from pickle import FALSE
import functools
from turtle import forward
import haiku as hk
import jax
import math
import jax.numpy as jnp
from torch import isin

# Adapted by Vikash Sehwag (https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py)
# Note: By default jax assumes NHWC ordering, thus the networks expects input in NHWC format.


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings (https://github.com/openai/guided-diffusion).

    :param timesteps: a 1-D arrays of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] array of positional embeddings.
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) *
                    jnp.arange(start=0, stop=half).astype(float) / half)
    args = timesteps[:, None].astype(float) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embedding

# TODO: is this fxn making the whole code super slow


def interp_Nd(images, target_shape):
    """
    Note that a poor implmentation of interpolation can make code 50-100x slower in jax.

    E.g, following jnp.stack([upsample(x, jnp.meshgrid(jnp.arange(target_shape[0]), jnp.arange(target_shape[1]), 
                                jnp.arange(x.shape[-1]))) for x in images]) 
        upsampling is 10-50x slower than simply using the in-build verions jax.image.resize(.) 
    """
    if len(target_shape) == 1:
        out = jax.image.resize(
            images, [len(images), target_shape[0], images.shape[-1]], method="nearest")
    elif len(target_shape) == 2:
        out = jax.image.resize(images, [len(
            images), target_shape[0], target_shape[1], images.shape[-1]], method="nearest")
    else:
        raise ValueError("Only 1-d and 2-d interpolation are supported")

    return out


def forward_sequential(layers, x, emb=None):
    out = x
    for layer in layers:
        if isinstance(layer, ResBlock):
            out = layer(out, emb)
        else:
            out = layer(out)
    return out


class Identity(hk.Module):

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        return x


class Normalization(hk.Module):

    def __init__(self, cat="group_norm", name=None):
        super().__init__(name=name)
        self.cat = cat

    def __call__(self, x):
        if self.cat == "no_norm":
            return x
        elif self.cat == "group_norm":
            return hk.GroupNorm(groups=32)(x)
        elif self.cat == "group_norm_nchw":
            return hk.GroupNorm(groups=32, data_format="channels_first")(x)
        else:
            raise ValueError(f"{self.cat} norm not supported!")


class Activation(hk.Module):

    def __init__(self, cat="silu", name=None):
        super().__init__(name=name)
        self.cat = cat

    def __call__(self, x):
        if self.cat == "relu":
            return jax.nn.relu(x)
        elif self.cat == "silu":
            return jax.nn.silu(x)
        else:
            raise ValueError(f"{self.cat} activation not supported!")


class Upsample(hk.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, name=None):
        super().__init__(name=name)
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.proj_conv = hk.ConvND(
                num_spatial_dims=dims,
                output_channels=self.out_channels,
                kernel_shape=3,
                padding="SAME",
                stride=1,
                with_bias=False,
                name="upsample_conv")

    def __call__(self, x):
        assert x.shape[-1] == self.channels
        if self.dims == 2:
            x = interp_Nd(x, [2 * x.shape[1], 2 * x.shape[2]])
        elif self.dims == 1:
            x = interp_Nd(x, [2 * x.shape[1]])
        else:
            raise ValueError(
                f"Interpolation for {self.dims} dim not supported.")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(hk.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, name=None):
        super().__init__(name=name)
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = hk.ConvND(
                num_spatial_dims=dims,
                output_channels=self.out_channels,
                kernel_shape=3,
                padding="SAME",
                stride=stride,
                with_bias=False,
                name="downsample_conv")
        else:
            assert self.channels == self.out_channels
            assert self.dims == 2, "only 2-d pooling supported in haiku (AFAIK)"
            self.op = hk.AvgPool(window_shape=stride,
                                 strides=stride, padding="SAME",
                                 name="downsample_pool")

    def __call__(self, x):
        assert x.shape[-1] == self.channels
        return self.op(x)


class ResBlock(hk.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        name=None,
    ):
        super().__init__(name=name)
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = [
            Normalization("group_norm"),
            Activation("silu"),
            hk.ConvND(dims, self.out_channels, 3,
                      padding="SAME", with_bias=False)
        ]

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, name="upsample_h")
            self.x_upd = Upsample(channels, False, dims, name="upsample_x")
        elif down:
            self.h_upd = Downsample(channels, False, dims, name="downsample_h")
            self.x_upd = Downsample(channels, False, dims, name="downsample_x")
        else:
            self.h_upd = self.x_upd = Identity()

        # TODO: If using bias, initialize it from zeros
        self.emb_layers = [Activation("silu"), hk.Linear(
            2 * self.out_channels, with_bias=False)]

        self.out_layers = [Normalization(
            "group_norm"), Activation("silu"),
            hk.ConvND(dims, self.out_channels, 3, padding="SAME")]

        if self.out_channels == channels:
            self.skip_connection = Identity()
        elif use_conv:
            self.skip_connection = hk.ConvND(
                dims, self.out_channels, 3, padding="SAME", name="conv_skip_connnection")
        else:
            self.skip_connection = hk.ConvND(
                dims, self.out_channels, 1, padding="SAME", name="linear_skip_connnection")

    def __call__(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = forward_sequential(in_rest, x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = forward_sequential(self.in_layers, x)

        emb_out = forward_sequential(self.emb_layers, emb)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None, :]

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = jnp.split(emb_out, 2, axis=-1)
        h = out_norm(h) * (1 + scale) + shift
        h = forward_sequential(out_rest, h)

        return self.skip_connection(x) + h

# TODO: Revert the changes in attention


class AttentionBlock(hk.Module):
    # Ref: https://github.com/crowsonkb/v-diffusion-jax
    def __init__(self, num_heads=1, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads

    def __call__(self, x):
        x = x.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        n, c, h, w = x.shape
        assert c % self.num_heads == 0
        qkv_proj = hk.Conv2D(c * 3, 1, data_format='NCHW', name='qkv_proj')
        out_proj = hk.Conv2D(c, 1, data_format='NCHW', name='out_proj')
        x = Normalization("group_norm_nchw")(x)
        qkv = qkv_proj(x)
        qkv = jnp.swapaxes(
            qkv.reshape([n, self.num_heads * 3, c // self.num_heads, h * w]), 2, 3)
        q, k, v = jnp.split(qkv, 3, axis=1)
        scale = k.shape[3]**-0.25
        att = jax.nn.softmax((q * scale) @ (jnp.swapaxes(k, 2, 3) * scale),
                             axis=3)
        y = jnp.swapaxes(att @ v, 2, 3).reshape([n, c, h, w])
        return (x + Normalization("group_norm_nchw")(out_proj(y))).transpose(0, 2, 3, 1)


# TODO: Label all layers/sub-layers properly

class UNetModel(hk.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_heads=1,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4
        self.time_embed = [hk.Linear(time_embed_dim, with_bias=False), Activation(
            "silu"), hk.Linear(time_embed_dim, with_bias=False)]

        if self.num_classes is not None:
            self.label_emb = hk.Embed(
                vocab_size=num_classes, embed_dim=time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = [hk.ConvND(dims, ch, 3, padding="SAME")]

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            num_heads=num_heads,
                        )
                    )
                self.input_blocks.append(layers)
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=out_ch,
                        dims=dims,
                        down=True,
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)  # list of list-of-layers
                ds *= 2
                self._feature_size += ch

        self.middle_block = [
            ResBlock(
                ch,
                time_embed_dim,
                dims=dims,
            ),
            AttentionBlock(
                num_heads=num_heads,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dims=dims,
            ),
        ]
        self._feature_size += ch

        self.output_blocks = []
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            num_heads=num_heads,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            out_channels=out_ch,
                            dims=dims,
                            up=True,
                        )

                    )
                    ds //= 2
                self.output_blocks.append(layers)  # list of list-of-layers
                self._feature_size += ch

        self.out = [Normalization(
            "group_norm"), Activation("silu"), hk.ConvND(dims, out_channels, 3, padding="SAME")]

    def __call__(self, x, timesteps, y=None, is_training=False):
        """
        Apply the model to an input batch.
        :param x: an [N x H x ... X C] array of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] array of labels, if class-conditional.
        :return: an [N x H x ... X C] array of outputs.
        """

        ## Turning off this check for now, as jax-privacy automatically 
        ## passed num_classes to the model
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = forward_sequential(self.time_embed, timestep_embedding(
            timesteps, dim=self.model_channels))
        
        # TODO: use num_classes instead of the array y for this check
        if y is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + forward_sequential(self.label_emb, y)

        h = x
        for module in self.input_blocks:
            h = forward_sequential(module if isinstance(
                module, list) else [module], h, emb)
            hs.append(h)
        h = forward_sequential(self.middle_block, h, emb)
        for module in self.output_blocks:
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = forward_sequential(module if isinstance(
                module, list) else [module], h, emb)
        h = forward_sequential(self.out, h, emb)
        return h


def UNet(
    image_size,
    in_channels=3,
    out_channels=3,
    base_width=192,
    num_classes=None,
):
    if image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    if image_size == 28:
        attention_resolutions = "28,14,7"
    else:
        attention_resolutions = "32,16,8"
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=base_width,
        out_channels=out_channels,
        num_res_blocks=2,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        num_classes=num_classes,
        num_heads=4,
    )


def UNetTiny(
    image_size,
    in_channels=3,
    out_channels=3,
    base_width=32,
    num_classes=None,
):
    if image_size == 64:
        channel_mult = (1, 1, 2, 2)
    elif image_size == 32:
        channel_mult = (1, 1, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 1, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    if image_size == 28:
        attention_resolutions = "28,14,7"
    else:
        attention_resolutions = "32,16,8"
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=base_width,
        out_channels=out_channels,
        num_res_blocks=1,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        num_classes=num_classes,
        num_heads=4,
    )
