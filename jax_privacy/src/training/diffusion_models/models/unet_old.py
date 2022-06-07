# Ref: https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py
import haiku as hk
import jax
import math
import jax.numpy as jnp


class TimestepEmbedding(hk.Module):
    """
    Create sinusoidal timestep embeddings (https://github.com/openai/guided-diffusion).

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    def __init__(self, dim, max_period=10000, name=None):
        super().__init__(name=name)
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period

    def __call__(self, x):
        half = self.dim // 2
        freqs = jnp.exp(-math.log(self.max_period) *
                        jnp.arange(start=0, stop=half).astype(float) / half)
        args = x[:, None].astype(float) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding


class Normalization(hk.Module):

    def __init__(self, type="GroupNorm", name=None):
        super().__init__(name=name)
        self.type = type

    def __call__(self, x):
        if self.type == "NoNorm":
            return x
        elif self.type == "GroupNorm":
            return hk.GroupNorm(groups=16, data_format="channels_first")(x)
        elif self.type == "GroupNorm8":
            return hk.GroupNorm(groups=8, data_format="channels_first")(x)
        else:
            raise ValueError(f"{self.type} norm not supported!")


# class FourierFeatures(hk.Module):

#     def __init__(self, output_size, std=1., name=None):
#         super().__init__(name=name)
#         assert output_size % 2 == 0
#         self.output_size = output_size
#         self.std = std

#     def __call__(self, x):
#         w = hk.get_parameter('w', [self.output_size // 2, x.shape[1]],
#                              init=hk.initializers.RandomNormal(self.std, 0))
#         f = 2 * jnp.pi * x @ w.T
#         return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class SelfAttention2d(hk.Module):

    def __init__(self, n_head=1, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.n_head = n_head
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        n, c, h, w = x.shape
        assert c % self.n_head == 0
        qkv_proj = hk.Conv2D(c * 3, 1, data_format='NCHW', name='qkv_proj')
        out_proj = hk.Conv2D(c, 1, data_format='NCHW', name='out_proj')
        x = Normalization("GroupNorm")(x)
        qkv = qkv_proj(x)
        qkv = jnp.swapaxes(
            qkv.reshape([n, self.n_head * 3, c // self.n_head, h * w]), 2, 3)
        q, k, v = jnp.split(qkv, 3, axis=1)
        scale = k.shape[3]**-0.25
        att = jax.nn.softmax((q * scale) @ (jnp.swapaxes(k, 2, 3) * scale),
                             axis=3)
        y = jnp.swapaxes(att @ v, 2, 3).reshape([n, c, h, w])
        return x + Normalization("GroupNorm")(out_proj(y))


def res_conv_block(c_mid, c_out, last_norm=True):

    def inner(x):
        x_skip_layer = hk.Conv2D(c_out, 1, with_bias=False, data_format='NCHW')
        x_skip = x if x.shape[1] == c_out else x_skip_layer(x)
        x = hk.Conv2D(c_mid, 3, data_format='NCHW')(x)
        x = Normalization("GroupNorm")(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(c_out, 3, data_format='NCHW')(x)
        if last_norm:
            x = Normalization("GroupNorm")(x)
        x = jax.nn.relu(x)
        return x + x_skip

    return inner


class unet(hk.Module):

    def __init__(self, base_width=128, attn=True, time_embed_dim=64, name=None, *args, **kwargs):
        super().__init__(name=name)
        self.width = base_width
        self.attn = attn
        self.time_embed_dim = time_embed_dim

    def __call__(self, x, timesteps, y=None, is_training=False):
        c = self.width
        x = jnp.transpose(x, (0, 3, 1, 2))  # NCHW
        timestep_embed = TimestepEmbedding(
            dim=self.time_embed_dim)(timesteps)
        te_planes = jnp.tile(timestep_embed[..., None, None],
                             [1, 1, x.shape[2], x.shape[3]])  # N x time_embed_dim x h x w
        x = jnp.concatenate([x, te_planes], axis=1)  # 128x128
        x = res_conv_block(c, c)(x)
        x = res_conv_block(c, c)(x)
        x_2 = hk.AvgPool(2, 2, 'SAME', 1)(x)  # 64x64
        x_2 = res_conv_block(c * 2, c * 2)(x_2)
        x_2 = res_conv_block(c * 2, c * 2)(x_2)
        x_3 = hk.AvgPool(2, 2, 'SAME', 1)(x_2)  # 32x32
        x_3 = res_conv_block(c * 2, c * 2)(x_3)
        x_3 = res_conv_block(c * 2, c * 2)(x_3)
        x_4 = hk.AvgPool(2, 2, 'SAME', 1)(x_3)  # 16x16
        x_4 = res_conv_block(c * 4, c * 4)(x_4)
        if self.attn:
            x_4 = SelfAttention2d(c * 4 // 128)(x_4)
        x_4 = res_conv_block(c * 4, c * 4)(x_4)
        if self.attn:
            x_4 = SelfAttention2d(c * 4 // 128)(x_4)
        x_5 = hk.AvgPool(2, 2, 'SAME', 1)(x_4)  # 8x8
        x_5 = res_conv_block(c * 4, c * 4)(x_5)
        if self.attn:
            x_5 = SelfAttention2d(c * 4 // 128)(x_5)
        x_5 = res_conv_block(c * 4, c * 4)(x_5)
        if self.attn:
            x_5 = SelfAttention2d(c * 4 // 128)(x_5)
        x_6 = hk.AvgPool(2, 2, 'SAME', 1)(x_5)  # 4x4
        x_6 = res_conv_block(c * 8, c * 8)(x_6)
        if self.attn:
            x_6 = SelfAttention2d(c * 8 // 128)(x_6)
        x_6 = res_conv_block(c * 8, c * 8)(x_6)
        if self.attn:
            x_6 = SelfAttention2d(c * 8 // 128)(x_6)
        x_6 = res_conv_block(c * 8, c * 8)(x_6)
        if self.attn:
            x_6 = SelfAttention2d(c * 8 // 128)(x_6)
        x_6 = res_conv_block(c * 8, c * 4)(x_6)
        if self.attn:
            x_6 = SelfAttention2d(c * 4 // 128)(x_6)
        x_6 = jax.image.resize(x_6, [*x_6.shape[:2], *x_5.shape[2:]],
                               'nearest')
        x_5 = jnp.concatenate([x_5, x_6], axis=1)
        x_5 = res_conv_block(c * 4, c * 4)(x_5)
        if self.attn:
            x_5 = SelfAttention2d(c * 4 // 128)(x_5)
        x_5 = res_conv_block(c * 4, c * 4)(x_5)
        if self.attn:
            x_5 = SelfAttention2d(c * 4 // 128)(x_5)
        x_5 = jax.image.resize(x_5, [*x_5.shape[:2], *x_4.shape[2:]],
                               'nearest')
        x_4 = jnp.concatenate([x_4, x_5], axis=1)
        x_4 = res_conv_block(c * 4, c * 4)(x_4)
        if self.attn:
            x_4 = SelfAttention2d(c * 4 // 128)(x_4)
        x_4 = res_conv_block(c * 4, c * 2)(x_4)
        if self.attn:
            x_4 = SelfAttention2d(c * 2 // 128)(x_4)
        x_4 = jax.image.resize(x_4, [*x_4.shape[:2], *x_3.shape[2:]],
                               'nearest')
        x_3 = jnp.concatenate([x_3, x_4], axis=1)
        x_3 = res_conv_block(c * 2, c * 2)(x_3)
        x_3 = res_conv_block(c * 2, c * 2)(x_3)
        x_3 = jax.image.resize(x_3, [*x_3.shape[:2], *x_2.shape[2:]],
                               'nearest')
        x_2 = jnp.concatenate([x_2, x_3], axis=1)
        x_2 = res_conv_block(c * 2, c * 2)(x_2)
        x_2 = res_conv_block(c * 2, c)(x_2)
        x_2 = jax.image.resize(x_2, [*x_2.shape[:2], *x.shape[2:]], 'nearest')
        x = jnp.concatenate([x, x_2], axis=1)
        x = res_conv_block(c, c)(x)
        x = res_conv_block(c, 3, last_norm=False)(x)
        return x.transpose(0, 2, 3, 1)  # NHWC
