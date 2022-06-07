# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines train and evaluation functions that compute losses and metrics."""

from typing import Any, Dict, Tuple

import math
import jax
import chex
import haiku as hk
import jax.numpy as jnp
from jax_privacy.src.training import metrics as metrics_module
import optax

Model = hk.TransformedWithState


class MultiClassForwardFn:
    """Defines forward passes for multi-class classification."""

    def __init__(self, net: Model):
        """Initialization function.

        Args:
          net: haiku model to use for the forward pass.
        """
        self._net = net
        self.timesteps = 1000  # Hardcoding 1000 steps for diffusion process
        # TODO: toggle between identical vs random noise for each instance
        self.identical_noise = False
        self.scalars = self.get_scalars()

    def train_init(
        self,
        rng_key: chex.PRNGKey,
        inputs: chex.ArrayTree,
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """Initializes the model.

        Args:
          rng_key: random number generation key used for the random initialization.
          inputs: model inputs (of format `{'images': images, 'labels': labels}`),
            to infer shapes of the model parameters. Images are expected to be of
            shape [NKHWC] (K is augmult).
        Returns:
          Initialized model parameters and state.
        """
        images = inputs['images']
        timesteps = jnp.ones((len(images), ))
        # Images has shape [NKHWC] (K is augmult).
        return self._net.init(rng_key, images[:, 0], {"t": timesteps}, is_training=True)

    def get_scalars(self):
        all_scalars = {}
        def alpha_bar_scheduler(t): return math.cos(
            (t / self.timesteps + 0.008) / 1.008 * math.pi / 2)**2
        all_scalars["beta"] = jnp.asarray([
            min(
                1 -
                alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                0.999,
            ) for t in range(self.timesteps)
        ]).reshape(-1, 1, 1, 1)  # hardcoding beta_max to 0.999

        all_scalars["beta_log"] = jnp.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = jnp.cumprod(all_scalars["alpha"], axis=0)
        return all_scalars

    def train_forward(
        self,
        params: chex.ArrayTree,
        inputs: chex.ArrayTree,
        network_state: chex.ArrayTree,
        rng: chex.PRNGKey,
        frozen_params: chex.ArrayTree,
    ) -> Tuple[chex.Array, Any]:
        """Forward pass per example (training time).

        Args:
          params: model parameters that should get updated during training.
          inputs: model inputs (of format `{'images': images, 'labels': labels}`),
            where the labels are one-hot encoded. `images` is expected to be of
            shape [NKHWC] (K is augmult), and `labels` of shape [NKO].
          network_state: model state.
          rng: random number generation key.
          frozen_params: model parameters that should remain frozen. These will be
            merged with `params` before the forward pass, but are specified
            separately to make it easy to compute gradients w.r.t. the first
            argument `params` only.
        Returns:
          loss: loss function computed per-example on the mini-batch (averaged over
            the K augmentations).
          auxiliary information, including the new model state, metrics computed
            on the current mini-batch and the current loss value per-example.
        """
        images, labels = inputs['images'], inputs['labels']
        # TODO: Make sure that image is in [0, 1]
        images = (2 * images) - 1.0

        # TODO: Ablate along the choice: using identical timesteps for K augmult
        timesteps = jax.random.randint(key=rng, shape=(
            len(images), ), minval=0, maxval=self.timesteps)
        timesteps = timesteps.repeat(images.shape[1])  # [NK] size

        # `images` has shape [NKHWC] (K is augmult), while model accepts [NHWC], so
        # we use a single larger batch dimension.
        reshaped_images = images.reshape((-1,) + images.shape[2:])

        # TODO: Check that rng is different across all jobs
        if self.identical_noise:
            noise = jax.random.normal(
                key=rng, shape=(len(images), )+images.shape[2:])
            noise = noise.repeat(images.shape[1], axis=0)  # [NW, h, w, c]
        else:
            noise = jax.random.normal(
                key=rng, shape=reshaped_images.shape)
        alphas = self.scalars["alpha_bar"][timesteps]
        xt = jnp.sqrt(alphas) * reshaped_images + jnp.sqrt(1 - alphas) * noise

        all_params = hk.data_structures.merge(params, frozen_params)

        pred_noise, network_state = self._net.apply(
            all_params, network_state, rng, xt, {"t": timesteps}, is_training=True)
        loss = self._loss(pred_noise, noise)

        # We reshape back to [NK] and average across augmentations.
        loss = loss.reshape(images.shape[:2])
        loss = jnp.mean(loss, axis=1)

        metrics = {"loss": jnp.mean(loss)}
        return jnp.mean(loss), (network_state, metrics, loss)

    # TODO: Avoid eval or fix the loss in it
    def eval_forward(
        self,
        params: chex.ArrayTree,
        inputs: chex.ArrayTree,
        network_state: chex.ArrayTree,
        rng: chex.PRNGKey,
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """Forward pass per example (evaluation time).

        Args:
          params: model parameters that should get updated during training.
          inputs: model inputs (of format `{'images': images, 'labels': labels}`),
            where the labels are one-hot encoded. `images` is expected to be of
            shape [NHWC], and `labels` of shape [NO].
          network_state: model state.
          rng: random number generation key.
        Returns:
          logits: logits computed per-example on the mini-batch.
          metrics: metrics computed on the current mini-batch.
        """
        # logits, unused_network_state = self._net.apply(
        #     params, network_state, rng, inputs['images'], {"t": jnp.ones((len(inputs['images']), ))})
        # loss = jnp.mean(self._loss(logits, inputs['labels']))
        # dummy as logits are unused in current code
        logits = jnp.mean(inputs['images'])
        loss = jnp.mean(inputs['images'])  # just a dummy number

        metrics = {'loss': loss}
        return logits, metrics

    def _loss(self, pred_noise: chex.Array, noise: chex.Array) -> chex.Array:
        """Compute the loss per-example.

        Args:
          pred_noise: pred_noise by network.
          noise: Original noise added int he networks.
        Returns:
          MSE loss computed per-example on leading dimensions.
        """
        return jnp.mean((pred_noise - noise) ** 2, axis=list(range(1, len(noise.shape))))
