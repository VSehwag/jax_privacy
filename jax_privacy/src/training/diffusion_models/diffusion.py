from typing import Optional, Tuple

import math
import numpy as np
from easydict import EasyDict

import jax
import jax.numpy as jnp
import haiku as hk

TIMESTEPS = 1000

# TODO: Cleanup the code before moving to jax_privacy


class GuassianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps: int, dim: int = 4):
        self.timesteps = timesteps
        self.dim = dim
        self.alpha_bar_scheduler = (lambda t: math.cos(
            (t / self.timesteps + 0.008) / 1.008 * math.pi / 2)**2)
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps)

        self.clamp_x0 = lambda x: x.clip(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (self.clamp_x0(
            1 / jnp.sqrt(scalars.alpha_bar[t]) * (xt -
                                                  jnp.sqrt(1 - scalars.alpha_bar[t]) * eps)))
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: (jnp.sqrt(scalars.alpha_bar[t]) * scalars.beta[t]) / (
                (1 - scalars.alpha_bar[t]) * jnp.sqrt(scalars.alpha[t])) * x0
            + (scalars.alpha[t] - scalars.alpha_bar[t]) / (
                (1 - scalars.alpha_bar[t]) * jnp.sqrt(scalars.alpha[t])) * xt)

    # TODO: Figure out typing issues when passing function (alpha_bar_scheduler)
    def get_all_scalars(self, alpha_bar_scheduler, timesteps: int, betas: bool = None):
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = jnp.asarray([
                min(
                    1 -
                    alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                    0.999,
                ) for t in range(timesteps)
            ]).reshape([-1] + [1] * (self.dim-1))  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas

        all_scalars["beta_log"] = jnp.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = jnp.cumprod(all_scalars["alpha"], axis=0)
        all_scalars["beta_tilde"] = (all_scalars["beta"][1:] *
                                     (1 - all_scalars["alpha_bar"][:-1]) /
                                     (1 - all_scalars["alpha_bar"][1:]))
        all_scalars["beta_tilde"] = jnp.concatenate(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]])
        all_scalars["beta_tilde_log"] = jnp.log(all_scalars["beta_tilde"])
        return EasyDict(
            dict([(k, v.astype(float)) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0: np.ndarray, t: np.ndarray, rng: jnp.ndarray, eps = None):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        if eps is None:
            eps = jax.random.normal(key=rng, shape=x0.shape)
        xt = jnp.sqrt(self.scalars.alpha_bar[t]) * \
            x0 + jnp.sqrt(1 - self.scalars.alpha_bar[t]) * eps
        return xt, eps

    def sample_from_reverse_process(self,
                                    net,
                                    params: hk.Params,
                                    state: hk.State,
                                    rng: np.ndarray,
                                    xT: np.ndarray,
                                    y: Optional[np.ndarray] = None,
                                    timesteps: int = None,
                                    ddim: bool = False) -> Tuple[np.ndarray, hk.Params]:
        """Sampling images by iterating over all timesteps.
        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.
        Return: An image tensor with identical shape as XT.
        """
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = jnp.linspace(0,
                                     self.timesteps - 1,
                                     num=timesteps,
                                     endpoint=True,
                                     dtype=int)
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        alpha_bar_div = jnp.pad(alpha_bar.ravel(), [1, 0], constant_values=1.0)[
            :-1].reshape(alpha_bar.shape)
        new_betas = 1 - (alpha_bar / alpha_bar_div)
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, new_betas)

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            current_t = jnp.asarray([t] * len(final))
            current_sub_t = jnp.asarray([i] * len(final))

            pred_eps, state = net.apply(
                params, state, None, {"x": final, "t": current_t, "y": y}, is_training=False)

            # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
            pred_x0 = self.get_x0_from_xt_eps(final, pred_eps,
                                              current_sub_t, scalars)
            pred_mean = self.get_pred_mean_from_x0_xt(
                final, pred_x0, current_sub_t, scalars)
            if i == 0:
                final = pred_mean
            else:
                if ddim:
                    final = (
                        jnp.sqrt(scalars["alpha_bar"][current_sub_t -
                                                      1]) *
                        pred_x0 + jnp.sqrt(1 - scalars["alpha_bar"][
                            current_sub_t - 1]) * pred_eps)
                else:
                    final = pred_mean + jnp.sqrt(scalars.beta_tilde[
                        current_sub_t]) * jax.random.normal(key=rng, shape=final.shape)
        return final, state
