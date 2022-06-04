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
"""Helper to get a model given collection of models."""
import functools
from typing import Sequence

from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.src.training.diffusion_models.models import imagenet
from jax_privacy.src.training.image_classification.models.common import restore_from_path
import numpy as np

MODELS = {
    'unet': imagenet.unet,
}


def get_model_instance(model_type: str, model_kwargs):
    """Instantiates the model with the model type and kwargs."""
    assert model_type in MODELS, (
        f'Model type not supported in this expt. Currently support only '
        f'{list(MODELS.keys())}. Input model type is {model_type}')
    generative_module = MODELS[model_type]
    return generative_module(**model_kwargs)
