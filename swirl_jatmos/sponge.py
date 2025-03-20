# Copyright 2024 The swirl_jatmos Authors.
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

"""Module for applying sponges to fields."""

from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import sponge_config

Array: TypeAlias = jax.Array


def apply_sponge(
    f: Array,
    z: Array,
    domain_top: float,
    sponge_cfg: sponge_config.SpongeConfig,
) -> Array:
  """Apply sponge that damps the input field at the domain top."""
  coeff = sponge_cfg.coeff
  sponge_fraction = sponge_cfg.sponge_fraction
  c2 = sponge_cfg.c2

  beta = jnp.zeros_like(z)

  # z coordinate at which the sponge starts.  No sponge below z_s.
  z_s = (1 - sponge_fraction) * domain_top
  beta1 = coeff * jnp.sin(np.pi / 2 * (z - z_s) / (domain_top - z_s)) ** 2

  beta2 = jnp.exp(c2 * (z - z_s) / (domain_top - z_s)) - 1
  beta_z = beta1 * beta2

  beta_z = jnp.where(z <= z_s, jnp.zeros_like(z), beta_z)
  beta = jnp.maximum(beta, beta_z)
  return f / (1.0 + beta)
