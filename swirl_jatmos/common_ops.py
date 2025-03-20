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

"""Common ops."""

from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


def reshape_to_broadcastable(f_1d: Array, dim: Literal[0, 1, 2]) -> Array:
  """Reshapes a rank-1 tensor to a form broadcastable against 3D fields."""
  if dim == 0:
    return f_1d[:, jnp.newaxis, jnp.newaxis]
  elif dim == 1:
    return f_1d[jnp.newaxis, :, jnp.newaxis]
  else:  # dim == 2
    return f_1d[jnp.newaxis, jnp.newaxis, :]


