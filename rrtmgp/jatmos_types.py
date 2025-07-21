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

"""Common types."""

from absl import flags
import jax
import jax.numpy as jnp

USE_64BIT_DTYPES = flags.DEFINE_bool(
    'use_64bit_dtypes',
    False,
    'If true, 64-bit dtypes will be used for the simulation. If false, a 32-bit'
    ' dtypes will be used.',
    allow_override=True,
)


# Note: This setting is intended to affect all but a small number of simulation
# variables. However, these are not wired up for most variables yet. Variables
# that are not affected by this setting are: 't_ns', 'dt_ns', and 'step_id'.
# Those variables always use 64-bit dtypes. For simulations, it is always
# required that JAX has x64 enabled.
def __getattr__(name):
  try:
    f_dtype: jax.typing.DTypeLike = (
        jnp.float64 if USE_64BIT_DTYPES.value else jnp.float32
    )
    i_dtype: jax.typing.DTypeLike = (
        jnp.int64 if USE_64BIT_DTYPES.value else jnp.int32
    )
  except flags.UnparsedFlagAccessError:
    # Fall-back default.
    f_dtype = jnp.float32
    i_dtype = jnp.int32
  if name == 'f_dtype':
    return f_dtype
  elif name == 'i_dtype':
    return i_dtype
  else:
    raise AttributeError(f'Unknown attribute: {name}')
