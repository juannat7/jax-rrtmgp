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

"""Module for checking whether states are valid."""

from collections.abc import Iterable
from typing import TypeAlias

from absl import flags
import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array

DEBUG_CHECK_FOR_NANS = flags.DEFINE_bool(
    'debug_check_for_nans',
    False,
    'If true, check for NaNs in the u field at the end of each step.  If NaNs'
    ' are found, exit the simulation early.',
    allow_override=True,
)


def check_no_nan_inf1(fields: Iterable[Array]) -> Array:
  """Check that all input fields are finite."""
  # Reduce on each field to a single scalar (boolean type).
  all_finite_for_each_field = jnp.array([jnp.isfinite(x).all() for x in fields])
  # Then reduce over each field's scalar.
  return jnp.all(all_finite_for_each_field)


def check_no_nan_inf2(fields: Iterable[Array]) -> Array:
  """Alternate implementation of above that may have different performance."""
  # Compute isnan on each field.
  isfinite_for_each_field = jnp.array([jnp.isfinite(x) for x in fields])
  # Apply a reduce operation using logical-and.
  isfinite_for_all_fields = jax.tree.reduce(
      jnp.logical_and, isfinite_for_each_field
  )
  # Reduce to a single scalar.
  return jnp.all(isfinite_for_all_fields)


def check_states_are_finite(states: dict[str, Array]) -> Array:
  fieldnames = ['dtheta_li', 'q_t', 'q_r', 'q_s', 'u', 'v', 'w']
  fields_to_check = [states[fieldname] for fieldname in fieldnames]
  return check_no_nan_inf1(fields_to_check)
