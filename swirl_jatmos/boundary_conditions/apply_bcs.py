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

"""Module for applying specific boundary conditions."""

from typing import TypeAlias

import jax
from swirl_jatmos.boundary_conditions import monin_obukhov

Array: TypeAlias = jax.Array
MoninObukhovParameters: TypeAlias = monin_obukhov.MoninObukhovParameters

_HW = 1  # Halo width.
_HALO_VALUE = -4e8


def set_z_halo_nodes(f_xxc: Array, value: float, halo_width: int) -> Array:
  """Return a copy of `f_xxc` with the z halo nodes set to `value`."""
  # halo_width is assumed to be 1.
  del halo_width
  f_xxc = f_xxc.at[:, :, 0].set(value)
  f_xxc = f_xxc.at[:, :, -1].set(value)
  return f_xxc


def set_z_halo_faces(f_xxf: Array, value: float, halo_width: int) -> Array:
  """Return a copy of `f_xxc` with the z halo nodes set to `value`."""
  # halo_width is assumed to be 1.
  del halo_width
  # With a halo width of 1, only the bottom has a "halo face."  The top face is
  # the physical wall.
  f_xxf = f_xxf.at[:, :, 0].set(value)
  return f_xxf


def set_z_halo_nodes_to_large_value(f_xxc: Array, halo_width: int) -> Array:
  """Return a copy of `f_xxc` with the z halo nodes set to a large value."""
  return set_z_halo_nodes(f_xxc, _HALO_VALUE, halo_width)


def set_z_halo_faces_to_large_value(f_xxf: Array, halo_width: int) -> Array:
  """Return a copy of `f_xxf` with the z halo faces set to a large value."""
  return set_z_halo_faces(f_xxf, _HALO_VALUE, halo_width)


def enforce_flux_at_z_bottom_bdy(
    flux_z_xxf: Array, flux_bc: Array | float
) -> Array:
  """Return a flux with a specified flux enforced on the bottom z boundary.

  Args:
    flux_z_xxf: The flux to update, a 3D field.
    flux_bc: The flux value to set at the bottom boundary.  Either a scalar or a
      2D (nx, ny) field.
  Returns:
    The updated flux.
  """
  flux_z_xxf = flux_z_xxf.at[:, :, _HW].set(flux_bc)
  return flux_z_xxf


def enforce_flux_at_z_top_bdy(
    flux_z_xxf: Array, flux_bc: Array | float
) -> Array:
  """Return a flux with a specified flux enforced on the top z boundary.

  Args:
    flux_z_xxf: The flux to update, a 3D field.
    flux_bc: The flux value to set at the top boundary.  Either a scalar or a
      2D (nx, ny) field.
  Returns:
    The updated flux.
  """
  flux_z_xxf = flux_z_xxf.at[:, :, -_HW].set(flux_bc)
  return flux_z_xxf
