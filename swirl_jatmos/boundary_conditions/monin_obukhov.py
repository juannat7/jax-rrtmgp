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

"""Library for the Monin-Obukhov similarity theory boundary layer.

For now, the full nonlinear Monin-Obukhov similarity theory is not yet
implemented.  Curently, fixed exchange coefficients are used.

The velocity at the surface is assumed to be zero.  A drag law is used.

Momentum flux: -rho * drag_coefficient * u_j * u_mag
Scalar flux: -rho * exchange_coefficient * u_mag * (scalar_1 - sfc_scalar)

where scalar_1 is the scalar at the first interior node above ground.
"""

import dataclasses
from typing import TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


@dataclasses.dataclass(frozen=True, kw_only=True)
class MoninObukhovParameters:
  """Parameters for the Monin-Obukhov Similarity Theory."""

  # Whether to use pre-set, fixed exchange coefficients for momentum and scalar
  # (theta_li, q_t) fluxes.
  use_constant_exchange_coefficients: bool = True
  # Constant drag coefficient for momentum flux.
  momentum_flux_drag_coeff: float = 2e-3
  # Constant exchange coefficient for scalar flux.
  scalar_flux_exchange_coeff: float = 3e-3

  # Amplitude of the surface gustiness (m/s)
  surface_gustiness: float = 1.0


def _surface_momentum_flux(
    rho: Array,
    u_j: Array,
    u_mag: Array,
    drag_coefficient: float | Array,
) -> Array:
  """Compute one component of momentum flux for a given drag coefficient.

  Reference: CLIMA Atmosphere Model.

  Args:
    rho: The density at the first level above ground.
    u_j: The j-th component of the fluid velocity in the first level above
      ground.
    u_mag: The magnitude of the horizontal fluid velocity in the first level
      above ground.
    drag_coefficient: The drag coefficient.

  Returns:
    The surface vertical momentum flux for a given velocity component.
  """
  return -rho * drag_coefficient * u_j * u_mag


def surface_momentum_flux(
    u_fcc: Array,
    v_cfc: Array,
    rho_xxc: Array,
    mop: MoninObukhovParameters,
) -> tuple[Array, Array]:
  """Compute the surface vertical momentum fluxes tau02, tau12.

  Args:
    u_fcc: The x velocity, a 3D field.
    v_cfc: The y velocity, a 3D field.
    rho_xxc: The reference density.
    mop: The Monin-Obukhov parameters.

  Returns:
    A 2-tuple containing surface values for tau_02 and tau_12, which are 2D
    arrays of shape (nx, ny) containing the vertical momentum fluxes for u and
    v, respectively, computed from the Monin-Obukhov similarity theory.
  """
  if mop.use_constant_exchange_coefficients:
    drag_coeff = mop.momentum_flux_drag_coeff
  else:
    raise NotImplementedError('Nonlinear drag coefficient not implemented.')

  # Extract the values from the first interior node.
  u_fcc_first_layer = u_fcc[:, :, 1]
  v_cfc_first_layer = v_cfc[:, :, 1]
  rho_first_layer = rho_xxc[:, :, 1]

  # We are forming the magnitude of the horizontal velocity vector using u_fcc
  # and v_cfc, which are at different locations, but for the purposes of just
  # getting the magnitude, this should be sufficiently accurate.
  u_mag_first_layer = jnp.sqrt(
      mop.surface_gustiness**2 + u_fcc_first_layer**2 + v_cfc_first_layer**2
  )

  # Compute the surface shear stresses using the drag law, assuming the
  # horizontal velocity is zero at the surface.
  tau02_fcf_surface = _surface_momentum_flux(
      rho_first_layer, u_fcc_first_layer, u_mag_first_layer, drag_coeff
  )
  tau12_cff_surface = _surface_momentum_flux(
      rho_first_layer, v_cfc_first_layer, u_mag_first_layer, drag_coeff
  )
  return tau02_fcf_surface, tau12_cff_surface


def surface_scalar_flux(
    scalar_ccc: Array,
    u_fcc: Array,
    v_cfc: Array,
    rho_xxc: Array,
    sfc_scalar: Array,
    mop: MoninObukhovParameters,
) -> Array:
  """Compute the surface scalar flux.

  Args:
    scalar_ccc: The input scalar to find the Monin-Obukhov flux for, a 3D field.
    u_fcc: the x velocity, a 3D field.
    v_cfc: The y velocity, a 3D field.
    rho_xxc: The reference density field.
    sfc_scalar: The surface values (z=0) of the relevant scalar on, a 2D field
      on (cc).
    mop: The Monin-Obukhov parameters.

  Returns:
    The surface scalar flux, a 2D field on (cc).
  """
  if mop.use_constant_exchange_coefficients:
    exchange_coeff = mop.scalar_flux_exchange_coeff
  else:
    raise NotImplementedError('Nonlinear drag coefficient not implemented.')

  # Extract the values from the first interior node.
  scalar_ccc_first_layer = scalar_ccc[:, :, 1]
  u_fcc_first_layer = u_fcc[:, :, 1]
  v_cfc_first_layer = v_cfc[:, :, 1]
  rho_first_layer = rho_xxc[:, :, 1]

  u_mag_first_layer = jnp.sqrt(
      mop.surface_gustiness**2 + u_fcc_first_layer**2 + v_cfc_first_layer**2
  )

  # Sign check on flux: If scalar at the first layer is smaller than the surface
  # value, then the flux should be positive (upward directed).  This is
  # satisfied.
  scalar_flux = (
      -rho_first_layer
      * exchange_coeff
      * u_mag_first_layer
      * (scalar_ccc_first_layer - sfc_scalar)
  )
  return scalar_flux
