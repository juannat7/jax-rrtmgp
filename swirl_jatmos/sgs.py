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

"""Smagorinsky-Lilly model for dry air."""

from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import constants
from swirl_jatmos import derivatives
from swirl_jatmos import interpolation
from swirl_jatmos import stretched_grid_util
from swirl_jatmos.utils import utils


Array: TypeAlias = jax.Array


def _z_deriv_with_neumann_lower_and_neumann_upper(
    f_c: Array,
    sg_map: dict[str, Array],
    z_c: np.ndarray,
    deriv_lib: derivatives.Derivatives,
    halo_width: int,
) -> Array:
  """Compute ∂f/∂z on z nodes with Neumann BCs on lower & upper z faces.

  Here: use neumann boundary conditions for f in z, enforcing ∂f/∂z = 0
  at the top and bottom boundaries, which are z faces.

  Since we need ∂f/∂z on z nodes, but ∂f/∂z is "naturally" computed on z faces,
  we use the following approach to incorporate the BCs.

  wall
  v
  |  o  |  o  |
     ^
     f0    f1

  To compute [∂f/∂z]_0 at the first interior node, we interpolate by averaging
  [∂f/∂z] from the wall (where ∂f/∂z = 0) and from the first interior face.
  Thus, use the following formula:
      [∂f/∂z]_0 = 0.5 * (f1 - f0) / (z1 - z0)

  The same approach is used at the last interior node.

  Args:
    f_c: The function f on (ccc).
    sg_map: The stretched grid map.
    z_c: Numpy array of z coordinates on nodes, used to statically compute the
      required grid spacings at the boundaries.
    deriv_lib: The derivative library.
    halo_width: The halo width in the z dimension.

  Returns:
    The derivative of f with respect to z on (ccc).
  """
  hw = halo_width
  df_dz_c = deriv_lib.deriv_centered(f_c, 2, sg_map)

  # Apply BCs to the first and last interior nodes.
  # Distance between 1st and 2nd interior nodes.
  dz_first = float(z_c[hw + 1] - z_c[hw])
  # Distance between last and 2nd-last interior nodes.
  dz_last = float(z_c[-(hw + 1)] - z_c[-(hw + 2)])

  df_dz_first = 0.5 * (f_c[:, :, hw + 1] - f_c[:, :, hw]) / dz_first
  df_dz_last = 0.5 * (f_c[:, :, -(hw + 1)] - f_c[:, :, -(hw + 2)]) / dz_last

  df_dz_c = df_dz_c.at[:, :, hw].set(df_dz_first)
  df_dz_c = df_dz_c.at[:, :, -(hw + 1)].set(df_dz_last)
  return df_dz_c


def _z_deriv_with_dirichlet_lower_and_neumann_upper(
    f_c: Array,
    f_bndy: Array | float,
    sg_map: dict[str, Array],
    z_c: np.ndarray,
    z_f: np.ndarray,
    deriv_lib: derivatives.Derivatives,
    halo_width: int,
) -> Array:
  """Returns ∂f/∂z on z nodes with Dirichlet / Neumann BCs on lower/upper faces.

  Here, the input f is assumed to be given on z nodes.  The derivative ∂f/∂z is
  also computed on z nodes.

  The assumed boundary conditions are:
    1) f = f_bndy on the lower z face.
    2) ∂f/∂z = 0 on the upper z face.

  To account for these BCs (which are on z faces), when both f and the output
  are on z nodes, we have to directly enforce the BCs.

  Here: use Dirichlet BCs for f on the lower z boundary face, and neumann BCs
  for f on the upper z boundary face, enforcing ∂f/∂z = 0.

  Since we need ∂f/∂z on z nodes, but ∂f/∂z is "naturally computed" on z faces,
  we use the following approach to incorporate the BCs.

  f_bndy (wall)
  v
  |  o  |  o  |
     ^
     f0    f1

  To compute [∂f/∂z]_0 at the first interior node, we use the difference
  between f0 and f_bndy, which are separated by the distance from the wall
  (face) to the first interior node.  Thus, use the following formula:
      [∂f/∂z]_0 = (f0 - f_bndy) / (0.5*(z1 - z0))

  For the upper Neumann BC, use the same approach as described in function
  `_z_deriv_with_neumann_lower_and_neumann_upper`.

  Args:
    f_c: The function f on (ccc).
    f_bndy: The Dirichlet boundary condition on f.
    sg_map: The stretched grid map.
    z_c: Numpy array of z coordinates on nodes, used to statically compute the
      required grid spacings at the boundaries.
    z_f: Numpy array of z coordinates on faces, used to statically compute the
      required grid spacings at the boundaries.
    deriv_lib: The derivative library.
    halo_width: The halo width in the z dimension.

  Returns:
    The derivative of f with respect to z on (ccc).
  """
  df_dz_c = deriv_lib.deriv_centered(f_c, 2, sg_map)
  hw = halo_width

  # Fix up the first and last interior nodes as specified in the docstring.
  # Distance from the wall to the first interior node.
  dz_wall_to_first_node = float(z_c[hw] - z_f[hw])
  # Distance between last and 2nd-last interior nodes
  dz_last = float(z_c[-(hw + 1)] - z_c[-(hw + 2)])

  df_dz_c_first = (f_c[:, :, hw] - f_bndy) / dz_wall_to_first_node
  df_dz_c_last = 0.5 * (f_c[:, :, -(hw + 1)] - f_c[:, :, -(hw + 2)]) / dz_last

  df_dz_c = df_dz_c.at[:, :, hw].set(df_dz_c_first)
  df_dz_c = df_dz_c.at[:, :, -(hw + 1)].set(df_dz_c_last)
  return df_dz_c


def _compute_delta_squared_ccc(
    sg_map: dict[str, Array],
    grid_spacings: tuple[float, float, float],
) -> Array | float:
  """Compute delta_squared_ccc = (dx_c * dy_c * dz_c)^(2/3)."""
  dx_c, dy_c, dz_c = stretched_grid_util.get_dxdydz(
      sg_map, grid_spacings, faces=False
  )
  delta_squared_ccc = (dx_c * dy_c * dz_c) ** (2 / 3)
  return delta_squared_ccc


def _compute_s_squared(strain_rate_tensor: utils.StrainRateTensor) -> Array:
  """Compute 2 * S_ij * S_ij on (ccc)."""
  s00_ccc = strain_rate_tensor.s00_ccc
  s01_ffc = strain_rate_tensor.s01_ffc
  s02_fcf = strain_rate_tensor.s02_fcf
  s11_ccc = strain_rate_tensor.s11_ccc
  s12_cff = strain_rate_tensor.s12_cff
  s22_ccc = strain_rate_tensor.s22_ccc

  # Interpolate to (ccc).
  s01_ccc = interpolation.x_f_to_c(interpolation.y_f_to_c(s01_ffc))
  s02_ccc = interpolation.x_f_to_c(interpolation.z_f_to_c(s02_fcf))
  s12_ccc = interpolation.y_f_to_c(interpolation.z_f_to_c(s12_cff))

  # Use the s_ij = s_ji symmetry in computing sum_ij (S_ij^2).
  s_squared_ccc = 2 * (
      (s00_ccc**2 + s11_ccc**2 + s22_ccc**2)  # Diagonal terms.
      + 2 * (s01_ccc**2 + s02_ccc**2 + s12_ccc**2)  # Off-diagonal terms.
  )
  return s_squared_ccc


def n_squared_unsaturated(
    theta_ccc: Array,
    sg_map: dict[str, Array],
    z_c: np.ndarray,
    deriv_lib: derivatives.Derivatives,
    halo_width: int,
) -> Array:
  """Return the squared buoyancy frequency for unsaturated air on (ccc).

  This assumes dry air (no moisture).  If there is moisture, then one needs to
  use the density potential temperature, and not just the potential temperature.

  N^2 = g * ∂θ/∂z / θ

  Here: use neumann boundary conditions for θ in z, enforcing ∂θ/∂z = 0
  at the top and bottom boundaries, which are z faces.

  Args:
    theta_ccc: Potential temperature θ.
    sg_map: The stretched grid map.
    z_c: Numpy array of z coordinates on nodes, used to statically compute the
      required grid spacings at the boundaries.
    deriv_lib: The derivative library.
    halo_width: The halo width in the z dimension.

  Returns:
    Squared buoyancy frequency [in s^-2] on (ccc).
  """
  g = constants.G  # m/s^2

  # Compute ∂θ/∂z on (ccc), handling boundary conditions carefully.
  dtheta_dz_ccc = _z_deriv_with_neumann_lower_and_neumann_upper(
      theta_ccc, sg_map, z_c, deriv_lib, halo_width
  )
  n_squared_ccc = g * dtheta_dz_ccc / theta_ccc
  return n_squared_ccc


def stability_correction_factor(richardson_number: Array, pr_t: float) -> Array:
  """Returns the stability correction factor for Smagorinsky-Lilly model."""
  ri_pr = richardson_number / pr_t
  factor = (jnp.maximum(0.0, 1 - ri_pr)) ** (1 / 2)

  # Prevent amplification of the turbulent viscosity by capping this factor at
  # a maximum value of 1.0.
  factor = jnp.where(richardson_number < 0, 1.0, factor)
  return factor


def smagorinsky_lilly_nu_t(
    strain_rate_tensor: utils.StrainRateTensor,
    theta_ccc: Array,
    pr_t: float,
    sg_map: dict[str, Array],
    z_c: np.ndarray,
    deriv_lib: derivatives.Derivatives,
    halo_width: int,
):
  """Compute the turbulent viscosity using the Smagorinsky-Lilly model.

  Use the unsaturated (and dry) version of the buoyancy frequency.

  Args:
    strain_rate_tensor: The strain rate tensor.
    theta_ccc: Potential temperature on (ccc).
    pr_t: The turbulent Prandtl number.
    sg_map: The stretched grid map.
    z_c: Numpy array of z coordinates on nodes, used to statically compute the
      required grid spacings at the boundaries.
    deriv_lib: The derivative library.
    halo_width: The halo width in the z dimension.

  Returns:
    The turbulent viscosity [m^2/s] on (ccc).
  """
  n_squared_ccc = n_squared_unsaturated(
      theta_ccc, sg_map, z_c, deriv_lib, halo_width
  )

  # Compute delta_ccc = (dx_ccc * dy_ccc * dz_ccc)^(1/3).
  delta_squared_ccc = _compute_delta_squared_ccc(
      sg_map, deriv_lib._grid_spacings  # pylint: disable=protected-access
  )

  # Get the magnitude of the rate-of-strain tensor.
  s_squared_ccc = _compute_s_squared(strain_rate_tensor)

  s_ccc = jnp.sqrt(s_squared_ccc)
  c_s = 0.18

  eps = 1e-4  # Small number to prevent division by zero.
  richardson_number_ccc = n_squared_ccc / (s_squared_ccc + eps)
  correction_factor_ccc = stability_correction_factor(
      richardson_number_ccc, pr_t
  )

  eddy_viscosity_ccc = (
      c_s**2 * delta_squared_ccc * s_ccc * correction_factor_ccc
  )
  return eddy_viscosity_ccc


