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

"""Utility functions."""

from typing import NamedTuple, TypeAlias

from absl import flags
import jax
import numpy as np

from swirl_jatmos import derivatives
from swirl_jatmos.boundary_conditions import apply_bcs
from swirl_jatmos.boundary_conditions import boundary_conditions

_PRINT_FOR_COLAB = flags.DEFINE_bool(
    'print_for_colab',
    False,
    'If true, calls to `print_if_flag` will print to STDOUT in addition to'
    ' logging, which is useful for colab.',
    allow_override=True,
)

Array: TypeAlias = jax.Array


class StrainRateTensor(NamedTuple):
  """Representation of the symmetric strain-rate tensor."""

  s00_ccc: Array
  s01_ffc: Array
  s02_fcf: Array
  s11_ccc: Array
  s12_cff: Array
  s22_ccc: Array


def print_if_flag(msg: str) -> None:
  """Conditionally print to STDOUT if flag `print_for_colab` is set.

  Print outputs are useful e.g. in Colab cells.

  Args:
    msg: String to print to console.
  """
  if _PRINT_FOR_COLAB.value:
    print(msg)


def uniform_grid(
    domain: tuple[float, float],
    num_cores: int,
    n_per_core: int,
    halo_width: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Sets up a grid with n nodes within the specified domain.

  The domain ends are on faces.  The face that aligns with domain[0] is the
  first non-halo face on the left, while the face that aligns with domain[1] is
  the first halo face on the right (if halos are included).

  E.g., with halo_width = 0 (appropriate for a periodic grid):

    v <-- domain[0]         v <-- domain[1] : not included in grid
    | o | o | o | o | o | o


  E.g., with halo_width = 1:

        v <-- domain[0]         v <-- domain[1]
    | o | o | o | o | o | o | o | o
    ^ ^                         ^ ^
    halos                       halos

  Args:
    domain: Tuple of the left and right ends of the domain.
    num_cores: The number of cores.
    n_per_core: The number of nodes per core.
    halo_width: The halo width.

  Returns:
    A 2-tuple of arrays containing the coordinates of the nodes and the
    coordinates of the faces, for a uniform grid.
  """
  length = domain[1] - domain[0]
  n_total_no_halos = n_per_core * num_cores - 2 * halo_width
  n_total_with_halos = n_per_core * num_cores
  dx = length / n_total_no_halos
  x_face_min = domain[0] - halo_width * dx
  x_face_max = domain[1] + (halo_width - 1) * dx

  x_faces = np.linspace(x_face_min, x_face_max, n_total_with_halos)
  x_nodes = x_faces + dx / 2
  return x_nodes, x_faces


def compute_strain_rate_tensor(
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    deriv_lib: derivatives.Derivatives,
    sg_map: dict[str, Array],
    z_c: np.ndarray,
    z_f: np.ndarray,
    z_bcs: boundary_conditions.ZBoundaryConditions,
) -> StrainRateTensor:
  """Compute the strain rate tensor."""
  du_dx_ccc = deriv_lib.dx_f_to_c(u_fcc, sg_map)
  du_dy_ffc = deriv_lib.dy_c_to_f(u_fcc, sg_map)
  du_dz_fcf = deriv_lib.dz_c_to_f(u_fcc, sg_map)

  dv_dx_ffc = deriv_lib.dx_c_to_f(v_cfc, sg_map)
  dv_dy_ccc = deriv_lib.dy_f_to_c(v_cfc, sg_map)
  dv_dz_cff = deriv_lib.dz_c_to_f(v_cfc, sg_map)

  dw_dx_fcf = deriv_lib.dx_c_to_f(w_ccf, sg_map)
  dw_dy_cff = deriv_lib.dy_c_to_f(w_ccf, sg_map)
  dw_dz_ccc = deriv_lib.dz_f_to_c(w_ccf, sg_map)

  s00_ccc = du_dx_ccc
  s01_ffc = 0.5 * (du_dy_ffc + dv_dx_ffc)
  s02_fcf = 0.5 * (du_dz_fcf + dw_dx_fcf)

  # Note that s10_ffc = s01_ffc by symmetry.
  s11_ccc = dv_dy_ccc
  s12_cff = 0.5 * (dv_dz_cff + dw_dy_cff)

  # Note that s20_fcf = s02_fcf & s21_cff = s12_cff by symmetry.
  s22_ccc = dw_dz_ccc

  # Enforce z boundary conditions on s02, s12.
  # Bottom boundary.
  if z_bcs.bottom.bc_type == 'no_flux':  # Free-slip wall BCs.
    s02_fcf = apply_bcs.enforce_flux_at_z_bottom_bdy(s02_fcf, 0.0)
    s12_cff = apply_bcs.enforce_flux_at_z_bottom_bdy(s12_cff, 0.0)
  elif z_bcs.bottom.bc_type == 'monin_obukhov':
    # With the Monin-Obukhov BC, calculate the strain rates at the surface
    # assuming u=0, v=0 at the bottom boundary.  This is a 1-sided derivative.
    # The strain rates here are used when computing the SGS eddy viscosity and
    # diffusivity terms.
    u_fcc_first_layer = u_fcc[:, :, 1]
    v_cfc_first_layer = v_cfc[:, :, 1]
    # Distance between first node and the bottom boundary (face).
    dz = float(z_c[1] - z_f[1])
    # At the bottom surface, w=0, so S_02 = 0.5 * ∂u/∂z, S_12 = 0.5 * ∂v/∂z.
    s02_fcf_surface = 0.5 * u_fcc_first_layer / dz
    s12_cff_surface = 0.5 * v_cfc_first_layer / dz
    s02_fcf = apply_bcs.enforce_flux_at_z_bottom_bdy(s02_fcf, s02_fcf_surface)
    s12_cff = apply_bcs.enforce_flux_at_z_bottom_bdy(s12_cff, s12_cff_surface)
  else:
    raise ValueError(f'Bad z_bcs.bottom.bc_type: {z_bcs.bottom.bc_type}')

  # Top boundary.
  if z_bcs.top.bc_type == 'no_flux':  # Free-slip wall BCs.
    s02_fcf = apply_bcs.enforce_flux_at_z_top_bdy(s02_fcf, 0.0)
    s12_cff = apply_bcs.enforce_flux_at_z_top_bdy(s12_cff, 0.0)
  else:
    raise ValueError(
        f'Unsupported z boundary condition type: {z_bcs.top.bc_type}'
    )

  return StrainRateTensor(s00_ccc, s01_ffc, s02_fcf, s11_ccc, s12_cff, s22_ccc)


def buoyancy(
    rho_thermal_ccf: Array,
    rho_xxf: Array,
    q_r_ccf: Array | None = None,
    q_s_ccf: Array | None = None,
) -> Array:
  """Compute the buoyancy acceleration."""
  g = 9.81  # In m/s^2.

  buoyancy_ccf = -g * (rho_thermal_ccf - rho_xxf) / rho_thermal_ccf
  if q_r_ccf is not None and q_s_ccf is not None:
    buoyancy_ccf = buoyancy_ccf - g * (q_r_ccf + q_s_ccf)
  return buoyancy_ccf
