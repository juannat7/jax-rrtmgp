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

"""Interface for Jatmos to the Fast Diagonalization Solver.

Covers cases for the anelastic Navier Stokes equations, possibly with a
stretched grid.

Pressure poisson equation arising from the anelastic Navier Stokes equations:

      div (rho grad p) = rhs

which we write in possibly stretched coordinates (but we still write the
coordinates as x,y,z even though more properly it's q0, q1, q2).  In
coordinates, the Poisson equation is written as

      ∂/∂x (w0 ∂p/∂x) + ∂/∂y (w1 ∂p/∂y) + ∂/∂z (w2 ∂p/∂z) = f

where, with stretched grid given by hx, hy, hz, the weights are:

      w0 = rho * hy * hz / hx
      w1 = rho * hx * hz / hy
      w2 = rho * hx * hy / hz,
      f = rhs * hx * hy * hz

The discretized operator for the fast diagonalization solver is written in the
form:

      L = (Ax ⊗ By ⊗ Bz + Bx ⊗ Ay ⊗ Bz + Bx ⊗ By ⊗ Az)

  Since rho depends only on z, we have: the matrices Ai and Bi arise from
      A_x = ∂/∂x [(1/hx) ∂p/∂x]
      A_y = ∂/∂y [(1/hy) ∂p/∂y]
      A_z = ∂/∂z [(rho/hz) ∂p/∂z]
      B_x = hx
      B_y = hy
      B_z = rho * hz

When instantiating a fast diagonalization solver, the density and stretched
grids must be provided so that the A and B matrices are known, and so the
operators can be diagonalized.

When solving a given problem (with a given RHS), the stretched grid factors must
be known to multiply the RHS by.
"""

from typing import TypeAlias

import jax
import numpy as np
from swirl_jatmos import stretched_grid_util
from swirl_jatmos.linalg import fast_diagonalization_solver_impl


Array: TypeAlias = jax.Array
FastDiagSolver: TypeAlias = fast_diagonalization_solver_impl.Solver
BCType: TypeAlias = fast_diagonalization_solver_impl.BCType


def solver_factory(
    rho_xxc: np.ndarray,
    rho_xxf: np.ndarray,
    hx_c: np.ndarray,
    hx_f: np.ndarray,
    hy_c: np.ndarray,
    hy_f: np.ndarray,
    hz_c: np.ndarray,
    hz_f: np.ndarray,
    grid_spacings: tuple[float, float, float],
    dtype: jax.typing.DTypeLike,
    uniform_y_2d: bool = False,
    uniform_z_2d: bool = False,
) -> FastDiagSolver:
  """Take in densities and stretched grids, and return a FastDiagSolver.

  This assumes the anelastic case where the reference density rho(z) depends
  only on z.
  """
  # BCs for Jatmos: periodic x,y, and wall in z (so Neumann for pressure).
  bc_types = (BCType.PERIODIC, BCType.PERIODIC, BCType.NEUMANN)
  nx, ny, nz = len(hx_c), len(hy_c), len(hz_c)
  poisson_solver = fast_diagonalization_solver_impl.Solver(
      (nx, ny, nz),
      grid_spacings,
      bc_types,
      (0, 0, 1),
      dtype,
      w_Ax_f=1 / hx_f,
      w_Ay_f=1 / hy_f,
      w_Az_f=rho_xxf / hz_f,
      w_Bx_c=hx_c,
      w_By_c=hy_c,
      w_Bz_c=rho_xxc * hz_c,
      uniform_y_2d=uniform_y_2d,
      uniform_z_2d=uniform_z_2d,
  )
  return poisson_solver


def _generate_modified_rhs(rhs_ccc: Array, sg_map: dict[str, Array]) -> Array:
  """Multiply the RHS by the stretched grid scale factors.

      rhs_modified = rhs * hx * hy * hz

  Args:
    rhs_ccc: The RHS of the Poisson equation.
    sg_map: The stretched grid map.

  Returns:
    The modified RHS.
  """
  use_stretched_grid = stretched_grid_util.get_use_stretched_grid(sg_map)
  for dim in (0, 1, 2):
    if use_stretched_grid[dim]:
      hc_dim = sg_map[stretched_grid_util.hc_key(dim)]
      rhs_ccc = rhs_ccc * hc_dim
  return rhs_ccc


def solve(
    poisson_solver: FastDiagSolver,
    rhs_ccc: Array,
    sg_map: dict[str, Array],
):
  """Solves the Poisson equation with the Fast Diagonalization solver."""
  rhs_ccc = _generate_modified_rhs(rhs_ccc, sg_map)
  dp_ccc = poisson_solver.solve(rhs_ccc)
  return dp_ccc
