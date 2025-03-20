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

"""Interface for Jatmos to the Jacobi Poisson solvers.

Covers cases for the anelastic Navier Stokes equations, possibly with a
stretched grid.
"""

from typing import TypeAlias

import jax
import jax.numpy as jnp
from swirl_jatmos import stretched_grid_util
from swirl_jatmos.linalg import jacobi_solver_impl

Array: TypeAlias = jax.Array
HaloUpdateFn: TypeAlias = jacobi_solver_impl.HaloUpdateFn

BaseJacobiSolver: TypeAlias = jacobi_solver_impl.BaseJacobiSolver
JacobiPlainPoisson: TypeAlias = jacobi_solver_impl.PlainPoisson
JacobiVariableCoefficientZFn: TypeAlias = (
    jacobi_solver_impl.VariableCoefficientZFn
)
JacobiThreeWeight: TypeAlias = jacobi_solver_impl.ThreeWeight
JacobiThreeWeightZFn: TypeAlias = jacobi_solver_impl.ThreeWeightZFn


def solver_factory(
    num_iterations: int,
    omega: float,
    grid_spacings: tuple[float, float, float],
    use_stretched_grid: tuple[bool, bool, bool],
) -> BaseJacobiSolver:
  """Take in densities and stretched grids, and return a Jacobi Solver.

  This assumes the anelastic case where the reference density rho(z) depends
  only on z.
  """
  halo_width = -1  # not used.
  if (
      not use_stretched_grid[0]
      and not use_stretched_grid[1]
      and use_stretched_grid[2]
  ):
    # If only stretching in z, use ThreeWeightZFn.  More efficient than the
    # general ThreeWeight solver.
    return JacobiThreeWeightZFn(
        grid_spacings, omega, num_iterations, halo_width
    )
  elif any(use_stretched_grid):
    # If there is any stretching (besides just z), use ThreeWeight.
    return JacobiThreeWeight(grid_spacings, omega, num_iterations, halo_width)
  else:
    # For uniform grid, use VariableCoefficientZFn, suitable for the anelastic
    # equations.
    return JacobiVariableCoefficientZFn(
        grid_spacings, omega, num_iterations, halo_width
    )


def _generate_weights_and_modified_rhs_for_three_weight(
    rhs_ccc: Array,
    rho_xxc: Array,
    rho_xxf: Array,
    sg_map: dict[str, Array],
) -> tuple[Array, Array, Array, Array]:
  """Generate the weights and modified RHS for the Jacobi solver.

  For the anelastic equations, the weights and the modified RHS are:
  J = h0 * h1 * h2
  w0_fcc = rho * h1 * h2 / h0
  w1_cfc = rho * h0 * h2 / h1
  w2_ccf = rho * h0 * h1 / h2
  rhs_ccc = rhs_ccc * h0 * h1 * h2

  Args:
    rhs_ccc: The right-hand-side of the Poisson equation.
    rho_xxc: The density on the center nodes.
    rho_xxf: The density on the face nodes.
    sg_map: The stretched grid map.
  Returns:
    A tuple of 4 elements: the 3 weighting cofficients w0, w1, w2, and the
    modified RHS.
  """
  use_stretched_grid = stretched_grid_util.get_use_stretched_grid(sg_map)
  hc = []
  hf = []
  for dim in (0, 1, 2):
    if use_stretched_grid[dim]:
      hc_key = stretched_grid_util.hc_key(dim)
      hf_key = stretched_grid_util.hf_key(dim)
      hc.append(sg_map[hc_key])
      hf.append(sg_map[hf_key])
    else:
      hc.append(1)
      hf.append(1)
  hx_c, hy_c, hz_c = hc[0], hc[1], hc[2]
  hx_f, hy_f, hz_f = hf[0], hf[1], hf[2]

  w0_fcc = rho_xxc * hy_c * hz_c / hx_f
  w1_cfc = rho_xxc * hx_c * hz_c / hy_f
  w2_ccf = rho_xxf * hx_c * hy_c / hz_f
  rhs_ccc = rhs_ccc * hx_c * hy_c * hz_c
  return w0_fcc, w1_cfc, w2_ccf, rhs_ccc


def _generate_weights_and_modified_rhs_for_three_weight_zfn(
    rhs_ccc: Array,
    rho_xxc: Array,
    rho_xxf: Array,
    sg_map: dict[str, Array],
) -> tuple[Array, Array, Array, Array]:
  """Generate the weights and modified RHS for the Jacobi solver.

  Here, stretching is assumed to occur only in z (and it must be used, else
  there is no reason to use this solver instead of VariableCoefficientZFn).
  Hence, h0=h1=1, but h2 is not just ones.

  For the anelastic equations, the weights and the modified RHS are:
  w0_xxc = rho * h2
  w1_xxc = rho * h2
  w2_xxf = rho / h2
  rhs_ccc = rhs_ccc * h2_xxc

  Args:
    rhs_ccc: The right-hand-side of the Poisson equation.
    rho_xxc: The density on the center nodes.
    rho_xxf: The density on the face nodes.
    sg_map: The stretched grid map.
  Returns:
    A tuple of 4 elements: the 3 weighting cofficients w0, w1, w2, and the
    modified RHS.
  """
  use_stretched_grid = stretched_grid_util.get_use_stretched_grid(sg_map)
  assert (
      not use_stretched_grid[0]
      and not use_stretched_grid[1]
      and use_stretched_grid[2]
  ), 'Must stretch in z and not in any other dim when using ThreeWeightZFn.'

  hz_c = sg_map[stretched_grid_util.hc_key(2)]
  hz_f = sg_map[stretched_grid_util.hf_key(2)]

  w0_xxc = rho_xxc * hz_c
  w1_xxc = rho_xxc * hz_c
  w2_xxf = rho_xxf / hz_f
  rhs_ccc = rhs_ccc * hz_c

  return w0_xxc, w1_xxc, w2_xxf, rhs_ccc


def solve(
    poisson_solver: BaseJacobiSolver,
    rhs_ccc: Array,
    rho_xxc: Array,
    rho_xxf: Array,
    sg_map: dict[str, Array],
    halo_update_fn: HaloUpdateFn,
):
  """Solves the Poisson equation with the Jacobi solver.

  Inputs are: possibly stretched grids, and rho.
  """
  p0 = jnp.zeros_like(rhs_ccc)
  if isinstance(poisson_solver, JacobiVariableCoefficientZFn):
    return poisson_solver.solve(rho_xxc, rho_xxf, rhs_ccc, p0, halo_update_fn)
  elif isinstance(poisson_solver, JacobiThreeWeight):
    w0_fcc, w1_cfc, w2_ccf, rhs_ccc = (
        _generate_weights_and_modified_rhs_for_three_weight(
            rhs_ccc, rho_xxc, rho_xxf, sg_map
        )
    )
    return poisson_solver.solve(
        w0_fcc, w1_cfc, w2_ccf, rhs_ccc, p0, halo_update_fn
    )
  elif isinstance(poisson_solver, JacobiThreeWeightZFn):
    w0_xxc, w1_xxc, w2_xxf, rhs_ccc = (
        _generate_weights_and_modified_rhs_for_three_weight_zfn(
            rhs_ccc, rho_xxc, rho_xxf, sg_map
        )
    )
    return poisson_solver.solve(
        w0_xxc, w1_xxc, w2_xxf, rhs_ccc, p0, halo_update_fn
    )
  elif isinstance(poisson_solver, JacobiPlainPoisson):
    del rho_xxc, rho_xxf, sg_map
    return poisson_solver.solve(rhs_ccc, p0, halo_update_fn)
  else:
    raise ValueError(
        f'Should not be here. Jacobi solver is: {poisson_solver}'
    )
