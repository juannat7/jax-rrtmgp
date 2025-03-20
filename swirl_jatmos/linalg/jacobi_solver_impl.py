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

"""Jacobi solvers for the Poisson equation.

The functionality here is used to solve the Poisson equation, i.e. ∇²p = b.
"""

import functools
from typing import Callable, TypeAlias

from absl import logging
import jax
from jax import lax
from swirl_jatmos import kernel_ops

Array: TypeAlias = jax.Array
HaloUpdateFn: TypeAlias = Callable[[Array], Array]


class BaseJacobiSolver:
  """Base class for weighted Jacobi solvers."""

  def __init__(
      self,
      grid_spacings: tuple[float, float, float],
      omega: float,
      num_iters: int,
      halo_width: int,
  ):
    if max(grid_spacings) / min(grid_spacings) > 1e3:
      logging.warning(
          'Ratio of maximum to minimum grid spacing is large. This has been'
          ' known to cause numerical issues in converging to high precision.'
      )

    self._grid_spacings = grid_spacings
    self._omega = omega
    self._num_iters = num_iters
    self._halo_width = halo_width

    self._reciprocal_square_grid_spacings = (
        1.0 / grid_spacings[0] ** 2,
        1.0 / grid_spacings[1] ** 2,
        1.0 / grid_spacings[2] ** 2,
    )

  def _apply_underrelaxation(
      self, p_star: Array, p: Array
  ) -> Array:
    """Apply underrelaxation for weighted Jacobi solver."""
    return self._omega * p_star + (1.0 - self._omega) * p

  def _do_iterations(
      self,
      p0: Array,
      one_iteration_fn: Callable[[Array], Array],
      halo_update_fn: HaloUpdateFn,
  ) -> Array:
    """Run the iterations for weighted Jacobi solver."""

    def body_fn(i: int, p: Array) -> Array:
      del i
      p = halo_update_fn(p)
      p_next = one_iteration_fn(p)
      return p_next

    p_end = lax.fori_loop(0, self._num_iters, body_fn, p0)
    return halo_update_fn(p_end)


class PlainPoisson(BaseJacobiSolver):
  """Jacobi solver for the Poisson equation.

  Solve the equation

      (∂^2/∂x^2 + ∂^2/∂y^2 + ∂^2/∂z^2) p = f

  using the weighted Jacobi method.
  """

  def __init__(
      self,
      grid_spacings: tuple[float, float, float],
      omega: float,
      num_iters: int,
      halo_width: int,
  ):
    """Initialize the Jacobi solver for the Poisson equation."""
    super().__init__(grid_spacings, omega, num_iters, halo_width)

    # Initialize some constants.
    self._factor_b = 1 / (2 * sum(self._reciprocal_square_grid_spacings))

    self._factor_x = self._factor_b / grid_spacings[0]**2
    self._factor_y = self._factor_b / grid_spacings[1]**2
    self._factor_z = self._factor_b / grid_spacings[2]**2

  def _poisson_step(self, p: Array, rhs: Array) -> Array:
    """Compute one step of the Jacobi iteration."""
    # Compute the right hand side function for interior points.
    x_term = self._factor_x * kernel_ops.centered_sum(p, 0)
    y_term = self._factor_y * kernel_ops.centered_sum(p, 1)
    z_term = self._factor_z * kernel_ops.centered_sum(p, 2)
    p_jacobi = x_term + y_term + z_term - self._factor_b * rhs
    return self._apply_underrelaxation(p_jacobi, p)

  def solve(
      self, rhs: Array, p0: Array, halo_update_fn: HaloUpdateFn
  ) -> Array:
    """Solve the Poisson equation."""
    p_next_fn = functools.partial(self._poisson_step, rhs=rhs)
    return self._do_iterations(p0, p_next_fn, halo_update_fn)


class VariableCoefficientZFn(BaseJacobiSolver):
  """Jacobi solver for the variable-coefficient Poisson equation.

  Solve the variable-coefficient Poisson equation:

      ∂/∂x (w ∂p/∂x) + ∂/∂y (w ∂p/∂y) + ∂/∂z (w ∂p/∂z) = f

  where w(z) is the variable coefficent and depends only on z.  w will be given
  on both z centers & z faces.

  The weighted Jacobi method is used.

  This solver is suitable for a staggered grid, in which w is evaluated on both
  centers and faces, as appropriate.
  """

  def _variable_coefficient_poisson_step(
      self, p: Array, w_c: Array, w_f: Array, rhs: Array
  ) -> Array:
    """Perform one Jacobi iteration."""
    inv_dq0_sq = self._reciprocal_square_grid_spacings[0]
    inv_dq1_sq = self._reciprocal_square_grid_spacings[1]
    inv_dq2_sq = self._reciprocal_square_grid_spacings[2]

    # Compute the diagonal factor.
    factor_diag_x = 2 * inv_dq0_sq * w_c
    factor_diag_y = 2 * inv_dq1_sq * w_c
    factor_diag_z = inv_dq2_sq * kernel_ops.forward_sum(w_f, 2)

    factor_diag_sum = factor_diag_x + factor_diag_y + factor_diag_z

    t_x = inv_dq0_sq * w_c * kernel_ops.centered_sum(p, 0)
    t_y = inv_dq1_sq * w_c * kernel_ops.centered_sum(p, 1)

    t_z_1 = kernel_ops.shift_from_plus(w_f * p, 2)
    t_z_2 = w_f * kernel_ops.shift_from_minus(p, 2)
    t_z = inv_dq2_sq * (t_z_1 + t_z_2)

    rhs_factor = t_x + t_y + t_z - rhs
    p_jacobi = rhs_factor / factor_diag_sum
    return self._apply_underrelaxation(p_jacobi, p)

  def residual(self, p: Array, w_c: Array, w_f: Array, rhs: Array) -> Array:
    """Compute the residual (LHS - RHS) of the Poisson equation."""
    inv_dq0_sq = self._reciprocal_square_grid_spacings[0]
    inv_dq1_sq = self._reciprocal_square_grid_spacings[1]
    inv_dq2_sq = self._reciprocal_square_grid_spacings[2]

    # Compute the diagonal factor.
    factor_diag_x = 2 * inv_dq0_sq * w_c
    factor_diag_y = 2 * inv_dq1_sq * w_c
    factor_diag_z = inv_dq2_sq * kernel_ops.forward_sum(w_f, 2)

    factor_diag_sum = factor_diag_x + factor_diag_y + factor_diag_z

    t_x = inv_dq0_sq * w_c * kernel_ops.centered_sum(p, 0)
    t_y = inv_dq1_sq * w_c * kernel_ops.centered_sum(p, 1)

    t_z_1 = kernel_ops.shift_from_plus(w_f * p, 2)
    t_z_2 = w_f * kernel_ops.shift_from_minus(p, 2)
    t_z = inv_dq2_sq * (t_z_1 + t_z_2)

    return -factor_diag_sum * p + t_x + t_y + t_z - rhs

  def solve(
      self,
      w_c: Array,
      w_f: Array,
      rhs: Array,
      p0: Array,
      halo_update_fn: Callable[[Array], Array],
  ) -> Array:
    """Solve the variable-coefficient Poisson equation.

    Args:
      w_c: The variable-coefficient weight w(z) on centers.
      w_f: The variable-coefficient weight w(z) on faces.
      rhs: The right-hand-side of the Poisson equation.
      p0: The initial guess for the solution.
      halo_update_fn: A function that updates the halos of the solution with
        boundary conditions.

    Returns:
      The numerical solution to the Poisson equation after the designated number
      of iterations.
    """
    p_next_fn = functools.partial(
        self._variable_coefficient_poisson_step,
        w_c=w_c,
        w_f=w_f,
        rhs=rhs,
    )

    return self._do_iterations(p0, p_next_fn, halo_update_fn)


class ThreeWeight(BaseJacobiSolver):
  """Jacobi solver for the three-weight Poisson equation.

  Solves the three-weight Poisson equation:

      ∂/∂x (w0 ∂p/∂x) + ∂/∂y (w1 ∂p/∂y) + ∂/∂z (w2 ∂p/∂z) = f

  where w0(x,y,z), w1(x,y,z), w2(x,ymz) are the three weighting coefficients.
  This occurs, for example, when stretched grids are used.

  The equation is solved using the weighted Jacobi method.

  This solver is suitable for a staggered grid, in which w0  & w1 are evaluated
  on z centers, while w2 is evaluated on z faces.
  """

  def _precompute_reciprocal_diagonal_factor(
      self,
      w0_fcc: Array,
      w1_cfc: Array,
      w2_ccf: Array,
  ) -> Array:
    """Compute the reciprocal diagonal factor for the 3-weight Poisson eqn."""
    inv_dq0_sq = self._reciprocal_square_grid_spacings[0]
    inv_dq1_sq = self._reciprocal_square_grid_spacings[1]
    inv_dq2_sq = self._reciprocal_square_grid_spacings[2]

    # Compute the diagonal factor.
    factor_diag_x = inv_dq0_sq * kernel_ops.forward_sum(w0_fcc, 0)
    factor_diag_y = inv_dq1_sq * kernel_ops.forward_sum(w1_cfc, 1)
    factor_diag_z = inv_dq2_sq * kernel_ops.forward_sum(w2_ccf, 2)
    return 1 / (factor_diag_x + factor_diag_y + factor_diag_z)

  def _three_weight_poisson_step(
      self,
      p_ccc: Array,
      w0_fcc: Array,
      w1_cfc: Array,
      w2_ccf: Array,
      rhs_ccc: Array,
      reciprocal_diagonal_factor: Array,
  ) -> Array:
    """Perform one Jacobi iteration."""
    inv_dq0_sq = self._reciprocal_square_grid_spacings[0]
    inv_dq1_sq = self._reciprocal_square_grid_spacings[1]
    inv_dq2_sq = self._reciprocal_square_grid_spacings[2]

    t_x_1 = kernel_ops.shift_from_plus(w0_fcc * p_ccc, 0)
    t_x_2 = w0_fcc * kernel_ops.shift_from_minus(p_ccc, 0)
    t_x = inv_dq0_sq * (t_x_1 + t_x_2)

    t_y_1 = kernel_ops.shift_from_plus(w1_cfc * p_ccc, 1)
    t_y_2 = w1_cfc * kernel_ops.shift_from_minus(p_ccc, 1)
    t_y = inv_dq1_sq * (t_y_1 + t_y_2)

    t_z_1 = kernel_ops.shift_from_plus(w2_ccf * p_ccc, 2)
    t_z_2 = w2_ccf * kernel_ops.shift_from_minus(p_ccc, 2)
    t_z = inv_dq2_sq * (t_z_1 + t_z_2)

    numerator = t_x + t_y + t_z - rhs_ccc
    p_ccc_jacobi = numerator * reciprocal_diagonal_factor
    return self._apply_underrelaxation(p_ccc_jacobi, p_ccc)

  def solve(
      self,
      w0_fcc: Array,
      w1_cfc: Array,
      w2_ccf: Array,
      rhs_ccc: Array,
      p0_ccc: Array,
      halo_update_fn: Callable[[Array], Array],
  ) -> Array:
    """Solve the variable-coefficient Poisson equation.

    Args:
      w0_fcc: The first weighting coefficient w0(x,y,z) on (fcc).
      w1_cfc: The second weighting coefficient w1(x,y,z) on (cfc).
      w2_ccf: The third weighting coefficient w2(x,y,z) on (ccf).
      rhs_ccc: The right-hand-side of the Poisson equation, on (ccc).
      p0_ccc: The initial guess for the solution, on (ccc).
      halo_update_fn: A function that updates the halos of the solution with
        boundary conditions.

    Returns:
      The numerical solution to the Poisson equation after the designated number
      of iterations.
    """
    reciprocal_diagonal_factor = self._precompute_reciprocal_diagonal_factor(
        w0_fcc, w1_cfc, w2_ccf
    )
    p_next_fn = functools.partial(
        self._three_weight_poisson_step,
        w0_fcc=w0_fcc,
        w1_cfc=w1_cfc,
        w2_ccf=w2_ccf,
        rhs_ccc=rhs_ccc,
        reciprocal_diagonal_factor=reciprocal_diagonal_factor,
    )
    return self._do_iterations(p0_ccc, p_next_fn, halo_update_fn)


class ThreeWeightZFn(BaseJacobiSolver):
  """Jacobi solver for the three-weight Poisson equation.

  Solves the three-weight Poisson equation:

      ∂/∂x (w0 ∂p/∂x) + ∂/∂y (w1 ∂p/∂y) + ∂/∂z (w2 ∂p/∂z) = f

  where w0(z), w1(z), w2(z) are the three weighting coefficients and depend only
  on z.  This occurs, for example, in the anelastic equations with stretched
  grid only in the z dimension.

  The equation is solved using the weighted Jacobi method.

  This solver is suitable for a staggered grid, in which w0 & w1 are evaluated
  on z centers, while w2 is evaluated on z faces.
  """

  def _precompute_reciprocal_diagonal_factor(
      self,
      w0_xxc: Array,
      w1_xxc: Array,
      w2_xxf: Array,
  ) -> Array:
    """Compute the reciprocal diagonal factor for the 3-weight Poisson eqn."""
    inv_dq0_sq = self._reciprocal_square_grid_spacings[0]
    inv_dq1_sq = self._reciprocal_square_grid_spacings[1]
    inv_dq2_sq = self._reciprocal_square_grid_spacings[2]

    # Compute the diagonal factor.
    factor_diag_x = 2 * inv_dq0_sq * w0_xxc
    factor_diag_y = 2 * inv_dq1_sq * w1_xxc
    factor_diag_z = inv_dq2_sq * kernel_ops.forward_sum(w2_xxf, 2)
    return 1 / (factor_diag_x + factor_diag_y + factor_diag_z)

  def _three_weight_poisson_step(
      self,
      p_ccc: Array,
      w0_xxc: Array,
      w1_xxc: Array,
      w2_xxf: Array,
      rhs_ccc: Array,
      reciprocal_diagonal_factor: Array,
  ) -> Array:
    """Perform one Jacobi iteration."""
    inv_dq0_sq = self._reciprocal_square_grid_spacings[0]
    inv_dq1_sq = self._reciprocal_square_grid_spacings[1]
    inv_dq2_sq = self._reciprocal_square_grid_spacings[2]

    t_x = inv_dq0_sq * w0_xxc * kernel_ops.centered_sum(p_ccc, 0)
    t_y = inv_dq1_sq * w1_xxc * kernel_ops.centered_sum(p_ccc, 1)

    t_z_1 = kernel_ops.shift_from_plus(w2_xxf * p_ccc, 2)
    t_z_2 = w2_xxf * kernel_ops.shift_from_minus(p_ccc, 2)
    t_z = inv_dq2_sq * (t_z_1 + t_z_2)

    numerator = t_x + t_y + t_z - rhs_ccc
    p_ccc_jacobi = numerator * reciprocal_diagonal_factor
    return self._apply_underrelaxation(p_ccc_jacobi, p_ccc)

  def solve(
      self,
      w0_xxc: Array,
      w1_xxc: Array,
      w2_xxf: Array,
      rhs_ccc: Array,
      p0_ccc: Array,
      halo_update_fn: Callable[[Array], Array],
  ) -> Array:
    """Solve the variable-coefficient Poisson equation.

    Args:
      w0_xxc: The first weighting coefficient w0(z) on z centers.
      w1_xxc: The second weighting coefficient w1(z) on z centers.
      w2_xxf: The third weighting coefficient w2(z) on z faces.
      rhs_ccc: The right-hand-side of the Poisson equation, on (ccc).
      p0_ccc: The initial guess for the solution, on (ccc).
      halo_update_fn: A function that updates the halos of the solution with
        boundary conditions.

    Returns:
      The numerical solution to the Poisson equation after the designated number
      of iterations.
    """
    reciprocal_diagonal_factor = self._precompute_reciprocal_diagonal_factor(
        w0_xxc, w1_xxc, w2_xxf
    )
    p_next_fn = functools.partial(
        self._three_weight_poisson_step,
        w0_xxc=w0_xxc,
        w1_xxc=w1_xxc,
        w2_xxf=w2_xxf,
        rhs_ccc=rhs_ccc,
        reciprocal_diagonal_factor=reciprocal_diagonal_factor,
    )
    return self._do_iterations(p0_ccc, p_next_fn, halo_update_fn)
