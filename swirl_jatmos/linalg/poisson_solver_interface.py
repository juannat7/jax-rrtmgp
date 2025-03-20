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

"""Interface to the various Poisson solvers."""

from typing import TypeAlias

from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import config
from swirl_jatmos import interpolation
from swirl_jatmos.linalg import fast_diagonalization_solver
from swirl_jatmos.linalg import fast_diagonalization_solver_impl
from swirl_jatmos.linalg import jacobi_solver
from swirl_jatmos.linalg import jacobi_solver_impl
from swirl_jatmos.utils import utils

Array: TypeAlias = jax.Array
HaloUpdateFn: TypeAlias = jacobi_solver_impl.HaloUpdateFn

BaseJacobiSolver: TypeAlias = jacobi_solver.BaseJacobiSolver
FastDiagSolver: TypeAlias = fast_diagonalization_solver_impl.Solver
PoissonSolver: TypeAlias = BaseJacobiSolver | FastDiagSolver

_NUM_JACOBI_ITERATIONS = flags.DEFINE_integer(
    'num_jacobi_iterations',
    10,
    'Number of iterations to run the Jacobi Poisson solver for.',
    allow_override=True,
)


def get_poisson_solver(
    rho_ref_xxc: np.ndarray,
    cfg: config.Config,
) -> PoissonSolver:
  """Create the Poisson solver."""
  if cfg.poisson_solver_type is config.PoissonSolverType.JACOBI:
    num_iterations = _NUM_JACOBI_ITERATIONS.value
    msg = f'Initializing Jacobi solver with {num_iterations} iterations.'
    logging.info(msg)
    utils.print_if_flag(msg)
    poisson_solver = jacobi_solver.solver_factory(
        num_iterations=num_iterations,
        omega=2 / 3,
        grid_spacings=cfg.grid_spacings,
        use_stretched_grid=cfg.use_stretched_grid,
    )
  elif cfg.poisson_solver_type is config.PoissonSolverType.FAST_DIAGONALIZATION:
    msg = 'Initializing fast diagonalization solver.'
    logging.info(msg)
    utils.print_if_flag(msg)
    # rho_ref_xxc is assumed to be a 1D array, representing the reference
    # density as a function of z, evaluated on z nodes.
    assert rho_ref_xxc.ndim == 1
    # Interpolate reference density from z nodes to z faces.
    rho_ref_xxf = interpolation.centered_node_to_face(
        jnp.asarray(rho_ref_xxc), 0
    )
    dtype = jnp.float32
    poisson_solver = fast_diagonalization_solver.solver_factory(
        np.squeeze(np.array(rho_ref_xxc, dtype=np.float64)),
        np.squeeze(np.array(rho_ref_xxf, dtype=np.float64)),
        cfg.hx_c,
        cfg.hx_f,
        cfg.hy_c,
        cfg.hy_f,
        cfg.hz_c,
        cfg.hz_f,
        cfg.grid_spacings,
        dtype,
        cfg.uniform_y_2d,
    )
  else:
    raise ValueError(
        f'Unhandled poisson solver type: {cfg.poisson_solver_type}'
    )
  return poisson_solver


def solve(
    poisson_solver: PoissonSolver,
    rhs_ccc: Array,
    rho_xxc: Array,
    rho_xxf: Array,
    sg_map: dict[str, Array],
    halo_update_fn: HaloUpdateFn,
):
  """Solve the Poisson equation."""
  if isinstance(poisson_solver, FastDiagSolver):
    return fast_diagonalization_solver.solve(poisson_solver, rhs_ccc, sg_map)
  elif isinstance(poisson_solver, BaseJacobiSolver):
    return jacobi_solver.solve(
        poisson_solver, rhs_ccc, rho_xxc, rho_xxf, sg_map, halo_update_fn
    )
  else:
    raise ValueError(f'Unhandled Poisson solver type: {type(poisson_solver)}')
