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

"""Test RK3 integration of Navier-Stokes equations for 2D Taylor-Green Vortex.

Note: this test does not use the `driver.py` driver, serving as an example of
how to set up and run a simulation without using that driver.
"""

import functools
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from swirl_jatmos import config
from swirl_jatmos import navier_stokes_step
from swirl_jatmos import sim_initializer
from swirl_jatmos import stretched_grid_util
from swirl_jatmos import test_util
from swirl_jatmos.linalg import fast_diagonalization_solver
from swirl_jatmos.linalg import jacobi_solver
from swirl_jatmos.sim_setups import taylor_green_vortex_2d

Array: TypeAlias = jax.Array
ProblemSpec: TypeAlias = taylor_green_vortex_2d.ProblemSpec
PoissonSolverType: TypeAlias = config.PoissonSolverType


def stretched_coordinate(zeta: npt.ArrayLike, alpha: float = 0.1) -> np.ndarray:
  """Transforms coordinate using sin-based stretching.

  This is a transform x=x(zeta), where 0 <= x <= 1 and 0 <= zeta <= 1, i.e. both
  x and zeta are normalized.

  Args:
    zeta: Values for normalized input coordinate.
    alpha: The amplitude of the sinusoidal distortion.

  Returns:
    The transformed, sinusoidally stretched, normalized coordinate.
  """
  if alpha > 1 / (2 * np.pi):
    raise ValueError('alpha must be <= 1 / (2 * pi).')

  return zeta - alpha * np.sin(2 * np.pi * zeta)


def get_stretched_grid(num_grid_points: int) -> np.ndarray:
  """Returns a stretched grid using sinusoidal stretching."""
  # Define the zeta grid on nodes.  The domain boundaries are placed at zeta=0
  # and zeta=1.
  dzeta = 1 / num_grid_points
  zeta_c = np.linspace(dzeta / 2, 1 - dzeta / 2, num_grid_points)
  x_c = stretched_coordinate(zeta_c)
  return x_c


def create_stretched_grid_tempfile(
    test: absltest.TestCase, num_grid_points: int
) -> str:
  """Creates file with stretched grid coordinates and returns the path."""
  # Get the stretched grid coordinates on nodes.
  coord = get_stretched_grid(num_grid_points)
  # Create a file with the stretched grid coordinates and get the path.
  return test_util.save_1d_array_to_tempfile(test, coord)


def _run_a_simulation(
    cfg: config.Config, spec: ProblemSpec, num_steps: int
) -> dict[str, Array]:
  dtype = jnp.float32

  if cfg.poisson_solver_type is PoissonSolverType.JACOBI:
    num_iterations = 10
    poisson_solver = jacobi_solver.solver_factory(
        num_iterations=num_iterations,
        omega=2 / 3,
        grid_spacings=cfg.grid_spacings,
        use_stretched_grid=cfg.use_stretched_grid,
    )
  elif cfg.poisson_solver_type is PoissonSolverType.FAST_DIAGONALIZATION:
    poisson_solver = fast_diagonalization_solver.solver_factory(
        np.ones_like(cfg.hz_c),
        np.ones_like(cfg.hz_f),
        cfg.hx_c,
        cfg.hx_f,
        cfg.hy_c,
        cfg.hy_f,
        cfg.hz_c,
        cfg.hz_f,
        cfg.grid_spacings,
        dtype,
        uniform_z_2d=True,
    )
  else:
    raise ValueError(
        f'Unhandled poisson solver type: {cfg.poisson_solver_type}'
    )

  # Initial condition function
  init_cond_fn = functools.partial(
      taylor_green_vortex_2d.analytical_solution_u_v_p, t=0, spec=spec
  )

  def one_step_fn(loop_i, states):
    del loop_i
    # Discard `aux_output` here.
    states, _ = navier_stokes_step.step(states, poisson_solver, cfg)
    return states

  @jax.jit
  def do_simulation():
    # Initialize sharded broadcastable grid variables.
    grid_map_sharded = sim_initializer.initialize_grids(cfg)
    x_c, x_f = grid_map_sharded['x_c'], grid_map_sharded['x_f']
    y_c, y_f = grid_map_sharded['y_c'], grid_map_sharded['y_f']

    u_fcc, v_cfc, p_ccc = init_cond_fn(x_c, x_f, y_c, y_f)
    u_fcc = jnp.tile(u_fcc, reps=(1, 1, cfg.nz))
    v_cfc = jnp.tile(v_cfc, reps=(1, 1, cfg.nz))
    p_ccc = jnp.tile(p_ccc, reps=(1, 1, cfg.nz))
    w_ccf = jnp.zeros_like(u_fcc)

    states = {'u': u_fcc, 'v': v_cfc, 'w': w_ccf, 'p': p_ccc}
    other_states = {
        'p_ref_xxc': jnp.ones((1, 1, cfg.nz), dtype=dtype),
        'rho_xxc': jnp.ones((1, 1, cfg.nz), dtype=dtype),
        'rho_xxf': jnp.ones((1, 1, cfg.nz), dtype=dtype),
        'theta_li_0': 300 * jnp.ones_like(u_fcc),
        'dtheta_li': jnp.zeros_like(u_fcc),
        'q_t': jnp.zeros_like(u_fcc),
        'q_r': jnp.zeros_like(u_fcc),
        'q_s': jnp.zeros_like(u_fcc),
    }
    states |= other_states  # Not used for TGV sim but needs to be present.
    states |= grid_map_sharded  # Add grid variables to states.
    # Add t, dt, and step_id to states.
    states |= sim_initializer.initialize_time_and_step_id(cfg)

    states = jax.lax.fori_loop(0, num_steps, one_step_fn, states)
    return states

  states = do_simulation()
  return states


class TaylorGreenVortex2DTest(parameterized.TestCase):

  @parameterized.product(
      poisson_solver_type=[
          PoissonSolverType.JACOBI,
          PoissonSolverType.FAST_DIAGONALIZATION,
      ],
      use_stretched_grid_in_dim0=[True, False],
      use_stretched_grid_in_dim1=[True, False],
  )
  def test_second_order_spatial_convergence_weno5(
      self,
      poisson_solver_type: PoissonSolverType,
      use_stretched_grid_in_dim0: bool,
      use_stretched_grid_in_dim1: bool,
  ):
    # SETUP
    dtype = jnp.float32
    dt = 0.0005
    num_steps = 4000
    convection_scheme = 'weno5_js'
    spec = ProblemSpec(domain_size=1.0, viscosity=6.25e-4, amplitude=1.0)

    def run_sim_and_get_error(num_points):
      # No sharding.
      # Set simulation parameters

      if use_stretched_grid_in_dim0:
        stretched_grid_path_x = create_stretched_grid_tempfile(self, num_points)
      else:
        stretched_grid_path_x = ''
      if use_stretched_grid_in_dim1:
        stretched_grid_path_y = create_stretched_grid_tempfile(self, num_points)
      else:
        stretched_grid_path_y = ''

      cfg_ext = taylor_green_vortex_2d.get_cfg(
          num_points,
          dt,
          spec,
          poisson_solver_type,
          convection_scheme,
          stretched_grid_path_x,
          stretched_grid_path_y,
      )
      cfg = config.config_from_config_external(cfg_ext)

      x_c, x_f = cfg.x_c, cfg.x_f
      y_c, y_f = cfg.y_c, cfg.y_f

      # Create broadcastable jax arrays (3D) and cast to desired dtype, assuming
      # we are starting with float64 numpy arrays.
      x_c_b = jnp.array(x_c[:, jnp.newaxis, jnp.newaxis], dtype=dtype)
      x_f_b = jnp.array(x_f[:, jnp.newaxis, jnp.newaxis], dtype=dtype)
      y_c_b = jnp.array(y_c[jnp.newaxis, :, jnp.newaxis], dtype=dtype)
      y_f_b = jnp.array(y_f[jnp.newaxis, :, jnp.newaxis], dtype=dtype)
      states = _run_a_simulation(cfg, spec, num_steps)

      # VERIFICATION
      # Remove halos
      u = states['u'][:, :, 1]
      v = states['v'][:, :, 1]

      # Compute the analytical solution at the final time.
      t_final = num_steps * dt
      u_analytic_fc, v_analytic_cf, _ = (
          taylor_green_vortex_2d.analytical_solution_u_v_p(
              x_c_b, x_f_b, y_c_b, y_f_b, t_final, spec
          )
      )
      u_analytic_fc = np.squeeze(u_analytic_fc, axis=2)
      v_analytic_cf = np.squeeze(v_analytic_cf, axis=2)

      u_error = test_util.l_infinity_error(u, u_analytic_fc)
      v_error = test_util.l_infinity_error(v, v_analytic_cf)
      max_error = max(u_error, v_error)
      return max_error

    n_per_core_vec = np.array([16, 32, 64, 128])

    n_vec = n_per_core_vec * 1  # 1 core.
    error_vec = np.zeros_like(n_per_core_vec, dtype=np.float64)

    for j, n_per_core in enumerate(n_per_core_vec):
      error_vec[j] = run_sim_and_get_error(n_per_core)

    # For 2nd-order accuracy, error should scale as ~ N^-2, for N grid points.
    error_order = test_util.compute_power_exponent(n_vec, error_vec)
    expected_error_order = -2
    tol = 0.1
    self.assertGreater(abs(error_order), abs(expected_error_order) - tol)

  def test_second_order_spatial_convergence_quick(self):
    # SETUP
    dtype = jnp.float32
    dt = 0.0005
    num_steps = 4000
    convection_scheme = 'quick'
    poisson_solver_type = PoissonSolverType.FAST_DIAGONALIZATION
    spec = ProblemSpec(domain_size=1.0, viscosity=6.25e-4, amplitude=1.0)

    def run_sim_and_get_error(num_points):
      # No sharding.
      # Set simulation parameters

      # No grid stretching.
      cfg_ext = taylor_green_vortex_2d.get_cfg(
          num_points,
          dt,
          spec,
          poisson_solver_type,
          convection_scheme,
      )
      cfg = config.config_from_config_external(cfg_ext)

      x_c, x_f = cfg.x_c, cfg.x_f
      y_c, y_f = cfg.y_c, cfg.y_f

      # Create broadcastable jax arrays (3D) and cast to desired dtype, assuming
      # we are starting with float64 numpy arrays.
      x_c_b = jnp.array(x_c[:, jnp.newaxis, jnp.newaxis], dtype=dtype)
      x_f_b = jnp.array(x_f[:, jnp.newaxis, jnp.newaxis], dtype=dtype)
      y_c_b = jnp.array(y_c[jnp.newaxis, :, jnp.newaxis], dtype=dtype)
      y_f_b = jnp.array(y_f[jnp.newaxis, :, jnp.newaxis], dtype=dtype)
      states = _run_a_simulation(cfg, spec, num_steps)

      # VERIFICATION
      # Remove halos
      u = states['u'][:, :, 1]
      v = states['v'][:, :, 1]

      # Compute the analytical solution at the final time.
      t_final = num_steps * dt
      u_analytic_fc, v_analytic_cf, _ = (
          taylor_green_vortex_2d.analytical_solution_u_v_p(
              x_c_b, x_f_b, y_c_b, y_f_b, t_final, spec
          )
      )
      u_analytic_fc = np.squeeze(u_analytic_fc, axis=2)
      v_analytic_cf = np.squeeze(v_analytic_cf, axis=2)

      u_error = test_util.l_infinity_error(u, u_analytic_fc)
      v_error = test_util.l_infinity_error(v, v_analytic_cf)
      max_error = max(u_error, v_error)
      return max_error

    n_per_core_vec = np.array([16, 32, 64, 128])

    n_vec = n_per_core_vec * 1  # 1 core.
    error_vec = np.zeros_like(n_per_core_vec, dtype=np.float64)

    for j, n_per_core in enumerate(n_per_core_vec):
      error_vec[j] = run_sim_and_get_error(n_per_core)

    # For 2nd-order accuracy, error should scale as ~ N^-2, for N grid points.
    error_order = test_util.compute_power_exponent(n_vec, error_vec)
    expected_error_order = -2

    tol = 0.1
    # Note: with uniform grid, the numerical result of `error_order` is actually
    # about -3 here, higher order than expected, due to not being in the
    # asymptotic regime. This probably arises because the interpolation with
    # QUICK is 3rd order, even though the overall scheme is only 2nd order
    # asymptotically.
    self.assertGreater(abs(error_order), abs(expected_error_order) - tol)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
