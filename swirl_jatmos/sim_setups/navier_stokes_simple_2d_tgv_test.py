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

"""Test the 2D Taylor-Green Vortex using the simple Navier Stokes module."""

import functools
from typing import Literal, NamedTuple, TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import derivatives
from swirl_jatmos import test_util
from swirl_jatmos.sim_setups import navier_stokes_simple
from swirl_jatmos.utils import utils

Array: TypeAlias = jax.Array
PartitionSpec: TypeAlias = jax.sharding.PartitionSpec
NamedSharding: TypeAlias = jax.sharding.NamedSharding


class ProblemSpec(NamedTuple):
  """Specification for the 2D Taylor-Green Vortex problem."""

  domain_size: float
  viscosity: float  # Kinematic viscosity.
  amplitude: float  # Amplitude of the initial velocity.


def analytical_solution_u(
    x: Array, y: Array, t: float, spec: ProblemSpec
) -> Array:
  """Gets the analytical solution for the 2D Taylor-Green vortex problem."""
  domain_size = spec.domain_size
  viscosity = spec.viscosity
  u0 = spec.amplitude

  tc = (domain_size / (2 * np.pi)) ** 2 / (2 * viscosity)
  u = (
      u0
      * jnp.sin(2.0 * np.pi * x / domain_size)
      * jnp.cos(2.0 * np.pi * y / domain_size)
      * jnp.exp(-t / tc)
  )
  return u


def analytical_solution_v(
    x: Array, y: Array, t: float, spec: ProblemSpec
) -> Array:
  """Gets the analytical solution for the 2D Taylor-Green vortex problem."""
  domain_size = spec.domain_size
  viscosity = spec.viscosity
  u0 = spec.amplitude

  tc = (domain_size / (2 * np.pi)) ** 2 / (2 * viscosity)
  v = (
      -u0
      * jnp.cos(2.0 * np.pi * x / domain_size)
      * jnp.sin(2.0 * np.pi * y / domain_size)
      * jnp.exp(-t / tc)
  )
  return v


def analytical_solution_p(
    x: Array, y: Array, t: float, spec: ProblemSpec
) -> Array:
  """Gets the analytical solution for the 2D Taylor-Green vortex problem."""
  domain_size = spec.domain_size
  viscosity = spec.viscosity
  u0 = spec.amplitude

  tc = (domain_size / (2 * np.pi)) ** 2 / (2 * viscosity)
  p = (
      u0**2
      / 4.0
      * (
          jnp.cos(4 * np.pi * x / domain_size)
          + jnp.cos(4 * np.pi * y / domain_size)
      )
      * jnp.exp(-2 * t / tc)
  )
  return p


def analytical_solution_u_v_p(
    xx_nodes, xx_faces, yy_nodes, yy_faces, t: float, spec: ProblemSpec
) -> tuple[Array, Array, Array]:
  """Gets the analytical solution for the 2D Taylor-Green vortex problem."""
  u_fc = analytical_solution_u(xx_faces, yy_nodes, t, spec)
  v_cf = analytical_solution_v(xx_nodes, yy_faces, t, spec)
  p_cc = analytical_solution_p(xx_nodes, yy_nodes, t, spec)
  return u_fc, v_cf, p_cc


class SimulationTest(parameterized.TestCase):

  @parameterized.parameters(('jacobi',), ('fast_diag',))
  def test_second_order_spatial_convergence(
      self, poisson_solver_type: Literal['jacobi', 'fast_diag']
  ):
    # SETUP
    dtype = jnp.float32

    def do_a_sim(n_per_core):
      # No sharding.
      # Set simulation parameters
      dt = 0.0005
      num_steps = 4000
      length = 1.0
      lx = length
      ly = length
      lz = 1.0
      spec = ProblemSpec(domain_size=length, viscosity=6.25e-4, amplitude=1.0)
      n = n_per_core
      nx_per_core = n  # Use ny = nx.
      ny_per_core = n
      nz_per_core = 1

      hw = 0
      num_cores = 1

      x_c, x_f = utils.uniform_grid((0, lx), num_cores, nx_per_core, hw)
      y_c, y_f = utils.uniform_grid((0, ly), num_cores, ny_per_core, hw)
      z_c, z_f = utils.uniform_grid((0, lz), num_cores, nz_per_core, hw)
      dx = float(x_c[1] - x_c[0])
      dy = float(y_c[1] - y_c[0])
      # Caution: if dz is too small (eg 1e-4) there appear to be floating point
      # truncation errors in the Jacobi solver, and convergence is not as good
      # as it should be.
      dz = 10

      deriv_lib = derivatives.Derivatives(
          grid_spacings=(dx, dy, dz), use_stretched_grid=(False, False, False)
      )

      # Create broadcastable jax arrays (3D) and cast to desired dtype, assuming
      # we are starting with float64 numpy arrays.
      x_c_b = jnp.array(x_c[:, jnp.newaxis, jnp.newaxis], dtype=dtype)
      x_f_b = jnp.array(x_f[:, jnp.newaxis, jnp.newaxis], dtype=dtype)
      y_c_b = jnp.array(y_c[jnp.newaxis, :, jnp.newaxis], dtype=dtype)
      y_f_b = jnp.array(y_f[jnp.newaxis, :, jnp.newaxis], dtype=dtype)
      z_c_b = jnp.array(z_c[jnp.newaxis, jnp.newaxis, :], dtype=dtype)
      z_f_b = jnp.array(z_f[jnp.newaxis, jnp.newaxis, :], dtype=dtype)

      # Initial condition function
      init_cond_fn = functools.partial(
          analytical_solution_u_v_p, t=0, spec=spec
      )

      def one_step_fn(step_id, states):
        return navier_stokes_simple.step(
            step_id, dt, states, deriv_lib, spec.viscosity, poisson_solver_type
        )

      @jax.jit
      def do_simulation(x_c, x_f, y_c, y_f, z_c, z_f):
        del z_c, z_f
        u_fcc, v_cfc, p_ccc = init_cond_fn(x_c, x_f, y_c, y_f)
        w_ccf = jnp.zeros_like(u_fcc)

        states = {'u': u_fcc, 'v': v_cfc, 'w': w_ccf, 'p': p_ccc}
        states = jax.lax.fori_loop(0, num_steps, one_step_fn, states)
        return states

      states = do_simulation(
          x_c_b,
          x_f_b,
          y_c_b,
          y_f_b,
          z_c_b,
          z_f_b,
      )

      # VERIFICATION
      u = states['u']
      v = states['v']
      # Compute the analytical solution at the final time.
      t_final = num_steps * dt
      u_analytic_fc, v_analytic_cf, _ = analytical_solution_u_v_p(
          x_c_b, x_f_b, y_c_b, y_f_b, t_final, spec
      )

      u_error = test_util.l_infinity_error(u, u_analytic_fc)
      v_error = test_util.l_infinity_error(v, v_analytic_cf)
      max_error = max(u_error, v_error)
      return max_error

    n_per_core_vec = np.array([16, 32, 64, 128])

    n_vec = n_per_core_vec * 1  # 1 core.
    error_vec = np.zeros_like(n_per_core_vec, dtype=np.float64)

    for j, n_per_core in enumerate(n_per_core_vec):
      error_vec[j] = do_a_sim(n_per_core)

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
  # jax.config.update('jax_enable_x64', True)
  absltest.main()
