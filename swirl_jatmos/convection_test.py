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

from collections.abc import Callable
import functools
from typing import Literal, TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import convection
from swirl_jatmos import derivatives
from swirl_jatmos import test_util

Array: TypeAlias = jax.Array


def initial_condition_gaussian(
    x: np.ndarray, velocity_sign: Literal[1.0, -1.0]
) -> np.ndarray:
  """Creates a Gaussian pulse for a convection test."""
  if velocity_sign == 1.0:  # Place pulse towards the left side of the domain.
    x0 = 0.3
  else:  # Place pulse towards the right side of the domain.
    x0 = 0.7
  phi = np.exp(-((x - x0) ** 2) / 0.07**2)
  return phi


def initial_condition_squarewave(
    x: np.ndarray, velocity_sign: Literal[1.0, -1.0]
) -> np.ndarray:
  """Creates a square-wave pulse for a convection test."""
  phi = np.zeros_like(x)
  if velocity_sign == 1.0:  # Place pulse towards the left side of the domain.
    ind = np.logical_and(x >= 0.05, x <= 0.25)
  else:  # Place pulse towards the right side of the domain.
    ind = np.logical_and(x >= 0.75, x <= 0.95)
  phi[ind] = 1
  return phi


def step_rk4(dt, phi, rhs: Callable[[Array], Array]):
  """Performs the 4th-order Runge-Kutta step."""
  k1 = rhs(phi)
  k2 = rhs(phi + dt * k1 / 2)
  k3 = rhs(phi + dt * k2 / 2)
  k4 = rhs(phi + dt * k3)
  phi_new = phi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
  return phi_new


class ConvectionSchemeNumericalPropertiesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dt = 0.005

  def set_up_convection_problem(
      self,
      n: int,
      velocity_sign: Literal[1, -1],
      dim: Literal[0, 1, 2],
      initial_condition_fn: Callable[[np.ndarray], np.ndarray],
  ) -> tuple[np.ndarray, derivatives.Derivatives, Array, Array, float]:
    """Sets up the problem for a convection test."""
    dtype = jnp.float64
    u0 = velocity_sign * 0.25
    x = np.linspace(0, 1, n, dtype=dtype)
    dx = float(x[1] - x[0])
    deriv_lib = derivatives.Derivatives(
        grid_spacings=(dx, dx, dx),
        use_stretched_grid=(False, False, False),
    )
    phi = jnp.array(initial_condition_fn(x), dtype=dtype)
    phi = test_util.convert_to_3d_array_and_tile(phi, dim, num_repeats=1)

    u = u0 * np.ones_like(phi)
    if dim == 2:
      # Enforce wall BC that normal velocity is zero on the walls, which is an
      # assumption in the advection schemes (the WENO5 scheme is particularly
      # sensitive to this).
      u[0, 0, 1] = 0
      u[0, 0, -1] = 0
    u = jnp.array(u, dtype=dtype)
    return x, deriv_lib, phi, u, u0

  @parameterized.product(
      convection_scheme=[
          'quick',
          'upwind1',
          'van_leer_upwind2',
          'weno3',
          'weno5_js',
          'weno5_z'
      ],
      velocity_sign=[1.0, -1.0],
      dim=[0, 1, 2],
  )
  def test_convergence_of_uniform_scalar_advection_gaussian(
      self,
      convection_scheme: Literal[
          'upwind1', 'quick', 'van_leer_upwind2', 'weno3', 'weno5_js', 'weno5_z'
      ],
      velocity_sign: Literal[1.0, -1.0],
      dim: Literal[0, 1, 2],
  ):
    # SETUP
    initial_condition_fn = functools.partial(
        initial_condition_gaussian, velocity_sign=velocity_sign
    )
    n_vec = np.array([256, 512, 1024])
    err_vec = []
    for n in n_vec:
      x, deriv_lib, phi, u, u0 = self.set_up_convection_problem(
          n, velocity_sign, dim, initial_condition_fn
      )

      def rhs(
          phi: Array,
          deriv_lib: derivatives.Derivatives,
          u: Array,
          dim: Literal[0, 1, 2],
      ):
        # Set the velocity into the desired dimension.
        if dim == 0:
          u_fcc = u
          v_cfc = jnp.zeros_like(u)
          w_ccf = jnp.zeros_like(u)
        elif dim == 1:
          u_fcc = jnp.zeros_like(u)
          v_cfc = u
          w_ccf = jnp.zeros_like(u)
        else:
          u_fcc = jnp.zeros_like(u)
          v_cfc = jnp.zeros_like(u)
          w_ccf = u

        # Compute the convective fluxes of the scalar
        flux_x_fcc, flux_y_cfc, flux_z_ccf = convection.convective_flux_scalar(
            rho_xxc=jnp.ones((1, 1, 1), dtype=phi.dtype),
            rho_xxf=jnp.ones((1, 1, 1), dtype=phi.dtype),
            phi_ccc=phi,
            u_fcc=u_fcc,
            v_cfc=v_cfc,
            w_ccf=w_ccf,
            halo_width=0,
            interp_method=convection_scheme,
        )
        # Return the negative divergence of the flux.
        return -deriv_lib.divergence_ccc(flux_x_fcc, flux_y_cfc, flux_z_ccf, {})

      rhs_reduced = functools.partial(rhs, deriv_lib=deriv_lib, u=u, dim=dim)

      # ACTION
      t_final = 1.0
      nsteps = int(t_final / self.dt)
      for _ in range(nsteps):
        phi = step_rk4(self.dt, phi, rhs_reduced)

      phi_final = np.array(
          test_util.extract_1d_slice_in_dim(phi, dim, other_idx=0)
      )
      t_final = nsteps * self.dt  # Recalculate to get exact final time.

      exact_solution = initial_condition_fn(x - u0 * t_final)
      # Determine the l_infinity error for this resolution.
      # If dim == 2, strip halos (WENO5 puts junk into the halos).
      if dim == 2:
        phi_final = phi_final[1:-1]
        exact_solution = exact_solution[1:-1]
      err = test_util.l_infinity_error(phi_final, exact_solution)
      err_vec.append(err)

    err_vec = np.array(err_vec)
    n_vec = np.array(n_vec)

    # VERIFICATION
    # Check that error at the highest resolution is small.
    abs_error_tol = {
        'upwind1': 0.05,
        'quick': 2e-4,
        'van_leer_upwind2': 3e-3,
        'weno3': 0.003,
        'weno5_js': 5e-7,
        'weno5_z': 2e-7,
    }
    self.assertLessEqual(err_vec[-1], abs_error_tol[convection_scheme])

    # Check the order of convergence.
    error_order = test_util.compute_power_exponent(n_vec, err_vec)

    order_and_tol = {
        'upwind1': (-1, 0.15),
        'quick': (-2, 0.05),
        # The l_infinity error converges slowly for this scheme (numerical order
        # is about -1.27 for the resolutions used). The l_2 error has a
        # numerical order closer to -2.
        'van_leer_upwind2': (-2, 0.75),
        'weno3': (-2, 0.1),
        'weno5_js': (-5, 1.3),
        'weno5_z': (-5, 3.5)  # Tol is high because error starts small.
    }

    expected_order, order_tol = order_and_tol[convection_scheme]
    self.assertAlmostEqual(error_order, expected_order, delta=order_tol)

  @parameterized.product(
      convection_scheme=['upwind1', 'van_leer_upwind2'],
      velocity_sign=[1.0, -1.0],
      dim=[0, 1, 2],
  )
  def test_monotone_property_using_square_wave_advection(
      self,
      convection_scheme: Literal['upwind1'],
      velocity_sign: Literal[1.0, -1.0],
      dim: Literal[0, 1, 2],
  ):
    """Checks that advection of a square wave does not produce negative values.

    This tests check the monotonicity-preserving property of convection schemes,
    which should arise for all Total Variation Diminishing (TVD) schemes.

    Args:
      convection_scheme: The convection scheme to use.
      velocity_sign: The sign of the velocity, determining the convection
        direction.
      dim: The dimension in which the 1D convection problem is performed.
    """
    n = 64
    initial_condition_fn = functools.partial(
        initial_condition_squarewave, velocity_sign=velocity_sign
    )
    _, deriv_lib, phi, u, _ = self.set_up_convection_problem(
        n, velocity_sign, dim, initial_condition_fn
    )

    def rhs(
        phi: Array,
        deriv_lib: derivatives.Derivatives,
        u: Array,
        dim: Literal[0, 1, 2],
    ):
      # Set the velocity into the desired dimension.
      if dim == 0:
        u_fcc = u
        v_cfc = jnp.zeros_like(u)
        w_ccf = jnp.zeros_like(u)
      elif dim == 1:
        u_fcc = jnp.zeros_like(u)
        v_cfc = u
        w_ccf = jnp.zeros_like(u)
      else:
        u_fcc = jnp.zeros_like(u)
        v_cfc = jnp.zeros_like(u)
        w_ccf = u

      # Compute the convection term of the scalar.
      (flux_x_fcc, flux_y_cfc, flux_z_ccf) = convection.convective_flux_scalar(
          rho_xxc=jnp.ones((1, 1, 1), dtype=phi.dtype),
          rho_xxf=jnp.ones((1, 1, 1), dtype=phi.dtype),
          phi_ccc=phi,
          u_fcc=u_fcc,
          v_cfc=v_cfc,
          w_ccf=w_ccf,
          halo_width=0,
          interp_method=convection_scheme,
      )

      sg_map = {}
      return -deriv_lib.divergence_ccc(
          flux_x_fcc, flux_y_cfc, flux_z_ccf, sg_map
      )
    rhs_reduced = functools.partial(rhs, deriv_lib=deriv_lib, u=u, dim=dim)

    # Action.
    t_final = 1.0
    nsteps = int(t_final / self.dt)
    for _ in range(nsteps):
      phi = step_rk4(self.dt, phi, rhs_reduced)

    phi_final = np.array(
        test_util.extract_1d_slice_in_dim(phi, dim, other_idx=0)
    )

    # Verification.
    eps = 1e-13
    is_nonnegative = phi_final + eps >= 0
    self.assertTrue(np.all(is_nonnegative))

if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
