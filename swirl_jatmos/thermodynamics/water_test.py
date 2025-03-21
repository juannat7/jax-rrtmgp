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

from typing import TypeAlias

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array


class WaterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Parameters used to generate expected values.
    self.wp = water.WaterParams(
        r_v=461.89,
        t_0=273.0,
        t_min=150.0,
        t_freeze=273.15,
        t_triple=273.16,
        t_icenuc=233.0,
        p_triple=611.7,
        exner_reference_pressure=1.013e5,
        lh_v0=2.258e6,
        lh_s0=2.592e6,
        cp_v=1859.0,
        cp_l=4219.9,
        cp_i=2050.0,
        num_density_iterations=10,
        max_newton_solver_iterations=4
    )

  def test_liquid_fraction(self):
    # SETUP
    temperature = jnp.array([266.0, 293.0])

    # ACTION
    liquid_frac = water.liquid_fraction(temperature, self.wp)

    # VERIFICATION
    expected_liquid_frac = np.array([0.821918, 1.0])
    np.testing.assert_allclose(
        liquid_frac, expected_liquid_frac, rtol=1e-5, atol=1e-5
    )

  def test_saturation_vapor_pressure(self):
    # SETUP
    temperature = jnp.array([273.0, 400.0])

    # ACTION
    p_v_sat = water.saturation_vapor_pressure(temperature, self.wp)

    # VERIFICATION
    expected_p_v_sat = np.array([605.3146, 128229.67])
    np.testing.assert_allclose(p_v_sat, expected_p_v_sat, rtol=1e-5, atol=1e-5)

  def test_saturation_vapor_humidity(self):
    # SETUP
    temperature = jnp.array([266.0, 293.0])
    rho = jnp.array([1.2, 1.0])

    # ACTION
    q_v_sat = water.saturation_vapor_humidity(temperature, rho, self.wp)

    # VERIFICATION
    expected_q_v_sat = np.array([0.00252687, 0.01499703])
    np.testing.assert_allclose(q_v_sat, expected_q_v_sat, rtol=1e-5, atol=1e-5)

  def test_equilibrium_phase_partition(self):
    # SETUP
    temperature = jnp.array([266.0, 293.0])
    rho = jnp.array([1.2, 1.0])
    q_t = jnp.array([0.01, 0.05])

    # ACTION
    q_liq, q_ice = water.equilibrium_phase_partition(
        temperature, rho, q_t, self.wp
    )

    # VERIFICATION
    expected_q_liq = np.array([0.0061423, 0.03500297])
    expected_q_ice = np.array([0.00133083, 0.0])
    np.testing.assert_allclose(q_liq, expected_q_liq, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(q_ice, expected_q_ice, rtol=1e-5, atol=1e-5)

  def test_saturation_adjustment(self):
    # SETUP
    theta_li = jnp.array([250.0, 300.0], dtype=jnp.float32)
    rho = jnp.array([1.0, 0.8], dtype=jnp.float32)
    q_t = jnp.array([0.01, 0.003], dtype=jnp.float32)
    p_ref = jnp.array(
        [99000, self.wp.exner_reference_pressure], dtype=jnp.float32
    )

    # ACTION
    temperature = water.saturation_adjustment(
        theta_li, rho, q_t, p_ref, self.wp
    )

    # VERIFICATION
    expected_temperature = np.array([265.04584, 300.0])
    np.testing.assert_allclose(
        temperature, expected_temperature, rtol=1e-5, atol=1e-5
    )

  def test_saturation_adjustment_3d(self):
    # SETUP
    theta_li = 250.0 * jnp.ones((2, 3, 4), dtype=jnp.float32)
    rho = 1.0 * jnp.ones_like(theta_li)
    q_t = 0.01 * jnp.ones_like(theta_li)
    p_ref = 99000 * jnp.ones_like(theta_li)

    # ACTION
    temperature = water.saturation_adjustment(
        theta_li, rho, q_t, p_ref, self.wp
    )

    # VERIFICATION
    expected_temperature = 265.04584 * np.ones_like(theta_li)
    np.testing.assert_allclose(
        temperature, expected_temperature, rtol=1e-5, atol=1e-5
    )

  def test_saturation_adjustment_mixed_3d_1d(self):
    """Test saturation works with 3D fields when `p_ref` is 1D."""
    # SETUP
    theta_li = 250.0 * jnp.ones((2, 3, 4), dtype=jnp.float32)
    rho = 1.0 * jnp.ones_like(theta_li)
    q_t = 0.01 * jnp.ones_like(theta_li)
    p_ref = 99000. * jnp.ones((1, 1, 4), dtype=jnp.float32)

    # ACTION
    temperature = water.saturation_adjustment(
        theta_li, rho, q_t, p_ref, self.wp
    )

    # VERIFICATION
    expected_temperature = 265.04584 * np.ones_like(theta_li)
    np.testing.assert_allclose(
        temperature, expected_temperature, rtol=1e-5, atol=1e-5
    )

  def test_density_and_temperature_from_theta_li_q_t(self):
    # SETUP
    theta_li = jnp.array([283.00, 500.0])
    q_t = jnp.array([0.01, 0.003])
    p_ref = jnp.array([99000, self.wp.exner_reference_pressure])
    rho_0 = jnp.array([1.0, 1.0])

    # ACTION
    rho, _ = water.density_and_temperature_from_theta_li_q_t(
        theta_li, q_t, p_ref, rho_0, self.wp
    )

    # VERIFICATION
    expected_rho = np.array([1.2067902, 0.70539343])
    np.testing.assert_allclose(rho, expected_rho, rtol=1e-5, atol=1e-5)

    # The fixed-point iteration converges much faster than I anticipated.
    # 1 iteration: rho =   1.2126
    # 2 iterations: rho =  1.2066
    # 3 iterations: rho =  1.2067938
    # 10 iterations: rho = 1.2067902  0.70539343.  SwirlLM 1.2067893
    # 30 iterations: rho = 1.2067902

  def test_density_and_temperature_from_theta_li_q_t_single_iteration_loop(
      self,
  ):
    # SETUP
    theta_li = jnp.array([283.00, 300.0])
    q_t = jnp.array([0.01, 0.003])
    p_ref = jnp.array([99000.0, 92000.0])
    rho_initial_guess = jnp.array([1.0, 1.0])

    # ACTION
    rho, temperature = (
        water.density_and_temperature_from_theta_li_q_t_single_iteration_loop(
            theta_li, q_t, p_ref, rho_initial_guess, self.wp
        )
    )

    # VERIFICATION
    expected_rho = np.array([1.2067902, 1.0974858])
    expected_temperature = np.array([285.24493, 291.86423])
    np.testing.assert_allclose(rho, expected_rho, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        temperature, expected_temperature, rtol=1e-5, atol=1e-5
    )

  def test_density_and_temperature_from_theta_li_q_t_single_iteration_loop_3d(
      self,
  ):
    """Test Newton iteration loop works with 3D fields when `p_ref` is 1D."""
    # SETUP
    theta_li = 283.00 * jnp.ones((2, 3, 4), dtype=jnp.float32)
    q_t = 0.01 * jnp.ones_like(theta_li)
    p_ref = 99000. * jnp.ones((1, 1, 4), dtype=jnp.float32)
    rho_initial_guess = 1.0 * jnp.ones_like(theta_li)

    # ACTION
    rho, _ = (
        water.density_and_temperature_from_theta_li_q_t_single_iteration_loop(
            theta_li, q_t, p_ref, rho_initial_guess, self.wp
        )
    )

    # VERIFICATION
    expected_rho = 1.2067902 * np.ones_like(theta_li)
    np.testing.assert_allclose(rho, expected_rho, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  # jax.config.update('jax_enable_x64', True)
  absltest.main()
