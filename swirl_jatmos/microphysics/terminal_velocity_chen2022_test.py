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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from scipy import special
from swirl_jatmos.microphysics import microphysics_config
from swirl_jatmos.microphysics import terminal_velocity_chen2022
from swirl_jatmos.microphysics import terminal_velocity_chen2022_config


class TerminalVelocityChen2022Test(absltest.TestCase):

  def test_compute_raindrop_coefficients(self):
    """Check if the raindrop coefficients are correctly computed."""
    # SETUP
    rho_air = jnp.array(1.2, dtype=jnp.float32)

    # ACTION
    coeffs = terminal_velocity_chen2022._compute_raindrop_coefficients(rho_air)

    # VERIFICATION
    expected_a = (286768.0204795, -1691643.34, 9843.240767)
    expected_b = (2.249342, 2.249342, 1.098942)
    expected_c = (0.0, 184.325, 184.325)

    np.testing.assert_allclose(coeffs.a, expected_a, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.b, expected_b, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.c, expected_c, rtol=1e-6, atol=0)

  def test_compute_ice_coefficients(self):
    """Check if the ice crystal coefficients are correctly computed."""
    # SETUP
    rho_air = jnp.array(1.2, dtype=jnp.float32)
    rho_ice = 917.0  # Apparent density of ice [kg/m^3]
    ice_table_b3_coeffs = (
        terminal_velocity_chen2022_config.compute_ice_table_b3_coeffs(rho_ice)
    )

    # ACTION
    coeffs = terminal_velocity_chen2022._compute_ice_coefficients(
        rho_air, ice_table_b3_coeffs
    )

    # VERIFICATION
    expected_a = (380.9935627, -378.9867846)
    expected_b = (0.7065393, 0.7065393)
    expected_c = (0.0, 5195.1357221)

    np.testing.assert_allclose(coeffs.a, expected_a, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.b, expected_b, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.c, expected_c, rtol=1e-6, atol=0)

  def test_compute_snow_coefficients(self):
    """Check if the snow coefficients are correctly computed."""
    # SETUP
    rho_air = jnp.array(1.2, dtype=jnp.float32)
    rho_snow = 50.0  # Apparent density of snow [kg/m^3]
    snow_table_b5_coeffs = (
        terminal_velocity_chen2022_config.compute_snow_table_b5_coeffs(rho_snow)
    )

    # ACTION
    coeffs = terminal_velocity_chen2022._compute_snow_coefficients(
        rho_air, snow_table_b5_coeffs
    )

    # VERIFICATION
    expected_a = (37.80304786, -1.44428551)
    expected_b = (0.57636626, 0.22821180)
    expected_c = (0.0, 25.344366009794136)

    np.testing.assert_allclose(coeffs.a, expected_a, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.b, expected_b, rtol=1e-6, atol=0)
    np.testing.assert_allclose(coeffs.c, expected_c, rtol=1e-6, atol=0)

  def test_rain_bulk_terminal_velocity(self):
    """Check if the rain bulk terminal velocity is correctly computed."""
    # SETUP
    rho_air = jnp.array(0.7, dtype=jnp.float32)
    q_r = jnp.array(1e-3, dtype=jnp.float32)
    rain_params = microphysics_config.RainParams()

    # ACTION
    terminal_velocity = terminal_velocity_chen2022.rain_terminal_velocity(
        rho_air, q_r, rain_params
    )

    # VERIFICATION
    # Unraveling an entire computation using the following precomputed
    # coefficients consistent with the density and rain mass fraction above.
    lam = 4895.707
    a = ([309176.2495886791], [-1823829.393420029], [13696.72701356858])
    b = ([2.2685745], [2.2685745], [1.1181745])
    c = ([0.0], [184.325], [184.325])
    a, b, c = (np.array(x) for x in (a, b, c))
    delta = 4.0
    gamma_delta = special.gamma(delta)
    expected_terminal_velocity = (
        lam**4
        / gamma_delta
        * (
            (a[0] * special.gamma(b[0] + delta))
            / (lam + c[0]) ** (b[0] + delta)
            + (a[1] * special.gamma(b[1] + delta))
            / (lam + c[1]) ** (b[1] + delta)
            + (a[2] * special.gamma(b[2] + delta))
            / (lam + c[2]) ** (b[2] + delta)
        )
    )

    np.testing.assert_allclose(
        terminal_velocity, expected_terminal_velocity, rtol=2e-6, atol=0
    )

  def test_snow_terminal_velocity(self):
    """Check if the snow bulk terminal velocity is correctly computed."""
    # SETUP
    rho_air = jnp.array(0.7, dtype=jnp.float32)
    q_s = jnp.array(1e-3, dtype=jnp.float32)
    rho_snow = 50.0  # Apparent density of snow [kg/m^3]
    snow_params = microphysics_config.SnowParams(rho=rho_snow)
    snow_table_b5_coeffs = (
        terminal_velocity_chen2022_config.compute_snow_table_b5_coeffs(rho_snow)
    )

    # ACTION
    terminal_velocity = terminal_velocity_chen2022.snow_terminal_velocity(
        rho_air, q_s, snow_params, snow_table_b5_coeffs
    )

    # VERIFICATION
    # Unraveling an entire computation using the following precomputed
    # coefficients consistent with the density and rain mass fraction above.
    lam = 2340.3284
    a = [49.0118012, -2.25309539]
    b = [0.5763663, 0.22821180401885524]
    c = [0.0, 25.3443660]

    delta = 4.0
    gamma_delta = special.gamma(delta)

    phi_0, alpha = snow_params.phi_0, snow_params.alpha
    phi_oblate = phi_0 / lam**alpha
    expected_terminal_velocity = (
        phi_oblate ** (1 / 3)
        * lam**4
        / gamma_delta
        * (
            (a[0] * special.gamma(b[0] + delta))
            / (lam + c[0]) ** (b[0] + delta)
            + (a[1] * special.gamma(b[1] + delta))
            / (lam + c[1]) ** (b[1] + delta)
        )
    )

    np.testing.assert_allclose(
        terminal_velocity, expected_terminal_velocity, rtol=3e-6, atol=0
    )

  def test_condensate_terminal_velocity_for_cloud_liquid(self):
    """Check the sedimentation velocity of cloud liquid droplets is correct."""
    # SETUP
    rho_air = jnp.array(0.7, dtype=jnp.float32)
    q_liq = jnp.array(1e-3, dtype=jnp.float32)
    rain_params = microphysics_config.RainParams()

    # ACTION
    terminal_velocity = (
        terminal_velocity_chen2022.liquid_condensate_terminal_velocity(
            rho_air, q_liq, rain_params
        )
    )

    # VERIFICATION
    # Unraveling an entire computation using the following precomputed
    # coefficients consistent with the density and cloud mass fraction above.
    a = (309176.2495886791, -1823829.393420029, 13696.72701356858)
    b = (2.2685745, 2.2685745, 1.1181745)
    c = (0.0, 184.325, 184.325)

    # Diameter of a single droplet.
    diameter = np.cbrt(0.7 * 1e-3 / 1e8 / 1e3)

    # Evaluating equation 19 of Chen et al. (2022).
    expected_terminal_velocity = 0.1 * (
        a[0] * diameter ** b[0] * np.exp(-c[0] * diameter)
        + a[1] * diameter ** b[1] * np.exp(-c[1] * diameter)
        + a[2] * diameter ** b[2] * np.exp(-c[2] * diameter)
    )

    np.testing.assert_allclose(
        terminal_velocity, expected_terminal_velocity, rtol=1e-6, atol=0
    )

  def test_condensate_terminal_velocity_for_cloud_ice(self):
    """Check the sedimentation velocity of cloud ice is correct."""
    # SETUP
    rho_air = jnp.array(0.7, dtype=jnp.float32)
    q_ice = jnp.array(1e-3, dtype=jnp.float32)
    rho_ice = 917.0  # Apparent density of ice [kg/m^3]
    ice_params = microphysics_config.IceParams(rho=rho_ice)
    ice_table_b3_coeffs = (
        terminal_velocity_chen2022_config.compute_ice_table_b3_coeffs(rho_ice)
    )

    # ACTION
    terminal_velocity = (
        terminal_velocity_chen2022.ice_condensate_terminal_velocity(
            rho_air, q_ice, ice_params, ice_table_b3_coeffs
        )
    )

    # VERIFICATION
    # Unraveling an entire computation using the following precomputed
    # coefficients consistent with the density and cloud mass fraction above.
    a = [678.99649479, -675.42006872]
    b = [0.75578913, 0.75578913]
    c = [0.0, 5195.1357221]
    # Diameter of a single droplet.
    diameter = np.cbrt(0.7 * 1e-3 / 1e8 / 917.0)
    # Evaluating equation 19 of Chen et al. (2022).
    expected_terminal_velocity = a[0] * diameter ** b[0] * np.exp(
        -c[0] * diameter
    ) + a[1] * diameter ** b[1] * np.exp(-c[1] * diameter)

    np.testing.assert_allclose(
        terminal_velocity, expected_terminal_velocity, rtol=1e-6, atol=0
    )


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
