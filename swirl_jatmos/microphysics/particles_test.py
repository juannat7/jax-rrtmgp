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
import jax.numpy as jnp
import numpy as np
from swirl_jatmos.microphysics import microphysics_config
from swirl_jatmos.microphysics import particles


class ParticlesTest(absltest.TestCase):

  def test_marshall_palmer_distribution_parameter_lambda_inverse(self):
    """Check the Marshall Palmer lambda parameter for rain, snow, and ice."""
    # SETUP
    pp_rain = microphysics_config.RainParams()
    pp_snow = microphysics_config.SnowParams()
    q = jnp.array([1e-3, 0.0])
    rho = jnp.array([1.2, 1.2])
    pp_ice = microphysics_config.IceParams()

    # ACTION
    lambda_inverse_rain = (
        particles.marshall_palmer_distribution_parameter_lambda_inverse(
            pp_rain, rho, q
        )
    )
    lambda_inverse_snow = (
        particles.marshall_palmer_distribution_parameter_lambda_inverse(
            pp_snow, rho, q
        )
    )
    lambda_inverse_ice = (
        particles.marshall_palmer_distribution_parameter_lambda_inverse(
            pp_ice, rho, q
        )
    )

    # VERIFICATION
    expected_lambda_inverse_rain = jnp.array([1 / 4278.5303, 0.0])
    expected_lambda_inverse_snow = jnp.array([1 / 2189.8105, 0.0])
    expected_lambda_inverse_ice = jnp.array([0.000263, 0.0])

    np.testing.assert_allclose(
        lambda_inverse_rain, expected_lambda_inverse_rain, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        lambda_inverse_snow, expected_lambda_inverse_snow, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        lambda_inverse_ice, expected_lambda_inverse_ice, rtol=1e-5, atol=1e-5
    )


if __name__ == '__main__':
  absltest.main()
