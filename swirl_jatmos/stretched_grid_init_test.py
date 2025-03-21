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
import numpy as np
from swirl_jatmos import stretched_grid_init


class StretchedGridInitTest(absltest.TestCase):

  def test_create_nonperiodic_grid(self):
    # SETUP
    domain = (2.9, 9.15)
    q_c_interior = np.array([3.4, 4.4, 5.1, 6.9, 7.5, 8.6])

    # ACTION
    q_c, q_f = stretched_grid_init.create_nonperiodic_grid(q_c_interior, domain)

    # VERIFICATION
    expected_q_f = np.array([1.9, 2.9, 3.9, 4.75, 6.0, 7.2, 8.05, 9.15])
    expected_q_c = np.array([2.4, 3.4, 4.4, 5.1, 6.9, 7.5, 8.6, 9.7])
    np.testing.assert_allclose(q_c, expected_q_c, rtol=1e-12, atol=0)
    np.testing.assert_allclose(q_f, expected_q_f, rtol=1e-12, atol=0)

  def test_create_periodic_grid(self):
    # SETUP
    domain = (3.0, 17.0)
    q_c_interior = np.array([4.1, 9.1, 13.3, 15.9])

    self.assertAlmostEqual(
        q_c_interior[0] - domain[0], domain[1] - q_c_interior[-1], delta=1e-12
    )

    # ACTION
    q_c, q_f = stretched_grid_init.create_periodic_grid(q_c_interior, domain)

    # VERIFICATION
    expected_q_f = np.array([3.0, 6.6, 11.2, 14.6])
    expected_q_c = np.array([4.1, 9.1, 13.3, 15.9])
    np.testing.assert_allclose(q_c, expected_q_c, rtol=1e-12, atol=0)
    np.testing.assert_allclose(q_f, expected_q_f, rtol=1e-12, atol=0)

  def test_create_periodic_grid_exception_for_invalid_grid(self):
    # SETUP
    domain = (3.0, 17.0)
    # Create a grid that is inconsistent with the periodic domain, and as such
    # is invalid.
    q_c_interior = np.array([4.1, 9.1, 13.3, 15.7])
    self.assertNotAlmostEqual(
        q_c_interior[0] - domain[0], domain[1] - q_c_interior[-1], delta=1e-12
    )

    # ACTION / VERIFICATION
    with self.assertRaisesRegex(
        ValueError,
        "the distance of the first node to the left wall must be equal",
    ):
      _, _ = stretched_grid_init.create_periodic_grid(q_c_interior, domain)

  def test_create_nonperiodic_h(self):
    # SETUP
    q_f = np.array([1.9, 2.9, 3.9, 4.75, 6.0, 7.2, 8.05, 9.15])
    q_c = np.array([2.4, 3.4, 4.4, 5.1, 6.9, 7.5, 8.6, 9.7])

    # ACTION
    h_c, h_f = stretched_grid_init.create_nonperiodic_h(q_c, q_f)

    expected_h_f = np.array([1.0, 1.0, 1.0, 0.7, 1.8, 0.6, 1.1, 1.1])
    expected_h_c = np.array([1.0, 1.0, 0.85, 1.25, 1.2, 0.85, 1.1, 1.1])
    np.testing.assert_allclose(h_c, expected_h_c, rtol=1e-12, atol=0)
    np.testing.assert_allclose(h_f, expected_h_f, rtol=1e-12, atol=0)

  def test_create_periodic_h(self):
    # SETUP
    domain = (3.0, 17.0)
    q_f = np.array([3.0, 6.6, 11.2, 14.6])
    q_c = np.array([4.1, 9.1, 13.3, 15.9])

    # ACTION
    h_c, h_f = stretched_grid_init.create_periodic_h(q_c, q_f, domain)

    # VERIFICATION
    expected_h_f = np.array([2.2, 5.0, 4.2, 2.6])
    expected_h_c = np.array([3.6, 4.6, 3.4, 2.4])
    np.testing.assert_allclose(h_c, expected_h_c, rtol=1e-12, atol=0)
    np.testing.assert_allclose(h_f, expected_h_f, rtol=1e-12, atol=0)


if __name__ == "__main__":
  absltest.main()
