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

from typing import Literal, TypeAlias

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from swirl_jatmos.utils import check_states_valid

Array: TypeAlias = jax.Array


class CheckStatesValidTest(parameterized.TestCase):

  @parameterized.product(method=['1', '2'])
  def test_check_no_nan_inf(self, method: Literal['1', '2']):
    if method == '1':
      check_no_nan_inf = check_states_valid.check_no_nan_inf1
    elif method == '2':
      check_no_nan_inf = check_states_valid.check_no_nan_inf2
    else:
      raise ValueError(f'Unknown method: {method}')

    # SETUP
    x = jnp.array([3.0, 4.0, 5.0], dtype=jnp.float32)
    y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    w = jnp.array([1.0, 2.0, jnp.nan], dtype=jnp.float32)
    z = jnp.array([2.0, 3.0, jnp.inf], dtype=jnp.float32)

    self.assertTrue(check_no_nan_inf([x, y]))
    self.assertFalse(check_no_nan_inf([w, x, y]))
    self.assertFalse(check_no_nan_inf([x, z, y]))


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
