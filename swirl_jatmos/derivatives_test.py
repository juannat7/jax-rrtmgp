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

from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import common_ops
from swirl_jatmos import derivatives
from swirl_jatmos import stretched_grid_util
from swirl_jatmos import test_util


class DerivativesTest(parameterized.TestCase):

  def _helper_setup(
      self,
      deriv_dim: Literal[0, 1, 2],
      use_stretched_grid_in_deriv_dim: bool,
  ):
    grid_spacing = 0.7
    grid_spacings = (grid_spacing, grid_spacing, grid_spacing)

    # Set up the scale factor for use along deriv_dim, if desired.
    hc_1d = jnp.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    hf_1d = jnp.array([1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75])

    # Reshape arrays so they are broadcastable for 3D fields.
    hc = common_ops.reshape_to_broadcastable(hc_1d, deriv_dim)
    hf = common_ops.reshape_to_broadcastable(hf_1d, deriv_dim)

    use_stretched_grid = [False, False, False]
    use_stretched_grid[deriv_dim] = use_stretched_grid_in_deriv_dim
    use_stretched_grid = tuple(use_stretched_grid)

    # Initialize derivatives library.
    deriv_lib = derivatives.Derivatives(
        grid_spacings,
        use_stretched_grid,
    )

    # Create data of which to take the derivative.
    f_1d = jnp.array([7.0, 8.1, 9.3, 10.0, 9.9, 9.7, 9.5, 9.1])

    states = {}
    if use_stretched_grid_in_deriv_dim:
      states[stretched_grid_util.hc_key(deriv_dim)] = hc
      states[stretched_grid_util.hf_key(deriv_dim)] = hf

    return grid_spacing, hc_1d, hf_1d, deriv_lib, f_1d, states

  @parameterized.product(
      deriv_dim=[0, 1, 2],
      use_stretched_grid_in_deriv_dim=[True, False],
  )
  def test_deriv_node_to_face(
      self, deriv_dim: Literal[0, 1, 2], use_stretched_grid_in_deriv_dim: bool
  ):
    """Tests first derivative taking input on nodes and output on faces.

    Args:
      deriv_dim: The dimension along which to take derivatives. 0 for x, 1 for
        y, 2 for z.
      use_stretched_grid_in_deriv_dim: Whether to use a stretched grid along the
        deriv_dim.
    """
    # Setup.
    # Get common setup data.
    grid_spacing, _, hf_1d, deriv_lib, f_1d, states = self._helper_setup(
        deriv_dim, use_stretched_grid_in_deriv_dim
    )

    # Tile f to a 3D tensor by repeating num_repeats times in dimensions other
    # than deriv_dim.
    num_repeats = 3
    f = test_util.convert_to_3d_array_and_tile(f_1d, deriv_dim, num_repeats)

    # Action.
    # Compute the derivative in the desired dimension.
    dfdx_face = deriv_lib.deriv_node_to_face(f, deriv_dim, states)

    # Verification.
    # Extract a 1D slice from dfdx to compare with the expected result.
    dfdx_face_1d = test_util.extract_1d_slice_in_dim(
        dfdx_face, deriv_dim, other_idx=1
    )

    # Remove the endpoints because the results there are not meaningful/valid.
    dfdx_face_1d = dfdx_face_1d[1:-1]

    # Compute expected result, removing endpoints here too.
    if use_stretched_grid_in_deriv_dim:
      dfdx_face_1d_expected = (f_1d[1:] - f_1d[:-1]) / (
          hf_1d[1:] * grid_spacing
      )
    else:
      dfdx_face_1d_expected = (f_1d[1:] - f_1d[:-1]) / grid_spacing
    dfdx_face_1d_expected = dfdx_face_1d_expected[:-1]

    np.testing.assert_allclose(dfdx_face_1d, dfdx_face_1d_expected)

  @parameterized.product(
      deriv_dim=[0, 1, 2],
      use_stretched_grid_in_deriv_dim=[True, False],
  )
  def test_deriv_face_to_node(
      self, deriv_dim: int, use_stretched_grid_in_deriv_dim: bool
  ):
    """Tests first derivative taking input on faces and output on nodes.

    Args:
      deriv_dim: The dimension along which to take derivatives. 0 for x, 1 for
        y, 2 for z.
      use_stretched_grid_in_deriv_dim: Whether to use a stretched grid along the
        deriv_dim.
    """
    # Setup.
    # Get common setup data.
    grid_spacing, hc_1d, _, deriv_lib, f_face_1d, states = (
        self._helper_setup(deriv_dim, use_stretched_grid_in_deriv_dim)
    )

    # Tile f to a 3D tensor by repeating num_repeats times in dimensions other
    # than deriv_dim.
    num_repeats = 3
    f_face = test_util.convert_to_3d_array_and_tile(
        f_face_1d, deriv_dim, num_repeats
    )

    # Action.
    # Compute the derivative in the desired dimension.
    dfdx = deriv_lib.deriv_face_to_node(f_face, deriv_dim, states)

    # Verification.
    # Extract a 1D slice from dfdx to compare with the expected result.
    dfdx_1d = test_util.extract_1d_slice_in_dim(dfdx, deriv_dim, other_idx=1)

    # Remove the endpoints because the results there are not meaningful/valid.
    dfdx_1d = dfdx_1d[1:-1]

    # Compute expected result, removing endpoints here too.
    if use_stretched_grid_in_deriv_dim:
      dfdx_1d_expected = (f_face_1d[1:] - f_face_1d[:-1]) / (
          hc_1d[:-1] * grid_spacing
      )
    else:
      dfdx_1d_expected = (f_face_1d[1:] - f_face_1d[:-1]) / grid_spacing
    dfdx_1d_expected = dfdx_1d_expected[1:]

    np.testing.assert_allclose(dfdx_1d, dfdx_1d_expected)

  @parameterized.product(
      deriv_dim=[0, 1, 2],
      use_stretched_grid_in_deriv_dim=[True, False],
  )
  def test_deriv_centered(
      self,
      deriv_dim: int,
      use_stretched_grid_in_deriv_dim: bool,
  ):
    """Tests first derivative taking input on nodes and output on nodes.

    Args:
      deriv_dim: The dimension along which to take derivatives. 0 for x, 1 for
        y, 2 for z.
      use_stretched_grid_in_deriv_dim: Whether to use a stretched grid along the
        deriv_dim.
    """
    # Setup.
    # Get common setup data.
    grid_spacing, hc_1d, _, deriv_lib, f_1d, states = (
        self._helper_setup(deriv_dim, use_stretched_grid_in_deriv_dim)
    )

    # Tile f to a 3D tensor by repeating num_repeats times in dimensions other
    # than deriv_dim.
    num_repeats = 3
    f = test_util.convert_to_3d_array_and_tile(f_1d, deriv_dim, num_repeats)

    # Action.
    # Compute the derivative in the desired dimension.
    dfdx = deriv_lib.deriv_centered(f, deriv_dim, states)

    # Verification.
    # Extract a 1D slice from dfdx to compare with the expected result.
    dfdx_1d = test_util.extract_1d_slice_in_dim(dfdx, deriv_dim, other_idx=1)

    # Remove the endpoints because the results there are not meaningful/valid.
    dfdx_1d = dfdx_1d[1:-1]

    # Compute expected result, removing endpoints here too.
    if use_stretched_grid_in_deriv_dim:
      dfdx_1d_expected = (f_1d[2:] - f_1d[:-2]) / (
          hc_1d[1:-1] * 2 * grid_spacing
      )
    else:
      dfdx_1d_expected = (f_1d[2:] - f_1d[:-2]) / (2 * grid_spacing)

    np.testing.assert_allclose(dfdx_1d, dfdx_1d_expected)

if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
