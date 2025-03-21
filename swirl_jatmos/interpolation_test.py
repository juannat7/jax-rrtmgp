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

import functools
from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from swirl_jatmos import interpolation
from swirl_jatmos import test_util
from swirl_jatmos.utils import utils


# Helper function for testing WENO interpolation.
def sin_cell_avg(x: npt.ArrayLike, dx: float) -> np.ndarray:
  """Cell average of sin(x) at x, given uniform grid spacing dx."""
  return (np.cos(x - dx / 2) - np.cos(x + dx / 2)) / dx


class InterpolationTest(parameterized.TestCase):

  @parameterized.product(dim=[0, 1, 2])
  def test_centered_node_to_face(self, dim):
    # Setup.
    f_1d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    # Tile f to a 3D tensor by repeating num_repeats times in dimensions other
    # than dim.
    num_repeats = 3
    f = test_util.convert_to_3d_array_and_tile(f_1d, dim, num_repeats)

    # ACTION
    # Perform 2nd-order, centered interpolation,
    f_face = interpolation.centered_node_to_face(f, dim)

    # VERIFICATION
    # Extract a 1D slice to compare with the expected result.
    f_face_1d = test_util.extract_1d_slice_in_dim(f_face, dim, other_idx=1)

    # Remove the endpoints because the results there are not meaningful/valid.
    f_face_1d = f_face_1d[1:-1]

    # Compute expected result, removing endpoints here too.
    f_face_1d_expected = (f_1d[1:-1] + f_1d[:-2]) / 2

    np.testing.assert_allclose(f_face_1d, f_face_1d_expected)

  @parameterized.product(dim=[0, 1, 2])
  def test_centered_face_to_node_interpolation(self, dim):
    # Setup.
    f_1d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    # Tile f to a 3D tensor by repeating num_repeats times in dimensions other
    # than dim.
    num_repeats = 3
    f = test_util.convert_to_3d_array_and_tile(f_1d, dim, num_repeats)

    # ACTION
    # Perform 2nd-order, centered interpolation,
    f_face = interpolation.centered_face_to_node(f, dim)

    # VERIFICATION
    # Extract a 1D slice to compare with the expected result.
    f_face_1d = test_util.extract_1d_slice_in_dim(f_face, dim, other_idx=1)

    # Remove the endpoints because the results there are not meaningful/valid.
    f_face_1d = f_face_1d[1:-1]

    # Compute expected result, removing endpoints here too.
    f_face_1d_expected = (f_1d[1:-1] + f_1d[2:]) / 2

    np.testing.assert_allclose(f_face_1d, f_face_1d_expected)

  def test_interp4_convergence(self):
    """Check convergence of 4th-order centered interpolation."""
    # SETUP
    domain = (0, 2 * np.pi)
    n_vec = np.array([64, 128, 256, 512, 1024])
    error_vec = np.zeros_like(n_vec, dtype=np.float64)

    for j, n in enumerate(n_vec):
      # Create a uniform periodic gric.
      x_c, x_f = utils.uniform_grid(
          domain, num_cores=1, n_per_core=n, halo_width=0
      )
      u = np.sin(x_c)

      # ACTION
      u_face = interpolation.interp4_node_to_face(u, dim=0)
      u_exact_f = np.sin(x_f)
      max_error = jnp.max(jnp.abs(u_face - u_exact_f))
      error_vec[j] = max_error

    # VERIFICATION
    error_order = test_util.compute_power_exponent(n_vec, error_vec)
    expected_error_order = -4
    tol = 0.05
    self.assertAlmostEqual(error_order, expected_error_order, delta=tol)

  def test_interp4_cell_avg_node_to_face_convergence(self):
    """Check convergence of 4th-order interpolation from cell averages."""
    # SETUP
    domain = (0, 2 * np.pi)
    n_vec = np.array([64, 128, 256, 512, 1024])
    error_vec = np.zeros_like(n_vec, dtype=np.float64)

    for j, n in enumerate(n_vec):
      # Create a uniform periodic gric.
      x_c, x_f = utils.uniform_grid(
          domain, num_cores=1, n_per_core=n, halo_width=0
      )
      dx = float(x_c[1] - x_c[0])
      u_cell_avg = sin_cell_avg(x_c, dx)

      # ACTION
      u_face = interpolation.interp4_cell_avg_node_to_face(u_cell_avg, dim=0)
      u_exact_f = np.sin(x_f)
      max_error = jnp.max(jnp.abs(u_face - u_exact_f))
      error_vec[j] = max_error

    # VERIFICATION
    error_order = test_util.compute_power_exponent(n_vec, error_vec)
    expected_error_order = -4
    tol = 0.05
    self.assertAlmostEqual(error_order, expected_error_order, delta=tol)

  def test_interp4_cell_avg_face_to_node_convergence(self):
    """Check convergence of 4th-order interpolation from cell averages."""
    # SETUP
    domain = (0, 2 * np.pi)
    n_vec = np.array([64, 128, 256, 512, 1024])
    error_vec = np.zeros_like(n_vec, dtype=np.float64)

    for j, n in enumerate(n_vec):
      # Create a uniform periodic gric.
      x_c, x_f = utils.uniform_grid(
          domain, num_cores=1, n_per_core=n, halo_width=0
      )
      dx = float(x_c[1] - x_c[0])
      u_cell_avg_f = sin_cell_avg(x_f, dx)

      # ACTION
      u_c = interpolation.interp4_cell_avg_face_to_node(u_cell_avg_f, dim=0)
      u_exact_c = np.sin(x_c)
      max_error = jnp.max(jnp.abs(u_c - u_exact_c))
      error_vec[j] = max_error

    # VERIFICATION
    error_order = test_util.compute_power_exponent(n_vec, error_vec)
    expected_error_order = -4
    tol = 0.05
    self.assertAlmostEqual(error_order, expected_error_order, delta=tol)

  @parameterized.product(method=['weno3', 'weno5_js', 'weno5_z'])
  def test_weno_node_to_face_convergence(
      self, method: Literal['weno3', 'weno5_js', 'weno5_z']
  ):
    """Check convergence of WENO(3,5) for node-to-face interpolation.

    The function we will interpolate is sin(x). WENO5 is based on having the
    cell average values at discrete points, and determines a pointwise value
    on an interface.
    For a uniform grid, the cell average values is computed by
          1/dx * integral(f, x-dx/2, x+dx/2).
    For sin(x), the integral can be computed analytically, so the cell average
    at x is:
          (cos(x - dx/2) - cos(x + dx/2)) / dx

    Args:
      method: The WENO method to use: 'weno3', 'weno5_js', or 'weno5_z'.
    """
    # SETUP
    domain = (0, 2 * np.pi)
    n_vec = np.array([64, 128, 256, 512, 1024])
    error_vec_plus = np.zeros_like(n_vec, dtype=np.float64)
    error_vec_minus = np.zeros_like(n_vec, dtype=np.float64)
    if method == 'weno3':
      interp_node_to_face_fn = interpolation.weno3_node_to_face
      expected_error_order = -3
    elif method == 'weno5_js':
      interp_node_to_face_fn = interpolation.weno5_js_node_to_face
      expected_error_order = -5
    elif method == 'weno5_z':
      interp_node_to_face_fn = interpolation.weno5_z_node_to_face
      expected_error_order = -5
    else:
      raise ValueError(f'Unknown method: {method}')

    for j, n in enumerate(n_vec):
      # Create a uniform periodic grid.
      x_c, x_f = utils.uniform_grid(
          domain, num_cores=1, n_per_core=n, halo_width=0
      )
      dx = float(x_c[1] - x_c[0])
      u_cell_avg = sin_cell_avg(x_c, dx)

      # ACTION
      u_face_plus, u_face_minus = interp_node_to_face_fn(u_cell_avg, dim=0)
      u_exact_f = jnp.sin(x_f)
      max_error_plus = jnp.max(jnp.abs(u_face_plus - u_exact_f))
      max_error_minus = jnp.max(jnp.abs(u_face_minus - u_exact_f))
      error_vec_plus[j] = max_error_plus
      error_vec_minus[j] = max_error_minus

    # VERIFICATION
    # For kth-order accuracy, error should scale as ~ N^-k, for N grid points.
    error_order_plus = test_util.compute_power_exponent(n_vec, error_vec_plus)
    error_order_minus = test_util.compute_power_exponent(n_vec, error_vec_minus)

    tol = 0.10
    self.assertGreater(abs(error_order_plus), abs(expected_error_order) - tol)
    self.assertGreater(abs(error_order_minus), abs(expected_error_order) - tol)

  @parameterized.product(method=['weno3', 'weno5_js', 'weno5_z'])
  def test_weno_face_to_node_convergence(
      self, method: Literal['weno3', 'weno5_js', 'weno5_z']
  ):
    """Check convergence of WENO(3,5) for face-to-node interpolation.

    For further details, see docstring for test_weno_node_to_face_convergence.

    Args:
      method: The WENO method to use, either 'weno3', 'weno5_js', or 'weno5_z'.
    """
    # SETUP
    domain = (0, 2 * np.pi)
    n_vec = np.array([64, 128, 256, 512, 1024])
    error_vec_plus = np.zeros_like(n_vec, dtype=np.float64)
    error_vec_minus = np.zeros_like(n_vec, dtype=np.float64)
    if method == 'weno3':
      interp_face_to_node_fn = interpolation.weno3_face_to_node
      expected_error_order = -3
    elif method == 'weno5_js':
      interp_face_to_node_fn = interpolation.weno5_js_face_to_node
      expected_error_order = -5
    elif method == 'weno5_z':
      interp_face_to_node_fn = interpolation.weno5_z_face_to_node
      expected_error_order = -5
    else:
      raise ValueError(f'Unknown method: {method}')

    for j, n in enumerate(n_vec):
      # Create a uniform periodic grid.
      x_c, x_f = utils.uniform_grid(
          domain, num_cores=1, n_per_core=n, halo_width=0
      )
      dx = float(x_c[1] - x_c[0])
      u_cell_avg_f = sin_cell_avg(x_f, dx)

      # ACTION
      u_plus_c, u_minus_c = interp_face_to_node_fn(u_cell_avg_f, dim=0)
      u_exact_c = jnp.sin(x_c)
      max_error_plus = jnp.max(jnp.abs(u_plus_c - u_exact_c))
      max_error_minus = jnp.max(jnp.abs(u_minus_c - u_exact_c))
      error_vec_plus[j] = max_error_plus
      error_vec_minus[j] = max_error_minus

    # VERIFICATION
    # For kth-order accuracy, error should scale as ~ N^-k, for N grid points.
    error_order_plus = test_util.compute_power_exponent(n_vec, error_vec_plus)
    error_order_minus = test_util.compute_power_exponent(n_vec, error_vec_minus)

    tol = 0.1
    self.assertGreater(abs(error_order_plus), abs(expected_error_order) - tol)
    self.assertGreater(abs(error_order_minus), abs(expected_error_order) - tol)

  def test_weno5_node_to_face_for_rrtmgp(self):
    n = 16
    offset = 3  # akin to the halo width

    dim = 2
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=dim, num_repeats=2
    )

    dx = 2 * np.pi / (n - (2 * offset - 1))
    x = dx * np.arange(n, dtype=np.float32) - offset * dx
    f_1d = np.sin(x)

    dtype = jnp.float32
    f_1d = jnp.array(f_1d, dtype=dtype)
    f = convert_to_3d(f_1d)

    # ACTION
    f_face_plus, f_face_minus = interpolation.weno5_node_to_face_for_rrtmgp(
        f, dim
    )

    # VERIFICATION
    # Extract a 1D slice to compare with the expected result.
    f_face_plus_1d = test_util.extract_1d_slice_in_dim(
        f_face_plus, dim, other_idx=0
    )
    f_face_minus_1d = test_util.extract_1d_slice_in_dim(
        f_face_minus, dim, other_idx=0
    )

    # Remove halos from output because results on the endpoints are not
    # meaningful/valid.
    f_face_plus_1d = f_face_plus_1d[offset:-offset]
    f_face_minus_1d = f_face_minus_1d[offset:-offset]

    expected_f_face_plus_1d = np.array([
        -0.290055, 0.27891243, 0.76181704, 1.0051062, 0.92013156,
        0.54974455, 0.00615088, -0.5419955, -0.9206312, -1.0050877,
    ])  # pyformat: disable
    expected_f_face_minus_1d = np.array([
        -0.278913, 0.29005462, 0.76466835, 1.0050876, 0.92063123,
        0.5419959, -0.00615061, -0.5497443, -0.9201313, -1.0051063,
    ])  # pyformat: disable

    np.testing.assert_allclose(
        f_face_plus_1d, expected_f_face_plus_1d, rtol=2.5e-5, atol=0
    )
    np.testing.assert_allclose(
        f_face_minus_1d, expected_f_face_minus_1d, rtol=2.5e-5, atol=0
    )


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
