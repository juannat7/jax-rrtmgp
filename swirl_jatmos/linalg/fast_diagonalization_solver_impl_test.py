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
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos.linalg import fast_diagonalization_solver_impl
from swirl_jatmos.utils import utils

Array: TypeAlias = jax.Array
NEUMANN_BC = fast_diagonalization_solver_impl.BCType.NEUMANN
PERIODIC_BC = fast_diagonalization_solver_impl.BCType.PERIODIC

_HALO_WIDTH = 2


def add_ones_for_matrix_halos(mat: np.ndarray, halo_width: int) -> np.ndarray:
  """Adds ones to the matrix for halos."""
  if halo_width != 0:
    n_no_halos, _ = mat.shape
    n = n_no_halos + 2 * halo_width
    mat_with_halos = np.eye(n, dtype=np.float64)
    mat_with_halos[halo_width:-halo_width, halo_width:-halo_width] = mat
    return mat_with_halos
  else:
    return mat


class FastDiagonalizationSolverImplTest(parameterized.TestCase):

  def _set_up_mesh(self, n: int = 64, halo_width: int = _HALO_WIDTH):
    nx = ny = nz = n
    length = 1.0
    domain = (0.0, length)

    # Generate the global mesh.  We enforce a homogeneous Neumann boundary
    # condition on the face.
    num_cores = 1
    x_nodes, x_faces = utils.uniform_grid(domain, num_cores, nx, halo_width)
    y_nodes, y_faces = utils.uniform_grid(domain, num_cores, ny, halo_width)
    z_nodes, z_faces = utils.uniform_grid(domain, num_cores, nz, halo_width)
    return (x_nodes, y_nodes, z_nodes, x_faces, y_faces, z_faces)

  @parameterized.parameters(0, 2)
  def test_matrix_creation_unweighted_neumann(self, halo_width: int):
    """Test the matrix creation for unweighted operator with Neumann BCs."""
    # pylint: disable=invalid-name
    # SETUP
    n_no_halos = 6
    n = n_no_halos + 2 * halo_width
    dx = 0.25

    # ACTION
    A = fast_diagonalization_solver_impl.create_matrix_dx_w_dx(
        n, dx, halo_width, NEUMANN_BC
    )

    # VERIFICATION
    # Create expected matrix.
    A_expected_no_halos = np.array(
        [
            [-1, 1, 0, 0, 0, 0],
            [1, -2, 1, 0, 0, 0],
            [0, 1, -2, 1, 0, 0],
            [0, 0, 1, -2, 1, 0],
            [0, 0, 0, 1, -2, 1],
            [0, 0, 0, 0, 1, -1],
        ],
        dtype=np.float64,
    )
    A_expected = add_ones_for_matrix_halos(A_expected_no_halos, halo_width)
    A_expected /= dx**2

    # pylint: enable=invalid-name
    np.testing.assert_allclose(A, A_expected, rtol=1e-10, atol=1e-10)

  @parameterized.parameters(0, 2)
  def test_matrix_creation_unweighted_periodic(self, halo_width: int):
    """Test the matrix creation for unweighted operator with periodic BCS."""
    # SETUP
    # pylint: disable=invalid-name
    n_no_halos = 6
    n = n_no_halos + 2 * halo_width
    dx = 0.25

    # ACTION
    A = fast_diagonalization_solver_impl.create_matrix_dx_w_dx(  # pylint: disable=invalid-name
        n, dx, halo_width, PERIODIC_BC
    )

    # VERIFICATION
    # Create expected matrix.
    A_expected_no_halos = np.array(
        [
            [-2, 1, 0, 0, 0, 1],
            [1, -2, 1, 0, 0, 0],
            [0, 1, -2, 1, 0, 0],
            [0, 0, 1, -2, 1, 0],
            [0, 0, 0, 1, -2, 1],
            [1, 0, 0, 0, 1, -2],
        ],
        dtype=np.float64,
    )
    A_expected = add_ones_for_matrix_halos(A_expected_no_halos, halo_width)
    A_expected /= dx**2

    # pylint: enable=invalid-name
    np.testing.assert_allclose(A, A_expected, rtol=1e-10, atol=1e-10)

  @parameterized.parameters(0, 2)
  def test_matrix_creation_weighted_neumann(self, halo_width: int):
    """Test the matrix creation for weighted operator with Neumann BCs."""
    # pylint: disable=invalid-name
    # SETUP
    n_no_halos = 6
    n = n_no_halos + 2 * halo_width
    dx = 0.25
    large_val = -9999  # Large value for halos; should be ignored.
    wf = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])  # Weights excluding halos.
    halos = large_val * np.ones(halo_width, dtype=np.float64)
    wf_with_halos = np.concatenate((halos, wf, halos))  # Weights with halos.

    # ACTION
    A = fast_diagonalization_solver_impl.create_matrix_dx_w_dx(
        n, dx, halo_width, NEUMANN_BC, wf_with_halos
    )

    # VERIFICATION
    # Create expected matrix.
    m = np.array([  # Main diagonal of the matrix.
        wf[1],
        wf[1] + wf[2],
        wf[2] + wf[3],
        wf[3] + wf[4],
        wf[4] + wf[5],
        wf[5],
    ])
    # pylint: disable=bad-whitespace
    # pyformat: disable
    # Note that the matrix:
    #   (1) Is symmetric.
    #   (2) Has rows sum to zero (the matrix is weakly diagonally dominant).
    A_expected_no_halos = np.array(
        [
            [-m[0], wf[1], 0,     0,     0,     0],
            [wf[1], -m[1], wf[2], 0,     0,     0],
            [0,     wf[2], -m[2], wf[3], 0,     0],
            [0,     0,     wf[3], -m[3], wf[4], 0],
            [0,     0,     0,     wf[4], -m[4], wf[5]],
            [0,     0,     0,     0,     wf[5], -m[5]],
        ],
        dtype=np.float64,
    )
    # pyformat: enable
    # pylint: enable=bad-whitespace
    A_expected = add_ones_for_matrix_halos(A_expected_no_halos, halo_width)
    A_expected /= dx**2

    # pylint: enable=invalid-name
    np.testing.assert_allclose(A, A_expected, rtol=1e-10, atol=1e-10)

  @parameterized.parameters(0, 2)
  def test_matrix_creation_weighted_periodic(self, halo_width: int):
    """Test the matrix creation for weighted operator with periodic BCs."""
    # pylint: disable=invalid-name
    # SETUP
    n_no_halos = 6
    n = n_no_halos + 2 * halo_width
    dx = 0.25
    large_val = -9999  # Large value for halos; should be ignored.
    wf = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])  # Weights excluding halos.
    halos = large_val * np.ones(halo_width, dtype=np.float64)
    wf_with_halos = np.concatenate((halos, wf, halos))  # Weights with halos.

    # ACTION
    A = fast_diagonalization_solver_impl.create_matrix_dx_w_dx(
        n, dx, halo_width, PERIODIC_BC, wf_with_halos
    )

    # VERIFICATION
    # Create expected matrix.
    m = np.array([  # Main diagonal of the matrix.
        wf[0] + wf[1],
        wf[1] + wf[2],
        wf[2] + wf[3],
        wf[3] + wf[4],
        wf[4] + wf[5],
        wf[5] + wf[0],
    ])
    # pylint: disable=bad-whitespace
    # pyformat: disable
    # Note that the matrix:
    #   (1) Is symmetric.
    #   (2) Has rows sum to zero (the matrix is weakly diagonally dominant).
    A_expected_no_halos = np.array(
        [
            [-m[0], wf[1], 0,     0,     0,     wf[0]],
            [wf[1], -m[1], wf[2], 0,     0,     0],
            [0,     wf[2], -m[2], wf[3], 0,     0],
            [0,     0,     wf[3], -m[3], wf[4], 0],
            [0,     0,     0,     wf[4], -m[4], wf[5]],
            [wf[0], 0,     0,     0,     wf[5], -m[5]],
        ],
        dtype=np.float64,
    )
    # pyformat: enable
    # pylint: enable=bad-whitespace
    A_expected = add_ones_for_matrix_halos(A_expected_no_halos, halo_width)
    A_expected /= dx**2

    # pylint: enable=invalid-name
    np.testing.assert_allclose(A, A_expected, rtol=1e-10, atol=1e-10)

  def test_plain_poisson_neumann_solver(self):
    """Test the plain Poisson solver with Neumann boundary conditions.

    Solve the Poisson equation:

        ∂/∂x (∂p/∂x) + ∂/∂y (∂p/∂y) + ∂/∂z (∂p/∂z) = f

    on a domain 0 <= {x,y,z} <= 1, with

        f(z) = 1 - 2z

    There is no variation in the x, y direction.  The boundary conditions are
    Neumann (∂p/∂n = 0) on all boundaries.

    The analytic solution is

        p(z) = z^2 / 2 - z^3 / 3 + const
    """
    # SETUP
    halo_width = _HALO_WIDTH
    x_nodes, y_nodes, z_nodes, _, _, _ = self._set_up_mesh()
    nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
    dx = x_nodes[1] - x_nodes[0]
    dy = y_nodes[1] - y_nodes[0]
    dz = z_nodes[1] - z_nodes[0]
    grid_spacings = (dx, dy, dz)
    neumann_bc = fast_diagonalization_solver_impl.BCType.NEUMANN
    bc_types = (neumann_bc, neumann_bc, neumann_bc)

    # solver = fast_diagonalization_solver_impl.PlainPoisson(
    #     (nx, ny, nz), grid_spacings, bc_types, halo_width
    # )
    solver = fast_diagonalization_solver_impl.Solver(
        (nx, ny, nz), grid_spacings, bc_types, halo_width, jnp.float32
    )

    _, _, zz = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
    f_ccc = 1 - 2 * zz

    # ACTION
    p_ccc = solver.solve(f_ccc)

    # VERIFICATION
    # Compare to the analytical solution.
    expected_p_ccc = zz**2 / 2 - zz**3 / 3

    # Remove halos when comparing solution.
    expected_p_ccc = expected_p_ccc[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]
    p_ccc = p_ccc[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]

    # Remove means
    expected_p_ccc = expected_p_ccc - np.mean(expected_p_ccc)

    np.testing.assert_allclose(p_ccc, expected_p_ccc, rtol=2e-5, atol=2e-5)

  def test_plain_poisson_periodic_solver(self):
    """Test the plain Poisson solver with periodic boundary conditions.

    Solve the Poisson equation:

        ∂/∂x (∂p/∂x) + ∂/∂y (∂p/∂y) + ∂/∂z (∂p/∂z) = f

    on a domain 0 <= {x,y,z} <= 1, with

        f(x, y) = -π^2 * (4/8 * cos(2πx) + 16/6 * cos(4πy))

    There is no variation in the z direction.  The boundary conditions are
    periodic in all dimensions.

    The analytic solution is

        p(x, y) = 1/8 * cos(2πx) + 1/6 * cos(4πy) + const
    """
    # SETUP
    halo_width = _HALO_WIDTH
    n = 128
    x_nodes, y_nodes, z_nodes, _, _, _ = self._set_up_mesh(n)
    nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
    dx = x_nodes[1] - x_nodes[0]
    dy = y_nodes[1] - y_nodes[0]
    dz = z_nodes[1] - z_nodes[0]
    grid_spacings = (dx, dy, dz)
    periodic_bc = fast_diagonalization_solver_impl.BCType.PERIODIC
    bc_types = (periodic_bc, periodic_bc, periodic_bc)

    # solver = fast_diagonalization_solver_impl.PlainPoisson(
    #     (nx, ny, nz), grid_spacings, bc_types, halo_width
    # )
    solver = fast_diagonalization_solver_impl.Solver(
        (nx, ny, nz), grid_spacings, bc_types, halo_width, jnp.float32
    )

    xx, yy, _ = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
    pi = np.pi
    f_ccc = -(pi**2) * (
        4 / 8 * jnp.cos(2 * pi * xx) + 16 / 6 * jnp.cos(4 * pi * yy)
    )

    # ACTION
    p_ccc = solver.solve(f_ccc)

    # VERIFICATION
    # Compare to the analytical solution.
    expected_p_ccc = 1/8 * jnp.cos(2 * pi * xx) + 1/6 * jnp.cos(4 * pi * yy)

    # Remove halos when comparing solution.
    expected_p_ccc = expected_p_ccc[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]
    p_ccc = p_ccc[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]

    # Remove means
    expected_p_ccc = expected_p_ccc - np.mean(expected_p_ccc)

    # Note: the error seen in this test is larger than the previous test because
    # a higher-frequency function is used in f(x, y), namely cos(4πy).
    # Importantly, the convergence is still 2nd order in the number of grid
    # points.
    np.testing.assert_allclose(p_ccc, expected_p_ccc, rtol=3e-4, atol=3e-4)

  @parameterized.parameters(False, True)
  def test_variable_coefficient_zfn_poisson_solver(
      self, use_general_solver: bool
  ):
    """Test the variable-coefficient Poisson solver with Neumann BCs.

    Solve the variable-coefficient Poisson equation:

        ∂/∂x (w ∂p/∂x) + ∂/∂y (w ∂p/∂y) + ∂/∂z (w ∂p/∂z) = f

    on a domain 0 <= {x,y,z} <= 1, with

        w(z) = exp(-z)
        f(z) = exp(-z) * [-4π^2 / 20 * cos(2πx) + (1 - 3z + z^2)

    There is no variation in the y direction.  The boundary conditions are
    Neumann (∂p/∂n = 0) on all boundaries.

    The analytic solution is

        p(z) = (1/20) * cos(2πx) + z^2 / 2 - z^3 / 3 + const.

    Args:
      use_general_solver: Whether to use the general solver or the specialized
        variable-coefficient ZFn solver.
    """
    # SETUP
    halo_width = _HALO_WIDTH
    n = 64
    x_nodes, y_nodes, z_nodes, _, _, z_faces = self._set_up_mesh(n, halo_width)
    nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
    dx = x_nodes[1] - x_nodes[0]
    dy = y_nodes[1] - y_nodes[0]
    dz = z_nodes[1] - z_nodes[0]
    grid_spacings = (dx, dy, dz)
    neumann_bc = fast_diagonalization_solver_impl.BCType.NEUMANN
    bc_types = (neumann_bc, neumann_bc, neumann_bc)

    w_fun = lambda z: np.exp(-z)

    w_c = w_fun(z_nodes)
    w_f = w_fun(z_faces)

    assert w_c.dtype == np.float64
    assert w_f.dtype == np.float64

    if use_general_solver:
      solver = fast_diagonalization_solver_impl.Solver(
          (nx, ny, nz),
          grid_spacings,
          bc_types,
          halo_width,
          jnp.float32,
          w_Az_f=w_f,
          w_Bz_c=w_c,
      )
    else:
      solver = fast_diagonalization_solver_impl.VariableCoefficientZFn(
          (nx, ny, nz),
          grid_spacings,
          bc_types,
          halo_width,
          jnp.float32,
          w_c,
          w_f,
      )

    xx, _, zz = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')

    f_ccc = jnp.exp(-zz) * (
        -4 * np.pi**2 / 20 * jnp.cos(2 * np.pi * xx) + (1 - 3 * zz + zz**2)
    )

    # ACTION
    p_ccc = solver.solve(f_ccc)

    # VERIFICATION
    # Compare to the analytical solution.
    expected_p_ccc = (1 / 20) * jnp.cos(2 * np.pi * xx) + zz**2 / 2 - zz**3 / 3

    # Remove halos when comparing solution.
    if halo_width != 0:
      expected_p_ccc = expected_p_ccc[
          halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
      ]
      p_ccc = p_ccc[
          halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
      ]

    # Remove means
    p_ccc = p_ccc - np.mean(p_ccc)
    expected_p_ccc = expected_p_ccc - np.mean(expected_p_ccc)

    np.testing.assert_allclose(p_ccc, expected_p_ccc, rtol=6e-5, atol=6e-5)

  @parameterized.parameters(False, True)
  def test_periodic_xy_variable_coefficient_zfn_poisson_solver(
      self, use_general_solver: bool
  ):
    """Test the variable-coefficient Poisson solver with periodic/neumann BCs.

    Solve the variable-coefficient Poisson equation:

        ∂/∂x (w ∂p/∂x) + ∂/∂y (w ∂p/∂y) + ∂/∂z (w ∂p/∂z) = f

    on a domain 0 <= {x,y,z} <= 1, with

        w(z) = exp(-z)
        f(z) = exp(-z) * [-4π^2 / 8 * sin(2πx) - 4π^2 / 6 * sin(2πy) +
                          (1 - 3z + z^2)]

    There is no variation in the y direction.  The boundary conditions are
    periodic in the x and y directions, and Neumann in the z direction, ∂p/∂z=0
    at z=0 and z=1.

    The analytic solution is

        p(z) = sin(2πx) / 8 + sin(2πy) / 6 + z^2 / 2 - z^3 / 3 + const.

    Args:
      use_general_solver: Whether to use the general solver or the specialized
        variable-coefficient ZFn solver.
    """
    # SETUP
    halo_width = _HALO_WIDTH
    n = 128
    x_nodes, y_nodes, z_nodes, _, _, z_faces = self._set_up_mesh(n, halo_width)
    nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
    dx = x_nodes[1] - x_nodes[0]
    dy = y_nodes[1] - y_nodes[0]
    dz = z_nodes[1] - z_nodes[0]
    grid_spacings = (dx, dy, dz)
    bc_types = (PERIODIC_BC, PERIODIC_BC, NEUMANN_BC)

    w_fun = lambda z: np.exp(-z)

    w_c = w_fun(z_nodes)
    w_f = w_fun(z_faces)

    assert w_c.dtype == np.float64
    assert w_f.dtype == np.float64

    if use_general_solver:
      solver = fast_diagonalization_solver_impl.Solver(
          (nx, ny, nz),
          grid_spacings,
          bc_types,
          halo_width,
          jnp.float32,
          w_Az_f=w_f,
          w_Bz_c=w_c,
      )
    else:
      solver = fast_diagonalization_solver_impl.VariableCoefficientZFn(
          (nx, ny, nz),
          grid_spacings,
          bc_types,
          halo_width,
          jnp.float32,
          w_c,
          w_f,
      )

    xx, yy, zz = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')

    pi = np.pi
    f_ccc = jnp.exp(-zz) * (
        -4 * pi**2 / 8 * jnp.sin(2 * pi * (xx + 0.11))
        - 4 * pi**2 / 6 * jnp.sin(2 * pi * yy)
        + (1 - 3 * zz + zz**2)
    )

    # ACTION
    p_ccc = solver.solve(f_ccc)

    # VERIFICATION
    # Compare to the analytical solution.
    expected_p_ccc = (
        jnp.sin(2 * pi * (xx + 0.11)) / 8
        + jnp.sin(2 * pi * yy) / 6
        + zz**2 / 2
        - zz**3 / 3
    )

    # Remove halos when comparing solution.
    if halo_width != 0:
      expected_p_ccc = expected_p_ccc[
          halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
      ]
      p_ccc = p_ccc[
          halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
      ]

    # Remove means
    p_ccc = p_ccc - np.mean(p_ccc)
    expected_p_ccc = expected_p_ccc - np.mean(expected_p_ccc)
    np.testing.assert_allclose(p_ccc, expected_p_ccc, rtol=6e-5, atol=6e-5)


if __name__ == '__main__':
  absltest.main()
