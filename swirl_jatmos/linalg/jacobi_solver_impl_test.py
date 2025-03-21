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
from swirl_jatmos.linalg import jacobi_solver_impl

Array: TypeAlias = jax.Array

_HALO_WIDTH = 2


def _neumann_halo_update_fn(p_ccc: Array) -> Array:
  """Return a version of input with updated halos using homogeneous Neumann BC.

  The Neumann BC is applied in all 3 dimensions, i.e., ∂p/∂n = 0 on every
  boundary (face).

  Here, input `p_ccc` is evaluated on nodes in every dimension.

  Args:
    p_ccc: The input array.

  Returns:
    The input array with updated halos.
  """
  halo_width = _HALO_WIDTH
  face0_idx = halo_width - 1
  face1_idx = -halo_width

  p_ccc = p_ccc.at[face0_idx, :, :].set(p_ccc[face0_idx + 1, :, :])
  p_ccc = p_ccc.at[face1_idx, :, :].set(p_ccc[face1_idx - 1, :, :])

  p_ccc = p_ccc.at[:, face0_idx, :].set(p_ccc[:, face0_idx + 1, :])
  p_ccc = p_ccc.at[:, face1_idx, :].set(p_ccc[:, face1_idx - 1, :])

  p_ccc = p_ccc.at[:, :, face0_idx].set(p_ccc[:, :, face0_idx + 1])
  p_ccc = p_ccc.at[:, :, face1_idx].set(p_ccc[:, :, face1_idx - 1])
  return p_ccc


def _generate_global_grid_with_halos(
    n_per_core: int, num_cores: int, halo_width: int, length: float
) -> tuple[np.ndarray, np.ndarray]:
  """Generate a global grid, including halos."""
  n_total_no_halos = (n_per_core - 2 * halo_width) * num_cores
  n_total_with_halos = n_total_no_halos + 2 * halo_width

  dx = length / n_total_no_halos

  x_node_min = -3 * dx / 2
  x_node_max = length + 3 * dx / 2
  x_nodes = np.linspace(x_node_min, x_node_max, n_total_with_halos)
  x_faces = x_nodes - dx / 2
  return x_nodes, x_faces


class JacobiSolverImplTest(absltest.TestCase):

  def _set_up_mesh(self):
    halo_width = _HALO_WIDTH
    nx = ny = nz = 32
    length = 1.0

    # Generate the global mesh.  We enforce a homogeneous Neumann boundary
    # condition on the face.
    num_cores = 1
    x_nodes, x_faces = _generate_global_grid_with_halos(
        nx, num_cores, halo_width, length
    )
    y_nodes, y_faces = _generate_global_grid_with_halos(
        ny, num_cores, halo_width, length
    )
    z_nodes, z_faces = _generate_global_grid_with_halos(
        nz, num_cores, halo_width, length
    )
    return (x_nodes, y_nodes, z_nodes, x_faces, y_faces, z_faces)

  def test_plain_poisson_solver(self):
    """Checks if the plain Poisson solver can solve a problem correctly.

    Solve the Poisson equation:

        ∂/∂x (∂p/∂x) + ∂/∂y (∂p/∂y) + ∂/∂z (∂p/∂z) = f

    on a domain 0 <= {x,y,z} <= 1, with

        f(z) = 1 - 2z

    There is no variation in the x, y direction.  The boundary conditions are
    Neumann (∂p/∂n = 0) on all boundaries.

    The analytic solution is

        p(z) = z^2 / 2 - z^3 / 3 + const
    """
    dtype = jnp.float32
    halo_width = _HALO_WIDTH
    x_nodes, y_nodes, z_nodes, _, _, _ = self._set_up_mesh()
    dx = float(x_nodes[1] - x_nodes[0])
    dy = float(y_nodes[1] - y_nodes[0])
    dz = float(z_nodes[1] - z_nodes[0])
    grid_spacings = (dx, dy, dz)
    omega = 2 / 3
    num_iters = 10000
    solver = jacobi_solver_impl.PlainPoisson(
        grid_spacings, omega, num_iters, _HALO_WIDTH
    )

    _, _, zz = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
    zz = jnp.array(zz, dtype=dtype)
    f_ccc = 1 - 2 * zz
    p0_ccc = jnp.zeros_like(f_ccc)

    p_ccc_final = solver.solve(f_ccc, p0_ccc, _neumann_halo_update_fn)

    # Compare to the analytical solution.
    expected_p_ccc = zz**2 / 2 - zz**3 / 3

    # Remove halos when comparing solution.
    expected_p_ccc = expected_p_ccc[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]
    p_ccc_final = p_ccc_final[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]

    # Remove means
    expected_p_ccc = expected_p_ccc - np.mean(expected_p_ccc)
    p_ccc_final = p_ccc_final - np.mean(p_ccc_final)

    np.testing.assert_allclose(
        p_ccc_final, expected_p_ccc, rtol=5e-3, atol=5e-4
    )

  def test_variable_coefficient_zfn_poisson_solver(self):
    """Checks if the variable-coefficient Poisson problem is solved correctly.

    Solve the variable-coefficient Poisson equation:

        ∂/∂x (w ∂p/∂x) + ∂/∂y (w ∂p/∂y) + ∂/∂z (w ∂p/∂z) = f

    on a domain 0 <= {x,y,z} <= 1, with

        w(z) = exp(-z)
        f(z) = exp(-z) * (1 - 3z + z^2)

    There is no variation in the x, y direction.  The boundary conditions are
    Neumann (∂p/∂n = 0) on all boundaries.

    The analytic solution is

        p(z) = z^2 / 2 - z^3 / 3 + const.
    """
    halo_width = _HALO_WIDTH
    x_nodes, y_nodes, z_nodes, _, _, z_faces = self._set_up_mesh()
    dx = x_nodes[1] - x_nodes[0]
    dy = y_nodes[1] - y_nodes[0]
    dz = z_nodes[1] - z_nodes[0]
    grid_spacings = (dx, dy, dz)
    omega = 2 / 3
    num_iters = 10000
    solver = jacobi_solver_impl.VariableCoefficientZFn(
        grid_spacings, omega, num_iters, _HALO_WIDTH
    )

    # (nx, ny, nz) z nodes.
    _, _, zz_ccc = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')

    # (1, 1, nz) z nodes and faces.
    zz_nodes = z_nodes[jnp.newaxis, jnp.newaxis, :]
    zz_faces = z_faces[jnp.newaxis, jnp.newaxis, :]

    f_ccc = jnp.exp(-zz_ccc) * (1 - 3 * zz_ccc + zz_ccc**2)
    p0_ccc = jnp.zeros_like(f_ccc)
    w_c = jnp.exp(-zz_nodes)
    w_f = jnp.exp(-zz_faces)

    p_ccc_final = solver.solve(w_c, w_f, f_ccc, p0_ccc, _neumann_halo_update_fn)

    # Compare to the analytical solution.
    expected_p_ccc = zz_ccc**2 / 2 - zz_ccc**3 / 3

    # Remove halos when comparing solution.
    expected_p_ccc = expected_p_ccc[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]
    p_ccc_final = p_ccc_final[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]

    # Remove means
    expected_p_ccc = expected_p_ccc - np.mean(expected_p_ccc)
    p_ccc_final = p_ccc_final - np.mean(p_ccc_final)

    np.testing.assert_allclose(
        p_ccc_final, expected_p_ccc, rtol=5e-3, atol=5e-4
    )

  def test_zfn_poisson_solver_residual_one_core(self):
    """Checks that the residual is computed properly.

    This test uses the same test problem as in the previous test,
    `test_variable_coefficient_zfn_poisson_solver`.  The analytic solution is
    not required to be known in order to compute the residual.
    """
    # SETUP
    halo_width = _HALO_WIDTH
    x_nodes, y_nodes, z_nodes, _, _, z_faces = self._set_up_mesh()
    dx = x_nodes[1] - x_nodes[0]
    dy = y_nodes[1] - y_nodes[0]
    dz = z_nodes[1] - z_nodes[0]
    grid_spacings = (dx, dy, dz)
    omega = 2 / 3
    num_iters = 10000
    solver = jacobi_solver_impl.VariableCoefficientZFn(
        grid_spacings, omega, num_iters, _HALO_WIDTH
    )

    # (nx, ny, nz) z nodes.
    _, _, zz_ccc = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')

    # (1, 1, nz) z nodes and faces.
    zz_nodes = z_nodes[jnp.newaxis, jnp.newaxis, :]
    zz_faces = z_faces[jnp.newaxis, jnp.newaxis, :]

    f_ccc = jnp.exp(-zz_ccc) * (1 - 3 * zz_ccc + zz_ccc**2)
    p0_ccc = jnp.zeros_like(f_ccc)
    w_c = jnp.exp(-zz_nodes)
    w_f = jnp.exp(-zz_faces)

    # ACTION
    p_ccc_final = solver.solve(w_c, w_f, f_ccc, p0_ccc, _neumann_halo_update_fn)
    # Compute the residual
    residual = solver.residual(p_ccc_final, w_c, w_f, f_ccc)

    # VERIFICATION
    # Remove halos when comparing solution.
    hw = halo_width
    residual = residual[hw:-hw, hw:-hw, hw:-hw]

    # Compute root mean square of the residual
    residual_rms = np.sqrt(np.sum(residual**2) / residual.size)

    self.assertLess(residual_rms, 1e-3)

    np.testing.assert_allclose(residual, 0, rtol=5e-3, atol=5e-4)

  def test_three_weight_zfn_poisson_solver(self):
    """Checks if the 3-weight Poisson problem is solved correctly.

    Solve the 3-weight Poisson equation:

        ∂/∂x (w0 ∂p/∂x) + ∂/∂y (w1 ∂p/∂y) + ∂/∂z (w2 ∂p/∂z) = f

    on a domain 0 <= {x,y,z} <= 1, with

        w0(z)  = exp(-2z)
        w1     = 1
        w2(z)  = exp(-z)
        f(x,z) = exp(-2z) * (-pi^2 cos(pi*x) * (1-2x) * (z^2 / 2 - z^3 / 3))
                 + (x^2 / 2 - x^3 / 3) * exp(-z) * (1 - 3z + z^2)

    There is no variation in the y direction.  The boundary conditions are
    Neumann (∂p/∂n = 0) on all boundaries.  The coefficients w0, w1, and w2
    depend only on z, so that the z-fn version of the 3-weight solver can be
    used.

    The analytic solution is

        p(x,z) = cos(pi*x) + (x^2 / 2 - x^3 / 3) * (z^2 / 2 - z^3 / 3) + const.
    """
    halo_width = _HALO_WIDTH
    x_nodes, y_nodes, z_nodes, _, _, z_faces = self._set_up_mesh()
    dx = x_nodes[1] - x_nodes[0]
    dy = y_nodes[1] - y_nodes[0]
    dz = z_nodes[1] - z_nodes[0]
    grid_spacings = (dx, dy, dz)
    omega = 2 / 3
    num_iters = 10000
    solver = jacobi_solver_impl.ThreeWeightZFn(
        grid_spacings, omega, num_iters, _HALO_WIDTH
    )

    # (nx, ny, nz) x, y, z nodes.
    xx_ccc, _, zz_ccc = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')

    # (1, 1, nz) z nodes and faces
    zz_nodes = z_nodes[jnp.newaxis, jnp.newaxis, :]
    zz_faces = z_faces[jnp.newaxis, jnp.newaxis, :]

    f_ccc = jnp.exp(-zz_ccc) * (1 - 3 * zz_ccc + zz_ccc**2)
    p0_ccc = jnp.zeros_like(f_ccc)
    w0_xxc = jnp.exp(-2 * zz_nodes)
    w1_xxc = jnp.ones_like(zz_nodes)
    w2_xxf = jnp.exp(-zz_faces)

    term1 = jnp.exp(-2 * zz_ccc) * (
        -np.pi**2 * jnp.cos(np.pi * xx_ccc)
        + (1 - 2 * xx_ccc) * (zz_ccc**2 / 2 - zz_ccc**3 / 3)
    )
    term2 = (
        (xx_ccc**2 / 2 - xx_ccc**3 / 3)
        * jnp.exp(-zz_ccc)
        * (1 - 3 * zz_ccc + zz_ccc**2)
    )
    f_ccc = term1 + term2

    p_ccc_final = solver.solve(
        w0_xxc, w1_xxc, w2_xxf, f_ccc, p0_ccc, _neumann_halo_update_fn
    )

    # Compare to the analytical solution.
    expected_p_ccc = np.cos(np.pi * xx_ccc) + (
        xx_ccc**2 / 2 - xx_ccc**3 / 3
    ) * (zz_ccc**2 / 2 - zz_ccc**3 / 3.0)

    # Remove halos when comparing solution.
    expected_p_ccc = expected_p_ccc[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]
    p_ccc_final = p_ccc_final[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]

    # Remove means
    expected_p_ccc = expected_p_ccc - np.mean(expected_p_ccc)
    p_ccc_final = p_ccc_final - np.mean(p_ccc_final)

    np.testing.assert_allclose(
        p_ccc_final, expected_p_ccc, rtol=5e-3, atol=5e-4
    )

  def test_three_weight_poisson_solver(self):
    """Checks if the 3-weight Poisson problem is solved correctly.

    Solve the 3-weight Poisson equation:

        ∂/∂x (w0 ∂p/∂x) + ∂/∂y (w1 ∂p/∂y) + ∂/∂z (w2 ∂p/∂z) = f

    on a domain 0 <= {x,y,z} <= 1, with

        w0(z)  = exp(-2z)
        w1     = 1
        w2(z)  = exp(-z)
        f(x,z) = exp(-2z) * (-pi^2 cos(pi*x) * (1-2x) * (z^2 / 2 - z^3 / 3))
                 + (x^2 / 2 - x^3 / 3) * exp(-z) * (1 - 3z + z^2)

    There is no variation in the y direction.  The boundary conditions are
    Neumann (∂p/∂n = 0) on all boundaries.  The coefficients w0, w1, and w2
    depend only on z, so that the z-fn version of the 3-weight solver can be
    used.

    The analytic solution is

        p(x,z) = cos(pi*x) + (x^2 / 2 - x^3 / 3) * (z^2 / 2 - z^3 / 3) + const.
    """
    halo_width = _HALO_WIDTH
    x_nodes, y_nodes, z_nodes, _, _, z_faces = self._set_up_mesh()
    dx = x_nodes[1] - x_nodes[0]
    dy = y_nodes[1] - y_nodes[0]
    dz = z_nodes[1] - z_nodes[0]
    grid_spacings = (dx, dy, dz)
    omega = 2 / 3
    num_iters = 10000
    solver = jacobi_solver_impl.ThreeWeight(
        grid_spacings, omega, num_iters, _HALO_WIDTH
    )

    # (nx, ny, nz) x, y, z nodes.
    nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
    xx_ccc, _, zz_ccc = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')

    # (1, 1, nz) z nodes and faces
    zz_nodes = z_nodes[jnp.newaxis, jnp.newaxis, :]
    zz_faces = z_faces[jnp.newaxis, jnp.newaxis, :]

    f_ccc = jnp.exp(-zz_ccc) * (1 - 3 * zz_ccc + zz_ccc**2)
    p0_ccc = jnp.zeros_like(f_ccc)
    w0_xxc = jnp.exp(-2 * zz_nodes)
    w1_xxc = jnp.ones_like(zz_nodes)
    w2_xxf = jnp.exp(-zz_faces)

    # Convert to full 3d (nx, ny, nz) arrays
    w0_fcc = jnp.tile(w0_xxc, reps=(nx, ny, 1))
    w1_cfc = jnp.tile(w1_xxc, reps=(nx, ny, 1))
    w2_ccf = jnp.tile(w2_xxf, reps=(nx, ny, 1))

    term1 = jnp.exp(-2 * zz_ccc) * (
        -np.pi**2 * jnp.cos(np.pi * xx_ccc)
        + (1 - 2 * xx_ccc) * (zz_ccc**2 / 2 - zz_ccc**3 / 3)
    )
    term2 = (
        (xx_ccc**2 / 2 - xx_ccc**3 / 3)
        * jnp.exp(-zz_ccc)
        * (1 - 3 * zz_ccc + zz_ccc**2)
    )
    f_ccc = term1 + term2

    p_ccc_final = solver.solve(
        w0_fcc, w1_cfc, w2_ccf, f_ccc, p0_ccc, _neumann_halo_update_fn
    )

    # Compare to the analytical solution.
    expected_p_ccc = np.cos(np.pi * xx_ccc) + (
        xx_ccc**2 / 2 - xx_ccc**3 / 3
    ) * (zz_ccc**2 / 2 - zz_ccc**3 / 3.0)

    # Remove halos when comparing solution.
    expected_p_ccc = expected_p_ccc[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]
    p_ccc_final = p_ccc_final[
        halo_width:-halo_width, halo_width:-halo_width, halo_width:-halo_width
    ]

    # Remove means
    expected_p_ccc = expected_p_ccc - np.mean(expected_p_ccc)
    p_ccc_final = p_ccc_final - np.mean(p_ccc_final)

    np.testing.assert_allclose(
        p_ccc_final, expected_p_ccc, rtol=5e-3, atol=5e-4
    )


if __name__ == '__main__':
  # jax.config.update('jax_enable_x64', True)
  absltest.main()
