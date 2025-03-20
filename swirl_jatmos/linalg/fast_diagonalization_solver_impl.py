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

"""Tensor-product based, fast diagonalization solvers for the Poisson equation.

The functionality here is used to solve the Poisson equation, i.e. ∇²p = b.
"""

import enum
from typing import TypeAlias

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

Array: TypeAlias = jax.Array


class BCType(enum.Enum):
  """An enum defining the boundary condition used.

  NEUMANN: The Neumann boundary condition, ∂p/∂n = 0 at both faces.
  PERIODIC: The periodic boundary condition.
  """

  NEUMANN = 1  # Neumann boundary condition.
  PERIODIC = 2  # Periodic boundary condition.


def create_matrix_dx_w_dx(
    n: int,
    dx: float,
    halo_width: int,
    bc_type: BCType,
    w_f: np.ndarray | None = None,
) -> np.ndarray:
  """Create a matrix representing ∂/∂x (w ∂/∂x) on a grid."""
  if w_f is None:
    w_f = np.ones((n), dtype=np.float64)
  # Remove halos from w_f.
  if halo_width != 0:
    w_f = w_f[halo_width:-halo_width]

  # Upper diagonal / lower diagonal band without halos.
  upper_diag = w_f[1:]  # Exclude the first face (the wall).

  # Main diagonal without halos
  if bc_type is BCType.NEUMANN:
    w_f2 = np.copy(w_f)
    w_f2[0] = 0.0
  elif bc_type is BCType.PERIODIC:
    w_f2 = w_f
  else:
    raise ValueError(f'Unhandled BC type: {bc_type}')
  w_f2_shifted = np.roll(w_f2, -1)
  main_diag = -(w_f2 + w_f2_shifted)

  # Add halos. Note: we add ones to the main diagonal to represent the halos,
  # and then later divide by dx^2 so the actual entries on the matrix are not
  # actually 1.0. This doesn't matter because the values solved for inside the
  # halo nodes are not used. What matters is that the halos are kept
  # independent from the interior nodes, whose solutions we do use.
  one_vec = np.ones(halo_width, dtype=np.float64)
  zero_vec = np.zeros(halo_width, dtype=np.float64)
  main_diag = np.concatenate((one_vec, main_diag, one_vec))
  upper_diag = np.concatenate((zero_vec, upper_diag, zero_vec))

  # Construct matrix.
  A = np.diag(main_diag) + np.diag(upper_diag, k=1) + np.diag(upper_diag, k=-1)  # pylint: disable=invalid-name

  # Fill in values in necessary places for periodic BCs.
  if bc_type is BCType.PERIODIC:
    A[halo_width, -(halo_width + 1)] = w_f[0]
    A[-(halo_width + 1), halo_width] = w_f[0]

  return A / dx**2


# pylint: disable=invalid-name
def perform_diagonalization(
    A: np.ndarray, w_B_c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Perform diagonalization."""
  sqrt_w_B_c = np.sqrt(w_B_c)  # Represent B^(-1/2) when B is diagonal.
  # Efficiently accomplish the transformation M = B^(-1/2) @ A @ B^(-1/2).
  M = A / (sqrt_w_B_c[:, np.newaxis] * sqrt_w_B_c[np.newaxis, :])
  eigenvalues, O = np.linalg.eigh(M)
  # Efficiently accomplish the transformation V = B^(-1/2) @ O.
  V = O / sqrt_w_B_c[:, np.newaxis]
  return eigenvalues, V
# pylint: enable=invalid-name


class VariableCoefficientZFn:
  """Solver for the variable-coefficient Poisson equation.

  Solve the variable-coefficient Poisson equation:

      ∂/∂x (w ∂p/∂x) + ∂/∂y (w ∂p/∂y) + ∂/∂z (w ∂p/∂z) = f

  where w(z) is the variable coefficent and depends only on z.  w will be given
  on both z centers & z faces.

  This solver is suitable for a staggered grid, in which w is evaluated on both
  centers and faces, as appropriate.

  The linear operator acting on p is written with a tensor product decomposition
  as:
    L = (A_x ⊗ I ⊗ B + I ⊗ A_y ⊗ B + I ⊗ I ⊗ A_z)
  Here, B represents w(z), the positive definite weighting factor that depends
  only on z.
  """

  def __init__(
      self,
      nx_ny_nz: tuple[int, int, int],
      grid_spacings: tuple[float, float, float],
      bc_types: tuple[BCType, BCType, BCType],
      halo_width: int,
      dtype: jax.typing.DTypeLike,
      w_c: np.ndarray,
      w_f: np.ndarray,
      uniform_y_2d: bool = False,
  ):
    self._halo_width = halo_width
    self._uniform_y_2d = uniform_y_2d

    # pylint: disable=invalid-name
    nx, ny, nz = nx_ny_nz
    Ax = create_matrix_dx_w_dx(nx, grid_spacings[0], halo_width, bc_types[0])
    Ay = create_matrix_dx_w_dx(ny, grid_spacings[1], halo_width, bc_types[1])
    Az = create_matrix_dx_w_dx(
        nz, grid_spacings[2], halo_width, bc_types[2], w_f
    )

    eigenvalues_x, Vx = np.linalg.eigh(Ax)
    self._eigenvalues_x = eigenvalues_x.astype(dtype)
    self.Vx = Vx.astype(dtype)

    eigenvalues_y, Vy = np.linalg.eigh(Ay)
    self._eigenvalues_y = eigenvalues_y.astype(dtype)
    self.Vy = Vy.astype(dtype)

    # For z.  Here, represent B^(-1/2) = np.diag(1 / np.sqrt(w_c)), which is a
    # diagonal matrix.
    sqrt_w_c = np.sqrt(w_c)
    # Efficiently accomplish the transformation M = B^(-1/2) @ Az @ B^(-1/2).
    M = Az / (sqrt_w_c[:, np.newaxis] * sqrt_w_c[np.newaxis, :])

    eigenvalues_z, O = np.linalg.eigh(M)
    # Efficiently accomplish the transformation Vz = B^(-1/2) @ O.
    Vz = O / sqrt_w_c[:, np.newaxis]
    self._eigenvalues_z = eigenvalues_z.astype(dtype)
    self.Vz = Vz.astype(dtype)

    # Compute a cutoff using the smallest eigenvalue, since we know that the
    # differential operators here have an eigenvector of zero, which is the one
    # that we want to avoid using.
    if uniform_y_2d:  # 2D problem
      self._cutoff = 1.01 * (
          np.min(np.abs(self._eigenvalues_x))
          + np.min(np.abs(self._eigenvalues_z))
      )
    else:  # 3D problem
      self._cutoff = 1.01 * (
          np.min(np.abs(self._eigenvalues_x))
          + np.min(np.abs(self._eigenvalues_y))
          + np.min(np.abs(self._eigenvalues_z))
      )

    # Dummy array to pass to einsum_path.
    f = np.zeros((nx, ny, nz), dtype=dtype)

    self._einsum_path = jnp.einsum_path(
        'ai,bj,ck,ijk->abc', Vx.T, Vy.T, Vz.T, f, optimize='optimal'
    )[0]
    # pylint: enable=invalid-name

  def solve(self, f: Array):
    """Solve the Poisson equation."""
    if self._uniform_y_2d:
      # For a 2D simulation with no y variation, use the specialized 2D solver,
      # which is faster and more accurate (and most importantly, stable).
      return self.solve_uniform_y(f)

    Vx, Vy, Vz = self.Vx, self.Vy, self.Vz  # pylint: disable=invalid-name

    # Step 1: Transform f to spectral space.
    fhat = jnp.einsum(
        'ai,bj,ck,ijk->abc',
        Vx.T,
        Vy.T,
        Vz.T,
        f,
        optimize=self._einsum_path,
        precision=lax.Precision.HIGHEST,
    )

    # Step 2: Solve the Poisson equation in spectral space, which amounts to
    # simply dividing by the eigenvalues.
    eigenvalues = (
        self._eigenvalues_x[:, np.newaxis, np.newaxis]
        + self._eigenvalues_y[np.newaxis, :, np.newaxis]
        + self._eigenvalues_z[np.newaxis, np.newaxis, :]
    )
    # Remove eigenvalues below cutoff.  The eigenvalue of zero correspond to a
    # constant eigenvector, which is just the mean value of the solution.  We
    # are free to set the component of this eigenvector to zero.
    inv_eigenvalues = np.where(
        np.abs(eigenvalues) > self._cutoff, 1 / eigenvalues, 0.0
    )
    uhat = inv_eigenvalues * fhat

    # Step 3: Inverse transform back to physical space.
    u = jnp.einsum(
        'ai,bj,ck,ijk->abc',
        Vx,
        Vy,
        Vz,
        uhat,
        optimize=self._einsum_path,
        precision=lax.Precision.HIGHEST,
    )

    return u

  def solve_uniform_y(self, f: Array):
    """Solve the 2D Poisson equation under the assumption of uniform y."""
    Vx, Vz = self.Vx, self.Vz  # pylint: disable=invalid-name
    # Take a slice at some y value to get a 2D array.
    f_2d = f[:, self._halo_width, :]

    # Step 1: Transform f to spectral space.
    fhat_2d = jnp.einsum(
        'ai,ck,ik->ac',
        Vx.T,
        Vz.T,
        f_2d,
        optimize='greedy',
        precision=lax.Precision.HIGHEST,
    )

    # Step 2: Solve the Poisson equation in spectral space, which amounts to
    # simply dividing by the eigenvalues.
    eigenvalues = (
        self._eigenvalues_x[:, np.newaxis] + self._eigenvalues_z[np.newaxis, :]
    )
    # Remove eigenvalues below cutoff.  The eigenvalue of zero correspond to a
    # constant eigenvector, which is just the mean value of the solution.  We
    # are free to set the component of this eigenvector to zero.
    inv_eigenvalues = np.where(
        np.abs(eigenvalues) > self._cutoff, 1 / eigenvalues, 0.0
    )
    uhat_2d = inv_eigenvalues * fhat_2d

    # Step 3: Inverse transform back to physical space.
    u_2d = jnp.einsum(
        'ai,ck,ik->ac',
        Vx,
        Vz,
        uhat_2d,
        optimize='greedy',
        precision=lax.Precision.HIGHEST,
    )

    # Reconstruct 3D u
    _, ny, _ = f.shape
    u = jnp.tile(u_2d[:, jnp.newaxis, :], reps=(1, ny, 1))
    return u


class Solver:
  """Solver for the Poisson equation with variable coefficients.

  The discretized operator is written in the form
      L = (Ax ⊗ By ⊗ Bz + Bx ⊗ Ay ⊗ Bz + Bx ⊗ By ⊗ Az)
  and the linear system is Lu = f.  In general, the Ai (i=x,y,z) should be
  regarded as the differential operators, and the Bi should be regarded as the
  positive definite weighting factors.

  In particular, this form arises from the variable-coefficient Poisson
  equation, possibly written in stretched coordinates, where the stretched
  coordinates or coordinate transforms are assumed to occur in each dimension
  independently.

  In accordance with the above assumption as to how the Ai, Bi matrices arise
  from a 2nd-order discretization of the Poisson equation, the Bi are assumed to
  be diagonal matrices.  Moreover, the Ai are assumed to be in the form
      ∂/∂x_i (w_Ai ∂/∂x_i)
  To specify the linear operator, the vector weighting factors w_Ai and w_Bi
  must be given, where the w_Bi are the diagonals of the Bi matrices.  On a
  staggered grid, the w_Ai are evaluated at coordinate faces, and the w_Bi are
  evaluated at coordinate nodes.

  This solver is suitable for a staggered grid.
  """

  # pylint: disable=invalid-name
  def __init__(
      self,
      nx_ny_nz: tuple[int, int, int],
      grid_spacings: tuple[float, float, float],
      bc_types: tuple[BCType, BCType, BCType],
      halo_width: int | tuple[int, int, int],
      dtype: jax.typing.DTypeLike,
      w_Ax_f: np.ndarray | None = None,
      w_Ay_f: np.ndarray | None = None,
      w_Az_f: np.ndarray | None = None,
      w_Bx_c: np.ndarray | None = None,
      w_By_c: np.ndarray | None = None,
      w_Bz_c: np.ndarray | None = None,
      uniform_y_2d: bool = False,
      uniform_z_2d: bool = False,
  ):
    assert not (
        uniform_y_2d and uniform_z_2d
    ), 'It is not valid to enable both `uniform_y_2d` and `uniform_z_2d`.'

    if isinstance(halo_width, int):
      halo_width = (halo_width, halo_width, halo_width)
    self._halo_width = halo_width
    self._uniform_y_2d = uniform_y_2d
    self._uniform_z_2d = uniform_z_2d

    nx, ny, nz = nx_ny_nz
    if w_Ax_f is None:
      w_Ax_f = np.ones(nx, dtype=np.float64)
    if w_Ay_f is None:
      w_Ay_f = np.ones(ny, dtype=np.float64)
    if w_Az_f is None:
      w_Az_f = np.ones(nz, dtype=np.float64)
    if w_Bx_c is None:
      w_Bx_c = np.ones(nx, dtype=np.float64)
    if w_By_c is None:
      w_By_c = np.ones(ny, dtype=np.float64)
    if w_Bz_c is None:
      w_Bz_c = np.ones(nz, dtype=np.float64)

    # Repeat this for x, y, z
    Ax = create_matrix_dx_w_dx(
        nx, grid_spacings[0], halo_width[0], bc_types[0], w_Ax_f
    )

    # Perform the diagonalization in numpy in float64 to have the best accuracy,
    # then convert to the desired dtype.  Hence, if the simulation is being
    # performed in float32, then the eigenvalues and eigenvectors will be
    # computed once in float64, but then converted to float32 to be used in the
    # simulation.
    eigenvalues_x, Vx = perform_diagonalization(Ax, w_Bx_c)
    self.eigenvalues_x = eigenvalues_x.astype(dtype)
    self.Vx = Vx.astype(dtype)

    Ay = create_matrix_dx_w_dx(
        ny, grid_spacings[1], halo_width[1], bc_types[1], w_Ay_f
    )
    eigenvalues_y, Vy = perform_diagonalization(Ay, w_By_c)
    self.eigenvalues_y = eigenvalues_y.astype(dtype)
    self.Vy = Vy.astype(dtype)

    Az = create_matrix_dx_w_dx(
        nz, grid_spacings[2], halo_width[2], bc_types[2], w_Az_f
    )
    eigenvalues_z, Vz = perform_diagonalization(Az, w_Bz_c)
    self.eigenvalues_z = eigenvalues_z.astype(dtype)
    self.Vz = Vz.astype(dtype)

    # Compute a cutoff using the smallest eigenvalue, since we know that the
    # differential operators here have an eigenvector of zero, which is the one
    # that we want to avoid using.
    if uniform_y_2d:  # 2D problem
      self._cutoff = 1.01 * (
          np.min(np.abs(self.eigenvalues_x))
          + np.min(np.abs(self.eigenvalues_z))
      )
    elif uniform_z_2d:  # 2D problem with uniform z
      self._cutoff = 1.01 * (
          np.min(np.abs(self.eigenvalues_x))
          + np.min(np.abs(self.eigenvalues_y))
      )
    else:  # 3D problem
      self._cutoff = 1.01 * (
          np.min(np.abs(self.eigenvalues_x))
          + np.min(np.abs(self.eigenvalues_y))
          + np.min(np.abs(self.eigenvalues_z))
      )

    # Dummy array to pass to einsum_path.
    f = np.zeros((nx, ny, nz), dtype=dtype)

    self._einsum_path = jnp.einsum_path(
        'ai,bj,ck,ijk->abc', self.Vx, self.Vy, self.Vz, f, optimize='optimal'
    )[0]
    # pylint: enable=invalid-name

  def solve(self, f: Array):
    """Solve the Poisson equation."""
    if self._uniform_y_2d or self._uniform_z_2d:
      # For a 2D simulation, use the specialized 2D solver, which is faster and
      # more accurate (and most importantly, stable).
      return self.solve_uniform_2d(f)

    Vx, Vy, Vz = self.Vx, self.Vy, self.Vz  # pylint: disable=invalid-name

    # Step 1: Transform f to spectral space.
    fhat = jnp.einsum(
        'ai,bj,ck,ijk->abc',
        Vx.T,
        Vy.T,
        Vz.T,
        f,
        optimize=self._einsum_path,
        precision=lax.Precision.HIGHEST,
    )

    # Step 2: Solve the Poisson equation in spectral space, which amounts to
    # simply dividing by the eigenvalues.
    eigenvalues = (
        self.eigenvalues_x[:, np.newaxis, np.newaxis]
        + self.eigenvalues_y[np.newaxis, :, np.newaxis]
        + self.eigenvalues_z[np.newaxis, np.newaxis, :]
    )
    # Remove eigenvalues below cutoff.  The eigenvalue of zero corresponds to a
    # constant eigenvector, which is just the mean value of the solution.  We
    # are free to set the component of this eigenvector to zero.
    inv_eigenvalues = np.where(
        np.abs(eigenvalues) > self._cutoff, 1 / eigenvalues, 0.0
    )
    uhat = inv_eigenvalues * fhat

    # Step 3: Inverse transform back to physical space.
    u = jnp.einsum(
        'ai,bj,ck,ijk->abc',
        Vx,
        Vy,
        Vz,
        uhat,
        optimize=self._einsum_path,
        precision=lax.Precision.HIGHEST,
    )

    return u

  def solve_uniform_2d(self, f: Array):
    """Solve the 2D Poisson equation under the assumption of uniform y or z."""
    # pylint: disable=invalid-name
    if self._uniform_z_2d:
      V1, V2 = self.Vx, self.Vy
      eigenvalues1, eigenvalues2 = self.eigenvalues_x, self.eigenvalues_y
      f_2d = f[:, :, self._halo_width[2]]
    else:  # uniform_y_2d
      V1, V2 = self.Vx, self.Vz
      eigenvalues1, eigenvalues2 = self.eigenvalues_x, self.eigenvalues_z
      f_2d = f[:, self._halo_width[1], :]
    # pylint: enable=invalid-name

    # Step 1: Transform f to spectral space.
    fhat_2d = jnp.einsum(
        'ai,ck,ik->ac',
        V1.T,
        V2.T,
        f_2d,
        optimize='greedy',
        precision=lax.Precision.HIGHEST,
    )

    # Step 2: Solve the Poisson equation in spectral space, which amounts to
    # simply dividing by the eigenvalues.
    eigenvalues = eigenvalues1[:, np.newaxis] + eigenvalues2[np.newaxis, :]
    # Remove eigenvalues below cutoff.  The eigenvalue of zero corresponds to a
    # constant eigenvector, which is just the mean value of the solution.  We
    # are free to set the component of this eigenvector to zero.
    inv_eigenvalues = np.where(
        np.abs(eigenvalues) > self._cutoff, 1 / eigenvalues, 0.0
    )
    uhat_2d = inv_eigenvalues * fhat_2d

    # Step 3: Inverse transform back to physical space.
    u_2d = jnp.einsum(
        'ai,ck,ik->ac',
        V1,
        V2,
        uhat_2d,
        optimize='greedy',
        precision=lax.Precision.HIGHEST,
    )


    # Reconstruct 3D u
    if self._uniform_z_2d:
      # u_2d is shape (nx, ny).  f is shape (nx, ny, nz).
      u = u_2d[:, :, jnp.newaxis] * jnp.ones_like(f)
    else:  # uniform_y_2d
      # u_2d is shape (nx, nz).  f is shape (nx, ny, nz).
      u = u_2d[:, jnp.newaxis, :] * jnp.ones_like(f)
    return u
