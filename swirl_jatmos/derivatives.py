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

"""Library for computing derivatives.

Provides functionality for computing derivatives on 3D fields.  The aim is to
abstract away the underlying algebraic kernel operations on raw arrays.
"""

from typing import Literal, TypeAlias

import jax
from swirl_jatmos import kernel_ops
from swirl_jatmos import stretched_grid_util

Array: TypeAlias = jax.Array
StatesMap: TypeAlias = dict[str, Array]


class Derivatives:
  """Library for computing derivatives.

  Currently, the derivatives implemented are in the context of a second-order
  finite-difference code, where variable values are located at nodes on a
  colocated mesh, and fluxes are located at faces.  The computational mesh must
  be uniform in the coordinates.

  The library allows for 1D coordinate transforms in each dimension, i.e.,
  allows for x=x(q⁰), y=y(q¹), z=z(q²).  If a coordinate transform
  ("stretched grid") is used, the scale factors are h₀=dx/dq⁰, h₁=dy/dq¹, and
  h₂=dz/dq².


  The library computes ∂f/∂xⱼ, where x=x₀, y=x₁, z=x₂.  Note that even when
  using transformed coordinates, for which ∂f/∂xⱼ = 1/hⱼ ∂f/∂qʲ (no sum over j),
  this library still computes and returns ∂f/∂xⱼ rather than ∂f/∂qʲ.

  Implementation details:
  The convention used is that faces to the *left* of a node are given the same
  index. See the diagram below, where `.` represents a node and `|` represents
  a face. For simplicity in this documentation, the diagram and descriptions
  use 1 dimension, but the functionality works on 3D fields.

  Index & coord loc.         i-1        i        i+1
                         |    .    |    .    |    .    |
  Index                 i-1        i        i+1
  Coord. loc.           i-3/2     i-1/2     i+1/2

  * For an array f evaluated on nodes: index i <==> coordinate location x_i
  * For an array f_face evaluated at faces: index i <==> coordinate location
    x_{i-1/2}.

  * Note, the value of f_face at the boundary (e.g., index 0 on a boundary
    replica) is not necessarily meaningful, because its location would be
    outside of the domain. This won't pollute computations, because field values
    at these points will be updated by the halo exchange / boundary conditions.

  Example: Given f on nodes, compute ∂f/∂x on faces.
    * Given f on nodes, the derivative at i-1/2 is stored at index i, and is
    given by df_dx_face[i] = (f[i] - f[i-1]) / dx. Note a backward sum is used.
    If a coordinate transform x(q) is involved, we divide by (h_face[i] * dq)
    rather than by dx.
    * Use method deriv_node_to_face()

  Example: Given f on faces, compute ∂f/∂x on nodes.
    * Given f on faces, the derivative at i is stored at index i, and is given
    by df_dx[i] = (f_face[i+1] - f_face[i]) / dx. Note a forward sum is used.
    If a coordinate transform x(q) is involved, we divide by (h[i] * dq) rather
    than by dx.
    * Use method deriv_face_to_node()

  Example: Given f on nodes, compute ∂f/∂x on nodes.
    * Given f on nodes, use a centered stencil to compute ∂f/∂x on nodes.  Then
    df_dx[i] = (f[i+1] - f[i-1]) / (2 * dx). If a coordinate transform x(q) is
    involved, we divide by (h[i] * dq) rather than by dx.
    * Use method deriv_centered()
  """

  def __init__(
      self,
      grid_spacings: tuple[float, float, float],
      use_stretched_grid: tuple[bool, bool, bool],
  ):
    """Instantiate a derivatives library."""
    self._grid_spacings = grid_spacings
    self._use_stretched_grid = use_stretched_grid

  def deriv_node_to_face(
      self, f: Array, dim: Literal[0, 1, 2], sg_map: StatesMap
  ) -> Array:
    """Compute the derivative of f in the given dimension, evaluated at faces."""
    df_dim_face = kernel_ops.backward_difference(f, dim)
    if self._use_stretched_grid[dim]:
      hf_key = stretched_grid_util.hf_key(dim)
      hf = sg_map[hf_key]
      df_dxdim_face = df_dim_face / (hf * self._grid_spacings[dim])
    else:
      df_dxdim_face = df_dim_face / self._grid_spacings[dim]
    return df_dxdim_face

  def deriv_face_to_node(
      self, f_face: Array, dim: Literal[0, 1, 2], sg_map: StatesMap
  ) -> Array:
    """Compute the derivative of f in the given dimension, evaluated at nodes."""
    df_dim = kernel_ops.forward_difference(f_face, dim)
    if self._use_stretched_grid[dim]:
      hc_key = stretched_grid_util.hc_key(dim)
      hc = sg_map[hc_key]
      df_dx_dim = df_dim / (hc * self._grid_spacings[dim])
    else:
      df_dx_dim = df_dim / self._grid_spacings[dim]
    return df_dx_dim

  def deriv_centered(
      self, f: Array, dim: Literal[0, 1, 2], sg_map: StatesMap
  ) -> Array:
    """Compute the centered derivative of f in the given dimension."""
    df_dim = kernel_ops.centered_difference(f, dim)
    if self._use_stretched_grid[dim]:
      hc_key = stretched_grid_util.hc_key(dim)
      hc = sg_map[hc_key]
      df_dx_dim = df_dim / (2 * hc * self._grid_spacings[dim])
    else:
      df_dx_dim = df_dim / (2 * self._grid_spacings[dim])
    return df_dx_dim

  # Define aliases for derivatives in each dimension, for clearer naming.
  def dx_f_to_c(self, f: Array, sg_map: StatesMap) -> Array:
    return self.deriv_face_to_node(f, 0, sg_map)

  def dy_f_to_c(self, f: Array, sg_map: StatesMap) -> Array:
    return self.deriv_face_to_node(f, 1, sg_map)

  def dz_f_to_c(self, f: Array, sg_map: StatesMap) -> Array:
    return self.deriv_face_to_node(f, 2, sg_map)

  def dx_c_to_f(self, f: Array, sg_map: StatesMap) -> Array:
    return self.deriv_node_to_face(f, 0, sg_map)

  def dy_c_to_f(self, f: Array, sg_map: StatesMap) -> Array:
    return self.deriv_node_to_face(f, 1, sg_map)

  def dz_c_to_f(self, f: Array, sg_map: StatesMap) -> Array:
    return self.deriv_node_to_face(f, 2, sg_map)

  def divergence_ccc(
      self, vx_fcc: Array, vy_cfc: Array, vz_ccf: Array, sg_map: StatesMap
  ) -> Array:
    """Compute ∇·V on (ccc), given vector V."

    The components of V are assumed to be vx (fcc), vy (cfc), and vz (ccf).
    """
    dvx_dx_ccc = self.dx_f_to_c(vx_fcc, sg_map)
    dvy_dy_ccc = self.dy_f_to_c(vy_cfc, sg_map)
    dvz_dz_ccc = self.dz_f_to_c(vz_ccf, sg_map)
    return dvx_dx_ccc + dvy_dy_ccc + dvz_dz_ccc
