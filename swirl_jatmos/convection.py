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

"""Convection.

In this file, enforce wall boundary conditions when considering convection in
the z direction.  In practice, what this means is that to get the flux next to
the wall, we can't use QUICK because the values outside the domain are not
valid.  Instead, we will use centered interpolation to get the interpolated
value of the convected variable at its control-volume face.

For a variable defined on z nodes, we'll assume the flux at the z-boundary face
is 0, but we need to determine the flux at the 1st interior face.  Consider the
case where the wall is on the left:

  wall
    |  o  |  o  |  o
          ^ <----- Need the variable interpolated to this face.

The required variable on the face can be determined with QUICK if the velocity
is to the left, as then the 2 points to the right can be used.  But if the
velocity is to the right, then with QUICK 2 points to the left would be needed,
which we do not have.  Therefore, in the case where the velocity is to the
right, we use centered interpolation to get the value at the face.

For a variable defined on z faces (in particular, the w variable)

  wall
    |  o  |  o  |  o
       ^ <----- Need the variable interpolated to this face.

Similarly to the above, if the velocity is to the right, then we use centered
interpolation to get the value at the face.  For wall boundaries, w=0 on the
wall.

Everything is the same for the wall on the right.

                  wall
    o  |  o  |  o  |


Under the anelastic assumption, rho depends only on z, so rho_xxc is used
whenever the location is on a z-center, regardless of whether the location is on
a face or center in x or y.  Similarly, rho_xxf is used whenever the location is
on a z-face, regardless of whether the location is on a center or face in x or
y.

Let the full 3D fields have shape (nx, ny, nz).

rho_xxc and rho_xxf are assumed to typically be arrays of shape (1, 1, nz),
although anything that broadcasts to an (nx, ny, nz) array will be allowed.
"""

import functools
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from swirl_jatmos import interpolation
from swirl_jatmos.boundary_conditions import apply_bcs


Array: TypeAlias = jax.Array


def update_centered_interp4_node_to_face_for_z_bcs(
    phi_face: Array, phi_node_original: Array, halo_width: int
) -> Array:
  """Handle centered4 node-to-face interpolation near a z wall."""
  hw = halo_width
  # Drop to centered interpolation for the first interior face and last interior
  # face.
  phi_bottom_face_centered = 0.5 * (
      phi_node_original[:, :, hw + 1] + phi_node_original[:, :, hw]
  )
  phi_face = phi_face.at[:, :, hw + 1].set(phi_bottom_face_centered)

  phi_top_face_centered = 0.5 * (
      phi_node_original[:, :, -(hw + 1)] + phi_node_original[:, :, -(hw + 2)]
  )
  phi_face = phi_face.at[:, :, -hw - 1].set(phi_top_face_centered)
  return phi_face


def update_centered_interp4_face_to_node_for_z_bcs(
    phi_node: Array, phi_face_original: Array, halo_width: int
) -> Array:
  """Handle centered4 face-to-node interpolation near a z wall."""
  hw = halo_width
  # Drop to centered interpolation for the first interior node and last interior
  # node.
  phi_bottom_node_centered = 0.5 * (
      phi_face_original[:, :, hw + 1] + phi_face_original[:, :, hw]
  )
  phi_node = phi_node.at[:, :, hw].set(phi_bottom_node_centered)

  phi_top_node_centered = 0.5 * (
      phi_face_original[:, :, -hw] + phi_face_original[:, :, -(hw + 1)]
  )
  phi_node = phi_node.at[:, :, -(hw + 1)].set(phi_top_node_centered)
  return phi_node


def handle_wall_adjacent_interpolations_node_to_face(
    phi_face_if_positive_velocity: Array,
    phi_face_if_negative_velocity: Array,
    phi_node: Array,
    halo_width: int,
):
  """Handle interpolations for node to face adjacent to a wall.

  Cannot use methods that take multiple points for interpolation, eg QUICK or
  2nd-order upwind.  Instead, use centered interpolation (if this is unstable,
  can use 1st-order upwind instead).

  Args:
    phi_face_if_positive_velocity: The value on the face if the velocity is
      positive.
    phi_face_if_negative_velocity: The value on the face if the velocity is
      negative.
    phi_node: The value on nodes.
    halo_width: The halo width used for the z dimension.

  Returns:
    The updated values of the left-biased and right-biased interpolations on
    faces, with special handling near the z boundaries.
  """
  del halo_width
  halo_width = 1

  # Special handling for dim 2, in which we assume there is a wall.
  # Handle the first interior face by using centered interpolation.
  phi_bottom_face_centered = 0.5 * (
      phi_node[:, :, halo_width + 1] + phi_node[:, :, halo_width]
  )
  phi_face_if_positive_velocity = phi_face_if_positive_velocity.at[
      :, :, halo_width + 1
  ].set(phi_bottom_face_centered)

  # For the face on the bottom corresponding to the wall (z=0), the terminal
  # velocities may be nonzero (pointing down), so we use an upwind interpolation
  # to get the value on the face.
  phi_bottom_wall_upwind = phi_node[:, :, halo_width]
  phi_face_if_negative_velocity = phi_face_if_negative_velocity.at[
      :, :, halo_width
  ].set(phi_bottom_wall_upwind)
  # For the face on the bottom wall, the vertical velocity should never be
  # positive, but just in case set the scalar zero if there is a positive
  # velocity.
  phi_face_if_positive_velocity = phi_face_if_positive_velocity.at[
      :, :, halo_width
  ].set(0)

  # Handle the last interior face.
  phi_top_face_centered = 0.5 * (
      phi_node[:, :, -(halo_width + 1)] + phi_node[:, :, -(halo_width + 2)]
  )
  phi_face_if_negative_velocity = phi_face_if_negative_velocity.at[
      :, :, -halo_width - 1
  ].set(phi_top_face_centered)

  # For the face on the top corresponding to the wall, we assume the terminal
  # velocities are set to zero there, so it does not matter what the
  # interpolated scalar value is (that is, the halo value is not used.)
  return (phi_face_if_positive_velocity, phi_face_if_negative_velocity)


def handle_wall_adjacent_interpolations_face_to_node(
    phi_node_if_positive_velocity: Array,
    phi_node_if_negative_velocity: Array,
    phi_face: Array,
    halo_width: int,
) -> tuple[Array, Array]:
  """Handle interpolations for face to node adjacent to a wall."""
  hw = halo_width
  # Special handling for dim 2, in which we assume there is a wall.
  # Handle the first interior node by using centered interpolation.
  s_bottom_node_centered = 0.5 * (phi_face[:, :, hw + 1] + phi_face[:, :, hw])
  phi_node_if_positive_velocity = phi_node_if_positive_velocity.at[
      :, :, hw
  ].set(s_bottom_node_centered)

  # Handle the last interior node.
  s_top_node_centered = 0.5 * (phi_face[:, :, -hw] + phi_face[:, :, -(hw + 1)])
  phi_node_if_negative_velocity = phi_node_if_negative_velocity.at[
      :, :, -(hw + 1)
  ].set(s_top_node_centered)
  return phi_node_if_positive_velocity, phi_node_if_negative_velocity


def interp_node_to_face_upwind1(
    phi_node: Array, dim: Literal[0, 1, 2], halo_width: int
) -> tuple[Array, Array]:
  """Perform interpolation with the 1st order upwind scheme."""
  del halo_width
  phi_node_iminus1 = jnp.roll(phi_node, 1, axis=dim)
  state_face_if_positive_velocity = phi_node_iminus1
  state_face_if_negative_velocity = phi_node
  return (state_face_if_positive_velocity, state_face_if_negative_velocity)


def interp_face_to_node_upwind1(
    phi_face: Array, dim: Literal[0, 1, 2], halo_width: int
) -> tuple[Array, Array]:
  """Perform interpolation with the 1st order upwind scheme."""
  del halo_width
  phi_face_iplus1 = jnp.roll(phi_face, -1, axis=dim)
  state_node_if_positive_velocity = phi_face
  state_node_if_negative_velocity = phi_face_iplus1
  return (state_node_if_positive_velocity, state_node_if_negative_velocity)


def interp_node_to_face_quick(
    phi_node: Array, dim: Literal[0, 1, 2], halo_width: int
) -> tuple[Array, Array]:
  """Perform interpolation with the QUICK scheme."""
  phi_node_iminus1 = jnp.roll(phi_node, 1, axis=dim)
  phi_node_iminus2 = jnp.roll(phi_node, 2, axis=dim)
  phi_node_iplus1 = jnp.roll(phi_node, -1, axis=dim)
  phi_face_left_biased = (
      -0.125 * phi_node_iminus2 + 0.75 * phi_node_iminus1 + 0.375 * phi_node
  )
  phi_face_right_biased = (
      -0.125 * phi_node_iplus1 + 0.75 * phi_node + 0.375 * phi_node_iminus1
  )

  # Special handling for dim 2, in which we assume there is a wall.
  if dim == 2:
    phi_face_left_biased, phi_face_right_biased = (
        handle_wall_adjacent_interpolations_node_to_face(
            phi_face_left_biased, phi_face_right_biased, phi_node, halo_width
        )
    )

  return (phi_face_left_biased, phi_face_right_biased)


def interp_face_to_node_quick(
    phi_face: Array, dim: Literal[0, 1, 2], halo_width: int
) -> tuple[Array, Array]:
  """Perform interpolation with the QUICK scheme."""
  phi_face_iminus1 = jnp.roll(phi_face, 1, axis=dim)
  phi_face_iplus1 = jnp.roll(phi_face, -1, axis=dim)
  phi_face_iplus2 = jnp.roll(phi_face, -2, axis=dim)
  state_node_if_positive_velocity = (
      -0.125 * phi_face_iminus1 + 0.75 * phi_face + 0.375 * phi_face_iplus1
  )
  state_node_if_negative_velocity = (
      -0.125 * phi_face_iplus2 + 0.75 * phi_face_iplus1 + 0.375 * phi_face
  )

  # Special handling for dim 2, in which we assume there is a wall.
  if dim == 2:
    (state_node_if_positive_velocity, state_node_if_negative_velocity) = (
        handle_wall_adjacent_interpolations_face_to_node(
            state_node_if_positive_velocity,
            state_node_if_negative_velocity,
            phi_face,
            halo_width,
        )
    )

  return (state_node_if_positive_velocity, state_node_if_negative_velocity)


def van_leer_limiter(r: Array) -> Array:
  """Compute the van leer limiter function ψ(r)."""
  return jnp.where(r > 0, 2 * r / (1 + r), 0.0)


def interp_node_to_face_van_leer_upwind2(
    phi_node: Array, dim: Literal[0, 1, 2], halo_width: int
) -> tuple[Array, Array]:
  """Perform interpolation with the Van-Leer-limited 2nd-order upwind scheme."""
  phi_node_iminus1 = jnp.roll(phi_node, 1, axis=dim)
  phi_node_iminus2 = jnp.roll(phi_node, 2, axis=dim)
  phi_node_iplus1 = jnp.roll(phi_node, -1, axis=dim)

  diff1 = phi_node_iplus1 - phi_node
  diff2 = phi_node - phi_node_iminus1
  diff3 = phi_node_iminus1 - phi_node_iminus2

  # Compute r = (state_D - state_U) / (state_U - state_UU).
  r_pos = jnp.where(diff3 == 0, 0.0, diff2 / diff3)
  r_neg = jnp.where(diff1 == 0, 0.0, diff2 / diff1)

  psi_pos = van_leer_limiter(r_pos)
  psi_neg = van_leer_limiter(r_neg)

  # Compute the value on the face, given by for 2nd-order upwind:
  #   state_face = state_U + 0.5 * psi(r) * (state_U - state_UU)
  state_face_if_positive_velocity = phi_node_iminus1 + 0.5 * psi_pos * diff3
  state_face_if_negative_velocity = phi_node - 0.5 * psi_neg * diff1

  # Special handling for dim 2, in which we assume there is a wall.
  if dim == 2:
    (state_face_if_positive_velocity, state_face_if_negative_velocity) = (
        handle_wall_adjacent_interpolations_node_to_face(
            state_face_if_positive_velocity,
            state_face_if_negative_velocity,
            phi_node,
            halo_width,
        )
    )
  return (state_face_if_positive_velocity, state_face_if_negative_velocity)


def interp_face_to_node_van_leer_upwind2(
    phi_face: Array, dim: Literal[0, 1, 2]
) -> tuple[Array, Array]:
  del phi_face, dim
  raise NotImplementedError(
      'van leer upwind2 for face-to-node interpolation is not implemented.'
  )


def interp_node_to_face_weno3(
    phi_node: Array, dim: Literal[0, 1, 2], halo_width: int
) -> tuple[Array, Array]:
  """Perform interpolation with the WENO3 scheme."""
  phi_face_left_biased, phi_face_right_biased = (
      interpolation.weno3_node_to_face(phi_node, dim)
  )
  # Special handling for dim 2, in which we assume there is a wall.
  if dim == 2:
    phi_face_left_biased, phi_face_right_biased = (
        handle_wall_adjacent_interpolations_node_to_face(
            phi_face_left_biased, phi_face_right_biased, phi_node, halo_width
        )
    )
  return phi_face_left_biased, phi_face_right_biased


def interp_face_to_node_weno3(
    phi_face: Array, dim: Literal[0, 1, 2], halo_width: int
) -> tuple[Array, Array]:
  """Perform interpolation with the WENO3 scheme."""
  phi_node_left_biased, phi_node_right_biased = (
      interpolation.weno3_face_to_node(phi_face, dim)
  )
  # Special handling for dim 2, in which we assume there is a wall.
  if dim == 2:
    phi_node_left_biased, phi_node_right_biased = (
        handle_wall_adjacent_interpolations_face_to_node(
            phi_node_left_biased, phi_node_right_biased, phi_face, halo_width
        )
    )
  return phi_node_left_biased, phi_node_right_biased


def interp_node_to_face_weno5(
    phi_node: Array,
    dim: Literal[0, 1, 2],
    halo_width: int,
    method: Literal['JS', 'Z'],
) -> tuple[Array, Array]:
  """Perform interpolation with the WENO5 scheme."""
  # Special handling for dim 2.
  # Set the halo nodes to very large values so that WENO5 stencil adapts to
  # avoid using the halos near the boundary.
  if dim == 2:
    phi_node = apply_bcs.set_z_halo_nodes_to_large_value(phi_node, halo_width)

  phi_face_left_biased, phi_face_right_biased = (
      interpolation.weno5_node_to_face(phi_node, dim, method)
  )
  # Special handling for dim 2, in which we assume there is a wall.
  # This fixes up the face values at the wall and the first interior face; the
  # 2nd interior face is handled in the WENO5 stencil.
  if dim == 2:
    phi_face_left_biased, phi_face_right_biased = (
        handle_wall_adjacent_interpolations_node_to_face(
            phi_face_left_biased, phi_face_right_biased, phi_node, halo_width
        )
    )
  return phi_face_left_biased, phi_face_right_biased


def interp_face_to_node_weno5(
    phi_face: Array,
    dim: Literal[0, 1, 2],
    halo_width: int,
    method: Literal['JS', 'Z'],
) -> tuple[Array, Array]:
  """Perform interpolation with the WENO5 scheme."""
  if dim == 2:  # Special handling due to wall in dim 2.
    # Set the halo faces to very large values so that WENO5 stencil adapts to
    # avoid using the halos near the boundary.
    phi_face = apply_bcs.set_z_halo_faces_to_large_value(phi_face, halo_width)

  phi_node_left_biased, phi_node_right_biased = (
      interpolation.weno5_face_to_node(phi_face, dim, method)
  )
  # Special handling for dim 2, in which we assume there is a wall.
  if dim == 2:
    phi_node_left_biased, phi_node_right_biased = (
        handle_wall_adjacent_interpolations_face_to_node(
            phi_node_left_biased, phi_node_right_biased, phi_face, halo_width
        )
    )
    # Additional special handling for WENO5 -- the 2nd-last interior node at the
    # top isn't treated properly with WENO5, so we set the right-based version
    # to use the upwind2 scheme.
    hw = halo_width
    upwind2_val = (
        3 / 2 * phi_face[:, :, -(hw + 1)] - 1 / 2 * phi_face[:, :, -hw]
    )
    phi_node_right_biased = phi_node_right_biased.at[
        :, :, -(halo_width + 2)
    ].set(upwind2_val)
  return phi_node_left_biased, phi_node_right_biased


def compute_flux_face(
    rho_face: Array,
    u_face: Array,
    phi_node: Array,
    dim: Literal[0, 1, 2],
    halo_width: int,
    interp_method: Literal[
        'upwind1', 'quick', 'van_leer_upwind2', 'weno3', 'weno5_js', 'weno5_z'
    ],
) -> Array:
  """Compute the flux on faces in dim, assuming phi is given on nodes in dim.

  Args:
    rho_face: The density on the edges of the control volume for phi.
    u_face: The velocity in dim on the edges of the control volume for phi.
    phi_node: The convected variable, given on nodes.
    dim: The dimension in which the flux is being computed.
    halo_width: The halo width used for the z dimension.
    interp_method: The interpolation method to use.

  Returns:
    The flux on faces in dim.
  """
  if interp_method == 'upwind1':
    interp_node_to_face_fn = interp_node_to_face_upwind1
  elif interp_method == 'quick':
    interp_node_to_face_fn = interp_node_to_face_quick
  elif interp_method == 'van_leer_upwind2':
    interp_node_to_face_fn = interp_node_to_face_van_leer_upwind2
  elif interp_method == 'weno3':
    interp_node_to_face_fn = interp_node_to_face_weno3
  elif interp_method == 'weno5_js':
    interp_node_to_face_fn = functools.partial(
        interp_node_to_face_weno5, method='JS'
    )
  elif interp_method == 'weno5_z':
    interp_node_to_face_fn = functools.partial(
        interp_node_to_face_weno5, method='Z'
    )
  else:
    raise ValueError(f'Unsupported interp_method: {interp_method}')

  phi_face_if_positive_velocity, phi_face_if_negative_velocity = (
      interp_node_to_face_fn(phi_node, dim, halo_width)
  )
  phi_face = jnp.where(
      u_face > 0, phi_face_if_positive_velocity, phi_face_if_negative_velocity
  )
  return rho_face * u_face * phi_face


def compute_flux_node(
    rho_node: Array,
    u_node: Array,
    phi_face: Array,
    dim: Literal[0, 1, 2],
    halo_width: int,
    interp_method: Literal['upwind1', 'quick', 'weno3', 'weno5_js', 'weno5_z'],
) -> Array:
  """Compute the flux on nodes in dim, assuming phi is given on faces in dim.

  Args:
    rho_node: The density on nodes, i.e., on edges of the control volume for
      phi.
    u_node: The velocity in dim on on nodes, i.e., on edges of the control
      volume for phi.
    phi_face: The convected variable, given on faces.
    dim: The dimension in which the flux is being computed.
    halo_width: The halo width used for the z dimension.
    interp_method: The interpolation method to use.

  Returns:
    The flux on nodes in dim.
  """
  if interp_method == 'upwind1':
    interp_face_to_node_fn = interp_face_to_node_upwind1
  elif interp_method == 'quick':
    interp_face_to_node_fn = interp_face_to_node_quick
  elif interp_method == 'weno3':
    interp_face_to_node_fn = interp_face_to_node_weno3
  elif interp_method == 'weno5_js':
    interp_face_to_node_fn = functools.partial(
        interp_face_to_node_weno5, method='JS'
    )
  elif interp_method == 'weno5_z':
    interp_face_to_node_fn = functools.partial(
        interp_face_to_node_weno5, method='Z'
    )
  else:
    raise ValueError(f'Unsupported interp_method: {interp_method}')
  phi_node_if_positive_velocity, phi_face_if_negative_velocity = (
      interp_face_to_node_fn(phi_face, dim, halo_width)
  )
  phi_node = jnp.where(
      u_node > 0, phi_node_if_positive_velocity, phi_face_if_negative_velocity
  )
  return rho_node * u_node * phi_node


def convective_flux_scalar(
    rho_xxc: Array,
    rho_xxf: Array,
    phi_ccc: Array,
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    halo_width: int,
    interp_method: Literal[
        'upwind1', 'quick', 'van_leer_upwind2', 'weno3', 'weno5_js', 'weno5_z'
    ],
) -> tuple[Array, Array, Array]:
  """Compute the convective flux ρVphi.

  The x-component of the flux is on (fcc), the y-component on (cfc), and the
  z-component on (ccf).
  """
  hw = halo_width
  flux_x_fcc = compute_flux_face(rho_xxc, u_fcc, phi_ccc, 0, hw, interp_method)
  flux_y_cfc = compute_flux_face(rho_xxc, v_cfc, phi_ccc, 1, hw, interp_method)
  flux_z_ccf = compute_flux_face(rho_xxf, w_ccf, phi_ccc, 2, hw, interp_method)
  return flux_x_fcc, flux_y_cfc, flux_z_ccf


def convective_flux_u(
    rho_xxc: Array,
    rho_xxf: Array,
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    halo_width: int,
    interp_method: Literal['upwind1', 'quick', 'weno3', 'weno5_js', 'weno5_z'],
) -> tuple[Array, Array, Array]:
  """Compute the convective flux ρVu of x-momentum.

  The divergence of this flux ∇·(ρVu) is on (fcc).
  """
  if interp_method in ['upwind1', 'quick', 'weno3']:
    ubar_ccc = interpolation.centered_face_to_node(u_fcc, 0)
    vbar_ffc = interpolation.centered_node_to_face(v_cfc, 0)
    wbar_fcf = interpolation.centered_node_to_face(w_ccf, 0)
  elif interp_method in ['weno5_js', 'weno5_z']:
    # interp4_cell_avg or interp4???  Test on buoyant bubble / supercell.
    # interp4_fn = interpolation.interp4_cell_avg_face_to_node
    ubar_ccc = interpolation.interp4_cell_avg_face_to_node(u_fcc, 0)
    vbar_ffc = interpolation.interp4_cell_avg_node_to_face(v_cfc, 0)
    wbar_fcf = interpolation.interp4_cell_avg_node_to_face(w_ccf, 0)

    # ubar_ccc = interpolation.interp4_face_to_node(u_fcc, 0)
    # vbar_ffc = interpolation.interp4_node_to_face(v_cfc, 0)
    # wbar_fcf = interpolation.interp4_node_to_face(w_ccf, 0)

    # ubar_ccc = interpolation.centered_face_to_node(u_fcc, 0)
    # vbar_ffc = interpolation.centered_node_to_face(v_cfc, 0)
    # wbar_fcf = interpolation.centered_node_to_face(w_ccf, 0)
  else:
    raise ValueError(f'Unsupported interp_method: {interp_method}')
  rhou_flux_x_ccc = compute_flux_node(
      rho_xxc, ubar_ccc, u_fcc, 0, halo_width, interp_method
  )
  rhou_flux_y_ffc = compute_flux_face(
      rho_xxc, vbar_ffc, u_fcc, 1, halo_width, interp_method
  )
  rhou_flux_z_fcf = compute_flux_face(
      rho_xxf, wbar_fcf, u_fcc, 2, halo_width, interp_method
  )
  return rhou_flux_x_ccc, rhou_flux_y_ffc, rhou_flux_z_fcf


def convective_flux_v(
    rho_xxc: Array,
    rho_xxf: Array,
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    halo_width: int,
    interp_method: Literal['upwind1', 'quick', 'weno3', 'weno5_js', 'weno5_z'],
) -> tuple[Array, Array, Array]:
  """Compute the convective flux ρVv of y-momentum."""
  if interp_method in ['upwind1', 'quick', 'weno3']:
    ubar_ffc = interpolation.centered_node_to_face(u_fcc, 1)
    vbar_ccc = interpolation.centered_face_to_node(v_cfc, 1)
    wbar_cff = interpolation.centered_node_to_face(w_ccf, 1)
  elif interp_method in ['weno5_js', 'weno5_z']:
    # interp4_cell_avg or interp4?
    # interp4_fn = interpolation.interp4_cell_avg_face_to_node
    ubar_ffc = interpolation.interp4_cell_avg_node_to_face(u_fcc, 1)
    vbar_ccc = interpolation.interp4_cell_avg_face_to_node(v_cfc, 1)
    wbar_cff = interpolation.interp4_cell_avg_node_to_face(w_ccf, 1)

    # ubar_ffc = interpolation.interp4_node_to_face(u_fcc, 1)
    # vbar_ccc = interpolation.interp4_face_to_node(v_cfc, 1)
    # wbar_cff = interpolation.interp4_node_to_face(w_ccf, 1)

    # ubar_ffc = interpolation.centered_node_to_face(u_fcc, 1)
    # vbar_ccc = interpolation.centered_face_to_node(v_cfc, 1)
    # wbar_cff = interpolation.centered_node_to_face(w_ccf, 1)
  else:
    raise ValueError(f'Unsupported interp_method: {interp_method}')
  rhov_flux_x_ffc = compute_flux_face(
      rho_xxf, ubar_ffc, v_cfc, 0, halo_width, interp_method
  )
  rhov_flux_y_ccc = compute_flux_node(
      rho_xxc, vbar_ccc, v_cfc, 1, halo_width, interp_method
  )
  rhov_flux_z_cff = compute_flux_face(
      rho_xxf, wbar_cff, v_cfc, 2, halo_width, interp_method
  )
  return rhov_flux_x_ffc, rhov_flux_y_ccc, rhov_flux_z_cff


def convective_flux_w(
    rho_xxc: Array,
    rho_xxf: Array,
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    halo_width: int,
    interp_method: Literal['upwind1', 'quick', 'weno3', 'weno5_js', 'weno5_z'],
) -> tuple[Array, Array, Array]:
  """Compute the convective flux ρVw of z-momentum."""
  if interp_method in ['upwind1', 'quick', 'weno3']:
    ubar_fcf = interpolation.centered_node_to_face(u_fcc, 2)
    vbar_cff = interpolation.centered_node_to_face(v_cfc, 2)
    wbar_ccc = interpolation.centered_face_to_node(w_ccf, 2)
  elif interp_method in ['weno5_js', 'weno5_z']:
    # interp4_cell_avg or interp4?
    # interp4_fn = interpolation.interp4_cell_avg_face_to_node

    ubar_fcf = interpolation.interp4_cell_avg_node_to_face(u_fcc, 2)
    vbar_cff = interpolation.interp4_cell_avg_node_to_face(v_cfc, 2)
    wbar_ccc = interpolation.interp4_cell_avg_face_to_node(w_ccf, 2)
    # Fix up the nodes near the boundary, as necessary.
    ubar_fcf = update_centered_interp4_node_to_face_for_z_bcs(
        ubar_fcf, u_fcc, halo_width
    )
    vbar_cff = update_centered_interp4_node_to_face_for_z_bcs(
        vbar_cff, v_cfc, halo_width
    )
    wbar_ccc = update_centered_interp4_face_to_node_for_z_bcs(
        wbar_ccc, w_ccf, halo_width
    )

    # ubar_fcf = interpolation.centered_node_to_face(u_fcc, 2)
    # vbar_cff = interpolation.centered_node_to_face(v_cfc, 2)
    # wbar_ccc = interpolation.centered_face_to_node(w_ccf, 2)
  else:
    raise ValueError(f'Unsupported interp_method: {interp_method}')
  rhow_flux_x_fcf = compute_flux_face(
      rho_xxf, ubar_fcf, w_ccf, 0, halo_width, interp_method
  )
  rhow_flux_y_cff = compute_flux_face(
      rho_xxf, vbar_cff, w_ccf, 1, halo_width, interp_method
  )
  rhow_flux_z_ccc = compute_flux_node(
      rho_xxc, wbar_ccc, w_ccf, 2, halo_width, interp_method
  )
  return rhow_flux_x_fcf, rhow_flux_y_cff, rhow_flux_z_ccc
