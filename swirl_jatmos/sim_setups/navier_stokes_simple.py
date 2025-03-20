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

"""Step of the Navier Stokes Equations.

Simplified version of the incompressible Navier-Stokes step that assumes uniform
density (does not use the anelastic terms), and there is no thermodynamics,
microphysics, or radiative transfer.
"""

from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import convection
from swirl_jatmos import derivatives
from swirl_jatmos import diffusion
from swirl_jatmos.boundary_conditions import boundary_conditions
from swirl_jatmos.linalg import fast_diagonalization_solver_impl
from swirl_jatmos.linalg import jacobi_solver_impl
from swirl_jatmos.utils import utils

Array: TypeAlias = jax.Array

# RK3 coefficients
a = [8 / 15, 5 / 12, 3 / 4]
b = [0, -17 / 60, -5 / 12]
c = [8 / 15, 2 / 15, 1 / 3]


def null_halo_update_fn(x: Array) -> Array:
  return x


def update_halo_fn_for_pressure_poisson_eqn(
    dp_ccc: Array, halo_width: int = 1
) -> Array:
  """Update halos of `dp` for the pressure Poisson equation.

  Assume periodicity in x,y, and Neumann BCs in z.  That is, ∂p/∂z = 0 is
  applied on the z boundaries, which means the values of p in the halo nodes are
  set equal to the values in the first interior node.
  """
  # Update BCs in the z dimension, for Neumann BCs.
  face0_idx = halo_width - 1  # 0
  face1_idx = -halo_width  # -1

  dp_ccc = dp_ccc.at[:, :, face0_idx].set(dp_ccc[:, :, face0_idx + 1])
  dp_ccc = dp_ccc.at[:, :, face1_idx].set(dp_ccc[:, :, face1_idx - 1])

  # dp_ccc = dp_ccc.at[:, :, 0].set(dp_ccc[:, :, 1])
  # dp_ccc = dp_ccc.at[:, :, -1].set(dp_ccc[:, :, -2])
  return dp_ccc


def u_v_w_rhs_explicit_fn(
    deriv_lib: derivatives.Derivatives,
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    viscosity: float,
    states: dict[str, Array],
):
  """Compute the explicit RHS for u and v."""
  # ∂(u)/∂t = -∇·(V*u) + ∇·(nu ∇u) - ∇p
  # ∂(v)/∂t = -∇·(V*v) + ∇·(nu ∇v) - ∇p
  # ∂(w)/∂t = -∇·(V*w) + ∇·(nu ∇w) - ∇p
  rho = jnp.ones((1, 1, 1), dtype=u_fcc.dtype)
  nu_ccc = viscosity * jnp.ones((1, 1, 1), dtype=u_fcc.dtype)
  hw = 0
  interp_method = 'quick'
  sg_map = {}

  # Create some dummy data required in the following computation.  The z data is
  # irrelevant.
  z_c = np.zeros((1, 1, 1))
  z_f = np.zeros((1, 1, 1))
  z_bcs = boundary_conditions.ZBoundaryConditions()
  strain_rate_tensor = utils.compute_strain_rate_tensor(
      u_fcc, v_cfc, w_ccf, deriv_lib, sg_map, z_c, z_f, z_bcs
  )

  # Compute F_u = -∇·(V*u) + ∇·(nu ∇u)
  # Get convective fluxes.
  rhou_conv_flux_x_ccc, rhou_conv_flux_y_ffc, rhou_conv_flux_z_fcf = (
      convection.convective_flux_u(
          rho, rho, u_fcc, v_cfc, w_ccf, hw, interp_method
      )
  )

  rhov_conv_flux_x_ffc, rhov_conv_flux_y_ccc, rhov_conv_flux_z_cff = (
      convection.convective_flux_v(
          rho, rho, u_fcc, v_cfc, w_ccf, hw, interp_method
      )
  )

  rhow_conv_flux_x_fcf, rhow_conv_flux_y_cff, rhow_conv_flux_z_ccc = (
      convection.convective_flux_w(
          rho, rho, u_fcc, v_cfc, w_ccf, hw, interp_method
      )
  )
  # Compute the divergences of the fluxes, for the right-hand sides.
  tau00_ccc, tau01_ffc, tau02_fcf, tau11_ccc, tau12_cff, tau22_ccc = (
      diffusion.diffusive_fluxes_uvw(
          rho, rho, nu_ccc, strain_rate_tensor, u_fcc, v_cfc, z_bcs
      )
  )

  # Rename from tauij to diffusive fluxes for clarity.
  rhou_diff_flux_x_ccc = tau00_ccc  # The x component of the flux of x-momentum.
  rhou_diff_flux_y_ffc = tau01_ffc  # The y component of the flux of x-momentum.
  rhou_diff_flux_z_fcf = tau02_fcf  # And so on.

  rhov_diff_flux_y_ccc = tau11_ccc
  rhov_diff_flux_z_cff = tau12_cff
  rhow_diff_flux_z_ccc = tau22_ccc

  # Components determined through symmetry, since tau_ij = tau_ji, and we need
  # these components at the same locations.
  rhov_diff_flux_x_ffc = tau01_ffc  # tau10 = tau01
  rhow_diff_flux_x_fcf = tau02_fcf  # tau20 = tau02
  rhow_diff_flux_y_cff = tau12_cff  # tau21 = tau12

  # Combine convective and diffusive fluxes.
  rhou_flux_x_ccc = rhou_conv_flux_x_ccc + rhou_diff_flux_x_ccc
  rhou_flux_y_ffc = rhou_conv_flux_y_ffc + rhou_diff_flux_y_ffc
  rhou_flux_z_fcf = rhou_conv_flux_z_fcf + rhou_diff_flux_z_fcf

  rhov_flux_x_ffc = rhov_conv_flux_x_ffc + rhov_diff_flux_x_ffc
  rhov_flux_y_ccc = rhov_conv_flux_y_ccc + rhov_diff_flux_y_ccc
  rhov_flux_z_cff = rhov_conv_flux_z_cff + rhov_diff_flux_z_cff

  rhow_flux_x_fcf = rhow_conv_flux_x_fcf + rhow_diff_flux_x_fcf
  rhow_flux_y_cff = rhow_conv_flux_y_cff + rhow_diff_flux_y_cff
  rhow_flux_z_ccc = rhow_conv_flux_z_ccc + rhow_diff_flux_z_ccc

  # Compute the divergences of the fluxes, for the right-hand sides.
  div_rhou_flux_fcc = (
      deriv_lib.dx_c_to_f(rhou_flux_x_ccc, sg_map)
      + deriv_lib.dy_f_to_c(rhou_flux_y_ffc, sg_map)
      + deriv_lib.dz_f_to_c(rhou_flux_z_fcf, sg_map)
  )

  div_rhov_flux_cfc = (
      deriv_lib.dx_f_to_c(rhov_flux_x_ffc, sg_map)
      + deriv_lib.dy_c_to_f(rhov_flux_y_ccc, sg_map)
      + deriv_lib.dz_f_to_c(rhov_flux_z_cff, sg_map)
  )

  div_rhow_flux_ccf = (
      deriv_lib.dx_f_to_c(rhow_flux_x_fcf, sg_map)
      + deriv_lib.dy_f_to_c(rhow_flux_y_cff, sg_map)
      + deriv_lib.dz_c_to_f(rhow_flux_z_ccc, sg_map)
  )

  f_u_fcc = -div_rhou_flux_fcc
  f_v_cfc = -div_rhov_flux_cfc
  f_w_ccf = -div_rhow_flux_ccf
  return f_u_fcc, f_v_cfc, f_w_ccf


def step(
    step_id: Array,
    dt: float,
    states: dict[str, Array],
    deriv_lib: derivatives.Derivatives,
    viscosity: float,
    poisson_solver_type: Literal['jacobi', 'fast_diag'],
) -> dict[str, Array]:
  """Step forward by dt."""
  del step_id

  if poisson_solver_type == 'jacobi':
    poisson_solver = jacobi_solver_impl.PlainPoisson(
        grid_spacings=deriv_lib._grid_spacings,  # pylint: disable=protected-access
        omega=2 / 3,
        num_iters=10,
        halo_width=0,
    )
  elif poisson_solver_type == 'fast_diag':
    periodic = fast_diagonalization_solver_impl.BCType.PERIODIC
    bc_types = (periodic, periodic, periodic)
    nx_ny_nz = states['u'].shape
    poisson_solver = fast_diagonalization_solver_impl.Solver(
        nx_ny_nz,
        deriv_lib._grid_spacings,  # pylint: disable=protected-access
        bc_types,
        0,
        states['u'].dtype,
        uniform_z_2d=True,
    )
  else:
    raise ValueError(f'Unknown poisson solver type: {poisson_solver_type}')

  # Treating density as uniform and constant equal to 1 here.

  # Take an explicit step on phi using the equation
  # ∂(u)/∂t = -∇·(V*u) + ∇·(nu ∇u) - ∇p  =  F_u - ∇_x p
  # ∂(v)/∂t = -∇·(V*v) + ∇·(nu ∇v) - ∇p  =  F_v - ∇_y p
  # ∂(w)/∂t = -∇·(V*w) + ∇·(nu ∇w) - ∇p  =  F_w - ∇_z p

  # start with Python for loop.  Change this to a jax.lax.fori_loop later.
  #  3 stages, k = 1,2,3.
  f_u_fcc_prev = jnp.zeros_like(states['u'])
  f_v_cfc_prev = jnp.zeros_like(states['v'])
  f_w_ccf_prev = jnp.zeros_like(states['w'])

  u_fcc, v_cfc, w_ccf = states['u'], states['v'], states['w']
  p_ccc = states['p']

  for k in range(3):
    # Compute terms on RHS treated explicitly (all except pressure gradient).
    f_u_fcc, f_v_cfc, f_w_ccf = u_v_w_rhs_explicit_fn(
        deriv_lib, u_fcc, v_cfc, w_ccf, viscosity, states
    )

    # predictor step
    dpdx_fcc = deriv_lib.deriv_node_to_face(p_ccc, 0, states)
    dpdy_cfc = deriv_lib.deriv_node_to_face(p_ccc, 1, states)
    dpdz_ccf = deriv_lib.deriv_node_to_face(p_ccc, 2, states)

    u_fcc_hat = u_fcc + dt * (
        a[k] * f_u_fcc + b[k] * f_u_fcc_prev - c[k] * dpdx_fcc
    )
    v_cfc_hat = v_cfc + dt * (
        a[k] * f_v_cfc + b[k] * f_v_cfc_prev - c[k] * dpdy_cfc
    )
    w_ccf_hat = w_ccf + dt * (
        a[k] * f_w_ccf + b[k] * f_w_ccf_prev - c[k] * dpdz_ccf
    )

    # Compute RHS for Poisson eqn.  First compute ∇·(uhat)
    term1_ccc = deriv_lib.deriv_face_to_node(u_fcc_hat, 0, states)
    term2_ccc = deriv_lib.deriv_face_to_node(v_cfc_hat, 1, states)
    term3_ccc = deriv_lib.deriv_face_to_node(w_ccf_hat, 2, states)
    div_u_hat_ccc = term1_ccc + term2_ccc + term3_ccc
    rhs_ccc = div_u_hat_ccc / (dt * c[k])

    # Solve Poisson equation for delta p_k
    halo_update_fn = update_halo_fn_for_pressure_poisson_eqn

    if poisson_solver_type == 'jacobi':
      dp_ccc = poisson_solver.solve(
          rhs_ccc, jnp.zeros_like(u_fcc), halo_update_fn
      )
    elif poisson_solver_type == 'fast_diag':
      dp_ccc = poisson_solver.solve(rhs_ccc)
    else:
      raise ValueError(f'Unknown poisson solver type: {poisson_solver_type}')

    # Correction step (projection to divergence-free velocity).
    ddpdx_fcc = deriv_lib.deriv_node_to_face(dp_ccc, 0, states)
    ddpdy_cfc = deriv_lib.deriv_node_to_face(dp_ccc, 1, states)
    ddpdz_ccf = deriv_lib.deriv_node_to_face(dp_ccc, 2, states)
    u_fcc = u_fcc_hat - dt * c[k] * ddpdx_fcc
    v_cfc = v_cfc_hat - dt * c[k] * ddpdy_cfc
    w_ccf = w_ccf_hat - dt * c[k] * ddpdz_ccf

    p_ccc = p_ccc + dp_ccc

    f_u_fcc_prev, f_v_cfc_prev, f_w_ccf_prev = f_u_fcc, f_v_cfc, f_w_ccf

  states_new = {'u': u_fcc, 'v': v_cfc, 'w': w_ccf, 'p': p_ccc}
  return states_new
