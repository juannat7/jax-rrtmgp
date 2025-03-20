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

"""Module for velocity equations.

This is identical to the other velocity module, except that the buoyancy term
includes the corrections accounting for precipitation.
"""

from typing import TypeAlias

import jax
from swirl_jatmos import convection
from swirl_jatmos import convection_config
from swirl_jatmos import derivatives
from swirl_jatmos import diffusion
from swirl_jatmos import interpolation
from swirl_jatmos.boundary_conditions import boundary_conditions
from swirl_jatmos.thermodynamics import water
from swirl_jatmos.utils import utils

Array: TypeAlias = jax.Array
ThermoFields: TypeAlias = water.ThermoFields

G = 9.81  # The gravitational acceleration constant, in units of m/s^2.


def u_v_w_rhs_explicit_fn(
    rho_xxc: Array,
    rho_xxf: Array,
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    q_r_ccc: Array,
    q_s_ccc: Array,
    viscosity_ccc: Array,
    rho_thermal_ccc: Array,
    strain_rate_tensor: utils.StrainRateTensor,
    deriv_lib: derivatives.Derivatives,
    sg_map: dict[str, Array],
    convection_cfg: convection_config.ConvectionConfig,
    include_buoyancy: bool,
    halo_width: int,
    z_bcs: boundary_conditions.ZBoundaryConditions,
) -> tuple[Array, Array, Array]:
  """Compute the explicit RHS for u, v, w (excluding pressure gradient).

  ∂(u)/∂t = F_x - (1 / ρ_ref) dp_dx
  ∂(v)/∂t = F_y - (1 / ρ_ref) dp_dy
  ∂(w)/∂t = F_z - (1 / ρ_ref) dp_dz

  The terms treated explicitly are:
    F_x = -(1 / ρ_ref) * [∇·(ρ_ref V*u) + ∇·(t_x)]
    F_x = -(1 / ρ_ref) * [∇·(ρ_ref V*v) + ∇·(t_y)]
    F_x = -(1 / ρ_ref) * [∇·(ρ_ref V*w) + ∇·(t_z)] + b

  where V = (u,v,w) is the velocity vector, and t_x, t_y, t_z are the diffusive
  fluxes of the x-momentum, y-momentum, and z-momentum, respectively:
    t_x = (tau_00, tau_01, tau_02)
    t_y = (tau_10, tau_11, tau_12)
    t_z = (tau_20, tau_21, tau_22)

  and b is the buoyancy acceleration, given by
    b = -g * (ρ_thermal - ρ_ref) / ρ_thermal
  """
  interp_method = convection_cfg.momentum_scheme

  # Get convective fluxes.
  rhou_conv_flux_x_ccc, rhou_conv_flux_y_ffc, rhou_conv_flux_z_fcf = (
      convection.convective_flux_u(
          rho_xxc, rho_xxf, u_fcc, v_cfc, w_ccf, halo_width, interp_method
      )
  )

  rhov_conv_flux_x_ffc, rhov_conv_flux_y_ccc, rhov_conv_flux_z_cff = (
      convection.convective_flux_v(
          rho_xxc, rho_xxf, u_fcc, v_cfc, w_ccf, halo_width, interp_method
      )
  )

  rhow_conv_flux_x_fcf, rhow_conv_flux_y_cff, rhow_conv_flux_z_ccc = (
      convection.convective_flux_w(
          rho_xxc, rho_xxf, u_fcc, v_cfc, w_ccf, halo_width, interp_method
      )
  )

  # Get the diffusive momentum fluxes.
  tau00_ccc, tau01_ffc, tau02_fcf, tau11_ccc, tau12_cff, tau22_ccc = (
      diffusion.diffusive_fluxes_uvw(
          rho_xxc,
          rho_xxf,
          viscosity_ccc,
          strain_rate_tensor,
          u_fcc,
          v_cfc,
          z_bcs,
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

  # Compute the buoyancy acceleration on (ccf)
  if include_buoyancy:
    rho_thermal_ccf = interpolation.z_c_to_f(rho_thermal_ccc)
    q_r_ccf = interpolation.z_c_to_f(q_r_ccc)
    q_s_ccf = interpolation.z_c_to_f(q_s_ccc)
    buoyancy_ccf = utils.buoyancy(rho_thermal_ccf, rho_xxf, q_r_ccf, q_s_ccf)
  else:
    buoyancy_ccf = 0

  # Add any additional terms as needed (e.g., buoyancy in the z direction).
  f_u_fcc = -div_rhou_flux_fcc / rho_xxc
  f_v_cfc = -div_rhov_flux_cfc / rho_xxc
  f_w_ccf = -div_rhow_flux_ccf / rho_xxf + buoyancy_ccf

  return f_u_fcc, f_v_cfc, f_w_ccf
