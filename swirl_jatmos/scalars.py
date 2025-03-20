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

"""Module to compute the terms in the scalar equations for the cloud sim.

Scalars:
* theta_li - the energy-like prognostic variable [K]
* q_t - total specific humidity [kg/kg]
* q_r - specific humidity of rain precipitation [kg/kg]
* q_s - specific humidity of snow precipitation [kg/kg]
"""

import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp
from swirl_jatmos import constants
from swirl_jatmos import convection
from swirl_jatmos import convection_config
from swirl_jatmos import derivatives
from swirl_jatmos import diffusion
from swirl_jatmos import interpolation
from swirl_jatmos.boundary_conditions import boundary_conditions
from swirl_jatmos.microphysics import microphysics_config
from swirl_jatmos.microphysics import microphysics_coupling
from swirl_jatmos.microphysics import microphysics_one_moment
from swirl_jatmos.rrtmgp.config import radiative_transfer
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array
ThermoFields: TypeAlias = water.ThermoFields


def compute_terminal_velocities(
    q_t: Array,
    q_r: Array,
    q_s: Array,
    thermo_fields: ThermoFields,
    include_qt_sedimentation: bool,
    microphysics_cfg: microphysics_config.MicrophysicsConfig,
) -> tuple[Array, Array, Array, Array]:
  """Compute the terminal / sedimentation velocities w_r, w_s, w_liq, w_ice."""
  rho_thermal = thermo_fields.rho_thermal

  w_r = microphysics_one_moment.terminal_velocity(
      'q_r', q_r, rho_thermal, microphysics_cfg
  )
  w_s = microphysics_one_moment.terminal_velocity(
      'q_s', q_s, rho_thermal, microphysics_cfg
  )

  if include_qt_sedimentation:
    q_liq = jnp.clip(thermo_fields.q_liq, 0.0, None)
    q_ice = jnp.clip(thermo_fields.q_ice, 0.0, None)
    w_liq = microphysics_one_moment.terminal_velocity(
        'q_liq', q_liq, rho_thermal, microphysics_cfg
    )
    w_ice = microphysics_one_moment.terminal_velocity(
        'q_ice', q_ice, rho_thermal, microphysics_cfg
    )
  else:
    w_liq = jnp.zeros_like(q_t)
    w_ice = jnp.zeros_like(q_t)

  # All of the above velocities are at (ccc), because they are computed from
  # scalars, and all scalars are at (ccc).
  return w_r, w_s, w_liq, w_ice


def get_terminal_velocities_on_faces(
    q_t: Array,
    w_r_ccc: Array,
    w_s_ccc: Array,
    w_liq_ccc: Array,
    w_ice_ccc: Array,
    thermo_fields: ThermoFields,
    halo_width: int,
) -> tuple[Array, Array, Array]:
  """Compute the terminal / sedimentation velocities w_r, w_s, w_liq, w_ice."""
  q_liq, q_ice = thermo_fields.q_liq, thermo_fields.q_ice

  sedimentation_flux = q_liq * w_liq_ccc + q_ice * w_ice_ccc
  # Effective velocity of w_t is the flux divided by the scalar value.
  w_t_ccc = jnp.where(q_t > 0, sedimentation_flux / q_t, 0)

  # All of the above velocities are at (ccc), because they are computed from
  # scalars, and all scalars are at (ccc).  We must interpolate the velocities
  # to (ccf) and then enforce boundary conditions for use in convection.

  # Alternate approach --- don't set w to 0 for terminal velocities.
  # This is to be consistent with Swirl's approach.  However, for the velocities
  # on the bottom face, just use the velocity at ccc.
  def interp_terminal_velocities(w_ccc: Array) -> Array:
    w_ccf = interpolation.z_c_to_f(w_ccc)
    # Lower boundary: w|wall = w_ccc
    w_ccf = w_ccf.at[:, :, halo_width].set(w_ccc[:, :, halo_width])
    # Upper boundary: w|wall = 0
    w_ccf = w_ccf.at[:, :, -halo_width].set(0)
    return w_ccf

  w_r_ccf = interp_terminal_velocities(w_r_ccc)
  w_s_ccf = interp_terminal_velocities(w_s_ccc)
  w_t_ccf = interp_terminal_velocities(w_t_ccc)
  return w_r_ccf, w_s_ccf, w_t_ccf


def compute_sedimentation_fluxes(
    q_r_ccc: Array,
    q_s_ccc: Array,
    w_r_ccc: Array,
    w_s_ccc: Array,
    w_liq_ccc: Array,
    w_ice_ccc: Array,
    thermo_fields: ThermoFields,
    include_qt_sedimentation: bool,
    halo_width: int,
):
  """Compute the condensate flux due to sedimentation with 1st-order upwind."""
  hw = halo_width
  # 1st-order upwinded flux, at z faces.
  qr_sedimentation_flux_ccf = -(q_r_ccc * w_r_ccc)
  qs_sedimentation_flux_ccf = -(q_s_ccc * w_s_ccc)
  # Upper boundary: flux at wall = 0
  qr_sedimentation_flux_ccf = qr_sedimentation_flux_ccf.at[:, :, -hw].set(0)
  qs_sedimentation_flux_ccf = qs_sedimentation_flux_ccf.at[:, :, -hw].set(0)

  if include_qt_sedimentation:
    q_liq_ccc, q_ice_ccc = thermo_fields.q_liq, thermo_fields.q_ice
    qt_sedimentation_flux_ccf = -(q_liq_ccc * w_liq_ccc + q_ice_ccc * w_ice_ccc)
    # Upper boundary: flux|wall = 0
    qt_sedimentation_flux_ccf = qt_sedimentation_flux_ccf.at[:, :, -hw].set(0)
  else:
    qt_sedimentation_flux_ccf = jnp.zeros_like(w_liq_ccc)
  return (
      qr_sedimentation_flux_ccf,
      qs_sedimentation_flux_ccf,
      qt_sedimentation_flux_ccf,
  )


def compute_total_flux_divergences(
    rho_xxc: Array,
    rho_xxf: Array,
    theta_li_ccc: Array,
    q_t_ccc: Array,
    q_r_ccc: Array,
    q_s_ccc: Array,
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    diffusivity_ccc: Array,
    thermo_fields: ThermoFields,
    deriv_lib: derivatives.Derivatives,
    sg_map: dict[str, Array],
    convection_cfg: convection_config.ConvectionConfig,
    microphysics_cfg: microphysics_config.MicrophysicsConfig,
    include_qt_sedimentation: bool,
    halo_width: int,
    z_bcs: boundary_conditions.ZBoundaryConditions,
    sfc_temperature: Array | None = None,
    sfc_q_t: Array | None = None,
) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
  """Compute the total flux divergence for each scalar.

  Let G = combined convective and diffusive flux.  Compute div G for each
  scalar theta_li, q_t, q_r, q_s.
  """
  aux_output = {}

  # Get the terminal velocities for various water types.
  w_r_ccc, w_s_ccc, w_liq_ccc, w_ice_ccc = compute_terminal_velocities(
      q_t_ccc,
      q_r_ccc,
      q_s_ccc,
      thermo_fields,
      include_qt_sedimentation,
      microphysics_cfg,
  )

  w_r_ccf, w_s_ccf, w_t_ccf = get_terminal_velocities_on_faces(
      q_t_ccc,
      w_r_ccc,
      w_s_ccc,
      w_liq_ccc,
      w_ice_ccc,
      thermo_fields,
      halo_width,
  )

  if microphysics_cfg.sedimentation_method in [1, 2]:
    # Don't use w_t_ccf for sedimentation flux.  Instead compute a 1st-order
    # upwinded condensate flux directly.
    (
        qr_sedimentation_flux_ccf,
        qs_sedimentation_flux_ccf,
        qt_sedimentation_flux_ccf,
    ) = compute_sedimentation_fluxes(
        q_r_ccc,
        q_s_ccc,
        w_r_ccc,
        w_s_ccc,
        w_liq_ccc,
        w_ice_ccc,
        thermo_fields,
        include_qt_sedimentation,
        halo_width,
    )
    if microphysics_cfg.sedimentation_method == 1:
      # Use upwind1 for all particle terminal velocity fluxes.
      w_r_ccf = 0.0
      w_s_ccf = 0.0
      w_t_ccf = 0.0
    elif microphysics_cfg.sedimentation_method == 2:
      # Use upwind1 only for qt terminal velocity flux.
      w_t_ccf = 0.0
      qr_sedimentation_flux_ccf = 0.0
      qs_sedimentation_flux_ccf = 0.0
    else:
      raise ValueError(
          'sedimentation_method must be 0, 1, or 2.  Got'
          f' {microphysics_cfg.sedimentation_method}.'
      )

  else:  # sedimentation_method == 0.
    qr_sedimentation_flux_ccf = 0.0
    qs_sedimentation_flux_ccf = 0.0
    qt_sedimentation_flux_ccf = 0.0

  # Compute the convective fluxes for each scalar.

  # Create a partial function where the only arguments remaining are the scalar
  # itself, the vertical velocity, and the interpolation method. Do this because
  # the horizontal velocities are the same for all scalars, but different
  # scalars may use different vertical velocities due to the sedimentation
  # velocities.
  conv_flux_scalar = functools.partial(
      convection.convective_flux_scalar,
      rho_xxc=rho_xxc,
      rho_xxf=rho_xxf,
      u_fcc=u_fcc,
      v_cfc=v_cfc,
      halo_width=halo_width,
  )

  th_conv_flux_x_fcc, th_conv_flux_y_cfc, th_conv_flux_z_ccf = conv_flux_scalar(
      phi_ccc=theta_li_ccc,
      w_ccf=w_ccf,
      interp_method=convection_cfg.theta_li_scheme,
  )
  # For convection of q_t, q_r, q_s: Modify the vertical velocity from the bulk
  # air velocity to account for the sedimentation velocities (if using such a
  # scheme)
  qt_conv_flux_x_fcc, qt_conv_flux_y_cfc, qt_conv_flux_z_ccf = conv_flux_scalar(
      phi_ccc=q_t_ccc,
      w_ccf=w_ccf - w_t_ccf,
      interp_method=convection_cfg.q_t_scheme,
  )
  qr_conv_flux_x_fcc, qr_conv_flux_y_cfc, qr_conv_flux_z_ccf = conv_flux_scalar(
      phi_ccc=q_r_ccc,
      w_ccf=w_ccf - w_r_ccf,
      interp_method=convection_cfg.q_r_scheme,
  )
  qs_conv_flux_x_fcc, qs_conv_flux_y_cfc, qs_conv_flux_z_ccf = conv_flux_scalar(
      phi_ccc=q_s_ccc,
      w_ccf=w_ccf - w_s_ccf,
      interp_method=convection_cfg.q_s_scheme,
  )

  # Compute the diffusive fluxes for each scalar.

  # Create a partial function where the only arguments remaining are the scalar
  # itself and the boundary condition argument, assuming the diffusivity is the
  # same for all scalars.
  diff_flux_scalar = functools.partial(
      diffusion.diffusive_flux_scalar,
      rho_xxc=rho_xxc,
      rho_xxf=rho_xxf,
      diffusivity_ccc=diffusivity_ccc,
      deriv_lib=deriv_lib,
      sg_map=sg_map,
      u_fcc=u_fcc,
      v_cfc=v_cfc,
      z_bcs=z_bcs,
  )

  # Note: at the surface, theta_li = temperature assuming no condensation and
  # using Exner reference pressure as the surface pressure.
  th_diff_flux_x_fcc, th_diff_flux_y_cfc, th_diff_flux_z_ccf = diff_flux_scalar(
      phi_ccc=theta_li_ccc,
      scalar_name='theta_li',
      phi_surface_value=sfc_temperature,
  )
  qt_diff_flux_x_fcc, qt_diff_flux_y_cfc, qt_diff_flux_z_ccf = diff_flux_scalar(
      phi_ccc=q_t_ccc, scalar_name='q_t', phi_surface_value=sfc_q_t
  )
  qr_diff_flux_x_fcc, qr_diff_flux_y_cfc, qr_diff_flux_z_ccf = diff_flux_scalar(
      phi_ccc=q_r_ccc, scalar_name='q_r'
  )
  qs_diff_flux_x_fcc, qs_diff_flux_y_cfc, qs_diff_flux_z_ccf = diff_flux_scalar(
      phi_ccc=q_s_ccc, scalar_name='q_s'
  )

  # Combine convective and diffusive fluxes.
  th_flux_x_fcc = th_conv_flux_x_fcc + th_diff_flux_x_fcc
  th_flux_y_cfc = th_conv_flux_y_cfc + th_diff_flux_y_cfc
  th_flux_z_ccf = th_conv_flux_z_ccf + th_diff_flux_z_ccf

  qt_flux_x_fcc = qt_conv_flux_x_fcc + qt_diff_flux_x_fcc
  qt_flux_y_cfc = qt_conv_flux_y_cfc + qt_diff_flux_y_cfc
  qt_flux_z_ccf = (
      qt_conv_flux_z_ccf + qt_sedimentation_flux_ccf + qt_diff_flux_z_ccf
  )

  qr_flux_x_fcc = qr_conv_flux_x_fcc + qr_diff_flux_x_fcc
  qr_flux_y_cfc = qr_conv_flux_y_cfc + qr_diff_flux_y_cfc
  qr_flux_z_ccf = (
      qr_conv_flux_z_ccf + qr_sedimentation_flux_ccf + qr_diff_flux_z_ccf
  )

  qs_flux_x_fcc = qs_conv_flux_x_fcc + qs_diff_flux_x_fcc
  qs_flux_y_cfc = qs_conv_flux_y_cfc + qs_diff_flux_y_cfc
  qs_flux_z_ccf = (
      qs_conv_flux_z_ccf + qs_sedimentation_flux_ccf + qs_diff_flux_z_ccf
  )

  # Compute the divergence of the flux.
  th_div_flux_ccc = deriv_lib.divergence_ccc(
      th_flux_x_fcc, th_flux_y_cfc, th_flux_z_ccf, sg_map
  )
  qt_div_flux_ccc = deriv_lib.divergence_ccc(
      qt_flux_x_fcc, qt_flux_y_cfc, qt_flux_z_ccf, sg_map
  )
  qr_div_flux_ccc = deriv_lib.divergence_ccc(
      qr_flux_x_fcc, qr_flux_y_cfc, qr_flux_z_ccf, sg_map
  )
  qs_div_flux_ccc = deriv_lib.divergence_ccc(
      qs_flux_x_fcc, qs_flux_y_cfc, qs_flux_z_ccf, sg_map
  )

  hw = halo_width
  aux_output['rain_precip_surf_2d_xy'] = qr_sedimentation_flux_ccf[:, :, hw]
  aux_output['surf_theta_flux_2d_xy'] = th_diff_flux_z_ccf[:, :, hw]
  aux_output['surf_q_t_flux_2d_xy'] = qt_diff_flux_z_ccf[:, :, hw]

  return (
      th_div_flux_ccc,
      qt_div_flux_ccc,
      qr_div_flux_ccc,
      qs_div_flux_ccc,
      aux_output,
  )


def compute_all_sources(
    q_t: Array,
    q_r: Array,
    q_s: Array,
    p_ref: Array,
    dt: Array,
    thermo_fields: ThermoFields,
    microphysics_cfg: microphysics_config.MicrophysicsConfig,
    radiative_transfer_cfg: radiative_transfer.RadiativeTransfer | None,
    radiation_source: Array | None,
) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
  """Compute all of the sources for the scalar equations."""
  T, rho_thermal = thermo_fields.T, thermo_fields.rho_thermal  # pylint: disable=invalid-name
  q_liq, q_ice = thermo_fields.q_liq, thermo_fields.q_ice
  q_c, q_v = thermo_fields.q_c, thermo_fields.q_v
  wp = thermo_fields.wp
  aux_output = {}

  (
      dq_t_dt_source,
      dq_r_dt_source,
      dq_s_dt_source,
      dtheta_li_dt_source,
      aux_out,
  ) = microphysics_coupling.scalar_tendencies_from_microphysics(
      q_t,
      q_r,
      q_s,
      q_v,
      q_liq,
      q_ice,
      T,
      rho_thermal,
      p_ref,
      dt,
      wp,
      microphysics_cfg,
  )
  aux_output |= aux_out

  # Compute the theta_li source due to atmospheric radiative transfer.
  if radiative_transfer_cfg is not None and radiation_source is not None:
    # The input radiation source is the temperature tendency [K/s].  Convert
    # this to the potential temperature tendency due to radiation [K/s] by
    # dividing by the Exner function.
    rm = (1 - q_t) * constants.R_D + (q_t - q_c) * constants.R_V
    cpm = (1 - q_t) * constants.CP_D + (q_t - q_c) * constants.CP_V
    exner_inv = (p_ref / wp.exner_reference_pressure) ** (-rm / cpm)
    radiative_heating_rate_theta = exner_inv * radiation_source

    # Add the radiative source to the theta_li source.
    dtheta_li_dt_source = dtheta_li_dt_source + radiative_heating_rate_theta

  # These sources are all on (ccc).
  return (
      dtheta_li_dt_source,
      dq_t_dt_source,
      dq_r_dt_source,
      dq_s_dt_source,
      aux_output,
  )


def rhs_explicit_fn(
    rho_xxc: Array,
    rho_xxf: Array,
    theta_li_ccc: Array,
    q_t_ccc: Array,
    q_r_ccc: Array,
    q_s_ccc: Array,
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    p_ref_xxc: Array,
    diffusivity_ccc: Array,
    thermo_fields: ThermoFields,
    dt: Array,
    deriv_lib: derivatives.Derivatives,
    sg_map: dict[str, Array],
    convection_cfg: convection_config.ConvectionConfig,
    microphysics_cfg: microphysics_config.MicrophysicsConfig,
    include_qt_sedimentation: bool,
    halo_width: int,
    z_bcs: boundary_conditions.ZBoundaryConditions,
    sfc_temperature: Array | None,
    sfc_q_t: Array | None,
    radiative_transfer_cfg: radiative_transfer.RadiativeTransfer | None,
    radiation_source: Array | None,
) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
  """Compute the explicit RHS for θₗᵢ, q_t, q_r, q_s.

  All terms are treated explicitly, including diffusion.

  ∂(θₗᵢ)/∂t = F_θ
  ∂(q_t)/∂t = F_q_t
  ∂(q_r)/∂t = F_q_r
  ∂(q_s)/∂t = F_q_s

  where
    F_θ = -(1 / ρ_ref) [∇·(ρ_ref Vθₗᵢ) - ∇·(ρ_ref D_t ∇θₗᵢ)] + S_θ
    F_q_t = -(1 / ρ_ref) [∇·(ρ_ref V_t q_t) - ∇·(ρ_ref D_t ∇q_t)] + S_q_t
    F_q_r = -(1 / ρ_ref) [∇·(ρ_ref V_r q_r) - ∇·(ρ_ref D_t ∇q_r)] + S_q_r
    F_q_s = -(1 / ρ_ref) [∇·(ρ_ref V_s q_s) - ∇·(ρ_ref D_t ∇q_s)] + S_q_s

  Here, V = (u,v,w) is the fluid velocity vector.  The velocities of the
  specific humidities are modified to account for the sedimentation (terminal)
  velocities, with
    V_t = V - w_c ẑ
    V_r = V - w_r ẑ
    V_s = V - w_s ẑ
  The negative sign is because terminal velocities are defined as positive when
  they are directed downward.

  The terms S_θ, S_q_t, S_q_r, and S_q_s are the source terms, which are
  computed by the microphysics model.

  * S_θ accounts for temperature changes due to conversion between water types,
    including phase changes.
  Water conversions include autoconversion, accretion, and evaporation and
  sublimation/deposition.
  """
  aux_output = {}
  # Compute ∇·(Flux) for each scalar, where Flux contains convective and
  # diffusive fluxes.
  (
      th_div_flux_ccc,
      qt_div_flux_ccc,
      qr_div_flux_ccc,
      qs_div_flux_ccc,
      aux_output1,
  ) = compute_total_flux_divergences(
      rho_xxc,
      rho_xxf,
      theta_li_ccc,
      q_t_ccc,
      q_r_ccc,
      q_s_ccc,
      u_fcc,
      v_cfc,
      w_ccf,
      diffusivity_ccc,
      thermo_fields,
      deriv_lib,
      sg_map,
      convection_cfg,
      microphysics_cfg,
      include_qt_sedimentation,
      halo_width,
      z_bcs,
      sfc_temperature,
      sfc_q_t,
  )
  aux_output |= aux_output1

  # Get all of the sources, which are on (ccc).
  dtheta_li_dt_source, dq_t_dt_source, dq_r_dt_source, dq_s_dt_source, aux = (
      compute_all_sources(
          q_t_ccc,
          q_r_ccc,
          q_s_ccc,
          p_ref_xxc,
          dt,
          thermo_fields,
          microphysics_cfg,
          radiative_transfer_cfg,
          radiation_source,
      )
  )
  aux_output |= aux

  f_theta_li_ccc = -th_div_flux_ccc / rho_xxc + dtheta_li_dt_source
  f_q_t_ccc = -qt_div_flux_ccc / rho_xxc + dq_t_dt_source
  f_q_r_ccc = -qr_div_flux_ccc / rho_xxc + dq_r_dt_source
  f_q_s_ccc = -qs_div_flux_ccc / rho_xxc + dq_s_dt_source

  return f_theta_li_ccc, f_q_t_ccc, f_q_r_ccc, f_q_s_ccc, aux_output
