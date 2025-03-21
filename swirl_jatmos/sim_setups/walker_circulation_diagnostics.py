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

"""Diagnostics for the Mock-Walker Circulation (RCEMIP-II) simulation."""

from typing import TypeAlias

import jax
import jax.numpy as jnp

from swirl_jatmos import config
from swirl_jatmos import constants
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array


def z_integral_removing_halos(f: Array) -> Array:
  """Compute the integral of `f` over the z dimension, excluding the halos."""
  f_no_halos = f[:, :, 1:-1]
  return jnp.sum(f_no_halos, axis=2)


def horiz_mean(f: Array) -> Array:
  """Compute the horizontal mean of `f`.

  Accumulate means in float64 to avoid potential numerical issues when
  accumulating, but then convert back to float32.

  Args:
    f: The array to compute the horizontal mean of.

  Returns:
    The horizontal mean of `f`.
  """
  return jnp.mean(f, axis=(0, 1), dtype=jnp.float64).astype(jnp.float32)


def diagnostics_update_fn(
    states: dict[str, Array],
    aux_output: dict[str, Array],
    diagnostics: dict[str, Array],
    dt_over_cycle_time: Array,
    cfg: config.Config,
) -> dict[str, Array]:
  """Diagnostics update function for the Mock-Walker Circulation simulation.

  `dt_over_cycle_time` is the current time step divided by the cycle time, dt/T.
  This is used to implement the temporal means, where

      mean(f) = 1/T ∫ dt f(t) ≈ ∑_i dt_i/T * f(t_i)
  """
  hw = 1  # halo width.
  a = dt_over_cycle_time  #  Shorthand for readability.
  up_d = {}  # Shorthand for 'updated diagnostics'.

  # Calculate diagnostics, involving temporal and/or horizontal means.
  if (v := 'T_1d_z') in cfg.diagnostic_fields:
    up_d[v] = diagnostics[v] + a * jnp.mean(aux_output['T'], axis=(0, 1))
  if (v := 'q_t_1d_z') in cfg.diagnostic_fields:
    up_d[v] = diagnostics[v] + a * jnp.mean(states['q_t'], axis=(0, 1))
  if (v := 'dtheta_li_1d_z') in cfg.diagnostic_fields:
    up_d[v] = diagnostics[v] + a * jnp.mean(states['dtheta_li'], axis=(0, 1))
  if (v := 'q_liq_1d_z') in cfg.diagnostic_fields:
    up_d[v] = diagnostics[v] + a * jnp.mean(aux_output['q_liq'], axis=(0, 1))
  if (v := 'q_ice_1d_z') in cfg.diagnostic_fields:
    up_d[v] = diagnostics[v] + a * jnp.mean(aux_output['q_ice'], axis=(0, 1))

  #  Surface fluxes; no spatial means.
  for v in (
      'rain_precip_surf_2d_xy',
      'surf_theta_flux_2d_xy',
      'surf_q_t_flux_2d_xy',
  ):
    if v in cfg.diagnostic_fields:
      up_d[v] = diagnostics[v] + a * aux_output[v]

  # To get outgoing longwave radiation, use toa_lw_flux_outgoing_2d_xy.
  if (v := 'toa_lw_flux_outgoing_2d_xy') in cfg.diagnostic_fields:
    up_d[v] = diagnostics[v] + a * states[v]

  return up_d

RCEMIP_2D_DIAGNOSTICS = [
    'surf_precip_rate_2d_xy',
    'surf_latent_heat_flux_2d_xy',
    'surf_sensible_heat_flux_2d_xy',
    'water_vapor_path_2d_xy',
    'sat_water_vapor_path_2d_xy',
    'condensed_water_path_2d_xy',
    'ice_water_path_2d_xy',
    'T_bottom_2d_xy',
    'u_bottom_2d_xy',
    'v_bottom_2d_xy',
    'w_500_2d_xy',
]
RCEMIP_2D_RRTMGP_DIAGNOSTICS = [
    # RRTMGP diagnostics.
    'surf_lw_flux_down_2d_xy',
    'surf_lw_flux_up_2d_xy',
    'surf_sw_flux_down_2d_xy',
    'surf_sw_flux_up_2d_xy',
    'toa_sw_flux_incoming_2d_xy',
    'toa_sw_flux_outgoing_2d_xy',
    'toa_lw_flux_outgoing_2d_xy',
    # RRTMGP clear-sky diagnostics.
    'surf_lw_flux_down_clearsky_2d_xy',
    'surf_lw_flux_up_clearsky_2d_xy',
    'surf_sw_flux_down_clearsky_2d_xy',
    'surf_sw_flux_up_clearsky_2d_xy',
    'toa_sw_flux_outgoing_clearsky_2d_xy',
    'toa_lw_flux_outgoing_clearsky_2d_xy',
]
RCEMIP_1D_DIAGNOSTICS = [
    'T_1d_z',
    'u_1d_z',
    'v_1d_z',
    'q_t_1d_z',
    'rel_humidity_1d_z',
    'q_liq_1d_z',
    'q_ice_1d_z',
    'q_r_1d_z',
    'q_s_1d_z',
    'theta_1d_z',
    'thetae_1d_z',
    'cloud_frac_1d_z',
]
RCEMIP_1D_RRTMGP_DIAGNOSTICS = [
    # RRTMGP diagnostics.
    'rad_heat_lw_1d_z',
    'rad_heat_sw_1d_z',
    # RRTMGP clear-sky diagnostics.
    'rad_heat_lw_clearsky_1d_z',
    'rad_heat_sw_clearsky_1d_z',
]
RCEMIP_0D_DIAGNOSTICS = [
    'surf_precip_rate_0d',
    'surf_latent_heat_flux_0d',
    'surf_sensible_heat_flux_0d',
    'water_vapor_path_0d',
    'sat_water_vapor_path_0d',
    'condensed_water_path_0d',
    'ice_water_path_0d',
    # RRTMGP diagnostics.
    'surf_lw_flux_down_0d',
    'surf_lw_flux_up_0d',
    'surf_sw_flux_down_0d',
    'surf_sw_flux_up_0d',
    'toa_sw_flux_incoming_0d',
    'toa_sw_flux_outgoing_0d',
    'toa_lw_flux_outgoing_0d',
    # RRTMGP clear-sky diagnostics.
    'surf_lw_flux_down_clearsky_0d',
    'surf_lw_flux_up_clearsky_0d',
    'surf_sw_flux_down_clearsky_0d',
    'surf_sw_flux_up_clearsky_0d',
    'toa_sw_flux_outgoing_clearsky_0d',
    'toa_lw_flux_outgoing_clearsky_0d',
]

RCEMIP_DIAGNOSTICS_SET = (
    set(RCEMIP_2D_DIAGNOSTICS)
    | set(RCEMIP_2D_RRTMGP_DIAGNOSTICS)
    | set(RCEMIP_1D_DIAGNOSTICS)
    | set(RCEMIP_1D_RRTMGP_DIAGNOSTICS)
    | set(RCEMIP_0D_DIAGNOSTICS)
)


def rcemip_diagnostics_update_fn(
    states: dict[str, Array],
    aux_output: dict[str, Array],
    diagnostics: dict[str, Array],
    dt_over_cycle_time: Array,
    cfg: config.Config,
) -> dict[str, Array]:
  """Diagnostics update function with RCEMIP-II diagnostics.

  Here, all diagnostics are required (an error will occur if the fields are not
  present in cfg.diagnostics_fields).

  `dt_over_cycle_time` is the current time step divided by the cycle time, dt/T.
  This is used to implement the temporal means, where

      mean(f) = 1/T ∫ dt f(t) ≈ ∑_i dt_i/T * f(t_i)

  Reference: Wing et al, RCEMIP-II: mock-Walker simulations as phase II
  of the radiative-convective equilibrium model intercomparison project,
  Geosci. Model Dev., 17, 6195-6225, 2024.
  """
  assert RCEMIP_DIAGNOSTICS_SET.issubset(
      set(cfg.diagnostic_fields)
  ), '`cfg.diagnostic_fields` must contain all RCEMIP diagnostics.'

  hw = 1  # halo width.
  a = dt_over_cycle_time  # Convenient shorthand.
  up_d = dict(diagnostics)  # Make copy; (shorthand for 'updated diagnostics').

  u = states['u']
  v = states['v']
  w = states['w']
  q_t = states['q_t']
  q_r = states['q_r']
  q_s = states['q_s']
  q_v = aux_output['q_v']
  q_v_sat = aux_output['q_v_sat']
  q_liq = aux_output['q_liq']
  q_ice = aux_output['q_ice']
  rho = states['rho_xxc']  # rho_ref(z)
  p_ref = states['p_ref_xxc']  # p_ref(z)
  T = aux_output['T']  # pylint: disable=invalid-name
  q_c = q_liq + q_ice

  # 2D diagnostics, as a function of x & y.
  # Surface precipitation rate.  Take the negative to get the downward flux.
  up_d['surf_precip_rate_2d_xy'] -= a * aux_output['rain_precip_surf_2d_xy']

  # Surface upward latent heat flux.
  # Latent heat of vaporization; used as the conversion factor from evaporation
  # flux to latent heat flux.
  humidity_flux_to_latent_heat_flux = cfg.wp.lh_v0
  up_d['surf_latent_heat_flux_2d_xy'] += (
      humidity_flux_to_latent_heat_flux * a * aux_output['surf_q_t_flux_2d_xy']
  )

  # Surface upward sensible heat flux.
  q_t_bot = q_t[:, :, hw]
  cp_m_surf = (1 - q_t_bot) * constants.CP_D + q_t_bot * constants.CP_V
  # Exner function = 1 at the surface because the reference pressure is the
  # surface pressure.
  exner_surf = 1.0
  theta_flux_to_sensible_heat_flux = exner_surf * cp_m_surf
  up_d['surf_sensible_heat_flux_2d_xy'] += (
      theta_flux_to_sensible_heat_flux * a * aux_output['surf_theta_flux_2d_xy']
  )

  # ********  Radiation 2D diagnostics  ********
  for b in RCEMIP_2D_RRTMGP_DIAGNOSTICS:
    up_d[b] += a * states[b]

  # ********  Water path diagnostics  ********
  up_d['water_vapor_path_2d_xy'] += a * z_integral_removing_halos(rho * q_v)
  up_d['sat_water_vapor_path_2d_xy'] += a * z_integral_removing_halos(
      rho * q_v_sat
  )
  up_d['condensed_water_path_2d_xy'] += a * z_integral_removing_halos(
      rho * (q_liq + q_ice)
  )
  up_d['ice_water_path_2d_xy'] += a * z_integral_removing_halos(rho * q_ice)

  # ********  Other 2D diagnostics  ********
  # Air temperature and horizontal wind at lowest model level.
  up_d['T_bottom_2d_xy'] += a * T[:, :, hw]
  up_d['u_bottom_2d_xy'] += a * u[:, :, hw]
  up_d['v_bottom_2d_xy'] += a * v[:, :, hw]

  # Vertical velocity at 500 hPa level.
  # For the RCEMIP z levels, this occurs at index 20 for w.  If the z levels are
  # changed, we'll need to modify this index.
  z_f_index_500_hpa = 20
  up_d['w_500_2d_xy'] += a * w[:, :, z_f_index_500_hpa]

  # ******** 1D Diagnostics -- horizontal means, giving functions of z. ********
  up_d['T_1d_z'] += a * horiz_mean(T)
  up_d['u_1d_z'] += a * horiz_mean(u)
  up_d['v_1d_z'] += a * horiz_mean(v)
  up_d['q_t_1d_z'] += a * horiz_mean(q_t)

  up_d['rel_humidity_1d_z'] += a * horiz_mean(q_v / q_v_sat)
  up_d['q_liq_1d_z'] += a * horiz_mean(q_liq)
  up_d['q_ice_1d_z'] += a * horiz_mean(q_ice)
  up_d['q_r_1d_z'] += a * horiz_mean(q_r)
  up_d['q_s_1d_z'] += a * horiz_mean(q_s)

  # Compute the potential temperature.
  rm = (1 - q_t) * constants.R_D + (q_t - q_c) * constants.R_V
  cpm = (1 - q_t) * constants.CP_D + (q_t - q_c) * constants.CP_V
  p_0 = cfg.wp.exner_reference_pressure
  exner_inv = (p_ref / p_0) ** (-rm / cpm)
  theta = exner_inv * T
  up_d['theta_1d_z'] += a * horiz_mean(theta)

  # Equivalent potential temperature with relative humidity = 1.
  R_D, CP_D, CP_L = constants.R_D, constants.CP_D, cfg.wp.cp_l  # pylint: disable=invalid-name
  lh_v = water.lh_v(T, cfg.wp)
  exp_factor = jnp.exp(lh_v * q_v / ((CP_D + q_t * CP_L) * T))
  theta_e = T * (p_0 / p_ref) ** (R_D / (CP_D + q_t * CP_L)) * exp_factor
  up_d['thetae_1d_z'] += a * horiz_mean(theta_e)

  # Global cloud fraction: the fraction of points (for each z level) where the
  # condensed cloud water mixing ratio > 1e-5.  Treat mixing ratio as
  # approximately equal to specific humidity.
  mx, my = len(cfg.x_c), len(cfg.y_c)  # Total number of x,y grid points.
  cloud_frac = jnp.count_nonzero(q_c > 1e-5, axis=(0, 1)) / (mx * my)
  up_d['cloud_frac_1d_z'] += a * cloud_frac.astype(jnp.float32)

  # Radiation 1D diagnostics.
  for b in RCEMIP_1D_RRTMGP_DIAGNOSTICS:
    up_d[b] += a * states[b]

  return up_d

# ******************** 0D Diagnostics ********************
# All of these can be computed as horizontal averages from the 2D diagnostics.


def rcemip_diagnostics_end_of_cycle_fn(
    diagnostics: dict[str, Array],
) -> dict[str, Array]:
  """Compute the 0D diagnostics from the 2D diagnostics.

  These don't need to be computed every step, since they don't need to be
  accumulated over time but can be computed directly from the 2D diagnostics.
  They can be computed at the end of each cycle (or in postprocessing), but
  for convenience they are done here.

  Args:
    diagnostics: The diagnostics dictionary.

  Returns:
    An updated diagnostics dictionary containing the 0D diagnostics.
  """
  up_d = dict(diagnostics)

  for b in RCEMIP_0D_DIAGNOSTICS:
    b_2d = b.replace('_0d', '_2d_xy')
    up_d[b] = horiz_mean(diagnostics[b_2d])
  return up_d
