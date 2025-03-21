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

"""Setup for the Mock-Walker Circulation (RCEMIP-II) simulation."""

import functools
from typing import TypeAlias

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from swirl_jatmos import config
from swirl_jatmos import constants
from swirl_jatmos import convection_config
from swirl_jatmos import interpolation
from swirl_jatmos import sim_initializer
from swirl_jatmos import sponge_config
from swirl_jatmos import timestep_control_config
from swirl_jatmos.boundary_conditions import boundary_conditions
from swirl_jatmos.boundary_conditions import monin_obukhov
from swirl_jatmos.rrtmgp import radiation_update
from swirl_jatmos.rrtmgp.config import radiative_transfer
from swirl_jatmos.sim_setups import walker_circulation_parameters
from swirl_jatmos.sim_setups import walker_circulation_utils
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array
WaterParams: TypeAlias = water.WaterParams
WalkerCirculationParameters: TypeAlias = (
    walker_circulation_parameters.WalkerCirculationParameters
)

_EPS = constants.R_V / constants.R_D
# Default value for the surface [Pa] and for the reference pressure in the Exner
# function for potential temperature.
_P0 = 1.0148e5


def _humidity_profile_smooth(z: jax.typing.ArrayLike, q_0: float) -> Array:
  """Provides smooth humidity profile."""
  zq1 = 4e3
  zq2 = 7.5e3
  zq3 = 17e3
  q_inf = 1e-14

  q = (
      q_0
      * jnp.exp(-z / zq1)
      * jnp.exp(-((z / zq2) ** 2))
      * jnp.exp(-8 * (z / zq3) ** 10)
  )
  q = jnp.where(q < q_inf, q_inf, q)
  return q


def analytic_profiles_from_paper(
    z: Array, wcp: WalkerCirculationParameters, p_0: float
) -> tuple[Array, Array, Array, Array]:
  """Provides the initial and reference profiles from the paper.

  Initial profiles for q_t, p, T are taken from the analytic profiles given in
  Wing et al (2018), the RCEMIP-I paper.

  Args:
    z: The vertical coordinates.
    wcp: The Walker Circulation parameters.
    p_0: The pressure at the surface [Pa].

  Returns:
    Get initial conditions for  initial profile of q_t, the reference
    hydrostatic pressure p_ref, the initial (unperturbed) temperature T, and the
    reference density rho_ref.
  """
  q_0, z_t = wcp.q_0, wcp.z_t
  # Initialize the specific humidity.

  # These formulas are from Wing (2018) but have a large discontinuity.
  # q_0, q_t, z_q1, z_q2, z_t = wcp.q_0, wcp.q_t, wcp.z_q1, wcp.z_q2, wcp.z_t
  # q_below_tropopause = q_0 * jnp.exp(-z / z_q1) * jnp.exp(-((z / z_q2) ** 2))
  # q_t_above_tropopause = q_t * jnp.ones_like(z)
  # q_t = jnp.where(z > z_t, q_t_above_tropopause, q_below_tropopause)

  # Smoothed specific humidity.
  q_t = _humidity_profile_smooth(z, q_0)

  # Initialize the virtual temperature.
  # pylint: disable=invalid-name
  T0 = wcp.sst_0  # Sea-surface temperature (using the average value)
  gamma = wcp.gamma
  Tv0 = T0 * (1 + (_EPS - 1) * q_0)  # Virtual temperature at the surface.
  Tvt = Tv0 - gamma * z_t  # Virtual temperature at the tropopause.
  Tv_above_tropopause = Tvt * jnp.ones_like(z)
  Tv_below_tropopause = Tv0 - gamma * z
  Tv = jnp.where(z > z_t, Tv_above_tropopause, Tv_below_tropopause)

  # Calculate the temperature from the virtual temperature.
  T = Tv / (1 + (_EPS - 1) * q_0)
  # pylint: enable=invalid-name

  # Initialize the hydrostatic/reference pressure.
  g, r_d = constants.G, constants.R_D
  p_t = p_0 * (Tvt / Tv0) ** (g / (r_d * gamma))  # Pressure at the tropopause.
  # Formulas given for p_ref below and above the tropopause.
  p_ref_below_tropopause = p_0 * (1 - gamma * z / Tv0) ** (g / (r_d * gamma))
  p_ref_above_tropopause = p_t * jnp.exp(-(g * (z - z_t) / (r_d * Tvt)))
  p_ref = jnp.where(z > z_t, p_ref_above_tropopause, p_ref_below_tropopause)

  # Initialize the density.
  rho_ref = p_ref / (r_d * Tv)  # This takes moisture into account.

  return q_t, p_ref, T, rho_ref


def init_fn(
    cfg: config.Config, wcp: WalkerCirculationParameters
) -> dict[str, Array]:
  """Create sharded initial conditions for the Walker Circulation simulation."""
  grid_map_sharded = sim_initializer.initialize_grids(cfg)

  # Extract required grid variables.
  x_c = grid_map_sharded['x_c']
  y_c = grid_map_sharded['y_c']
  z_c = grid_map_sharded['z_c']
  nx, ny, _ = x_c.shape[0], y_c.shape[1], z_c.shape[2]
  wp = cfg.wp
  lx = cfg.domain_x[1]

  # Set the pressure at the surface to be the same as the reference pressure
  # used in the Exner function.
  p_0 = wp.exner_reference_pressure

  # pylint: disable=invalid-name
  q_t_c, p_ref_xxc, T_c, rho_ref_xxc = analytic_profiles_from_paper(
      z_c, wcp, p_0
  )
  # pylint: enable=invalid-name

  # If using a numerical sounding for θ_li and q_t, we override the analytic
  # profiles for T and q_t.
  use_numerical_sounding = bool(wcp.sounding_dirname)
  if use_numerical_sounding:
    logging.info('Using numerical sounding for θ_li and q_t.')
    theta_li_0_c, q_t_c = (
        walker_circulation_utils.load_sounding_and_interpolate(
            wcp.sounding_dirname, z_c
        )
    )

  # Use interpolation in z rather than evaluating exactly at the face points.
  # This is numerically better because we interpolate rho_thermal from z nodes
  # to z faces, as we don't have an exact evaluation for rho_thermal at z
  # faces.  Therefore, since buoyancy is determined by the difference of
  # rho_thermal and rho_ref, it is more consistent to interpolate rho_ref
  # rather than evaluate a prescribed formula on faces.  A nice benefit of
  # this approach is that it makes the initial pressure balancing buoyancy
  # exactly equal to zero (assuming we include moisture in the reference).
  rho_ref_xxf = interpolation.z_c_to_f(rho_ref_xxc)

  q_t_ccc = jnp.tile(q_t_c, reps=(nx, ny, 1))
  u_fcc = jnp.zeros_like(q_t_ccc)
  v_cfc = jnp.zeros_like(q_t_ccc)
  w_ccf = jnp.zeros_like(q_t_ccc)
  p_ccc = jnp.zeros_like(q_t_ccc)
  q_r_ccc = jnp.zeros_like(q_t_ccc)
  q_s_ccc = jnp.zeros_like(q_t_ccc)

  shard_3d = functools.partial(sim_initializer.shard_3d, cfg=cfg)
  shard_2d_horiz = functools.partial(sim_initializer.shard_2d_horiz, cfg=cfg)
  shard_1d_z = functools.partial(
      sim_initializer.shard_broadcastable_arr, cfg=cfg, dim_names='z'
  )

  p_ref_xxc = shard_1d_z(p_ref_xxc)
  rho_ref_xxc = shard_1d_z(rho_ref_xxc)
  rho_ref_xxf = shard_1d_z(rho_ref_xxf)
  q_t_ccc = shard_3d(q_t_ccc)
  u_fcc = shard_3d(u_fcc)
  v_cfc = shard_3d(v_cfc)
  q_r_ccc = shard_3d(q_r_ccc)
  q_s_ccc = shard_3d(q_s_ccc)
  w_ccf = shard_3d(w_ccf)
  p_ccc = shard_3d(p_ccc)

  # Add perturbation to the temperature in the lowest 5 layers as prescribed in
  # Wing et al. (2018), for symmetry breaking.
  random_seed = 4242  # Random seed for the perturbation.
  key = jax.random.key(random_seed)
  pert_val = jax.random.normal(key, shape=(nx, ny, 5), dtype=q_t_c.dtype)
  amplitudes = jnp.array([0.1, 0.08, 0.06, 0.04, 0.02], dtype=q_t_c.dtype)
  amplitudes = amplitudes[jnp.newaxis, jnp.newaxis, :]
  pert_val = pert_val * amplitudes * wcp.theta_li_pert_scaling

  # If using analytic profile for T, add perturbation to T.  But if we use the
  # numerical sounding for theta_li, add the perturbation to theta_li instead.
  if not use_numerical_sounding:
    # pylint: disable=invalid-name
    # Add the perturbation to the lowest 5 layers (ignore the halo).
    T_pert_only_5_layers = T_c[:, :, 1:6] + pert_val
    # Create a new temperature field with the perturbation.
    T_pert_ccc = jnp.broadcast_to(T_c, q_t_ccc.shape)
    T_pert_ccc = T_pert_ccc.at[:, :, 1:6].set(T_pert_only_5_layers)
    T_pert_ccc = shard_3d(T_pert_ccc)
    # pylint: enable=invalid-name

    # Construct theta_li from the temperature.
    # For now, assume no condensate, so theta_li = theta.
    # First, construct theta_li_0.
    r_m_1d = (1 - q_t_c) * constants.R_D + q_t_c * constants.R_V
    cp_m_1d = (1 - q_t_c) * constants.CP_D + q_t_c * constants.CP_V
    exner_1d = (p_ref_xxc / wp.exner_reference_pressure) ** (r_m_1d / cp_m_1d)
    theta_li_0_c = T_c / exner_1d  # (1, 1, nz) field.

    # Now, construct full theta from the perturbed T.
    r_m = (1 - q_t_ccc) * constants.R_D + q_t_ccc * constants.R_V
    cp_m = (1 - q_t_ccc) * constants.CP_D + q_t_ccc * constants.CP_V

    # Compute the temperature from the potential temperature.
    exner = (p_ref_xxc / wp.exner_reference_pressure) ** (r_m / cp_m)
    theta_li_ccc = T_pert_ccc / exner  # (nx, ny, nz) field.
    dtheta_li_ccc = theta_li_ccc - theta_li_0_c
  else:
    # Use the numerical sounding for theta_li.  Use the random perturbation as
    # the initial value for dtheta_li_ccc.
    dtheta_li_ccc = jnp.zeros_like(q_t_ccc)
    dtheta_li_ccc = dtheta_li_ccc.at[:, :, 1:6].set(pert_val)

  theta_li_0_c = shard_1d_z(theta_li_0_c)  # pylint: disable=undefined-variable
  dtheta_li_ccc = shard_3d(dtheta_li_ccc)

  # Initialize the sharded radiation states.
  assert (radiative_transfer_cfg := cfg.radiative_transfer_cfg) is not None
  rad_states = radiation_update.initialize_radiation_states(
      radiative_transfer_cfg, cfg, q_t_ccc
  )

  # Initialize the sea-surface temperature.
  sst_0, delta_sst = wcp.sst_0, wcp.delta_sst
  sfc_temperature = sst_0 - 0.5 * delta_sst * jnp.cos(2 * np.pi * x_c / lx)
  sfc_temperature = jnp.tile(sfc_temperature, reps=(1, ny, 1))
  sfc_temperature = sfc_temperature[:, :, 0]  # (nx, ny) field.
  # Add 1 Kelvin to the sea-surface temperature; used for Monin-Obukhov BC and
  # for the radiation temperature BC.
  sfc_temperature = sfc_temperature + 1.0
  sfc_temperature = shard_2d_horiz(sfc_temperature)

  # Initialize the surface humidity q_t to be the saturation vapor humidity at
  # the sea-surface temperature.  That is:
  #     q_t_surface = q_vap_sat(T_surface, rho_surface)
  sfc_rho = rho_ref_xxf[:, :, 1]  # Shape (1, 1).
  sfc_q_t = water.saturation_vapor_humidity_over_liquid_water(
      sfc_temperature, sfc_rho, wp
  )  # Shape (nx, ny).

  # Or, use:
  # sfc_q_t = 0.019 * jnp.ones_like(sfc_temperature)
  sfc_q_t = shard_2d_horiz(sfc_q_t)

  states = {
      'p_ref_xxc': p_ref_xxc,
      'rho_xxc': rho_ref_xxc,
      'rho_xxf': rho_ref_xxf,
      'theta_li_0': theta_li_0_c,
      'dtheta_li': dtheta_li_ccc,
      'q_t': q_t_ccc,
      'q_r': q_r_ccc,
      'q_s': q_s_ccc,
      'u': u_fcc,
      'v': v_cfc,
      'w': w_ccf,
      'p': p_ccc,
  }
  states |= rad_states
  states['sfc_temperature_2d_xy'] = sfc_temperature
  states['sfc_q_t_2d_xy'] = sfc_q_t
  return states


def preprocess_update_fn(
    states: dict[str, Array],
    cfg: config.Config,
) -> tuple[dict[str, Array], dict[str, Array]]:
  """Preprocess update function for the Walker Circulation simulation.

  The output states are used to update the main states dict in the driver (they
  do update rather than replace it).

  Args:
    states: The current states.
    cfg: The configuration.

  Returns:
    A dict of states & aux_output that are updated as part of the preprocess.
    Here, return the radiation states and an empty aux_output.
  """
  rad_states = radiation_update.radiation_preprocess_update_fn(states, cfg)
  aux_output = {}
  return rad_states, aux_output
