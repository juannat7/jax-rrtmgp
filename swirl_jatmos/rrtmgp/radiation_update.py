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

"""Module for handling the sporadic update of the radiation source."""

import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp
from swirl_jatmos import config
from swirl_jatmos import sim_initializer
from swirl_jatmos import stretched_grid_util
from swirl_jatmos.rrtmgp import rrtmgp
from swirl_jatmos.rrtmgp import rrtmgp_common
from swirl_jatmos.rrtmgp.config import radiative_transfer
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array
StatesMap: TypeAlias = dict[str, Array]

KEY_APPLIED_RADIATION = rrtmgp_common.KEY_APPLIED_RADIATION
KEY_STORED_RADIATION = rrtmgp_common.KEY_STORED_RADIATION


def initialize_radiation_states(
    radiative_transfer_cfg: radiative_transfer.RadiativeTransfer,
    cfg: config.Config,
    template_field_3d: Array,
) -> StatesMap:
  """Initialize the radiation states.

  Args:
    radiative_transfer_cfg: The radiative transfer configuration.
    cfg: The simulation configuration.
    template_field_3d: A field for which the shape, dtype, and sharding will be
      used to initialize the 3D radiation states.

  Returns:
    A dictionary of radiation states with the required keys initialized to zero.
  """
  # rad_keys = 'rad_heat_src_applied', 'rad_heat_src', 'rad_flux_{lw,sw}'
  # rad_keys = rrtmgp_common.required_keys(radiative_transfer_cfg)
  rad_keys = ['rad_heat_src', 'rad_heat_src_applied']
  if radiative_transfer_cfg.save_lw_sw_heating_rates:
    rad_keys.extend(['rad_heat_lw_3d', 'rad_heat_sw_3d'])
    if radiative_transfer_cfg.do_clear_sky:
      rad_keys.extend(['rad_heat_lw_clearsky_3d', 'rad_heat_sw_clearsky_3d'])

  shard_3d = functools.partial(sim_initializer.shard_3d, cfg=cfg)
  rad_states = {
      k: shard_3d(jnp.zeros_like(template_field_3d)) for k in rad_keys
  }

  shard_0d = functools.partial(sim_initializer.shard_0d, cfg=cfg)
  # Initialize update time to -1, which means the radiation update will trigger
  # on the first step (t=0).
  rad_states['rad_next_update_time_ns'] = shard_0d(
      jnp.array(-1, dtype=jnp.int64)
  )

  # When supercycling is used, initialize next apply time to -1, which means the
  # radiation will be applied on the first step (t=0) (but with negligible
  # heating, dt = 1 ns).
  if radiative_transfer_cfg.apply_cadence_seconds > 0:
    next_apply_time_ns = shard_0d(jnp.array(-1, dtype=jnp.int64))
    rad_states['rad_next_apply_time_ns'] = next_apply_time_ns

  #  ******** Initialize states for Diagnostics *********
  allowed_rad_diagnostics = rrtmgp_common.DIAGNOSTICS_KEYS
  for fieldname in allowed_rad_diagnostics:
    if fieldname in cfg.diagnostic_fields:
      rad_states[fieldname] = sim_initializer.initialize_zeros_from_varname(
          fieldname, cfg
      )

  return rad_states


def filter_radiation_states(
    states: StatesMap,
    radiative_transfer_cfg: radiative_transfer.RadiativeTransfer,
) -> StatesMap:
  """From the input `states`, keep only radiation states."""
  # rad_keys = 'rad_heat_src_applied', 'rad_heat_src', 'rad_flux_{lw,sw}'
  rad_keys = rrtmgp_common.required_keys(radiative_transfer_cfg)
  # Remove 'rad_heat_src_applied' because we want to filter it out so we don't
  # update it here.
  rad_keys.remove(KEY_APPLIED_RADIATION)
  rad_keys.extend(['rad_heat_lw_3d', 'rad_heat_sw_3d'])
  rad_keys.extend(['rad_heat_lw_clearsky_3d', 'rad_heat_sw_clearsky_3d'])
  rad_keys.extend(rrtmgp_common.DIAGNOSTICS_KEYS)
  rad_keys.append('rad_next_update_time_ns')
  return {k: states[k] for k in states if k in rad_keys}


def radiation_update_fn(states: StatesMap, cfg: config.Config) -> StatesMap:
  """Perform the radiative transfer calculation."""
  assert cfg.radiative_transfer_cfg is not None
  radiative_transfer_cfg = cfg.radiative_transfer_cfg
  update_cycle_ns = int(1e9 * radiative_transfer_cfg.update_cycle_seconds)

  # This should be instantiated statically when inside JIT.  Check this...
  rrtmgp_ = rrtmgp.RRTMGP(
      radiative_transfer_cfg,
      cfg.wp,
      cfg.grid_spacings[2],
      cfg.diagnostic_fields,
  )

  # Extract required states.
  rho_xxc = states['rho_xxc']
  q_t = states['q_t']
  theta_li = states['theta_li_0'] + states['dtheta_li']
  p_ref_xxc = states['p_ref_xxc']
  sfc_temperature = states['sfc_temperature_2d_xy']

  # Compute thermodynamic fields.
  rho_thermal_initial_guess = jnp.broadcast_to(rho_xxc, theta_li.shape)
  thermo_fields = water.compute_thermodynamic_fields_from_prognostic_fields(
      theta_li, q_t, p_ref_xxc, rho_thermal_initial_guess, cfg.wp
  )
  temperature = thermo_fields.T
  q_liq = thermo_fields.q_liq
  q_ice = thermo_fields.q_ice
  q_c = thermo_fields.q_c

  sg_map = stretched_grid_util.sg_map_from_states(states)

  # Keys returned: 'rad_heat_src', 'rad_flux_{lw,sw}', and
  # 'rad_next_update_time_ns'.
  output = rrtmgp_.compute_heating_rate(
      rho_xxc,
      q_t,
      q_liq,
      q_ice,
      q_c,
      temperature,
      sfc_temperature,
      p_ref_xxc,
      sg_map,
      use_scan=radiative_transfer_cfg.use_scan,
  )

  output['rad_next_update_time_ns'] = states['t_ns'] + update_cycle_ns
  return output


def radiation_preprocess_update_fn(
    states: StatesMap,
    cfg: config.Config,
) -> StatesMap:
  """Run the RRTMGP calculation sporadically.

  There are two separate but complementary calculations here:
  1. The radiative heating rate is recomputed every `update_cycle_seconds`, and
  in between updates the most recently computed heating rate is carried over.

  2. The radiative heating rate is applied every `apply_cadence_seconds` and
  scaled appropriately.  This is accomplished by using another field in `states`
  that is zero when not applied, but is nonzero when the radiative heating is
  applied to the fluid.

  This function is called just before the Navier-Stokes step.

  Args:
    states: The simulation states.
    cfg: The simulation configuration.

  Returns:
    A dictionary of radiation states with the required keys updated.
  """
  assert cfg.radiative_transfer_cfg is not None, 'RRTMGP is not enabled.'
  assert (
      'sfc_temperature_2d_xy' in states
  ), 'Field `sfc_temperature_2d_xy` is required in `states`.'
  assert (
      states['sfc_temperature_2d_xy'].ndim == 2
  ), 'Field `sfc_temperature_2d_xy` must be a 2D array.'

  radiative_transfer_cfg = cfg.radiative_transfer_cfg
  apply_cadence_ns = int(1e9 * radiative_transfer_cfg.apply_cadence_seconds)

  update_condition = states['t_ns'] >= states['rad_next_update_time_ns']

  # The radiative heating rate is recomputed every `update_cycle_seconds`, and
  # between updates the most recently computed heating rate is carried over.
  # This enables the radiative heating rate to have a persistent contribution to
  # to the energy equation even if it is being updated somewhat infrequently.
  # The staleness is justified because the time scale of radiative flux profile
  # changes is very large compared to the typical timestep of the LES.

  # Keys returned: 'rad_heat_src', 'rad_flux_{lw,sw}', and
  # 'rad_next_update_time_ns'.
  # When `update_condition` is triggered, update these radiation keys, otherwise
  # just pass them through `states` unchanged.
  rad_states = jax.lax.cond(
      pred=update_condition,
      true_fun=lambda: radiation_update_fn(states, cfg),
      false_fun=lambda: filter_radiation_states(states, radiative_transfer_cfg),
  )

  # The radiative heating rate is applied every `apply_cadence_ns`. This is a
  # form of supercycling of the radiation heating, where it is applied not every
  # step, only every certain number of steps. This is necessary because the
  # the radiative heating rate can be so small that the updates are lost under
  # single precision. The radiative heating is applied based on
  # `additional_states[rrtmgp_common.KEY_APPLIED_RADIATION]`. When we
  # supercycle, we multiply the radiative heating rate by the amount of time
  # divided by dt that has passed since the last time it was applied.

  # If `apply_cadence_seconds` is set to 0 (the default), we do no supercycling
  # and apply the radiative heating at every step.  This `if` is compiled
  # statically, so if there is no supercycling, we can avoid the jax.lax.cond.
  if apply_cadence_ns == 0:
    rad_states2 = {KEY_APPLIED_RADIATION: states[KEY_STORED_RADIATION]}
  else:
    apply_condition = states['t_ns'] >= states['rad_next_apply_time_ns']

    def apply_fun() -> dict[str, Array]:
      time_elapsed_since_last_apply_ns = (
          states['t_ns'] - states['rad_next_apply_time_ns'] + apply_cadence_ns
      )
      # Description of scaling factor:
      # Over a time interval 𝜏, the net heating effect (change in temperature T)
      # is ∫P dt, where P is the "heating rate" (∂T/∂t).  If the heating is
      # applied every dt, then the net heating effect is P * dt.  If the heating
      # is applied over a time interval 𝜏, the net heating effect is
      # approximately P𝜏.  When the equation of motion is stepped forward, the
      # heating is scaled by dt.  Therefore, the appropriate scaling factor here
      # is 𝜏 / dt.  Note that when 𝜏 = dt, the scaling factor is 1.
      scaling_factor = (
          time_elapsed_since_last_apply_ns / states['dt_ns']
      ).astype(states[KEY_STORED_RADIATION].dtype)
      return {
          KEY_APPLIED_RADIATION: scaling_factor * states[KEY_STORED_RADIATION],
          'rad_next_apply_time_ns': states['t_ns'] + apply_cadence_ns,
      }

    rad_states2 = jax.lax.cond(
        pred=apply_condition,
        true_fun=apply_fun,
        false_fun=lambda: {
            KEY_APPLIED_RADIATION: jnp.zeros_like(states[KEY_STORED_RADIATION]),
            'rad_next_apply_time_ns': states['rad_next_apply_time_ns'],
        },
    )
  rad_states |= rad_states2

  # Note: Since the heating rate is the only quantity directly used, the
  # diagnostic states like rad_flux_lw, rad_flux_sw, etc., would typically go
  # in to `aux_output`.  However, that is not possible because of the sporadic
  # radiative transfer calculations.  Since `aux_output` is never an input, the
  # only way to pass states through from one step to the next is to include them
  # in `states`.  Therefore, we include the diagnostic states in `states`.
  return rad_states
