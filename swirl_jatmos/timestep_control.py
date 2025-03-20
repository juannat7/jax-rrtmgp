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

"""Module for adaptive timestep control."""

from typing import TypeAlias

from absl import logging
import jax
import jax.numpy as jnp
from swirl_jatmos import timestep_control_config

Array: TypeAlias = jax.Array
TimestepControlConfig: TypeAlias = timestep_control_config.TimestepControlConfig


def compute_next_dt(
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    dx_f: Array | float,
    dy_f: Array | float,
    dz_f: Array | float,
    dt_ns: Array,
    step_id: Array,
    timestep_control_cfg: TimestepControlConfig,
):
  """Compute the next timestep [ns], if the update condition is met."""
  if timestep_control_cfg.disable_adaptive_timestep:
    logging.info('Adaptive timestep control is disabled.')
    return dt_ns

  update_condition = step_id % timestep_control_cfg.update_interval_steps == 0
  next_dt_ns = jax.lax.cond(
      pred=update_condition,
      true_fun=lambda: _compute_next_dt(
          u_fcc, v_cfc, w_ccf, dx_f, dy_f, dz_f, dt_ns, timestep_control_cfg
      ),
      false_fun=lambda: dt_ns,
  )
  return next_dt_ns


def _compute_next_dt(
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    dx_f: Array | float,
    dy_f: Array | float,
    dz_f: Array | float,
    dt_ns: Array,
    timestep_control_cfg: TimestepControlConfig,
) -> Array:
  """Compute the next timestep [ns], returning a scalar array."""
  # Local advection timescale [s] for each grid point.
  adv_timescale = advection_timescale(u_fcc, v_cfc, w_ccf, dx_f, dy_f, dz_f)

  # Target timestep [s], for each grid point.
  target_dt = timestep_control_cfg.desired_cfl * adv_timescale
  dt = (dt_ns * 1e-9).astype(u_fcc.dtype)  # Convert dt to seconds.
  next_dt = _compute_next_dt_from_target_dt(target_dt, dt, timestep_control_cfg)

  # Perform an all-reduce, determining the minimum timestep across the entire
  # grid.  The result is replicated across all cores.  Before performing the
  # all-reduce, remove halos.
  halo_width = 1
  next_dt = jnp.min(next_dt[:, :, halo_width:-halo_width])

  # Convert back to nanoseconds.
  next_dt_ns = (next_dt * 1e9).astype(dt_ns.dtype)
  return next_dt_ns


def advection_timescale(
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    dx_f: Array | float,
    dy_f: Array | float,
    dz_f: Array | float,
) -> Array:
  """Compute the pointwise advection timescale [s], for each grid point."""
  inverse_timescale_x = jnp.abs(u_fcc) / dx_f
  inverse_timescale_y = jnp.abs(v_cfc) / dy_f
  inverse_timescale_z = jnp.abs(w_ccf) / dz_f
  inverse_timescale = (
      inverse_timescale_x + inverse_timescale_y + inverse_timescale_z
  )
  return 1 / inverse_timescale


def _compute_next_dt_from_target_dt(
    target_dt: Array,
    dt: Array,
    timestep_control_cfg: TimestepControlConfig,
) -> Array:
  """Compute a target timestep pointwise from the advection timescale.

  This function returns an array of timesteps of the same shape as the input
  `target_dt`.  All operations are pointwise.

  Args:
    target_dt: The target timescale for each grid point [s].
    dt: The current timestep [s].
    timestep_control_cfg: The timestep control configuration.

  Returns:
    A new timestep [s] computed for each grid point.
  """
  max_dt = timestep_control_cfg.max_dt
  min_dt = timestep_control_cfg.min_dt
  max_change_factor = timestep_control_cfg.max_change_factor
  min_change_factor = timestep_control_cfg.min_change_factor

  # Clip the target dt to the allowed range.
  allowed_min_dt = jnp.maximum(min_dt, min_change_factor * dt)
  allowed_max_dt = jnp.minimum(max_dt, max_change_factor * dt)
  next_dt = jnp.clip(target_dt, allowed_min_dt, allowed_max_dt)

  return next_dt
