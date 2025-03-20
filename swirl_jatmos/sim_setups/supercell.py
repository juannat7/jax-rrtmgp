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

"""Supercell simulation setup."""

import dataclasses
import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from swirl_jatmos import config
from swirl_jatmos import interpolation
from swirl_jatmos import sim_initializer
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array
WaterParams: TypeAlias = water.WaterParams


@dataclasses.dataclass(frozen=True)
class SupercellSettings:
  """Defines parameters to set up the Supercell simulation."""

  # The minimum initial velocity (m/s)
  u_min: float = 7.0
  # The maximum initial velocity (m/s)
  u_max: float = 31.0
  # z1 is the top of the initial rotating wind ground layer (m).
  # Between 0 and z1, the wind in the x-y plane is
  # u_min * [1-cos(angle), sin(angle)]^T,
  # where angle = pi/2 * (z/z1).
  z1: float = 2000.0
  # z2 is the top of the initial linearly increasing wind layer (m).
  # between z1 and z2, the wind in the x-y plane is
  # [u_min + (z-z1)*(u_max-u_min)/(z2-z1), u_min]^T
  # Above z2, the wind in the x-y plane is is constant with a value of
  # [u_max, u_min].
  z2: float = 6000.0
  # Tropopause height (m).
  z_t: float = 1.2e4
  # The potential temperature at the sea surface (K).
  theta_s: float = 300.0
  # The potential temperature at the tropopause (K).
  theta_t: float = 343.0
  # The vapor mass fraction at the sea surface.
  q_vs: float = 0.014
  # The temperature at tropopause (K).
  t_t: float = 213.0
  # The maximum number of iterations used to determine the total humidity.
  q_t_max_iter: int = 10
  # The perturbation of potential temperature in the warm bubble [K].
  theta_pert: float = 1.0
  # The perturbation of rain water mass fraction in the warm bubble [K].
  q_r_pert: float = 0.0
  # The height of the center of the warm bubble [m].
  z_bubble: float = 1.4e3
  # The horizontal radius of the warm bubble [m].
  rh_bubble: float = 1e4
  # The vertical radius of the warm bubble [m].
  rv_bubble: float = 1.4e3
  # The velocity shift so that the updraft stays at the center of the domain.
  # The default values are derived from CM1.
  u_shift: float = 12.466293
  v_shift: float = 2.31388


sim_params = SupercellSettings()


def u_initial_condition(z: Array) -> Array:
  """Initial condition for u velocity component."""
  u_min = sim_params.u_min
  u_max = sim_params.u_max
  z1 = sim_params.z1
  z2 = sim_params.z2
  u_shift = sim_params.u_shift

  # Angle changes linearly from 0 to 90 degrees as z goes from zero to z1.
  angle = 0.5 * np.pi * (z / z1)
  u_below_z1 = u_min * (1.0 - jnp.cos(angle))
  u_z1_to_z2 = u_min + (z - z1) * ((u_max - u_min) / (z2 - z1))
  u_above_z2 = u_max * jnp.ones_like(z)
  u = jnp.where(
      z < z1,
      u_below_z1,
      jnp.where(z < z2, u_z1_to_z2, u_above_z2),
  )

  # Shift the velocity.
  u = u - u_shift
  return u


def v_initial_condition(z: Array) -> Array:
  """Initial condition for v velocity component; Careful to make z 3d."""
  u_min = sim_params.u_min
  z1 = sim_params.z1
  v_shift = sim_params.v_shift

  # Angle changes linearly from 0 to 90 degrees as z goes from zero to z1.
  angle = 0.5 * np.pi * (z / z1)
  v_below_z1 = u_min * jnp.sin(angle)
  v_above_z1 = u_min * jnp.ones_like(z)
  v = jnp.where(z < z1, v_below_z1, v_above_z1)

  # Shift the velocity.
  v = v - v_shift
  return v


def theta_initial_condition(z: Array) -> Array:
  """Initial condition for potential temperature (excluding perturbation).

  Args:
    z: The vertical coordinates.

  Returns:
    The potential temperature profile prior to perturbation.
  """
  theta_s = sim_params.theta_s
  theta_t = sim_params.theta_t
  z_t = sim_params.z_t
  t_t = sim_params.t_t
  g = 9.81
  cp_d = water.CP_D

  # Ensure z is non-negative; important for taking the 1.25 power.
  # Values in halos (with negative z) should not matter.
  z_clip = jnp.clip(z, 0.0, None)
  theta_below_z_t = theta_s + (theta_t - theta_s) * (z_clip / z_t) ** 1.25
  theta_above_z_t = theta_t * jnp.exp(g * (z - z_t) / (t_t * cp_d))
  theta = jnp.where(z < z_t, theta_below_z_t, theta_above_z_t)
  return theta


def q_t_initial_condition(
    z: Array, theta: Array, p_ref: Array, wp: WaterParams
) -> Array:
  """Initial condition for total humidity."""
  q_t_max_iter = sim_params.q_t_max_iter
  z_t = sim_params.z_t
  cp_d = water.CP_D
  r_d = water.R_D
  r_v = wp.r_v
  cp_v = wp.cp_v

  zc = jnp.maximum(z, 0.0)

  # Start with an assumed value for the relative humidity profile h.
  h_below_z_t = 1 - 0.75 * (zc / z_t) ** (1.25)
  h_above_z_t = 0.25 * jnp.ones_like(z)
  h = jnp.where(z < z_t, h_below_z_t, h_above_z_t)

  def q_t_iteration(q_t: Array) -> Array:
    """One iteration of the q_t iteration procedure."""
    # Compute Rm, cp_m, assuming unsaturated air (no condensate).
    r_m = (1 - q_t) * r_d + q_t * r_v
    cp_m = (1 - q_t) * cp_d + q_t * cp_v

    # Compute the temperature from the potential temperature.
    exner = (p_ref / wp.exner_reference_pressure) ** (r_m / cp_m)
    T = exner * theta  # pylint: disable=invalid-name

    # Compute saturation vapor humidity.
    p_v_sat = water.saturation_vapor_pressure(T, wp)
    # Formula for q_v_sat that doesn't depend on density.
    q_v_sat = r_d / r_v * (1 - q_t) * p_v_sat / (p_ref - p_v_sat)
    q_t_next = h * q_v_sat
    return q_t_next

  q_t = jnp.zeros_like(z)
  for _ in range(q_t_max_iter):
    q_t = q_t_iteration(q_t)

  # Enforce maximum value of q_t.
  q_t = jnp.clip(q_t, 0.0, 0.014)
  return q_t


def warm_bubble_potential_temperature(
    x: Array, y: Array, z: Array, x0: float, y0: float
) -> Array:
  """Create a perturbation in potential temperature for a warm bubble."""
  radius_horiz = sim_params.rh_bubble
  radius_vert = sim_params.rv_bubble
  z0 = sim_params.z_bubble
  theta_pert = sim_params.theta_pert

  normalized_distance_sq = (
      ((x - x0) / radius_horiz) ** 2
      + ((y - y0) / radius_horiz) ** 2
      + ((z - z0) / radius_vert) ** 2
  )
  normalized_distance = jnp.sqrt(normalized_distance_sq)

  theta_bubble_perturbation = jnp.where(
      normalized_distance < 1.0,
      theta_pert * jnp.cos(0.5 * np.pi * normalized_distance) ** 2,
      jnp.zeros_like(normalized_distance),
  )
  return theta_bubble_perturbation


def p_ref_fn(z: Array, wp: WaterParams) -> Array:
  """Computes the reference pressure."""
  theta_s = sim_params.theta_s
  theta_t = sim_params.theta_t
  z_t = sim_params.z_t
  t_t = sim_params.t_t
  g = 9.81
  r_d = water.R_D
  cp_d = water.CP_D
  p0 = wp.exner_reference_pressure

  def integral_below_zt(z: Array | float) -> Array:
    theta_diff = theta_t - theta_s
    return z_t / theta_diff * jnp.log(theta_s + theta_diff * z / z_t)

  theta_inv_integral_below_z_t = integral_below_zt(z) - integral_below_zt(0.0)

  def integral_above_zt(z: Array | float) -> Array:
    return (
        -(t_t * cp_d) / (theta_t * g) * jnp.exp(-g * (z - z_t) / (t_t * cp_d))
    )

  theta_inv_integral_above_z_t = (
      integral_above_zt(z)
      - integral_above_zt(z_t)
      + integral_below_zt(z_t)
      - integral_below_zt(0.0)
  )

  theta_inv_integral = jnp.where(
      z < z_t, theta_inv_integral_below_z_t, theta_inv_integral_above_z_t
  )

  p_ref = p0 * (1.0 - g / cp_d * theta_inv_integral) ** (cp_d / r_d)
  return p_ref


def rho_ref_fn(
    theta: Array, q_t: Array, p_ref: Array, wp: WaterParams
) -> Array:
  """Compute rho_ref from theta, q_t, p_ref."""
  r_d = water.R_D
  cp_d = water.CP_D
  r_v = wp.r_v
  cp_v = wp.cp_v

  r_m = (1 - q_t) * r_d + q_t * r_v
  cp_m = (1 - q_t) * cp_d + q_t * cp_v

  # Compute the temperature from the potential temperature.
  exner = (p_ref / wp.exner_reference_pressure) ** (r_m / cp_m)
  T = exner * theta  # pylint: disable=invalid-name

  rho_ref = p_ref / (r_m * T)
  return rho_ref


def thermodynamic_initial_condition(
    z: Array, wp: WaterParams
) -> tuple[Array, Array, Array, Array]:
  """Get thermodynamic initial conditions for theta_li, q_t, p_ref, rho_ref."""
  p_ref = p_ref_fn(z, wp)
  # theta = theta_li, assuming no condensate.
  theta_li = theta_initial_condition(z)

  q_t = q_t_initial_condition(z, theta_li, p_ref, wp)

  rho_ref = rho_ref_fn(theta_li, q_t, p_ref, wp)
  return theta_li, q_t, p_ref, rho_ref


def init_fn(cfg: config.Config) -> dict[str, Array]:
  """Create sharded initial conditions."""
  grid_map_sharded = sim_initializer.initialize_grids(cfg)

  # Extract required grid variables.
  x_c = grid_map_sharded['x_c']
  y_c = grid_map_sharded['y_c']
  z_c = grid_map_sharded['z_c']
  nx, ny, _ = x_c.shape[0], y_c.shape[1], z_c.shape[2]
  wp = cfg.wp
  lx, ly = cfg.domain_x[1], cfg.domain_y[1]

  theta_li_init_c, q_t_c, p_ref_xxc, rho_ref_xxc = (
      thermodynamic_initial_condition(z_c, wp)
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

  u_c = u_initial_condition(z_c)
  v_c = v_initial_condition(z_c)

  q_t_ccc = jnp.tile(q_t_c, reps=(nx, ny, 1))
  u_fcc = jnp.tile(u_c, reps=(nx, ny, 1))
  v_cfc = jnp.tile(v_c, reps=(nx, ny, 1))

  q_r_ccc = jnp.zeros_like(q_t_ccc)
  q_s_ccc = jnp.zeros_like(q_t_ccc)
  w_ccf = jnp.zeros_like(u_fcc)
  p_ccc = jnp.zeros_like(u_fcc)

  shard_3d = functools.partial(sim_initializer.shard_3d, cfg=cfg)

  q_t_ccc = shard_3d(q_t_ccc)
  u_fcc = shard_3d(u_fcc)
  v_cfc = shard_3d(v_cfc)
  q_r_ccc = shard_3d(q_r_ccc)
  q_s_ccc = shard_3d(q_s_ccc)
  w_ccf = shard_3d(w_ccf)
  p_ccc = shard_3d(p_ccc)

  # Initialize θ perturbation.
  x0 = lx / 2
  y0 = ly / 2
  theta_pert_ccc = warm_bubble_potential_temperature(x_c, y_c, z_c, x0, y0)
  theta_pert_ccc = shard_3d(theta_pert_ccc)

  theta_li_0_c = theta_li_init_c
  dtheta_li_ccc = theta_pert_ccc

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
  return states
