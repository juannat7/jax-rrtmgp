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

"""Setup for the buoyant bubble simulation."""

import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import config
from swirl_jatmos import constants
from swirl_jatmos import convection_config
from swirl_jatmos import interpolation
from swirl_jatmos import sim_initializer
from swirl_jatmos import timestep_control_config
from swirl_jatmos.boundary_conditions import boundary_conditions
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = jax.typing.ArrayLike

cfg_ext = config.ConfigExternal(
    cx=1,
    cy=1,
    cz=1,
    nx=512,
    ny=2,
    nz=256,
    domain_x=(0, 40e3),
    domain_y=(0, 157),
    domain_z=(0, 15e3),
    halo_width=1,  # only used in z dimension.
    dt=1.0,
    timestep_control_cfg=timestep_control_config.TimestepControlConfig(
        disable_adaptive_timestep=True
    ),
    wp=water.WaterParams(),
    convection_cfg=convection_config.ConvectionConfig(
        momentum_scheme='weno5_z',
        theta_li_scheme='weno5_z',
    ),
    use_sgs=True,
    uniform_y_2d=True,
    z_bcs=boundary_conditions.ZBoundaryConditions(
        bottom=boundary_conditions.ZBC(bc_type='no_flux'),
        top=boundary_conditions.ZBC(bc_type='no_flux'),
    ),
    solve_pressure_only_on_last_rk3_stage=False,
    poisson_solver_type=config.PoissonSolverType.FAST_DIAGONALIZATION,
    viscosity=1e-3,
    diffusivity=1e-3,
    disable_checkpointing=True,
)


def warm_bubble_potential_temperature(
    x: ArrayLike, z: ArrayLike, lx: float
) -> Array:
  """Create a perturbation in potential temperature for a warm bubble."""
  # Buoyant bubble settings
  theta_pert = 10.0  # perturbation in potential temperature for bubble [K]
  bubble_height = 1.4e3  # Height of the center of the warm bubble [m]
  bubble_x_radius = 10e3  # Horizontal radius of the warm bubble [m]
  bubble_z_radius = 1.4e3  # Vertical radius of the warm bubble [m]

  normalized_distance_sq = ((x - lx / 2) / bubble_x_radius) ** 2 + (
      (z - bubble_height) / bubble_z_radius
  ) ** 2
  normalized_distance = jnp.sqrt(normalized_distance_sq)

  theta_bubble_perturbation = jnp.where(
      normalized_distance < 1.0,
      theta_pert * jnp.cos(0.5 * np.pi * normalized_distance) ** 2,
      jnp.zeros_like(normalized_distance),
  )
  return theta_bubble_perturbation


def initial_profiles(z: ArrayLike) -> tuple[Array, Array, Array, Array]:
  """Provides initial profiles for p_ref, T, rho_ref, theta_init.

  Args:
    z: The vertical coordinate.

  Returns:
    Tuple of reference and initial conditions for p_ref, rho_ref, theta_li_init,
    and q_t, all returned with the same shape as the input `z`.
  """
  z = jnp.asarray(z)
  wp = water.WaterParams()
  g, r_d, cp_d = constants.G, constants.R_D, constants.CP_D
  p0 = wp.exner_reference_pressure

  T_s = 300.0  # surface temperature, K  # pylint: disable=invalid-name
  gamma = 6.7e-3  # temperature lapse rate, K/m
  T = T_s - gamma * z  # pylint: disable=invalid-name

  # Initialize the hydrostatic reference pressure
  p_ref = p0 * (1 - gamma * z / T_s) ** (g / (r_d * gamma))
  rho_ref = p_ref / (r_d * T)

  # no moisture
  q_t = jnp.zeros_like(z)

  # Initialize theta_li (assuming no condensate, so theta = theta_li)
  rm = (1 - q_t) * r_d + q_t * wp.r_v
  cpm = (1 - q_t) * cp_d + q_t * wp.cp_v
  exner = (p_ref / wp.exner_reference_pressure) ** (rm / cpm)
  # Compute the initial (liquid-ice) potential temperature.
  theta_li = T / exner

  return p_ref, rho_ref, theta_li, q_t


def init_fn(cfg: config.Config) -> dict[str, Array]:
  """Create sharded initial conditions for the buoyant bubble simulation."""
  grid_map_sharded = sim_initializer.initialize_grids(cfg)

  # Extract required grid variables.
  x_c = grid_map_sharded['x_c']
  y_c = grid_map_sharded['y_c']
  z_c = grid_map_sharded['z_c']
  nx, ny, nz = x_c.shape[0], y_c.shape[1], z_c.shape[2]
  lx = cfg.domain_x[1]
  # Initialize the anelastic reference states, p_ref, ρ_ref
  p_ref_xxc, rho_ref_xxc, theta_li_init_c, q_t_c = initial_profiles(z_c)
  rho_ref_xxf = interpolation.z_c_to_f(rho_ref_xxc)
  q_t_ccc = jnp.tile(q_t_c, reps=(nx, ny, 1))

  q_r_ccc = jnp.zeros_like(q_t_ccc)
  q_s_ccc = jnp.zeros_like(q_t_ccc)
  p_ccc = jnp.zeros_like(q_t_ccc)

  # Initialize 3D u, v, w
  u_fcc = jnp.zeros_like(q_t_ccc)
  v_cfc = jnp.zeros_like(u_fcc)
  w_ccf = jnp.zeros_like(u_fcc)

  shard_3d = functools.partial(sim_initializer.shard_3d, cfg=cfg)
  q_t_ccc = shard_3d(q_t_ccc)
  u_fcc = shard_3d(u_fcc)
  v_cfc = shard_3d(v_cfc)
  q_r_ccc = shard_3d(q_r_ccc)
  q_s_ccc = shard_3d(q_s_ccc)
  w_ccf = shard_3d(w_ccf)
  p_ccc = shard_3d(p_ccc)

  # Initialize potential temperature perturbation to θ.
  xx_nodes_3d = jnp.tile(x_c, reps=(1, ny, nz))
  zz_nodes_3d = jnp.tile(z_c, reps=(nx, ny, 1))
  theta_pert_ccc = warm_bubble_potential_temperature(
      xx_nodes_3d, zz_nodes_3d, lx
  )  # Shape is (nx, ny, nz)
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
