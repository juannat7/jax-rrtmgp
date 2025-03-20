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

"""Step of the Anelastic Navier-Stokes Equations.

- Anelastic equations are used
- Moist thermodynamics: equation of state for water including phase transitions.
- One-moment microphysics is included.

Boundary conditions:
* periodic in x, y for all prognostic variables - u,v,w,theta,q_t,q_r, q_s
* free-slip wall in z.
  * Dirichlet condition w=0 on z boundaries
  * No diffusive flux conditions on z boundaries - for u,v,theta_li,q_t,q_r,q_s.
"""

import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp
from swirl_jatmos import config
from swirl_jatmos import scalars
from swirl_jatmos import sgs
from swirl_jatmos import sponge
from swirl_jatmos import stretched_grid_util
from swirl_jatmos import velocity
from swirl_jatmos.linalg import fast_diagonalization_solver_impl
from swirl_jatmos.linalg import jacobi_solver_impl
from swirl_jatmos.linalg import poisson_solver_interface
from swirl_jatmos.rrtmgp import rrtmgp_common
from swirl_jatmos.thermodynamics import water
from swirl_jatmos.utils import check_states_valid
from swirl_jatmos.utils import utils

Array: TypeAlias = jax.Array
WaterParams: TypeAlias = water.WaterParams
ThermoFields: TypeAlias = water.ThermoFields
BCType: TypeAlias = fast_diagonalization_solver_impl.BCType

DEBUG_CHECK_FOR_NANS = check_states_valid.DEBUG_CHECK_FOR_NANS

# RK3 coefficients
a = [8 / 15, 5 / 12, 3 / 4]
b = [0, -17 / 60, -5 / 12]
c = [8 / 15, 2 / 15, 1 / 3]

# Large value to set for z halos, to smoke out any improper use of the halos.
HALO_VALUE = -4e7


def _enforce_max_diffusivity(
    diffusivity_ccc: Array,
    viscosity_ccc: Array,
    dz_in: float,
    dt: Array,
    sg_map: dict[str, Array],
) -> tuple[Array, Array]:
  """Enforce a maximum diffusivity and viscosity for numerical stability.

  Diffusion is treated explicitly, so the generic timestep restriction is
  dt < dx^2 / diffusivity.  We assume an isotropic diffusivity.  We also assume
  that the grid spacing in z is the important one, so we only use dz here to
  set the maximum diffusivity.

  More generally, we could use all dimensions if desired.  But in standard
  atmospheric simulations, enforcing this on the z grid spacing is sufficient.

  Args:
    diffusivity_ccc: Input scalar diffusivity [m^2/s].
    viscosity_ccc: Input kinematic viscosity [m^2/s].
    dz_in: The grid spacing in the z dimension; only used for uniform z grid.
    dt: The timestep.
    sg_map: The stretched grid map.

  Returns:
    A 2-tuple of the diffusivity and viscosity, both in [m^2/s], clipped to
    maximum values determined by stability limits.
  """
  hz_f_key = stretched_grid_util.hf_key(dim=2)
  dz = sg_map.get(hz_f_key, dz_in)

  coeff = 0.2  # Tolerance factor for the diffusive stability limit.
  max_diffusivity = coeff * dz**2 / dt

  diffusivity_ccc = jnp.clip(diffusivity_ccc, None, max_diffusivity)
  viscosity_ccc = jnp.clip(viscosity_ccc, None, max_diffusivity)
  return diffusivity_ccc, viscosity_ccc


def update_w_dirichlet_bc(w: Array, halo_width: int) -> Array:
  """Update `w` to apply Dirichlet boundary conditions."""
  w = w.at[:, :, halo_width].set(0)
  w = w.at[:, :, -halo_width].set(0)

  # Smoke out if halos are inadvertently used.
  w = w.at[:, :, 0].set(HALO_VALUE)
  return w


def update_bcs_uvw(
    u: Array, v: Array, w: Array, halo_width: int
) -> tuple[Array, Array, Array]:
  w = update_w_dirichlet_bc(w, halo_width)

  # Smoke out if halos are inadvertently used.  Can delete this later.
  u = update_z_halos_bcs(u, halo_width)
  v = update_z_halos_bcs(v, halo_width)
  return u, v, w


def update_halos_bcs(
    theta_li: Array,
    q_t: Array,
    q_r: Array,
    q_s: Array,
    u: Array,
    v: Array,
    w: Array,
    halo_width: int,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
  """Update z BCs."""
  u, v, w = update_bcs_uvw(u, v, w, halo_width)

  # Smoke out if halos are inadvertently used.  Can delete this later.
  theta_li = update_z_halos_bcs(theta_li, halo_width)
  q_t = update_z_halos_bcs(q_t, halo_width)
  q_r = update_z_halos_bcs(q_r, halo_width)
  q_s = update_z_halos_bcs(q_s, halo_width)
  return theta_li, q_t, q_r, q_s, u, v, w


def update_z_halos_bcs(f: Array, halo_width: int) -> Array:
  """Z halos should not matter.  Set to arbitrary value."""
  # Update BCs in the z dimension, for Neumann BCs. For arrays on z nodes.
  # Smoke out if halos are inadvertently used by setting to large value.
  # halo_width is assumed to be 1.
  del halo_width
  f = f.at[:, :, 0].set(HALO_VALUE)
  f = f.at[:, :, -1].set(HALO_VALUE)
  return f


def update_halo_fn_for_pressure_poisson_eqn(
    dp_ccc: Array, halo_width: int
) -> Array:
  """Update halos of `dp` for the pressure Poisson equation.

  Assume periodicity in x,y, and Neumann BCs in z.  That is, ∂p/∂z = 0 is
  applied on the z boundaries, which means the values of p in the halo nodes are
  set equal to the values in the first interior node.
  """
  # Update BCs in the z dimension, for Neumann BCs.
  face0_idx = halo_width - 1
  face1_idx = -halo_width

  dp_ccc = dp_ccc.at[:, :, face0_idx].set(dp_ccc[:, :, face0_idx + 1])
  dp_ccc = dp_ccc.at[:, :, face1_idx].set(dp_ccc[:, :, face1_idx - 1])
  return dp_ccc


def step(
    states: dict[str, Array],
    poisson_solver: poisson_solver_interface.PoissonSolver,
    cfg: config.Config,
) -> tuple[dict[str, Array], dict[str, Array]]:
  """Step forward by dt."""
  u_fcc = states['u']

  f_states_prev = dict(
      f_theta_li_ccc=jnp.zeros_like(u_fcc),
      f_q_t_ccc=jnp.zeros_like(u_fcc),
      f_q_r_ccc=jnp.zeros_like(u_fcc),
      f_q_s_ccc=jnp.zeros_like(u_fcc),
      f_u_fcc=jnp.zeros_like(u_fcc),
      f_v_cfc=jnp.zeros_like(u_fcc),
      f_w_ccf=jnp.zeros_like(u_fcc),
  )

  print('tracing: 3 substeps of the RK3 step.')
  substep = functools.partial(_substep, poisson_solver=poisson_solver, cfg=cfg)

  # Compute 1st stage of the RK3 step.
  states, f_states_prev, aux_output = substep(states, f_states_prev, k=0)  # pylint: disable=unused-variable

  if not DEBUG_CHECK_FOR_NANS.value:
    # Compute 2nd and 3rd stages without any checks on NaNs.
    states, f_states_prev, aux_output = substep(states, f_states_prev, k=1)  # pylint: disable=unused-variable
    states, _, aux_output = substep(states, f_states_prev, k=2)
  else:
    # Compute 2nd stage only if states are valid.
    states, f_states_prev, aux_output = jax.lax.cond(
        pred=check_states_valid.check_states_are_finite(states),
        true_fun=lambda: substep(states, f_states_prev, k=1),
        false_fun=lambda: (states, f_states_prev, aux_output),
    )
    # Compute 3rd stage only if states are valid.
    states, _, aux_output = jax.lax.cond(
        pred=check_states_valid.check_states_are_finite(states),
        true_fun=lambda: substep(states, f_states_prev, k=2),
        false_fun=lambda: (states, f_states_prev, aux_output),
    )

  states_new = dict(states)
  states_new['step_id'] = states['step_id'] + 1
  states_new['t_ns'] = states['t_ns'] + states['dt_ns']
  return states_new, aux_output


def _substep(
    states: dict[str, Array],
    f_prev_states: dict[str, Array],
    poisson_solver: poisson_solver_interface.PoissonSolver,
    cfg: config.Config,
    k: int,
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array]]:
  """Compute one substep (one stage) of the RK3 step."""
  aux_output: dict[str, Array] = {}
  halo_width = cfg.halo_width
  assert halo_width == 1, 'Only halo_width=1 is supported for parallel sim.'
  wp = cfg.wp
  deriv_lib = cfg.deriv_lib

  # Reference states: rho_ref and p_ref.
  rho_xxc = states['rho_xxc']
  rho_xxf = states['rho_xxf']
  p_ref_xxc = states['p_ref_xxc']
  theta_li_0_ccc = states['theta_li_0']

  # Convert dt to seconds for use in timestepping the equations of motion.
  dt = (1e-9 * states['dt_ns']).astype(rho_xxc.dtype)

  # Pull out 2D surface conditions for surface temperature and humidity, if
  # present (they are used when the Monin-Obukhov BC is specified).
  sfc_temperature = states.get('sfc_temperature_2d_xy', None)
  sfc_q_t = states.get('sfc_q_t_2d_xy', None)

  # Pull out the radiative heating source, if present.  This term is not
  # recomputed inside the RK3 substeps, so it is constant throughout the step.
  radiation_source = states.get(rrtmgp_common.KEY_APPLIED_RADIATION, None)

  sg_map = stretched_grid_util.sg_map_from_states(states)

  dtheta_li_ccc = states['dtheta_li']
  u_fcc, v_cfc, w_ccf = states['u'], states['v'], states['w']
  q_t_ccc, q_r_ccc, q_s_ccc = states['q_t'], states['q_r'], states['q_s']
  p_ccc = states['p']  # Hydrodynamic pressure.

  f_theta_li_ccc_prev = f_prev_states['f_theta_li_ccc']
  f_q_t_ccc_prev = f_prev_states['f_q_t_ccc']
  f_q_r_ccc_prev = f_prev_states['f_q_r_ccc']
  f_q_s_ccc_prev = f_prev_states['f_q_s_ccc']
  f_u_fcc_prev = f_prev_states['f_u_fcc']
  f_v_cfc_prev = f_prev_states['f_v_cfc']
  f_w_ccf_prev = f_prev_states['f_w_ccf']

  # Get the base kinematic viscosity and scalar diffusivity from the config.
  const_diffusivity_ccc = cfg.diffusivity
  const_viscosity_ccc = cfg.viscosity

  # Compute total theta_li as background + perturbation.
  theta_li_ccc = theta_li_0_ccc + dtheta_li_ccc

  # Compute the strain rate tensor and the turbulent viscosity.
  pr_t = 0.33  # Turbulent Prandtl number.

  rho_thermal_initial_guess = rho_xxc * jnp.ones_like(theta_li_ccc)
  thermo_fields = water.compute_thermodynamic_fields_from_prognostic_fields(
      theta_li_ccc, q_t_ccc, p_ref_xxc, rho_thermal_initial_guess, wp
  )

  strain_rate_tensor = utils.compute_strain_rate_tensor(
      u_fcc, v_cfc, w_ccf, deriv_lib, sg_map, cfg.z_c, cfg.z_f, cfg.z_bcs
  )
  if cfg.use_sgs:
    # Use the unsaturated version of the buoyancy frequency.  Can also use the
    # version assuming saturated air.
    eddy_viscosity_ccc = sgs.smagorinsky_lilly_nu_t(
        strain_rate_tensor,
        theta_li_ccc,  # Use theta_li in place of theta here... temporary.
        pr_t,
        sg_map,
        cfg.z_c,
        deriv_lib,
        halo_width,
    )
  else:
    # eddy_viscosity_ccc = 0
    eddy_viscosity_ccc = jnp.zeros_like(theta_li_ccc)
  viscosity_ccc = const_viscosity_ccc + eddy_viscosity_ccc
  diffusivity_ccc = const_diffusivity_ccc + eddy_viscosity_ccc / pr_t
  if cfg.enforce_max_diffusivity:
    diffusivity_ccc, viscosity_ccc = _enforce_max_diffusivity(
        diffusivity_ccc, viscosity_ccc, cfg.grid_spacings[2], dt, sg_map
    )

  # Compute terms on RHS treated explicitly (all except pressure gradient).
  f_theta_li_ccc, f_q_t_ccc, f_q_r_ccc, f_q_s_ccc, aux_out_scalars = (
      scalars.rhs_explicit_fn(
          rho_xxc,
          rho_xxf,
          theta_li_ccc,
          q_t_ccc,
          q_r_ccc,
          q_s_ccc,
          u_fcc,
          v_cfc,
          w_ccf,
          p_ref_xxc,
          diffusivity_ccc,
          thermo_fields,
          dt,
          deriv_lib,
          sg_map,
          cfg.convection_cfg,
          cfg.microphysics_cfg,
          cfg.include_qt_sedimentation,
          halo_width,
          cfg.z_bcs,
          sfc_temperature,
          sfc_q_t,
          cfg.radiative_transfer_cfg,
          radiation_source,
      )
  )
  aux_output |= aux_out_scalars

  f_u_fcc, f_v_cfc, f_w_ccf = velocity.u_v_w_rhs_explicit_fn(
      rho_xxc,
      rho_xxf,
      u_fcc,
      v_cfc,
      w_ccf,
      q_r_ccc,
      q_s_ccc,
      viscosity_ccc,
      thermo_fields.rho_thermal,
      strain_rate_tensor,
      deriv_lib,
      sg_map,
      cfg.convection_cfg,
      cfg.include_buoyancy,
      halo_width,
      cfg.z_bcs,
  )

  # RK3 stage for scalars d_theta_li, q_t, q_r, q_s.
  dtheta_li_ccc = dtheta_li_ccc + dt * (
      a[k] * f_theta_li_ccc + b[k] * f_theta_li_ccc_prev
  )
  q_t_ccc = q_t_ccc + dt * (a[k] * f_q_t_ccc + b[k] * f_q_t_ccc_prev)
  q_r_ccc = q_r_ccc + dt * (a[k] * f_q_r_ccc + b[k] * f_q_r_ccc_prev)
  q_s_ccc = q_s_ccc + dt * (a[k] * f_q_s_ccc + b[k] * f_q_s_ccc_prev)

  # RK3 stage for velocity, with a predictor step, then a projection step.
  # Predictor step.
  dpdx_fcc = deriv_lib.dx_c_to_f(p_ccc, sg_map)
  dpdy_cfc = deriv_lib.dy_c_to_f(p_ccc, sg_map)
  dpdz_ccf = deriv_lib.dz_c_to_f(p_ccc, sg_map)

  u_fcc_hat = u_fcc + dt * (
      a[k] * f_u_fcc + b[k] * f_u_fcc_prev - c[k] * dpdx_fcc
  )
  v_cfc_hat = v_cfc + dt * (
      a[k] * f_v_cfc + b[k] * f_v_cfc_prev - c[k] * dpdy_cfc
  )
  w_ccf_hat = w_ccf + dt * (
      a[k] * f_w_ccf + b[k] * f_w_ccf_prev - c[k] * dpdz_ccf
  )

  if cfg.sponge_cfg is not None:
    # Apply implicit sponge (not using the RK3 coeffs here because we do not
    # worry about high-order temporal accuracy for the sponge.)
    domain_height = cfg.domain_z[1]
    w_ccf_hat = sponge.apply_sponge(
        w_ccf_hat, states['z_f'], domain_height, cfg.sponge_cfg
    )
    v_cfc_hat = sponge.apply_sponge(
        v_cfc_hat, states['z_c'], domain_height, cfg.sponge_cfg
    )
    u_fcc_hat = sponge.apply_sponge(
        u_fcc_hat, states['z_c'], domain_height, cfg.sponge_cfg
    )

  u_fcc_hat, v_cfc_hat, w_ccf_hat = update_bcs_uvw(
      u_fcc_hat, v_cfc_hat, w_ccf_hat, halo_width
  )

  if cfg.solve_pressure_only_on_last_rk3_stage and k != 2:
    # Under the Le & Moin procedure, do not solve the pressure Poisson
    # equation in the first 2 RK3 stages, and just take the predictor
    # velocities as the velocities in the next stage.
    u_fcc = u_fcc_hat
    v_cfc = v_cfc_hat
    w_ccf = w_ccf_hat
  else:
    # In the general RK3 case, or for the 3rd stage under the Le & Moin
    # procedure, solve the pressure Poisson equation.

    # Compute the pressure Poisson eqn's RHS.  First compute ∇·(rho*uhat).
    div_rho_u_hat_ccc = deriv_lib.divergence_ccc(
        rho_xxc * u_fcc_hat, rho_xxc * v_cfc_hat, rho_xxf * w_ccf_hat, sg_map
    )

    # If solving the pressure equation only on the last RK3 stage, then adjust
    # the c_k coefficient modifying the pressure poisson equation and
    # correction.
    c_k = 1.0 if cfg.solve_pressure_only_on_last_rk3_stage else c[k]
    rhs_ccc = div_rho_u_hat_ccc / (dt * c_k)

    # Solve Poisson equation for delta p_k
    if isinstance(poisson_solver, jacobi_solver_impl.BaseJacobiSolver):
      halo_update_fn = functools.partial(
          update_halo_fn_for_pressure_poisson_eqn, halo_width=halo_width
      )
    else:  # Fast Diagonalization Solver
      # Unlike the Jacobi solver, there is no halo update inside the fast
      # diagonalization solver. The halos in z are not actually used, though.
      halo_update_fn = lambda dp_ccc: dp_ccc

    dp_ccc = poisson_solver_interface.solve(
        poisson_solver, rhs_ccc, rho_xxc, rho_xxf, sg_map, halo_update_fn
    )

    # Correction step (projection to divergence-free mass flux).
    ddpdx_fcc = deriv_lib.dx_c_to_f(dp_ccc, sg_map)
    ddpdy_cfc = deriv_lib.dy_c_to_f(dp_ccc, sg_map)
    ddpdz_ccf = deriv_lib.dz_c_to_f(dp_ccc, sg_map)
    u_fcc = u_fcc_hat - dt * c_k * ddpdx_fcc
    v_cfc = v_cfc_hat - dt * c_k * ddpdy_cfc
    w_ccf = w_ccf_hat - dt * c_k * ddpdz_ccf
    p_ccc = p_ccc + dp_ccc

  f_prev_states_new = {}
  f_prev_states_new['f_theta_li_ccc'] = f_theta_li_ccc
  f_prev_states_new['f_q_t_ccc'] = f_q_t_ccc
  f_prev_states_new['f_q_r_ccc'] = f_q_r_ccc
  f_prev_states_new['f_q_s_ccc'] = f_q_s_ccc
  f_prev_states_new['f_u_fcc'] = f_u_fcc
  f_prev_states_new['f_v_cfc'] = f_v_cfc
  f_prev_states_new['f_w_ccf'] = f_w_ccf

  dtheta_li_ccc, q_t_ccc, q_r_ccc, q_s_ccc, u_fcc, v_cfc, w_ccf = (
      update_halos_bcs(
          dtheta_li_ccc,
          q_t_ccc,
          q_r_ccc,
          q_s_ccc,
          u_fcc,
          v_cfc,
          w_ccf,
          halo_width,
      )
  )
  states_new = dict(states)
  states_new['dtheta_li'] = dtheta_li_ccc
  states_new['q_t'] = q_t_ccc
  states_new['q_r'] = q_r_ccc
  states_new['q_s'] = q_s_ccc
  states_new['u'] = u_fcc
  states_new['v'] = v_cfc
  states_new['w'] = w_ccf
  states_new['p'] = p_ccc

  # Add aux_output
  if 'eddy_viscosity' in cfg.aux_output_fields:
    aux_output['eddy_viscosity'] = eddy_viscosity_ccc
  if 'q_c' in cfg.aux_output_fields:
    aux_output['q_c'] = thermo_fields.q_c  # Note, q_c = q_liq + q_ice.
  if 'T' in cfg.aux_output_fields:
    aux_output['T'] = thermo_fields.T
  if 'q_liq' in cfg.aux_output_fields:
    aux_output['q_liq'] = thermo_fields.q_liq
  if 'q_ice' in cfg.aux_output_fields:
    aux_output['q_ice'] = thermo_fields.q_ice
  if 'T_1d_z' in cfg.aux_output_fields:
    aux_output['T_1d_z'] = jnp.mean(thermo_fields.T, axis=(0, 1))

  aux_output['q_v'] = thermo_fields.q_v
  aux_output['q_v_sat'] = thermo_fields.q_v_sat
  return states_new, f_prev_states_new, aux_output
