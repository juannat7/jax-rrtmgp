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

"""Perform a step of the temperature equation ∂T/∂t = R.

Where R is the heating due to radiation.

* Assume a constant relative humidity h < 1 (no condensate; clear sky).
* Assume a constant surface temperature.
"""

import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp
from swirl_jatmos import constants
from swirl_jatmos.rrtmgp import rrtmgp
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array
WaterParams: TypeAlias = water.WaterParams

# Large value to set for z halos, to smoke out any improper use of the halos.
HALO_VALUE = -4e7


def update_z_halos_bcs(f: Array, halo_width: int) -> Array:
  """Z halos should not matter.  Set to arbitrary value."""
  # Update BCs in the z dimension, for Neumann BCs. For arrays on z nodes.
  # Smoke out if halos are inadvertently used by setting to large value.
  # halo_width is assumed to be 1.
  del halo_width
  f = f.at[:, :, 0].set(HALO_VALUE)
  f = f.at[:, :, -1].set(HALO_VALUE)
  return f


def compute_q_t(
    T: Array,  # pylint: disable=invalid-name
    p_ref_xxc: Array,
    wp: WaterParams,
    relative_humidity: float,
) -> Array:
  """Compute the vapor specific humidity given constant relative humidity."""
  p_v_sat = water.saturation_vapor_pressure(T, wp)
  # Using the formula for q_v_sat that doesn't depend on thermodynamic density.
  r_d, r_v = constants.R_D, constants.R_V
  alpha = relative_humidity * r_d / r_v * p_v_sat / (p_ref_xxc - p_v_sat)
  q_t = alpha / (1 + alpha)
  return q_t


# pylint: disable=invalid-name
def T_rhs(
    T: Array,
    q_t: Array,
    rho_xxc: Array,
    p_ref_xxc: Array,
    sfc_temperature: Array,
    sg_map: dict[str, Array],
    rrtmgp_: rrtmgp.RRTMGP,
    use_scan: bool = True,
) -> dict[str, Array]:
  """Compute the right-hand side of the temperature equation."""
  q_liq = jnp.zeros_like(T)
  q_ice = jnp.zeros_like(T)
  q_c = jnp.zeros_like(T)

  rad_states = rrtmgp_.compute_heating_rate(
      rho_xxc,
      q_t,
      q_liq,
      q_ice,
      q_c,
      T,
      sfc_temperature,
      p_ref_xxc,
      sg_map,
      use_scan=use_scan,
  )
  return rad_states
# pylint: enable=invalid-name


def step(
    T: Array,  # pylint: disable=invalid-name
    rho_xxc: Array,
    p_ref_xxc: Array,
    sfc_temperature: Array,
    sg_map: dict[str, Array],
    relative_humidity: float,
    wp: WaterParams,
    rrtmgp_: rrtmgp.RRTMGP,
    dt: float,
    use_scan: bool = True,
) -> tuple[Array, dict[str, Array]]:
  """Step forward by dt using RK2 midpoint method."""
  # pylint: disable=invalid-name
  # Compute the vapor specific humidity, q_t = q_v, from the relative humidity.
  q_t0 = compute_q_t(T, p_ref_xxc, wp, relative_humidity)

  T_rhs_func = functools.partial(
      T_rhs,
      rho_xxc=rho_xxc,
      p_ref_xxc=p_ref_xxc,
      sfc_temperature=sfc_temperature,
      sg_map=sg_map,
      rrtmgp_=rrtmgp_,
      use_scan=use_scan,
  )

  # Stage 1
  rad_states1 = T_rhs_func(T, q_t0)
  rad_heat_src1 = rad_states1['rad_heat_src']

  f1 = rad_heat_src1  # dT/dt due to radiation, in K/s.
  T1 = T + dt / 2 * f1  # Estimate T halfway through the timestep.

  # Stage 2
  q_t1 = compute_q_t(T1, p_ref_xxc, wp, relative_humidity)
  rad_states_2 = T_rhs_func(T1, q_t1)
  rad_heat_src2 = rad_states_2['rad_heat_src']

  f2 = rad_heat_src2  # dT/dt due to radiation, in K/s.
  T2 = T + dt * f2

  # Contains `rad_heat_src`, and optionally, `rad_heat_lw_3d` and
  # `rad_heat_sw_3d` if `rrtmgp_.save_lw_sw_heating_rates` is True.
  aux_output = rad_states_2

  # Put junk into T2 halos.
  T2 = update_z_halos_bcs(T2, halo_width=1)

  return T2, aux_output

  # pylint: enable=invalid-name
