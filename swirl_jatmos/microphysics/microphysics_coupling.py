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

"""Module for computing coupling of microphysics to evolution of scalars.

This module has functionality for computing the tendencies in the scalar
evolution equations, using the one-moment microphysics model.
"""

import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp
from swirl_jatmos.microphysics import microphysics_config
from swirl_jatmos.microphysics import microphysics_one_moment
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array


def _limit_src(src: Array, q: Array, dt: Array, k: float) -> Array:
  """Return a limited version of source `src` which decreases `q`.

  Consider the evolution of `q` due solely to `src`:

        dq/dt = -src

  It is important that it is assumed here that we know `src` is nonnegative.  If
  using an Euler step, we have

        q_new = q - dt * src

  The constraint that q_new >= 0 leads to the inequality

      src <= q / dt

  We used a generalized constraint, that q_new >= k * q, with 0 < k <= 1, to
  account for the fact that other terms affect the evolution of q as well.  The
  constraint is

      src <= (1-k)q/dt

  I.e., the maximum value the `src` is allowed to be is (1-k) * q / dt.

  Args:
    src: The source term to be limited.
    q: The scalar value that is being decreased by `src`.
    dt: The time step.
    k: The coefficient for the limiting constraint.
  """
  return jnp.clip(src, None, (1-k) * q / dt)


# pylint: disable=invalid-name
def scalar_tendencies_from_microphysics(
    q_t: Array,
    q_r: Array,
    q_s: Array,
    q_v: Array,
    q_liq: Array,
    q_ice: Array,
    T: Array,
    rho: Array,
    p_ref: Array,
    dt: Array,
    wp: water.WaterParams,
    microphysics_cfg: microphysics_config.MicrophysicsConfig,
) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
  """Compute the tendencies due to microphysics for scalar evolution."""
  rain_params = microphysics_cfg.rain_params
  snow_params = microphysics_cfg.snow_params
  accretion_params = microphysics_cfg.accretion_params
  autoconversion_params = microphysics_cfg.autoconversion_params

  # Apply clipping on inputs (note that q_liq, q_ice are nonnegative already).
  q_t = jnp.clip(q_t, 0.0, None)
  q_v = jnp.clip(q_v, 0.0, None)
  q_r = jnp.clip(q_r, 0.0, None)
  q_s = jnp.clip(q_s, 0.0, None)

  R_m = (1 - q_t) * water.R_D + q_v * wp.r_v
  cp_m = (1 - q_t) * water.CP_D + q_v * wp.cp_v
  exner_inv = (p_ref / wp.exner_reference_pressure) ** (-R_m / cp_m)
  Lv0 = wp.lh_v0
  Ls0 = wp.lh_s0
  Lv = water.lh_v(T, wp)
  Ls = water.lh_s(T, wp)
  Lf = water.lh_f(T, wp)
  cv_l = wp.cv_l
  cv_m = water.cv_m(q_t, q_liq, q_ice, wp)

  T_freeze = wp.t_freeze

  tendency_q_t = jnp.zeros_like(q_t)
  tendency_q_r = jnp.zeros_like(q_t)
  tendency_q_s = jnp.zeros_like(q_t)
  tendency_theta_li = jnp.zeros_like(q_t)
  aux_output = {}

  # Function for limiting the source terms.
  limit = functools.partial(_limit_src, dt=dt, k=microphysics_cfg.k_coeff)

  #  ************  Tendencies from autoconversion  ************  #
  # S_qr_autocnv and S_qs_autocnv are nonnegative.
  # Rain autoconversion, q_liq -> q_r.
  S_qr_autocnv = microphysics_one_moment.autoconversion_rain(
      q_liq, autoconversion_params
  )
  S_qr_autocnv = limit(S_qr_autocnv, q_liq)

  # Snow autoconversion, q_ice -> q_s.
  S_qs_autocnv = microphysics_one_moment.autoconversion_snow_nosupersat(
      q_ice, autoconversion_params
  )
  S_qs_autocnv = limit(S_qs_autocnv, q_ice)
  tendency_q_r = tendency_q_r + S_qr_autocnv
  tendency_q_s = tendency_q_s + S_qs_autocnv
  tendency_q_t = tendency_q_t - (S_qr_autocnv + S_qs_autocnv)
  tendency_theta_li = tendency_theta_li + (exner_inv / cp_m) * (
      Lv0 * S_qr_autocnv + Ls0 * S_qs_autocnv
  )

  #  **************  Tendencies from accretion  ***************  #
  # Accretion (1): cloud water & rain, q_liq -> q_r.
  S_qr = microphysics_one_moment.accretion(
      'liq', q_liq, rain_params, q_r, rho, accretion_params
  )
  S_qr = limit(S_qr, q_liq)
  # S_qr >= 0
  tendency_q_r = tendency_q_r + S_qr
  tendency_q_t = tendency_q_t - S_qr
  tendency_theta_li = tendency_theta_li + (exner_inv / cp_m) * Lv0 * S_qr

  # Accretion (2): cloud ice & snow, q_ice -> q_s.
  S_qs = microphysics_one_moment.accretion(
      'ice', q_ice, snow_params, q_s, rho, accretion_params
  )
  S_qs = limit(S_qs, q_ice)
  # S_qs >= 0
  tendency_q_s = tendency_q_s + S_qs
  tendency_q_t = tendency_q_t - S_qs
  tendency_theta_li = tendency_theta_li + (exner_inv / cp_m) * Ls0 * S_qs

  # Accretion (3): cloud water + snow, to snow or rain.
  sink_qt = microphysics_one_moment.accretion(
      'liq', q_liq, snow_params, q_s, rho, accretion_params
  )
  # sink_qt is positive, and it decreases q_t (decreases q_liq).
  sink_qt = limit(sink_qt, q_liq)

  # Tendencies for T < T_freeze; the transition is q_liq -> q_s
  d_tndncy_q_r_less = jnp.zeros_like(q_t)
  d_tndncy_q_s_less = sink_qt
  d_tndncy_q_t_less = -sink_qt
  d_tndncy_theta_li_less = exner_inv / cp_m * sink_qt * Lf * (1 + R_m / cv_m)

  # Tendencies for T > T_freeze; the transition is q_liq + q_s -> q_r
  alpha = (cv_l / Lf) * (T - T_freeze)
  d_tndncy_q_r_grtr = (1 + alpha) * sink_qt
  d_tndncy_q_s_grtr = -alpha * sink_qt
  d_tndncy_q_t_grtr = -sink_qt
  d_tndncy_theta_li_grtr = (
      -exner_inv / cp_m * sink_qt * (alpha * Lf * (1 + R_m / cv_m) - Lv0)
  )

  # Select based on T.
  d_tndncy_q_r = jnp.where(T < T_freeze, d_tndncy_q_r_less, d_tndncy_q_r_grtr)
  d_tndncy_q_s = jnp.where(T < T_freeze, d_tndncy_q_s_less, d_tndncy_q_s_grtr)
  d_tndncy_q_t = jnp.where(T < T_freeze, d_tndncy_q_t_less, d_tndncy_q_t_grtr)
  d_tndncy_theta_li = jnp.where(
      T < T_freeze, d_tndncy_theta_li_less, d_tndncy_theta_li_grtr
  )

  tendency_q_r = tendency_q_r + d_tndncy_q_r
  tendency_q_s = tendency_q_s + d_tndncy_q_s
  tendency_q_t = tendency_q_t + d_tndncy_q_t
  tendency_theta_li = tendency_theta_li + d_tndncy_theta_li

  # Accretion (4): q_ice + q_r -> q_s.
  # Commenting out because this is probably unimportant for now.
  # sink_qi = microphysics_one_moment.accretion(
  #     'ice', q_ice, rain_params, q_r, rho, accretion_params
  # )
  # # sink_qi >= 0 and decreases q_ice (decreases q_t).
  # sink_qi = limit(sink_qi, q_ice)

  # sink_qr = microphysics_one_moment.accretion_rain_sink(
  #     q_ice, q_r, rho, rain_params, ice_params, accretion_params
  # )
  # # sink_qr >= 0 and decreases q_r.
  # sink_qr = limit(sink_qr, q_r)

  # tendency_q_r = tendency_q_r - sink_qr
  # tendency_q_t = tendency_q_t - sink_qi
  # tendency_q_s = tendency_q_s + sink_qi + sink_qr
  # tendency_theta_li = tendency_theta_li + (exner_inv / cp_m) * (
  #     sink_qr * Lf * (1 + R_m / cv_m) + sink_qi * Ls0
  # )

  # Accretion (5):
  # Commenting out because this is probably unimportant for now.
  #   If t < T_freeze:  q_r -> q_s
  #   If t > T_freeze:  q_s -> q_r
  # S_qs_T_less_than_T_freeze = microphysics_one_moment.accretion_snow_rain(
  #     snow_params, rain_params, q_s, q_r, rho, microphysics_cfg
  # )
  # S_qs_T_less_than_T_freeze = limit(S_qs_T_less_than_T_freeze, q_r)

  # S_qr_T_greater_than_T_freeze = microphysics_one_moment.accretion_snow_rain(
  #     rain_params, snow_params, q_r, q_s, rho, microphysics_cfg
  # )
  # S_qr_T_greater_than_T_freeze = limit(S_qr_T_greater_than_T_freeze, q_s)

  # # S_qr_T_greater_than_T_freeze >= 0 and S_qs_T_less_than_T_freeze >= 0
  # # But now define S_qs, which can be of either sign, depending on which
  # # transition is occurring.

  # S_qs = jnp.where(
  #     T < T_freeze,
  #     S_qs_T_less_than_T_freeze,
  #     -S_qr_T_greater_than_T_freeze,
  # )

  # tendency_q_r = tendency_q_r - S_qs
  # tendency_q_s = tendency_q_s + S_qs
  # tendency_theta_li = tendency_theta_li + (exner_inv / cp_m) * S_qs * Lf

  #  ** Tendencies from evaporation, sublimation/deposition, and snow melt ** #
  # Evaporation: q_r -> q_v (q_t).
  S_qt_evap = microphysics_one_moment.evaporation_sublimation(
      rain_params, T, rho, q_v, q_r, wp
  )
  S_qt_evap = limit(S_qt_evap, q_r)

  # Sublimation / deposition:  q_s <--> q_v.
  tmp = microphysics_one_moment.evaporation_sublimation(
      snow_params, T, rho, q_v, q_s, wp
  )
  # tmp may be either positive or negative.
  # if tmp > 0: sublimation.  q_s -> q_v.  Want to limit based on q_s.
  # if tmp < 0: deposition.  q_v -> q_s.  Want to limit based on q_v.
  S_qt_subdep = jnp.where(tmp > 0, limit(tmp, q_s), -limit(-tmp, q_v))

  # Snow melt: q_s -> q_r.
  S_qr_melt = microphysics_one_moment.snow_melt(T, rho, q_s, wp, snow_params)
  S_qr_melt = limit(S_qr_melt, q_s)

  tendency_q_r = tendency_q_r + S_qr_melt - S_qt_evap
  tendency_q_s = tendency_q_s - S_qr_melt - S_qt_subdep
  tendency_q_t = tendency_q_t + S_qt_evap + S_qt_subdep
  tendency_theta_li = tendency_theta_li - (exner_inv / cp_m) * (
      S_qt_evap * (Lv - wp.r_v * T) * (1 + R_m / cv_m)
      + S_qt_subdep * (Ls - wp.r_v * T) * (1 + R_m / cv_m)
      + S_qr_melt * Lf * (1 + R_m / cv_m)
  )

  return tendency_q_t, tendency_q_r, tendency_q_s, tendency_theta_li, aux_output
