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

"""Microphysics one moment model."""

from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special
from swirl_jatmos import constants
from swirl_jatmos.microphysics import microphysics_config
from swirl_jatmos.microphysics import particles
from swirl_jatmos.microphysics import terminal_velocity_chen2022
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array
WaterParams: TypeAlias = water.WaterParams

AccretionParams: TypeAlias = microphysics_config.AccretionParams
AutoconversionParams: TypeAlias = microphysics_config.AutoconversionParams
IceParams: TypeAlias = microphysics_config.IceParams
RainParams: TypeAlias = microphysics_config.RainParams
SnowParams: TypeAlias = microphysics_config.SnowParams

# Some densities in kg/m^3.
RHO_ICE = microphysics_config.RHO_ICE
RHO_AIR = microphysics_config.RHO_AIR
RHO_WATER = microphysics_config.RHO_WATER

K_COND = 2.4e-2  # The thermal conductivity of air [J/(m s K)].
NU_AIR = 1.6e-5  # The kinematic visocity of air [m^2/s].
D_VAP = 2.26e-5  # The molecular diffusivity of water vapor [m^2/s].

ASYMMETRY_CLOUD = 0.8  # The optical asymmetry factor of cloud droplets.
DROPLET_N = microphysics_config.DROPLET_N


def _v_0(pp: RainParams | SnowParams, rho: Array | None = None) -> Array:
  """Compute normalized terminal velocity for precip particle of size r_0."""
  if isinstance(pp, RainParams):
    assert rho is not None
    g = constants.G  # gravitational acceleration [m/s^2]
    return jnp.sqrt(8 * pp.r_0 * g / (3 * pp.c_d) * (RHO_WATER / rho - 1))
  elif isinstance(pp, SnowParams):
    return jnp.array(2.0**2.25 * pp.r_0**0.25)
  else:
    raise ValueError(
        f'One of Rain or Snow is required but {type(pp)} was provided.'
    )


def _conduction_and_diffusion(
    pp: RainParams | SnowParams, temperature: Array, wp: WaterParams
) -> Array:
  """Compute G(T), the effect of thermal conduction and water diffusion.

  Args:
    pp: A RainParams or SnowParams object.
    temperature: The temperature of the fluid [K].
    wp: The water thermodynamics parameters.

  Returns:
    The G(T) term in the microphysics equations.
  """
  # pylint: disable=invalid-name
  T = temperature
  if isinstance(pp, SnowParams):
    Lh = water.lh_s(T, wp)
  else:
    Lh = water.lh_v(T, wp)

  R_v = wp.r_v
  p_v_sat = water.saturation_vapor_pressure(T, wp)
  G = 1 / (
      Lh / (K_COND * T) * (Lh / (R_v * T) - 1.0) + (R_v * T) / (p_v_sat * D_VAP)
  )
  return G
  # pylint: enable=invalid-name


def autoconversion_rain(
    q_liq: Array, autoconversion_params: AutoconversionParams
) -> Array:
  """Compute the increase rate of rain precipitation q_r due to autoconversion.

  Rain autoconversion is the process of cloud liquid water converting to rain
  due to collisions between cloud droplets.

  Args:
    q_liq: The specific humidity of cloud water in the liquid phase.
    autoconversion_params: Autoconversion parameters.

  Returns:
    dq_r/dt|_acnv, the rate of increase of rain precipitation due to
    autoconversion.
  """
  q_liq_threshold = autoconversion_params.q_l_threshold
  tau_autoconversion_lr = autoconversion_params.tau_lr
  dqr_dt_acnv = (q_liq - q_liq_threshold) / tau_autoconversion_lr
  return jnp.clip(dqr_dt_acnv, 0.0, None)


def autoconversion_snow_nosupersat(
    q_ice: Array,
    autoconversion_params: AutoconversionParams,
) -> Array:
  """Compute the increase of snow precipitation q_s due to autoconversion.

  Snow autoconversion is the process of cloud ice converting to snow due to
  growth of cloud ice by vapor deposition.  This function is a simple
  parameterization of autoconversion, where the model does not allow for
  supersaturation.

  Args:
    q_ice: The specific humidity of cloud ice in the ice phase.
    autoconversion_params: Autoconversion parameters.

  Returns:
    dq_s/dt|_acnv, the rate of increase of snow due to autoconversion.
  """
  q_ice_threshold = autoconversion_params.q_i_threshold
  tau_autoconversion_is = autoconversion_params.tau_is
  dqs_dt_acnv = (q_ice - q_ice_threshold) / tau_autoconversion_is
  return jnp.clip(dqs_dt_acnv, 0.0, None)


def autoconversion_snow(
    temperature: Array,
    rho: Array,
    q_v: Array,
    q_ice: Array,
    wp: WaterParams,
    autoconversion_params: AutoconversionParams,
    snow_params: SnowParams,
    ice_params: IceParams,
) -> Array:
  """Compute the increase rate of snow precipitation q_s due to autoconversion.

  Snow autoconversion is the process of cloud ice converting to snow due to
  growth of cloud ice by vapor deposition.

  ** Note that according to the CliMa web page on 1-moment microphysics, q_v_sat
  in this formula is supposed to be the saturation specific humidity of the
  vapor over *ice*.  Need to be careful about how to compute this.  Here, we are
  using the parameters of a mixture of water and ice, following SwirlLM.

  Args:
    temperature: The fluid thermodynamic temperature [K].
    rho: The density of the moist air [kg/m^3].
    q_v: The specific humidity of the gas phase [kg/kg].
    q_ice: The specific humidity of the ice phase [kg/kg].
    wp: The water thermodynamics parameters.
    autoconversion_params: Autoconversion parameters.
    snow_params: Snow parameters.
    ice_params: Ice parameters.

  Returns:
    dq_s/dt|_acnv, the rate of increase of snow due to autoconversion.
  """
  # pylint: disable=invalid-name
  q_v_sat = water.saturation_vapor_humidity(temperature, rho, wp)
  S = q_v / q_v_sat  # Saturation ratio.

  G = _conduction_and_diffusion(snow_params, temperature, wp)
  lambda_inv = particles.marshall_palmer_distribution_parameter_lambda_inverse(
      ice_params, rho, q_ice
  )
  r_is = autoconversion_params.r_is
  n_0 = particles.n_0(ice_params)

  # Compute exp(-λr).  For the case when 1/λ is 0, λ is +∞, so exp(-λr) = 0.
  exp_term = jnp.where(lambda_inv > 0, jnp.exp(-r_is / lambda_inv), 0.0)
  factor1 = 4 * np.pi / rho * (S - 1) * G * n_0 * exp_term

  m_e = ice_params.m_e
  del_m = ice_params.del_m
  factor2 = r_is**2 / (m_e + del_m) + lambda_inv * (r_is + lambda_inv)

  dqs_dt_acnv = factor1 * factor2
  return jnp.clip(dqs_dt_acnv, 0.0, None)
  # pylint: enable=invalid-name


def _collision_efficiency(
    s1: Literal['liq', 'ice', 'rain', 'snow'],
    s2: Literal['rain', 'snow'],
    accretion_params: AccretionParams,
) -> float:
  """Return the collision efficiency for two colliding species.

  This function requires that cloud condensate (liquid or ice), if present, is
  input as the first species argument `s1`.

  Args:
    s1: The first species argument.
    s2: The second species argument.
    accretion_params: Accretion parameters.

  Returns:
    The collision efficiency between the two species.
  """
  if s1 == 'liq' and s2 == 'rain':
    return accretion_params.e_liq_rain
  elif s1 == 'liq' and s2 == 'snow':
    return accretion_params.e_liq_snow
  elif s1 == 'ice' and s2 == 'rain':
    return accretion_params.e_ice_rain
  elif s1 == 'ice' and s2 == 'snow':
    return accretion_params.e_ice_snow
  elif (s1 == 'rain' and s2 == 'snow') | (s1 == 'snow' and s2 == 'rain'):
    return accretion_params.e_rain_snow
  else:
    raise ValueError(
        f'Collision efficiency between {s1} and {s2} is not defined.'
    )


def accretion(
    cloud_type: Literal['liq', 'ice'],
    q_clo: Array,
    pp: RainParams | SnowParams,
    q_p: Array,
    rho: Array,
    accretion_params: AccretionParams,
) -> Array:
  """Compute the accretion source term due to collisions with cloud water.

  Accretion is the process in which precipitating water is grown due to
  collisions.  This function returns the source of precipitate (q_r or q_s) due
  to collisions with cloud water (q_liq or q_ice).

  Args:
    cloud_type: The type of cloud water, which can be 'liq' or 'ice'.
    q_clo: The specific humidity [kg/kg]of the cloud water, either q_liq or
      q_ice.
    pp: Particle parameters, which is a RainParams or SnowParams.
    q_p: The precipitation water mass fraction corresponding to `pp`, which can
      be rain (q_r) or snow (q_s). [kg/kg].
    rho: The density of the moist air [kg/m^3].
    accretion_params: Accretion parameters.

  Returns:
    dq_p/dt|_accr, the rate of change of specific humidity of the precipitation
    (rain or snow) due to the collision with cloud water (q_liq or q_ice).  The
    returned rate is nonnegative.
  """
  lambda_inv = particles.marshall_palmer_distribution_parameter_lambda_inverse(
      pp, rho, q_p
  )
  n_0 = particles.n_0(pp, rho, q_p)
  v_0 = _v_0(pp, rho)

  precip_type = 'rain' if isinstance(pp, RainParams) else 'snow'
  coll_eff = _collision_efficiency(cloud_type, precip_type, accretion_params)

  sigma_av = pp.a_e + pp.v_e + pp.del_a + pp.del_v
  pi_av_coeff = pp.a_0 * v_0 * pp.chi_a * pp.chi_v
  gamma = float(special.gamma(sigma_av + 1))
  factor1 = n_0 * pi_av_coeff * q_clo * coll_eff * gamma * lambda_inv
  factor2 = (lambda_inv / pp.r_0) ** sigma_av
  return factor1 * factor2


def accretion_rain_sink(
    q_ice: Array,
    q_r: Array,
    rho: Array,
    rain_params: RainParams,
    ice_params: IceParams,
    accretion_params: AccretionParams,
) -> Array:
  """Compute the accretion sink of rain due to (ice + rain -> snow) collisions.

  The output value is nonnegative and contributes as a sink of rain and source
  of snow.

  The output is the contribution to the snow source, that is,

      dq_s/dt|_accr = -dq_r/dt|_accr

  Args:
    q_ice: The specific humidity of the ice phase [kg/kg].
    q_r: The specific humidity of the rain phase [kg/kg].
    rho: The density of the moist air [kg/m^3].
    rain_params: Rain parameters.
    ice_params: Ice parameters.
    accretion_params: Accretion parameters.

  Returns:
    The accretion sink of rain due to (ice + rain -> snow) collisions.
  """
  lambda_ice_inv = (
      particles.marshall_palmer_distribution_parameter_lambda_inverse(
          ice_params, rho, q_ice
      )
  )
  lambda_rain_inv = (
      particles.marshall_palmer_distribution_parameter_lambda_inverse(
          rain_params, rho, q_r
      )
  )
  n_0_ice = particles.n_0(ice_params)
  n_0_r = particles.n_0(rain_params)
  coll_eff = _collision_efficiency('ice', 'rain', accretion_params)

  v0_r = _v_0(rain_params, rho)

  rp = rain_params
  pi_r_mav = rp.m_0 * rp.a_0 * v0_r * rp.chi_m * rp.chi_a * rp.chi_v
  sigma_r_mav = rp.m_e + rp.a_e + rp.v_e + rp.del_m + rp.del_a + rp.del_v
  gamma = float(special.gamma(sigma_r_mav + 1))

  factor1 = coll_eff * n_0_r * n_0_ice * pi_r_mav * gamma
  factor2 = lambda_ice_inv * lambda_rain_inv / rho
  factor3 = (lambda_rain_inv / rp.r_0) ** sigma_r_mav

  neg_dq_r_dt_accr = factor1 * factor2 * factor3
  return neg_dq_r_dt_accr


def accretion_snow_rain(
    s_a_params: RainParams | SnowParams,
    s_b_params: RainParams | SnowParams,
    q_a: Array,
    q_b: Array,
    rho: Array,
    microphysics_cfg: microphysics_config.MicrophysicsConfig,
):
  """Compute the accretion rate for collisions between snow and rain.

  This function gives the source of species a, i.e. the transition q_b -> q_a.

  If T < T_freeze, these collisions result in snow (q_r -> q_s).  For this case,
  this function should be called with s_a = snow, s_b = rain.

  If T > T_freeze, these collisions result in rain (q_s -> q_r).  For this case,
  this function should be called with s_a = rain, s_b = snow.

  Args:
    s_a_params: The parameters of species a, which can be rain or snow.
    s_b_params: The parameters of species b, which can be rain or snow.
    q_a: The specific humidity of species a [kg/kg], either q_r or q_s.
    q_b: The specific humidity of species b [kg/kg], either q_r or q_s.
    rho: The density of the moist air [kg/m^3].
    microphysics_cfg: The microphysics configuration parameters.

  Returns:
    The accretion rate source term of species a, which is also the sink of
    species b.  This value is nonnegative.
  """
  assert type(s_a_params) != type(s_b_params)  # pylint: disable=unidiomatic-typecheck
  n0_a = particles.n_0(s_a_params, rho, q_a)
  n0_b = particles.n_0(s_b_params, rho, q_b)

  r0_b, m0_b, me_b = s_b_params.r_0, s_b_params.m_0, s_b_params.m_e
  del_m_b, chi_m_b = s_b_params.del_m, s_b_params.chi_m

  s_a = 'rain' if isinstance(s_a_params, RainParams) else 'snow'
  s_b = 'rain' if isinstance(s_b_params, RainParams) else 'snow'
  coll_eff = _collision_efficiency(s_a, s_b, microphysics_cfg.accretion_params)

  lambda_inv_a = (
      particles.marshall_palmer_distribution_parameter_lambda_inverse(
          s_a_params, rho, q_a
      )
  )
  lambda_inv_b = (
      particles.marshall_palmer_distribution_parameter_lambda_inverse(
          s_b_params, rho, q_b
      )
  )

  # Compute the terminal velocity of the two species.
  # print('use Chen terminal velocity in accretion_snow_rain()')
  # varname_a = 'q_r' if isinstance(s_a_params, RainParams) else 'q_s'
  # v_ta = terminal_velocity(varname_a, q_a, rho, microphysics_cfg)
  # varname_b = 'q_r' if isinstance(s_b_params, RainParams) else 'q_s'
  # v_tb = terminal_velocity(varname_b, q_b, rho, microphysics_cfg)

  # Use the power-law terminal velocity here.
  print('use power-law terminal velocity in accretion_snow_rain()')
  v_ta = _terminal_velocity_power_law(s_a_params, rho, q_a)
  v_tb = _terminal_velocity_power_law(s_b_params, rho, q_b)

  factor1 = np.pi * n0_a * n0_b * m0_b * chi_m_b * coll_eff
  factor2 = jnp.abs(v_ta - v_tb) / rho / (r0_b ** (me_b + del_m_b))

  gamma1 = float(special.gamma(me_b + del_m_b + 1))
  gamma2 = float(special.gamma(me_b + del_m_b + 2))
  gamma3 = float(special.gamma(me_b + del_m_b + 2))

  term1 = 2 * gamma1 * lambda_inv_a**3 * lambda_inv_b ** (me_b + del_m_b + 1)
  term2 = 2 * gamma2 * lambda_inv_a**2 * lambda_inv_b ** (me_b + del_m_b + 2)
  term3 = gamma3 * lambda_inv_a * lambda_inv_b ** (me_b + del_m_b + 3)
  factor3 = term1 + term2 + term3
  accr_rate = factor1 * factor2 * factor3
  return accr_rate


def evaporation_sublimation(
    pp: RainParams | SnowParams,
    temperature: Array,
    rho: Array,
    q_v: Array,
    q_p: Array,
    wp: WaterParams,
) -> Array:
  """Compute the rate of change of precipitation by evaporation/sublimation.

  Note that for the case of rain we only consider evaporation, i.e., unsaturated
  conditions (S < 1), where S is the saturation ratio.  For the case of snow, we
  consider both the source term due to vapor deposition on snow (S > 1) and the
  sink due to vapor sublimation (S < 1).

  For evaporation (`pp` is a `RainParams`), the returned rate is nonnegative.
  For sublimation/deposition (`pp` is a `SnowParams`), the returned rate can be
  positive or negative.

  Args:
    pp: A RainParams or SnowParams object.
    temperature: The temperature of the fluid [K].
    rho: The density of the moist air [kg/m^3].
    q_v: The specific humidity of the gas phase [kg/kg].
    q_p: The precipitation specific humidity (or mass fraction), which can be
      rain (q_r) or snow (q_s) [kg/kg].
    wp: The water thermodynamics parameters.

  Returns:
    The rate [1/s] of evaporation/sublimation, i.e., dq_t/dt|_evap or
    dq_t/dt|_subl.  Here, dq_t/dt|_evap = -dq_r/dt|_evap, and similarly for
    snow.  If the rate dq_t/dt is > 0, then the amount of rain or snow is
    decreasing and the amount of water vapor (q_t) is increasing.  Note that
    deposition is also considered so rate can be < 0 in the case of snow, but
    for evaporation the rate can only be >= 0.
  """
  # pylint: disable=invalid-name
  if isinstance(pp, RainParams):
    q_v_sat_over_water = water.saturation_vapor_humidity_over_liquid_water(
        temperature, rho, wp
    )
    S = q_v / q_v_sat_over_water
  else:  # SnowParams
    q_v_sat_over_ice = water.saturation_vapor_humidity_over_ice(
        temperature, rho, wp
    )
    S = q_v / q_v_sat_over_ice

  G = _conduction_and_diffusion(pp, temperature, wp)
  lambda_inv = particles.marshall_palmer_distribution_parameter_lambda_inverse(
      pp, rho, q_p
  )

  a_vent, b_vent, r_0 = pp.a_vent, pp.b_vent, pp.r_0
  v_e, del_v, chi_v = pp.v_e, pp.del_v, pp.chi_v
  v_0 = _v_0(pp, rho)

  gamma = float(special.gamma(0.5 * (v_e + del_v + 5)))
  f_vent = (
      a_vent
      + b_vent
      * (NU_AIR / D_VAP) ** (1.0 / 3.0)
      * (lambda_inv / r_0) ** (0.5 * (v_e + del_v))
      * jnp.sqrt(2 * chi_v * v_0 * lambda_inv / NU_AIR)
      * gamma
  )

  n_0 = particles.n_0(pp, rho, q_p)

  # Compute dq_t/dt due to evaporation or sublimation.
  dq_t_dt = -4 * np.pi * n_0 / rho * (S - 1) * G * lambda_inv**2 * f_vent

  if isinstance(pp, RainParams):
    # For rain, only allow dq_t/dt to be >= 0, i.e., evaporation is considered
    # only for unsaturated conditions S < 1.
    dq_t_dt = jnp.clip(dq_t_dt, 0.0, None)
  return dq_t_dt
  # pylint: enable=invalid-name


def snow_melt(
    temperature: Array,
    rho: Array,
    q_s: Array,
    wp: WaterParams,
    snow_params: SnowParams,
) -> Array:
  """Compute the rate of change of snow melting into rain.

  Args:
    temperature: The temperature of the fluid [K].
    rho: The density of the moist air [kg/m^3].
    q_s: The specific humidity of the snow phase [kg/kg].
    wp: The water thermodynamics parameters.
    snow_params: Snow parameters.

  Returns:
    The rate [1/s] of snow melting into rain, i.e., dq_r/dt|_melt, which is
    equal to -dq_s/dt|_melt.  The value is nonnegative.
  """
  T = temperature  # pylint: disable=invalid-name
  T_freeze = wp.t_freeze  # pylint: disable=invalid-name
  a_vent, b_vent, r_0 = snow_params.a_vent, snow_params.b_vent, snow_params.r_0
  v_e, del_v, chi_v = snow_params.v_e, snow_params.del_v, snow_params.chi_v
  v_0 = _v_0(snow_params, rho)

  lambda_inv = particles.marshall_palmer_distribution_parameter_lambda_inverse(
      snow_params, rho, q_s
  )
  lh_f = water.lh_f(temperature, wp)

  n_0 = particles.n_0(snow_params, rho, q_s)
  term1 = 4 * np.pi * n_0 * K_COND / rho / lh_f * (T - T_freeze) * lambda_inv**2

  gamma = float(special.gamma(0.5 * (v_e + del_v + 5)))
  f_vent = (
      a_vent
      + b_vent
      * (NU_AIR / D_VAP) ** (1.0 / 3.0)
      * (lambda_inv / r_0) ** (0.5 * (v_e + del_v))
      * jnp.sqrt(2 * chi_v * v_0 * lambda_inv / NU_AIR)
      * gamma
  )
  dq_r_dt_melt = term1 * f_vent
  dq_r_dt_melt = jnp.where(T > T_freeze, dq_r_dt_melt, 0.0)
  return dq_r_dt_melt


def _terminal_velocity_power_law(
    pp: RainParams | SnowParams,
    rho: Array,
    q_p: Array,
) -> Array:
  """Compute the bulk terminal velocity using a power-law parameterization.

  Args:
    pp: A RainParams or SnowParams object.
    rho: The density of the moist air [kg/m^3].
    q_p: The precipitation water mass fraction, which can be rain (q_r) or snow
      (q_s) [kg/kg].  [*Can also be q_liq or q_ice if using this function
      to compute sedimentation velocities of cloud water or cloud ice.]

  Returns:
    The terminal velocity of rain or snow [m/s].
  """
  lambda_inv = particles.marshall_palmer_distribution_parameter_lambda_inverse(
      pp, rho, q_p
  )

  chi_v, r_0, v_e, del_v = pp.chi_v, pp.r_0, pp.v_e, pp.del_v
  m_e, del_m = pp.m_e, pp.del_m
  v_0 = _v_0(pp, rho)

  gamma1 = float(special.gamma(m_e + v_e + del_m + del_v + 1))
  gamma2 = float(special.gamma(m_e + del_m + 1))
  return chi_v * v_0 * (lambda_inv / r_0) ** (v_e + del_v) * gamma1 / gamma2


def terminal_velocity(
    varname: Literal['q_r', 'q_s', 'q_liq', 'q_ice'],
    q: Array,
    rho: Array,
    microphysics_cfg: microphysics_config.MicrophysicsConfig,
) -> Array:
  """Compute the terminal velocity of rain, snow, or cloud water/ice.

  Args:
    varname: The variable name for which to compute the terminal velocity, which
      can be 'q_r', 'q_s', 'q_liq', or 'q_ice'.
    q: The specific humidity of the variable corresponding to `varname`.
    rho: The density of the moist air [kg/m^3].
    microphysics_cfg: The microphysics configuration parameters.

  Returns:
    The terminal velocity of the variable corresponding to `varname`.
  """
  assert varname in ['q_r', 'q_s', 'q_liq', 'q_ice']
  terminal_velocity_method = microphysics_cfg.terminal_velocity_method
  TerminalVelocityMethod: TypeAlias = microphysics_config.TerminalVelocityMethod

  if terminal_velocity_method is TerminalVelocityMethod.POWER_LAW:
    # Compute terminal velocity using power-law parameterization.
    if varname in ['q_r', 'q_liq']:
      pp = microphysics_cfg.rain_params
    else:  # ['q_s', 'q_ice']
      pp = microphysics_cfg.snow_params
    return _terminal_velocity_power_law(pp, rho, q)
  elif terminal_velocity_method is TerminalVelocityMethod.CHEN_2022:
    # Compute terminal velocity using the Chen et al. (2022) parameterization.
    if varname == 'q_r':
      return terminal_velocity_chen2022.rain_terminal_velocity(
          rho, q, microphysics_cfg.rain_params
      )
    elif varname == 'q_s':
      return terminal_velocity_chen2022.snow_terminal_velocity(
          rho,
          q,
          microphysics_cfg.snow_params,
          microphysics_cfg.snow_table_b5_coeffs,
      )
    elif varname == 'q_liq':
      return terminal_velocity_chen2022.liquid_condensate_terminal_velocity(
          rho, q, microphysics_cfg.rain_params
      )
    else:  # varname == 'q_ice'
      return terminal_velocity_chen2022.ice_condensate_terminal_velocity(
          rho,
          q,
          microphysics_cfg.ice_params,
          microphysics_cfg.ice_table_b3_coeffs,
      )
  else:
    raise ValueError(
        f'Unknown terminal velocity method: {terminal_velocity_method}'
    )


def cloud_particle_effective_radius(
    rho: Array, q_c: Array, phase: Literal['liq', 'ice']
) -> Array:
  """Compute the 1-moment approximation of cloud particle effective radius.

  This follows the formulation from Liu and Hallett (1997) equation 8.  The
  concentration of cloud particles is assumed to be constant and the 1/3 power
  law between effective radius and water content is used.  Particles are assumed
  to be spherical for both liquid and ice, which is an oversimplification for
  ice.  The same asymmetry factor is used for liquid and ice.

  Args:
    rho: The density of the moist air [kg/m^3].
    q_c: The condensed-phase specific humidity [kg/kg].  Note that this is the
      specific humidity of either ice or liquid, NOT their sum.
    phase: The phase of the cloud water, either 'liq' or 'ice'.

  Returns:
    The effective radius [m] of the cloud droplets or ice particles.
  """
  # Density of the condensate.
  rho_c = RHO_WATER if phase == 'liq' else RHO_ICE

  alpha = (4.0 / 3.0 * np.pi * rho_c * ASYMMETRY_CLOUD) ** (-1 / 3)
  return alpha * (rho * q_c / DROPLET_N) ** (1 / 3)
