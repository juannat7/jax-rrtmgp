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

"""Water thermodynamics assuming the prognostic variable is theta_li."""

from collections.abc import Callable
import dataclasses
import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array

R_D = 286.69  # Gas constant for dry air [J/kg/K].

GAMMA = 1.4  # The heat capacity ratio of dry air, dimensionless.

# The specific heat of dry air at constant pressure [J/kg/K].
CP_D = GAMMA * R_D / (GAMMA - 1)


@dataclasses.dataclass(frozen=True)
class WaterParams:
  """Parameters for water thermodynamics."""
  r_v: float = 461.89  # Gas constant for water vapor.
  t_0: float = 273.16  # Reference temperature, used for latent heats
  t_min: float = 150.0  # Minimum allowed temperature
  t_freeze: float = 273.15  # Freezing point of water
  t_triple: float = 273.16  # Triple point of water
  t_icenuc: float = 233.0  # Ice nucleus temperature
  p_triple: float = 611.657  # Triple point pressure of water
  # Reference pressure [Pa] used in the Exner function, for the definition of
  # potential temperature.
  exner_reference_pressure: float = 1.013e5
  lh_v0: float = 2.5008e6  # Latent heat of vaporization at T=t_0 [J/kg].
  lh_s0: float = 2.8344e6  # Latent heat of sublimation at T=t_0 [J/kg].
  # Specific heat capacities:
  cv_v: float = 1397.11  # Isochoric specific heat of water vapor [J/kg/K].
  cp_v: float = 1846.0  # Isobaric specific heat of water vapor [J/kg/K].
  cv_l: float = 4165.0  # Isochoric specific heat of liquid water [J/kg/K].
  cp_l: float = 4181.9  # Isobaric specific heat of liquid water [J/kg/K].
  cv_i: float = 2050.0  # Isochoric specific heat of ice [J/kg/K].
  cp_i: float = 2100.0  # Isobaric specific heat of ice [J/kg/K].
  cv_d: float = 717.4664  # Isochoric specific heat of dry air [J/kg/K].

  single_iteration_loop: bool = True  # Set this to True after testing it's ok.
  max_newton_solver_iterations: int = 4
  # Number of outer iterations on density.  Only used when single_iteration_loop
  # is False.
  num_density_iterations: int = 10


@dataclasses.dataclass(frozen=True)
class ThermoFields:
  """Thermodynamic fields obtained from equilibrium calculation."""

  T: Array  # pylint: disable=invalid-name
  rho_thermal: Array
  q_liq: Array
  q_ice: Array
  q_c: Array  # Note, q_c = q_liq + q_ice.
  q_v: Array  # Note, q_v = q_t - q_c.
  q_v_sat: Array
  wp: WaterParams  # Water parameters, included here for convenience.


def lh_v(temperature: Array, wp: WaterParams) -> Array:
  """Compute the latent heat of vaporization at `temperature`.

  Args:
    temperature: The temperature of the fluid [K].
    wp: The water parameters.

  Returns:
    The latent heat of vaporization at the input temperature.
  """
  return wp.lh_v0 + (wp.cp_v - wp.cp_l) * (temperature - wp.t_0)


def lh_s(temperature: Array, wp: WaterParams) -> Array:
  """Compute the latent heat of sublimation at `temperature`.

  Args:
    temperature: The temperature of the fluid [K].
    wp: The water parameters.

  Returns:
    The latent heat of sublimation at the input temperature.
  """
  return wp.lh_s0 + (wp.cp_v - wp.cp_i) * (temperature - wp.t_0)


def lh_f(temperature: Array, wp: WaterParams) -> Array:
  """Compute the latent heat of fusion at `temperature`.

  Args:
    temperature: The temperature of the fluid [K].
    wp: The water parameters.

  Returns:
    The latent heat of fusion at the input temperature.
  """
  lh_f0 = wp.lh_s0 - wp.lh_v0
  return lh_f0 + (wp.cp_l - wp.cp_i) * (temperature - wp.t_0)


def cv_m(q_t: Array, q_liq: Array, q_ice: Array, wp: WaterParams) -> Array:
  """Computes the isochoric specific heat capacity of moist air.

  cv_m = cv_d + (cv_v - cv_d)q_t + (cv_l - cv_v)q_liq + (cv_i - cv_v)q_ice

  Args:
    q_t: The total specific humidity [kg/kg].
    q_liq: The liquid phase specific humidity [kg/kg].
    q_ice: The solid phase specific humidity [kg/kg].
    wp: The water parameters.

  Returns:
    The isochoric specific heat capacity of moist air.
  """
  cv_d, cv_v, cv_l, cv_i = wp.cv_d, wp.cv_v, wp.cv_l, wp.cv_i
  return (
      cv_d + (cv_v - cv_d) * q_t + (cv_l - cv_v) * q_liq + (cv_i - cv_v) * q_ice
  )


def liquid_fraction(temperature: Array, wp: WaterParams):
  """Determine the fraction of liquid in the condensed phase.

  The fraction of liquid is f_liq = q_l / q_c.  The liquid fraction function is
  defined as:

  f_liq(T) =
    0                                         T < T_icenuc
    (T - T_icenuc)/(T_freeze - T_icenuc)      T_icenuc < T < T_freeze
    1                                         T > T_freeze

  Args:
    temperature: The temperature of the fluid.
    wp: The water settings.

  Returns:
    The fraction of liquid phase over the condensed phase, which is between 0
    and 1.
  """

  def linear_fn(t: Array) -> Array:
    return (t - wp.t_icenuc) / (wp.t_freeze - wp.t_icenuc)

  liquid_frac_a = jnp.where(
      temperature < wp.t_icenuc,
      0.0,
      linear_fn(temperature),
  )
  liquid_frac_b = jnp.where(temperature > wp.t_freeze, 1.0, liquid_frac_a)
  return liquid_frac_b


def saturation_vapor_pressure(temperature: Array, wp: WaterParams) -> Array:
  """Get the saturation vapor pressure over a mixed liquid-ice plane surface.

  Assumes a mixture of liquid and ice, based on temperature.

  Clausius-Clapeyron relation is used to compute the saturation vapor pressure,
  which is:
    dlog(p_v_sat) / dT = L/ (R_v T^2).
  L is the specific latent heat with constant isobaric specific heats of the
  phase, which is represented by the Kirchholf's relation as:
    L = LH_0 + Δcp (T - T_0).
  Note that the linear dependency of L on T allows analytical integration.

  Args:
    temperature: The temperature of the fluid.
    wp: The water settings.

  Returns:
    The saturation vapor pressure, in Pa.
  """
  liquid_frac = liquid_fraction(temperature, wp)
  ice_frac = 1.0 - liquid_frac

  lh_0 = liquid_frac * wp.lh_v0 + ice_frac * wp.lh_s0
  delta_cp = liquid_frac * (wp.cp_v - wp.cp_l) + ice_frac * (wp.cp_v - wp.cp_i)

  return saturation_vapor_pressure_formula(temperature, lh_0, delta_cp, wp)


def saturation_vapor_pressure_over_liquid_water(
    temperature: Array, wp: WaterParams
) -> Array:
  """Get the saturation vapor pressure over a liquid water surface."""
  lh_0 = wp.lh_v0
  delta_cp = wp.cp_v - wp.cp_l
  return saturation_vapor_pressure_formula(temperature, lh_0, delta_cp, wp)


def saturation_vapor_pressure_over_ice(
    temperature: Array, wp: WaterParams
) -> Array:
  """Get the saturation vapor pressure over an ice surface."""
  lh_0 = wp.lh_s0
  delta_cp = wp.cp_v - wp.cp_i
  return saturation_vapor_pressure_formula(temperature, lh_0, delta_cp, wp)


def saturation_vapor_pressure_formula(
    temperature: Array,
    lh_0: Array | float,
    delta_cp: Array | float,
    wp: WaterParams,
) -> Array:
  """Computes the saturation vapor pressure over a plane surface.

  The plane surface can be a liquid surface, an ice surface, or a mixture of
  liquid and ice.  The surface is determined through the parameters `lh_0` and
  `delta_cp`.  A mixture of liquid and ice have weighted values of `lh_0` and
  `delta_cp`.

  Args:
    temperature: The temperature of the fluid.
    lh_0: The latent heat at the reference temperature of the surface fluid.
    delta_cp: The difference between the specific heat of the vapor and the
      specific heat of the surface fluid at isobaric condition.
    wp: The water settings.

  Returns:
    The saturation vapor pressure, in Pa.
  """
  return (
      wp.p_triple
      * (temperature / wp.t_triple) ** (delta_cp / wp.r_v)
      * jnp.exp(
          (lh_0 - delta_cp * wp.t_0)
          / wp.r_v
          * (1 / wp.t_triple - 1 / temperature)
      )
  )


def saturation_vapor_humidity(
    temperature: Array, rho: Array, wp: WaterParams
) -> Array:
  """Computes the saturation vapor humidity over a liquid-ice plane surface.

  Assumes a mixture of liquid and ice, based on temperature.

  Uses the ideal gas equation of state for the vapor component:

    qᵥₛ = pₛₐₜ / (ϱ Rᵥ T)

  Args:
    temperature: Temperature of the fluid.
    rho: The density of the moist air.
    wp: The water settings.

  Returns:
    The saturation specific vapor humidity, in kg/kg.
  """
  p_v_sat = saturation_vapor_pressure(temperature, wp)
  return p_v_sat / (rho * wp.r_v * temperature)


def saturation_vapor_humidity_over_liquid_water(
    temperature: Array, rho: Array, wp: WaterParams
) -> Array:
  """Computes the saturation vapor humidity over a liquid water surface.

  Uses the ideal gas equation of state for the vapor component:

    qᵥₛ = pₛₐₜ / (ϱ Rᵥ T)

  Args:
    temperature: Temperature of the fluid.
    rho: The density of the moist air.
    wp: The water settings.

  Returns:
    The saturation specific vapor humidity, in kg/kg.
  """
  p_v_sat = saturation_vapor_pressure_over_liquid_water(temperature, wp)
  return p_v_sat / (rho * wp.r_v * temperature)


def saturation_vapor_humidity_over_ice(
    temperature: Array, rho: Array, wp: WaterParams
) -> Array:
  """Computes the saturation vapor humidity over an ice surface.

  Uses the ideal gas equation of state for the vapor component:

    qᵥₛ = pₛₐₜ / (ϱ Rᵥ T)

  Args:
    temperature: Temperature of the fluid.
    rho: The density of the moist air.
    wp: The water settings.

  Returns:
    The saturation specific vapor humidity, in kg/kg.
  """
  p_v_sat = saturation_vapor_pressure_over_ice(temperature, wp)
  return p_v_sat / (rho * wp.r_v * temperature)


def equilibrium_phase_partition(
    temperature: Array,
    rho: Array,
    q_t: Array,
    wp: WaterParams,
) -> tuple[Array, Array]:
  """Determines the partition of condensate into liquid and ice phases.

  Given a fluid in thermal equilibrium, and a given temperature, density, and
  total humidity, this function determines the partition of the condensate
  between the liquid and ice phases.

  Args:
    temperature: The temeprature of the flow field.
    rho: The density of the moist air.
    q_t: The total specific humidity.
    wp: The water parameters.

  Returns:
    The specific humidities q_liq, q_ice of the liquid and ice phase, in kg/kg.
    The returned q_liq and q_ice are nonnegative.
  """
  q_v_sat = saturation_vapor_humidity(temperature, rho, wp)
  q_c = jnp.maximum(0, q_t - q_v_sat)

  liquid_frac = liquid_fraction(temperature, wp)
  q_liq = liquid_frac * q_c
  q_ice = (1.0 - liquid_frac) * q_c
  return q_liq, q_ice


def newton_root_finder(
    f: Callable[[Array], Array],
    fprime: Callable[[Array], Array],
    x_initial_guess: Array,
    max_iterations: int,
) -> Array:
  """Apply Newton's method to find the elementwise roots of f."""

  def body_fn(i, x):
    del i
    # What to do if divide by 0 or divide by a very small number?
    # clamp in x?
    return x - f(x) / fprime(x)

  x = jax.lax.fori_loop(0, max_iterations, body_fn, x_initial_guess)
  return x


def theta_li_from_temperature_rho_qt(
    T: Array, rho: Array, q_t: Array, p_ref: Array, wp: WaterParams  # pylint: disable=invalid-name
) -> Array:
  """Compute theta_li, from T, rho, q_t, p_ref in equilibrium."""
  # Determine the amount of saturation as well as the split into liquid and ice
  # phases.
  q_liq, q_ice = equilibrium_phase_partition(T, rho, q_t, wp)
  q_c = q_liq + q_ice

  # Get the versions of R_m, cp_m, and the Exner function that account for both
  # moisture and the condensed phase.
  rm = (1 - q_t) * R_D + (q_t - q_c) * wp.r_v
  cpm = (1 - q_t) * CP_D + (q_t - q_c) * wp.cp_v
  exner_inv = (p_ref / wp.exner_reference_pressure) ** (-rm / cpm)
  # Compute the liquid-ice potential temperature.
  theta_li = exner_inv * (T - (wp.lh_v0 * q_liq + wp.lh_s0 * q_ice) / cpm)
  return theta_li


def saturation_adjustment(
    theta_li: Array,
    rho: Array,
    q_t: Array,
    p_ref: Array,
    wp: WaterParams,
) -> Array:
  """Determine the temperature assuming thermal equilibrium.

  Given theta_li, rho, q_t, p_ref, determine the temperature T such that
  theta_li_from_temperature_rho_qt(T, rho, q_t, p_ref, wp) = theta_li.

  Complexity arises due to saturation conditions, where q_l and q_i can
  suddenly jump from being zero to being nonzero.

  Args:
    theta_li: The liquid-ice potential temperature.
    rho: The density of the moist air.
    q_t: The total specific humidity.
    p_ref: The reference hydrostatic pressure as a function of height.
    wp: The water parameters.

  Returns:
    The temperature T such that theta_li_from_temperature_rho_qt(T, rho, q_t,
    p_ref, wp) = theta_li.
  """
  # pylint: disable=invalid-name
  # Find T1, the temperature computed assuming the air is unsaturated with
  # q_liq = q_ice = q_c = 0.
  R_m_unsat = (1 - q_t) * R_D + q_t * wp.r_v
  cp_m_unsat = (1 - q_t) * CP_D + q_t * wp.cp_v
  exner = (p_ref / wp.exner_reference_pressure) ** (R_m_unsat / cp_m_unsat)
  T1 = exner * theta_li
  T1 = jnp.maximum(wp.t_min, T1)  # Apply a cutoff for numerical reasons.

  # Calculate the saturation humidity at temperature T1.
  q_v_sat = saturation_vapor_humidity(T1, rho, wp)

  # Case 2: temperature at freezing point.
  # Compute theta_li at the freezing point for unsaturated (assuming q_c=0)
  theta_li_freeze = wp.t_freeze / exner

  # Case 3: temperature at saturation condition.
  # Goal: define f(T) = theta_li_eqb(T) - theta_li
  #  Find T to make f(T) = 0.
  def f(
      T: Array, rho: Array, q_t: Array, p_ref: Array, theta_li: Array
  ) -> Array:
    theta_li_sat = theta_li_from_temperature_rho_qt(
        T, rho, q_t, p_ref, wp
    )
    return theta_li_sat - theta_li

  # Get a function for ∂f/∂T using the exact, autodiff-based gradient.
  # Each application of vmap applies over the first dimension of the input to
  # all arguments.
  vmap = jax.vmap
  if len(T1.shape) == 1:
    # Just for testing purposes: allow rank-1 inputs.
    df_dT = vmap(jax.grad(f, argnums=0))
  elif len(T1.shape) == 3:
    # For the typical case of rank-3 inputs, need to vmap for each dimension.
    df_dT = vmap(vmap(vmap(jax.grad(f, argnums=0))))
  else:
    raise ValueError(f'T1 must be 1D or 3D, got shape {T1.shape}')

  f_onearg = functools.partial(
      f, rho=rho, q_t=q_t, p_ref=p_ref, theta_li=theta_li
  )
  # Materialize p_ref into 3D if it is 1D, for vmap to work correctly.
  if len(p_ref.shape) == 3 and p_ref.shape[0] == 1 and p_ref.shape[1] == 1:
    # p_ref shape is (1, 1, nz).  theta_li shape is (nx, ny, nz).
    p_ref = p_ref * jnp.ones_like(theta_li)
  df_dT_onearg = functools.partial(
      df_dT, rho=rho, q_t=q_t, p_ref=p_ref, theta_li=theta_li
  )

  T_saturation = newton_root_finder(
      f_onearg, df_dT_onearg, T1, wp.max_newton_solver_iterations
  )

  # Decide between the cases:
  # If unsaturated (Case 1), T = T1
  #    q_l=0, q_i=0
  # If freezing (Case 2), T = T_freeze
  #    q_l=0, q_i=0, but discontinuous derivative so treat this specially?
  # Else (Case 3, not unsaturated, not right at the freezing temperature)
  #    q_c > 0

  # Start with unsaturated (case 1) vs saturated (case 3)
  T_eqb = jnp.where(q_t < q_v_sat, T1, T_saturation)

  # Now throw in case 2, the freezing point.
  # JP: there is probably a bug. The above calculation assumes that there is no
  # condensation (q_liq = q_ice = 0 used), but I'm not sure why we assume that.
  # However, this is what SwirlLM does, so let's keep it for now.
  # Since case 2 is almost never hit (only when
  # |theta_li_freeze - theta_li | < 1e-6), this probably doesn't matter anyway.

  T_eqb = jnp.where(
      jnp.abs(theta_li_freeze - theta_li) < 1e-6,
      wp.t_freeze,
      T_eqb,
  )
  return T_eqb
  # pylint: enable=invalid-name


def density_and_temperature_from_theta_li_q_t(
    theta_li: Array,
    q_t: Array,
    p_ref: Array,
    rho_initial_guess: Array,
    wp: WaterParams,
) -> tuple[Array, Array]:
  """Compute density and temperature from theta_li, q_t, and p_ref.

  Solve 2 equations for 2 unknowns (T and rho):
        p_ref = rho * R_m * T
        theta_li_eqb(T, rho, q_t) = theta_li

  This function follows the SwirlLM procedure for computing density and
  temperature.  There is an outer iteration (fixed-point iteration) over rho
  using rho = p_ref / (R_m T).  There is an inner iteration (Newton iteraitons)
  over T assuming rho is known, using `saturation_adjustment`.  Ten fixed-point
  iterations for rho are used.

  Args:
    theta_li: The liquid-ice potential temperature.
    q_t: The total specific humidity.
    p_ref: The reference hydrostatic pressure as a function of height.
    rho_initial_guess: The initial guess for the density used in the fixed-point
      iteration.
    wp: The water parameters.

  Returns:
    The thermodynamic density and temperature.
  """

  def density_update_fn(rho: Array) -> Array:
    # Compute the temperature assuming the density is known.
    T = saturation_adjustment(theta_li, rho, q_t, p_ref, wp)  # pylint: disable=invalid-name

    # Determine the amount of condensate.
    q_liq, q_ice = equilibrium_phase_partition(T, rho, q_t, wp)
    q_c = q_liq + q_ice

    # Get R_m, accounting for both moisture and the condensed phase.
    rm = (1 - q_t) * R_D + (q_t - q_c) * wp.r_v

    # Return the next iterate of density.
    return p_ref / (rm * T)

  def body_fn(i, rho):
    return density_update_fn(rho)

  rho = jax.lax.fori_loop(
      0, wp.num_density_iterations, body_fn, rho_initial_guess
  )

  # Recalculate temperature from final density.
  T = saturation_adjustment(theta_li, rho, q_t, p_ref, wp)  # pylint: disable=invalid-name
  return rho, T


# pylint: disable=invalid-name
def _rho_temperature_newton_solver(
    f2: Callable[[Array, Array], Array],
    grad_f2: Callable[[Array, Array], tuple[Array, Array]],
    rho_initial_guess: Array,
    T_initial_guess: Array,
    q_t: Array,
    p_ref: Array,
    wp: WaterParams,
):
  """Solve 2 equations for 2 unknowns (rho and T) with Newton solver."""

  def body_fn(j, rho_T):
    del j
    rho, T = rho_T

    # Need to calculate Rm
    # Determine the amount of saturation.
    q_liq, q_ice = equilibrium_phase_partition(T, rho, q_t, wp)
    q_c = q_liq + q_ice

    # Get R_m that accounts for both moisture and the condensed phase.
    Rm = (1 - q_t) * R_D + (q_t - q_c) * wp.r_v

    # Construct the Jacobian matrix of the 2x2 system for f1, f2.
    # Note: we neglect the derivatives of Rm with respect to rho, T. These
    # derivatives are very small, and this very tiny error in the Jacobian
    # should not prevent convergence of the Newton method to the desired root.
    # We could use autodiff to compute the exact derivatives, but it is more
    # expensive
    # and unnecessary.
    J11 = Rm * T  # ∂f1/∂ρ
    J12 = Rm * rho  # ∂f1/∂T
    J21, J22 = grad_f2(rho, T)

    determinant = J11 * J22 - J12 * J21
    Jinv_11 = J22 / determinant
    Jinv_12 = -J12 / determinant
    Jinv_21 = -J21 / determinant
    Jinv_22 = J11 / determinant
    f1val = rho * Rm * T - p_ref
    f2val = f2(rho, T)

    rho_new = rho - (Jinv_11 * f1val + Jinv_12 * f2val)
    T_new = T - (Jinv_21 * f1val + Jinv_22 * f2val)
    return rho_new, T_new

  max_iterations = wp.max_newton_solver_iterations
  rho, T = jax.lax.fori_loop(
      0, max_iterations, body_fn, (rho_initial_guess, T_initial_guess)
  )
  return rho, T
  # pylint: enable=invalid-name


def density_and_temperature_from_theta_li_q_t_single_iteration_loop(
    theta_li: Array,
    q_t: Array,
    p_ref: Array,
    rho_initial_guess: Array,
    wp: WaterParams,
) -> tuple[Array, Array]:
  """Compute density and temperature from theta_li, q_t, and p_ref.

  Solve 2 equations for 2 unknowns (T and rho):
        p_ref = rho * R_m * T
        theta_li_eqb(T, rho, q_t) = theta_li

  Use Newton solve of the system of 2 equations in 2 unknowns, rho, T.
        R_m * rho * T - p_ref = 0
        theta_li_eqb(T, rho, q_t) - theta_li = 0

  Let
        f1(rho, T) = R_m * rho * T - p_ref
        f2(rho, T) = theta_li_eqb(T, rho, q_t) - theta_li

  * Initial guess for T: use unsaturated conditions, with R_m = R_D.
  * The partial derivatives of f1 are easy.

  Args:
    theta_li: The liquid-ice potential temperature.
    q_t: The total specific humidity.
    p_ref: The reference hydrostatic pressure as a function of height.
    rho_initial_guess: The initial guess for the density used in the fixed-point
      iteration.
    wp: The water parameters.

  Returns:
    The thermodynamic density and temperature.
  """
  # pylint: disable=invalid-name
  # Compute the temperature assuming the air is unsaturated.  This T will serve
  # as the initial guess for the Newton iterations.
  R_m_unsat = (1 - q_t) * R_D + q_t * wp.r_v
  cp_m_unsat = (1 - q_t) * CP_D + q_t * wp.cp_v
  exner = (p_ref / wp.exner_reference_pressure) ** (R_m_unsat / cp_m_unsat)
  T1 = exner * theta_li
  T1 = jnp.maximum(wp.t_min, T1)  # Apply a cutoff for numerical reasons.

  # Define f2(rho, T) = theta_li_eqb(rho, T) - theta_li
  def f2(
      rho: Array, T: Array, q_t: Array, p_ref: Array, theta_li: Array
  ) -> Array:
    theta_li_sat = theta_li_from_temperature_rho_qt(T, rho, q_t, p_ref, wp)
    return theta_li_sat - theta_li

  # Get a function for ∇f2 = (∂f2/∂ρ, ∂f2/∂T) using the exact, autodiff-based
  # gradient.  Each application of vmap applies over the first dimension of the
  # input to all arguments.
  vmap = jax.vmap
  if len(T1.shape) == 1:
    # Just for testing purposes: allow rank-1 inputs.
    grad_f2 = vmap(jax.grad(f2, argnums=[0, 1]))
  elif len(T1.shape) == 3:
    # For the typical case of rank-3 inputs, need to vmap for each dimension.
    grad_f2 = vmap(vmap(vmap(jax.grad(f2, argnums=[0, 1]))))
  else:
    raise ValueError(f'T1 must be 1D or 3D, got shape {T1.shape}')

  f2_twoargs = functools.partial(f2, q_t=q_t, p_ref=p_ref, theta_li=theta_li)
  # Materialize p_ref into 3D if it is 1D, for vmap to work correctly.
  if len(p_ref.shape) == 3 and p_ref.shape[0] == 1 and p_ref.shape[1] == 1:
    # p_ref shape is (1, 1, nz).  theta_li shape is (nx, ny, nz).
    p_ref = p_ref * jnp.ones_like(theta_li)
  grad_f2_twoargs = functools.partial(
      grad_f2, q_t=q_t, p_ref=p_ref, theta_li=theta_li
  )
  rho, T_saturation = _rho_temperature_newton_solver(
      f2_twoargs, grad_f2_twoargs, rho_initial_guess, T1, q_t, p_ref, wp
  )

  # Handle the freezing case.
  # Compute theta_li at the freezing point for unsaturated (assuming q_c=0)
  theta_li_freeze = wp.t_freeze / exner
  # If at the freezing point, use the freezing point temperature and density.
  T_eqb = jnp.where(
      jnp.abs(theta_li_freeze - theta_li) < 1e-6,
      wp.t_freeze,
      T_saturation,
  )
  rho_eqb = jnp.where(
      jnp.abs(theta_li_freeze - theta_li) < 1e-6,
      p_ref / (R_m_unsat * T_eqb),
      rho,
  )

  return rho_eqb, T_eqb
  # pylint: enable=invalid-name


def compute_thermodynamic_fields_from_prognostic_fields(
    theta_li: Array,
    q_t: Array,
    p_ref: Array,
    rho_initial_guess: Array,
    wp: WaterParams,
) -> ThermoFields:
  """Compute thermodynamic fields from the prognostic variables."""
  # pylint: disable=invalid-name
  if wp.single_iteration_loop:
    with jax.named_scope(
        'thermodynamic_fields_from_prognostic_fields Single Loop'
    ):
      rho_thermal, T = (
          density_and_temperature_from_theta_li_q_t_single_iteration_loop(
              theta_li, q_t, p_ref, rho_initial_guess, wp
          )
      )
  else:
    with jax.named_scope(
        'thermodynamic_fields_from_prognostic_fields double loop'
    ):
      rho_thermal, T = density_and_temperature_from_theta_li_q_t(
          theta_li, q_t, p_ref, rho_initial_guess, wp
      )

  # Get the other thermodynamic fields and return all of them.
  q_liq, q_ice = equilibrium_phase_partition(T, rho_thermal, q_t, wp)
  q_c = q_liq + q_ice
  q_v = q_t - q_c
  q_v_sat = saturation_vapor_humidity(T, rho_thermal, wp)
  # pylint: enable=invalid-name
  return ThermoFields(T, rho_thermal, q_liq, q_ice, q_c, q_v, q_v_sat, wp)
