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

import dataclasses
from typing import TypeAlias

import jax
import jax.numpy as jnp
from scipy import special
from swirl_jatmos.microphysics import microphysics_config
from swirl_jatmos.microphysics import particles
from swirl_jatmos.microphysics import terminal_velocity_chen2022_config

Array: TypeAlias = jax.Array
# Eventually remove these, just use one microphysics config input that allows
# non-default input.
RainParams: TypeAlias = microphysics_config.RainParams
SnowParams: TypeAlias = microphysics_config.SnowParams
IceParams: TypeAlias = microphysics_config.IceParams
IceTableB3Coeffs: TypeAlias = terminal_velocity_chen2022_config.IceTableB3Coeffs
SnowTableB5Coeffs: TypeAlias = (
    terminal_velocity_chen2022_config.SnowTableB5Coeffs
)

# pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class TerminalVelocityCoefficients:
  """Coefficients for the gamma-type terminal velocity formulas.

  The a_i, b_i, c_i appear in Chen et al. (2022) formulas.  For rain, i=1,2,3,
  and for snow & ice, i=1,2.
  """

  a: tuple[Array, ...]
  b: tuple[Array, ...]
  c: tuple[Array, ...]


def _convert_coefficients_to_si_units_and_wrap(
    a: tuple[Array, ...], b: tuple[Array, ...], c: tuple[Array, ...]
) -> TerminalVelocityCoefficients:
  """Convert the coefficients to SI units and puts them in a dataclass."""
  # Convert a from mm^-b to m^-b.
  a = tuple(a_i * 1e3**b_i for a_i, b_i in zip(a, b))
  # Convert c from mm^-1 to m^-1.
  c = tuple(c_i * 1e3 for c_i in c)
  return TerminalVelocityCoefficients(a, b, c)


def _compute_raindrop_coefficients(rho: Array):
  """Compute raindrop coefficients as a function of air density (Table B1).

  Args:
    rho: The density of the air [kg/m^3].

  Returns:
    The coefficients for the raindrop terminal velocity.
  """

  q = jnp.exp(0.115231 * rho)
  # The a_i, b_i, c_i coefficeints for i=1,2,3.
  a = (0.044612 * q, -0.263166 * q, 4.7178 * q * rho**-0.47335)  # a_1, a_2, a_3
  b = (
      2.2955 - 0.038465 * rho,
      2.2955 - 0.038465 * rho,
      1.1451 - 0.038465 * rho,
  )
  array = lambda x: jnp.array(x, dtype=rho.dtype)
  c = (array(0), array(0.184325), array(0.184325))
  return _convert_coefficients_to_si_units_and_wrap(a, b, c)


def _compute_ice_coefficients(
    rho: Array, ice_table_b3_coeffs: IceTableB3Coeffs
) -> TerminalVelocityCoefficients:
  """Compute ice coefficients as a function of air density (Table B2)."""
  A_s, B_s = ice_table_b3_coeffs.A_s, ice_table_b3_coeffs.B_s
  C_s, E_s = ice_table_b3_coeffs.C_s, ice_table_b3_coeffs.E_s
  F_s, G_s = ice_table_b3_coeffs.F_s, ice_table_b3_coeffs.G_s

  # The a_i, b_i, c_i coefficients for i=1,2.
  a = (E_s * rho**A_s, F_s * rho**A_s)
  b = (B_s + C_s * rho, B_s + C_s * rho)
  array = lambda x: jnp.array(x, dtype=rho.dtype)
  c = (array(0), array(G_s))
  # check sharding on c?
  return _convert_coefficients_to_si_units_and_wrap(a, b, c)


def _compute_snow_coefficients(
    rho: Array, snow_table_b5_coeffs: SnowTableB5Coeffs
) -> TerminalVelocityCoefficients:
  """Compute snow coefficients as a function of air density (Table B4)."""
  A_L, B_L = snow_table_b5_coeffs.A_L, snow_table_b5_coeffs.B_L
  C_L, E_L = snow_table_b5_coeffs.C_L, snow_table_b5_coeffs.E_L
  F_L, G_L = snow_table_b5_coeffs.F_L, snow_table_b5_coeffs.G_L
  H_L = snow_table_b5_coeffs.H_L

  # The a_i, b_i, c_i coefficients for i=1,2.
  a = (B_L * rho**A_L, E_L * rho**A_L * jnp.exp(H_L * rho))
  array = lambda x: jnp.array(x, dtype=rho.dtype)
  b = (array(C_L), array(F_L))
  c = (array(0), array(G_L))
  return _convert_coefficients_to_si_units_and_wrap(a, b, c)


def _bulk_terminal_velocity(
    coeffs: TerminalVelocityCoefficients, lambda_inv: Array
) -> Array:
  """Compute the mass-weighted bulk terminal velocity.

  Calculate Eq. (20) of Chen et al. (2022), which corresponds to an integration
  of a gamma-type velocity distribution over the particle size spectrum.  Here,
  we assume the particle size follows the Marshall-Palmer distribution.  Since
  the size distribution is exponential, we set `mu` to 0; and since this is a
  mass-weighted average we fix the moment `k` to 3.

  Args:
    coeffs: Holder for the terminal velocity coefficients `a`, `b`, and `c` of
      the gamma-type function in Eq. (19) of Chen et al. (2022).
    lambda_inv: 1/λ, where λ is the Marshall-Palmer distribution rate parameter.

  Returns:
    The mass-weighted bulk terminal velocity for a particle group.  Note that
    this result should still be scaled by the volume-weighted mean aspect ratio
    of the particle group.
  """
  # Exponential particle size distribution implies mu = 0 (equation 2).
  # The volume-weighted, or mass-weighted, fall speed corresponds to the third
  # moment (equation 3), k = 3.  Then delta = mu + k + 1 = 4.
  delta = 4
  gamma_delta = float(special.gamma(delta))  # This is just Γ(4) = 6.0.

  def compute_one_term(a_i: Array, b_i: Array, c_i: Array) -> Array:
    lambda_independent_factor = (
        a_i * jax.scipy.special.gamma(b_i + delta) / gamma_delta
    )
    lambda_dependent_factor = lambda_inv**b_i / (1 + c_i * lambda_inv) ** (
        b_i + delta
    )
    return lambda_independent_factor * lambda_dependent_factor

  return sum(
      compute_one_term(a_i, b_i, c_i)
      for a_i, b_i, c_i in zip(coeffs.a, coeffs.b, coeffs.c)
  )


def _individual_terminal_velocity(
    coeffs: TerminalVelocityCoefficients, diameter: Array
) -> Array:
  """Compute the terminal velocity of a single particle.

  This evaluates the multi-term gamma-type function in Eq. (19) of Chen et al.
  (2022) excluding the aspect ratio factor, which must be computed separately.

  Args:
    coeffs: A wrapper for the terminal velocity coefficients `a`, `b`, and `c`
      of the gamma-type function in Eq. (19).
    diameter: Pointwise estimated group droplet diameter.

  Returns:
    The mass-weighted terminal velocity for an individual particle.  Note that
    this result should still be scaled by the volume-weighted aspect ratio.
  """

  def compute_one_term(a_i: Array, b_i: Array, c_i: Array) -> Array:
    return a_i * diameter**b_i * jnp.exp(-c_i * diameter)

  return sum(
      compute_one_term(a_i, b_i, c_i)
      for a_i, b_i, c_i in zip(coeffs.a, coeffs.b, coeffs.c)
  )


def rain_terminal_velocity(rho: Array, q_r: Array, rp: RainParams) -> Array:
  """Compute the bulk terminal velocity of rain.

  Args:
    rho: The density of air [kg/m^3].
    q_r: The rain mass fraction [kg/kg].
    rp: A RainParams object.

  Returns:
    The bulk terminal velocity of rain drops [m/s].
  """
  lambda_inv = particles.marshall_palmer_distribution_parameter_lambda_inverse(
      rp, rho, q_r
  )
  coeffs = _compute_raindrop_coefficients(rho)
  v_terminal = _bulk_terminal_velocity(coeffs, lambda_inv)
  # Handle the special case when q_r==0, v_terminal ends up large due to the
  # specific way the computation is handled numerically, because lambda should
  # be infinity when q=0 but it is not.  v_terminal should be 0 when q_r = 0.
  v_terminal = jnp.where(q_r > 0, v_terminal, 0)
  return jnp.clip(v_terminal, 0.0, None)


def snow_terminal_velocity(
    rho: Array,
    q_s: Array,
    sp: SnowParams,
    snow_table_b5_coeffs: SnowTableB5Coeffs,
) -> Array:
  """Compute the bulk terminal velocity of snow.

  Args:
    rho: The density of air [kg/m^3].
    q_s: The snow mass fraction [kg/kg].
    sp: A SnowParams object.
    snow_table_b5_coeffs: The coefficients for the snow terminal velocity
      formulas.

  Returns:
    The bulk terminal velocity of snow [m/s].
  """
  # The exponent applied to the mass-weighted aspect ratio of snow (ϕ) in the
  # bulk fall speed equation. We assume snow particles are shaped like oblate
  # spheroids, which corresponds to κ = 1/3 as derived from the relationship
  # between the volume-equivalent diameter and the axial half-lengths of the
  # spheroid.
  kappa = 1 / 3

  lambda_inv = particles.marshall_palmer_distribution_parameter_lambda_inverse(
      sp, rho, q_s
  )

  # Mass-weighted aspect ratio.
  psi_avg = sp.phi_0 * lambda_inv**sp.alpha
  # Handle a NaN that occurs when lambda_inv=0 and sp.alpha=-1.
  psi_avg = jnp.where(lambda_inv > 0, psi_avg, 0.0)

  coeffs = _compute_snow_coefficients(rho, snow_table_b5_coeffs)
  v_terminal = psi_avg**kappa * _bulk_terminal_velocity(coeffs, lambda_inv)
  return jnp.clip(v_terminal, 0.0, None)


def liquid_condensate_terminal_velocity(
    rho: Array,
    q_liq: Array,
    rp: RainParams,
) -> Array:
  """Compute the sedimentation terminal velocity of cloud liquid droplets.

  This function uses the individual fall speed, not the bulk fall speed
  integrated over all particle diameters.

  Args:
    rho: The density of air [kg/m^3].
    q_liq: The cloud liquid droplet condensate mass fraction [kg/kg].
    rp: A RainParams object.

  Returns:
    The sedimentation terminal velocity of cloud liquid droplets [m/s].
  """
  abc_coeffs = _compute_raindrop_coefficients(rho)
  # The correction factor applied to the terminal velocity of cloud droplets.
  # This is needed because we use the same coefficients for cloud droplets that
  # we use for rain, but those are only directly applicable to particle
  # diameters greater than 100 μm.
  correction_factor = 0.1
  q_liq = jnp.clip(q_liq, 0.0, None)
  diameter = (rho * q_liq / microphysics_config.DROPLET_N / rp.rho) ** (1 / 3)
  terminal_velocity = _individual_terminal_velocity(abc_coeffs, diameter)
  return jnp.clip(correction_factor * terminal_velocity, 0.0, None)


def ice_condensate_terminal_velocity(
    rho: Array,
    q_ice: Array,
    ip: IceParams,
    ice_table_b3_coeffs: IceTableB3Coeffs,
) -> Array:
  """Compute the sedimentation terminal velocity of cloud ice.

  This function uses the individual fall speed, not the bulk fall speed
  integrated over all particle diameters.

  Args:
    rho: The density of air [kg/m^3].
    q_ice: The cloud ice condensate mass fraction [kg/kg].
    ip: An IceParams object.
    ice_table_b3_coeffs: The coefficients for the ice terminal velocity
      formulas.

  Returns:
    The sedimentation terminal velocity of cloud ice [m/s].
  """
  abc_coeffs = _compute_ice_coefficients(rho, ice_table_b3_coeffs)
  q_ice = jnp.clip(q_ice, 0.0, None)
  diameter = (rho * q_ice / microphysics_config.DROPLET_N / ip.rho) ** (1 / 3)
  terminal_velocity = _individual_terminal_velocity(abc_coeffs, diameter)
  return jnp.clip(terminal_velocity, 0.0, None)


# pylint: enable=invalid-name
