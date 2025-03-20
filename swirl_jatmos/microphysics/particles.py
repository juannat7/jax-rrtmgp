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

"""Utilities and wrappers for parameters of rain, snow, and ice particles."""

from typing import TypeAlias

import jax
import jax.numpy as jnp
from scipy import special
from swirl_jatmos.microphysics import microphysics_config

Array: TypeAlias = jax.Array

IceParams: TypeAlias = microphysics_config.IceParams
RainParams: TypeAlias = microphysics_config.RainParams
SnowParams: TypeAlias = microphysics_config.SnowParams


def n_0(
    pp: RainParams | SnowParams | IceParams,
    rho: Array | None = None,
    q_s: Array | None = None,
) -> Array:
  """Computes the Marshall-Palmer distribution parameter [m^-4]."""
  if isinstance(pp, SnowParams):
    assert q_s is not None, 'q_s is required for Snow, but None was provided.'
    assert rho is not None, 'rho is required for Snow, but None was provided.'
    # Remove negative values here, otherwise the power causes NaNs.
    q_s = jnp.clip(q_s, 0.0, None)
    return pp.mu * (rho * q_s / microphysics_config.RHO_AIR) ** pp.nu
  elif isinstance(pp, IceParams):
    return jnp.array(2e7)
  elif isinstance(pp, RainParams):
    return jnp.array(1.6e7)
  else:
    raise ValueError(
        f'One of Snow, Ice, or Rain is required but {type(pp)} was provided.'
    )


def marshall_palmer_distribution_parameter_lambda_inverse(
    pp: RainParams | SnowParams | IceParams,
    rho: Array,
    q: Array,
) -> Array:
  """Computes 1/λ in the Marshall-Palmer distribution parameter.

  Note that λ goes to infinity as q goes to 0, so computing the inverse is
  better behaved numerically.

  Args:
    pp: Particle parameters, which is a RainParams, SnowParams, or IceParams.
    rho: The density of the moist air [kg/m^3].
    q: The water mass fraction [kg/kg], where q = q_r, q_s, or q_ice.

  Returns:
    The reciprocal of the lambda parameter in the Marshall-Palmer distribution.
  """
  r_0 = pp.r_0
  m_0 = pp.m_0
  m_e = pp.m_e
  del_m = pp.del_m
  chi_m = pp.chi_m
  n_0_ = n_0(pp, rho, q)
  m = m_e + del_m + 1.0

  gamma = float(special.gamma(m))
  lambda_inverse = (
      rho * q * r_0 ** (m_e + del_m) / (chi_m * m_0 * n_0_ * gamma)
  ) ** (1.0 / m)

  # Note that for snow, n_0 is 0 when q is 0, so we have 0/0.  However,
  # n_0 ~ q^nu, with nu_snow = 0.63, so lambda_inverse ~ q / n_0 -> 0 as q -> 0.
  # Furthermore, if q < 1e-35 or so, then lambda_inverse has NaN when it should
  # be very small.  Therefore, set lambda_inverse to 0 when q is smaller than
  # some small value.
  lambda_inverse = jnp.where(q > 1e-31, lambda_inverse, 0.0)

  return lambda_inverse
