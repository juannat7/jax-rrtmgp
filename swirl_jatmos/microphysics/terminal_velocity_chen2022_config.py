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

"""Configuration for the Chen terminal velocity method."""

import dataclasses

import numpy as np

# pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True, kw_only=True)
class IceTableB3Coeffs:
  """Coefficients from table B3 of Chen et al. (2022)."""

  A_s: float
  B_s: float
  C_s: float
  E_s: float
  F_s: float
  G_s: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class SnowTableB5Coeffs:
  """Coefficients from table B5 of Chen et al. (2022)."""

  A_L: float
  B_L: float
  C_L: float
  E_L: float
  F_L: float
  G_L: float
  H_L: float


def compute_ice_table_b3_coeffs(
    rho: float,
) -> IceTableB3Coeffs:
  """Compute coefficient formulas from table B3 of Chen et al. (2022).

  Args:
    rho: The apparent density of ice [kg/m^3].

  Returns:
    The coefficients for the terminal velocity of single ice particles.
  """
  A_s = -0.263503 + 0.00174079 * np.log(rho) ** 2 - 0.0378769 * np.log(rho)
  B_s = (0.575231 + 0.0909307 * np.log(rho) + 0.515579 / np.sqrt(rho)) ** -1
  C_s = (
      -0.345387
      + 0.177362 * np.exp(-0.000427794 * rho)
      + 0.00419647 * np.sqrt(rho)
  )
  E_s = -0.156593 - 0.0189334 * np.log(rho) ** 2 + 0.1377817 * np.sqrt(rho)
  F_s = -np.exp(
      -3.35641 - 0.0156199 * np.log(rho) ** 2 + 0.765337 * np.log(rho)
  )
  G_s = (
      -0.0309715 + 1.55054 / np.log(rho) - 0.518349 * np.log(rho) / rho
  ) ** -1
  # Convert from numpy scalars to floats so that downstream JAX array dtypes are
  # not overridden.
  return IceTableB3Coeffs(
      A_s=float(A_s),
      B_s=float(B_s),
      C_s=float(C_s),
      E_s=float(E_s),
      F_s=float(F_s),
      G_s=float(G_s),
  )


def compute_snow_table_b5_coeffs(
    rho: float,
) -> SnowTableB5Coeffs:
  """Compute coefficient formulas from table B5 of Chen et al. (2022).

  Args:
    rho: The apparent density of snow [kg/m^3].

  Returns:
    The coefficients for the terminal velocity of single snow particles.
  """
  A_L = -0.475897 - 0.00231270 * np.log(rho) + 1.12293 * rho ** (-3 / 2)
  B_L = np.exp(
      -2.56289 - 0.00513504 * np.log(rho) ** 2 + 0.608459 * np.log(rho)
  )
  C_L = np.exp(-0.756064 + 0.935922 / np.log(rho) - 1.70952 / rho)
  E_L = (
      0.00639847
      + 0.00906454 * np.log(rho) * np.sqrt(rho)
      - 0.108232 * np.sqrt(rho)
  )
  # Slightly rewrote F_L to convert 10^19 * exp(-rho_i) into
  # exp(19 * log(10) - rho_i), which is more numerically stable.
  F_L = (
      0.515453
      - 0.0725042 * np.log(rho)
      - 1.86810 * np.exp(43.749116766887 - rho)
  )
  G_L = (
      2.65236 + 0.00158269 * np.log(rho) * np.sqrt(rho) + 259.935 / np.sqrt(rho)
  ) ** -1
  # Slightly rewrote H_L to convert 10^20 * exp(-rho_i) into
  # exp(20 * log(10) - rho_i), which is more numerically stable.
  H_L = (
      -0.346044
      - 7.17829e-11 * rho ** (5 / 2)
      - 1.24394 * np.exp(46.0517018599 - rho)
  )
  # Convert from numpy scalars to floats so that downstream JAX array dtypes are
  # not overridden.
  return SnowTableB5Coeffs(
      A_L=float(A_L),
      B_L=float(B_L),
      C_L=float(C_L),
      E_L=float(E_L),
      F_L=float(F_L),
      G_L=float(G_L),
      H_L=float(H_L),
  )


# pylint: enable=invalid-name
