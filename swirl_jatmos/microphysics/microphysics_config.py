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

"""Configuration for the microphysics."""

import dataclasses
import enum
from typing import TypeAlias

import dataclasses_json  # Used for JSON serialization.
import numpy as np
from scipy import special
from swirl_jatmos.microphysics import terminal_velocity_chen2022_config

IceTableB3Coeffs: TypeAlias = terminal_velocity_chen2022_config.IceTableB3Coeffs
SnowTableB5Coeffs: TypeAlias = (
    terminal_velocity_chen2022_config.SnowTableB5Coeffs
)

# Some densities in kg/m^3.
RHO_ICE = 500.0  # Apparent density of ice (actual ice density ~917).
RHO_AIR = 1.0
RHO_WATER = 1e3

DROPLET_N = 1e8  # The number of cloud droplets per cubic meter.


@dataclasses.dataclass(frozen=True, kw_only=True)
class IceParams:
  """Constants for ice related quantities."""

  # The apparent density of an ice crystal [kg/m^3].
  rho: float = RHO_ICE

  # Typical ice crystal radius [m].
  r_0: float = 1e-5

  # Unit mass of an ice crystal [kg].
  m_0: float = dataclasses.field(init=False)

  # Exponent to the radius ratio in the mass equation.
  m_e: float = 3.0
  # The calibration coefficients in the mass equation.
  chi_m: float = 1.0
  del_m: float = 0.0
  # The `n_0` parameter in the Marshall-Palmer distribution.
  n_0: float = 2e7

  def __post_init__(self):
    object.__setattr__(self, 'm_0', 4.0 / 3.0 * np.pi * self.rho * self.r_0**3)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RainParams:
  """Constants for rain related quantities."""

  # The density of a rain drop [kg/m^3].
  rho: float = RHO_WATER

  # The drag coefficient of a rain drop [1].
  c_d: float = 0.55

  # Typical rain drop radius [m].
  r_0: float = 1e-3

  # Unit mass of a rain drop [kg].
  m_0: float = dataclasses.field(init=False)

  # Exponent to the radius ratio in the mass equation.
  m_e: float = 3.0
  # The calibration coefficients in the mass equation.
  chi_m: float = 1.0
  del_m: float = 0.0

  # Unit cross section area of a rain drop [m^2].
  a_0: float = dataclasses.field(init=False)
  # Exponent to the radius ratio in the cross section area equation.
  a_e: float = 2.0
  # The calibration coefficients in the cross section area equation.
  chi_a: float = 1.0
  del_a: float = 0.0

  # Exponent to the radius ratio in the terminal velocity equation.
  v_e: float = 0.5
  # The calibration coefficients in the terminal velocity equation.
  chi_v: float = 1.0
  del_v: float = 0.0

  # The ventilation factor coefficients [1].
  a_vent: float = 1.5
  b_vent: float = 0.53
  # The `n_0` parameter in the Marshall-Palmer distribution.
  n_0: float = 1.6e7

  def __post_init__(self):
    object.__setattr__(self, 'm_0', 4.0 / 3.0 * np.pi * self.rho * self.r_0**3)
    object.__setattr__(self, 'a_0', np.pi * self.r_0**2)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SnowParams:
  """Constants for snow related quantities."""

  # The apparent density [kg/m^3] of a falling snow crystal for calculating the
  # terminal velocity.  The apparent density is defined as the mass of the snow
  # particle divided by the volume of a circumscribing spheroid, and its value
  # can range from 50 to 900 kg/m^3.  In the atmosphere, a typical value is
  # around 100 kg/m^3.
  rho: float = 100.0

  # Typical snow crystal radius [m].
  r_0: float = 1e-3

  # Unit mass of a snow crystal [kg] [1].
  m_0: float = dataclasses.field(init=False)
  # Exponent to the radius ratio in the mass equation [1].
  m_e: float = 2.0
  # The calibration coefficients in the mass equation.
  chi_m: float = 1.0
  del_m: float = 0.0

  # Unit cross section area of a snow crystal [m^2] [1].
  a_0: float = dataclasses.field(init=False)
  # Exponent to the radius ratio in the cross section area equation.
  a_e: float = 2.0
  # The calibration coefficients in the cross section area equation.
  chi_a: float = 1.0
  del_a: float = 0.0

  # Exponent to the radius ratio in the terminal velocity equation [1].
  v_e: float = 0.25
  # The calibration coefficients in the terminal velocity equation.
  chi_v: float = 1.0
  del_v: float = 0.0

  # The snow size distribution parameter exponent [3].
  nu: float = 0.63
  # The snow size distribution parameter coefficient [m^-4] [3].
  mu: float = 4.36e9 * RHO_AIR**nu

  # The ventilation factor coefficients [7].
  a_vent: float = 0.65
  b_vent: float = 0.44

  # The exponent of the radius in the aspect ratio approximation.
  alpha: float = dataclasses.field(init=False)

  # The constant factor of the aspect ratio (independent of radius).
  phi_0: float = dataclasses.field(init=False)

  def __post_init__(self):
    object.__setattr__(self, 'm_0', 0.1 * self.r_0**2)
    object.__setattr__(self, 'a_0', 0.3 * np.pi * self.r_0**2)
    object.__setattr__(
        self, 'alpha', self.m_e + self.del_m - 1.5 * (self.a_e + self.del_a)
    )

    # 3rd-order moment for volume- or mass-weighted average.
    k = 3
    # Compute the scale factor of the mass-weighted average of the aspect ratio.
    phi_0 = float(
        special.gamma(self.alpha + k + 1)
        * 3.0
        * np.sqrt(np.pi)
        / (special.gamma(k + 1) * 4.0 * self.rho)
        * self.chi_m
        * self.m_0
        / (self.chi_a * self.a_0) ** (3 / 2)
        / (2.0 * self.r_0) ** self.alpha
    )
    object.__setattr__(self, 'phi_0', phi_0)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AutoconversionParams:
  """Constant coefficients in autoconversion."""

  # Timescale for cloud liquid to rain water autoconversion [s] [1].
  tau_lr: float = 1e3
  # Timescale for cloud ice to snow autoconversion [s].
  tau_is: float = 1e2
  # Threshold for cloud liquid to rain water autoconversion [1].
  q_l_threshold: float = 5e-4
  # Threshold for cloud ice to snow autoconversion.
  q_i_threshold: float = 1e-6
  # Threshold particle radius between ice and snow [m] [4].
  r_is: float = 6.25e-5


@dataclasses.dataclass(frozen=True, kw_only=True)
class AccretionParams:
  """Collision efficiencies in accretion."""

  # Collision efficiency between cloud liquid and rain [1].
  e_liq_rain: float = 0.8
  # Collision efficiency between cloud liquid and snow [5].
  e_liq_snow: float = 0.1
  # Collision efficiency between cloud ice and rain [5].
  e_ice_rain: float = 1.0
  # Collision efficiency between cloud ice and snow [6].
  e_ice_snow: float = 0.1
  # Collision efficiency between rain and snow [6].
  e_rain_snow: float = 1.0


class TerminalVelocityMethod(enum.Enum):
  """An enum defining the methodology used for the terminal velocities."""

  POWER_LAW = 1  # Use simple power-law based velocity expressions.
  CHEN_2022 = 2  # Use terminal velocity from Chen et al. (2022).


@dataclasses.dataclass(frozen=True, kw_only=True)
class MicrophysicsConfig(dataclasses_json.DataClassJsonMixin):
  """Parameters for one-moment microphysics."""

  rain_params: RainParams = RainParams()
  snow_params: SnowParams = SnowParams()
  ice_params: IceParams = IceParams()
  autoconversion_params: AutoconversionParams = AutoconversionParams()
  accretion_params: AccretionParams = AccretionParams()
  terminal_velocity_method: TerminalVelocityMethod = (
      TerminalVelocityMethod.POWER_LAW
  )
  k_coeff: float = 0.5  # Coefficient used for clipping source terms.
  # Sedimentation method (1 or 2 is recommended).
  # 0: Combine all terminal velocities with fluid velocities for q_t, q_r, q_s.
  # 1: Use upwind1 for terminal velocity fluxes for q_t, q_r, q_s.
  # 2: Use upwind1 for q_t terminal velocity flux and combine velocities for
  #      q_r, q_s.
  sedimentation_method: int = 1
  # The following 2 fields are only used if `terminal_velocity_method` is
  # `TerminalVelocityMethod.CHEN_2022`.
  ice_table_b3_coeffs: IceTableB3Coeffs = dataclasses.field(init=False)
  snow_table_b5_coeffs: SnowTableB5Coeffs = dataclasses.field(init=False)

  def __post_init__(self):
    ice_table_b3_coeffs = (
        terminal_velocity_chen2022_config.compute_ice_table_b3_coeffs(
            self.ice_params.rho
        )
    )
    snow_table_b5_coeffs = (
        terminal_velocity_chen2022_config.compute_snow_table_b5_coeffs(
            self.snow_params.rho
        )
    )
    object.__setattr__(self, 'ice_table_b3_coeffs', ice_table_b3_coeffs)
    object.__setattr__(self, 'snow_table_b5_coeffs', snow_table_b5_coeffs)
