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

"""Convection config."""

import dataclasses
from typing import Literal, TypeAlias

import dataclasses_json  # Used for JSON serialization.

MomentumScheme: TypeAlias = Literal[
    'quick', 'upwind1', 'weno3', 'weno5_js', 'weno5_z'
]
ScalarScheme: TypeAlias = Literal[
    'quick', 'upwind1', 'van_leer_upwind2', 'weno3', 'weno5_js', 'weno5_z'
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConvectionConfig(dataclasses_json.DataClassJsonMixin):
  """Parameters determining the convection config."""

  momentum_scheme: MomentumScheme = 'quick'
  theta_li_scheme: ScalarScheme = 'quick'
  q_t_scheme: ScalarScheme = 'van_leer_upwind2'
  q_r_scheme: ScalarScheme = 'van_leer_upwind2'
  q_s_scheme: ScalarScheme = 'van_leer_upwind2'

  def __post_init__(self):
    allowed_momentum_schemes = [
        'quick',
        'upwind1',
        'weno3',
        'weno5_js',
        'weno5_z',
    ]
    if self.momentum_scheme not in allowed_momentum_schemes:
      raise ValueError(
          f'Unsupported momentum scheme: {self.momentum_scheme}.  Must be one'
          f' of {allowed_momentum_schemes}.'
      )

    allowed_scalar_schemes = [
        'quick',
        'upwind1',
        'van_leer_upwind2',
        'weno3',
        'weno5_js',
        'weno5_z',
    ]
    scalar_names = ['theta_li', 'q_t', 'q_r', 'q_s']
    scalar_schemes = [
        self.theta_li_scheme,
        self.q_t_scheme,
        self.q_r_scheme,
        self.q_s_scheme,
    ]
    for name, scheme in zip(scalar_names, scalar_schemes):
      if scheme not in allowed_scalar_schemes:
        raise ValueError(
            f'Unsupported {name} scheme: {scheme}.  Must be one of'
            f' {allowed_scalar_schemes}.'
        )
