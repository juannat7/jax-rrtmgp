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

"""Sponge configuration."""

import dataclasses


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpongeConfig:
  """Configuration for applying sponge.

  Sponge is applied to the velocity fields at the domain top.
  """
  coeff: float = 6.0
  # Fraction of the domain height to which the sponge is applied.
  sponge_fraction: float = 0.25
  c2: float = 0.5
