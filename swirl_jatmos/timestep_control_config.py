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

"""Timestep control config."""

import dataclasses


@dataclasses.dataclass(frozen=True, kw_only=True)
class TimestepControlConfig:
  """Parameters determining the adaptive timestep control."""

  disable_adaptive_timestep: bool = False
  desired_cfl: float = 0.7  # Target CFL number.
  max_dt: float = 25.0  # Maximum timestep [s].
  min_dt: float = 1e-4  # Minimum timestep [s].
  max_change_factor: float = 1.1  # Maximum amount to change dt by.
  min_change_factor: float = 0.5  # Minimum amount to change dt by.
  update_interval_steps: int = 5  # Number of steps between timestep updates.

  def __post_init__(self):
    # Perform validation.
    if self.max_change_factor <= 1 or self.min_change_factor >= 1:
      raise ValueError(
          'max_change_factor must be greater than 1 and min_change_factor must'
          ' be less than 1.'
      )
