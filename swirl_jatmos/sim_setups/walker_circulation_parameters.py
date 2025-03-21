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

"""Parameters for the Mock-Walker Circulation simulation."""

import dataclasses

import dataclasses_json
from etils import epath
import jax


@dataclasses.dataclass(frozen=True, kw_only=True)
class WalkerCirculationParameters(dataclasses_json.DataClassJsonMixin):
  """Defines parameters for the Walker Circulation simulation."""

  # The analytic sounding parameters
  z_t: float = 15e3  # Height of tropopause, used in initialization.
  z_q1: float = 4e3  # Height used in initialization.
  z_q2: float = 7.5e3  # Height used in initialization.
  q_0: float = 0.01865  # 0.012, 0.01865, 0.024 for T=295, 300, 305 K. [kg/kg]
  q_t: float = 1e-14  # Initial value for q_t above the tropopause.
  gamma: float = 6.7e-3  # Lapse rate, used in initialization.
  sst_0: float = 300.0  # Sea-surface temperature, average value [K].
  delta_sst: float = 0  # Amplitude of variation in sea-surface temperature [K].
  # Amplitude scaling factor of the initial perturbation.
  theta_li_pert_scaling: float = 1.0
  # If nonempty, use sounding for initial theta_li(z) and q_t(z) from specified
  # directory with filename 'theta_li_qt_sounding.csv', instead of the analytic.
  # profiles. The analytic profiles are still used for p_ref(z) and rho_ref(z).
  sounding_dirname: str = ''


def _should_write_file() -> bool:
  """Returns True if this process should write a file."""
  # For multihost, ensure only one host saves the config.
  return jax.process_index() == 0


def save_json(wcp: WalkerCirculationParameters, output_dir: str) -> None:
  """Save a `WalkerCirculationParameters` to a JSON file."""
  if not _should_write_file():
    return

  dirpath = epath.Path(output_dir)
  dirpath.mkdir(mode=0o775, parents=True, exist_ok=True)
  filepath = dirpath / 'walker_circulation_parameters.json'
  filepath.write_text(wcp.to_json(indent=2))


def load_json(output_dir: str) -> WalkerCirculationParameters:
  """Load a `WalkerCirculationParameters` from a JSON file."""
  path = epath.Path(output_dir) / 'walker_circulation_parameters.json'
  return WalkerCirculationParameters.from_json(path.read_text())
