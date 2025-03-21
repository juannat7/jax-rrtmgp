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

"""Buoyant bubble simulation test."""

from typing import TypeAlias

from absl.testing import absltest
import jax
import numpy as np
import numpy.typing as npt
from swirl_jatmos import config
from swirl_jatmos import driver
from swirl_jatmos.sim_setups import buoyant_bubble


Array: TypeAlias = jax.Array

_HALO_WIDTH = 1


def remove_halos(f: npt.ArrayLike) -> npt.ArrayLike:
  hw = _HALO_WIDTH
  return f[:, :, hw:-hw]


class BuoyantBubbleTest(absltest.TestCase):

  def test_buoyant_bubble(self):
    # SETUP
    output_dir = self.create_tempdir().full_path
    t_final = 1200.0
    sec_per_cycle = 100.0

    cfg = config.config_from_config_external(buoyant_bubble.cfg_ext)

    _, rho_ref_xxc, _, _ = buoyant_bubble.initial_profiles(cfg.z_c)

    # ACTION
    states, _, _ = driver.run_driver(
        buoyant_bubble.init_fn,
        np.array(rho_ref_xxc, dtype=np.float64),
        output_dir,
        t_final,
        sec_per_cycle,
        cfg,
    )

    # VERIFICATION
    # Check the simulation completed with no NaNs.
    u_final = remove_halos(np.array(states['u']))
    dtheta_li_final = remove_halos(np.array(states['dtheta_li']))

    self.assertFalse(np.isnan(u_final).any())
    self.assertFalse(np.isnan(dtheta_li_final).any())


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
