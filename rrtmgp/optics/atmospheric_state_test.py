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

"""Tests whether the atmospheric conditions are loaded properly from a proto."""

import unittest
from pathlib import Path
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import atmospheric_state

_GLOBAL_MEAN_VMR_FILENAME = 'rrtmgp/optics/test_data/vmr_global_means.json'
_VMR_SOUNDING_FILENAME = 'rrtmgp/optics/test_data/vmr_interpolated.csv'

root = Path()
_GLOBAL_MEAN_VMR_FILEPATH = root / _GLOBAL_MEAN_VMR_FILENAME
_VMR_SOUNDING_FILEPATH = root / _VMR_SOUNDING_FILENAME


class AtmosphericStateTest(absltest.TestCase):

  def test_atmospheric_state_lookup_loads_data_from_proto(self):
    atmospheric_state_cfg = radiative_transfer.AtmosphericStateCfg(
        vmr_global_mean_filepath=_GLOBAL_MEAN_VMR_FILEPATH,
        vmr_sounding_filepath=_VMR_SOUNDING_FILEPATH,
        sfc_emis=0.92,
        sfc_alb=0.01,
        zenith=1.5,
        irrad=1000.0,
        toa_flux_lw=50.0,
    )
    atmos_state = atmospheric_state.from_config(atmospheric_state_cfg)

    self.assertEqual(atmos_state.sfc_emis, 0.92)
    self.assertEqual(atmos_state.sfc_alb, 0.01)
    self.assertEqual(atmos_state.zenith, 1.5)
    self.assertEqual(atmos_state.irrad, 1000.0)
    self.assertEqual(atmos_state.toa_flux_lw, 50.0)
    self.assertLen(atmos_state.vmr.profiles['ch4'], 64)


if __name__ == '__main__':
  absltest.main()
