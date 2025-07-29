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

import unittest
from pathlib import Path
from rrtmgp.optics import lookup_cloud_optics

_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'

root = Path()
_LW_LOOKUP_TABLE_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME


class LookupCloudOpticsTest(unittest.TestCase):

  def test_longwave_optics_lookup_loads_data(self):
    # ACTION
    lookup_cld = lookup_cloud_optics.from_nc_file(_LW_LOOKUP_TABLE_FILEPATH)

    # VERIFICATION
    self.assertEqual(lookup_cld.n_size_liq, 20)
    self.assertEqual(lookup_cld.n_size_ice, 18)
    self.assertEqual(lookup_cld.ext_liq.shape, (16, 20))
    self.assertEqual(lookup_cld.ssa_liq.shape, (16, 20))
    self.assertEqual(lookup_cld.asy_liq.shape, (16, 20))
    self.assertEqual(lookup_cld.ext_ice.shape, (3, 16, 18))
    self.assertEqual(lookup_cld.ssa_ice.shape, (3, 16, 18))
    self.assertEqual(lookup_cld.asy_ice.shape, (3, 16, 18))
    self.assertEqual(lookup_cld.ice_roughness.value, 1)


if __name__ == '__main__':
  unittest.main()
