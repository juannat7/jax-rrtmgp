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

"""Utilities for radiative_eqb_test.py.

The config is based on the RCEMIP-I setup for the gas concentrations.
"""

from pathlib import Path
import numpy as np
from rrtmgp import config
from rrtmgp import timestep_control_config
from rrtmgp.config import radiative_transfer
from rrtmgp.thermodynamics import water

HALO_WIDTH = 1
_VMR_GLOBAL_MEAN_FILENAME = 'rrtmgp/optics/test_data/rcemip_global_mean_vmr.json'
_VMR_SOUNDING_FILENAME = 'rrtmgp/optics/test_data/rcemip_vmr_sounding.csv'
_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g128.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g112.nc'
_CLD_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
_CLD_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc'

root = Path()
_VMR_GLOBAL_MEAN_FILEPATH = root / _VMR_GLOBAL_MEAN_FILENAME
_VMR_SOUNDING_FILEPATH = root / _VMR_SOUNDING_FILENAME
_LONGWAVE_NC_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME
_SHORTWAVE_NC_FILEPATH = root / _SW_LOOKUP_TABLE_FILENAME
_CLOUD_LONGWAVE_NC_FILEPATH = root / _CLD_LW_LOOKUP_TABLE_FILENAME
_CLOUD_SHORTWAVE_NC_FILEPATH = root / _CLD_SW_LOOKUP_TABLE_FILENAME


# z levels from Wing et al (2018), with 74 vertical levels.
STRETCHED_GRID_Z = np.fromstring(
    """37
111
194
288
395
520
667
843
1062
1331
1664
2055
2505
3000
3500
4000
4500
5000
5500
6000
6500
7000
7500
8000
8500
9000
9500
10000
10500
11000
11500
12000
12500
13000
13500
14000
14500
15000
15500
16000
16500
17000
17500
18000
18500
19000
19500
20000
20500
21000
21500
22000
22500
23000
23500
24000
24500
25000
25500
26000
26500
27000
27500
28000
28750
29750
31000
32500
34250
36250
38500
41000
44000
47000
50000
53000
""",
    sep='\n',
)

z = STRETCHED_GRID_Z
lz = z[-1] + (z[-1] - z[-2]) / 2
lz = float(lz)
num_z_levels = len(z)


def _get_radiative_transfer_cfg(
    include_ozone: bool = True,
) -> radiative_transfer.RadiativeTransfer:
  """Gets the radiative transfer config for a Walker Circulation simulation."""
  if include_ozone:
    vmr_sounding_filepath = _VMR_SOUNDING_FILEPATH
  else:
    vmr_sounding_filepath = ''

  radiative_transfer_cfg = radiative_transfer.RadiativeTransfer(
      optics=radiative_transfer.OpticsParameters(
          optics=radiative_transfer.RRTMOptics(
              longwave_nc_filepath=_LONGWAVE_NC_FILEPATH,
              shortwave_nc_filepath=_SHORTWAVE_NC_FILEPATH,
              cloud_longwave_nc_filepath=_CLOUD_LONGWAVE_NC_FILEPATH,
              cloud_shortwave_nc_filepath=_CLOUD_SHORTWAVE_NC_FILEPATH,
          )
      ),
      atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
          sfc_emis=0.98,
          sfc_alb=0.07,
          zenith=0.733911,
          irrad=551.58,
          toa_flux_lw=0.0,
          vmr_sounding_filepath=vmr_sounding_filepath,
          vmr_global_mean_filepath=_VMR_GLOBAL_MEAN_FILEPATH,
      ),
      use_scan=True,
      save_lw_sw_heating_rates=True,
  )
  return radiative_transfer_cfg

