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

"""Utilities for walker_circulation_test.py."""

from etils import epath
import numpy as np

from swirl_jatmos import config
from swirl_jatmos import convection_config
from swirl_jatmos import sponge_config
from swirl_jatmos import timestep_control_config
from swirl_jatmos.boundary_conditions import boundary_conditions
from swirl_jatmos.boundary_conditions import monin_obukhov
from swirl_jatmos.rrtmgp.config import radiative_transfer
from swirl_jatmos.thermodynamics import water

HALO_WIDTH = 1
_VMR_GLOBAL_MEAN_FILENAME = (
    'rrtmgp/optics/test_data/rcemip_global_mean_vmr.json'
)
_VMR_SOUNDING_FILENAME = 'rrtmgp/optics/test_data/rcemip_vmr_sounding.csv'
_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g128.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g112.nc'
_CLD_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
_CLD_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc'

root = epath.resource_path('swirl_jatmos')
_VMR_GLOBAL_MEAN_FILEPATH = root / _VMR_GLOBAL_MEAN_FILENAME
_VMR_SOUNDING_FILEPATH = root / _VMR_SOUNDING_FILENAME
_LW_LOOKUP_TABLE_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME
_SW_LOOKUP_TABLE_FILEPATH = root / _SW_LOOKUP_TABLE_FILENAME
_CLD_LW_LOOKUP_TABLE_FILEPATH = root / _CLD_LW_LOOKUP_TABLE_FILENAME
_CLD_SW_LOOKUP_TABLE_FILEPATH = root / _CLD_SW_LOOKUP_TABLE_FILENAME


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
28500
29000
29500
30000
30500
31000
31500
32000
32500
33000
""",
    sep='\n',
)


def _get_radiative_transfer_cfg() -> radiative_transfer.RadiativeTransfer:
  """Gets the radiative transfer config for a Walker Circulation simulation."""
  radiative_transfer_cfg = radiative_transfer.RadiativeTransfer(
      optics=radiative_transfer.OpticsParameters(
          optics=radiative_transfer.RRTMOptics(
              longwave_nc_filepath=_LW_LOOKUP_TABLE_FILEPATH,
              shortwave_nc_filepath=_SW_LOOKUP_TABLE_FILEPATH,
              cloud_longwave_nc_filepath=_CLD_LW_LOOKUP_TABLE_FILEPATH,
              cloud_shortwave_nc_filepath=_CLD_SW_LOOKUP_TABLE_FILEPATH,
          )
      ),
      atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
          sfc_emis=0.98,
          sfc_alb=0.07,
          zenith=0.733911,
          irrad=551.58,
          toa_flux_lw=0.0,
          vmr_sounding_filepath=_VMR_SOUNDING_FILEPATH,
          vmr_global_mean_filepath=_VMR_GLOBAL_MEAN_FILEPATH,
      ),
      update_cycle_seconds=900.0,  # 15 minutes.
      apply_cadence_seconds=100.0,
      use_scan=True,  # Compare performance for True vs False.
      save_lw_sw_heating_rates=True,
  )
  return radiative_transfer_cfg


def get_cfg_ext(stretched_grid_path_z: str = '') -> config.ConfigExternal:
  """Gets the config external for a Walker Circulation simulation."""
  radiative_transfer_cfg = _get_radiative_transfer_cfg()
  _P0 = 1.0148e5  # Pressure at the surface [Pa]. # pylint: disable=invalid-name

  cfg_ext = config.ConfigExternal(
      cx=4,
      cy=2,
      cz=1,
      nx=56,
      ny=56,
      nz=76,
      domain_x=(0, 200e3),
      domain_y=(0, 100e3),
      domain_z=(0, 33_250),
      halo_width=1,  # only used in z dimension.
      dt=8.0,
      timestep_control_cfg=timestep_control_config.TimestepControlConfig(
          desired_cfl=0.9,
          max_dt=25.0,
          min_dt=1e-2,
          max_change_factor=2,
          update_interval_steps=5,
      ),
      wp=water.WaterParams(exner_reference_pressure=_P0),
      convection_cfg=convection_config.ConvectionConfig(
          momentum_scheme='weno5_js',
          theta_li_scheme='weno5_js',
          q_t_scheme='weno5_js',
          q_r_scheme='upwind1',
          q_s_scheme='upwind1',
      ),
      use_sgs=True,
      stretched_grid_path_z=stretched_grid_path_z,
      z_bcs=boundary_conditions.ZBoundaryConditions(
          bottom=boundary_conditions.ZBC(
              bc_type='monin_obukhov',
              mop=monin_obukhov.MoninObukhovParameters(),
          ),
          top=boundary_conditions.ZBC(bc_type='no_flux'),
      ),
      solve_pressure_only_on_last_rk3_stage=False,
      include_qt_sedimentation=True,  # Watch out for this setting.
      sponge_cfg=sponge_config.SpongeConfig(
          coeff=6.0, sponge_fraction=0.55, c2=0.5
      ),
      # poisson_solver_type=config.PoissonSolverType.JACOBI,
      poisson_solver_type=config.PoissonSolverType.FAST_DIAGONALIZATION,
      aux_output_fields=('q_c', 'T'),
      diagnostic_fields=('T_1d_z',),
      viscosity=1e-3,
      diffusivity=1e-3,
      enforce_max_diffusivity=True,
      radiative_transfer_cfg=radiative_transfer_cfg,
      disable_checkpointing=True,
  )
  return cfg_ext
