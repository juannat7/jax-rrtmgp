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

import functools
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
from swirl_jatmos import constants
from swirl_jatmos import jatmos_types
from swirl_jatmos import kernel_ops
from swirl_jatmos import test_util
from swirl_jatmos.rrtmgp.config import radiative_transfer
from swirl_jatmos.rrtmgp.optics import atmospheric_state
from swirl_jatmos.rrtmgp.optics import optics
from swirl_jatmos.rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

_VMR_GLOBAL_MEAN_FILENAME = 'rrtmgp/optics/test_data/rcemip_global_mean_vmr.json'
_ATMOSPHERIC_STATE_FILENAME = 'rrtmgp/optics/test_data/cloudysky_as.nc'
_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g256.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g224.nc'
_LW_LOOKUP_TABLE_COMPACT_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g128.nc'
_SW_LOOKUP_TABLE_COMPACT_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g112.nc'
_CLD_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
_CLD_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc'
_ALL_SKY_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/cloudysky_lut.nc'

root = epath.resource_path('swirl_jatmos')
_VMR_GLOBAL_MEAN_FILEPATH = root / _VMR_GLOBAL_MEAN_FILENAME
_ATMOSPHERIC_STATE_FILEPATH = root / _ATMOSPHERIC_STATE_FILENAME
_LW_LOOKUP_TABLE_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME
_SW_LOOKUP_TABLE_FILEPATH = root / _SW_LOOKUP_TABLE_FILENAME
_LW_LOOKUP_TABLE_COMPACT_FILEPATH = root / _LW_LOOKUP_TABLE_COMPACT_FILENAME
_SW_LOOKUP_TABLE_COMPACT_FILEPATH = root / _SW_LOOKUP_TABLE_COMPACT_FILENAME

_CLOUD_LONGWAVE_NC_FILEPATH = root / _CLD_LW_LOOKUP_TABLE_FILENAME
_CLOUD_SHORTWAVE_NC_FILEPATH = root / _CLD_SW_LOOKUP_TABLE_FILENAME
_ALL_SKY_REFERENCE_FILEPATH = root / _ALL_SKY_REFERENCE_FILENAME


def _remove_halos(f: Array) -> Array:
  """Remove the halos from the output."""
  return f[:, :, 1:-1]


def _setup_radiation_params(
    use_compact_lookup: bool,
) -> radiative_transfer.RadiativeTransfer:
  """Create an instance of `RadiativeTransfer`."""
  if use_compact_lookup:
    lw_lookup_table_nc_filepath = _LW_LOOKUP_TABLE_COMPACT_FILEPATH
    sw_lookup_table_nc_filepath = _SW_LOOKUP_TABLE_COMPACT_FILEPATH
  else:
    lw_lookup_table_nc_filepath = _LW_LOOKUP_TABLE_FILEPATH
    sw_lookup_table_nc_filepath = _SW_LOOKUP_TABLE_FILEPATH

  return radiative_transfer.RadiativeTransfer(
      optics=radiative_transfer.OpticsParameters(
          optics=radiative_transfer.RRTMOptics(
              longwave_nc_filepath=lw_lookup_table_nc_filepath,
              shortwave_nc_filepath=sw_lookup_table_nc_filepath,
              cloud_longwave_nc_filepath=_CLOUD_LONGWAVE_NC_FILEPATH,
              cloud_shortwave_nc_filepath=_CLOUD_SHORTWAVE_NC_FILEPATH,
          )
      ),
      atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
          sfc_emis=0.98,
          sfc_alb=0.06,
          zenith=0.535526654,
          irrad=1360.8585174,
          toa_flux_lw=0.0,
          vmr_global_mean_filepath=_VMR_GLOBAL_MEAN_FILEPATH,
      ),
  )


def _setup_atmospheric_profiles() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
]:
  """Load vertical profiles of pressure and temperature from RFMIP."""
  halo_width = 1
  atmos_state_ds = nc.Dataset(_ATMOSPHERIC_STATE_FILEPATH, 'r')

  # Reverse order of the profiles so they correspond to increasing altitude.
  p_internal = atmos_state_ds['p_lay'][:].data
  pres_level = atmos_state_ds['p_lev'][:].data

  # Set the boundary values using linear extrapolation from the outermost
  # levels.
  paddings_2d = ((0, 0), (halo_width, halo_width))
  # Transpose so the vertical profile is stored in the last axis.
  pressure = np.pad(np.transpose(p_internal), paddings_2d, mode='edge')
  pressure_level = np.pad(
      np.transpose(pres_level), ((0, 0), (halo_width, halo_width - 1))
  )
  # After padding, shape should be (42, 42 + 2) = (42, 44).

  temp_internal = np.transpose(atmos_state_ds['t_lay'][:].data)
  temp_level = np.transpose(atmos_state_ds['t_lev'][:].data)

  nx, nz = temp_internal.shape  # 42, 42
  nz_with_halos = nz + 2 * halo_width
  temperature = np.zeros((nx, nz_with_halos), dtype=jatmos_types.f_dtype)
  temperature[:, halo_width:-halo_width] = temp_internal
  # Fill in halos by extrapolating from face (temp_level) and node values.
  temperature[:, 0] = 2 * temp_level[:, 0] - temp_internal[:, 0]
  temperature[:, -1] = 2 * temp_level[:, -1] - temp_internal[:, -1]

  temperature_level = np.zeros_like(temperature)
  temperature_level[:, 1:] = temp_level
  # Fill in halos with same value as in `temperature`.
  temperature_level[:, 0] = temperature[:, 0]

  vmr_profiles = {}
  for k in atmos_state_ds.variables:
    if not k.startswith('vmr_'):
      continue
    chem_formula = k[len('vmr_') :]
    vmr_profiles[chem_formula] = np.pad(
        np.transpose(atmos_state_ds[k][:].data), paddings_2d, mode='edge'
    )

  sfc_temperature = atmos_state_ds['t_sfc'][:].data
  # Here `_level` means these are the values on faces.
  return (
      pressure,
      pressure_level,
      temperature,
      temperature_level,
      vmr_profiles,
      sfc_temperature,
  )


def _load_expected_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
  reference_data = nc.Dataset(_ALL_SKY_REFERENCE_FILEPATH, 'r')

  halo_width = 1
  paddings = ((0, 0), (halo_width, halo_width - 1))

  lw_flux_up = np.pad(
      np.transpose(reference_data['lw_flux_up'][:].data), paddings
  )
  lw_flux_down = np.pad(
      np.transpose(reference_data['lw_flux_dn'][:].data), paddings
  )
  sw_flux_up = np.pad(
      np.transpose(reference_data['sw_flux_up'][:].data), paddings
  )
  sw_flux_down = np.pad(
      np.transpose(reference_data['sw_flux_dn'][:].data), paddings
  )
  sw_flux_dir = np.pad(
      np.transpose(reference_data['sw_flux_dir'][:].data), paddings
  )
  return lw_flux_up, lw_flux_down, sw_flux_up, sw_flux_down, sw_flux_dir


def _air_molecules_per_area(p_bottom: Array, vmr_h2o: Array) -> Array:
  """Compute the number of molecules in an atmospheric grid cell per area."""
  dp = kernel_ops.forward_difference(p_bottom, dim=2)
  mol_m_air = constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o
  return -(dp / constants.G) * constants.AVOGADRO / mol_m_air


class AllSkyTest(parameterized.TestCase):

  @parameterized.product(
      use_compact_lookup=[True, False],
      use_scan=[True, False],
  )
  def test_two_stream_solver_with_cloudy_sky(
      self, use_compact_lookup: bool, use_scan: bool
  ):
    # SETUP
    site = 0  # Use data from site 0.

    (
        pressure_allsites,
        pressure_level_allsites,
        temperature_allsites,
        temperature_level_allsites,
        vmr_profiles_allsites,
        _,
    ) = _setup_atmospheric_profiles()
    # pressure, pressure_level, etc. are 2D arrays, where the 2nd dimension is
    # the vertical dimension.  The 1st dimension is a dimension corresponding to
    # the 'site' the data comes from.

    radiation_params = _setup_radiation_params(use_compact_lookup)
    atmos_state = atmospheric_state.from_config(
        radiation_params.atmospheric_state_cfg
    )
    optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

    # Model inputs.
    n_horiz = 2
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
    )
    sfc_temperature = temperature_level_allsites[site, 1] * jnp.ones(
        (n_horiz, n_horiz), dtype=jatmos_types.f_dtype
    )
    vmr_fields = {
        k: convert_to_3d(v[site, :]) for k, v in vmr_profiles_allsites.items()
    }
    p = convert_to_3d(pressure_allsites[site, :])
    pressure_level = convert_to_3d(pressure_level_allsites[site, :])
    temperature = convert_to_3d(temperature_allsites[site, :])

    # Effective radius of condensate particles.
    r_liq = 1.2e-5
    # The division by 2 is to enable compatibility with the old RRTMGP
    # tables, after the ice radius -> diameter fix was made. We should remove
    # the division by 2 after the tables with expected values are updated.
    r_ice = 9.5e-5 / 2
    # Cloud path for liquid and ice in kg/kg.
    cld_path = 1e-2
    # Let there be condensates only between 100 hPa and 900 hPa.

    ones = jnp.ones_like(p)
    ind_valid_p_range = jnp.logical_and(p > 10000, p < 90000)
    r_eff_liq = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature > 263),
        r_liq * ones,
        0.0,
    )
    cld_path_liq = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature > 263),
        cld_path * ones,
        0.0,
    )
    r_eff_ice = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature < 273),
        r_ice * ones,
        0.0,
    )
    cld_path_ice = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature < 273),
        cld_path * ones,
        0.0,
    )

    molecules = _air_molecules_per_area(pressure_level, vmr_fields['h2o'])

    # ACTION
    output_lw = two_stream.solve_lw(
        p,
        temperature,
        molecules,
        optics_lib,
        atmos_state,
        vmr_fields,
        sfc_temperature,
        cloud_r_eff_liq=r_eff_liq,
        cloud_path_liq=cld_path_liq,
        cloud_r_eff_ice=r_eff_ice,
        cloud_path_ice=cld_path_ice,
        use_scan=use_scan,
    )
    output_sw = two_stream.solve_sw(
        p,
        temperature,
        molecules,
        optics_lib,
        atmos_state,
        vmr_fields,
        cloud_r_eff_liq=r_eff_liq,
        cloud_path_liq=cld_path_liq,
        cloud_r_eff_ice=r_eff_ice,
        cloud_path_ice=cld_path_ice,
        use_scan=use_scan,
    )

    # VERIFICATION
    (
        expected_lw_flux_up,
        expected_lw_flux_down,
        expected_sw_flux_up,
        expected_sw_flux_down,
        _,
    ) = _load_expected_data()

    expected_flux_up_lw = convert_to_3d(expected_lw_flux_up[site, :])
    expected_flux_down_lw = convert_to_3d(expected_lw_flux_down[site, :])
    expected_flux_up_sw = convert_to_3d(expected_sw_flux_up[site, :])
    expected_flux_down_sw = convert_to_3d(expected_sw_flux_down[site, :])

    np.testing.assert_allclose(
        _remove_halos(output_lw['flux_down']),
        _remove_halos(expected_flux_down_lw),
        rtol=2e-5,
        atol=0.3 if use_compact_lookup else 0.22,
    )
    np.testing.assert_allclose(
        _remove_halos(output_lw['flux_up']),
        _remove_halos(expected_flux_up_lw),
        rtol=2e-3,
        atol=0.1,
    )
    np.testing.assert_allclose(
        _remove_halos(output_sw['flux_down']),
        _remove_halos(expected_flux_down_sw),
        rtol=1.4e-3 if use_compact_lookup else 1e-3,
        atol=0,
    )
    np.testing.assert_allclose(
        _remove_halos(output_sw['flux_up']),
        _remove_halos(expected_flux_up_sw),
        rtol=3e-3 if use_compact_lookup else 1e-3,
        atol=0,
    )


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
