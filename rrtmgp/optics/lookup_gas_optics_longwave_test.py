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

"""Tests whether the longwave optics data for atmospheric gases are loaded properly."""

from absl.testing import absltest
from pathlib import Path
import jax
from rrtmgp.optics import lookup_gas_optics_longwave

_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g256.nc'

root = Path()
_LW_LOOKUP_TABLE_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME


class LookupGasOpticsLongwaveTest(absltest.TestCase):

  def test_longwave_optics_lookup_loads_data(self):
    lookup_longwave = lookup_gas_optics_longwave.from_nc_file(
        _LW_LOOKUP_TABLE_FILEPATH
    )
    expected_dims = {
        'n_gases': 19,
        'n_maj_absrb': 19,
        'n_atmos_layers': 2,
        'n_bnd': 16,
        'n_contrib_lower': 960,
        'n_contrib_upper': 544,
        'n_gpt': 256,
        'n_minor_absrb': 21,
        'n_minor_absrb_lower': 60,
        'n_minor_absrb_upper': 34,
        'n_mixing_fraction': 9,
        'n_p_ref': 59,
        'n_t_ref': 14,
        'n_t_plnk': 196,
    }
    actual_dims = {
        'n_gases': lookup_longwave.n_gases,
        'n_maj_absrb': lookup_longwave.n_maj_absrb,
        'n_atmos_layers': lookup_longwave.n_atmos_layers,
        'n_bnd': lookup_longwave.n_bnd,
        'n_contrib_lower': lookup_longwave.n_contrib_lower,
        'n_contrib_upper': lookup_longwave.n_contrib_upper,
        'n_gpt': lookup_longwave.n_gpt,
        'n_minor_absrb': lookup_longwave.n_minor_absrb,
        'n_minor_absrb_lower': lookup_longwave.n_minor_absrb_lower,
        'n_minor_absrb_upper': lookup_longwave.n_minor_absrb_upper,
        'n_mixing_fraction': lookup_longwave.n_mixing_fraction,
        'n_p_ref': lookup_longwave.n_p_ref,
        'n_t_ref': lookup_longwave.n_t_ref,
        'n_t_plnk': lookup_longwave.n_t_plnk,
    }
    self.assertEqual(actual_dims, expected_dims)
    self.assertEqual(lookup_longwave.kmajor.shape, (14, 60, 9, 256))
    self.assertEqual(lookup_longwave.planck_fraction.shape, (14, 60, 9, 256))
    self.assertEqual(lookup_longwave.t_planck.shape, (196,))
    self.assertEqual(lookup_longwave.totplnk.shape, (16, 196))


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
