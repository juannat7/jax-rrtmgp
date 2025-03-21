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

"""Mock-Walker circulation simulation test."""

import functools
from typing import TypeAlias

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from swirl_jatmos import config
from swirl_jatmos import constants
from swirl_jatmos import driver
from swirl_jatmos import test_util
from swirl_jatmos.sim_setups import walker_circulation
from swirl_jatmos.sim_setups import walker_circulation_diagnostics
from swirl_jatmos.sim_setups import walker_circulation_parameters
from swirl_jatmos.sim_setups import walker_circulation_test_util
from swirl_jatmos.sim_setups import walker_circulation_utils

FLAGS = flags.FLAGS
Array: TypeAlias = jax.Array

_HALO_WIDTH = 1


def remove_halos(f: npt.ArrayLike) -> npt.ArrayLike:
  hw = _HALO_WIDTH
  return f[..., hw:-hw]


class WalkerCirculationTest(absltest.TestCase):

  def test_walker_circulation(self):
    """Test a few cycles of walker circulation sim."""
    # SETUP
    FLAGS.use_rcemip_ozone_profile = True
    stretched_grid_path_z = test_util.save_1d_array_to_tempfile(
        self, walker_circulation_test_util.STRETCHED_GRID_Z
    )
    cfg_ext = walker_circulation_test_util.get_cfg_ext(stretched_grid_path_z)

    output_dir = self.create_tempdir().full_path
    t_final = 3600.0
    sec_per_cycle = 900.0

    cfg = config.config_from_config_external(cfg_ext)
    wcp = walker_circulation_parameters.WalkerCirculationParameters()
    dtype = jnp.float32
    p_0 = cfg.wp.exner_reference_pressure

    _, _, _, rho_ref_xxc = walker_circulation.analytic_profiles_from_paper(
        jnp.array(cfg.z_c, dtype=dtype), wcp, p_0
    )
    init_fn = functools.partial(walker_circulation.init_fn, wcp=wcp)

    # ACTION
    states, _, _ = driver.run_driver(
        init_fn,
        np.array(rho_ref_xxc, dtype=np.float64),
        output_dir,
        t_final,
        sec_per_cycle,
        cfg,
        preprocess_update_fn=walker_circulation.preprocess_update_fn,
        diagnostics_update_fn=walker_circulation_diagnostics.diagnostics_update_fn,
    )

    # VERIFICATION
    # Check the simulation completed with no NaNs.
    u_final = remove_halos(np.array(states['u']))
    dtheta_li_final = remove_halos(np.array(states['dtheta_li']))

    self.assertFalse(np.isnan(u_final).any())
    self.assertFalse(np.isnan(dtheta_li_final).any())

  def test_sim_with_sounding(self):
    """Test sim when using files for sounding of θ_li and q_t."""
    # SETUP
    FLAGS.use_rcemip_ozone_profile = True
    stretched_grid_path_z = test_util.save_1d_array_to_tempfile(
        self, walker_circulation_test_util.STRETCHED_GRID_Z
    )
    cfg_ext = walker_circulation_test_util.get_cfg_ext(stretched_grid_path_z)
    dtype = jnp.float32
    wp = cfg_ext.wp
    p_0 = wp.exner_reference_pressure

    output_dir = self.create_tempdir().full_path
    # Create a sounding and save it to the `output_dir`.
    default_wcp = walker_circulation_parameters.WalkerCirculationParameters()

    z_c = walker_circulation_test_util.STRETCHED_GRID_Z
    # pylint: disable=invalid-name
    q_t_c, p_ref_xxc, T_c, _ = walker_circulation.analytic_profiles_from_paper(
        jnp.array(z_c, dtype=dtype), default_wcp, p_0
    )
    T_c = np.squeeze(np.array(T_c))
    # pylint: enable=invalid-name
    q_t = np.squeeze(np.array(q_t_c))
    p_ref_xxc = np.squeeze(np.array(p_ref_xxc))
    r_m_1d = (1 - q_t_c) * constants.R_D + q_t_c * constants.R_V
    cp_m_1d = (1 - q_t_c) * constants.CP_D + q_t_c * constants.CP_V
    exner_1d = (p_ref_xxc / wp.exner_reference_pressure) ** (r_m_1d / cp_m_1d)
    theta_li = T_c / exner_1d

    print(f'{theta_li.shape=}')
    print(f'{q_t.shape=}')
    walker_circulation_utils.save_sounding(output_dir, z_c, theta_li, q_t)

    t_final = 20.0
    sec_per_cycle = 20.0

    cfg = config.config_from_config_external(cfg_ext)
    wcp = walker_circulation_parameters.WalkerCirculationParameters(
        sounding_dirname=output_dir
    )

    _, _, _, rho_ref_xxc = walker_circulation.analytic_profiles_from_paper(
        jnp.array(cfg.z_c, dtype=dtype), wcp, p_0
    )
    init_fn = functools.partial(walker_circulation.init_fn, wcp=wcp)

    # ACTION
    states, _, _ = driver.run_driver(
        init_fn,
        np.array(rho_ref_xxc, dtype=np.float64),
        output_dir,
        t_final,
        sec_per_cycle,
        cfg,
        preprocess_update_fn=walker_circulation.preprocess_update_fn,
        diagnostics_update_fn=walker_circulation_diagnostics.diagnostics_update_fn,
    )

    # VERIFICATION
    # Check the simulation completed with no NaNs.
    u_final = remove_halos(np.array(states['u']))
    dtheta_li_final = remove_halos(np.array(states['dtheta_li']))

    self.assertFalse(np.isnan(u_final).any())
    self.assertFalse(np.isnan(dtheta_li_final).any())


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  jax.config.update('jax_threefry_partitionable', True)
  absltest.main()
