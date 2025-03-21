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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from swirl_jatmos import config
from swirl_jatmos import timestep_control_config
from swirl_jatmos.rrtmgp.config import radiative_transfer


class ConfigTest(parameterized.TestCase):

  def test_load_array_from_file(self):
    # SETUP
    data = """# Header comment
    1.1  # Comment
    3.1
    5.1
    6.1
    """
    output_file_path = self.create_tempfile(content=data).full_path

    # ACTION
    array = config._load_array_from_file(output_file_path)

    # VERIFICATION
    expected = np.array([1.1, 3.1, 5.1, 6.1])
    np.testing.assert_array_equal(array, expected)

  def test_create_config_from_config_external(self):
    # SETUP
    cfgext = config.ConfigExternal(
        cx=1,
        cy=1,
        cz=1,
        nx=10,
        ny=16,
        nz=32,
        domain_x=(0.0, 1.0),
        domain_y=(0.0, 1.0),
        domain_z=(0.0, 1.0),
        dt=1.2,
        timestep_control_cfg=timestep_control_config.TimestepControlConfig(
            disable_adaptive_timestep=True
        ),
        use_sgs=False,
        solve_pressure_only_on_last_rk3_stage=False,
        include_qt_sedimentation=False,
        uniform_y_2d=False,
        poisson_solver_type=config.PoissonSolverType.FAST_DIAGONALIZATION,
    )

    # ACTION
    cfg = config.config_from_config_external(cfgext)

    # VERIFICATION
    self.assertEqual(cfg.use_sgs, False)
    self.assertEqual(cfg.use_stretched_grid, (False, False, False))
    np.testing.assert_allclose(
        cfg.grid_spacings, (0.1, 1 / 16, 1 / 30), rtol=1e-12, atol=0
    )

  @parameterized.parameters(True, False)
  def test_serialize_and_deserialize_config(
      self, include_radiative_transfer_cfg: bool
  ):
    """Test de/serialization containing union-type nested dataclasses."""
    if include_radiative_transfer_cfg:
      radiative_transfer_cfg = radiative_transfer.RadiativeTransfer(
          optics=radiative_transfer.OpticsParameters(
              optics=radiative_transfer.GrayAtmosphereOptics()
          ),
          atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(),
      )
    else:
      radiative_transfer_cfg = None

    # SETUP
    cfgext = config.ConfigExternal(
        cx=1,
        cy=1,
        cz=1,
        nx=10,
        ny=16,
        nz=32,
        domain_x=(0.0, 1.0),
        domain_y=(0.0, 1.0),
        domain_z=(0.0, 1.0),
        dt=1.2,
        timestep_control_cfg=timestep_control_config.TimestepControlConfig(
            disable_adaptive_timestep=True
        ),
        use_sgs=False,
        radiative_transfer_cfg=radiative_transfer_cfg,
    )
    # ACTION
    # Serialize.
    cfg_json = cfgext.to_json()
    # Deserialize.
    cfgext_loaded = config.ConfigExternal.from_json(cfg_json)

    # VERIFICATION
    self.assertEqual(cfgext_loaded, cfgext)

  def test_load_json(self):
    # SETUP
    cfgext = config.ConfigExternal(
        cx=1,
        cy=1,
        cz=1,
        nx=10,
        ny=16,
        nz=32,
        domain_x=(0.0, 1.0),
        domain_y=(0.0, 1.0),
        domain_z=(0.0, 1.0),
        dt=1.2,
        timestep_control_cfg=timestep_control_config.TimestepControlConfig(
            disable_adaptive_timestep=True
        ),
        use_sgs=False,
    )
    output_dir = self.create_tempdir().full_path
    config.save_json(cfgext, output_dir)
    cfgext_loaded = config.load_json(output_dir)
    self.assertEqual(cfgext_loaded, cfgext)

if __name__ == '__main__':
  absltest.main()
