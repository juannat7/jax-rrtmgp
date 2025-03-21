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

"""Supercell simulation test using the driver."""

from typing import TypeAlias

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from swirl_jatmos import config
from swirl_jatmos import driver
from swirl_jatmos import timestep_control_config
from swirl_jatmos.sim_setups import supercell
from swirl_jatmos.thermodynamics import water

Array: TypeAlias = jax.Array
PartitionSpec: TypeAlias = jax.sharding.PartitionSpec
NamedSharding: TypeAlias = jax.sharding.NamedSharding
PoissonSolverType: TypeAlias = config.PoissonSolverType

_HALO_WIDTH = 1

# Set the problem config.
cfg_ext = config.ConfigExternal(
    cx=2,
    cy=2,
    cz=2,
    nx=128,
    ny=128,
    nz=128,
    domain_x=(0, 100e3),
    domain_y=(0, 100e3),
    domain_z=(0, 20e3),
    halo_width=_HALO_WIDTH,  # only used in z dimension.
    dt=1.5,
    timestep_control_cfg=timestep_control_config.TimestepControlConfig(
        disable_adaptive_timestep=True
    ),
    wp=water.WaterParams(),
    use_sgs=True,
    # poisson_solver_type=config.PoissonSolverType.JACOBI,
    poisson_solver_type=config.PoissonSolverType.FAST_DIAGONALIZATION,
    aux_output_fields=('q_c',),
    viscosity=1e-3,
    diffusivity=1e-3,
)


def remove_halos(f: npt.ArrayLike) -> npt.ArrayLike:
  hw = _HALO_WIDTH
  return f[:, :, hw:-hw]


class SupercellTest(absltest.TestCase):

  def test_supercell(self):
    # SETUP
    output_dir = self.create_tempdir().full_path
    t_final = 600.0  # Simulate 10 minutes.
    sec_per_cycle = 60

    config.save_json(cfg_ext, output_dir)

    cfg = config.config_from_config_external(cfg_ext)
    dtype = jnp.float32

    _, _, _, rho_ref_xxc = supercell.thermodynamic_initial_condition(
        jnp.array(cfg.z_c, dtype=dtype), cfg.wp
    )

    # ACTION
    states, _, _ = driver.run_driver(
        supercell.init_fn,
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
