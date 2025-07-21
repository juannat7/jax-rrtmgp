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

"""Test radiative equilibrium solution."""

from typing import TypeAlias

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from rrtmgp import config
from rrtmgp import constants
from rrtmgp import sim_initializer
from rrtmgp import stretched_grid_util
from rrtmgp import test_util
from rrtmgp import radiative_eqb_solver
from rrtmgp import radiative_eqb_test_util
from rrtmgp import rrtmgp

FLAGS = flags.FLAGS
Array: TypeAlias = jax.Array


def remove_halos(f: npt.ArrayLike) -> npt.ArrayLike:
  halo_width = 1
  return f[:, :, halo_width:-halo_width]


# pylint: disable=invalid-name


def temperature_initial_profile(z: Array, cfg: config.Config) -> Array:
  T0 = 300.0  # Sea-surface temperature [K].
  gamma = 6.7e-3  # Lapse rate [K/m].
  z_t = 15e3  # Height of tropopause [m].
  Tt = T0 - gamma * z_t  # Temperature at the tropopause.
  T_above_tropopause = Tt * jnp.ones_like(z)
  T_below_tropopause = T0 - gamma * z
  T = jnp.where(z > z_t, T_above_tropopause, T_below_tropopause)

  # Initialize the hydrostatic/reference pressure and density.
  # There is a slight inconsistency because we compute the density assuming
  # there is no moisture, but there is actually moisture.  However, vertical
  # momentum and buoyancy are not computed here, so this slight inconsistency
  # does not pose any numerical problems.
  g, r_d = constants.G, constants.R_D
  p_0 = cfg.wp.exner_reference_pressure
  p_t = p_0 * (Tt / T0) ** (g / (r_d * gamma))  # Pressure at the tropopause.
  # Formulas given for p_ref below and above the tropopause.
  p_ref_below_tropopause = p_0 * (1 - gamma * z / T0) ** (g / (r_d * gamma))
  p_ref_above_tropopause = p_t * jnp.exp(-(g * (z - z_t) / (r_d * Tt)))
  p_ref = jnp.where(z > z_t, p_ref_above_tropopause, p_ref_below_tropopause)

  # Initialize the density.
  rho_ref = p_ref / (r_d * T)

  sfc_temperature = T[:, :, 0] + 1.0
  # Don't bother sharding; only using 1 core for this test.
  return T, p_ref, rho_ref, sfc_temperature


def init_fn(cfg: config.Config) -> dict[str, Array]:
  """Initialize the state variables."""
  grid_map_sharded = sim_initializer.initialize_grids(cfg)

  z = grid_map_sharded['z_c']
  T, p_ref, rho_ref, sfc_temperature = temperature_initial_profile(z, cfg)
  sg_map = stretched_grid_util.sg_map_from_states(grid_map_sharded)
  return T, p_ref, rho_ref, sfc_temperature, z, sg_map


class RadiativeEqbTest(absltest.TestCase):

  def test_radiative_eqb_solver(self):
    # SETUP
    FLAGS.use_rcemip_ozone_profile = True
    stretched_grid_path_z = test_util.save_1d_array_to_tempfile(
        self, radiative_eqb_test_util.STRETCHED_GRID_Z
    )
    cfg_ext = radiative_eqb_test_util.get_cfg_ext(stretched_grid_path_z)
    cfg = config.config_from_config_external(cfg_ext)

    radiative_transfer_cfg = cfg.radiative_transfer_cfg
    rrtmgp_ = rrtmgp(
        radiative_transfer_cfg, cfg.wp, cfg.grid_spacings[2]
    )

    # Initialize the state.
    T, p_ref, rho_ref, sfc_temperature, z, sg_map = init_fn(cfg)

    relative_humidity = 0.75
    dt = 12 * 3600.0  # Timestep = 1 hours.

    num_steps_per_cycle = 4
    num_cycles = 10

    def one_step_fn(
        loop_i, T_and_aux_output: tuple[Array, dict[str, Array]]
    ) -> tuple[Array, dict[str, Array]]:
      del loop_i
      T, _ = T_and_aux_output
      return radiative_eqb_solver.step(
          T,
          rho_ref,
          p_ref,
          sfc_temperature,
          sg_map,
          relative_humidity,
          cfg.wp,
          rrtmgp_,
          dt,
          use_scan=True,
      )

    @jax.jit
    def one_cycle(cycle_id, T: Array) -> tuple[Array, dict[str, Array]]:
      del cycle_id
      aux_output = {
          'rad_heat_src': jnp.zeros_like(T),
          'rad_heat_lw_3d': jnp.zeros_like(T),
          'rad_heat_sw_3d': jnp.zeros_like(T),
      }
      init_val = (T, aux_output)
      return jax.lax.fori_loop(0, num_steps_per_cycle, one_step_fn, init_val)

    # ACTION
    for _ in range(num_cycles):
      T, aux_output = one_cycle(0, T)

    # VERIFICATION
    # Check the simulation completed with no NaNs.
    T_final = remove_halos(np.array(T))
    rad_heat_src_final = remove_halos(np.array(aux_output['rad_heat_src']))

    self.assertFalse(np.isnan(T_final).any())
    self.assertFalse(np.isnan(rad_heat_src_final).any())


# pylint: enable=invalid-name

if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
