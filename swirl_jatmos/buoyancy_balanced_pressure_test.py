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

from typing import TypeAlias

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import buoyancy_balanced_pressure
from swirl_jatmos import derivatives
from swirl_jatmos import interpolation
from swirl_jatmos import stretched_grid_util

Array: TypeAlias = jax.Array


class SimulationTest(absltest.TestCase):

  def test_buoyancy_balanced_pressure_from_rho(self):
    # SETUP
    nz = 32
    halo_width = 2

    # Uniform grid.
    z_nodes = np.linspace(0, 4.5, nz)
    dz = z_nodes[1] - z_nodes[0]

    # Create (1, 1, nz) rho_ref_xxf.
    rho_ref_xxf = jnp.linspace(1.0, 2.0, nz)[jnp.newaxis, jnp.newaxis, :]

    # Create (1, 1, nz) rho_thermal_xxc.
    rho_thermal_xxc = jnp.linspace(0.9, 1.8, nz)[jnp.newaxis, jnp.newaxis, :]
    rho_thermal_xxc = jnp.tile(rho_thermal_xxc, reps=(2, 1, 1))  # (2, 1, nz)

    # ACTION
    p_c = buoyancy_balanced_pressure.buoyancy_balanced_pressure_from_rho(
        rho_ref_xxf,
        rho_thermal_xxc,
        dz,
        halo_width,
        sg_map={},
    )
    p_c = p_c[0:1, :, :]  # Shape should be (1, 1, nz).

    # VERIFICATION
    # Instantiate derivatives to take the derivative and verify dp/dz = b.
    deriv_lib = derivatives.Derivatives(
        grid_spacings=(1.0, 1.0, dz), use_stretched_grid=(False, False, False)
    )
    dp_dz_f = deriv_lib.dz_c_to_f(p_c, sg_map={})
    # Remove halos and 1st element to extract meaningful values only
    dp_dz_f = dp_dz_f[0, 0, halo_width + 1 : -halo_width]

    # Recalculate the buoyancy
    g = 9.81
    rho_thermal_xxf = interpolation.centered_node_to_face(rho_thermal_xxc, 2)
    b_f = -g * (rho_thermal_xxf - rho_ref_xxf) / rho_thermal_xxf
    b_f = b_f[0, 0, halo_width + 1 : -halo_width]

    np.testing.assert_allclose(dp_dz_f, b_f, rtol=1e-10, atol=1e-10)

  def test_buoyancy_balanced_pressure_from_rho_stretched_grid(self):
    # SETUP
    nz = 32
    halo_width = 2

    # Stretched, nonuniform grid.
    grid_spacing = 1.0  # Grid spacing in transformed coordinate.
    z_nodes = np.arange(nz) ** 2 / 252
    z_faces = (z_nodes[:-1] + z_nodes[1:]) / 2
    dz_first = z_nodes[1] - z_nodes[0]
    z_faces = np.concatenate((np.array([z_nodes[0] - dz_first / 2]), z_faces))
    z_faces = z_faces[np.newaxis, np.newaxis, :]  # Change to (1, 1, nz) shape
    z_faces = jnp.array(z_faces)
    sg_map = {}
    sg_map[stretched_grid_util.hf_key(dim=2)] = z_faces

    # Create (1, 1, nz) rho_ref_xxf.
    rho_ref_xxf = jnp.linspace(1.0, 2.0, nz)[jnp.newaxis, jnp.newaxis, :]

    # Create (1, 1, nz) rho_thermal_xxc.
    rho_thermal_xxc = jnp.linspace(0.9, 1.8, nz)[jnp.newaxis, jnp.newaxis, :]
    rho_thermal_xxc = jnp.tile(rho_thermal_xxc, reps=(2, 1, 1))  # (2, 1, nz)

    # ACTION
    p_c = buoyancy_balanced_pressure.buoyancy_balanced_pressure_from_rho(
        rho_ref_xxf,
        rho_thermal_xxc,
        grid_spacing,
        halo_width,
        sg_map=sg_map,
    )
    p_c = p_c[0:1, :, :]  # Shape should be (1, 1, nz).

    # VERIFICATION
    # Instantiate derivatives to take the derivative and verify dp/dz = b.
    deriv_lib = derivatives.Derivatives(
        grid_spacings=(1.0, 1.0, 1.0), use_stretched_grid=(False, False, True)
    )
    dp_dz_f = deriv_lib.dz_c_to_f(p_c, sg_map)
    # Remove halos and 1st element to extract meaningful values only
    dp_dz_f = dp_dz_f[0, 0, halo_width + 1 : -halo_width]

    # Recalculate the buoyancy
    g = 9.81
    rho_thermal_xxf = interpolation.centered_node_to_face(rho_thermal_xxc, 2)
    b_f = -g * (rho_thermal_xxf - rho_ref_xxf) / rho_thermal_xxf
    b_f = b_f[0, 0, halo_width + 1 : -halo_width]

    np.testing.assert_allclose(dp_dz_f, b_f, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
