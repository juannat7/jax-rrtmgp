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
from swirl_jatmos import timestep_control
from swirl_jatmos import timestep_control_config

TimestepControlConfig: TypeAlias = timestep_control_config.TimestepControlConfig


class TimestepControlTest(absltest.TestCase):

  def test_timestep_control_single_point(self):
    # SETUP
    tol = 1e-5
    cfg = TimestepControlConfig(
        desired_cfl=0.7,
        max_dt=30.0,
        min_dt=1.0,
        max_change_factor=1.2,
        min_change_factor=0.4,
    )
    target_dt = jnp.array(7.0)

    # Unclamped.
    dt = jnp.array(6.8)
    new_dt = timestep_control._compute_next_dt_from_target_dt(
        target_dt, dt, cfg
    )
    self.assertAlmostEqual(new_dt, 7.0, delta=tol)

    dt = jnp.array(10.0)
    new_dt = timestep_control._compute_next_dt_from_target_dt(
        target_dt, dt, cfg
    )
    self.assertAlmostEqual(new_dt, 7.0, delta=tol)

    # Clamped by max_change_factor.
    dt = jnp.array(4.0)
    new_dt = timestep_control._compute_next_dt_from_target_dt(
        target_dt, dt, cfg
    )
    self.assertAlmostEqual(new_dt, 4.8, delta=tol)

    # Clamped by min_change_factor
    dt = jnp.array(25.0)
    new_dt = timestep_control._compute_next_dt_from_target_dt(
        target_dt, dt, cfg
    )
    self.assertAlmostEqual(new_dt, 10.0, delta=tol)

    # Clamped by min_dt
    target_dt = jnp.array(0.35)
    dt = jnp.array(1.01)
    new_dt = timestep_control._compute_next_dt_from_target_dt(
        target_dt, dt, cfg
    )
    self.assertAlmostEqual(new_dt, cfg.min_dt, delta=tol)

    # Clamped by max_dt
    target_dt = jnp.array(35.0)
    dt = jnp.array(28.0)
    new_dt = timestep_control._compute_next_dt_from_target_dt(
        target_dt, dt, cfg
    )
    self.assertAlmostEqual(new_dt, cfg.max_dt, delta=tol)

  def test_timestep_control_with_halos(self):
    # SETUP
    cfg = TimestepControlConfig(
        desired_cfl=0.7,
        max_dt=30.0,
        min_dt=1.0,
        max_change_factor=1.2,
        min_change_factor=0.4,
    )

    nx, ny, nz = 16, 16, 16
    # Advective timescale is 10.0.
    u_fcc = 0.1 * np.ones((nx, ny, nz), dtype=np.float32)
    dx_f = 1.0 * jnp.ones((nx, 1, 1), dtype=np.float32)

    # Fill z halos with junk values NaNs. (halo_width=1)
    u_fcc[:, :, 0] = np.nan
    u_fcc[:, :, -1] = -4e7

    # Convert to jax arrays.
    u_fcc = jnp.array(u_fcc)
    v_cfc = jnp.zeros_like(u_fcc)
    w_ccf = jnp.zeros_like(u_fcc)
    dy_f = 1.0 * jnp.ones((1, ny, 1), dtype=np.float32)
    dz_f = 1.0 * jnp.ones((1, 1, nz), dtype=np.float32)

    dt = jnp.array(6.6, dtype=jnp.float32)
    dt_ns = (dt * 1e9).astype(jnp.int64)

    # ACTION
    new_dt_ns = timestep_control._compute_next_dt(
        u_fcc, v_cfc, w_ccf, dx_f, dy_f, dz_f, dt_ns, cfg
    )
    new_dt = (new_dt_ns * 1e-9).astype(dt.dtype)

    # VERIFICATION
    # Verify that the halos are ignored and the proper timestep is
    # returned.
    expected_dt = 7.0
    self.assertAlmostEqual(new_dt, expected_dt, delta=1e-5)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
