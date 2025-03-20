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

"""Calculation of the hydrodynamic pressure balancing buoyancy."""

from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from swirl_jatmos import constants
from swirl_jatmos import interpolation
from swirl_jatmos import stretched_grid_util

Array: TypeAlias = jax.Array
G = constants.G


def buoyancy_balanced_pressure_from_rho(
    rho_ref_xxf: Array,
    rho_thermal_xxc: Array,
    dz: float,
    halo_width: int,
    sg_map: dict[str, Array],
) -> Array:
  """Compute hydrodynamic pressure on (ccc) that balances buoyancy, given ρ.

  For an equilibrium situation where the fluid is at rest, the
  anelastic equation for w (vertical velocity) reduces to dp/dz = b.

  For computational reasons, it is advantageous to initialize the
  pressure such that it is in balance with the buoyancy.  This helps
  to preserve equilibrium, and also prevent the formation of gravity
  waves.

  If b can be calculated, then the exact discrete solution to
  dp/dz = b can be computed for p.
  """
  nx, ny, nz = rho_thermal_xxc.shape
  assert rho_ref_xxf.shape == (1, 1, nz)

  # rho_thermal_xxf = interpolation.centered_node_to_face(rho_thermal_xxc, 2)
  rho_thermal_xxf = interpolation.z_c_to_f(rho_thermal_xxc)

  # Want to deal with 1D calculation, not 3D.  Then pass to 3D at the end.

  # Perform the calculation in numpy.
  rho_thermal_xxf = np.array(rho_thermal_xxf[0:1, 0:1, :])  # shape (1, 1, nz)
  rho_ref_xxf = np.array(rho_ref_xxf)  # shape (1, 1, nz)

  b_f = -G * (rho_thermal_xxf - rho_ref_xxf) / rho_thermal_xxf
  # Compute the buoyancy multiplied by the grid spacing.
  hz_face_key = stretched_grid_util.hf_key(dim=2)
  # Use stretched grid spacing if it exists, else use constant grid spacing dz.
  h_face = sg_map.get(hz_face_key, dz)

  # Multiply by dz.
  b_f = b_f * h_face  # shape (1, 1, nz)
  b_f = np.squeeze(b_f)  #  shape (nz)

  # Extract the halos, start at b^f_1.
  b_f = b_f[halo_width + 1 :]  # shape (nz - 3) (for halo_width=2)

  # p^c_k = p^c_0 + sum_{j=1}^k b^f_k.
  #   Setting p^c_0 = 0, we have
  # p^c_1 = dz * b^f_1
  # p^c_2 = dz * (b^f_1 + b^f_2)
  # p^c_3 = dz * (b^f_1 + b^f_2 + b^f_3)

  # Suppose b^f = [b^f_1, b^f_2, b^f_3, ...]
  # np.cumsum(b^f) = [b^f_1, b^f_1 + b^f_2, b^f_1 + b^f_2 + b^f_3, ...]
  p_c = np.cumsum(b_f)
  # We could add back p^c_0 here, but since it is zero, skip this step.

  # Add back the halos, massaging the 1D structure.
  p_c = np.concatenate((np.zeros(halo_width + 1), p_c))  # shape is (nz).

  p_c = p_c[np.newaxis, np.newaxis, :]  # Shape is (1, 1, nz).
  # Tile to 3D, and convert to a Jax array.
  p_c = jnp.tile(p_c, reps=(nx, ny, 1))  # (nx, ny, nz).

  return p_c
