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

"""Stretched grid initialization.

Note:
- For nonperiodic dimensions, we assume halo_width = 1.
- For periodic dimensions, we assume halo_width = 0.
"""

import numpy as np
import numpy.typing as npt


def create_nonperiodic_grid(
    q_c_interior: npt.ArrayLike, domain: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
  """Create grid on nodes and faces for a nonperiodic dimension."""
  # Assuming halo_width = 1 for nonperiodic dimensions.
  halo_width = 1
  q_c_interior = np.asarray(q_c_interior)
  assert np.all(
      q_c_interior[:-1] < q_c_interior[1:]
  ), 'q_c_interior must be sorted in ascending order without duplicates.'

  rtol = 1e-8
  atol = 1e-8
  distance_from_wall_left = q_c_interior[0] - domain[0]
  distance_from_wall_right = domain[1] - q_c_interior[-1]

  # Check for a common mistake by ensuring that the first and last nodes of the
  # interior are separated from the wall by some small amount, and that they are
  # not accidentally overlapping the wall.
  assert (
      distance_from_wall_left > rtol * domain[0] + atol
  ), 'q_c_interior[0] is not greater than left side of domain'
  assert (
      distance_from_wall_right > rtol * domain[1] + atol
  ), 'q_c_interior[-1] is not less than right side of domain'
  assert len(q_c_interior.shape) == 1

  # Produce q_c.
  n_interior = len(q_c_interior)
  n = n_interior + 2 * halo_width
  q_c = np.zeros(n, dtype=q_c_interior.dtype)

  dq_first = 2 * (q_c_interior[0] - domain[0])
  dq_last = 2 * (domain[1] - q_c_interior[-1])

  q_c[halo_width:-halo_width] = q_c_interior
  ext = np.arange(1, halo_width + 1)
  q_c[:halo_width] = q_c_interior[0] - dq_first * ext
  q_c[-halo_width:] = q_c_interior[-1] + dq_last * ext

  # Produce q_f.
  q_f = np.zeros_like(q_c)
  q_f[1:] = (q_c[0:-1] + q_c[1:]) / 2
  q_f[0] = q_c[0] - dq_first / 2
  return q_c, q_f


def create_periodic_grid(
    q_c_interior: npt.ArrayLike, domain: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
  """Create grid on nodes and faces for a periodic dimension."""
  # Assuming halo_width = 0 for periodic dimensions.
  q_c_interior = np.asarray(q_c_interior)
  assert np.all(
      q_c_interior[:-1] < q_c_interior[1:]
  ), 'q_c_interior must be sorted in ascending order without duplicates.'
  assert q_c_interior[0] > domain[0] and q_c_interior[-1] < domain[1]

  assert len(q_c_interior.shape) == 1

  eps = 1e-6
  dist_from_wall_left = q_c_interior[0] - domain[0]
  dist_from_wall_right = domain[1] - q_c_interior[-1]
  if np.abs(dist_from_wall_left - dist_from_wall_right) >= eps:
    raise ValueError(
        'For consistency in the periodic domain, the distance of the first node'
        ' to the left wall must be equal to the distance of the last node to'
        ' the right wall.'
    )

  q_c = q_c_interior  # Already complete on nodes.

  # Produce q_f
  q_f = np.zeros_like(q_c)
  q_f[1:] = (q_c[0:-1] + q_c[1:]) / 2
  q_f[0] = domain[0]
  return q_c, q_f


def create_nonperiodic_h(
    q_c: np.ndarray, q_f: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Compute scale factors for a nonperiodic dimension."""
  assert q_c.shape == q_f.shape
  h_c = np.zeros_like(q_c)
  h_c[0:-1] = q_f[1:] - q_f[0:-1]

  dq_last = q_c[-1] - q_c[-2]
  h_c[-1] = dq_last

  # Produce h_f.
  h_f = np.zeros_like(q_c)
  h_f[1:] = q_c[1:] - q_c[0:-1]
  dq_first = q_c[1] - q_c[0]
  h_f[0] = dq_first
  return h_c, h_f


def create_periodic_h(
    q_c: np.ndarray, q_f: np.ndarray, domain: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
  """Compute scale factors for a periodic dimension."""
  assert q_c.shape == q_f.shape
  h_c = np.zeros_like(q_c)
  h_c[0:-1] = q_f[1:] - q_f[0:-1]
  h_c[-1] = domain[1] - q_f[-1]

  h_f = np.zeros_like(q_c)
  h_f[1:] = q_c[1:] - q_c[0:-1]
  dq_first = 2 * (q_c[0] - domain[0])
  h_f[0] = dq_first
  return h_c, h_f
