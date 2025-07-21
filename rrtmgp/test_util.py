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

"""Library for common operations used in tests."""
import os
from typing import Literal, TypeAlias

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

Array: TypeAlias = jax.Array


def extract_1d_slice_in_dim(
    f_3d: Array, dim: Literal[0, 1, 2], other_idx: int
) -> Array:
  """Extracts 1D slice of `f_3d` along `dim`.

  For example, if dim == 1 and other_idx == 4, return f[4, :, 4].

  Args:
    f_3d: The 3D array to extract the slice from.
    dim: The dimension along which to extract the slice.
    other_idx: The other indices to extract the slice from.

  Returns:
    The 1D slice of `f_3d` along `dim`.
  """
  observe_slice = [other_idx, other_idx, other_idx]
  observe_slice[dim] = slice(None)
  f_1d_slice = f_3d[tuple(observe_slice)]
  return f_1d_slice


def convert_to_3d_array_and_tile(
    f_1d: Array, dim: Literal[0, 1, 2], num_repeats: int
) -> Array:
  """Converts 1D array `f_1d` to a tiled 3D array.

  For example, if len(f_1d) == 8, dim == 1, and num_repeats == 4, then
  f.shape = (4, 8, 4).

  Args:
    f_1d: The 1D tensor to convert to 3D.
    dim: The dimension along which `f_1d` is laid out.
    num_repeats: The number of times to repeat the array in dimensions other
      than `dim`.

  Returns:
    The 3D tensor `f_3d`, where f_3d.shape[dim] == len(f_1d), and
      f_3d.shape[j] == num_repeats for j != dim.
  """
  # Convert f_1d to 3D tensor f where the direction of variation is along
  # dim. Result: f.shape[dim] = len(f_1d), and f.shape[j] = 1 for j != dim.
  slices = [jnp.newaxis, jnp.newaxis, jnp.newaxis]
  slices[dim] = slice(None)
  f = f_1d[tuple(slices)]

  # Tile f to a 3D tensor that is repeated in other dimensions.
  repeats = [num_repeats, num_repeats, num_repeats]
  repeats[dim] = 1
  f_3d = jnp.tile(f, reps=repeats)
  return f_3d


def save_1d_array_to_tempfile(
    test: absltest.TestCase, array: npt.ArrayLike
) -> str:
  """Saves a 1D array to a tempfile and returns the path to the tempfile."""
  tempfile = test.create_tempfile()
  fname = os.path.join(tempfile)
  np.savetxt(fname, array)
  return fname


def l_infinity_norm(v: npt.ArrayLike) -> float:
  return np.max(np.abs(v))


def l_infinity_error(field1: npt.ArrayLike, field2: npt.ArrayLike) -> float:
  err = field1 - field2
  return l_infinity_norm(err)


def compute_power_exponent(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
  """Estimates a power-law exponent through regression.

  Assume that x, y have a power-law relationship, y = C * x^p. Estimate p
  through linear regression, using log y = log C + p log x.

  Args:
    x: Independent variable.
    y: Dependent variable.

  Returns:
    The exponent p characterizing the power law.
  """
  logy = np.log(y)
  logx = np.log(x)
  return np.polyfit(logx, logy, 1)[0]
