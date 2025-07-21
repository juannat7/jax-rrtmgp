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

"""A base loader for the RRTMGP lookup tables in netCDF format."""

from collections.abc import Sequence
import os
from typing import TypeAlias

from pathlib import Path
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np

Array: TypeAlias = jax.Array

_NETCDF_DATA_DIR = '/tmp/netcdf/data'


def _bytes_to_str(split_str):
  return str(split_str, 'utf-8').strip()


def create_index(name_arr: Sequence[str]) -> dict[str, int]:
  """Utility function for generating an index from a sequence of names."""
  return {name: idx for idx, name in enumerate(name_arr)}


def _create_local_file(filepath: str) -> str:
  """Copies remote files locally so they can be ingested by netCDF reader."""
  # Create local directory.
  if os.path.exists(_NETCDF_DATA_DIR):
    assert os.path.isdir(_NETCDF_DATA_DIR)
  else:
    os.makedirs(_NETCDF_DATA_DIR)
  local_filename = os.path.join(_NETCDF_DATA_DIR, os.path.basename(filepath))
  # Copy the file from remote location if not already present.
  if os.path.exists(local_filename):
    assert os.path.isfile(local_filename)
  else:
    path = Path(local_filename)
    path.write_bytes(Path(filepath).read_bytes())
  return local_filename


def parse_nc_file(
    path: str,
) -> tuple[nc.Dataset, dict[str, Array], dict[str, int]]:
  """Utility functions for unpacking RRTMGP files and loading arrays.

  Args:
    path: Full path of the netCDF dataset file.

  Returns:
    A 3-tuple of 1) the original netCDF Dataset, 2) a dictionary containing the
    data as Arrays, and 3) a dictionary of dimensions.
  """
  local_path = _create_local_file(path)
  ds = nc.Dataset(local_path, 'r')

  array_dict = {}
  dim_map = {k: v.size for k, v in ds.dimensions.items()}

  for key in ds.variables:
    val = ds[key][:].data
    if val.dtype == np.dtype('S1'):
      # The S1 dtype string arrays are not needed here, because later on they
      # are pulled directly from the Dataset object.  JAX cannot convert this
      # datatype to arrays anyway, so skip these data types.
      continue
    if np.issubdtype(val.dtype, np.floating):
      dtype = jnp.float_
    elif np.issubdtype(val.dtype, np.integer):
      dtype = jnp.int_
    else:
      raise ValueError(f'Unexpected dtype: {val.dtype}')
    array_dict[key] = jnp.array(val, dtype=dtype)
  return ds, array_dict, dim_map
