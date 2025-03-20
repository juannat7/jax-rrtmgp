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

"""Module for initializing a simulation."""

import functools
from typing import Literal, TypeAlias

from absl import flags
import jax
import jax.numpy as jnp
from swirl_jatmos import common_ops
from swirl_jatmos import config
from swirl_jatmos import stretched_grid_util

Array: TypeAlias = jax.Array
PartitionSpec: TypeAlias = jax.sharding.PartitionSpec
NamedSharding: TypeAlias = jax.sharding.NamedSharding

_ALLOW_SPLIT_PHYSICAL_AXES = flags.DEFINE_bool(
    'allow_split_physical_axes',
    False,
    'If true, allow splitting physical axes when creating the sharding mesh.',
    allow_override=True,
)


def get_sharding_mesh(cx: int, cy: int, cz: int) -> jax.sharding.Mesh:
  """Get the sharding mesh."""
  if not _ALLOW_SPLIT_PHYSICAL_AXES.value:
    return jax.make_mesh((cx, cy, cz), ('x', 'y', 'z'))
  else:
    # Note: Unlike jax.make_mesh, this command fails if the number of requested
    # devices is not equal to the number of available devices. E.g., if there
    # are 8 devices and cx=cy=cz=1, this command fails.
    devices = jax.experimental.mesh_utils.create_device_mesh(
        (cx, cy, cz), allow_split_physical_axes=True
    )
    mesh = jax.sharding.Mesh(devices, ('x', 'y', 'z'))
  return mesh


def shard_arr(
    arr: Array,
    cfg: config.Config,
    dim_names: Literal['', 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'],
) -> Array:
  """Shard a 0D, 1D, 2D, or 3D array across cores."""
  mesh = get_sharding_mesh(cfg.cx, cfg.cy, cfg.cz)
  partition_specs = {
      '': PartitionSpec(),
      'x': PartitionSpec('x'),
      'y': PartitionSpec('y'),
      'z': PartitionSpec('z'),
      'xy': PartitionSpec('x', 'y'),
      'xz': PartitionSpec('x', 'z'),
      'yz': PartitionSpec('y', 'z'),
      'xyz': PartitionSpec('x', 'y', 'z'),
  }
  spec = partition_specs[dim_names]
  sharding = NamedSharding(mesh, spec)
  return jax.lax.with_sharding_constraint(arr, sharding)


def shard_broadcastable_arr(
    arr: Array,
    cfg: config.Config,
    dim_names: Literal['x', 'y', 'z', 'xy', 'xz', 'yz'],
) -> Array:
  """Shard a braodcastable 1D or 2D array across cores.

  The 1D arrays have shape (nx, 1, 1), (1, ny, 1), or (1, 1, nz).
  The 2D arrays have shape (nx, ny, 1) or (nx, 1, nz), or (1, ny, nz).

  Args:
    arr: The array to shard.
    cfg: The config object.
    dim_names: The dimension names to shard across.

  Returns:
    The sharded, broadcastable array.
  """
  mesh = get_sharding_mesh(cfg.cx, cfg.cy, cfg.cz)
  partition_specs = {
      'x': PartitionSpec('x', None, None),
      'y': PartitionSpec(None, 'y', None),
      'z': PartitionSpec(None, None, 'z'),
      'xy': PartitionSpec('x', 'y', None),
      'xz': PartitionSpec('x', None, 'z'),
      'yz': PartitionSpec(None, 'y', 'z'),
  }
  spec = partition_specs[dim_names]
  sharding = NamedSharding(mesh, spec)
  return jax.lax.with_sharding_constraint(arr, sharding)


# Convenience definitions for 0d and 3d arrays since the dim names do not need
# to be specified.
def shard_0d(arr: Array, cfg: config.Config) -> Array:
  """Shard a 0D (scalar) array to be replicated across all cores."""
  return shard_arr(arr, cfg, '')


def shard_3d(arr: Array, cfg: config.Config) -> Array:
  """Shard a 3D array across all cores."""
  return shard_arr(arr, cfg, 'xyz')


def shard_2d_horiz(arr: Array, cfg: config.Config) -> Array:
  """Shard a horizontal 2D (x,y) array."""
  return shard_arr(arr, cfg, 'xy')


def convert_1d_array_to_jax_array_and_shard(
    u: jax.typing.ArrayLike,
    dim: Literal[0, 1, 2],
    cfg: config.Config,
    dtype: jax.typing.DTypeLike,
):
  """Convert a rank-1 array to a broadcastable, sharded Jax array."""
  # Convert input to Jax Array with specified dtype.
  u = jnp.array(u, dtype=dtype)
  # Reshape to a rank-3 broadcastable array, required for sharding.
  u = common_ops.reshape_to_broadcastable(u, dim)
  dim_name = ['x', 'y', 'z'][dim]
  return shard_broadcastable_arr(u, cfg, dim_name)


def initialize_grids(cfg: config.Config) -> dict[str, Array]:
  """Initialize sharded, broadcastable grid variables from config.

  Args:
    cfg: Config object.

  Returns:
    A dictionary of sharded grid variables, which includes the grid coordinates
    'x_c', 'x_f', 'y_c', 'y_f', 'z_c', 'z_f', and the stretched-grid h factors
    for the dimensions that are stretched.
  """
  dtype = jnp.float32
  shard_array = functools.partial(
      convert_1d_array_to_jax_array_and_shard, cfg=cfg, dtype=dtype
  )
  grid_map: dict[str, Array] = {}
  grid_map['x_c'] = shard_array(cfg.x_c, 0)
  grid_map['x_f'] = shard_array(cfg.x_f, 0)
  grid_map['y_c'] = shard_array(cfg.y_c, 1)
  grid_map['y_f'] = shard_array(cfg.y_f, 1)
  grid_map['z_c'] = shard_array(cfg.z_c, 2)
  grid_map['z_f'] = shard_array(cfg.z_f, 2)

  sgh_map: dict[str, Array] = {}
  for dim in (0, 1, 2):
    if cfg.use_stretched_grid[dim]:
      h_c = [cfg.hx_c, cfg.hy_c, cfg.hz_c][dim]
      h_f = [cfg.hx_f, cfg.hy_f, cfg.hz_f][dim]
      sgh_map[stretched_grid_util.hc_key(dim)] = shard_array(h_c, dim)
      sgh_map[stretched_grid_util.hf_key(dim)] = shard_array(h_f, dim)

  grid_map |= sgh_map
  return grid_map


def initialize_time_and_step_id(cfg: config.Config) -> dict[str, Array]:
  """Initialize time and step id variables."""
  step_id = jnp.array(0, dtype=jnp.int64)
  t_ns = jnp.array(0, dtype=jnp.int64)
  dt_ns = jnp.array(cfg.dt * 1e9, dtype=jnp.int64)
  all_valid = jnp.array(True, dtype=jnp.bool_)

  if t_ns.dtype != jnp.int64:
    raise ValueError(
        'x64 must be enabled for JAX.'
        " Please set `jax.config.update('jax_enable_x64', True)` in the binary."
    )

  step_id = shard_0d(step_id, cfg)
  t_ns = shard_0d(t_ns, cfg)
  dt_ns = shard_0d(dt_ns, cfg)
  all_valid = shard_0d(all_valid, cfg)
  return {
      'step_id': step_id,
      't_ns': t_ns,
      'dt_ns': dt_ns,
      'all_valid': all_valid,
  }


def initialize_zeros_from_varname(varname: str, cfg: config.Config) -> Array:
  """Initialize a sharded, zero-valued array based on the variable name.

  The variable name determines the shape and sharding of the array, based on the
  *ending* of the variable name.  The possible endings are:
    - '_0d',
    - '_1d_x', '1d_y', '1d_z',
    - '_2d_xy', '_2d_xz', '_2d_yz',
    - '_3d'
  If none of these endings are present, then it is assumed the array is 3D.

  The 1D and 2D arrays are NOT in broadcastable form (they are not rank-3
  arrays).

  The shape of the array is assumed to correspond to the full domain along the
  indicated dimensions, rather than some subset of the domain.

  The dtype of the array is float32 (not configurable currently).

  Args:
    varname: The name of the variable to initialize.  The ending of the name
      determines the shape and sharding of the array.
    cfg: The config object.

  Returns:
    A sharded, zero-valued array.
  """
  dtype = jnp.float32
  nx, ny, nz = len(cfg.x_c), len(cfg.y_c), len(cfg.z_c)
  if varname.endswith('_0d'):
    return shard_0d(jnp.array(0, dtype=dtype), cfg)
  elif varname.endswith('_1d_x'):
    return shard_arr(jnp.zeros(nx, dtype=dtype), cfg, 'x')
  elif varname.endswith('_1d_y'):
    return shard_arr(jnp.zeros(ny, dtype=dtype), cfg, 'y')
  elif varname.endswith('_1d_z'):
    return shard_arr(jnp.zeros(nz, dtype=dtype), cfg, 'z')
  elif varname.endswith('_2d_xy'):
    return shard_arr(jnp.zeros((nx, ny), dtype=dtype), cfg, 'xy')
  elif varname.endswith('_2d_xz'):
    return shard_arr(jnp.zeros((nx, nz), dtype=dtype), cfg, 'xz')
  elif varname.endswith('_2d_yz'):
    return shard_arr(jnp.zeros((ny, nz), dtype=dtype), cfg, 'yz')
  elif varname.endswith('_3d'):
    return shard_3d(jnp.zeros((nx, ny, nz), dtype=dtype), cfg)
  else:  # Default to 3D.
    return shard_3d(jnp.zeros((nx, ny, nz), dtype=dtype), cfg)
