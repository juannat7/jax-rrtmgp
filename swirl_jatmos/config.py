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

"""Jatmo configuration."""

import dataclasses
import enum

from absl import logging
import dataclasses_json
from etils import epath
import jax
import numpy as np
from swirl_jatmos import convection_config
from swirl_jatmos import derivatives
from swirl_jatmos import sponge_config
from swirl_jatmos import stretched_grid_init
from swirl_jatmos import timestep_control_config
from swirl_jatmos.boundary_conditions import boundary_conditions
from swirl_jatmos.microphysics import microphysics_config
from swirl_jatmos.rrtmgp.config import radiative_transfer
from swirl_jatmos.thermodynamics import water
from swirl_jatmos.utils import file_io
from swirl_jatmos.utils import text_util
from swirl_jatmos.utils import utils


class PoissonSolverType(enum.Enum):
  """Type of Poisson solver to use."""

  JACOBI = 'jacobi'
  FAST_DIAGONALIZATION = 'fast_diagonalization'


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigExternal(dataclasses_json.DataClassJsonMixin):
  """Config that is external to the simulation."""

  # Number of cores in each dimension.
  cx: int
  cy: int
  cz: int
  # Number of grid points in each dimension (including z halos).
  nx: int
  ny: int
  nz: int
  # Domain end points in each dimension.
  domain_x: tuple[float, float]
  domain_y: tuple[float, float]
  domain_z: tuple[float, float]
  halo_width: int = 1  # This is the only halo width allowed.
  dt: float  # Initial timestep [s].
  timestep_control_cfg: timestep_control_config.TimestepControlConfig

  wp: water.WaterParams = water.WaterParams()  # Water thermodynamics params.
  use_sgs: bool  # Whether to enable the subgrid-scale turbulence model.

  stretched_grid_path_x: str = ''
  stretched_grid_path_y: str = ''
  stretched_grid_path_z: str = ''

  # Boundary conditions in z.
  z_bcs: boundary_conditions.ZBoundaryConditions = (
      boundary_conditions.ZBoundaryConditions()
  )
  # Convection scheme.
  convection_cfg: convection_config.ConvectionConfig = (
      convection_config.ConvectionConfig()
  )

  # Microphysics config.
  microphysics_cfg: microphysics_config.MicrophysicsConfig = (
      microphysics_config.MicrophysicsConfig()
  )

  # If True, solve the pressure Poisson equation only on the last RK3 stage.
  # This should not be used with the Jacobi solver.
  solve_pressure_only_on_last_rk3_stage: bool = False
  include_buoyancy: bool = True
  include_qt_sedimentation: bool = False
  # If True, enable a sponge at the top of the domain that damps velocity.
  sponge_cfg: sponge_config.SpongeConfig | None = None
  uniform_y_2d: bool = False  # If True, assume 2D simulation with uniform y.
  poisson_solver_type: PoissonSolverType = (
      PoissonSolverType.FAST_DIAGONALIZATION
  )

  # Optional: RRTMGP radiative transfer config.
  radiative_transfer_cfg: radiative_transfer.RadiativeTransfer | None = None

  # Note: would use type annotation of Sequence[str], but that leads to an error
  # with dataclasses_json when deserializing.
  aux_output_fields: tuple[str, ...] = tuple()
  diagnostic_fields: tuple[str, ...] = tuple()

  # Physical constants
  viscosity: float = 1e-3  # Kinematic viscosity [m^2/s].
  diffusivity: float = 1e-3  # Scalar diffusivity [m^2/s].

  # If True, clip the diffusivity and viscosity to a maximum value determined
  # by stability limits. Currently, only the the z grid spacing is used to
  # determine the stability limit, which is sufficient for standard atmospheric
  # simulations.
  enforce_max_diffusivity: bool = False

  #  ******************** Checkpointing options ********************
  # Save a checkpoint every N cycles.  Diagnostics are saved every cycle.
  checkpoint_cycle_interval: int = 1
  # If True, driver saves no output (checkpoints or diagnostics) to disk.
  disable_checkpointing: bool = False


def _should_write_file() -> bool:
  """Returns True if this process should write a file."""
  # For multihost, ensure only one host saves the config.
  return jax.process_index() == 0


def save_json(cfgext: ConfigExternal, output_dir: str):
  """Save a ConfigExternal to a JSON file."""
  if not _should_write_file():
    return

  logging.info('Saving config in json format')
  dirpath = epath.Path(output_dir)
  dirpath.mkdir(mode=0o775, parents=True, exist_ok=True)
  filepath = dirpath / 'cfg.json'
  filepath.write_text(cfgext.to_json(indent=2))


def load_json(output_dir: str) -> ConfigExternal:
  """Load a ConfigExternal from a JSON file."""
  path = epath.Path(output_dir) / 'cfg.json'
  return ConfigExternal.from_json(path.read_text())


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
  """Jatmo configuration."""

  cx: int
  cy: int
  cz: int
  nx: int
  ny: int
  nz: int
  domain_x: tuple[float, float]
  domain_y: tuple[float, float]
  domain_z: tuple[float, float]
  halo_width: int
  dt: float
  timestep_control_cfg: timestep_control_config.TimestepControlConfig

  wp: water.WaterParams
  use_sgs: bool

  x_c: np.ndarray
  x_f: np.ndarray
  y_c: np.ndarray
  y_f: np.ndarray
  z_c: np.ndarray
  z_f: np.ndarray

  use_stretched_grid: tuple[bool, bool, bool]
  grid_spacings: tuple[float, float, float]
  deriv_lib: derivatives.Derivatives

  # Grid spacings.  If no stretched grid in a dimension, then h=constant array.
  hx_c: np.ndarray
  hx_f: np.ndarray
  hy_c: np.ndarray
  hy_f: np.ndarray
  hz_c: np.ndarray
  hz_f: np.ndarray

  z_bcs: boundary_conditions.ZBoundaryConditions
  convection_cfg: convection_config.ConvectionConfig
  microphysics_cfg: microphysics_config.MicrophysicsConfig

  # (Experimental) If True, solve the pressure Poisson equation only on the last
  # RK3 stage. This should not be used with the Jacobi solver.
  solve_pressure_only_on_last_rk3_stage: bool = False
  include_buoyancy: bool = True
  include_qt_sedimentation: bool = False
  sponge_cfg: sponge_config.SpongeConfig | None = None
  uniform_y_2d: bool = False  # If True, assume 2D simulation with uniform y.

  poisson_solver_type: PoissonSolverType = (
      PoissonSolverType.FAST_DIAGONALIZATION
  )

  radiative_transfer_cfg: radiative_transfer.RadiativeTransfer | None = None
  aux_output_fields: tuple[str, ...]
  diagnostic_fields: tuple[str, ...]
  checkpoint_cycle_interval: int
  disable_checkpointing: bool

  # Physical constants
  viscosity: float  # Kinematic viscosity [m^2/s].
  diffusivity: float  # Scalar diffusivity [m^2/s].

  enforce_max_diffusivity: bool

  # Checkpointing options
  checkpoint_cycle_interval: int
  disable_checkpointing: bool


def coordinates_and_h_factors_from_config(
    domain: tuple[float, float],
    num_cores: int,
    n: int,
    periodic: bool,
    stretched_grid_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Create coordinate and stretched grid arrays from config."""
  halo_width = 0 if periodic else 1

  if stretched_grid_path:
    q_c_interior = _load_array_from_file(stretched_grid_path)
    assert len(q_c_interior) == num_cores * n - 2 * halo_width, (
        f'Number of grid points {len(q_c_interior)} given in stretched grid'
        ' file does not match required number of grid points'
        f'={num_cores * n - 2 * halo_width} arising from {num_cores=} &'
        f' {n=} with {halo_width=}.'
    )
    if periodic:
      q_c, q_f = stretched_grid_init.create_periodic_grid(q_c_interior, domain)
      h_c, h_f = stretched_grid_init.create_periodic_h(q_c, q_f, domain)
    else:
      q_c, q_f = stretched_grid_init.create_nonperiodic_grid(
          q_c_interior, domain
      )
      h_c, h_f = stretched_grid_init.create_nonperiodic_h(q_c, q_f)
  else:
    q_c, q_f = utils.uniform_grid(domain, num_cores, n, halo_width)
    h_c, h_f = np.ones_like(q_c), np.ones_like(q_c)
  return q_c, q_f, h_c, h_f


def config_from_config_external(
    cfgext: ConfigExternal,
) -> Config:
  """Create a Config from a ConfigExternal."""
  use_stretched_grid = (
      True if cfgext.stretched_grid_path_x else False,
      True if cfgext.stretched_grid_path_y else False,
      True if cfgext.stretched_grid_path_z else False,
  )

  # Here, assume x,y are periodic, and z is nonperiodic.
  periodic = (True, True, False)
  xyz_c, xyz_f, hxyz_c, hxyz_f, grid_spacings_list = [], [], [], [], []
  for dim in (0, 1, 2):
    domain = (cfgext.domain_x, cfgext.domain_y, cfgext.domain_z)[dim]
    num_cores = (cfgext.cx, cfgext.cy, cfgext.cz)[dim]
    n = (cfgext.nx, cfgext.ny, cfgext.nz)[dim]
    stretched_grid_path = (
        cfgext.stretched_grid_path_x,
        cfgext.stretched_grid_path_y,
        cfgext.stretched_grid_path_z,
    )[dim]
    q_c, q_f, h_c, h_f = coordinates_and_h_factors_from_config(
        domain, num_cores, n, periodic[dim], stretched_grid_path
    )
    xyz_c.append(q_c)
    xyz_f.append(q_f)
    hxyz_c.append(h_c)
    hxyz_f.append(h_f)
    if use_stretched_grid[dim]:
      grid_spacings_list.append(1.0)
    elif len(q_c) == 1:
      # Handle the special case where there is only one grid point in this dim.
      grid_spacings_list.append(1.0)
    else:
      grid_spacings_list.append(float(q_c[1] - q_c[0]))
    grid_spacings = tuple(grid_spacings_list)

  deriv_lib = derivatives.Derivatives(grid_spacings, use_stretched_grid)

  return Config(
      cx=cfgext.cx,
      cy=cfgext.cy,
      cz=cfgext.cz,
      nx=cfgext.nx,
      ny=cfgext.ny,
      nz=cfgext.nz,
      domain_x=cfgext.domain_x,
      domain_y=cfgext.domain_y,
      domain_z=cfgext.domain_z,
      halo_width=cfgext.halo_width,
      dt=cfgext.dt,
      timestep_control_cfg=cfgext.timestep_control_cfg,
      wp=cfgext.wp,
      use_sgs=cfgext.use_sgs,
      x_c=xyz_c[0],
      x_f=xyz_f[0],
      y_c=xyz_c[1],
      y_f=xyz_f[1],
      z_c=xyz_c[2],
      z_f=xyz_f[2],
      use_stretched_grid=use_stretched_grid,
      grid_spacings=grid_spacings,
      deriv_lib=deriv_lib,
      hx_c=hxyz_c[0],
      hx_f=hxyz_f[0],
      hy_c=hxyz_c[1],
      hy_f=hxyz_f[1],
      hz_c=hxyz_c[2],
      hz_f=hxyz_f[2],
      z_bcs=cfgext.z_bcs,
      convection_cfg=cfgext.convection_cfg,
      microphysics_cfg=cfgext.microphysics_cfg,
      solve_pressure_only_on_last_rk3_stage=cfgext.solve_pressure_only_on_last_rk3_stage,
      include_buoyancy=cfgext.include_buoyancy,
      include_qt_sedimentation=cfgext.include_qt_sedimentation,
      sponge_cfg=cfgext.sponge_cfg,
      uniform_y_2d=cfgext.uniform_y_2d,
      poisson_solver_type=cfgext.poisson_solver_type,
      radiative_transfer_cfg=cfgext.radiative_transfer_cfg,
      aux_output_fields=cfgext.aux_output_fields,
      diagnostic_fields=cfgext.diagnostic_fields,
      viscosity=cfgext.viscosity,
      diffusivity=cfgext.diffusivity,
      enforce_max_diffusivity=cfgext.enforce_max_diffusivity,
      checkpoint_cycle_interval=cfgext.checkpoint_cycle_interval,
      disable_checkpointing=cfgext.disable_checkpointing,
  )


def _load_array_from_file(path: str) -> np.ndarray:
  """Loads a 1D array from a text file and returns it as a numpy array.

  Each element of the array must be on its own line.

  Args:
    path: The path to the file to load.

  Returns:
    A 1D numpy array of the data from the file.
  """
  contents = file_io.load_from_path(path)
  contents = text_util.strip_line_comments(contents, '#')
  return np.fromstring(contents, sep='\n')
