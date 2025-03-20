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

"""2D Taylor-Green vortex problem."""

import dataclasses
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from swirl_jatmos import config
from swirl_jatmos import convection_config
from swirl_jatmos import timestep_control_config

Array: TypeAlias = jax.Array


@dataclasses.dataclass(frozen=True)
class ProblemSpec:
  """Specification for the 2D Taylor-Green Vortex problem."""

  domain_size: float
  viscosity: float  # Kinematic viscosity.
  amplitude: float  # Amplitude of the initial velocity.


def analytical_solution_u(
    x: Array, y: Array, t: float, spec: ProblemSpec
) -> Array:
  """Gets the analytical solution for the 2D Taylor-Green vortex problem."""
  domain_size = spec.domain_size
  viscosity = spec.viscosity
  u0 = spec.amplitude

  tc = (domain_size / (2 * np.pi)) ** 2 / (2 * viscosity)
  u = (
      u0
      * jnp.sin(2.0 * np.pi * x / domain_size)
      * jnp.cos(2.0 * np.pi * y / domain_size)
      * jnp.exp(-t / tc)
  )
  return u


def analytical_solution_v(
    x: Array, y: Array, t: float, spec: ProblemSpec
) -> Array:
  """Gets the analytical solution for the 2D Taylor-Green vortex problem."""
  domain_size = spec.domain_size
  viscosity = spec.viscosity
  u0 = spec.amplitude

  tc = (domain_size / (2 * np.pi)) ** 2 / (2 * viscosity)
  v = (
      -u0
      * jnp.cos(2.0 * np.pi * x / domain_size)
      * jnp.sin(2.0 * np.pi * y / domain_size)
      * jnp.exp(-t / tc)
  )
  return v


def analytical_solution_p(
    x: Array, y: Array, t: float, spec: ProblemSpec
) -> Array:
  """Gets the analytical solution for the 2D Taylor-Green vortex problem."""
  domain_size = spec.domain_size
  viscosity = spec.viscosity
  u0 = spec.amplitude

  tc = (domain_size / (2 * np.pi)) ** 2 / (2 * viscosity)
  p = (
      u0**2
      / 4.0
      * (
          jnp.cos(4 * np.pi * x / domain_size)
          + jnp.cos(4 * np.pi * y / domain_size)
      )
      * jnp.exp(-2 * t / tc)
  )
  return p


def analytical_solution_u_v_p(
    xx_nodes, xx_faces, yy_nodes, yy_faces, t: float, spec: ProblemSpec
) -> tuple[Array, Array, Array]:
  """Gets the analytical solution for the 2D Taylor-Green vortex problem."""
  u_fc = analytical_solution_u(xx_faces, yy_nodes, t, spec)
  v_cf = analytical_solution_v(xx_nodes, yy_faces, t, spec)
  p_cc = analytical_solution_p(xx_nodes, yy_nodes, t, spec)
  return u_fc, v_cf, p_cc


def get_cfg(
    n_per_core: int,
    dt: float,
    spec: ProblemSpec,
    poisson_solver_type: config.PoissonSolverType,
    convection_scheme: Literal['quick', 'upwind1', 'weno3', 'weno5'] = 'quick',
    stretched_grid_path_x: str = '',
    stretched_grid_path_y: str = '',
) -> config.ConfigExternal:
  domain_size = spec.domain_size
  viscosity = spec.viscosity

  return config.ConfigExternal(
      cx=1,
      cy=1,
      cz=1,
      nx=n_per_core,
      ny=n_per_core,
      # Need nz >= 3 points because z is required to have halo_width=1. For some
      # reason, nz=3 leads to NaNs in the unit test, while nz=4 works properly.
      nz=4,
      domain_x=(0, domain_size),
      domain_y=(0, domain_size),
      # Caution: if dz is too small (eg 1e-4) there appear to be floating point
      # truncation errors in the Jacobi solver, and convergence is not as good
      # as it should be.  Make z domain large to avoid very small dz.
      domain_z=(0, 1),
      halo_width=1,
      dt=dt,
      timestep_control_cfg=timestep_control_config.TimestepControlConfig(
          disable_adaptive_timestep=True
      ),
      convection_cfg=convection_config.ConvectionConfig(
          momentum_scheme=convection_scheme,
      ),
      use_sgs=False,
      stretched_grid_path_x=stretched_grid_path_x,
      stretched_grid_path_y=stretched_grid_path_y,
      include_buoyancy=False,
      poisson_solver_type=poisson_solver_type,
      # poisson_solver_type=config.PoissonSolverType.JACOBI,
      viscosity=viscosity,
  )
