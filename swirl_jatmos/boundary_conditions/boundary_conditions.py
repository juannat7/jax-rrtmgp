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

"""Module for specifying boundary conditions."""

import dataclasses
from typing import Literal, TypeAlias

import dataclasses_json  # Used for JSON serialization.
from swirl_jatmos.boundary_conditions import monin_obukhov

MoninObukhovParameters: TypeAlias = monin_obukhov.MoninObukhovParameters


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZBC(dataclasses_json.DataClassJsonMixin):
  """Parameters determining the z boundary condition for one face."""

  # Description of 'no_flux' BC:
  #   The z boundaries are viewed as hard, free-slip walls.  The fluid w
  #   velocity is set to zero there. Thus, the convective flux is generally
  #   zero, *except* that some scalars (q_r, q_s, and sometimes q_t) can have a
  #   nonzero sedimentation velocity, even on the bottom wall, which causes
  #   scalar flux out of the system. For u, v, and theta_li, there is no
  #   vertical convective flux at the boundary.  Furthermore, for u, v, and all
  #   scalars, there is no vertical diffusive flux at the boundary.
  # Description of 'monin_obukhov' BC:
  #   A similarity theory is used to model the bottom boundary condition, with
  #   a logarithmic velocity profile.  Vertical fluxes of the horizontal
  #   momentum are prescribed (tau_02, tau_12), assuming the horizontal velocity
  #   is zero at the wall itself.  Scalar fluxes of theta_li and q_t are also
  #   prescribed, assuming some surface temperature and surface humidity values.
  #   The flux of theta_li is a heat source into the system (sensible heat
  #   flux).  The flux of q_t is a moisture source into the system -- an
  #   evaporative flux, corresponding to a latent heat flux.  In general, the
  #   coefficients determining these fluxes are determined nonlinearly by the
  #   Monin-Obukhov similarity theory, but for simplicity an alternative option
  #   is to specify fixed exchange coefficients.  The Monin-Obukhov-determined
  #   fluxes are set as the 'diffusive' fluxes, while the convective fluxes are
  #   what they would be as described above in the 'no_flux' case, because 'w'
  #   is still zero at the boundary (but there can be nonzero sedimentation
  #   velocities).
  bc_type: Literal['no_flux', 'monin_obukhov'] = 'no_flux'
  mop: MoninObukhovParameters | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZBoundaryConditions(dataclasses_json.DataClassJsonMixin):
  """Parameters determining the boundary conditions for a fluid simulation."""

  bottom: ZBC = ZBC()
  top: ZBC = ZBC()

  def __post_init__(self):
    # Validate inputs.
    if self.top.bc_type == 'monin_obukhov':
      raise ValueError('Monin-Obukhov BC cannot be used for the top boundary.')
    if self.bottom.bc_type == 'monin_obukhov' and self.bottom.mop is None:
      raise ValueError(
          'Monin-Obukhov parameters must be provided for the bottom boundary'
          ' when using the Monin-Obukhov BC.'
      )
