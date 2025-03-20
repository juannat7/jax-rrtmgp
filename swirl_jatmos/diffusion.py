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

"""Compute diffusion terms for scalar.

Under the anelastic assumption, rho depends only on z, so rho_xxc is used
whenever the location is on a z-center, regardless of whether the location is on
a face or center in x or y.  Similarly, rho_xxf is used whenever the location is
on a z-face, regardless of whether the location is on a center or face in x or
y.

Let the full 3D fields have shape (nx, ny, nz).

rho_xxc and rho_xxf are assumed to typically be arrays of shape (1, 1, nz),
although anything that broadcasts to an (nx, ny, nz) array will be allowed.
"""

from typing import Literal, TypeAlias

import jax
from swirl_jatmos import derivatives
from swirl_jatmos import interpolation
from swirl_jatmos.boundary_conditions import apply_bcs
from swirl_jatmos.boundary_conditions import boundary_conditions
from swirl_jatmos.boundary_conditions import monin_obukhov
from swirl_jatmos.utils import utils


Array: TypeAlias = jax.Array


def enforce_no_flux_at_z_boundaries(
    flux_z_xxf: Array, halo_width: int
) -> Array:
  """Update fluxes to enforce a no-flux condition on the z boundaries."""
  flux_z_xxf = flux_z_xxf.at[:, :, halo_width].set(0.0)
  flux_z_xxf = flux_z_xxf.at[:, :, -halo_width].set(0.0)
  return flux_z_xxf


def diffusive_flux_scalar(
    rho_xxc: Array,
    rho_xxf: Array,
    phi_ccc: Array,
    diffusivity_ccc: Array,
    deriv_lib: derivatives.Derivatives,
    sg_map: dict[str, Array],
    u_fcc: Array,
    v_cfc: Array,
    z_bcs: boundary_conditions.ZBoundaryConditions,
    scalar_name: Literal['theta_li', 'q_t', 'q_r', 'q_s'],
    phi_surface_value: Array | None = None,
) -> tuple[Array, Array, Array]:
  """Compute diffusive flux -ρD∇ϕ for scalar ρϕ.

  An isotropic diffusion tensor is assumed.

  Take D to just be D = diffusivity * I, where diffusivity is a scalar (but
  can be a field, dependent on space), and I is the identity tensor.

  Flux boundary conditions are applied here.

  Args:
    rho_xxc: The density on z-centers.
    rho_xxf: The density on z-faces.
    phi_ccc: The scalar field on (ccc).
    diffusivity_ccc: The diffusivity on (ccc).
    deriv_lib: An instance of the derivatives library.
    sg_map: Dict holding stretched-grid states.
    u_fcc: The x velocity (used for Monin-Obukhov BCs).
    v_cfc: The y velocity (used for Monin-Obukhov BCs).
    z_bcs: Specification of z boundary conditions.
    scalar_name: The name of the scalar for which to compute the diffusive flux.
      Must be one of 'theta_li', 'q_t', 'q_r', or 'q_s'.
    phi_surface_value: The value of the scalar at the surface. A 2D array. Must
      be present if using the Monin-Obukhov BC.

  Returns:
    The diffusive flux, with boundary conditions applied.
  """
  diffusivity_fcc = interpolation.centered_node_to_face(diffusivity_ccc, 0)
  diffusivity_cfc = interpolation.centered_node_to_face(diffusivity_ccc, 1)
  diffusivity_ccf = interpolation.centered_node_to_face(diffusivity_ccc, 2)

  # Compute diffusive fluxes -ρD ∂ϕ/∂x_j evaluated on faces in each dim.
  flux_x_fcc = -rho_xxc * diffusivity_fcc * deriv_lib.dx_c_to_f(phi_ccc, sg_map)
  flux_y_cfc = -rho_xxc * diffusivity_cfc * deriv_lib.dy_c_to_f(phi_ccc, sg_map)
  flux_z_ccf = -rho_xxf * diffusivity_ccf * deriv_lib.dz_c_to_f(phi_ccc, sg_map)

  # Enforce boundary conditions on the diffusive fluxes.  Specifically, enforce
  # BCs on the vertical diffusive flux.

  if scalar_name in ['q_r', 'q_s']:
    # The only option available is to enforce 0 diffusive flux on both z
    # boundaries.
    flux_z_ccf = apply_bcs.enforce_flux_at_z_bottom_bdy(flux_z_ccf, 0.0)
    flux_z_ccf = apply_bcs.enforce_flux_at_z_top_bdy(flux_z_ccf, 0.0)
    return (flux_x_fcc, flux_y_cfc, flux_z_ccf)

  assert scalar_name in ['theta_li', 'q_t']
  # Bottom boundary.
  if z_bcs.bottom.bc_type == 'no_flux':
    flux_z_ccf = apply_bcs.enforce_flux_at_z_bottom_bdy(flux_z_ccf, 0.0)
  elif z_bcs.bottom.bc_type == 'monin_obukhov':
    assert z_bcs.bottom.mop is not None
    assert (
        phi_surface_value is not None
    ), 'scalar_surface_value must be provided for Monin-Obukhov BCs'
    flux_z_ccf_surface = monin_obukhov.surface_scalar_flux(
        phi_ccc, u_fcc, v_cfc, rho_xxc, phi_surface_value, z_bcs.bottom.mop
    )
    flux_z_ccf = apply_bcs.enforce_flux_at_z_bottom_bdy(
        flux_z_ccf, flux_z_ccf_surface
    )
  else:
    raise ValueError(f'Bad z_bcs.bottom.bc_type: {z_bcs.bottom.bc_type}')

  # Top boundary.
  if z_bcs.top.bc_type == 'no_flux':
    flux_z_ccf = apply_bcs.enforce_flux_at_z_top_bdy(flux_z_ccf, 0.0)
  else:
    raise ValueError(
        f'Unsupported z boundary condition type: {z_bcs.top.bc_type}'
    )
  return (flux_x_fcc, flux_y_cfc, flux_z_ccf)


def diffusive_fluxes_uvw(
    rho_xxc: Array,
    rho_xxf: Array,
    nu_ccc: Array,
    strain_rate_tensor: utils.StrainRateTensor,
    u_fcc: Array,
    v_cfc: Array,
    z_bcs: boundary_conditions.ZBoundaryConditions,
) -> tuple[Array, Array, Array, Array, Array, Array]:
  """Compute diffusive momentum fluxes.

  The 3 components of the diffusive flux of x-momentum:
    rho_u_flux_x, rho_u_flux_y, rho_u_flux_z
  The 3 components of the diffusive flux of y-momentum:
    rho_v_flux_x, rho_v_flux_y, rho_v_flux_z
  The 3 components of the diffusive flux of z-momentum:
    rho_w_flux_x, rho_w_flux_y, rho_w_flux_z

  Flux boundary conditions (no flux on z boundaries) are optionally enforced.

  Args:
    rho_xxc: The density of the fluid.
    rho_xxf: The density of the fluid.
    nu_ccc: The viscosity of the fluid.
    strain_rate_tensor: The strain rate tensor.
    u_fcc: The x velocity (used for Monin-Obukhov BCs).
    v_cfc: The y velocity (used for Monin-Obukhov BCs).
    z_bcs: Specification of z boundary conditions.

  Returns:
    The diffusive momentum fluxes of the stress tensor.
  """
  s00_ccc = strain_rate_tensor.s00_ccc
  s01_ffc = strain_rate_tensor.s01_ffc
  s02_fcf = strain_rate_tensor.s02_fcf
  s11_ccc = strain_rate_tensor.s11_ccc
  s12_cff = strain_rate_tensor.s12_cff
  s22_ccc = strain_rate_tensor.s22_ccc

  nu_fcc = interpolation.centered_node_to_face(nu_ccc, 0)
  nu_ffc = interpolation.centered_node_to_face(nu_fcc, 1)
  nu_fcf = interpolation.centered_node_to_face(nu_fcc, 2)
  nu_cfc = interpolation.centered_node_to_face(nu_ccc, 1)
  nu_cff = interpolation.centered_node_to_face(nu_cfc, 2)

  # Compute (x,y,z) components of diffusive fluxes of x-momentum.
  tau00_ccc = -2 * rho_xxc * nu_ccc * s00_ccc
  tau01_ffc = -2 * rho_xxc * nu_ffc * s01_ffc
  tau02_fcf = -2 * rho_xxf * nu_fcf * s02_fcf

  # Compute (x,y,z) components of diffusive fluxes of y-momentum.
  # Note that tau10_ffc = tau01_ffc by symmetry.
  tau11_ccc = -2 * rho_xxc * nu_ccc * s11_ccc
  tau12_cff = -2 * rho_xxf * nu_cff * s12_cff

  # Compute (x,y,z) components of diffusive fluxes of z-momentum.
  # Note that tau20_fcf = tau02_fcf & tau21_cff = tau12_cff by symmetry.
  tau22_ccc = -2 * rho_xxc * nu_ccc * s22_ccc

  # Enforce boundary conditions on the diffusive fluxes; specifically, enforce
  # z boundary conditions on tau02, tau12 (the z-going flux of x, y momentum).
  # Bottom boundary.
  if z_bcs.bottom.bc_type == 'no_flux':  # Free-slip wall BCs.
    # Do nothing because we have already enforced s02, s12 = 0 at the bottom
    # when computing the strain rate tensor.
    pass
  elif z_bcs.bottom.bc_type == 'monin_obukhov':
    # Compute the fluxes arising from Monin-Obukhov similarity theory.
    assert z_bcs.bottom.mop is not None
    tau02_fcf_surface, tau12_cff_surface = monin_obukhov.surface_momentum_flux(
        u_fcc, v_cfc, rho_xxc, z_bcs.bottom.mop
    )
    tau02_fcf = apply_bcs.enforce_flux_at_z_bottom_bdy(
        tau02_fcf, tau02_fcf_surface
    )
    tau12_cff = apply_bcs.enforce_flux_at_z_bottom_bdy(
        tau12_cff, tau12_cff_surface
    )
  else:
    raise ValueError(f'Bad z_bcs.bottom.bc_type: {z_bcs.bottom.bc_type}')

  # Top boundary.
  if z_bcs.top.bc_type == 'no_flux':  # Free-slip wall BCs.
    # Do nothing because we have already enforced s02, s12 = 0 at the top when
    # computing the strain rate tensor.
    pass
  else:
    raise ValueError(
        f'Unsupported z boundary condition type: {z_bcs.top.bc_type}'
    )

  return tau00_ccc, tau01_ffc, tau02_fcf, tau11_ccc, tau12_cff, tau22_ccc
