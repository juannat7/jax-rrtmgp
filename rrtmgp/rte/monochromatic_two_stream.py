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

"""The radiative transfer equation solver.

Common symbols:
ssa: single-scattering albedo;
tau: optical depth;
g: asymmetry factor;
sw: shortwave;
lw: longwave;
gamma: exchange rate coefficient in the radiative transfer equation;
zenith: the zenith angle of collimated solar radiation.
"""

import math
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp import kernel_ops
from rrtmgp.rte import rte_utils

Array: TypeAlias = jax.Array
StatesMap: TypeAlias = dict[str, Array]

# Secant of the longwave diffusivity angle per Fu et al. (1997).
_LW_DIFFUSIVE_FACTOR = 1.66
_EPSILON = 1e-6
# Minimum longwave optical depth required for nonzero source.
_MIN_TAU_FOR_LW_SRC = 1e-4
# Minimum value of the k parameter used in the transmittance.
_K_MIN = 1e-2


def _shift_up(f: Array) -> Array:
  """output_i = f_{i-1}."""
  return kernel_ops.shift_from_minus(f, 2)


def _shift_down(f: Array) -> Array:
  """output_i = f_{i+1}."""
  return kernel_ops.shift_from_plus(f, 2)


def lw_combine_sources(planck_srcs: StatesMap) -> StatesMap:
  """Combine the longwave source functions at each cell face.

  RRTMGP provides two source functions at each cell interface using the
  spectral mapping of each adjacent layer. These source functions are combined
  here via a geometric mean, and the result can be used for two-stream
  calculations.

  Args:
    planck_srcs: A dictionary containing the longwave Planck sources at the cell
      faces [ccf]. The `planck_src_top` 3D variable contains the Planck source
      at the top cell face derived from the cell center's spectral mapping while
      the `planck_src_bottom` 3D variable contains the Planck source at the
      bottom cell face.

  Returns:
    A map of 3D variables for the combined Planck sources at the top face and
    the bottom cell face, respectively with the same keys as the ones in the
    input `planck_srcs`.
  """
  planck_src_top = planck_srcs['planck_src_top']
  planck_src_bottom = planck_srcs['planck_src_bottom']
  combined_src_top = jnp.sqrt(planck_src_top * _shift_down(planck_src_bottom))
  combined_src_bottom = _shift_up(combined_src_top)
  return {
      'planck_src_top': combined_src_top,
      'planck_src_bottom': combined_src_bottom,
  }


def _k_fn(gamma1: Array, gamma2: Array) -> Array:
  """Compute the k parameter used in the transmittance."""
  k = jnp.sqrt(jnp.maximum((gamma1 + gamma2) * (gamma1 - gamma2), _EPSILON))
  return jnp.maximum(k, _K_MIN)


def _rt_denominator_diffuse(gamma1: Array, gamma2: Array, tau: Array) -> Array:
  """Shared denominator of the diffuse reflectance and transmittance."""
  # As in the original RRTMGP Fortran code, this expression has been
  # refactored to avoid rounding errors when k, gamma1 are of very different
  # magnitudes.
  k = _k_fn(gamma1, gamma2)
  return k * (1 + jnp.exp(-2.0 * tau * k)) + gamma1 * (
      1 - jnp.exp(-2.0 * tau * k)
  )


def _rt_denominator_direct(
    gamma1: Array, gamma2: Array, tau: Array, ssa: Array, zenith: float
) -> Array:
  """Shared denominator of direct reflectance and transmittance functions."""
  k = _k_fn(gamma1, gamma2)
  denom = _rt_denominator_diffuse(gamma1, gamma2, tau)
  k_mu_squared = (k * jnp.cos(zenith)) ** 2

  # Equation 14, multiplying top and bottom by exp(-k*tau) and rearranging to
  # avoid division by 0.
  return jnp.where(
      jnp.abs(1.0 - k_mu_squared) >= _EPSILON,
      denom * (1.0 - k_mu_squared) / ssa,
      denom * _EPSILON / ssa,
  )


def _direct_reflectance(
    gamma1: Array,
    gamma2: Array,
    gamma3: Array,
    alpha2: Array,
    tau: Array,
    ssa: Array,
    zenith: float,
) -> Array:
  """Direct solar radiation reflectance (equation 14 of Meador and Weaver)."""
  k = _k_fn(gamma1, gamma2)
  denom = _rt_denominator_direct(gamma1, gamma2, tau, ssa, zenith)
  k_mu = k * math.cos(zenith)

  # Transmittance of direct, unscattered beam.
  t0 = jnp.exp(-tau / math.cos(zenith))

  # Equation 14 of Meador and Weaver (1980), multiplying top and bottom by
  # exp(-k*tau) and rearranging to avoid division by 0.
  exp_minusktau = jnp.exp(-k * tau)
  exp_minus2ktau = jnp.exp(-2.0 * k * tau)
  return (
      (1.0 - k_mu) * (alpha2 + k * gamma3)
      - (1.0 + k_mu) * (alpha2 - k * gamma3) * exp_minus2ktau
      - 2.0 * (k * gamma3 - alpha2 * k_mu) * exp_minusktau * t0
  ) / denom


def _direct_transmittance(
    gamma1: Array,
    gamma2: Array,
    gamma4: Array,
    alpha1: Array,
    tau: Array,
    ssa: Array,
    zenith: float,
) -> Array:
  """Direct solar radiation transmittance (equation 15 of Meador and Weaver)."""
  k = _k_fn(gamma1, gamma2)
  denom = _rt_denominator_direct(gamma1, gamma2, tau, ssa, zenith)
  k_mu = k * math.cos(zenith)
  k_y4 = k * gamma4

  # Transmittance of direct, unscattered beam.
  t0 = jnp.exp(-tau / math.cos(zenith))

  exp_minusktau = jnp.exp(-k * tau)
  exp_minus2ktau = jnp.exp(-2 * k * tau)

  # Equation 15 (Meador and Weaver (1980)), refactored for numerical stability
  # by 1) multiplying top and bottom by exp(-k*tau), 2) multiplying through by
  # exp(-tau/mu0) to prefer underflow to overflow, and 3) omitting direct
  # transmittance.
  out = (
      -(
          (1.0 + k_mu) * (alpha1 + k_y4) * t0
          - (1.0 - k_mu) * (alpha1 - k_y4) * exp_minus2ktau * t0
          - 2.0 * (k_y4 + alpha1 * k_mu) * exp_minusktau
      )
      / denom
  )
  return out


def _diffuse_reflectance(gamma1: Array, gamma2: Array, tau: Array) -> Array:
  """The diffuse reflectance (equation 25 of Meador and Weaver (1980))."""
  k = _k_fn(gamma1, gamma2)
  denom = _rt_denominator_diffuse(gamma1, gamma2, tau)
  return gamma2 * (1.0 - jnp.exp(-2.0 * tau * k)) / denom


def _diffuse_transmittance(gamma1: Array, gamma2: Array, tau: Array) -> Array:
  """The diffuse transmittance (equation 26 of Meador and Weaver (1980))."""
  k = _k_fn(gamma1, gamma2)
  denom = _rt_denominator_diffuse(gamma1, gamma2, tau)
  return 2.0 * k * jnp.exp(-tau * k) / denom


def lw_cell_source_and_properties(
    optical_depth: Array,
    ssa: Array,
    level_src_bottom: Array,
    level_src_top: Array,
    asymmetry_factor: Array,
) -> StatesMap:
  """Compute the longwave two-stream reflectance, transmittance, and sources.

  The upwelling and downwelling Planck functions and the optical properties
  (transmission and reflectance) are calculated at the cell centers. Equations
  are developed in Meador and Weaver (1980) and Toon et al. (1989).

  Args:
    optical_depth: The pointwise optical depth.
    ssa: The pointwise single-scattering albedo.
    level_src_bottom: The Planck source at the bottom cell face [W / m^2 / sr].
    level_src_top: The Planck source at the top cell face [W / m^2 / sr].
    asymmetry_factor: The pointwise asymmetry factor.

  Returns:
    A dictionary containing the following items:
      'r_diff': A 3D variable containing the pointwise reflectance.
      't_diff': A 3D variable containing the pointwise transmittance.
      'src_up': A 3D variable containing the pointwise upwelling Planck source.
      'src_down': A 3D variable with the pointwise downwelling Planck source.
  """
  # The coefficient of the parallel irradiance in the 2-stream RTE.
  gamma1 = _LW_DIFFUSIVE_FACTOR * (1 - 0.5 * ssa * (1 + asymmetry_factor))
  # The coefficient of the antiparallel irradiance in the 2-stream RTE.
  gamma2 = _LW_DIFFUSIVE_FACTOR * 0.5 * ssa * (1 - asymmetry_factor)

  r_diff = _diffuse_reflectance(gamma1, gamma2, optical_depth)
  t_diff = _diffuse_transmittance(gamma1, gamma2, optical_depth)

  # From Toon et al. (JGR 1989) Eqs 26-27, first-order coefficient of the
  # Taylor series expansion of the Planck function in terms of the optical
  # depth.
  b_1 = (level_src_bottom - level_src_top) / (optical_depth * (gamma1 + gamma2))

  # Compute longwave source function for upward and downward emission at cell
  # interfaces using linear-in-tau assumption.
  c_up_top = level_src_top + b_1
  c_up_bottom = level_src_bottom + b_1
  c_down_top = level_src_top - b_1
  c_down_bottom = level_src_bottom - b_1

  def cell_center_src_fn(
      downstream_out: Array,
      downstream_in: Array,
      upstream_in: Array,
      refl: Array,
      tran: Array,
      tau: Array,
  ) -> Array:
    """Compute the flux at the cell center consistent with face fluxes.

    The cell center source is the residual that remains when one subtracts
    from the downstream outward flux two contributions:
    1. the upstream inward flux that is transmitted through the cell and
    2. the downstream inward flux that is reflected off the cell.

    Args:
      downstream_out: Downstream outward flux.
      downstream_in: Downstream inward flux.
      upstream_in: Upstream inward flux.
      refl: The grid cell reflectance.
      tran: The grid cell transmittance.
      tau: The grid cell optical depth.

    Returns:
      The directional radiative source at the cell center consistent with the
      given face sources [W / m^2].
    """
    src = math.pi * (downstream_out - refl * downstream_in - tran * upstream_in)
    # Filter out sources where the optical depth is too small.
    return jnp.where(tau > _MIN_TAU_FOR_LW_SRC, src, 0.0)

  src_up = cell_center_src_fn(
      c_up_top, c_down_top, c_up_bottom, r_diff, t_diff, optical_depth
  )
  src_down = cell_center_src_fn(
      c_down_bottom, c_up_bottom, c_down_top, r_diff, t_diff, optical_depth
  )
  return {
      't_diff': t_diff,
      'r_diff': r_diff,
      'src_up': src_up,
      'src_down': src_down,
  }


def sw_cell_properties(
    zenith: float, optical_depth: Array, ssa: Array, asymmetry_factor: Array
) -> StatesMap:
  """Compute shortwave reflectance and transmittance.

  Two-stream solutions to direct and diffuse reflectance and transmittance as
  a function of optical depth, single-scattering albedo, and asymmetry factor.
  Equations are developed in Meador and Weaver (1980).

  Args:
    zenith: The zenith angle of the shortwave collimated radiation.
    optical_depth: A 3D variable containing the pointwise optical depth.
    ssa: A 3D variable containing the pointwise single-scattering albedo.
    asymmetry_factor: A 3D variable containing the pointwise asymmetry factor.

  Returns:
    A dictionary containing the following items:
    't_diff': A 3D variable containing the diffuse transmittance.
    'r_diff': A 3D variable containing the diffuse reflectance.
    't_dir': A 3D variable containing the direct transmittance.
    'r_dir': A 3D variable containing the direct reflectance.
  """
  # Exchange rate coefficients from Zdunkowski et al. (1980).
  g = asymmetry_factor
  gamma1 = 0.25 * (8 - ssa * (5 + 3 * g))
  gamma2 = 0.25 * 3 * ssa * (1 - g)
  gamma3 = 0.25 * (2 - 3 * math.cos(zenith) * g)
  gamma4 = 1 - gamma3
  alpha1 = gamma1 * gamma4 + gamma2 * gamma3
  alpha2 = gamma1 * gamma3 + gamma2 * gamma4

  # Diffuse reflectance and transmittance.
  r_diff = _diffuse_reflectance(gamma1, gamma2, optical_depth)
  t_diff = _diffuse_transmittance(gamma1, gamma2, optical_depth)

  # Direct reflectance and transmittance.
  r_dir_unconstrained = _direct_reflectance(
      gamma1, gamma2, gamma3, alpha2, optical_depth, ssa, zenith
  )
  t_dir_unconstrained = _direct_transmittance(
      gamma1, gamma2, gamma4, alpha1, optical_depth, ssa, zenith
  )

  # Constrain reflectance and transmittance to be positive and to not go above
  # physical limits by enforcing the constraint that the direct beam can
  # either be reflected, penetrate unscattetered to the bottom of the grid
  # cell, or penetrate through but be scattered on the way.

  # Direct transmittance.
  t0 = jnp.exp(-optical_depth / math.cos(zenith))
  r_dir = jnp.clip(r_dir_unconstrained, 0, 1 - t0)
  t_dir = jnp.clip(t_dir_unconstrained, 0, 1 - t0 - r_dir)

  return {
      't_diff': t_diff,
      'r_diff': r_diff,
      't_dir': t_dir,
      'r_dir': r_dir,
  }


def sw_cell_source(
    t_dir: Array,
    r_dir: Array,
    optical_depth: Array,
    toa_flux: Array,
    sfc_albedo_direct: Array,
    zenith: float,
    use_scan: bool = False,
) -> StatesMap:
  """Compute the monochromatic shortwave direct-beam flux and diffuse source.

  Args:
    t_dir: Direct-beam transmittance, a 3D field.
    r_dir: Direct-beam reflectance, a 3D field.
    optical_depth: Optical depth, a 3D field.
    toa_flux: The top-of-atmosphere incoming flux, a 2D field.
    sfc_albedo_direct: The surface albedo with respect to direct radiation, a 2D
      field.
    zenith: The solar zenith angle.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing the following items:
      'src_up': A 3D field for the cell center upward source.
      'src_down': A 3D field for the cell center downward source.
      'flux_down_dir': A 3D field for the solved downwelling direct-beam
        radiative flux at the bottom cell face.
      'sfc_src': A 2D field for the shortwave source emanating from the surface.
  """
  # Transmittance of direct, unscattered beam.
  t_noscat = jnp.exp(-optical_depth / math.cos(zenith))
  mu = math.cos(zenith)

  # The vertical component of incident flux at the top boundary.
  flux_down_direct_bc = toa_flux * mu

  # Global recurrent accumulation for the direct-beam downward flux at the
  # bottom cell face unraveling from the top of the atmosphere down to the
  # surface. The recurrence follows the simple relation:
  # flux_down_direct[i] = T_no_scatter[i] * flux_down_direct[i + 1]
  def op(carry, w):
    return w * carry, w * carry
  init = flux_down_direct_bc
  inputs = {'w': t_noscat}
  _, flux_down_direct = rte_utils.recurrent_op_with_halos(
      op, init, inputs, forward=False, use_scan=use_scan
  )

  # Upward source from direct-beam reflection at the cell center.
  src_up = r_dir * _shift_down(flux_down_direct)

  # Downward source from direct-beam transmittance at the cell center.
  src_down = t_dir * _shift_down(flux_down_direct)

  # Direct-beam flux incident on the surface.
  halo_width = 1
  flux_down_sfc = flux_down_direct[:, :, halo_width]

  # The surface source is the direct-beam downard flux that is reflected from
  # the surface.
  sfc_src = sfc_albedo_direct * flux_down_sfc

  srcs_primary = {
      'src_up': src_up,
      'src_down': src_down,
      'flux_down_dir': flux_down_direct,
      'sfc_src': sfc_src,
  }
  return srcs_primary


def _solve_rte_2stream(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    toa_flux_down: Array,
    sfc_emission: Array,
    sfc_reflectance: Array,
    use_scan: bool = False,
) -> StatesMap:
  """Solves the monochromatic two-stream radiative transfer equation.

  Given boundary conditions for the downward flux at the top of the atmosphere
  (`toa_flux_down`) and the upward surface emission (`sfc_emission`), this
  computes the two-stream approximation of the upwelling and downwelling
  radiative fluxes at the cell faces based on the equations of Shonk and Hogan
  (2008).  All the computations here assume a single absorption interval (or 'g'
  interval in RRTM nomenclature).  This function needs to be applied to each 'g'
  interval separately.

  Args:
    t_diff: Cell center transmittance, 3D field
    r_diff: Cell center reflectance, 3D field.
    src_up: Cell center upward emission, 3D field.
    src_down: Cell center downward emission, 3D field.
    toa_flux_down: The downward component of the incoming flux at the top
      boundary of the atmosphere.  This corresponds to the downward flux at the
      wall of the domain (e.g., the face above the last interior node).  2D
      field.
    sfc_emission: The upward surface emission.  This corresponds to the wall of
      the domain (e.g., the face below the first interior node).  2D field.
    sfc_reflectance: The surface reflectance.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing fluxes at the bottom cell face:
      'flux_up': The upwelling radiative flux.
      'flux_down': The downwelling radiative flux.
  """
  # Global recurrent accumulation for the albedo of the atmosphere below a
  # certain level, computed from the surface to the top boundary.  The
  # recurrence relation for albedo is taken from Shonk and Hogan, Eq. 9.

  def albedo_op(
      albedo_below: Array, r_diff: Array, t_diff: Array
  ) -> tuple[Array, Array]:
    """Recurrent formula for albedo solution, starting from the surface."""
    # Geometric series solution accounting for infinite reflection events.
    beta = 1 / (1 - r_diff * albedo_below)
    out = r_diff + t_diff**2 * beta * albedo_below
    return out, out  # Carry and output are the same.

  init = sfc_reflectance
  albedo_inputs = {'r_diff': r_diff, 't_diff': t_diff}
  _, albedo = rte_utils.recurrent_op_with_halos(
      albedo_op, init, albedo_inputs, forward=True, use_scan=use_scan
  )

  # Global recurrent accumulation for the aggregate upwelling source emission
  # computed from the surface to the top of the atmosphere.  The coefficient and
  # bias terms of the recurrent relation for emission are taken from Shonk and
  # Hogan, Eq. 11.  The upward emission is a combination of 1) the upward source
  # from the grid cell center, 2) aggregate emission from the atmosphere below,
  # transmitted through the cell, and 3) the downward source from the grid cell
  # center that is reflected from the atmosphere below and transmitted up
  # through the layer.

  def upward_emission_op(
      emission_from_below: Array,
      src_up: Array,
      src_down: Array,
      t_diff: Array,
      r_diff: Array,
      albedo: Array,
  ) -> tuple[Array, Array]:
    """Recurrent formula for upward emission, starting from the surface."""
    # Geometric series solution accounting for infinite reflection events.
    beta = 1 / (1 - r_diff * albedo)
    out = src_up + t_diff * beta * (emission_from_below + src_down * albedo)
    return out, out  # Carry and output are the same.

  init = sfc_emission
  emission_inputs = {
      'src_up': src_up,
      'src_down': src_down,
      't_diff': t_diff,
      'r_diff': r_diff,
      'albedo': _shift_up(albedo),
  }
  _, emission_up = rte_utils.recurrent_op_with_halos(
      upward_emission_op, init, emission_inputs, forward=True, use_scan=use_scan
  )

  # Global recurrent accumulation for the downwelling radiative flux solution at
  # the bottom face, unravelling from the top of the atmosphere down to the
  # surface.  The coefficient and bias terms are taken from Shonk and Hogan,
  # Eq. 13.  The downward flux at the bottom face is a combination of 1) the
  # downward source emitted from the grid cell, 2) the downward flux from the
  # face above transmitted through the cell, and 3) the aggregate upward
  # emissions from the atmosphere below that are reflected from the cell.

  def flux_down_op(
      flux_down_from_above: Array,
      emiss_up: Array,
      src_down: Array,
      t_diff: Array,
      r_diff: Array,
      albedo: Array,
  ) -> tuple[Array, Array]:
    """Recurrent formula for downwelling flux initiating at top boundar."""
    # Geometric series solution accounting for infinite reflection events.
    beta = 1 / (1 - r_diff * albedo)
    out = (t_diff * flux_down_from_above + r_diff * emiss_up + src_down) * beta
    return out, out  # Carry and output are the same.

  init = toa_flux_down
  flux_down_inputs = {
      'emiss_up': _shift_up(emission_up),
      'src_down': src_down,
      't_diff': t_diff,
      'r_diff': r_diff,
      'albedo': _shift_up(albedo),
  }
  _, flux_down = rte_utils.recurrent_op_with_halos(
      flux_down_op, init, flux_down_inputs, forward=False, use_scan=use_scan
  )

  # The upwelling radiative flux at the bottom face can now be computed
  # directly from the cumulative upward emissions, the cumulative albedo of
  # the atmosphere below, and the downwelling radiative flux at the same face.
  flux_up = flux_down * _shift_up(albedo) + _shift_up(emission_up)
  fluxes = {
      'flux_up': flux_up,
      'flux_down': flux_down,
  }
  return fluxes


def lw_transport(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    toa_flux_down: Array,
    sfc_src: Array,
    sfc_emissivity: Array,
    use_scan: bool = False,
) -> StatesMap:
  """Compute the monochromatic longwave diffusive flux of the atmosphere.

  The upwelling and downwelling fluxes are computed from the equations of Shonk
  and Hogan.  The net flux is also computed at every face.  Note that the net
  flux is computed only at cell faces and does not correspond to the net flux
  into or out of the grid cell.  For the overall grid cell net flux, one must
  take the difference of the net fluxes of the upper and bottom faces.

  Args:
    t_diff: Cell center transmittance, 3D field
    r_diff: Cell center reflectance, 3D field.
    src_up: Cell center Planck upward emission, 3D field.
    src_down: Cell center Planck downward emission, 3D field.
    toa_flux_down: The downward flux at the top boundary of the atmosphere, 2D
      field.
    sfc_src: The surface Planck source, 2D field.
    sfc_emissivity: The surface emissivity, 2D field.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing fluxes at the cell faces [W/m^2]:
      'flux_up': The upwelling radiative flux.
      'flux_down': The downwelling radiative flux.
      'net_flux': The net radiative flux.
  """
  # The source of diffuse radiation is the surface emission.
  sfc_emission = np.pi * sfc_emissivity * sfc_src
  # The surface reflectance is just the complement of the surface emissivity.
  sfc_reflectance = 1 - sfc_emissivity
  fluxes = _solve_rte_2stream(
      t_diff,
      r_diff,
      src_up,
      src_down,
      toa_flux_down,
      sfc_emission,
      sfc_reflectance,
      use_scan,
  )
  fluxes['flux_net'] = fluxes['flux_up'] - fluxes['flux_down']
  return fluxes


def sw_transport(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    sfc_src: Array,
    sfc_albedo: Array,
    flux_down_dir: Array,
    use_scan: bool = False,
) -> StatesMap:
  """Compute the monochromatic shortwave fluxes in a layered atmosphere.

  The direct-beam downward flux `flux_down_dir` is added to the downwelling
  diffuse flux in the final solution.

  The upwelling and downwelling diffuse fluxes are computed from the equations
  of Shonk and Hogan.  The net flux is also computed at every face.  Note that
  the net flux is computed only at cell interfaces and does not correspond to
  the net flux into or out of the cell.  For the overall grid cell net flux, one
  must take the difference of the net fluxes of the upper and bottom faces.

  Args:
    t_diff: Cell center transmittance, 3D field
    r_diff: Cell center reflectance, 3D field.
    src_up: Cell center upward source, 3D field.
    src_down: Cell center downward source, 3D field.
    sfc_src: Direct-beam shortwave radiation reflected upward from the surface,
      2D field.
    sfc_albedo: The surface albedo, 2D field.
    flux_down_dir: The downwelling direct-beam radiative flux at the cell faces,
      3D field.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing fluxes at the z faces:
      'flux_up': The upwelling radiative flux.
      'flux_down': The downwelling radiative flux.
      'net_flux': The net radiative flux.
  """
  fluxes = _solve_rte_2stream(
      t_diff,
      r_diff,
      src_up,
      src_down,
      jnp.zeros_like(sfc_src),
      sfc_src,
      sfc_albedo,
      use_scan,
  )

  # Add the direct-beam contribution to the downwelling flux.
  fluxes['flux_down'] = fluxes['flux_down'] + flux_down_dir

  # The net flux computed at cell faces.
  fluxes['flux_net'] = fluxes['flux_up'] - fluxes['flux_down']
  return fluxes
