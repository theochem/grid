# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2026 The GRID Development Team
#
# This file is part of GRID.
#
# GRID is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GRID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
r"""
Coulomb potential module for Gaussian charge densities.

Provides exact analytical formulas for evaluating the electrostatic potential
of s-type and p-type Gaussian functions.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

__all__ = ["coulomb_gaussian_p", "coulomb_gaussian_s", "coulomb_potential"]


def coulomb_gaussian_s(r: np.ndarray, alpha: float, normalized: bool = True) -> np.ndarray:
    r"""Compute the exact Coulomb potential of an s-type Gaussian charge density.

    If ``normalized`` is True, the charge density is:
    .. math::
        \rho(r) = \left(\frac{\alpha}{\pi}\right)^{3/2} e^{-\alpha r^2}

    and the potential is:
    .. math::
        V(r) = \frac{\text{erf}(\sqrt{\alpha} r)}{r}

    If ``normalized`` is False, the charge density is:
    .. math::
        \rho(r) = e^{-\alpha r^2}

    and the potential is:
    .. math::
        V(r) = \left(\frac{\pi}{\alpha}\right)^{3/2} \frac{\text{erf}(\sqrt{\alpha} r)}{r}

    Parameters
    ----------
    r : np.ndarray
        Radial distances from the center of the Gaussian.
    alpha : float
        Gaussian exponent.
    normalized : bool, default=True
        Whether to compute the potential of a normalized s-type Gaussian.

    Returns
    -------
    np.ndarray
        Coulomb potential evaluated at the radial distances.
    """
    r = np.asarray(r, dtype=float)
    out = np.empty_like(r)
    mask_zero = r < 1e-12
    mask_nonzero = ~mask_zero

    r_nonzero = r[mask_nonzero]
    sqrt_a = np.sqrt(alpha)

    if normalized:
        out[mask_nonzero] = erf(sqrt_a * r_nonzero) / r_nonzero
        out[mask_zero] = 2.0 * sqrt_a / np.sqrt(np.pi)
    else:
        prefactor = (np.pi / alpha) ** 1.5
        out[mask_nonzero] = prefactor * erf(sqrt_a * r_nonzero) / r_nonzero
        out[mask_zero] = prefactor * 2.0 * sqrt_a / np.sqrt(np.pi)

    return out


def coulomb_gaussian_p(r: np.ndarray, beta: float, normalized: bool = True) -> np.ndarray:
    r"""Compute the exact Coulomb potential of a p-type radial Gaussian charge density.

    If ``normalized`` is True, the charge density is:
    .. math::
        \rho(r) = \frac{2}{3} \frac{\beta^{5/2}}{\pi^{3/2}} r^2 e^{-\beta r^2}

    and the potential is:
    .. math::
        V(r) = \frac{\text{erf}(\sqrt{\beta} r)}{r}
        + \frac{4}{3} \sqrt{\frac{\beta}{\pi}} e^{-\beta r^2}

    If ``normalized`` is False, the charge density is:
    .. math::
        \rho(r) = r^2 e^{-\beta r^2}

    and the potential is:
    .. math::
        V(r) = \frac{3 \pi^{1.5}}{2 \beta^{2.5}} \frac{\text{erf}(\sqrt{\beta} r)}{r}
        + \frac{2\pi}{\beta^2} e^{-\beta r^2}

    Parameters
    ----------
    r : np.ndarray
        Radial distances from the center of the Gaussian.
    beta : float
        Gaussian exponent.
    normalized : bool, default=True
        Whether to compute the potential of a normalized p-type Gaussian.

    Returns
    -------
    np.ndarray
        Coulomb potential evaluated at the radial distances.
    """
    r = np.asarray(r, dtype=float)
    out = np.empty_like(r)
    mask_zero = r < 1e-12
    mask_nonzero = ~mask_zero

    r_nonzero = r[mask_nonzero]
    sqrt_b = np.sqrt(beta)

    if normalized:
        out[mask_nonzero] = erf(sqrt_b * r_nonzero) / r_nonzero + (4.0 / 3.0) * (
            sqrt_b / np.sqrt(np.pi)
        ) * np.exp(-beta * r_nonzero**2)
        out[mask_zero] = (10.0 / 3.0) * (sqrt_b / np.sqrt(np.pi))
    else:
        term1_pref = 1.5 * (np.pi**1.5) / (beta**2.5)
        term2_pref = 2.0 * np.pi / (beta**2)
        out[mask_nonzero] = term1_pref * erf(sqrt_b * r_nonzero) / r_nonzero + term2_pref * np.exp(
            -beta * r_nonzero**2
        )
        out[mask_zero] = 5.0 * np.pi / (beta**2)

    return out


def coulomb_potential(
    points: np.ndarray,
    centers_s: np.ndarray,
    coeffs_s: np.ndarray,
    expons_s: np.ndarray,
    centers_p: np.ndarray | None = None,
    coeffs_p: np.ndarray | None = None,
    expons_p: np.ndarray | None = None,
    normalized: bool = True,
) -> np.ndarray:
    """Compute the total Coulomb potential at evaluation points from a set of Gaussians.

    Parameters
    ----------
    points : np.ndarray
        Evaluation points, shape (N, 3).
    centers_s : np.ndarray
        Centers of the s-type Gaussians, shape (Ks, 3).
    coeffs_s : np.ndarray
        Coefficients of the s-type Gaussians, shape (Ks,).
    expons_s : np.ndarray
        Exponents of the s-type Gaussians, shape (Ks,).
    centers_p : np.ndarray, optional
        Centers of the p-type Gaussians, shape (Kp, 3).
    coeffs_p : np.ndarray, optional
        Coefficients of the p-type Gaussians, shape (Kp,).
    expons_p : np.ndarray, optional
        Exponents of the p-type Gaussians, shape (Kp,).
    normalized : bool, default=True
        Whether the coefficients correspond to normalized Gaussians.

    Returns
    -------
    np.ndarray
        The computed electrostatic potential, shape (N,).
    """
    points = np.asarray(points, dtype=float)
    coeffs_s = np.asarray(coeffs_s, dtype=float)
    expons_s = np.asarray(expons_s, dtype=float)
    centers_s = np.asarray(centers_s, dtype=float)

    V = np.zeros(len(points))

    # Accumulate s-type potential
    for c, alpha, center in zip(coeffs_s, expons_s, centers_s):
        r = np.linalg.norm(points - center, axis=-1)
        V += c * coulomb_gaussian_s(r, alpha, normalized=normalized)

    # Accumulate p-type potential if present
    if coeffs_p is not None and expons_p is not None and centers_p is not None:
        coeffs_p = np.asarray(coeffs_p, dtype=float)
        expons_p = np.asarray(expons_p, dtype=float)
        centers_p = np.asarray(centers_p, dtype=float)

        for c, beta, center in zip(coeffs_p, expons_p, centers_p):
            r = np.linalg.norm(points - center, axis=-1)
            V += c * coulomb_gaussian_p(r, beta, normalized=normalized)

    return V
