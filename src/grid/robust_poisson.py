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
"""
Robust Poisson Solver --- Split-1-Plus-Solve Architecture.

Architecture
------------
::

    Total Density rho(r)
          |
    [SPLIT 1]  Analytical core subtraction using pre-fitted Gaussian parameters
          |     (load_atomic_gaussian_params -> coulomb_potential)
          |
    residual(r) = rho - rho_core   <-- smooth, nuclear-cusp-free
          |
    [SOLVE]  solve_poisson_bvp on residual  --> phi_numerical
          |
    Total Potential = phi_core (analytical) + phi_numerical

Notes
-----
Split 2 (bonding/polarization residual fitting) and the full Double-Split
pipeline will be added in a subsequent PR.
"""

from __future__ import annotations

import numpy as np

from grid.atomgrid import AtomGrid
from grid.coulomb import coulomb_potential, load_atomic_gaussian_params
from grid.molgrid import MolGrid
from grid.poisson import solve_poisson_bvp
from grid.rtransform import BaseTransform

__all__ = ["solve_poisson_robust"]


def _build_core_density(
    points: np.ndarray, center: np.ndarray, coeffs_s: np.ndarray, alphas_s: np.ndarray
) -> np.ndarray:
    """Evaluate a sum of normalized s-type Gaussians at the given points.

    Returns ndarray(N,) of the core density:
    ``sum_k c_k * (alpha_k/pi)^1.5 * exp(-alpha_k * |r - center|^2)``.
    """
    r_sq = np.sum((points - center) ** 2, axis=1)
    rho = np.zeros(len(points))
    for c, alpha in zip(coeffs_s, alphas_s, strict=True):
        prefactor = c * (alpha / np.pi) ** 1.5
        rho += prefactor * np.exp(-alpha * r_sq)
    return rho


def solve_poisson_robust(
    molgrid: MolGrid | AtomGrid,
    density_vals: np.ndarray,
    transform: BaseTransform,
    atnums: np.ndarray,
    atcoords: np.ndarray,
    **bvp_kwargs,
) -> callable:
    r"""Solve the Poisson equation robustly using analytical core subtraction (Split 1).

    For each atomic center, pre-fitted Gaussian parameters are loaded via
    :func:`~grid.coulomb.load_atomic_gaussian_params` and the exact analytical
    core potential is computed with :func:`~grid.coulomb.coulomb_potential`.
    The corresponding core density is subtracted from the input to form a smooth
    residual, which is then solved numerically with
    :func:`~grid.poisson.solve_poisson_bvp`.

    Parameters
    ----------
    molgrid : MolGrid or AtomGrid
        Molecular or atomic grid used for integration and ODE solving.
    density_vals : ndarray(N,)
        Total electron density evaluated at all grid points.
    transform : BaseTransform
        Radial coordinate transform passed to ``solve_poisson_bvp``.
    atnums : ndarray(M,) of int
        Atomic numbers for each of the M atomic centers.
    atcoords : ndarray(M, 3)
        Cartesian coordinates of the M atomic centers in atomic units (Bohr).
    **bvp_kwargs
        Additional keyword arguments forwarded to :func:`~grid.poisson.solve_poisson_bvp`.

    Returns
    -------
    callable
        Function ``V(points)`` returning the total electrostatic potential at
        an array of Cartesian evaluation points, shape ``(N, 3) -> (N,)``.

    Raises
    ------
    ValueError
        If pre-fitted Gaussian parameters are not available for a given atomic number.

    Notes
    -----
    Pre-fitted parameters are currently available for H (1), C (6), N (7),
    O (8), and Cl (17).
    """
    residual = density_vals.copy()

    # Pre-cache Gaussian parameters for all atoms once (avoids repeated JSON lookups
    # inside the returned closure, which may be called many times).
    atom_params = [
        (load_atomic_gaussian_params(int(atnum)), center)
        for atnum, center in zip(atnums, atcoords, strict=True)
    ]

    # SPLIT 1: Analytical Core Subtraction
    for (coeffs_s, alphas_s), center in atom_params:
        residual -= _build_core_density(molgrid.points, center, coeffs_s, alphas_s)

    # SOLVE: BVP solver on the smooth, nuclear-cusp-free residual
    phi_residual_interp = solve_poisson_bvp(molgrid, residual, transform, **bvp_kwargs)

    # SUM: Total potential = analytical core + numerical residual
    def total_potential(points: np.ndarray) -> np.ndarray:
        """Evaluate total electrostatic potential at Cartesian points."""
        points = np.asarray(points, dtype=float)

        # Analytical core contribution — uses pre-cached params
        v_core = np.zeros(len(points))
        for (coeffs_s, alphas_s), center in atom_params:
            centers_rep = np.tile(center, (len(coeffs_s), 1))
            v_core += coulomb_potential(
                points,
                centers_s=centers_rep,
                coeffs_s=coeffs_s,
                alphas_s=alphas_s,
                normalized=True,
            )

        # Numerical residual contribution
        v_residual = phi_residual_interp(points)

        return v_core + v_residual

    return total_potential
