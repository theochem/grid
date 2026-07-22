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
Robust Poisson Solver --- Double-Split Architecture.

Split 1 only (default)
-----------------------
::

    Total Density rho(r)
          |
    [SPLIT 1]  Analytical core subtraction using pre-fitted Gaussian parameters
          |     (load_atomic_gaussian_params -> coulomb_potential)
          |
    residual_1(r) = rho - rho_core   <- smooth, nuclear-cusp-free
          |
    [SOLVE]  solve_poisson_bvp on residual_1  --> phi_numerical
          |
    Total Potential = phi_core (analytical) + phi_numerical

Double-Split (split2=True)
---------------------------
::

    Total Density rho(r)
          |
    [SPLIT 1]  Analytical core subtraction  --> residual_1
          |
    [SPLIT 2]  Per-atom NNLS Gaussian fit of residual_1  --> residual_2
          |     (scipy.optimize.nnls, exponents from _DEFAULT_ALPHAS_BASIS)
          |
    [SOLVE]  solve_poisson_bvp on residual_2  --> phi_numerical
          |
    Total Potential = phi_core + phi_bonding (analytical) + phi_numerical
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from grid.atomgrid import AtomGrid
from grid.coulomb import coulomb_potential, load_atomic_gaussian_params
from grid.molgrid import MolGrid
from grid.poisson import solve_poisson_bvp
from grid.rtransform import BaseTransform

__all__ = ["solve_poisson_robust"]

_DEFAULT_ALPHAS_BASIS = np.geomspace(0.05, 5000.0, 20)


def _build_core_density(
    points: np.ndarray, center: np.ndarray, coeffs_s: np.ndarray, alphas_s: np.ndarray
) -> np.ndarray:
    """Return ndarray(N,) core density: sum_k c_k*(alpha_k/pi)^1.5*exp(-alpha_k*|r-center|^2)."""
    r_sq = np.sum((points - center) ** 2, axis=1)
    rho = np.zeros(len(points))
    for c, alpha in zip(coeffs_s, alphas_s, strict=True):
        prefactor = c * (alpha / np.pi) ** 1.5
        rho += prefactor * np.exp(-alpha * r_sq)
    return rho


def _fit_residual_gaussians(
    grid_pts: np.ndarray, residual: np.ndarray, atcoords: np.ndarray, alphas_basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the residual density with s-type Gaussians at each center using NNLS.

    Sequential greedy strategy: each atom's NNLS fit is solved against the
    current residual and immediately subtracted in-place, so later atoms see
    a partially reduced residual (order-dependent for multi-atom systems).

    Returns (coeffs, alphas, centers) for all retained non-zero Gaussians.
    """
    all_coeffs = []
    all_alphas = []
    all_centers = []

    for center in atcoords:
        r_sq = np.sum((grid_pts - center) ** 2, axis=1)
        A = np.zeros((len(grid_pts), len(alphas_basis)))
        for i, alpha in enumerate(alphas_basis):
            prefactor = (alpha / np.pi) ** 1.5
            A[:, i] = prefactor * np.exp(-alpha * r_sq)

        coeffs, _ = nnls(A, residual)

        mask = coeffs > 0
        if np.any(mask):
            c_pos = coeffs[mask]
            a_pos = alphas_basis[mask]
            all_coeffs.extend(c_pos)
            all_alphas.extend(a_pos)
            all_centers.extend([center] * len(c_pos))

            residual -= A[:, mask] @ c_pos

    if not all_coeffs:
        return np.array([]), np.array([]), np.empty((0, 3))
    return np.array(all_coeffs), np.array(all_alphas), np.array(all_centers)


def solve_poisson_robust(
    molgrid: MolGrid | AtomGrid,
    density_vals: np.ndarray,
    transform: BaseTransform,
    atnums: np.ndarray,
    atcoords: np.ndarray,
    split2: bool = False,
    alphas_basis: np.ndarray | None = None,
    **bvp_kwargs,
) -> callable:
    r"""Solve the Poisson equation robustly using analytical core subtraction (Split 1).

    If ``split2=True``, an additional Non-Negative Least Squares (NNLS) fitting
    step is performed on the residual (Split 2) to analytically subtract bonding
    and polarization density features before the BVP solve.

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
    split2 : bool, default=False
        If True, perform a second split (NNLS fitting of the residual) before solving.
    alphas_basis : ndarray, optional
        Array of s-type Gaussian exponents to use for the Split 2 fit.
        If None, a default geometric sequence of 20 exponents is used.
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
    ValueError
        If ``density_vals`` is not a 1-D array of length N matching the number
        of grid points in ``molgrid``.

    Notes
    -----
    Pre-fitted parameters are currently available for H (1), C (6), N (7),
    O (8), and Cl (17).
    """
    residual = np.array(density_vals, dtype=float)
    if residual.ndim != 1 or residual.shape[0] != molgrid.points.shape[0]:
        raise ValueError(
            f"density_vals must be 1-D with length matching molgrid.points "
            f"({molgrid.points.shape[0]}); got shape {residual.shape}"
        )

    # Cache params once; the closure may be called many times.
    atom_params = [
        (load_atomic_gaussian_params(int(atnum)), center)
        for atnum, center in zip(atnums, atcoords, strict=True)
    ]

    # SPLIT 1: Analytical Core Subtraction
    for (coeffs_s, alphas_s), center in atom_params:
        residual -= _build_core_density(molgrid.points, center, coeffs_s, alphas_s)

    # SPLIT 2: Bonding/Residual Fitting (Optional)
    # Always initialized so the closure safely references all three variables.
    fit_coeffs, fit_alphas, fit_centers = np.array([]), np.array([]), np.empty((0, 3))
    if split2:
        if alphas_basis is None:
            alphas_basis = _DEFAULT_ALPHAS_BASIS
        fit_coeffs, fit_alphas, fit_centers = _fit_residual_gaussians(
            molgrid.points, residual, atcoords, alphas_basis
        )

    # SOLVE: BVP solver on the smooth, nuclear-cusp-free residual
    phi_residual_interp = solve_poisson_bvp(molgrid, residual, transform, **bvp_kwargs)

    # SUM: closure returns v_core + v_bonding + v_residual at evaluation time
    def total_potential(points: np.ndarray) -> np.ndarray:
        """Evaluate total electrostatic potential at Cartesian points."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3); got {points.shape}")

        # Analytical core contribution — uses pre-cached params
        v_core = np.zeros(points.shape[0])
        for (coeffs_s, alphas_s), center in atom_params:
            centers_rep = np.tile(center, (len(coeffs_s), 1))
            v_core += coulomb_potential(
                points,
                centers_s=centers_rep,
                coeffs_s=coeffs_s,
                alphas_s=alphas_s,
                normalized=True,
            )

        # Analytical bonding fit contribution (Split 2)
        v_bonding = np.zeros(points.shape[0])
        if len(fit_coeffs) > 0:
            v_bonding = coulomb_potential(
                points,
                centers_s=fit_centers,
                coeffs_s=fit_coeffs,
                alphas_s=fit_alphas,
                normalized=True,
            )

        # Numerical residual contribution
        v_residual = phi_residual_interp(points)

        return v_core + v_bonding + v_residual

    return total_potential
