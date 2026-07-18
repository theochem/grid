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
"""Tests for the robust Poisson solver (Split 1 + BVP)."""

import numpy as np
import pytest
from scipy.special import erf

from grid.atomgrid import AtomGrid
from grid.onedgrid import GaussLegendre
from grid.robust_poisson import solve_poisson_robust
from grid.rtransform import BeckeRTransform, InverseRTransform


def _make_single_center_grid(n_radial=100, l_degree=29, center=None):
    """Return an AtomGrid and InverseRTransform at a given center."""
    center = np.zeros(3) if center is None else np.asarray(center)
    oned = GaussLegendre(n_radial)
    tf = BeckeRTransform(1e-5, R=1.5)
    radial = tf.transform_1d_grid(oned)
    atgrid = AtomGrid(radial, degrees=[l_degree], center=center)
    inv_tf = InverseRTransform(tf)
    return atgrid, inv_tf


def _gaussian_density(points, alpha, center=None):
    """Normalized s-type Gaussian density: (alpha/pi)^(3/2) * exp(-alpha*r^2)."""
    center = np.zeros(3) if center is None else np.asarray(center)
    r_sq = np.sum((points - center) ** 2, axis=1)
    return (alpha / np.pi) ** 1.5 * np.exp(-alpha * r_sq)


def _gaussian_potential(points, alpha, center=None):
    """Exact analytical potential of a normalized s-type Gaussian: erf(sqrt(alpha)*r)/r."""
    center = np.zeros(3) if center is None else np.asarray(center)
    r = np.linalg.norm(points - center, axis=1)
    V = np.zeros_like(r)
    mask = r >= 1e-12
    V[mask] = erf(np.sqrt(alpha) * r[mask]) / r[mask]
    V[~mask] = 2.0 * np.sqrt(alpha / np.pi)
    return V


def test_robust_poisson_moderate_gaussian():
    """solve_poisson_robust should match exact Gaussian potential for alpha=0.5."""
    alpha = 0.5
    atgrid, tf = _make_single_center_grid(n_radial=100)

    density = _gaussian_density(atgrid.points, alpha)
    V_exact = _gaussian_potential(atgrid.points, alpha)

    pot_func = solve_poisson_robust(
        atgrid,
        density,
        tf,
        atnums=np.array([1]),
        atcoords=np.zeros((1, 3)),
    )
    V_computed = pot_func(atgrid.points)

    # Relative L2 error should be well below 1%
    rel_error = np.sqrt(np.mean((V_computed - V_exact) ** 2)) / np.sqrt(np.mean(V_exact**2))
    assert rel_error < 0.01, f"Moderate Gaussian: relative L2 error = {rel_error:.4e}"


def test_robust_poisson_sharp_gaussian():
    """solve_poisson_robust should stay stable for a sharp cusp-like density (alpha=50)."""
    alpha = 50.0
    atgrid, tf = _make_single_center_grid(n_radial=150)

    density = _gaussian_density(atgrid.points, alpha)
    V_exact = _gaussian_potential(atgrid.points, alpha)

    pot_func = solve_poisson_robust(
        atgrid,
        density,
        tf,
        atnums=np.array([1]),
        atcoords=np.zeros((1, 3)),
    )
    V_computed = pot_func(atgrid.points)

    rel_error = np.sqrt(np.mean((V_computed - V_exact) ** 2)) / np.sqrt(np.mean(V_exact**2))
    # Tighter than the plain BVP would achieve without Split 1
    assert rel_error < 0.05, f"Sharp Gaussian: relative L2 error = {rel_error:.4e}"


def test_robust_poisson_returns_callable():
    """solve_poisson_robust must return a callable that accepts (M, 3) point arrays."""
    alpha = 1.0
    atgrid, tf = _make_single_center_grid(n_radial=80)
    density = _gaussian_density(atgrid.points, alpha)

    result = solve_poisson_robust(
        atgrid,
        density,
        tf,
        atnums=np.array([1]),
        atcoords=np.zeros((1, 3)),
    )

    assert callable(result)

    # Evaluate on a small set of random points
    rng = np.random.default_rng(42)
    test_pts = rng.uniform(-3, 3, size=(10, 3))
    vals = result(test_pts)
    assert vals.shape == (10,)
    assert np.all(np.isfinite(vals)), "Potential values must be finite at all evaluation points"


def test_robust_poisson_unsupported_element_raises():
    """solve_poisson_robust must raise ValueError for an element with no pre-fitted params."""
    alpha = 1.0
    atgrid, tf = _make_single_center_grid(n_radial=80)
    density = _gaussian_density(atgrid.points, alpha)

    with pytest.raises(ValueError):
        solve_poisson_robust(
            atgrid,
            density,
            tf,
            atnums=np.array([2]),
            atcoords=np.zeros((1, 3)),
        )


def test_robust_poisson_carbon():
    """solve_poisson_robust should run without error for a Carbon center (Z=6)."""
    alpha = 5.0
    center = np.zeros(3)
    atgrid, tf = _make_single_center_grid(n_radial=120, center=center)
    density = _gaussian_density(atgrid.points, alpha, center=center)

    pot_func = solve_poisson_robust(
        atgrid,
        density,
        tf,
        atnums=np.array([6]),
        atcoords=center.reshape(1, 3),
    )

    V = pot_func(atgrid.points)
    assert np.all(np.isfinite(V)), "Potential values must be finite for Carbon test"
    assert np.all(V >= -1e-10), (
        "Electrostatic potential of positive density should be non-negative "
        "(within numerical tolerance)"
    )


def test_robust_poisson_transition_density():
    """solve_poisson_robust must remain stable when density can be negative.

    A transition density (off-diagonal element of the density matrix) changes
    sign in space. Here we simulate it as a difference of two Gaussians.
    The Split 1 core subtraction should not break on negative residuals.
    """
    center = np.zeros(3)
    atgrid, tf = _make_single_center_grid(n_radial=100, center=center)

    # Transition-like: broad positive Gaussian minus narrow positive Gaussian
    # The result can be negative at small r and positive at large r.
    rho_broad = _gaussian_density(atgrid.points, alpha=0.5, center=center)
    rho_narrow = _gaussian_density(atgrid.points, alpha=5.0, center=center)
    density = rho_broad - 0.8 * rho_narrow  # can be negative near origin

    pot_func = solve_poisson_robust(
        atgrid,
        density,
        tf,
        atnums=np.array([1]),
        atcoords=center.reshape(1, 3),
    )

    V = pot_func(atgrid.points)
    assert np.all(np.isfinite(V)), "Potential must be finite for transition-like density"


def test_robust_poisson_wrong_density_shape_raises():
    """solve_poisson_robust must raise ValueError when density_vals shape mismatches grid."""
    atgrid, tf = _make_single_center_grid(n_radial=80)

    with pytest.raises(ValueError, match="density_vals must be 1-D"):
        solve_poisson_robust(
            atgrid,
            np.ones(10),  # wrong length
            tf,
            atnums=np.array([1]),
            atcoords=np.zeros((1, 3)),
        )


def test_robust_poisson_wrong_points_shape_raises():
    """The returned callable must raise ValueError for wrong-shaped points."""
    atgrid, tf = _make_single_center_grid(n_radial=80)
    density = _gaussian_density(atgrid.points, alpha=1.0)

    pot_func = solve_poisson_robust(
        atgrid,
        density,
        tf,
        atnums=np.array([1]),
        atcoords=np.zeros((1, 3)),
    )

    with pytest.raises(ValueError, match="points must have shape"):
        pot_func(np.array([1.0, 2.0, 3.0]))  # (3,) instead of (1, 3)
