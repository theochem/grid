# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2019 The GRID Development Team
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
"""Poisson test module."""
from grid.atomgrid import AtomGrid
from grid.onedgrid import OneDGrid, GaussChebyshev, GaussLaguerre, Trapezoidal
from grid.poisson import interpolate_laplacian, solve_poisson_bvp, solve_poisson_ivp
from grid.rtransform import (
    BeckeRTransform, IdentityRTransform, InverseRTransform, KnowlesRTransform,
    LinearFiniteRTransform
)

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import erf


def test_interpolation_of_laplacian_with_spherical_harmonic():
    r"""Test the interpolation of Laplacian of spherical harmonic is eigenvector."""
    odg = OneDGrid(np.linspace(1e-5, 1, num=20), np.ones(20), (0, np.inf))
    degree = 6 * 2 + 2
    atgrid = AtomGrid.from_pruned(odg, 1, sectors_r=[], sectors_degree=[degree])

    def func(sph_points):
        # Spherical harmonic of degree 6 and order 0
        r, phi, theta = sph_points.T
        return np.sqrt(2.0) * np.sqrt(13) / (np.sqrt(np.pi) * 32) * (
                231 * np.cos(theta) ** 6.0 - 315 * np.cos(theta) ** 4.0 + 105 * np.cos(
            theta) ** 2.0 - 5.0
        )
    # Get spherical points from atomic grid
    spherical_pts = atgrid.convert_cartesian_to_spherical()
    func_values = func(spherical_pts)

    laplacian = interpolate_laplacian(atgrid, func_values)

    # Test on the same points used for interpolation and random points.
    for grid in [atgrid.points, np.random.uniform(-0.75, 0.75, (250, 3))]:
        actual = laplacian(grid)
        spherical_pts = atgrid.convert_cartesian_to_spherical(grid)
        # Eigenvector spherical harmonic times l(l + 1) / r^2
        with np.errstate(divide='ignore', invalid='ignore'):
            desired = -func(spherical_pts) * 6 * (6 + 1) / spherical_pts[:, 0]**2.0
        # desired[spherical_pts[:, 0]**2.0 < 1e-10] = -np.inf
        np.set_printoptions(threshold=np.inf)
        assert_allclose(actual, desired)


def test_interpolation_of_laplacian_of_exponential():
    r"""Test the interpolation of Laplacian of exponential."""
    oned = Trapezoidal(5000)
    btf = BeckeRTransform(0.0, 1.5)
    radial = btf.transform_1d_grid(oned)
    degree = 10
    atgrid = AtomGrid.from_pruned(radial, 1, sectors_r=[], sectors_degree=[degree])

    def func(cart_pts):
        radius = np.linalg.norm(cart_pts, axis=1)
        return np.exp(-radius)

    func_values = func(atgrid.points)

    laplacian = interpolate_laplacian(atgrid, func_values)

    # Test on the same points used for interpolation and random points.
    for grid in [atgrid.points, np.random.uniform(-0.5, 0.5, (250, 3))]:
        actual = laplacian(grid)
        spherical_pts = atgrid.convert_cartesian_to_spherical(grid)
        spherical_pts[:, 0][spherical_pts[:, 0] < 1e-6] = 1e-6
        # Laplacian of exponential is e^-x (x - 2) / x
        desired = np.exp(-spherical_pts[:, 0]) * (spherical_pts[:, 0] - 2.0) /\
                  spherical_pts[:, 0]
        assert_almost_equal(actual, desired, decimal=3)