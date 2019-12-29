# -*- coding: utf-8 -*-
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
"""Tests one-dimensional grids."""

from unittest import TestCase

from grid.onedgrid import GaussChebyshev, GaussLaguerre, GaussLegendre, HortonLinear

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from scipy.special import roots_legendre


class TestOneDGrid(TestCase):
    """OneDGrid test class."""

    def setUp(self):
        """Test setup function."""
        ...

    def test_gausslaguerre(self):
        """Test Guass Laguerre polynomial grid."""
        points, weights = np.polynomial.laguerre.laggauss(10)
        weights = weights * np.exp(points) * np.power(points, 0)
        grid = GaussLaguerre(10)
        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_gausslengendre(self):
        """Test Guass Lengendre polynomial grid."""
        points, weights = roots_legendre(10)
        grid = GaussLegendre(10)
        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_gausschebyshev(self):
        """Test Guass Chebyshev polynomial grid."""
        points, weights = np.polynomial.chebyshev.chebgauss(10)
        weights = weights * np.sqrt(1 - np.power(points, 2))
        grid = GaussChebyshev(10)
        assert_allclose(grid.points, np.sort(points))
        assert_allclose(grid.weights, weights)

    def test_horton_linear(self):
        """Test horton linear grids."""
        grid = HortonLinear(10)
        assert_allclose(grid.points, np.arange(10))
        assert_allclose(grid.weights, np.ones(10))

    def test_errors_raise(self):
        """Test errors raise."""
        with self.assertRaises(ValueError):
            GaussLaguerre(10, -1)

    @staticmethod
    def helper_gaussian(x):
        """Compute gauss function for integral between [-1, 1]."""
        # integrate (exp(-x^2)) x=[-1, 1], result = 1.49365
        return np.exp(-(x ** 2))

    @staticmethod
    def helper_quadratic(x):
        """Compute quadratic function for integral between [-1, 1]."""
        # integrate (-x^2 + 1) x=[-1, 1], result = 1.33333
        return -(x ** 2) + 1

    def test_oned_integral(self):
        """A simple integral tests for basic oned grid."""
        # create candidate function
        # add more ``quadratures: npoints`` if needed
        candidates_quadratures = {GaussChebyshev: 15, GaussLegendre: 15}
        # loop each pair to create pts instance
        for quadrature, n_points in candidates_quadratures.items():
            grid = quadrature(n_points)
            # compute gauss numpymerical integral value
            f1_value = np.sum(self.helper_gaussian(grid.points) * grid.weights)
            # ref value
            ref_value = 1.49365
            assert_almost_equal(f1_value, ref_value, decimal=3)

            # compute quadratic integral value
            f2_value = np.sum(self.helper_quadratic(grid.points) * grid.weights)
            # ref value
            ref_value = 1.33333
            assert_almost_equal(f2_value, ref_value, decimal=3)
            print(f"{quadrature.__name__} passed the tests.")
