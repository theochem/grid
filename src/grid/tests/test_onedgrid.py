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
from numpy.testing import assert_allclose

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
