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


from grid.onedgrid import GaussChebyshev, GaussLaguerre, GaussLegendre, HortonLinear

import numpy as np

import pytest

from scipy.special import roots_legendre


def test_gauss_laguerre():
    """Test Guass Laguerre polynomial grid."""
    points, weights = np.polynomial.laguerre.laggauss(10)
    weights = weights * np.exp(points) * np.power(points, 0)
    grid = GaussLaguerre(10)
    assert np.allclose(grid.points, points)
    assert np.allclose(grid.weights, weights)


def test_gauss_lengendre():
    """Test Guass Lengendre polynomial grid."""
    points, weights = roots_legendre(10)
    grid = GaussLegendre(10)
    assert np.allclose(grid.points, points)
    assert np.allclose(grid.weights, weights)


def test_gauss_chebyshev():
    """Test Guass Chebyshev polynomial grid."""
    points, weights = np.polynomial.chebyshev.chebgauss(10)
    weights = weights * np.sqrt(1 - np.power(points, 2))
    grid = GaussChebyshev(10)
    assert np.allclose(grid.points, np.sort(points))
    assert np.allclose(grid.weights, weights)


def test_horton_linear():
    """Test horton linear grids."""
    grid = HortonLinear(10)
    assert np.allclose(grid.points, np.arange(10))
    assert np.allclose(grid.weights, np.ones(10))


def test_errors_raise():
    """Test errors raise."""
    with pytest.raises(ValueError) as error:
        GaussLaguerre(10, -1)
    assert str(error.value) == "Alpha need to be bigger than -1, given -1"
