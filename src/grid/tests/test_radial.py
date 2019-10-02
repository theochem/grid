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
"""Radial grid test."""

from grid.basegrid import OneDGrid
from grid.onedgrid import HortonLinear
from grid.rtransform import ExpRTransform, PowerRTransform

import numpy as np
from numpy.testing import assert_almost_equal


def test_basics1():
    """Test basic radial grid transform properties."""
    oned = HortonLinear(4)
    rtf = ExpRTransform(0.1, 1e1)
    grid = rtf.transform_1d_grid(oned)
    assert isinstance(grid, OneDGrid)

    assert grid.size == 4
    # assert grid.shape == (4,)
    # assert grid.rtransform == rtf
    assert (grid.weights > 0).all()
    assert (grid.points == rtf.transform(oned.points)).all()
    # assert grid.zeros().shape == (4,)


def test_basics2():
    """Test basic radial grid transform properties for bigger grid."""
    oned = HortonLinear(100)
    rtf = ExpRTransform(1e-3, 1e1)
    grid = rtf.transform_1d_grid(oned)
    assert isinstance(grid, OneDGrid)

    assert grid.size == 100
    # assert grid.shape == (100,)
    # assert grid.rtransform == rtf
    assert (grid.weights > 0).all()
    assert (grid.points == rtf.transform(oned.points)).all()
    # assert grid.zeros().shape == (100,)


def test_integrate_gauss():
    """Test radial grid integral."""
    oned = HortonLinear(100)
    rtf = PowerRTransform(0.0005, 1e1)
    grid = rtf.transform_1d_grid(oned)
    assert isinstance(grid, OneDGrid)

    y = np.exp(-0.5 * grid.points ** 2)
    # time 4 \pi and r^2 to accommodate old horton test
    grid._weights = grid.weights * 4 * np.pi * grid.points ** 2
    assert_almost_equal(grid.integrate(y), (2 * np.pi) ** 1.5)
    # assert abs(grid.integrate(y) - (2*np.pi)**1.5) < 1e-9
