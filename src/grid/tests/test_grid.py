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
"""Grid tests file."""


from unittest import TestCase

from grid.basegrid import Grid, OneDGrid

import numpy as np
from numpy.testing import assert_allclose


class TestGrid(TestCase):
    """Grid testcase class."""

    def setUp(self):
        """Test setup function."""
        points = np.linspace(-1, 1, 21)
        weights = np.ones(21) * 0.1
        self.grid = Grid(points, weights)

    def test_init_grid(self):
        """Test Grid init."""
        # tests property
        assert isinstance(self.grid, Grid)
        ref_pts = np.arange(-1, 1.1, 0.1)
        ref_wts = np.ones(21) * 0.1
        assert_allclose(self.grid.points, ref_pts, atol=1e-7)
        assert_allclose(self.grid.weights, ref_wts)
        assert self.grid.size == 21

    def test_integrate(self):
        """Test Grid integral."""
        # integral test1
        result2 = self.grid.integrate(np.ones(21))
        assert_allclose(result2, 2.1)
        # integral test2
        value1 = np.linspace(-1, 1, 21)
        value2 = value1 ** 2
        result3 = self.grid.integrate(value1, value2)
        assert_allclose(result3, 0, atol=1e-7)

    def test_getitem(self):
        """Test Grid index and slicing."""
        # test index
        grid_index = self.grid[10]
        ref_grid = Grid(np.array([0]), np.array([0.1]))
        assert_allclose(grid_index.points, ref_grid.points)
        assert_allclose(grid_index.weights, ref_grid.weights)
        assert isinstance(grid_index, Grid)
        # test slice
        ref_grid_slice = Grid(np.linspace(-1, 0, 11), np.ones(11) * 0.1)
        grid_slice = self.grid[:11]
        assert_allclose(grid_slice.points, ref_grid_slice.points)
        assert_allclose(grid_slice.weights, ref_grid_slice.weights)
        assert isinstance(grid_slice, Grid)
        a = np.array([1, 3, 5])
        ref_smt_index = self.grid[a]
        assert_allclose(ref_smt_index.points, np.array([-0.9, -0.7, -0.5]))
        assert_allclose(ref_smt_index.weights, np.array([0.1, 0.1, 0.1]))

    def test_errors_raise(self):
        """Test errors raise."""
        # grid init
        weights = np.ones(4)
        points = np.arange(5)
        with self.assertRaises(ValueError):
            Grid(points, weights)
        # integral
        with self.assertRaises(ValueError):
            self.grid.integrate()
        with self.assertRaises(TypeError):
            self.grid.integrate(5)
        with self.assertRaises(ValueError):
            self.grid.integrate(points)


class TestOneDGrid(TestCase):
    """OneDGrid test class."""

    def test_errors_raises(self):
        """Test errors raised."""
        arr_1d = np.arange(10)
        arr_2d = np.arange(20).reshape(4, 5)
        with self.assertRaises(ValueError):
            OneDGrid(arr_2d, arr_1d)
        with self.assertRaises(ValueError):
            OneDGrid(arr_1d, arr_2d)
        with self.assertRaises(ValueError):
            OneDGrid(arr_1d, arr_1d, (0, 5))
        with self.assertRaises(ValueError):
            OneDGrid(arr_1d, arr_1d, (1, 5))
        with self.assertRaises(ValueError):
            OneDGrid(arr_1d, arr_1d, (1, 9))
        with self.assertRaises(ValueError):
            OneDGrid(arr_1d, arr_1d, (0, 1, 2))
        with self.assertRaises(ValueError):
            OneDGrid(arr_1d, arr_1d, (9, 0))

    def test_getitem(self):
        """Test grid indexing."""
        points = np.arange(20)
        weights = np.arange(20) * 0.1
        grid = OneDGrid(points, weights)
        assert grid.size == 20
        assert grid.domain == (0, 19)
        subgrid = grid[0]
        assert subgrid.size == 1
        assert np.allclose(subgrid.points, points[0])
        assert np.allclose(subgrid.weights, weights[0])
        assert subgrid.domain == (0, 19)
        subgrid = grid[3:7]
        assert subgrid.size == 4
        assert np.allclose(subgrid.points, points[3:7])
        assert np.allclose(subgrid.weights, weights[3:7])
        assert subgrid.domain == (0, 19)
