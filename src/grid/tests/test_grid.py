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

from grid.basegrid import Grid, OneDGrid, SubGrid

import numpy as np
from numpy.testing import assert_allclose


class TestGrid(TestCase):
    """Grid testcase class."""

    def setUp(self):
        """Test setup function."""
        self._ref_points = np.linspace(-1, 1, 21)
        self._ref_weights = np.ones(21) * 0.1
        self.grid = Grid(self._ref_points, self._ref_weights)

    def test_init_grid(self):
        """Test Grid init."""
        # tests property
        assert isinstance(self.grid, Grid)
        assert_allclose(self.grid.points, self._ref_points, atol=1e-7)
        assert_allclose(self.grid.weights, self._ref_weights)
        assert self.grid.size == self._ref_weights.size

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
        ref_grid = Grid(self._ref_points[10:11], self._ref_weights[10:11])
        assert_allclose(grid_index.points, ref_grid.points)
        assert_allclose(grid_index.weights, ref_grid.weights)
        assert isinstance(grid_index, Grid)
        # test slice
        ref_grid_slice = Grid(self._ref_points[:11], self._ref_weights[:11])
        grid_slice = self.grid[:11]
        assert_allclose(grid_slice.points, ref_grid_slice.points)
        assert_allclose(grid_slice.weights, ref_grid_slice.weights)
        assert isinstance(grid_slice, Grid)
        a = np.array([1, 3, 5])
        ref_smt_index = self.grid[a]
        assert_allclose(ref_smt_index.points, self._ref_points[a])
        assert_allclose(ref_smt_index.weights, self._ref_weights[a])

    def test_get_subgrid(self):
        """Test the creation of the subgrid with a normal radius."""
        center = self.grid.points[3]
        radius = 0.2
        subgrid = self.grid.get_subgrid(center, radius)
        # Just make sure we are testing with an actual subgrid with less (but
        # not zero) points.
        assert subgrid.size > 0
        assert subgrid.size < self.grid.size
        # Test that the subgrid contains the correct results.
        assert subgrid.points.ndim == self.grid.points.ndim
        assert subgrid.weights.ndim == self.grid.weights.ndim
        assert_allclose(subgrid.points, self.grid.points[subgrid.indices])
        assert_allclose(subgrid.weights, self.grid.weights[subgrid.indices])
        if self._ref_points.ndim == 2:
            assert (np.linalg.norm(subgrid.points - center, axis=1) <= radius).all()
        else:
            assert (abs(subgrid.points - center) <= radius).all()

    def test_get_subgrid_radius_inf(self):
        """Test the creation of the subgrid with an infinite radius."""
        subgrid = self.grid.get_subgrid(self.grid.points[3], np.inf)
        # Just make sure we are testing with a real subgrid
        assert subgrid.size == self.grid.size
        assert_allclose(subgrid.points, self.grid.points)
        assert_allclose(subgrid.weights, self.grid.weights)
        assert_allclose(subgrid.indices, np.arange(self.grid.size))

    def test_errors_raise(self):
        """Test errors raise."""
        # grid init
        with self.assertRaises(ValueError):
            Grid(self._ref_points, np.ones(len(self._ref_weights) + 1))
        with self.assertRaises(ValueError):
            Grid(self._ref_points.reshape(self.grid.size, 1, 1), self._ref_weights)
        with self.assertRaises(ValueError):
            Grid(self._ref_points, self._ref_weights.reshape(-1, 1))
        # integral
        with self.assertRaises(ValueError):
            self.grid.integrate()
        with self.assertRaises(TypeError):
            self.grid.integrate(5)
        if self._ref_points.ndim == 2:
            with self.assertRaises(ValueError):
                self.grid.integrate(self._ref_points)
        # get_subgrid
        with self.assertRaises(ValueError):
            self.grid.get_subgrid(self._ref_points[0], -1)
        with self.assertRaises(ValueError):
            self.grid.get_subgrid(self._ref_points[0], -np.inf)
        with self.assertRaises(ValueError):
            self.grid.get_subgrid(self._ref_points[0], np.nan)
        if self._ref_points.ndim == 2:
            with self.assertRaises(ValueError):
                self.grid.get_subgrid(np.zeros(self._ref_points.shape[1] + 1), 5.0)
        else:
            with self.assertRaises(ValueError):
                self.grid.get_subgrid(np.zeros(2), 5.0)


class TestGrid1D(TestGrid):
    """Grid testcase class for 1D point arrays."""

    def setUp(self):
        """Test setup function."""
        self._ref_points = np.linspace(-1, 1, 21).reshape(-1, 1)
        self._ref_weights = np.ones(21) * 0.1
        self.grid = Grid(self._ref_points, self._ref_weights)


class TestGrid2D(TestGrid):
    """Grid testcase class for 2D point arrays."""

    def setUp(self):
        """Test setup function."""
        self._ref_points = np.array([np.linspace(-1, 1, 21)] * 2).T
        self._ref_weights = np.ones(21) * 0.1
        self.grid = Grid(self._ref_points, self._ref_weights)


class TestGrid3D(TestGrid):
    """Grid testcase class for 3D point arrays."""

    def setUp(self):
        """Test setup function."""
        self._ref_points = np.array([np.linspace(-1, 1, 21)] * 3).T
        self._ref_weights = np.ones(21) * 0.1
        self.grid = Grid(self._ref_points, self._ref_weights)


class TestSubGrid(TestCase):
    """SubGrid test class."""

    def test_properties(self):
        """Test consistency of the properties with constructor arguments."""
        weights = np.ones(4)
        points = np.arange(12).reshape(4, 3)
        center = np.array([4.0, 5.0, 6.0])
        indices = np.arange(4)
        sg = SubGrid(points, weights, center, indices)
        assert_allclose(sg.weights, weights)
        assert_allclose(sg.points, points)
        assert_allclose(sg.center, center)
        assert_allclose(sg.indices, indices)

    def test_errors_raise(self):
        """Test exceptions with invalid construct arguments."""
        weights = np.ones(4)
        points = np.arange(12).reshape(4, 3)
        center = np.zeros(3)
        with self.assertRaises(ValueError):
            SubGrid(points, weights, center, np.arange(5))
        with self.assertRaises(ValueError):
            SubGrid(points, weights, center, np.arange(8).reshape(4, 2))


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
            OneDGrid(arr_1d, arr_1d[:, np.newaxis])
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
        assert grid.domain is None
        subgrid = grid[0]
        assert subgrid.size == 1
        assert np.allclose(subgrid.points, points[0])
        assert np.allclose(subgrid.weights, weights[0])
        assert subgrid.domain is None
        subgrid = grid[3:7]
        assert subgrid.size == 4
        assert np.allclose(subgrid.points, points[3:7])
        assert np.allclose(subgrid.weights, weights[3:7])
        assert subgrid.domain is None
