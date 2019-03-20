"""Grid tests file."""
from unittest import TestCase

from grid.grid import Grid

import numpy as np


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
        assert np.allclose(self.grid.points, ref_pts)
        assert np.allclose(self.grid.weights, ref_wts)
        assert self.grid.size == 21

    def test_integrate(self):
        """Test Grid integral."""
        # integral test1
        result2 = self.grid.integrate(np.ones(21))
        assert np.allclose(result2, 2.1)
        # integral test2
        value1 = np.linspace(-1, 1, 21)
        value2 = value1 ** 2
        result3 = self.grid.integrate(value1, value2)
        assert np.allclose(result3, 0)

    def test_getitem(self):
        """Test Grid index and slicing."""
        # test index
        grid_index = self.grid[10]
        ref_grid = Grid(np.array([0]), np.array([0.1]))
        assert np.allclose(grid_index.points, ref_grid.points)
        assert np.allclose(grid_index.weights, ref_grid.weights)
        assert isinstance(grid_index, Grid)
        # test slice
        ref_grid_slice = Grid(np.linspace(-1, 0, 11), np.ones(11) * 0.1)
        grid_slice = self.grid[:11]
        assert np.allclose(grid_slice.points, ref_grid_slice.points)
        assert np.allclose(grid_slice.weights, ref_grid_slice.weights)
        assert isinstance(grid_slice, Grid)
        a = np.array([1, 3, 5])
        ref_smt_index = self.grid[a]
        assert np.allclose(ref_smt_index.points, np.array([-0.9, -0.7, -0.5]))
        assert np.allclose(ref_smt_index.weights, np.array([0.1, 0.1, 0.1]))

    def test_raise(self):
        """Test errors raises."""
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
