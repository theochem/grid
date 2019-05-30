"""Integral test module."""
from unittest import TestCase

from grid.basegrid import Grid
from grid.integral import _v1_v2_low, elec_elec_integral, two_grid_integral

import numpy as np
from numpy.testing import assert_allclose  # , assert_almost_equal


class TestIntegral(TestCase):
    """Integral test class."""

    def test_r1_r2_low(self):
        """Test low level two array interaction function."""
        r1 = np.arange(4)
        r2 = np.arange(4)
        result = _v1_v2_low(r1, r2, lambda x, y: x - y)
        for i in r1:
            for j in r2:
                assert result[i, j] == i - j

        r1 = np.random.rand(10, 3)
        r2 = np.random.rand(9, 3)
        result = _v1_v2_low(r1, r2, lambda r1, r2: 1 / np.linalg.norm(r1 - r2, axis=-1))
        for i, n in enumerate(r1):
            for j, m in enumerate(r2):
                assert_allclose(result[i, j], 1 / np.linalg.norm(n - m))

        v1 = np.arange(1, 5)
        v2 = np.arange(2, 6)
        result = _v1_v2_low(v1, v2, lambda a, b: a * b)
        for i, n in enumerate(v1):
            for j, m in enumerate(v2):
                result[i, j] == n * m

    def test_two_grid_int(self):
        """Test two API for same ele ele interaction."""
        points1 = np.zeros((10, 3))
        points2 = np.zeros((10, 3))
        points1[:, 0] = np.arange(0, 5, 0.5)
        points2[:, 0] = np.arange(0.25, 5.25, 0.5)
        grid1 = Grid(points1, np.ones(10) * 0.5)
        grid2 = Grid(points2, np.ones(10) * 0.5)
        result = elec_elec_integral(
            grid1,
            grid2,
            np.sum(grid1.points ** 2, axis=-1),
            np.sum(grid2.points ** 2, axis=-1),
        )
        ref_v1 = np.arange(0, 5.0, 0.5) ** 2
        ref_v2 = np.arange(0.25, 5.25, 0.5) ** 2
        result2 = two_grid_integral(
            grid1,
            grid2,
            ref_v1,
            ref_v2,
            func_rad=lambda x, y: 1 / np.linalg.norm(x - y, axis=-1),
            func_val=lambda x, y: x * y,
        )
        assert_allclose(result, result2)

    def test_errors_raise(self):
        """Test different error raiese."""
        with self.assertRaises(TypeError):
            two_grid_integral(1, 1, 1, 1, func_rad=1, func_val=1)
        grid1 = Grid(np.arange(3), np.arange(3))
        grid2 = Grid(np.arange(3), np.arange(3))
        with self.assertRaises(TypeError):
            two_grid_integral(grid1, 1, 1, 1, func_rad=1, func_val=1)
        with self.assertRaises(TypeError):
            two_grid_integral(grid1, grid2, 1, 1, func_rad=1, func_val=1)
        with self.assertRaises(TypeError):
            two_grid_integral(grid1, grid2, np.arange(1, 4), 1, func_rad=1, func_val=1)
        with self.assertRaises(TypeError):
            two_grid_integral(
                grid1, grid2, np.arange(1, 4), np.arange(1, 4), func_rad=1, func_val=1
            )
        with self.assertRaises(TypeError):
            two_grid_integral(
                grid1,
                grid2,
                np.arange(1, 4),
                np.arange(1, 4),
                func_rad=lambda x, y: 1,
                func_val=1,
            )
        with self.assertRaises(ValueError):
            elec_elec_integral(grid1, grid2, np.arange(1, 4), np.arange(1, 4))
        grid1 = Grid(np.random.rand(3, 3), np.arange(3))
        with self.assertRaises(ValueError):
            elec_elec_integral(grid1, grid2, np.arange(1, 4), np.arange(1, 4))
