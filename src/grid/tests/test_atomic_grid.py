from unittest import TestCase

import numpy as np

from grid.atomic_grid import AtomicGridFactory
from grid.grid import Grid


class TestAtomicGrid(TestCase):
    def test_total_atomic_grid(self):
        radial_pts = np.arange(0.1, 1.1, 0.1)
        radial_wts = np.ones(10) * 0.1
        radial_grid = Grid(radial_pts, radial_wts)
        atomic_rad = 0.5
        scales = np.array([0.5, 1, 1.5])
        degs = np.array([6, 14, 14, 6])
        self.agf = AtomicGridFactory(radial_grid, atomic_rad, scales, degs)

    def test_find_l_for_rad_list(self):
        radial_pts = np.arange(0.1, 1.1, 0.1)
        radial_wts = np.ones(10) * 0.1
        radial_grid = Grid(radial_pts, radial_wts)
        atomic_rad = 1
        scales = np.array([0.2, 0.4, 0.8])
        degs = np.array([3, 5, 7, 3])
        atomic_grid_degree = AtomicGridFactory._find_l_for_rad_list(
            radial_grid.points, atomic_rad, scales, degs
        )
        assert np.allclose(atomic_grid_degree, [3, 3, 5, 5, 7, 7, 7, 7, 3, 3])

    def test_preload_unit_sphere_grid(self):
        degs = [3, 3, 5, 5, 7, 7]
        unit_sphere = AtomicGridFactory._preload_unit_sphere_grid(degs)
        assert len(unit_sphere) == 3
        degs = [3, 4, 5, 6, 7]
        unit_sphere2 = AtomicGridFactory._preload_unit_sphere_grid(degs)
        assert len(unit_sphere2) == 5
        assert np.allclose(unit_sphere2[4].points, unit_sphere2[5].points)
