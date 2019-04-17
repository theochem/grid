"""Test class for atomic grid."""
from unittest import TestCase

from grid.atomic_grid import AtomicGridFactory
from grid.grid import AtomicGrid, Grid
from grid.lebedev import generate_lebedev_grid

import numpy as np
from numpy.testing import assert_allclose, assert_equal


class TestAtomicGrid(TestCase):
    """Atomic grid factory test class."""

    def test_total_atomic_grid(self):
        """Normal initialization test."""
        radial_pts = np.arange(0.1, 1.1, 0.1)
        radial_wts = np.ones(10) * 0.1
        radial_grid = Grid(radial_pts, radial_wts)
        atomic_rad = 0.5
        scales = np.array([0.5, 1, 1.5])
        degs = np.array([6, 14, 14, 6])
        # generate a proper instance without failing.
        ag_ob = AtomicGridFactory(radial_grid, atomic_rad, scales=scales, degs=degs)
        assert isinstance(ag_ob.atomic_grid, AtomicGrid)
        assert len(ag_ob.indices) == 11

    def test_find_l_for_rad_list(self):
        """Test private method find_l_for_rad_list."""
        radial_pts = np.arange(0.1, 1.1, 0.1)
        radial_wts = np.ones(10) * 0.1
        radial_grid = Grid(radial_pts, radial_wts)
        atomic_rad = 1
        scales = np.array([0.2, 0.4, 0.8])
        degs = np.array([3, 5, 7, 3])
        atomic_grid_degree = AtomicGridFactory._find_l_for_rad_list(
            radial_grid.points, atomic_rad, scales, degs
        )
        assert_equal(atomic_grid_degree, [3, 3, 5, 5, 7, 7, 7, 7, 3, 3])

    def test_preload_unit_sphere_grid(self):
        """Test for private method to preload spherical grids."""
        degs = [3, 3, 5, 5, 7, 7]
        unit_sphere = AtomicGridFactory._preload_unit_sphere_grid(degs)
        assert len(unit_sphere) == 3
        degs = [3, 4, 5, 6, 7]
        unit_sphere2 = AtomicGridFactory._preload_unit_sphere_grid(degs)
        assert len(unit_sphere2) == 5
        assert_allclose(unit_sphere2[4].points, unit_sphere2[5].points)
        assert_allclose(unit_sphere2[6].points, unit_sphere2[7].points)
        assert not np.allclose(
            unit_sphere2[4].points.shape, unit_sphere2[6].points.shape
        )

    def test_generate_atomic_grid(self):
        """Test for generating atomic grid."""
        # setup testing class
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = Grid(rad_pts, rad_wts)
        degs = np.array([3, 5, 7])
        center = np.array([0, 0, 0])
        target_grid, ind = AtomicGridFactory._generate_atomic_grid(
            rad_grid, degs, center
        )
        assert target_grid.size == 46
        assert_equal(ind, [0, 6, 20, 46])
        # set tests for slicing grid from atomic grid
        for i in range(3):
            # set each layer of points
            ref_grid = generate_lebedev_grid(degree=degs[i])
            # check for each point
            assert_allclose(
                target_grid.points[ind[i] : ind[i + 1]], ref_grid.points * rad_pts[i]
            )
            # check for each weight
            assert_allclose(
                target_grid.weights[ind[i] : ind[i + 1]], ref_grid.weights * rad_wts[i]
            )

    def test_atomic_grid(center):
        """Test atomic grid center transilation."""
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = Grid(rad_pts, rad_wts)
        degs = np.array([3, 5, 7])
        # origin center
        center = np.array([0, 0, 0])
        # randome center
        ref_center = np.random.rand(3)
        target_grid, ind = AtomicGridFactory._generate_atomic_grid(
            rad_grid, degs, center
        )
        ref_grid, ref_ind = AtomicGridFactory._generate_atomic_grid(
            rad_grid, degs, ref_center
        )
        # diff grid points diff by center and same weights
        assert_allclose(target_grid.points + ref_center, ref_grid.points)
        assert_allclose(target_grid.weights, ref_grid.weights)
        assert_allclose(target_grid.center + ref_center, ref_grid.center)

    def test_error_raises(self):
        """Tests for error raises."""
        with self.assertRaises(TypeError):
            AtomicGridFactory(np.arange(3), 1.0, scales=np.arange(2), degs=np.arange(3))
        with self.assertRaises(ValueError):
            AtomicGridFactory(
                Grid(np.arange(3), np.arange(3)),
                1.0,
                scales=np.arange(2),
                degs=np.arange(0),
            )
        with self.assertRaises(ValueError):
            AtomicGridFactory(
                Grid(np.arange(3), np.arange(3)),
                1.0,
                scales=np.arange(2),
                degs=np.arange(4),
            )
        with self.assertRaises(ValueError):
            AtomicGridFactory._generate_atomic_grid(
                Grid(np.arange(3), np.arange(3)), np.arange(2), np.array([0, 0, 0])
            )
        with self.assertRaises(TypeError):
            AtomicGridFactory(
                Grid(np.arange(3), np.arange(3)),
                1.0,
                scales=np.array([0.3, 0.5, 0.7]),
                degs=np.array([3, 5, 7, 5]),
                center=(0, 0, 0),
            )
        with self.assertRaises(ValueError):
            AtomicGridFactory(
                Grid(np.arange(3), np.arange(3)),
                1.0,
                scales=np.array([0.3, 0.5, 0.7]),
                degs=np.array([3, 5, 7, 5]),
                center=np.array([0, 0, 0, 0]),
            )
