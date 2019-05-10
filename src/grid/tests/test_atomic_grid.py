"""Test class for atomic grid."""
from unittest import TestCase

from grid.atomic_grid import AtomicGrid
from grid.basegrid import Grid
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
        ag_ob = AtomicGrid(radial_grid, atomic_rad, scales=scales, degs=degs)
        assert isinstance(ag_ob, AtomicGrid)
        assert len(ag_ob.indices) == 11
        ag_ob = AtomicGrid(
            radial_grid, atomic_rad, scales=np.array([]), degs=np.array([6])
        )
        assert isinstance(ag_ob, AtomicGrid)
        assert len(ag_ob.indices) == 11

    def test_find_l_for_rad_list(self):
        """Test private method find_l_for_rad_list."""
        radial_pts = np.arange(0.1, 1.1, 0.1)
        radial_wts = np.ones(10) * 0.1
        radial_grid = Grid(radial_pts, radial_wts)
        atomic_rad = 1
        scales = np.array([0.2, 0.4, 0.8])
        degs = np.array([3, 5, 7, 3])
        atomic_grid_degree = AtomicGrid._find_l_for_rad_list(
            radial_grid.points, atomic_rad, scales, degs
        )
        assert_equal(atomic_grid_degree, [3, 3, 5, 5, 7, 7, 7, 7, 3, 3])

    def test_preload_unit_sphere_grid(self):
        """Test for private method to preload spherical grids."""
        degs = [3, 3, 5, 5, 7, 7]
        unit_sphere = AtomicGrid._preload_unit_sphere_grid(degs)
        assert len(unit_sphere) == 3
        degs = [3, 4, 5, 6, 7]
        unit_sphere2 = AtomicGrid._preload_unit_sphere_grid(degs)
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
        pts, wts, ind = AtomicGrid._generate_atomic_grid(
            rad_grid, degs, center
        )
        assert len(pts) == 46
        assert_equal(ind, [0, 6, 20, 46])
        # set tests for slicing grid from atomic grid
        for i in range(3):
            # set each layer of points
            ref_grid = generate_lebedev_grid(degree=degs[i])
            # check for each point
            assert_allclose(
                pts[ind[i] : ind[i + 1]], ref_grid.points * rad_pts[i]
            )
            # check for each weight
            assert_allclose(
                wts[ind[i] : ind[i + 1]],
                ref_grid.weights * rad_wts[i] * rad_pts[i] ** 2,
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
        pts, wts, ind = AtomicGrid._generate_atomic_grid(
            rad_grid, degs, center
        )
        ref_pts, ref_wts, ref_ind = AtomicGrid._generate_atomic_grid(
            rad_grid, degs, ref_center
        )
        # diff grid points diff by center and same weights
        assert_allclose(pts, ref_pts)
        assert_allclose(wts, ref_wts)
        # assert_allclose(target_grid.center + ref_center, ref_grid.center)

    def test_error_raises(self):
        """Tests for error raises."""
        with self.assertRaises(TypeError):
            AtomicGrid(np.arange(3), 1.0, scales=np.arange(2), degs=np.arange(3))
        with self.assertRaises(ValueError):
            AtomicGrid(
                Grid(np.arange(3), np.arange(3)),
                1.0,
                scales=np.arange(2),
                degs=np.arange(0),
            )
        with self.assertRaises(ValueError):
            AtomicGrid(
                Grid(np.arange(3), np.arange(3)),
                1.0,
                scales=np.arange(2),
                degs=np.arange(4),
            )
        with self.assertRaises(ValueError):
            AtomicGrid._generate_atomic_grid(
                Grid(np.arange(3), np.arange(3)), np.arange(2), np.array([0, 0, 0])
            )
        with self.assertRaises(TypeError):
            AtomicGrid(
                Grid(np.arange(3), np.arange(3)),
                1.0,
                scales=np.array([0.3, 0.5, 0.7]),
                degs=np.array([3, 5, 7, 5]),
                center=(0, 0, 0),
            )
        with self.assertRaises(ValueError):
            AtomicGrid(
                Grid(np.arange(3), np.arange(3)),
                1.0,
                scales=np.array([0.3, 0.5, 0.7]),
                degs=np.array([3, 5, 7, 5]),
                center=np.array([0, 0, 0, 0]),
            )
