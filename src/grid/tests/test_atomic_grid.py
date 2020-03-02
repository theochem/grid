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
"""Test class for atomic grid."""


from unittest import TestCase

from grid.atomic_grid import AtomicGrid
from grid.basegrid import AngularGrid, Grid, OneDGrid
from grid.lebedev import generate_lebedev_grid
from grid.onedgrid import HortonLinear
from grid.rtransform import PowerRTransform

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)

from scipy.spatial.transform import Rotation as R


class TestAtomicGrid(TestCase):
    """Atomic grid factory test class."""

    def test_total_atomic_grid(self):
        """Normal initialization test."""
        radial_pts = np.arange(0.1, 1.1, 0.1)
        radial_wts = np.ones(10) * 0.1
        rgrid = OneDGrid(radial_pts, radial_wts)
        rad = 0.5
        r_sectors = np.array([0.5, 1, 1.5])
        degs = np.array([6, 14, 14, 6])
        # generate a proper instance without failing.
        ag_ob = AtomicGrid.special_init(
            rgrid, radius=rad, r_sectors=r_sectors, degs=degs
        )
        assert isinstance(ag_ob, AtomicGrid)
        assert len(ag_ob.indices) == 11
        assert ag_ob.l_max == 15
        ag_ob = AtomicGrid.special_init(
            rgrid, radius=rad, r_sectors=np.array([]), degs=np.array([6])
        )
        assert isinstance(ag_ob, AtomicGrid)
        assert len(ag_ob.indices) == 11
        ag_ob = AtomicGrid(rgrid, nums=[110])
        assert ag_ob.l_max == 17
        assert_array_equal(ag_ob._rad_degs, np.ones(10) * 17)
        assert ag_ob.size == 110 * 10
        # new init AtomicGrid
        ag_ob2 = AtomicGrid(rgrid, degs=[17])
        assert ag_ob2.l_max == 17
        assert_array_equal(ag_ob2._rad_degs, np.ones(10) * 17)
        assert ag_ob2.size == 110 * 10
        assert isinstance(ag_ob.rad_grid, OneDGrid)
        assert_allclose(ag_ob.rad_grid.points, rgrid.points)
        assert_allclose(ag_ob.rad_grid.weights, rgrid.weights)

    def test_quick_grid(self):
        """Test grid construction with predefined grid."""
        # test coarse grid
        pts = HortonLinear(20)
        tf = PowerRTransform(7.0879993828935345e-06, 16.05937640019924)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomicGrid.quick_grid(1, rad_grid, "coarse")
        # 604 points for coarse H atom
        assert_equal(atgrid.size, 604)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points ** 2, axis=1)) * atgrid.weights),
            5.56840953,
        )

        # test medium grid
        pts = HortonLinear(24)
        tf = PowerRTransform(3.69705074304963e-06, 19.279558946793685)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomicGrid.quick_grid(1, rad_grid, "medium")
        # 928 points for coarse H atom
        assert_equal(atgrid.size, 928)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points ** 2, axis=1)) * atgrid.weights),
            5.56834559,
        )
        # test fine grid
        pts = HortonLinear(34)
        tf = PowerRTransform(2.577533167224667e-07, 16.276983371222354)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomicGrid.quick_grid(1, rad_grid, "fine")
        # 1984 points for coarse H atom
        assert_equal(atgrid.size, 1984)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points ** 2, axis=1)) * atgrid.weights),
            5.56832800,
        )
        # test veryfine grid
        pts = HortonLinear(41)
        tf = PowerRTransform(1.1774580743206259e-07, 20.140888089596444)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomicGrid.quick_grid(1, rad_grid, "veryfine")
        # 3154 points for coarse H atom
        assert_equal(atgrid.size, 3154)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points ** 2, axis=1)) * atgrid.weights),
            5.56832800,
        )
        # test ultrafine grid
        pts = HortonLinear(49)
        tf = PowerRTransform(4.883104847991021e-08, 21.05456999309752)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomicGrid.quick_grid(1, rad_grid, "ultrafine")
        # 4546 points for coarse H atom
        assert_equal(atgrid.size, 4546)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points ** 2, axis=1)) * atgrid.weights),
            5.56832800,
        )
        # test insane grid
        pts = HortonLinear(59)
        tf = PowerRTransform(1.9221827244049134e-08, 21.413278983919113)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomicGrid.quick_grid(1, rad_grid, "insane")
        # 6622 points for coarse H atom
        assert_equal(atgrid.size, 6622)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points ** 2, axis=1)) * atgrid.weights),
            5.56832800,
        )

    def test_special_init_with_degs_and_nums(self):
        """Test different initilize method."""
        radial_pts = np.arange(0.1, 1.1, 0.1)
        radial_wts = np.ones(10) * 0.1
        rgrid = OneDGrid(radial_pts, radial_wts)
        rad = 0.5
        r_sectors = np.array([0.5, 1, 1.5])
        degs = np.array([3, 5, 7, 5])
        nums = np.array([6, 14, 26, 14])
        # construct atomic grid with degs
        atgrid1 = AtomicGrid.special_init(
            rgrid, radius=rad, r_sectors=r_sectors, degs=degs
        )
        # construct atomic grid with nums
        atgrid2 = AtomicGrid.special_init(
            rgrid, radius=rad, r_sectors=r_sectors, nums=nums
        )
        # test two grids are the same
        assert_equal(atgrid1.size, atgrid2.size)
        assert_allclose(atgrid1.points, atgrid2.points)
        assert_allclose(atgrid1.weights, atgrid2.weights)

    def test_find_l_for_rad_list(self):
        """Test private method find_l_for_rad_list."""
        radial_pts = np.arange(0.1, 1.1, 0.1)
        radial_wts = np.ones(10) * 0.1
        rgrid = OneDGrid(radial_pts, radial_wts)
        rad = 1
        r_sectors = np.array([0.2, 0.4, 0.8])
        degs = np.array([3, 5, 7, 3])
        atomic_grid_degree = AtomicGrid._find_l_for_rad_list(
            rgrid.points, rad * r_sectors, degs
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
        rad_grid = OneDGrid(rad_pts, rad_wts)
        degs = np.array([3, 5, 7])
        pts, wts, ind = AtomicGrid._generate_atomic_grid(rad_grid, degs)
        assert len(pts) == 46
        assert_equal(ind, [0, 6, 20, 46])
        # set tests for slicing grid from atomic grid
        for i in range(3):
            # set each layer of points
            ref_grid = generate_lebedev_grid(degree=degs[i])
            # check for each point
            assert_allclose(pts[ind[i] : ind[i + 1]], ref_grid.points * rad_pts[i])
            # check for each weight
            assert_allclose(
                wts[ind[i] : ind[i + 1]],
                ref_grid.weights * rad_wts[i] * rad_pts[i] ** 2,
            )

    def test_atomic_grid(self):
        """Test atomic grid center transilation."""
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = OneDGrid(rad_pts, rad_wts)
        degs = np.array([3, 5, 7])
        # origin center
        # randome center
        pts, wts, ind = AtomicGrid._generate_atomic_grid(rad_grid, degs)
        ref_pts, ref_wts, ref_ind = AtomicGrid._generate_atomic_grid(rad_grid, degs)
        # diff grid points diff by center and same weights
        assert_allclose(pts, ref_pts)
        assert_allclose(wts, ref_wts)
        # assert_allclose(target_grid.center + ref_center, ref_grid.center)

    def test_atomic_rotate(self):
        """Test random rotation for atomic grid."""
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = OneDGrid(rad_pts, rad_wts)
        degs = [3, 5, 7]
        atgrid = AtomicGrid(rad_grid, degs=degs)
        # make sure True and 1 is not the same result
        atgrid1 = AtomicGrid(rad_grid, degs=degs, rotate=True)
        atgrid2 = AtomicGrid(rad_grid, degs=degs, rotate=1)
        # test diff points, same weights
        assert not np.allclose(atgrid.points, atgrid1.points)
        assert not np.allclose(atgrid.points, atgrid2.points)
        assert not np.allclose(atgrid1.points, atgrid2.points)
        assert_allclose(atgrid.weights, atgrid1.weights)
        assert_allclose(atgrid.weights, atgrid2.weights)
        assert_allclose(atgrid1.weights, atgrid2.weights)
        # test same integral
        value = np.prod(atgrid.points ** 2, axis=-1)
        value1 = np.prod(atgrid.points ** 2, axis=-1)
        value2 = np.prod(atgrid.points ** 2, axis=-1)
        res = atgrid.integrate(value)
        res1 = atgrid1.integrate(value1)
        res2 = atgrid2.integrate(value2)
        assert_almost_equal(res, res1)
        assert_almost_equal(res1, res2)
        # test rotated shells
        for i in range(len(degs)):
            non_rot_shell = atgrid.get_shell_grid(i).points
            rot_shell = atgrid2.get_shell_grid(i).points
            rot_mt = R.random(random_state=1 + i).as_matrix()
            assert_allclose(rot_shell, non_rot_shell @ rot_mt)

    def test_get_shell_grid(self):
        """Test angular grid get from get_shell_grid function."""
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = OneDGrid(rad_pts, rad_wts)
        degs = [3, 5, 7]
        atgrid = AtomicGrid(rad_grid, degs=degs)
        assert atgrid.n_shells == 3
        # grep shell with r^2
        for i in range(atgrid.n_shells):
            sh_grid = atgrid.get_shell_grid(i)
            assert isinstance(sh_grid, AngularGrid)
            ref_grid = generate_lebedev_grid(degree=degs[i])
            assert np.allclose(sh_grid.points, ref_grid.points * rad_pts[i])
            assert np.allclose(
                sh_grid.weights, ref_grid.weights * rad_wts[i] * rad_pts[i] ** 2
            )
        # grep shell without r^2
        for i in range(atgrid.n_shells):
            sh_grid = atgrid.get_shell_grid(i, r_sq=False)
            assert isinstance(sh_grid, AngularGrid)
            ref_grid = generate_lebedev_grid(degree=degs[i])
            assert np.allclose(sh_grid.points, ref_grid.points * rad_pts[i])
            assert np.allclose(sh_grid.weights, ref_grid.weights * rad_wts[i])

    def test_convert_points_to_sph(self):
        """Test convert random points to spherical based on atomic structure."""
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = OneDGrid(rad_pts, rad_wts)
        center = np.random.rand(3)
        atgrid = AtomicGrid(rad_grid, degs=[7], center=center)
        points = np.random.rand(100, 3)
        calc_sph = atgrid.convert_point_to_sph(points)
        # compute ref result
        ref_coor = points - center
        # radius
        r = np.linalg.norm(ref_coor, axis=-1)
        # azimuthal
        theta = np.arctan2(ref_coor[:, 1], ref_coor[:, 0])
        # polar
        phi = np.arccos(ref_coor[:, 2] / r)
        assert_allclose(np.stack([r, theta, phi]).T, calc_sph)

        # test single point
        point = np.random.rand(3)
        calc_sph = atgrid.convert_point_to_sph(point)
        ref_coor = point - center
        r = np.linalg.norm(ref_coor)
        theta = np.arctan2(ref_coor[1], ref_coor[0])
        phi = np.arccos(ref_coor[2] / r)
        assert_allclose(np.array([r, theta, phi]).reshape(-1, 3), calc_sph)

    def test_error_raises(self):
        """Tests for error raises."""
        with self.assertRaises(TypeError):
            AtomicGrid.special_init(
                np.arange(3), 1.0, r_sectors=np.arange(2), degs=np.arange(3)
            )
        with self.assertRaises(ValueError):
            AtomicGrid.special_init(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                r_sectors=np.arange(2),
                degs=np.arange(0),
            )
        with self.assertRaises(ValueError):
            AtomicGrid.special_init(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                r_sectors=np.arange(2),
                degs=np.arange(4),
            )
        with self.assertRaises(ValueError):
            AtomicGrid._generate_atomic_grid(
                OneDGrid(np.arange(3), np.arange(3)), np.arange(2)
            )
        with self.assertRaises(TypeError):
            AtomicGrid.special_init(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                r_sectors=np.array([0.3, 0.5, 0.7]),
                degs=np.array([3, 5, 7, 5]),
                center=(0, 0, 0),
            )
        with self.assertRaises(ValueError):
            AtomicGrid.special_init(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                r_sectors=np.array([0.3, 0.5, 0.7]),
                degs=np.array([3, 5, 7, 5]),
                center=np.array([0, 0, 0, 0]),
            )

        with self.assertRaises(TypeError):
            AtomicGrid(OneDGrid(np.arange(3), np.arange(3)), nums=110)
        with self.assertRaises(TypeError):
            AtomicGrid(OneDGrid(np.arange(3), np.arange(3)), degs=17)
        with self.assertRaises(ValueError):
            AtomicGrid(OneDGrid(np.arange(3), np.arange(3)), degs=[17], rotate=-1)
        with self.assertRaises(TypeError):
            AtomicGrid(OneDGrid(np.arange(3), np.arange(3)), degs=[17], rotate="asdfaf")
        # error of radial grid
        with self.assertRaises(TypeError):
            AtomicGrid(Grid(np.arange(1, 5, 1), np.ones(4)), degs=[2, 3, 4, 5])
        with self.assertRaises(TypeError):
            AtomicGrid(OneDGrid(np.arange(-2, 2, 1), np.ones(4)), degs=[2, 3, 4, 5])
        with self.assertRaises(TypeError):
            rgrid = OneDGrid(np.arange(1, 3, 1), np.ones(2), domain=(-1, 5))
            AtomicGrid(rgrid, degs=[2])
        with self.assertRaises(TypeError):
            rgrid = OneDGrid(np.arange(-1, 1, 1), np.ones(2))
            AtomicGrid(rgrid, degs=[2])
