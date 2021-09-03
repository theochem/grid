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

from grid.atomgrid import AtomGrid
from grid.basegrid import Grid, OneDGrid
from grid.lebedev import AngularGrid, LEBEDEV_DEGREES
from grid.onedgrid import GaussLegendre, HortonLinear
from grid.rtransform import BeckeTF, IdentityRTransform, PowerRTransform

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)

from scipy.spatial.transform import Rotation as R


class TestAtomGrid(TestCase):
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
        ag_ob = AtomGrid.from_pruned(
            rgrid, radius=rad, sectors_r=r_sectors, sectors_degree=degs
        )
        assert isinstance(ag_ob, AtomGrid)
        assert len(ag_ob.indices) == 11
        assert ag_ob.l_max == 15
        ag_ob = AtomGrid.from_pruned(
            rgrid, radius=rad, sectors_r=np.array([]), sectors_degree=np.array([6])
        )
        assert isinstance(ag_ob, AtomGrid)
        assert len(ag_ob.indices) == 11
        ag_ob = AtomGrid(rgrid, sizes=[110])
        assert ag_ob.l_max == 17
        assert_array_equal(ag_ob._degs, np.ones(10) * 17)
        assert ag_ob.size == 110 * 10
        # new init AtomGrid
        ag_ob2 = AtomGrid(rgrid, degrees=[17])
        assert ag_ob2.l_max == 17
        assert_array_equal(ag_ob2._degs, np.ones(10) * 17)
        assert ag_ob2.size == 110 * 10
        assert isinstance(ag_ob.rgrid, OneDGrid)
        assert_allclose(ag_ob.rgrid.points, rgrid.points)
        assert_allclose(ag_ob.rgrid.weights, rgrid.weights)

    def test_from_predefined(self):
        """Test grid construction with predefined grid."""
        # test coarse grid
        pts = HortonLinear(20)
        tf = PowerRTransform(7.0879993828935345e-06, 16.05937640019924)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomGrid.from_preset(rad_grid, atnum=1, preset="coarse")
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
        atgrid = AtomGrid.from_preset(rad_grid, atnum=1, preset="medium")
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
        atgrid = AtomGrid.from_preset(rad_grid, atnum=1, preset="fine")
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
        atgrid = AtomGrid.from_preset(rad_grid, atnum=1, preset="veryfine")
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
        atgrid = AtomGrid.from_preset(rad_grid, atnum=1, preset="ultrafine")
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
        atgrid = AtomGrid.from_preset(rad_grid, atnum=1, preset="insane")
        # 6622 points for coarse H atom
        assert_equal(atgrid.size, 6622)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points ** 2, axis=1)) * atgrid.weights),
            5.56832800,
        )

    def test_from_pruned_with_degs_and_size(self):
        """Test different initilize method."""
        radial_pts = np.arange(0.1, 1.1, 0.1)
        radial_wts = np.ones(10) * 0.1
        rgrid = OneDGrid(radial_pts, radial_wts)
        rad = 0.5
        r_sectors = np.array([0.5, 1, 1.5])
        degs = np.array([3, 5, 7, 5])
        size = np.array([6, 14, 26, 14])
        # construct atomic grid with degs
        atgrid1 = AtomGrid.from_pruned(
            rgrid, radius=rad, sectors_r=r_sectors, sectors_degree=degs
        )
        # construct atomic grid with size
        atgrid2 = AtomGrid.from_pruned(
            rgrid, radius=rad, sectors_r=r_sectors, sectors_size=size
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
        atomic_grid_degree = AtomGrid._find_l_for_rad_list(
            rgrid.points, rad * r_sectors, degs
        )
        assert_equal(atomic_grid_degree, [3, 3, 5, 5, 7, 7, 7, 7, 3, 3])

    # def test_preload_unit_sphere_grid(self):
    #     """Test for private method to preload spherical grids."""
    #     degs = [3, 3, 5, 5, 7, 7]
    #     unit_sphere = AtomGrid._preload_unit_sphere_grid(degs)
    #     assert len(unit_sphere) == 3
    #     degs = [3, 4, 5, 6, 7]
    #     unit_sphere2 = AtomGrid._preload_unit_sphere_grid(degs)
    #     assert len(unit_sphere2) == 5
    #     assert_allclose(unit_sphere2[4].points, unit_sphere2[5].points)
    #     assert_allclose(unit_sphere2[6].points, unit_sphere2[7].points)
    #     assert not np.allclose(
    #         unit_sphere2[4].points.shape, unit_sphere2[6].points.shape
    #     )

    def test_generate_atomic_grid(self):
        """Test for generating atomic grid."""
        # setup testing class
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = OneDGrid(rad_pts, rad_wts)
        degs = np.array([3, 5, 7])
        pts, wts, ind = AtomGrid._generate_atomic_grid(rad_grid, degs)
        assert len(pts) == 46
        assert_equal(ind, [0, 6, 20, 46])
        # set tests for slicing grid from atomic grid
        for i in range(3):
            # set each layer of points
            ref_grid = AngularGrid(degree=degs[i])
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
        pts, wts, ind = AtomGrid._generate_atomic_grid(rad_grid, degs)
        ref_pts, ref_wts, ref_ind = AtomGrid._generate_atomic_grid(rad_grid, degs)
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
        atgrid = AtomGrid(rad_grid, degrees=degs)
        # make sure True and 1 is not the same result
        atgrid1 = AtomGrid(rad_grid, degrees=degs, rotate=True)
        atgrid2 = AtomGrid(rad_grid, degrees=degs, rotate=1)
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
        atgrid = AtomGrid(rad_grid, degrees=degs)
        assert atgrid.n_shells == 3
        # grep shell with r^2
        for i in range(atgrid.n_shells):
            sh_grid = atgrid.get_shell_grid(i)
            assert isinstance(sh_grid, AngularGrid)
            ref_grid = AngularGrid(degree=degs[i])
            assert np.allclose(sh_grid.points, ref_grid.points * rad_pts[i])
            assert np.allclose(
                sh_grid.weights, ref_grid.weights * rad_wts[i] * rad_pts[i] ** 2
            )
        # grep shell without r^2
        for i in range(atgrid.n_shells):
            sh_grid = atgrid.get_shell_grid(i, r_sq=False)
            assert isinstance(sh_grid, AngularGrid)
            ref_grid = AngularGrid(degree=degs[i])
            assert np.allclose(sh_grid.points, ref_grid.points * rad_pts[i])
            assert np.allclose(sh_grid.weights, ref_grid.weights * rad_wts[i])

    def test_convert_points_to_sph(self):
        """Test convert random points to spherical based on atomic structure."""
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = OneDGrid(rad_pts, rad_wts)
        center = np.random.rand(3)
        atgrid = AtomGrid(rad_grid, degrees=[7], center=center)
        points = np.random.rand(100, 3)
        calc_sph = atgrid.convert_cart_to_sph(points)
        # compute ref result
        ref_coor = points - center
        # radius
        r = np.linalg.norm(ref_coor, axis=-1)
        # azimuthal
        theta = np.arctan2(ref_coor[:, 1], ref_coor[:, 0])
        # polar
        phi = np.arccos(ref_coor[:, 2] / r)
        assert_allclose(np.stack([r, theta, phi]).T, calc_sph)
        assert_equal(calc_sph.shape, (100, 3))

        # test single point
        point = np.random.rand(3)
        calc_sph = atgrid.convert_cart_to_sph(point)
        ref_coor = point - center
        r = np.linalg.norm(ref_coor)
        theta = np.arctan2(ref_coor[1], ref_coor[0])
        phi = np.arccos(ref_coor[2] / r)
        assert_allclose(np.array([r, theta, phi]).reshape(-1, 3), calc_sph)

    def test_spherical_complete(self):
        """Test atomitc grid consistence for spherical integral."""
        num_pts = len(LEBEDEV_DEGREES)
        pts = HortonLinear(num_pts)
        for _ in range(10):
            start = np.random.rand() * 1e-5
            end = np.random.rand() * 10 + 10
            tf = PowerRTransform(start, end)
            rad_grid = tf.transform_1d_grid(pts)
            atgrid = AtomGrid(rad_grid, degrees=list(LEBEDEV_DEGREES.keys()))
            values = np.random.rand(len(LEBEDEV_DEGREES))
            pt_val = np.zeros(atgrid.size)
            for index, value in enumerate(values):
                pt_val[atgrid._indices[index] : atgrid._indices[index + 1]] = value
                rad_int_val = (
                    value
                    * rad_grid.weights[index]
                    * 4
                    * np.pi
                    * rad_grid.points[index] ** 2
                )
                atgrid_int_val = np.sum(
                    pt_val[atgrid._indices[index] : atgrid._indices[index + 1]]
                    * atgrid.weights[
                        atgrid._indices[index] : atgrid._indices[index + 1]
                    ]
                )
                assert_almost_equal(rad_int_val, atgrid_int_val)
            ref_int_at = atgrid.integrate(pt_val)
            ref_int_rad = rad_grid.integrate(4 * np.pi * rad_grid.points ** 2 * values)
            assert_almost_equal(ref_int_at, ref_int_rad)

    # spherical harmonics and related methods tests
    def helper_func_gauss(self, points):
        """Compute gauss function value for test interpolation."""
        x, y, z = points.T
        return np.exp(-(x ** 2)) * np.exp(-(y ** 2)) * np.exp(-(z ** 2))

    def helper_func_power(self, points):
        """Compute function value for test interpolation."""
        return 2 * points[:, 0] ** 2 + 3 * points[:, 1] ** 2 + 4 * points[:, 2] ** 2

    def helper_func_power_deriv(self, points):
        """Compute function derivative for test derivation."""
        r = np.linalg.norm(points, axis=1)
        dxf = 4 * points[:, 0] * points[:, 0] / r
        dyf = 6 * points[:, 1] * points[:, 1] / r
        dzf = 8 * points[:, 2] * points[:, 2] / r
        return dxf + dyf + dzf

    def test_generate_spherical(self):
        """Test generated real spherical harmonics values."""
        atgrid = AngularGrid(degree=7)
        pts = atgrid.points
        wts = atgrid.weights
        r = np.linalg.norm(pts, axis=1)
        # polar
        phi = np.arccos(pts[:, 2] / r)
        # azimuthal
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        # generate spherical harmonics
        sph_h = AtomGrid._generate_real_sph_harm(3, theta, phi)  # l_max = 3
        assert sph_h.shape == (16, 26)
        for _ in range(100):
            n1, n2 = np.random.randint(0, 16, 2)
            re = sum(sph_h[n1] * sph_h[n2] * wts)
            if n1 != n2:
                print(n1, n2, re)
                assert_almost_equal(re, 0)
            else:
                print(n1, n2, re)
                assert_almost_equal(re, 1)

        for i in range(10):
            sph_h = AtomGrid._generate_real_sph_harm(i, theta, phi)
            assert sph_h.shape == ((i + 1) ** 2, 26)

    def test_value_fitting(self):
        """Test spline projection the same as spherical harmonics."""
        odg = OneDGrid(np.arange(10) + 1, np.ones(10), (0, np.inf))
        rad = IdentityRTransform().transform_1d_grid(odg)
        atgrid = AtomGrid.from_pruned(rad, 1, sectors_r=[], sectors_degree=[7])
        values = self.helper_func_power(atgrid.points)
        spls = atgrid.fit_values(values)
        assert len(spls) == 16

        for shell in range(1, 11):
            sh_grid = atgrid.get_shell_grid(shell - 1, r_sq=False)
            r = np.linalg.norm(sh_grid._points, axis=1)
            theta = np.arctan2(sh_grid._points[:, 1], sh_grid._points[:, 0])
            phi = np.arccos(sh_grid._points[:, 2] / r)
            l_max = atgrid.l_max // 2
            r_sph = atgrid._generate_real_sph_harm(l_max, theta, phi)
            r_sph_proj = np.sum(
                r_sph * self.helper_func_power(sh_grid.points) * sh_grid.weights,
                axis=-1,
            )
            assert_allclose(r_sph_proj, [spl(shell) for spl in spls], atol=1e-10)

    def test_cubicspline_and_interp_gauss(self):
        """Test cubicspline interpolation values."""
        oned = GaussLegendre(30)
        btf = BeckeTF(0.0001, 1.5)
        rad = btf.transform_1d_grid(oned)
        atgrid = AtomGrid.from_pruned(rad, 1, sectors_r=[], sectors_degree=[7])
        value_array = self.helper_func_gauss(atgrid.points)
        # random test points on gauss function
        for _ in range(20):
            r = np.random.rand(1)[0] * 2
            theta = np.random.rand(10)
            phi = np.random.rand(10)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            inters = atgrid.interpolate(np.array((x, y, z)).T, value_array)
            assert_allclose(
                self.helper_func_gauss(np.array([x, y, z]).T), inters, atol=1e-4
            )

    def test_cubicspline_and_interp_mol(self):
        """Test cubicspline interpolation values."""
        odg = OneDGrid(np.arange(10) + 1, np.ones(10), (0, np.inf))
        rad = IdentityRTransform().transform_1d_grid(odg)
        atgrid = AtomGrid.from_pruned(rad, 1, sectors_r=[], sectors_degree=[7])
        values = self.helper_func_power(atgrid.points)
        # spls = atgrid.fit_values(values)
        for i in range(10):
            interp = atgrid.interpolate(
                atgrid.points[atgrid.indices[i] : atgrid.indices[i + 1]], values
            )
            # same result from points and interpolation
            assert_allclose(interp, values[atgrid.indices[i] : atgrid.indices[i + 1]])

    def test_cubicspline_and_interp(self):
        """Test cubicspline interpolation values."""
        odg = OneDGrid(np.arange(10) + 1, np.ones(10), (0, np.inf))
        rad_grid = IdentityRTransform().transform_1d_grid(odg)
        for _ in range(10):
            degree = np.random.randint(5, 20)
            atgrid = AtomGrid.from_pruned(
                rad_grid, 1, sectors_r=[], sectors_degree=[degree]
            )
            values = self.helper_func_power(atgrid.points)
            # spls = atgrid.fit_values(values)

            for i in range(10):
                interp = atgrid.interpolate(
                    atgrid.points[atgrid.indices[i] : atgrid.indices[i + 1]], values
                )
                # same result from points and interpolation
                assert_allclose(
                    interp, values[atgrid.indices[i] : atgrid.indices[i + 1]]
                )

            # test random x, y, z
            for _ in range(10):
                xyz = np.random.rand(10, 3) * np.random.uniform(1, 6)
                # xyz /= np.linalg.norm(xyz, axis=-1)[:, None]
                # rad = np.random.normal() * np.random.randint(1, 11)
                # xyz *= rad
                ref_value = self.helper_func_power(xyz)

                interp = atgrid.interpolate(xyz, values)
                assert_allclose(interp, ref_value)

    def test_cubicspline_and_deriv(self):
        """Test spline for derivation."""
        odg = OneDGrid(np.arange(10) + 1, np.ones(10), (0, np.inf))
        rad = IdentityRTransform().transform_1d_grid(odg)
        for _ in range(10):
            degree = np.random.randint(5, 20)
            atgrid = AtomGrid.from_pruned(rad, 1, sectors_r=[], sectors_degree=[degree])
            values = self.helper_func_power(atgrid.points)
            # spls = atgrid.fit_values(values)

            for i in range(10):
                interp = atgrid.interpolate(
                    atgrid.points[atgrid.indices[i] : atgrid.indices[i + 1]],
                    values,
                    deriv=1,
                )
                # same result from points and interpolation
                ref_deriv = self.helper_func_power_deriv(
                    atgrid.points[atgrid.indices[i] : atgrid.indices[i + 1]]
                )
                assert_allclose(interp, ref_deriv)

            # test random x, y, z with fd
            for _ in range(10):
                xyz = np.random.rand(10, 3) * np.random.uniform(1, 6)
                ref_value = self.helper_func_power_deriv(xyz)
                interp = atgrid.interpolate(xyz, values, deriv=1)
                assert_allclose(interp, ref_value)

    def test_error_raises(self):
        """Tests for error raises."""
        with self.assertRaises(TypeError):
            AtomGrid.from_pruned(
                np.arange(3), 1.0, sectors_r=np.arange(2), sectors_degree=np.arange(3)
            )
        with self.assertRaises(ValueError):
            AtomGrid.from_pruned(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                sectors_r=np.arange(2),
                sectors_degree=np.arange(0),
            )
        with self.assertRaises(ValueError):
            AtomGrid.from_pruned(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                sectors_r=np.arange(2),
                sectors_degree=np.arange(4),
            )
        with self.assertRaises(ValueError):
            AtomGrid._generate_atomic_grid(
                OneDGrid(np.arange(3), np.arange(3)), np.arange(2)
            )
        with self.assertRaises(ValueError):
            AtomGrid.from_pruned(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                sectors_r=np.array([0.3, 0.5, 0.7]),
                sectors_degree=np.array([3, 5, 7, 5]),
                center=np.array([0, 0, 0, 0]),
            )

        # test preset
        with self.assertRaises(ValueError):
            AtomGrid.from_preset(atnum=1, preset="fine")

        with self.assertRaises(TypeError):
            AtomGrid(OneDGrid(np.arange(3), np.arange(3)), sizes=110)
        with self.assertRaises(TypeError):
            AtomGrid(OneDGrid(np.arange(3), np.arange(3)), degrees=17)
        with self.assertRaises(ValueError):
            AtomGrid(OneDGrid(np.arange(3), np.arange(3)), degrees=[17], rotate=-1)
        with self.assertRaises(TypeError):
            AtomGrid(
                OneDGrid(np.arange(3), np.arange(3)), degrees=[17], rotate="asdfaf"
            )
        # error of radial grid
        with self.assertRaises(TypeError):
            AtomGrid(Grid(np.arange(1, 5, 1), np.ones(4)), degrees=[2, 3, 4, 5])
        with self.assertRaises(TypeError):
            AtomGrid(OneDGrid(np.arange(-2, 2, 1), np.ones(4)), degrees=[2, 3, 4, 5])
        with self.assertRaises(TypeError):
            rgrid = OneDGrid(np.arange(1, 3, 1), np.ones(2), domain=(-1, 5))
            AtomGrid(rgrid, degrees=[2])
        with self.assertRaises(TypeError):
            rgrid = OneDGrid(np.arange(-1, 1, 1), np.ones(2))
            AtomGrid(rgrid, degrees=[2])

        with self.assertRaises(ValueError):
            AtomGrid._generate_real_sph_harm(-1, np.random.rand(10), np.random.rand(10))
        with self.assertRaises(ValueError):
            oned = GaussLegendre(30)
            btf = BeckeTF(0.0001, 1.5)
            rad = btf.transform_1d_grid(oned)
            atgrid = AtomGrid.from_preset(rad, atnum=1, preset="fine")
            atgrid.fit_values(np.random.rand(100))
