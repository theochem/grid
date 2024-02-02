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

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)
from scipy.spatial.transform import Rotation as R

from grid.angular import LEBEDEV_DEGREES, AngularGrid
from grid.atomgrid import AtomGrid, _get_rgrid_size
from grid.basegrid import Grid, OneDGrid
from grid.onedgrid import GaussLaguerre, GaussLegendre, UniformInteger
from grid.rtransform import BeckeRTransform, IdentityRTransform, PowerRTransform
from grid.utils import generate_real_spherical_harmonics


def test_get_rgrid_size():
    """Test to obtain number of radial poitns for pruned atomic grids"""
    n_rad = _get_rgrid_size("sg_0", atnums=list(range(1, 18)))
    assert n_rad == [23, 23, 23, 23, 25, 23, 23, 23, 23, 23, 26, 26, 26, 26, 26, 26, 26]
    n_rad = _get_rgrid_size("sg_1", atnums=[4, 8, 16, 28, 35, 40, 75])
    assert n_rad == [50, 50, 50, 50, 50, 50, 50]
    n_rad = _get_rgrid_size("sg_2", atnums=[4, 8, 16, 28, 35, 40, 75])
    assert n_rad == [75, 75, 75, 75, 75, 75, 75]
    n_rad = _get_rgrid_size("sg_3", atnums=[4, 8, 16, 28, 35, 40, 75])
    assert n_rad == [99, 99, 99, 99, 99, 99, 99]
    n_rad = _get_rgrid_size("g1", atnums=[6, 14, 32, 50, 82])
    assert n_rad == [40, 45, 55, 60, 65]
    n_rad = _get_rgrid_size("g2", atnums=[6, 14, 32, 50, 82])
    assert n_rad == [45, 50, 60, 65, 70]
    n_rad = _get_rgrid_size("g3", atnums=[6, 14, 32, 50, 82])
    assert n_rad == [55, 60, 70, 75, 80]
    n_rad = _get_rgrid_size("g4", atnums=[6, 14, 32, 50, 82])
    assert n_rad == [60, 65, 75, 80, 85]
    n_rad = _get_rgrid_size("g5", atnums=[6, 14, 32, 50, 82])
    assert n_rad == [65, 70, 80, 85, 90]
    n_rad = _get_rgrid_size("g6", atnums=[6, 14, 32, 50, 82])
    assert n_rad == [75, 80, 90, 95, 100]
    n_rad = _get_rgrid_size("g7", atnums=[6, 14, 32, 50, 82])
    assert n_rad == [85, 90, 100, 105, 110]


class TestAtomGrid:
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
        ag_ob = AtomGrid.from_pruned(rgrid, radius=rad, r_sectors=r_sectors, d_sectors=degs)
        assert isinstance(ag_ob, AtomGrid)
        assert len(ag_ob.indices) == 11
        assert ag_ob.l_max == 15
        ag_ob = AtomGrid.from_pruned(
            rgrid, radius=rad, r_sectors=np.array([]), d_sectors=np.array([6])
        )
        assert isinstance(ag_ob, AtomGrid)
        assert len(ag_ob.indices) == 11
        ag_ob = AtomGrid(rgrid, degrees=None, sizes=[110])
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

    @pytest.mark.filterwarnings("ignore:Lebedev weights are negative which can*")
    def test_from_predefined(self):
        """Test grid construction with predefined grid."""
        # test coarse grid
        pts = UniformInteger(20)
        tf = PowerRTransform(7.0879993828935345e-06, 16.05937640019924)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomGrid.from_preset(atnum=1, preset="coarse", rgrid=rad_grid)
        # 604 points for coarse H atom
        assert_equal(atgrid.size, 616)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points**2, axis=1)) * atgrid.weights),
            5.56840953,
        )

        # test medium grid
        pts = UniformInteger(24)
        tf = PowerRTransform(3.69705074304963e-06, 19.279558946793685)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomGrid.from_preset(atnum=1, preset="medium", rgrid=rad_grid)
        # 928 points for coarse H atom
        assert_equal(atgrid.size, 940)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points**2, axis=1)) * atgrid.weights),
            5.56834559,
        )
        # test fine grid
        pts = UniformInteger(34)
        tf = PowerRTransform(2.577533167224667e-07, 16.276983371222354)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomGrid.from_preset(atnum=1, preset="fine", rgrid=rad_grid)
        # 1984 points for coarse H atom
        assert_equal(atgrid.size, 1984 + 4 * 4)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points**2, axis=1)) * atgrid.weights),
            5.56832800,
        )
        # test fine grid but without default radial grid
        atgrid = AtomGrid.from_preset(atnum=1, preset="fine", rgrid=None)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points**2, axis=1)) * atgrid.weights),
            5.56832800,
        )
        # test veryfine grid
        pts = UniformInteger(41)
        tf = PowerRTransform(1.1774580743206259e-07, 20.140888089596444)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomGrid.from_preset(atnum=1, preset="veryfine", rgrid=rad_grid)
        # 3154 points for coarse H atom
        assert_equal(atgrid.size, 3154 + 4 * 6)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points**2, axis=1)) * atgrid.weights),
            5.56832800,
        )
        # test ultrafine grid
        pts = UniformInteger(49)
        tf = PowerRTransform(4.883104847991021e-08, 21.05456999309752)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomGrid.from_preset(atnum=1, preset="ultrafine", rgrid=rad_grid)
        # 4546 points for coarse H atom
        assert_equal(atgrid.size, 4546 + 4 * 6)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points**2, axis=1)) * atgrid.weights),
            5.56832800,
        )
        # test insane grid
        pts = UniformInteger(59)
        tf = PowerRTransform(1.9221827244049134e-08, 21.413278983919113)
        rad_grid = tf.transform_1d_grid(pts)
        atgrid = AtomGrid.from_preset(atnum=1, preset="insane", rgrid=rad_grid)
        # 6622 points for coarse H atom
        assert_equal(atgrid.size, 6622 + 4 * 7)
        assert_almost_equal(
            np.sum(np.exp(-np.sum(atgrid.points**2, axis=1)) * atgrid.weights),
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
        atgrid1 = AtomGrid.from_pruned(rgrid, radius=rad, r_sectors=r_sectors, d_sectors=degs)
        # construct atomic grid with size
        atgrid2 = AtomGrid.from_pruned(
            rgrid, radius=rad, r_sectors=r_sectors, d_sectors=None, s_sectors=size
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
        atomic_grid_degree = AtomGrid._find_degrees_for_radial_points(
            rgrid.points, rad * r_sectors, degs
        )
        assert_equal(atomic_grid_degree, [3, 3, 5, 5, 7, 7, 7, 7, 3, 3])

    def test_generate_atomic_grid(self):
        """Test for generating atomic grid."""
        # setup testing class
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = OneDGrid(rad_pts, rad_wts)
        degs = np.array([2, 5, 7])
        pts, wts, ind, degrees = AtomGrid._generate_atomic_grid(rad_grid, degs)
        assert len(pts) == 50
        assert_equal(ind, [0, 6, 24, 50])
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
        """Test atomic grid center translation."""
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = OneDGrid(rad_pts, rad_wts)
        degs = np.array([2, 5, 7])
        # origin center
        # randome center
        pts, wts, ind, _ = AtomGrid._generate_atomic_grid(rad_grid, degs)
        ref_pts, ref_wts, ref_ind, _ = AtomGrid._generate_atomic_grid(rad_grid, degs)
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
        atgrid = AtomGrid(rad_grid, degrees=degs, rotate=0)
        # make sure True and 1 is not the same result
        atgrid1 = AtomGrid(rad_grid, degrees=degs, rotate=10)
        atgrid2 = AtomGrid(rad_grid, degrees=degs, rotate=1)
        # test diff points, same weights
        assert not np.allclose(atgrid.points, atgrid1.points)
        assert not np.allclose(atgrid.points, atgrid2.points)
        assert not np.allclose(atgrid1.points, atgrid2.points)
        assert_allclose(atgrid.weights, atgrid1.weights)
        assert_allclose(atgrid.weights, atgrid2.weights)
        assert_allclose(atgrid1.weights, atgrid2.weights)
        # test same integral
        value = np.prod(atgrid.points**2, axis=-1)
        value1 = np.prod(atgrid.points**2, axis=-1)
        value2 = np.prod(atgrid.points**2, axis=-1)
        res = atgrid.integrate(value)
        res1 = atgrid1.integrate(value1)
        res2 = atgrid2.integrate(value2)
        assert_almost_equal(res, res1)
        assert_almost_equal(res1, res2)
        # test rotated shells
        for i in range(0, len(degs)):
            non_rot_shell = atgrid.get_shell_grid(i).points
            rot_shell = atgrid2.get_shell_grid(i).points

            rot_mt = R.random(random_state=atgrid2.rotate + i).as_matrix()
            assert_allclose(rot_shell, non_rot_shell @ rot_mt)

    @pytest.mark.parametrize("method", ["lebedev", "spherical"])
    def test_get_shell_grid(self, method):
        """Test angular grid get from get_shell_grid function."""
        rad_pts = np.array([0.1, 0.5, 1])
        rad_wts = np.array([0.3, 0.4, 0.3])
        rad_grid = OneDGrid(rad_pts, rad_wts)
        degs = [3, 5, 7]
        atgrid = AtomGrid(rad_grid, degrees=degs, method=method)
        assert atgrid.n_shells == 3
        # grep shell with r^2
        for i in range(atgrid.n_shells):
            sh_grid = atgrid.get_shell_grid(i)
            assert isinstance(sh_grid, AngularGrid)
            ref_grid = AngularGrid(degree=degs[i], method=method)
            assert np.allclose(sh_grid.points, ref_grid.points * rad_pts[i])
            assert np.allclose(sh_grid.weights, ref_grid.weights * rad_wts[i] * rad_pts[i] ** 2)
        # grep shell without r^2
        for i in range(atgrid.n_shells):
            sh_grid = atgrid.get_shell_grid(i, r_sq=False)
            assert isinstance(sh_grid, AngularGrid)
            ref_grid = AngularGrid(degree=degs[i], method=method)
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
        calc_sph = atgrid.convert_cartesian_to_spherical(points)
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
        calc_sph = atgrid.convert_cartesian_to_spherical(point)
        ref_coor = point - center
        r = np.linalg.norm(ref_coor)
        theta = np.arctan2(ref_coor[1], ref_coor[0])
        phi = np.arccos(ref_coor[2] / r)
        assert_allclose(np.array([r, theta, phi]).reshape(-1, 3), calc_sph)

    @pytest.mark.filterwarnings("ignore:Lebedev weights are negative which can*")
    @pytest.mark.parametrize("method", ["lebedev", "spherical"])
    def test_spherical_complete(self, method):
        """Test atomic grid consistence for spherical integral."""
        num_pts = len(LEBEDEV_DEGREES)
        pts = UniformInteger(num_pts)
        for _ in range(10):
            start = np.random.rand() * 1e-5
            end = np.random.rand() * 10 + 10
            tf = PowerRTransform(start, end)
            rad_grid = tf.transform_1d_grid(pts)
            atgrid = AtomGrid(
                rad_grid,
                degrees=list(LEBEDEV_DEGREES.keys()),
                method=method,
            )
            values = np.random.rand(len(LEBEDEV_DEGREES))
            pt_val = np.zeros(atgrid.size)
            for index, value in enumerate(values):
                pt_val[atgrid._indices[index] : atgrid._indices[index + 1]] = value
                rad_int_val = (
                    value * rad_grid.weights[index] * 4 * np.pi * rad_grid.points[index] ** 2
                )
                atgrid_int_val = np.sum(
                    pt_val[atgrid._indices[index] : atgrid._indices[index + 1]]
                    * atgrid.weights[atgrid._indices[index] : atgrid._indices[index + 1]]
                )
                assert_almost_equal(rad_int_val, atgrid_int_val)
            ref_int_at = atgrid.integrate(pt_val)
            ref_int_rad = rad_grid.integrate(4 * np.pi * rad_grid.points**2 * values)
            assert_almost_equal(ref_int_at, ref_int_rad)

    # spherical harmonics and related methods tests
    def helper_func_gauss(self, points, center=None):
        """Compute gauss function value for test interpolation."""
        if center is None:
            center = np.array([0.0, 0.0, 0.0])
        x, y, z = (points - center).T
        return np.exp(-(x**2)) * np.exp(-(y**2)) * np.exp(-(z**2))

    def helper_func_power(self, points, deriv=False):
        """Compute function value for test interpolation."""
        if deriv:
            deriv = np.zeros((points.shape[0], 3))
            deriv[:, 0] = 4.0 * points[:, 0]
            deriv[:, 1] = 6.0 * points[:, 1]
            deriv[:, 2] = 8.0 * points[:, 2]
            return deriv
        return 2 * points[:, 0] ** 2 + 3 * points[:, 1] ** 2 + 4 * points[:, 2] ** 2

    def helper_func_power_deriv(self, points):
        """Compute function derivative for test derivation."""
        r = np.linalg.norm(points, axis=1)
        dxf = 4 * points[:, 0] * points[:, 0] / r
        dyf = 6 * points[:, 1] * points[:, 1] / r
        dzf = 8 * points[:, 2] * points[:, 2] / r
        return dxf + dyf + dzf

    @pytest.mark.parametrize("method", ["lebedev", "spherical"])
    def test_integrating_angular_components_spherical(self, method):
        """Test integrating angular components of a spherical harmonics of maximum degree 3."""
        odg = OneDGrid(np.array([0.0, 1e-16, 1e-8, 1e-4, 1e-2]), np.ones(5), (0, np.inf))
        atom_grid = AtomGrid(odg, degrees=[3], method=method)
        spherical = atom_grid.convert_cartesian_to_spherical()
        # Evaluate all spherical harmonics on the atomic grid points (r_i, theta_j, phi_j).
        spherical_harmonics = generate_real_spherical_harmonics(
            3, spherical[:, 1], spherical[:, 2]  # theta, phi points
        )
        # Convert to three-dimensional array (Degrees, Order, Points)
        spherical_array = np.zeros((3, 2 * 3 + 1, len(atom_grid.points)))
        spherical_array[0, 0, :] = spherical_harmonics[0, :]  # (l,m) = (0,0)
        spherical_array[1, 0, :] = spherical_harmonics[1, :]  # = (1, 0)
        spherical_array[1, 1, :] = spherical_harmonics[2, :]  # = (1, 1)
        spherical_array[1, 2, :] = spherical_harmonics[3, :]  # = (1, -1)
        spherical_array[2, 0, :] = spherical_harmonics[4, :]  # = (2, 0)
        spherical_array[2, 1, :] = spherical_harmonics[5, :]  # = (2, 2)
        spherical_array[2, 2, :] = spherical_harmonics[6, :]  # = (2, 1)
        spherical_array[2, 3, :] = spherical_harmonics[7, :]  # = (2, -2)
        spherical_array[2, 4, :] = spherical_harmonics[8, :]  # = (2, -1)

        integral = atom_grid.integrate_angular_coordinates(spherical_array)
        assert integral.shape == (3, 2 * 3 + 1, 5)
        # Assert that all spherical harmonics except when l=0,m=0 are all zero.
        assert_almost_equal(integral[0, 0, :], np.sqrt(4.0 * np.pi))
        assert_almost_equal(integral[0, 1:, :], 0.0)
        assert_almost_equal(integral[1:, :, :], 0.0)

    @pytest.mark.parametrize("numb_rads, degs", [(10, 2), (10, 5), (10, 10)])
    def test_integrating_angular_components_with_gaussian_projected_to_spherical_harmonic(
        self, numb_rads, degs
    ):
        r"""Test integrating angular components of a Gaussian projected to spherical harmonic."""
        odg = OneDGrid(np.linspace(0.0, 1.0, numb_rads), np.ones(numb_rads), (0, np.inf))
        atom_grid = AtomGrid(odg, degrees=[degs])
        func_vals = self.helper_func_gauss(atom_grid.points)
        # Generate spherical harmonic basis
        theta, phi = atom_grid.convert_cartesian_to_spherical().T[1:]
        basis = generate_real_spherical_harmonics(degs // 2, theta, phi)
        # Multiply spherical harmonic basis with the Gaussian function values to project.
        values = np.einsum("ln,n->ln", basis, func_vals)
        # Take the integral of the projection of the Gaussian onto spherical harmonic basis.
        integrals = atom_grid.integrate_angular_coordinates(values)
        # Since the Gaussian is spherical, then this is just the integral of a spherical harmonic
        #  thus whenever l!= 0, we have it is zero everywhere.
        assert_allclose(integrals[1:, :], 0.0, atol=1e-6)
        # Integral of e^(-r^2) * int sin(theta) dtheta dphi / (sqrt(4 pi))
        assert_allclose(
            integrals[0, :], np.exp(-atom_grid.rgrid.points**2.0) * (4.0 * np.pi) ** 0.5
        )

    @pytest.mark.parametrize("numb_rads, degs", [(50, 10)])
    def test_integrating_angular_components_with_offcentered_gaussian(self, numb_rads, degs):
        r"""Test integrating angular components off-centered Gaussian is non-zero due symmetry."""
        # Go from 1e-3, since it is zero
        odg = OneDGrid(np.linspace(1e-3, 1.0, numb_rads), np.ones(numb_rads), (0, np.inf))
        atom_grid = AtomGrid(odg, degrees=[degs])
        # Since it is off-centered, one of the l=1, (p-z orbital) would be non-zero
        func_vals = self.helper_func_gauss(atom_grid.points, center=np.array([0.0, 0.0, 0.1]))
        # Generate spherical harmonic basis
        theta, phi = atom_grid.convert_cartesian_to_spherical().T[1:]
        basis = generate_real_spherical_harmonics(degs // 2, theta, phi)
        # Multiply spherical harmonic basis with the Gaussian function values to project.
        values = np.einsum("ln,n->ln", basis, func_vals)
        # Take the integral of the projection of the Gaussian onto spherical harmonic basis.
        integrals = atom_grid.integrate_angular_coordinates(values)
        # Integral of e^(-r^2) * int sin(\theta) d\theta d\phi / (sqrt(4 pi))
        assert np.all(integrals[1, :] > 1e-5)  # pz-orbital should be non-zero
        assert np.all(integrals[2, :] < 1e-5)  # px, py-orbital should be zero
        assert np.all(integrals[3, :] < 1e-5)

    @pytest.mark.parametrize("method", ["lebedev", "spherical"])
    def test_fitting_spherical_harmonics(self, method):
        r"""Test fitting the radial components of spherical harmonics is just a one, rest zeros."""
        max_degree = 10  # Maximum degree
        rad = OneDGrid(np.linspace(0.0, 1.0, num=10), np.ones(10), (0, np.inf))
        atom_grid = AtomGrid(rad, degrees=[max_degree], method=method)
        max_degree = atom_grid.l_max
        spherical = atom_grid.convert_cartesian_to_spherical()
        # Evaluate all spherical harmonics on the atomic grid points (r_i, theta_j, phi_j).
        spherical_harmonics = generate_real_spherical_harmonics(
            max_degree, spherical[:, 1], spherical[:, 2]  # theta, phi points
        )

        i = 0
        # Go through each spherical harmonic up to max_degree // 2 and check if projection
        # for its radial component is one and the rest are all zeros.
        for l_value in range(0, max_degree // 2):
            for _m in [0] + [x for x in range(1, l_value + 1)] + [x for x in range(-l_value, 0)]:
                spherical_harm = spherical_harmonics[i, :]
                radial_components = atom_grid.radial_component_splines(spherical_harm)
                assert len(radial_components) == (atom_grid.l_max // 2 + 1) ** 2.0

                radial_pts = np.arange(0.0, 1.0, 0.01)
                # Go Through each (l, m)
                for j, radial_comp in enumerate(radial_components):
                    # If the current index is j, then projection of spherical harmonic
                    # onto itself should be all ones, else they are all zero.
                    if i == j:
                        assert_almost_equal(radial_comp(radial_pts), 1.0)
                    else:
                        assert_almost_equal(radial_comp(radial_pts), 0.0, decimal=8)
                i += 1

    @pytest.mark.parametrize("method", ["lebedev", "spherical"])
    def test_fitting_product_of_spherical_harmonic(self, method):
        r"""Test fitting the radial components of r**2 times spherical harmonic."""
        max_degree = 7  # Maximum degree
        rad = OneDGrid(np.linspace(0.0, 1.0, num=10), np.ones(10), (0, np.inf))
        atom_grid = AtomGrid(rad, degrees=[max_degree], method=method)
        max_degree = atom_grid.l_max
        spherical = atom_grid.convert_cartesian_to_spherical()

        spherical_harmonic = generate_real_spherical_harmonics(
            4, spherical[:, 1], spherical[:, 2]  # theta, phi points
        )
        # Test on the function r^2 * Y^1_3
        func_vals = spherical[:, 0] ** 2.0 * spherical_harmonic[(3 + 1) ** 2 + 1, :]
        # Fit radial components
        fit = atom_grid.radial_component_splines(func_vals)
        radial_pts = np.arange(0.0, 1.0, 0.01)
        i = 0
        for l_value in range(0, max_degree // 2):
            for _m in [0] + [x for x in range(1, l_value + 1)] + [x for x in range(-l_value, 0)]:
                if i != (3 + 1) ** 2 + 1:
                    assert_almost_equal(fit[i](radial_pts), 0.0, decimal=8)
                else:
                    # Test that on the right spherical harmonic the function r* Y^1_3 projects
                    # to \rho^{1,3}(r) = r
                    assert_almost_equal(fit[i](radial_pts), radial_pts**2.0)
                i += 1

    def test_radial_component_spherical_harmonic_degree_6(self):
        r"""Test fitting the radial components of spherical harmonic of l=6,m=0."""
        max_degree = 6 * 2 + 2  # Maximum degree
        rad = OneDGrid(np.linspace(0.0, 1.0, num=10), np.ones(10), (0, np.inf))
        atom_grid = AtomGrid.from_pruned(rad, 1, r_sectors=[], d_sectors=[max_degree])
        max_degree = atom_grid.l_max
        spherical = atom_grid.convert_cartesian_to_spherical()

        def func(sph_points):
            # Spherical harmonic of degree 6 and order 0
            r, phi, theta = sph_points.T
            return (
                np.sqrt(2.0)
                * np.sqrt(13)
                / (np.sqrt(np.pi) * 32)
                * (
                    231 * np.cos(theta) ** 6.0
                    - 315 * np.cos(theta) ** 4.0
                    + 105 * np.cos(theta) ** 2.0
                    - 5.0
                )
            )

        # Test on the function Y^0_6
        func_vals = func(spherical)
        # Fit radial components
        fit = atom_grid.radial_component_splines(func_vals)
        radial_pts = np.arange(0.0, 1.0, 0.01)
        counter = 0
        for deg in range(0, max_degree // 2):
            for _ in [0] + [x for x in range(1, deg + 1)] + [x for x in range(-deg, 0)]:
                if counter == 36:
                    # Test that on the right spherical harmonic the function r* Y^1_3 projects
                    # to \rho^{1,3}(r) = r
                    assert_almost_equal(fit[counter](radial_pts), np.sqrt(2.0))
                else:
                    assert_almost_equal(fit[counter](radial_pts), 0.0, decimal=8)

                # Test the derivatives are all zero
                assert_almost_equal(fit[counter](radial_pts, 1), 0.0)
                assert_almost_equal(fit[counter](radial_pts, 2), 0.0)
                counter += 1

    @pytest.mark.parametrize("method", ["lebedev", "spherical"])
    def test_value_fitting(self, method):
        """Test spline projection the same as spherical harmonics."""
        odg = OneDGrid(np.arange(10) + 1, np.ones(10), (0, np.inf))
        rad = IdentityRTransform().transform_1d_grid(odg)
        atgrid = AtomGrid.from_pruned(rad, 1, r_sectors=[], d_sectors=[7], method=method)
        values = self.helper_func_power(atgrid.points)
        spls = atgrid.radial_component_splines(values)
        assert len(spls) == 16

        for shell in range(1, 11):
            sh_grid = atgrid.get_shell_grid(shell - 1, r_sq=False)
            spherical_pts = atgrid.convert_cartesian_to_spherical(sh_grid.points)
            _, theta, phi = spherical_pts.T
            l_max = atgrid.l_max // 2
            r_sph = generate_real_spherical_harmonics(l_max, theta, phi)
            r_sph_proj = np.sum(
                r_sph * self.helper_func_power(sh_grid.points) * sh_grid.weights,
                axis=-1,
            )
            assert_allclose(r_sph_proj, [spl(shell) for spl in spls], atol=1e-10)

    @pytest.mark.parametrize(
        "centers",
        [
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1e-3, 0.0, 0.0]]),  # Off-centered from AtomGrid
            np.array([[1e-3, 0.0, -1e-3]]),  # Off-centered from AtomGrid
        ],
    )
    def test_interpolation_of_gaussian(self, centers):
        """Test cubicspline interpolation values on a Gaussian."""
        oned = GaussLegendre(350)
        btf = BeckeRTransform(0.0001, 1.5)
        rad = btf.transform_1d_grid(oned)
        atgrid = AtomGrid(rad, degrees=[31], method="lebedev")
        value_array = self.helper_func_gauss(atgrid.points)
        # random test points on gauss function
        size = 1000
        r = np.random.uniform(0.01, np.max(rad.points), size=size)
        theta = np.random.uniform(0.0, np.pi, size=size)
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=size)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        input_points = np.array((x, y, z)).T
        interfunc = atgrid.interpolate(value_array)
        assert_allclose(
            self.helper_func_gauss(input_points, centers), interfunc(input_points), atol=1e-3
        )

    @pytest.mark.parametrize("method", ["lebedev", "spherical"])
    def test_cubicspline_and_interp_mol(self, method):
        """Test cubicspline interpolation values."""
        odg = OneDGrid(np.arange(10) + 1, np.ones(10), (0, np.inf))
        rad = IdentityRTransform().transform_1d_grid(odg)
        atgrid = AtomGrid.from_pruned(rad, 1, r_sectors=[], d_sectors=[7], method=method)
        values = self.helper_func_power(atgrid.points)
        # spls = atgrid.fit_values(values)
        for i in range(10):
            inter_func = atgrid.interpolate(values)
            # same result from points and interpolation
            assert_allclose(
                inter_func(atgrid.points[atgrid.indices[i] : atgrid.indices[i + 1]]),
                values[atgrid.indices[i] : atgrid.indices[i + 1]],
            )

    @pytest.mark.filterwarnings("ignore:Lebedev weights are negative which can*")
    def test_cubicspline_and_interp(self):
        """Test cubicspline interpolation values."""
        odg = OneDGrid(np.arange(10) + 1, np.ones(10), (0, np.inf))
        rad_grid = IdentityRTransform().transform_1d_grid(odg)
        for _ in range(10):
            degree = np.random.randint(5, 20)
            atgrid = AtomGrid.from_pruned(rad_grid, 1, r_sectors=[], d_sectors=[degree])
            values = self.helper_func_power(atgrid.points)
            # spls = atgrid.fit_values(values)

            for i in range(10):
                inter_func = atgrid.interpolate(values)
                interp = inter_func(atgrid.points[atgrid.indices[i] : atgrid.indices[i + 1]])
                # same result from points and interpolation
                assert_allclose(interp, values[atgrid.indices[i] : atgrid.indices[i + 1]])

            # test random x, y, z
            for _ in range(10):
                xyz = np.random.rand(10, 3) * np.random.uniform(1, 6)
                # xyz /= np.linalg.norm(xyz, axis=-1)[:, None]
                # rad = np.random.normal() * np.random.randint(1, 11)
                # xyz *= rad
                ref_value = self.helper_func_power(xyz)

                interp_func = atgrid.interpolate(values)
                assert_allclose(interp_func(xyz), ref_value)

    @pytest.mark.parametrize("method", ["lebedev", "spherical"])
    def test_spherical_average_of_gaussian(self, method):
        r"""Test spherical average of a Gaussian (radial) function is itself and its integral."""

        # construct helper function
        def func(sph_points):
            return np.exp(-sph_points[:, 0] ** 2.0)

        # Construct Radial Grid and atomic grid with spherical harmonics of degree 10
        #   for all points.
        oned_grid = np.arange(0.0, 5.0, 0.001)
        rad = OneDGrid(oned_grid, np.ones(len(oned_grid)), (0, np.inf))
        atgrid = AtomGrid(rad, degrees=[5], method=method)
        spherical_pts = atgrid.convert_cartesian_to_spherical(atgrid.points)
        func_values = func(spherical_pts)

        spherical_avg = atgrid.spherical_average(func_values)

        # Test that the spherical average of a Gaussian is itself
        numb_pts = 1000
        random_rad_pts = np.random.uniform(0.0, 1.5, size=(numb_pts, 3))
        spherical_avg2 = spherical_avg(random_rad_pts[:, 0])
        func_vals = func(random_rad_pts)
        assert_allclose(spherical_avg2, func_vals, atol=1e-4)

        # Test the integral of spherical average is the integral of Gaussian e^(-x^2)e^(-y^2)...
        #   from -infinity to infinity which is equal to pi^(3/2)
        integral = 4.0 * np.pi * np.trapz(y=spherical_avg(oned_grid) * oned_grid**2.0, x=oned_grid)
        actual_integral = np.sqrt(np.pi) ** 3.0
        assert_allclose(actual_integral, integral)

    def test_spherical_average_of_spherical_harmonic(self):
        r"""Test spherical average of spherical harmonic is zero."""

        # construct helper function
        def func(sph_points):
            # Spherical harmonic of order 6 and magnetic 0
            r, phi, theta = sph_points.T
            return (
                np.sqrt(13)
                / (np.sqrt(np.pi) * 32)
                * (
                    231 * np.cos(theta) ** 6.0
                    - 315 * np.cos(theta) ** 4.0
                    + 105 * np.cos(theta) ** 2.0
                    - 5.0
                )
            )

        # Construct Radial Grid and atomic grid with spherical harmonics of degree 10
        #   for all points.
        oned_grid = np.arange(0.0, 5.0, 0.001)
        rad = OneDGrid(oned_grid, np.ones(len(oned_grid)), (0, np.inf))
        atgrid = AtomGrid(rad, degrees=[10])
        spherical_pts = atgrid.convert_cartesian_to_spherical(atgrid.points)
        func_values = func(spherical_pts)

        spherical_avg = atgrid.spherical_average(func_values)

        # Test that the spherical average of a spherical harmonic is zero.
        numb_pts = 1000
        random_rad_pts = np.random.uniform(0.02, np.pi, size=(numb_pts, 3))
        spherical_avg2 = spherical_avg(random_rad_pts[:, 0])
        assert_allclose(spherical_avg2, 0.0, atol=1e-4)

    @pytest.mark.filterwarnings("ignore:Lebedev weights are negative which can*")
    @pytest.mark.parametrize("method", ["lebedev", "spherical"])
    def test_interpolate_and_its_derivatives_on_polynomial(self, method):
        """Test interpolation of derivative of polynomial function."""
        odg = OneDGrid(np.linspace(0.01, 10, num=50), np.ones(50), (0, np.inf))
        rad = IdentityRTransform().transform_1d_grid(odg)
        for _ in range(10):
            degree = np.random.randint(5, 20)
            atgrid = AtomGrid.from_pruned(
                rad,
                1,
                r_sectors=[],
                d_sectors=[degree],
                method=method,
            )
            values = self.helper_func_power(atgrid.points)
            # spls = atgrid.fit_values(values)

            for i in range(10):
                points = atgrid.points[atgrid.indices[i] : atgrid.indices[i + 1]]
                interp_func = atgrid.interpolate(values)
                # same result from points and interpolation
                ref_deriv = self.helper_func_power(points, deriv=1)
                assert_almost_equal(interp_func(points, deriv=1), ref_deriv)

            # test random x, y, z with fd
            for _ in range(10):
                xyz = np.random.rand(10, 3) * np.random.uniform(1, 6)
                ref_value = self.helper_func_power(xyz, deriv=1)
                interp_func = atgrid.interpolate(values)
                interp = interp_func(xyz, deriv=1)
                assert_allclose(interp, ref_value)

            # test random x, y, z with fd
            for _ in range(10):
                xyz = np.random.rand(10, 3) * np.random.uniform(1, 6)
                ref_value = self.helper_func_power_deriv(xyz)
                interp_func = atgrid.interpolate(values)
                interp = interp_func(xyz, deriv=1, only_radial_deriv=True)
                assert_allclose(interp, ref_value)

            with pytest.raises(ValueError):
                # Raise error since it can't do derivative beyond one.
                interp_func(xyz, deriv=3)

    def test_cartesian_moment_integral_with_gaussian_upto_order_1(self):
        r"""Test Cartesian moment integral of Gaussian up to order 1."""
        # The moment integral is computed analytically with wolframalpha in one-dimension.
        # Generate atomic grid.
        oned = GaussLegendre(10000)
        btf = BeckeRTransform(0.0001, 0.1)
        rad = btf.transform_1d_grid(oned)
        atgrid = AtomGrid.from_pruned(rad, 1, r_sectors=[], d_sectors=[7])

        # Create Gaussian function
        func_vals = self.helper_func_gauss(atgrid.points, np.array([0.175, 0.25, 0.15]))

        # Consider two centers.
        orders = 1
        centers = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.3]])
        true = atgrid.moments(orders, centers, func_vals)
        # Test Cartesian order: (0, 0, 0), which is integral e^{-(x - c)^2} in x-dim
        assert_allclose(true[0, 0], np.sqrt(np.pi) ** 3.0, atol=1e-4)
        assert_allclose(true[0, 1], np.sqrt(np.pi) ** 3.0, atol=1e-4)

        # Test Cartesian order: (1, 0, 0), which is integral (x - X_c) e^{-(x-c)^2}
        #  Wolfram: integral (x - c) e^(-(x - d)^2)  = sqrt(pi) (d - c)
        assert_allclose(true[1, 0], np.sqrt(np.pi) ** 3.0 * 0.175, atol=1e-5)
        assert_allclose(true[1, 1], np.sqrt(np.pi) ** 3.0 * (0.175 - 0.1), atol=1e-5)

        # # Test (0, 0, 1)
        assert_allclose(true[3, 0], np.sqrt(np.pi) ** 3.0 * 0.15, atol=1e-3)
        assert_allclose(true[3, 1], np.sqrt(np.pi) ** 3.0 * (0.15 - 0.3), atol=1e-3)

        with assert_raises(TypeError):
            # orders should be integer
            atgrid.moments(np.array([1, 1]), centers, func_vals)
        with assert_raises(ValueError):
            multidim_f = np.array([func_vals, func_vals])
            atgrid.moments(1, centers, multidim_f)  # func_vals should be ndim = 1
            # centers should be ndim =2
            atgrid.moments(1, np.array([1, 1, 1]), func_vals)
            atgrid.moments(1, np.array([[1, 1]]), func_vals)  # centers should be dim=3
            atgrid.moments(0, centers, func_vals, type_mom="pure_radial")  # l>0
            # func_vals too little points
            atgrid.moments(1, centers, np.array([1, 2, 3]))

    def test_pure_moment_integral_with_identity_function(self):
        r"""Test pure moment integral with identify function is mostly all zeros."""
        center = np.array([[0.0, 0.0, 0.0]])
        oned = GaussLaguerre(15)
        atgrid = AtomGrid(oned, degrees=[50])
        func_vals = np.ones(atgrid.points.shape[0])
        true = atgrid.moments(orders=2, centers=center, func_vals=func_vals, type_mom="pure")
        assert_allclose(true[range(1, len(true))], 0.0, atol=1e-3)

    def test_pure_moment_integrals_with_gaussian_upto_order_5(self):
        r"""Test pure multipole moment integral with Gaussian upto order 5."""
        center = np.array([[0.0, 0.5, 1.0]])
        # Obtained this from Horton on atomgrid
        horton_answer = np.array(
            [
                5.56832800,
                -5.56832800,
                8.75535482e-18,
                -2.78416400,
                4.87228700,
                1.96316697e-18,
                4.82231350,
                -1.20557838,
                -1.26575073e-17,
                -3.48020500,
                1.19286145e-16,
                -6.39354483,
                2.69575520,
                5.23442037e-17,
                -4.68564343e-17,
                5.50268726e-01,
                1.52258969,
                -3.26136843e-13,
                7.15349344,
                -4.47463560,
                -2.80536246e-13,
                -1.40807305e-12,
                -1.45587420,
                2.57364630e-01,
                -1.56214452e-13,
                7.39543562e-01,
                1.35462566e-12,
                -6.82363035,
                6.24076062,
                2.67670790e-12,
                5.05401764e-12,
                2.82075627,
                -7.72093891e-01,
                3.45622696e-12,
                -3.70252405e-13,
                -1.22078763e-01,
            ]
        )
        # Generate atomic grid.
        oned = GaussLaguerre(50)
        atgrid = AtomGrid(oned, degrees=[50])
        order = 5
        gaussian = np.exp(-np.linalg.norm(atgrid.points, axis=1) ** 2.0)
        true = atgrid.moments(orders=order, centers=center, func_vals=gaussian, type_mom="pure")
        assert_allclose(np.ravel(true), horton_answer, atol=1e-6)

    def test_pure_radial_moments_of_identity_function_against_pure_moments(self):
        r"""Test pure-radial multipole moments with identity function against pure moments."""
        center = np.array([[0.0, 0.0, 0.0]])
        # Generate atomic grid.
        oned = GaussLaguerre(100)
        atgrid = AtomGrid(oned, degrees=[50])
        order = 3
        ident_func = np.ones(atgrid.points.shape[0])
        true, orders = atgrid.moments(order, center, ident_func, "pure-radial", return_orders=True)

        radial = atgrid.convert_cartesian_to_spherical(atgrid.points, center=center[0])[:, 0]
        # Go through each (n, deg, m)
        for i, (n, deg, ord) in enumerate(orders):
            index = deg**2 + 2 * ord - 1 if ord > 0 else deg**2 - 2 * ord
            # Integrate Y_l^m r^l f(x) where f(x)=r^n
            desired, orders = atgrid.moments(deg, center, radial**n, "pure", return_orders=True)
            assert_allclose(desired[index], true[i], atol=1e-4)

    def test_radial_moments_of_gaussian_against_horton(self):
        r"""Test radial moments of Gausian against theochem/horton."""
        center = np.array([[0.0, 0.5, 0.0]])
        # Generate atomic grid.
        oned = GaussLaguerre(100)
        atgrid = AtomGrid(oned, degrees=[50])
        horton_answer = np.array([5.568328, 6.79414003, 9.744574, 15.78558743])
        order = 3
        gaussian = np.exp(-np.linalg.norm(atgrid.points, axis=1) ** 2.0)
        true = atgrid.moments(orders=order, centers=center, func_vals=gaussian, type_mom="radial")
        assert_allclose(true[:, 0], horton_answer, atol=1e-4)

    def test_pure_radial_moments_of_spherical_harmonics(self):
        r"""Test pure-radial multipole moments with spherical harmonics."""
        center = np.array([[0.0, 0.0, 0.0]])
        # Generate atomic grid.
        oned = GaussLaguerre(20)
        atgrid = AtomGrid(oned, degrees=[50])
        order = 5
        _, theta, phi = atgrid.convert_cartesian_to_spherical().T
        spherical = generate_real_spherical_harmonics(order, theta, phi)

        i_sph = 0
        for l_sph, m_sph in [
            (0, 0),
            (1, 0),
            (1, 1),
            (1, -1),
            (2, 0),
            (2, 1),
            (2, -1),
            (2, 2),
            (2, -2),
        ]:
            true, orders = atgrid.moments(
                order // 2, center, spherical[i_sph], "pure-radial", return_orders=True
            )
            for i_mom, (_n_mom, l_mom, m_mom) in enumerate(orders):
                # If the spherical harmonics match, then the integral over sph_coords is one
                # then we are left with a diverging integral.
                if l_mom == l_sph and m_sph == m_mom:
                    assert true[i_mom] > 1000
                else:
                    # If the spherical  harmonics don't match, then the integral over sph coords
                    # is zero.
                    assert_allclose(true[i_mom], 0.0, atol=1e-5)
            i_sph += 1

    def test_error_raises(self):
        """Tests for error raises."""
        with pytest.raises(TypeError):
            AtomGrid.from_pruned(np.arange(3), 1.0, r_sectors=np.arange(2), d_sectors=np.arange(3))
        with pytest.raises(ValueError):
            AtomGrid.from_pruned(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                r_sectors=np.arange(2),
                d_sectors=np.arange(0),
            )
        with pytest.raises(ValueError):
            AtomGrid.from_pruned(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                r_sectors=np.arange(2),
                d_sectors=np.arange(4),
            )
        with pytest.raises(ValueError):
            AtomGrid._generate_atomic_grid(OneDGrid(np.arange(3), np.arange(3)), np.arange(2))
        with pytest.raises(ValueError):
            AtomGrid.from_pruned(
                OneDGrid(np.arange(3), np.arange(3)),
                radius=1.0,
                r_sectors=np.array([0.3, 0.5, 0.7]),
                d_sectors=np.array([3, 5, 7, 5]),
                center=np.array([0, 0, 0, 0]),
            )

        # test preset
        with pytest.raises(TypeError):
            AtomGrid(OneDGrid(np.arange(3), np.arange(3)), sizes=110)
        with pytest.raises(TypeError):
            AtomGrid(OneDGrid(np.arange(3), np.arange(3)), degrees=17)
        with pytest.raises(ValueError):
            AtomGrid(OneDGrid(np.arange(3), np.arange(3)), degrees=[17], rotate=-1)
        with pytest.raises(TypeError):
            AtomGrid(OneDGrid(np.arange(3), np.arange(3)), degrees=[17], rotate="asdfaf")
        # error of radial grid
        with pytest.raises(TypeError):
            AtomGrid(Grid(np.arange(1, 5, 1), np.ones(4)), degrees=[2, 3, 4, 5])
        with pytest.raises(TypeError):
            AtomGrid(OneDGrid(np.arange(-2, 2, 1), np.ones(4)), degrees=[2, 3, 4, 5])
        with pytest.raises(TypeError):
            rgrid = OneDGrid(np.arange(1, 3, 1), np.ones(2), domain=(-1, 5))
            AtomGrid(rgrid, degrees=[2])
        with pytest.raises(TypeError):
            rgrid = OneDGrid(np.arange(-1, 1, 1), np.ones(2))
            AtomGrid(rgrid, degrees=[2])

        with pytest.raises(ValueError):
            oned = GaussLegendre(30)
            btf = BeckeRTransform(0.0001, 1.5)
            rad = btf.transform_1d_grid(oned)
            atgrid = AtomGrid.from_preset(atnum=1, preset="fine", rgrid=rad)
            atgrid.radial_component_splines(np.random.rand(100))
