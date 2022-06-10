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
"""Test lebedev grid."""


from unittest import TestCase

from grid.lebedev import (
    AngularGrid,
    LEBEDEV_CACHE,
    LEBEDEV_DEGREES,
    LEBEDEV_NPOINTS,
)
from grid.utils import generate_real_spherical_harmonics

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)


class TestLebedev(TestCase):
    """Lebedev test class."""

    def test_consistency(self):
        """Consistency tests from old grid."""
        for i in LEBEDEV_NPOINTS:
            assert_equal(
                AngularGrid._get_lebedev_size_and_degree(degree=LEBEDEV_NPOINTS[i]),
                (LEBEDEV_NPOINTS[i], i),
            )
        for j in LEBEDEV_DEGREES:
            assert_equal(
                AngularGrid._get_lebedev_size_and_degree(size=LEBEDEV_DEGREES[j]),
                (j, LEBEDEV_DEGREES[j]),
            )

    def test_lebedev_laikov_sphere(self):
        """Test the integration of sphere and its points and weights."""
        previous_npoint = 0
        for i in range(1, 132):
            npoint = AngularGrid._get_lebedev_size_and_degree(degree=i)[1]
            if npoint > previous_npoint:
                grid = AngularGrid(size=npoint)
                assert isinstance(grid, AngularGrid)
                assert_allclose(grid.weights.sum(), 1.0 * 4 * np.pi)
                # check surface area (i.e., integral of constant function 1)
                assert_allclose(grid.integrate(np.ones(grid.size)), 4 * np.pi)
                # check integral of x * y * z is zero (i.e., f orbital is orthogonal to s orbital)
                assert_allclose(
                    grid.integrate(np.product(grid.points, axis=1)), 0.0, atol=1.0e-12
                )
                assert_allclose(grid.points[:, 0].sum(), 0, atol=1e-10)
                assert_allclose(grid.points[:, 1].sum(), 0, atol=1e-10)
                assert_allclose(grid.points[:, 2].sum(), 0, atol=1e-10)
                assert_allclose(grid.points[:, 0] @ grid.weights, 0, atol=1e-10)
                assert_allclose(grid.points[:, 1] @ grid.weights, 0, atol=1e-10)
                assert_allclose(grid.points[:, 2] @ grid.weights, 0, atol=1e-10)
            previous_npoint = npoint

    def test_orthogonality_of_spherical_harmonic_up_to_degree_three(self):
        r"""Test orthogonality of spherical harmonic up to degree 3 is accurate."""
        degree = 3
        grid = AngularGrid(degree=10)
        # Concert to spherical coordinates from Cartesian.
        r = np.linalg.norm(grid.points, axis=1)
        phi = np.arccos(grid.points[:, 2] / r)
        theta = np.arctan2(grid.points[:, 1], grid.points[:, 0])
        # Generate All Spherical Harmonics Up To Degree = 3
        #   Returns a three dimensional array where [order m, degree l, points]
        sph_harm = generate_real_spherical_harmonics(degree, theta, phi)
        for l_deg in range(0, 4):
            for m_ord in range(-l_deg, l_deg + 1):
                for l2 in range(0, 4):
                    for m2 in range(-l2, l2 + 1):
                        sph_harm_one = sph_harm[l_deg ** 2 : (l_deg + 1) ** 2, :]
                        sph_harm_two = sph_harm[l2 ** 2 : (l2 + 1) ** 2, :]
                        integral = grid.integrate(
                            sph_harm_one[m_ord, :] * sph_harm_two[m2, :]
                        )
                        if l2 != l_deg or m2 != m_ord:
                            assert np.abs(integral) < 1e-8
                        else:
                            assert np.abs(integral - 1.0) < 1e-8

    def test_integration_of_spherical_harmonic_up_to_degree_three(self):
        r"""Test integration of spherical harmonic up to degree three is accurate."""
        degree = 3
        grid = AngularGrid(degree=100)
        # Concert to spherical coordinates from Cartesian.
        r = np.linalg.norm(grid.points, axis=1)
        phi = np.arccos(grid.points[:, 2] / r)
        theta = np.arctan2(grid.points[:, 1], grid.points[:, 0])
        # Generate All Spherical Harmonics Up To Degree = 3
        #   Returns a three dimensional array where [order m, degree l, points]
        sph_harm = generate_real_spherical_harmonics(degree, theta, phi)
        for l_deg in range(0, 4):
            for m_ord in range(-l_deg, l_deg):
                sph_harm_one = sph_harm[l_deg ** 2 : (l_deg + 1) ** 2, :]
                if l_deg == 0 and m_ord == 0:
                    actual = np.sqrt(4.0 * np.pi)
                    assert_equal(actual, grid.integrate(sph_harm_one[m_ord, :]))
                else:
                    assert_almost_equal(0.0, grid.integrate(sph_harm_one[m_ord, :]))

    def test_integration_of_spherical_harmonic_not_accurate_beyond_degree(self):
        r"""Test integration of spherical harmonic of degree higher than grid is not accurate."""
        grid = AngularGrid(degree=3)
        r = np.linalg.norm(grid.points, axis=1)
        phi = np.arccos(grid.points[:, 2] / r)
        theta = np.arctan2(grid.points[:, 1], grid.points[:, 0])

        sph_harm = generate_real_spherical_harmonics(l_max=6, theta=theta, phi=phi)
        # Check that l=4,m=0 gives inaccurate results
        assert np.abs(grid.integrate(sph_harm[(4) ** 2, :])) > 1e-8
        # Check that l=6,m=0 gives inaccurate results
        assert np.abs(grid.integrate(sph_harm[(6) ** 2, :])) > 1e-8

    def test_lebedev_cache(self):
        """Test cache behavior of spherical grid."""
        degrees = np.random.randint(1, 100, 50)
        LEBEDEV_CACHE.clear()
        for i in degrees:
            AngularGrid(degree=i, cache=False)
        assert len(LEBEDEV_CACHE) == 0

        for i in degrees:
            AngularGrid(degree=i)
            ref_d = AngularGrid._get_lebedev_size_and_degree(degree=i)[0]
            assert ref_d in LEBEDEV_CACHE

    def test_convert_lebedev_sizes_to_degrees(self):
        """Test size to degree conversion."""
        # first test
        nums = [38, 50, 74, 86, 110, 38, 50, 74]
        degs = AngularGrid.convert_lebedev_sizes_to_degrees(nums)
        ref_degs = [9, 11, 13, 15, 17, 9, 11, 13]
        assert_array_equal(degs, ref_degs)
        # second test
        nums = [6]
        degs = AngularGrid.convert_lebedev_sizes_to_degrees(nums)
        ref_degs = [3]
        assert_array_equal(degs, ref_degs)

    def test_errors_and_warnings(self):
        """Tests for errors and warning."""
        # low level function tests
        with self.assertRaises(ValueError):
            AngularGrid._get_lebedev_size_and_degree()
        with self.assertRaises(ValueError):
            AngularGrid._get_lebedev_size_and_degree(degree=-1)
        with self.assertRaises(ValueError):
            AngularGrid._get_lebedev_size_and_degree(degree=132)
        with self.assertRaises(ValueError):
            AngularGrid._get_lebedev_size_and_degree(size=-1)
        with self.assertRaises(ValueError):
            AngularGrid._get_lebedev_size_and_degree(size=6000)
        with self.assertWarns(RuntimeWarning):
            AngularGrid._get_lebedev_size_and_degree(degree=5, size=10)
        # load lebedev grid npz file
        with self.assertRaises(ValueError):
            AngularGrid._load_lebedev_grid(degree=2, size=6)
        with self.assertRaises(ValueError):
            AngularGrid._load_lebedev_grid(degree=3, size=2)
        # high level function tests
        with self.assertRaises(ValueError):
            AngularGrid()
        with self.assertRaises(ValueError):
            AngularGrid(size=6000)
        with self.assertRaises(ValueError):
            AngularGrid(size=-1)
        with self.assertRaises(ValueError):
            AngularGrid(degree=132)
        with self.assertRaises(ValueError):
            AngularGrid(degree=-2)
        with self.assertWarns(RuntimeWarning):
            AngularGrid(degree=5, size=10)
        with self.assertWarns(RuntimeWarning):
            pts = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
            wts = np.array([0.3, 0.4, 0.5])
            AngularGrid(pts, wts, degree=7)
        with self.assertWarns(RuntimeWarning):
            pts = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
            wts = np.array([0.3, 0.4, 0.5])
            AngularGrid(pts, wts, size=14)
