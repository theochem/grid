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

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)

from grid.angular import (
    LEBEDEV_CACHE,
    LEBEDEV_DEGREES,
    LEBEDEV_NPOINTS,
    SPHERICAL_DEGREES,
    AngularGrid,
)
from grid.utils import generate_real_spherical_harmonics


class TestLebedev(TestCase):
    """Lebedev test class."""

    def test_consistency(self):
        """Consistency tests from old grid."""
        for i in LEBEDEV_NPOINTS:
            assert_equal(
                AngularGrid._get_degree_and_size(
                    degree=LEBEDEV_NPOINTS[i], size=None, method="lebedev"
                ),
                (LEBEDEV_NPOINTS[i], i),
            )
        for j in LEBEDEV_DEGREES:
            assert_equal(
                AngularGrid._get_degree_and_size(
                    degree=None, size=LEBEDEV_DEGREES[j], method="lebedev"
                ),
                (j, LEBEDEV_DEGREES[j]),
            )

    def test_lebedev_cache(self):
        """Test cache behavior of spherical grid."""
        degrees = np.random.randint(1, 100, 50)
        # Add 13 so that the warning is guaranteed to happen so pytest warning works.
        degrees = np.append(degrees, [13])
        LEBEDEV_CACHE.clear()
        with pytest.warns(UserWarning, match="Lebedev weights are negative*"):
            for i in degrees:
                AngularGrid(degree=i, cache=False)
            assert len(LEBEDEV_CACHE) == 0
        with pytest.warns(UserWarning, match="Lebedev weights are negative*"):
            for i in degrees:
                AngularGrid(degree=i)
                ref_d = AngularGrid._get_degree_and_size(degree=i, size=None, method="lebedev")[0]
                assert ref_d in LEBEDEV_CACHE

    def test_convert_lebedev_sizes_to_degrees(self):
        """Test size to degree conversion."""
        # first test
        nums = [38, 50, 74, 86, 110, 38, 50, 74]
        degs = AngularGrid.convert_angular_sizes_to_degrees(nums, method="lebedev")
        ref_degs = [9, 11, 13, 15, 17, 9, 11, 13]
        assert_array_equal(degs, ref_degs)
        # second test
        nums = [6]
        degs = AngularGrid.convert_angular_sizes_to_degrees(nums, method="lebedev")
        ref_degs = [3]
        assert_array_equal(degs, ref_degs)

    def test_errors_and_warnings(self):
        """Tests for errors and warning."""
        # low level function tests
        with self.assertRaises(ValueError):
            AngularGrid._get_degree_and_size(degree=None, size=None, method="gibberish")
        with self.assertRaises(ValueError):
            AngularGrid._get_degree_and_size(degree=None, size=None, method="lebedev")
        with self.assertRaises(ValueError):
            AngularGrid._get_degree_and_size(degree=-1, size=None, method="lebedev")
        with self.assertRaises(ValueError):
            AngularGrid._get_degree_and_size(degree=132, size=None, method="lebedev")
        with self.assertRaises(ValueError):
            AngularGrid._get_degree_and_size(degree=None, size=-1, method="lebedev")
        with self.assertRaises(ValueError):
            AngularGrid._get_degree_and_size(degree=None, size=6000, method="lebedev")
        with self.assertWarns(RuntimeWarning):
            AngularGrid._get_degree_and_size(degree=5, size=10, method="lebedev")
        # load lebedev grid npz file
        with self.assertRaises(ValueError):
            AngularGrid._load_precomputed_angular_grid(degree=2, size=6, method="lebedev")
        with self.assertRaises(ValueError):
            AngularGrid._load_precomputed_angular_grid(degree=3, size=2, method="lebedev")
        # high level function tests
        with self.assertRaises(ValueError):
            AngularGrid(degree=None)
        with self.assertRaises(ValueError):
            AngularGrid(degree=None, size=6000)
        with self.assertRaises(ValueError):
            AngularGrid(degree=None, size=-1)
        with self.assertRaises(ValueError):
            AngularGrid(degree=132)
        with self.assertRaises(ValueError):
            AngularGrid(degree=-2)
        with self.assertWarns(RuntimeWarning):
            AngularGrid(degree=5, size=10)


@pytest.mark.parametrize("degree", [5, 10, 100])
@pytest.mark.parametrize("method", ["lebedev", "spherical"])
def test_integration_of_spherical_harmonic_up_to_degree(degree, method):
    r"""Test integration of spherical harmonic is accurate."""
    grid = AngularGrid(degree=degree, method=method)
    # Concert to spherical coordinates from Cartesian.
    r = np.linalg.norm(grid.points, axis=1)
    phi = np.arccos(grid.points[:, 2] / r)
    theta = np.arctan2(grid.points[:, 1], grid.points[:, 0])
    # Generate All Spherical Harmonics Up To Degree = 10
    #   Returns a three-dimensional array where [order m, degree l, points]
    sph_harm = generate_real_spherical_harmonics(degree, theta, phi)
    for l_deg in range(0, degree):
        for m_ord in range(-l_deg, l_deg):
            sph_harm_one = sph_harm[l_deg**2 : (l_deg + 1) ** 2, :]
            if l_deg == 0 and m_ord == 0:
                actual = np.sqrt(4.0 * np.pi)
                assert_equal(actual, grid.integrate(sph_harm_one[m_ord, :]))
            else:
                assert_almost_equal(0.0, grid.integrate(sph_harm_one[m_ord, :]))


@pytest.mark.parametrize("method", ["lebedev", "spherical"])
def test_integration_of_spherical_harmonic_not_accurate_beyond_degree(method):
    r"""Test integration of spherical harmonic of degree higher than grid is not accurate."""
    grid = AngularGrid(degree=3, method=method)
    r = np.linalg.norm(grid.points, axis=1)
    phi = np.arccos(grid.points[:, 2] / r)
    theta = np.arctan2(grid.points[:, 1], grid.points[:, 0])

    sph_harm = generate_real_spherical_harmonics(l_max=6, theta=theta, phi=phi)
    # Check that l=4,m=0 gives inaccurate results
    assert np.abs(grid.integrate(sph_harm[(4) ** 2, :])) > 1e-8
    # Check that l=6,m=0 gives inaccurate results
    assert np.abs(grid.integrate(sph_harm[(6) ** 2, :])) > 1e-8


@pytest.mark.parametrize("method", ["lebedev", "spherical"])
def test_orthogonality_of_spherical_harmonic_up_to_degree_three(method):
    r"""Test orthogonality of spherical harmonic up to degree 3 is accurate."""
    degree = 3
    grid = AngularGrid(degree=10, method=method)
    # Concert to spherical coordinates from Cartesian.
    r = np.linalg.norm(grid.points, axis=1)
    phi = np.arccos(grid.points[:, 2] / r)
    theta = np.arctan2(grid.points[:, 1], grid.points[:, 0])
    # Generate All Spherical Harmonics Up To Degree = 3
    #   Returns a three dimensional array where [order m, degree l, points]
    sph_harm = generate_real_spherical_harmonics(degree, theta, phi)
    for l_deg in range(0, 4):
        for m_ord in [0] + [x for x in range(1, l_deg + 1)] + [-x for x in range(1, l_deg + 1)]:
            for l2 in range(0, 4):
                for m2 in [0] + [x for x in range(1, l2 + 1)] + [-x for x in range(1, l2 + 1)]:
                    sph_harm_one = sph_harm[l_deg**2 : (l_deg + 1) ** 2, :]
                    sph_harm_two = sph_harm[l2**2 : (l2 + 1) ** 2, :]
                    integral = grid.integrate(sph_harm_one[m_ord, :] * sph_harm_two[m2, :])
                    if l2 != l_deg or m2 != m_ord:
                        assert np.abs(integral) < 1e-8
                    else:
                        assert np.abs(integral - 1.0) < 1e-8


def test_orthogonality_of_spherical_harmonic_at_high_degrees():
    r"""Test orthogonality of spherical harmonic is accurate at very high degrees."""
    degree = 88 * 2
    grid = AngularGrid(degree=degree, method="spherical")
    # Concert to spherical coordinates from Cartesian.
    r = np.linalg.norm(grid.points, axis=1)
    phi = np.arccos(grid.points[:, 2] / r)
    theta = np.arctan2(grid.points[:, 1], grid.points[:, 0])
    half_l = degree // 2
    sph_harm = generate_real_spherical_harmonics(half_l, theta, phi)
    for l_deg in range(half_l - 1, half_l):
        for m_ord in [0] + [j for i in [[i, -i] for i in range(1, l_deg)] for j in i]:
            for l2 in range(half_l - 1, half_l):
                for m2 in [0] + [j for i in [[i, -i] for i in range(1, l2)] for j in i]:
                    sph_harm_one = sph_harm[l_deg**2 : (l_deg + 1) ** 2, :]
                    sph_harm_two = sph_harm[l2**2 : (l2 + 1) ** 2, :]
                    integral = grid.integrate(sph_harm_one[m_ord, :] * sph_harm_two[m2, :])
                    if l2 != l_deg or m2 != m_ord:
                        assert np.abs(integral) < 1e-8
                    else:
                        assert np.abs(integral - 1.0) < 1e-8


def test_that_symmetric_spherical_design_is_symmetric():
    r"""Test the sum of all points on the sphere is zero."""
    for degree in SPHERICAL_DEGREES.keys():
        grid = AngularGrid(degree=degree, method="spherical", cache=False)
        assert np.all(np.abs(np.sum(grid.points, axis=0)) < 1e-8)
