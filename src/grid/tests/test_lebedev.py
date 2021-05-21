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

from grid.basegrid import AngularGrid
from grid.lebedev import (
    LEBEDEV_DEGREES,
    LEBEDEV_NPOINTS,
    _get_lebedev_size_and_degree,
    _load_lebedev_grid,
    cache_dict,
    convert_lebedev_sizes_to_degrees,
    generate_lebedev_grid,
)

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal


class TestLebedev(TestCase):
    """Lebedev test class."""

    def test_consistency(self):
        """Consistency tests from old grid."""
        for i in LEBEDEV_NPOINTS:
            assert_equal(
                _get_lebedev_size_and_degree(degree=LEBEDEV_NPOINTS[i]),
                (LEBEDEV_NPOINTS[i], i),
            )
        for j in LEBEDEV_DEGREES:
            assert_equal(
                _get_lebedev_size_and_degree(size=LEBEDEV_DEGREES[j]),
                (j, LEBEDEV_DEGREES[j]),
            )

    def test_lebedev_laikov_sphere(self):
        """Levedev grid tests from old grid."""
        previous_npoint = 0
        for i in range(1, 132):
            npoint = _get_lebedev_size_and_degree(degree=i)[1]
            if npoint > previous_npoint:
                grid = generate_lebedev_grid(size=npoint)
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

    def test_lebedev_cache(self):
        """Test cache behavior of spherical grid."""
        degrees = np.random.randint(1, 100, 50)
        cache_dict.clear()
        for i in degrees:
            generate_lebedev_grid(degree=i, cache=False)
        assert len(cache_dict) == 0

        for i in degrees:
            generate_lebedev_grid(degree=i)
            ref_d = _get_lebedev_size_and_degree(degree=i)[0]
            assert ref_d in cache_dict

    def test_convert_lebedev_sizes_to_degrees(self):
        """Test size to degree conversion."""
        # first test
        nums = [38, 50, 74, 86, 110, 38, 50, 74]
        degs = convert_lebedev_sizes_to_degrees(nums)
        ref_degs = [9, 11, 13, 15, 17, 9, 11, 13]
        assert_array_equal(degs, ref_degs)
        # second test
        nums = [6]
        degs = convert_lebedev_sizes_to_degrees(nums)
        ref_degs = [3]
        assert_array_equal(degs, ref_degs)

    def test_errors_and_warnings(self):
        """Tests for errors and warning."""
        # low level function tests
        with self.assertRaises(ValueError):
            _get_lebedev_size_and_degree()
        with self.assertRaises(ValueError):
            _get_lebedev_size_and_degree(degree=-1)
        with self.assertRaises(ValueError):
            _get_lebedev_size_and_degree(degree=132)
        with self.assertRaises(ValueError):
            _get_lebedev_size_and_degree(size=-1)
        with self.assertRaises(ValueError):
            _get_lebedev_size_and_degree(size=6000)
        with self.assertWarns(RuntimeWarning):
            _get_lebedev_size_and_degree(degree=5, size=10)
        # load lebedev grid npz file
        with self.assertRaises(ValueError):
            _load_lebedev_grid(degree=2, size=6)
        with self.assertRaises(ValueError):
            _load_lebedev_grid(degree=3, size=2)
        # high level function tests
        with self.assertRaises(ValueError):
            generate_lebedev_grid()
        with self.assertRaises(ValueError):
            generate_lebedev_grid(size=6000)
        with self.assertRaises(ValueError):
            generate_lebedev_grid(size=-1)
        with self.assertRaises(ValueError):
            generate_lebedev_grid(degree=132)
        with self.assertRaises(ValueError):
            generate_lebedev_grid(degree=-2)
        with self.assertWarns(RuntimeWarning):
            generate_lebedev_grid(degree=5, size=10)
