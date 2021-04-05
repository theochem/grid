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
    _select_grid_type,
    generate_lebedev_grid,
    lebedev_degrees,
    lebedev_npoints,
    size_to_degree,
)

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal


class TestLebedev(TestCase):
    """Lebedev test class."""

    def test_consistency(self):
        """Consistency tests from old grid."""
        for i in range(len(lebedev_npoints)):
            assert_equal(_select_grid_type(degree=lebedev_degrees[i])[1], lebedev_npoints[i])

    def test_lebedev_laikov_sphere(self):
        """Levedev grid tests from old grid."""
        previous_npoint = 0
        for i in range(1, 132):
            npoint = _select_grid_type(degree=i)[1]
            if npoint > previous_npoint:
                grid = generate_lebedev_grid(size=npoint)
                assert isinstance(grid, AngularGrid)
                assert_allclose(grid.weights.sum(), 1.0 * 4 * np.pi)
                assert_allclose(grid.points[:, 0].sum(), 0, atol=1e-10)
                assert_allclose(grid.points[:, 1].sum(), 0, atol=1e-10)
                assert_allclose(grid.points[:, 2].sum(), 0, atol=1e-10)
                assert_allclose(grid.points[:, 0] @ grid.weights, 0, atol=1e-10)
                assert_allclose(grid.points[:, 1] @ grid.weights, 0, atol=1e-10)
                assert_allclose(grid.points[:, 2] @ grid.weights, 0, atol=1e-10)
            previous_npoint = npoint

    def test_size_to_degree(self):
        """Test size to degree conversion."""
        # first test
        nums = [38, 50, 74, 86, 110, 38, 50, 74]
        degs = size_to_degree(nums)
        ref_degs = [9, 11, 13, 15, 17, 9, 11, 13]
        assert_array_equal(degs, ref_degs)
        # second test
        nums = [6]
        degs = size_to_degree(nums)
        ref_degs = [3]
        assert_array_equal(degs, ref_degs)

    def test_errors_and_warnings(self):
        """Tests for errors and warning."""
        # low level function tests
        with self.assertRaises(ValueError):
            _select_grid_type()
        with self.assertRaises(ValueError):
            _select_grid_type(degree=-1)
        with self.assertRaises(ValueError):
            _select_grid_type(degree=132)
        with self.assertRaises(ValueError):
            _select_grid_type(size=-1)
        with self.assertRaises(ValueError):
            _select_grid_type(size=6000)
        with self.assertWarns(RuntimeWarning):
            _select_grid_type(degree=5, size=10)
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
