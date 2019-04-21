# -*- coding: utf-8 -*-
# OLDGRIDS: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The OLDGRIDS Development Team
#
# This file is part of OLDGRIDS.
#
# OLDGRIDS is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# OLDGRIDS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Test lebedev grid."""
from unittest import TestCase

from grid.basegrid import AngularGrid
from grid.lebedev import _select_grid_type, generate_lebedev_grid, n_degree, n_points

import numpy as np
from numpy.testing import assert_allclose, assert_equal


class TestLebedev(TestCase):
    """Lebedev test class."""

    def test_consistency(self):
        """Consistency tests from old grid."""
        for i in range(len(n_points)):
            assert_equal(_select_grid_type(degree=n_degree[i])[1], n_points[i])

    def test_lebedev_laikov_sphere(self):
        """Levedev grid tests from old grid."""
        previous_npoint = 0
        for i in range(1, 132):
            npoint = _select_grid_type(degree=i)[1]
            if npoint > previous_npoint:
                grid = generate_lebedev_grid(size=npoint)
                assert isinstance(grid, AngularGrid)
                assert_allclose(grid.weights.sum(), 1.0)
                assert_allclose(grid.points[:, 0].sum(), 0, atol=1e-10)
                assert_allclose(grid.points[:, 1].sum(), 0, atol=1e-10)
                assert_allclose(grid.points[:, 2].sum(), 0, atol=1e-10)
                assert_allclose(np.dot(grid.points[:, 0], grid.weights), 0, atol=1e-15)
                assert_allclose(np.dot(grid.points[:, 1], grid.weights), 0, atol=1e-15)
                assert_allclose(np.dot(grid.points[:, 2], grid.weights), 0, atol=1e-15)
            previous_npoint = npoint

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
