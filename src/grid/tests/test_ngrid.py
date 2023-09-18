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
"""Ngrid tests file."""


from unittest import TestCase
from grid.onedgrid import UniformInteger
from grid.rtransform import LinearInfiniteRTransform
from grid.ngrid import Ngrid

import numpy as np
from numpy.testing import assert_allclose


class TestNgrid(TestCase):
    """Ngrid tests class."""

    def setUp(self):
        """Set up the test."""
        # define the number of points
        n = 500
        # create a linear grid with n points
        self.linear_grid = UniformInteger(n)
        # transform its boundaries to 0 and 1
        self.linear_grid = LinearInfiniteRTransform(rmin=1.0e-4, rmax=1.0).transform_1d_grid(
            self.linear_grid
        )
        # create a 3D grid with n equally spaced points between 0 and 1 along each axis
        self.ngrid = Ngrid([self.linear_grid], 3)

    def test_init_raises(self):
        """Assert that the init raises the correct error."""
        # case 1: the grid list is not given
        with self.assertRaises(ValueError):
            Ngrid(grid_list=None)
        # case 2: the grid list is empty
        with self.assertRaises(ValueError):
            Ngrid(grid_list=[])
        # case 3: the grid list is not a list of Grid
        with self.assertRaises(ValueError):
            Ngrid(grid_list=[self.linear_grid, 1], n=None)
        # case 4: n is negative
        with self.assertRaises(ValueError):
            Ngrid(grid_list=[self.linear_grid], n=-1)
        # case 5: n and the grid list have different lengths
        with self.assertRaises(ValueError):
            Ngrid(grid_list=[self.linear_grid] * 3, n=2)

    def test_init(self):
        """Assert that the init works as expected."""
        # case 1: the grid list is given and n is None
        ngrid = Ngrid(grid_list=[self.linear_grid, self.linear_grid, self.linear_grid])
        self.assertEqual(len(ngrid.grid_list), 3)
        self.assertEqual(ngrid.n, None)

        # case 2: the grid list is given (length 1) and n is not None
        ngrid = Ngrid(grid_list=[self.linear_grid], n=3)
        self.assertEqual(len(ngrid.grid_list), 1)
        self.assertEqual(ngrid.n, 3)

    def test_single_grid_integration(self):
        """Assert that the integration works as expected for a single grid."""

        # define a function to integrate (x**2)
        def f(x):
            return x**2

        # define a Ngrid with only one grid
        ngrid = Ngrid(grid_list=[self.linear_grid])
        # integrate it
        result = ngrid.integrate(f)
        # check that the result is correct
        self.assertAlmostEqual(result, 1.0 / 3.0, places=2)

    def test_2_grid_integration(self):
        """Assert that the integration works as expected for two grids."""

        # define a function to integrate (x**2+y**2)
        def f(x, y):
            return x**2 + y**2

        # define a Ngrid with two grids
        ngrid = Ngrid(grid_list=[self.linear_grid], n=2)
        # integrate it
        result = ngrid.integrate(f)
        # check that the result is correct
        self.assertAlmostEqual(result, 2.0 / 3.0, places=2)

    def test_3_grid_integration(self):
        """Assert that the integration works as expected for three grids."""

        # define a function to integrate (x**2+y**2+z**2)
        def f(x, y, z):
            return x * y * z

        # define a Ngrid with three grids
        ngrid = Ngrid(grid_list=[self.linear_grid], n=3)
        # integrate it
        result = ngrid.integrate(f)
        # check that the result is correct
        self.assertAlmostEqual(result, 1.0 / 8.0, places=2)
