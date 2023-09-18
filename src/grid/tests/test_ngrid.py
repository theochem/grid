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