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
from grid.onedgrid import UniformInteger, GaussLegendre
from grid.rtransform import BeckeRTransform, LinearInfiniteRTransform
from grid.atomgrid import AtomGrid
from grid.ngrid import Ngrid
import pytest
import itertools

import numpy as np
from numpy.testing import assert_allclose


class TestNgrid_linear(TestCase):
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


class TestNgrid_atom(TestCase):
    """Ngrid tests class."""

    def setUp(self):
        """Set up atomgrid to use in the tests."""
        # construct a radial grid with 150 points
        oned = GaussLegendre(npoints=150)
        rgrid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned)
        self.center1 = np.array([50.0, 0.0, -0.5], float)
        self.center2 = np.array([0.0, 50.0, 0.5], float)
        self.center3 = np.array([0.0, -0.5, 50.0], float)
        self.atgrid1 = AtomGrid(rgrid, degrees=[22], center=self.center1)
        self.atgrid2 = AtomGrid(rgrid, degrees=[22], center=self.center2)
        self.atgrid3 = AtomGrid(rgrid, degrees=[22], center=self.center3)

    def make_gaussian(self, center, height, width):
        """Make a gaussian function.

        Parameters
        ----------
        center : np.ndarray
            The center of the gaussian.
        height : float
            The height of the gaussian.
        width : float
            The width of the gaussian."""

        def f(x):
            r = np.linalg.norm(center - x, axis=1)
            return height * np.exp(-(r**2) / (2 * width**2))

        return f

    def gaussian_integral(self, height, width):
        """Calculate the integral of a gaussian function.

        Parameters
        ----------
        height : float
            The height of the gaussian.
        width : float
            The width of the gaussian."""
        return height * (width * np.sqrt(2 * np.pi)) ** 3

    def test_single_grid_integration(self):
        """Assert that the integration works as expected for a single grid."""

        height, width = 1.0, 2.0

        # define a function to integrate
        g = self.make_gaussian(self.center1, height, width)

        # define a Ngrid with only one atom grid
        ngrid = Ngrid(grid_list=[self.atgrid1])
        # integrate it
        result = ngrid.integrate(g)
        ref_val = self.gaussian_integral(height, width)
        # check that the result is correct
        self.assertAlmostEqual(result, ref_val, places=6)

    def test_2_grid_integration(self):
        """Assert that the integration works as expected for two grids."""

        height1, width1 = 1.0, 2.0
        height2, width2 = 0.7, 1.5

        # define a function to integrate
        g1 = self.make_gaussian(self.center1, height1, width1)
        g2 = self.make_gaussian(self.center2, height2, width2)

        # function is product of two gaussians
        func = lambda x, y: g1(x) * g2(y)

        # define a Ngrid with two grids
        ngrid = Ngrid(grid_list=[self.atgrid1, self.atgrid2])
        # integrate it and compare with reference value
        result = ngrid.integrate(func)
        ref_val = self.gaussian_integral(height1, width1) * self.gaussian_integral(height2, width2)

        # check that the result is correct
        self.assertAlmostEqual(result, ref_val, places=6)

    @pytest.mark.skip(reason="This is to slow.")
    def test_3_grid_integration(self):
        """Assert that the integration works as expected for three grids."""

        height1, width1 = 1.0, 2.0
        height2, width2 = 0.7, 1.5
        height3, width3 = 0.5, 1.0

        # define a function to integrate
        g1 = self.make_gaussian(self.center1, height1, width1)
        g2 = self.make_gaussian(self.center2, height2, width2)
        g3 = self.make_gaussian(self.center3, height3, width3)

        # function is product of two gaussians
        func = lambda x, y, z: g1(x) * g2(y) * g3(z)

        # define a Ngrid with two grids
        ngrid = Ngrid(grid_list=[self.atgrid1, self.atgrid2, self.atgrid3])
        # integrate it and compare with reference value
        result = ngrid.integrate(func)
        ref_val = (
            self.gaussian_integral(height1, width1)
            * self.gaussian_integral(height2, width2)
            * self.gaussian_integral(height3, width3)
        )

        # check that the result is correct
        self.assertAlmostEqual(result, ref_val, places=6)


class TestNgrid_mixed(TestCase):
    """Ngrid tests class."""

    def setUp(self):
        """Set up atomgrid to use in the tests."""
        # construct a radial grid with 150 points
        oned = GaussLegendre(npoints=150)
        rgrid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned)
        # the radial grid for points > 0, integrates half of the gaussian
        self.center1d = 0.0
        self.center3d = np.array([0.0, 50.0, 0.5], float)

        self.onedgrid = rgrid
        self.atgrid1 = AtomGrid(rgrid, degrees=[22], center=self.center3d)

    def make_gaussian_1d(self, center, height, width):
        """Make a gaussian function.

        Parameters
        ----------
        center : float
            The center of the gaussian.
        height : float
            The height of the gaussian.
        width : float
            The width of the gaussian."""

        def f(r):
            return height * np.exp(-(r**2) / (2 * width**2))

        return f

    def make_gaussian_3d(self, center, height, width):
        """Make a gaussian function.

        Parameters
        ----------
        center : np.ndarray
            The center of the gaussian.
        height : float
            The height of the gaussian.
        width : float
            The width of the gaussian."""

        def f(x):
            r = np.linalg.norm(center - x, axis=1)
            return height * np.exp(-(r**2) / (2 * width**2))

        return f

    def gaussian_integral_1d(self, height, width):
        """Calculate the integral of a gaussian function.

        Parameters
        ----------
        height : float
            The height of the gaussian.
        width : float
            The width of the gaussian."""
        return height * (width * np.sqrt(2 * np.pi))

    def gaussian_integral_3d(self, height, width):
        """Calculate the integral of a gaussian function.

        Parameters
        ----------
        height : float
            The height of the gaussian.
        width : float
            The width of the gaussian."""
        return height * (width * np.sqrt(2 * np.pi)) ** 3

    def test_single_mixed_1d_3d(self):
        """Assert that the integration works as expected for a single grid."""

        height_1d, width_1d = 1.0, 2.0
        height_3d, width_3d = 3.1, 1.5

        # define a function to integrate
        g1d = self.make_gaussian_1d(self.center1d, height_1d, width_1d)
        g3d = self.make_gaussian_3d(self.center3d, height_3d, width_3d)

        # define a function to integrate
        g1 = self.make_gaussian_1d(self.center1d, height_1d, width_1d)
        g2 = self.make_gaussian_3d(self.center3d, height_3d, width_3d)

        # function is product of two gaussians
        func = lambda x, y: g1(x) * g2(y)

        # define a Ngrid with only one atom grid
        ngrid = Ngrid(grid_list=[self.onedgrid, self.atgrid1])
        # integrate it
        result = ngrid.integrate(func)
        ref_val = (
            self.gaussian_integral_1d(height_1d, width_1d)
            / 2
            * self.gaussian_integral_3d(height_3d, width_3d)
        )
        # check that the result is correct
        self.assertAlmostEqual(result, ref_val, places=6)
