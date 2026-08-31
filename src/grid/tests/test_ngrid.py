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
"""MultiDomainGrid tests file."""


from unittest import TestCase
from grid.onedgrid import UniformInteger, GaussLegendre
from grid.rtransform import BeckeRTransform, LinearInfiniteRTransform
from grid.atomgrid import AtomGrid
from grid.ngrid import MultiDomainGrid
import itertools
import pytest

import numpy as np
from numpy.testing import assert_allclose


class TestMultiDomainGrid_linear(TestCase):
    """MultiDomainGrid tests class."""

    def setUp(self):
        """Set up the test."""
        # define the number of points
        n = 40
        # create a linear grid with n points
        self.linear_grid = UniformInteger(n)
        # transform its boundaries to 0 and 1
        self.linear_grid = LinearInfiniteRTransform(rmin=1.0e-4, rmax=1.0).transform_1d_grid(
            self.linear_grid
        )
        # create a 3D grid with n equally spaced points between 0 and 1 along each axis
        self.ngrid = MultiDomainGrid([self.linear_grid], 3)

    def test_init_raises(self):
        """Assert that the init raises the correct error."""
        # case 1: the grid list is not given
        with self.assertRaises(ValueError):
            MultiDomainGrid(grid_list=None)
        # case 2: the grid list is empty
        with self.assertRaises(ValueError):
            MultiDomainGrid(grid_list=[])
        # case 3: the grid list is not a list of Grid
        with self.assertRaises(ValueError):
            MultiDomainGrid(grid_list=[self.linear_grid, 1], num_domains=None)
        # case 4: n is negative
        with self.assertRaises(ValueError):
            MultiDomainGrid(grid_list=[self.linear_grid], num_domains=-1)
        # case 5: n and the grid list have different lengths
        with self.assertRaises(ValueError):
            MultiDomainGrid(grid_list=[self.linear_grid] * 3, num_domains=2)

    def test_size(self):
        """Assert that the size property works as expected."""
        ref_size = np.prod([grid.size for grid in self.ngrid.grid_list])

        # if there is only one grid the same grid is repeated num_domains times and all possible
        # combinations are taken
        if len(self.ngrid.grid_list) == 1:
            ref_size = ref_size**self.ngrid.num_domains
        self.assertEqual(ref_size, self.ngrid.size)

    def test_init(self):
        """Assert that the init works as expected."""
        # case 1: the grid list is given and n is None
        ngrid = MultiDomainGrid(grid_list=[self.linear_grid, self.linear_grid, self.linear_grid])
        self.assertEqual(len(ngrid.grid_list), 3)
        self.assertEqual(ngrid.num_domains, 3)

        # case 2: the grid list is given (length 1) and n is not None
        ngrid = MultiDomainGrid(grid_list=[self.linear_grid], num_domains=3)
        self.assertEqual(len(ngrid.grid_list), 1)
        self.assertEqual(ngrid.num_domains, 3)

    def test_single_grid_integration(self):
        """Assert that the integration works as expected for a single grid."""

        # define a function to integrate (x**2)
        def f(x):
            return x**2

        # define a MultiDomainGrid with only one grid
        ngrid = MultiDomainGrid(grid_list=[self.linear_grid])
        # integrate it
        result = ngrid.integrate(f)

        # reference value
        func_vals = f(self.linear_grid.points)
        ref_result = self.linear_grid.integrate(func_vals)

        self.assertAlmostEqual(ref_result, result, places=6)

    def test_2_grid_integration(self):
        """Assert that the integration works as expected for two grids."""

        # define a function to integrate (x**2+y**2)
        def f(x, y):
            return x**2 + y**2

        # reference points, weights and values (all possible combinations)
        weights = [np.prod(i) for i in itertools.product(self.linear_grid.weights, repeat=2)]
        values = [f(*i) for i in itertools.product(self.linear_grid.points, repeat=2)]
        ref_value = np.sum(np.array(weights) * np.array(values))

        # define a MultiDomainGrid with two grids
        ngrid = MultiDomainGrid(grid_list=[self.linear_grid], num_domains=2)
        # integrate it
        result = ngrid.integrate(f)
        # check that the result is correct
        self.assertAlmostEqual(ref_value, result, places=6)

    def test_2_grid_integration_non_vectorized(self):
        """Assert that the integration works as expected for two grids."""

        # define a function to integrate (x**2+y**2)
        def f(x, y):
            return x**2 + y**2

        # reference points, weights and values (all possible combinations)
        weights = [np.prod(i) for i in itertools.product(self.linear_grid.weights, repeat=2)]
        values = [f(*i) for i in itertools.product(self.linear_grid.points, repeat=2)]
        ref_value = np.sum(np.array(weights) * np.array(values))

        # define a MultiDomainGrid with two grids
        ngrid = MultiDomainGrid(grid_list=[self.linear_grid], num_domains=2)
        # integrate it
        result = ngrid.integrate(f, non_vectorized=True)
        # check that the result is correct
        self.assertAlmostEqual(ref_value, result, places=6)

    def test_3_grid_integration(self):
        """Assert that the integration works as expected for three grids."""

        # define a function to integrate
        def f(x, y, z):
            return x * y * z

        # reference points, weights and values (all possible combinations)
        weights = [np.prod(i) for i in itertools.product(self.linear_grid.weights, repeat=3)]
        values = [f(*i) for i in itertools.product(self.linear_grid.points, repeat=3)]
        ref_value = np.sum(np.array(weights) * np.array(values))

        # define a MultiDomainGrid with two grids
        ngrid = MultiDomainGrid(grid_list=[self.linear_grid], num_domains=3)
        # integrate it
        result = ngrid.integrate(f)
        # check that the result is correct
        self.assertAlmostEqual(ref_value, result, places=6)

    def test_3_grid_integration_non_vectorized(self):
        """Assert that the integration works as expected for three grids."""

        # define a function to integrate
        def f(x, y, z):
            return x * y * z

        # reference points, weights and values (all possible combinations)
        weights = [np.prod(i) for i in itertools.product(self.linear_grid.weights, repeat=3)]
        values = [f(*i) for i in itertools.product(self.linear_grid.points, repeat=3)]
        ref_value = np.sum(np.array(weights) * np.array(values))

        # define a MultiDomainGrid with two grids
        ngrid = MultiDomainGrid(grid_list=[self.linear_grid], num_domains=3)
        # integrate it
        result = ngrid.integrate(f, non_vectorized=True)
        # check that the result is correct
        self.assertAlmostEqual(ref_value, result, places=6)


class TestMultiDomainGrid_atom(TestCase):
    """MultiDomainGrid tests class."""

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
            # axis=1 is needed to broadcast the norm correctly in the 3D case
            r = np.linalg.norm(center - x, axis=-1)
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

        # # define a MultiDomainGrid with only one atom grid
        ngrid = MultiDomainGrid(grid_list=[self.atgrid1])
        ngrid.integrate(g)
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
        def func(x, y):
            return g1(x) * g2(y)

        # define a MultiDomainGrid with two grids
        ngrid = MultiDomainGrid(grid_list=[self.atgrid1, self.atgrid2])
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

        # define a MultiDomainGrid with two grids
        ngrid = MultiDomainGrid(grid_list=[self.atgrid1, self.atgrid2, self.atgrid3])
        # integrate it and compare with reference value
        result = ngrid.integrate(func)
        ref_val = (
            self.gaussian_integral(height1, width1)
            * self.gaussian_integral(height2, width2)
            * self.gaussian_integral(height3, width3)
        )

        # check that the result is correct
        self.assertAlmostEqual(result, ref_val, places=6)


class TestMultiDomainGrid_mixed_dimension_grids(TestCase):
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

        # define a MultiDomainGrid with only one atom grid
        ngrid = MultiDomainGrid(grid_list=[self.onedgrid, self.atgrid1])
        # integrate it
        result = ngrid.integrate(func)
        ref_val = (
            self.gaussian_integral_1d(height_1d, width_1d)
            / 2
            * self.gaussian_integral_3d(height_3d, width_3d)
        )
        # check that the result is correct
        self.assertAlmostEqual(result, ref_val, places=6)
