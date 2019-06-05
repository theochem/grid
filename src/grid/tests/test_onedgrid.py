"""Onedgrid tests file."""
from unittest import TestCase

from grid.onedgrid import (
    GaussChebyshev,
    GaussLaguerre,
    GaussLegendre,
    HortonLinear,
    generate_onedgrid,
)

import numpy as np
from numpy.testing import assert_allclose

from scipy.special import roots_legendre


class TestOneDGrid(TestCase):
    """OneDGrid test class."""

    def setUp(self):
        """Test setup function."""
        ...

    def test_generate_ondgrid(self):
        """Place holder tests for generate_onedgrid."""
        generate_onedgrid(0)

    def test_gausslaguerre(self):
        """Test Guass Laguerre polynomial grid."""
        points, weights = np.polynomial.laguerre.laggauss(10)
        weights = weights * np.exp(points) * np.power(points, 0)
        grid = GaussLaguerre(10)
        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_gausslengendre(self):
        """Test Guass Lengendre polynomial grid."""
        points, weights = roots_legendre(10)
        grid = GaussLegendre(10)
        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_gausschebyshev(self):
        """Test Guass Chebyshev polynomial grid."""
        points, weights = np.polynomial.chebyshev.chebgauss(10)
        weights = weights * np.sqrt(1 - np.power(points, 2))
        grid = GaussChebyshev(10)
        assert_allclose(grid.points, np.sort(points))
        assert_allclose(grid.weights, weights)

    def test_horton_linear(self):
        """Test horton linear grids."""
        grid = HortonLinear(10)
        assert_allclose(grid.points, np.arange(10))
        assert_allclose(grid.weights, np.ones(10))

    def test_errors_raise(self):
        """Test errors raise."""
        with self.assertRaises(ValueError):
            GaussLaguerre(10, -1)
