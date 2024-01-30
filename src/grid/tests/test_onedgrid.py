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
"""Tests one-dimensional grids."""

from unittest import TestCase

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.special import roots_chebyu, roots_legendre

from grid.onedgrid import (
    ClenshawCurtis,
    ExpExp,
    ExpSinh,
    FejerFirst,
    FejerSecond,
    GaussChebyshev,
    GaussChebyshevLobatto,
    GaussChebyshevType2,
    GaussLaguerre,
    GaussLegendre,
    LogExpSinh,
    MidPoint,
    RectangleRuleSineEndPoints,
    Simpson,
    SingleArcSinhExp,
    SingleExp,
    SingleTanh,
    TanhSinh,
    Trapezoidal,
    TrefethenCC,
    TrefethenGC2,
    TrefethenGeneral,
    TrefethenStripCC,
    TrefethenStripGC2,
    TrefethenStripGeneral,
    UniformInteger,
    _derg2,
    _derg3,
    _dergstrip,
    _g2,
    _g3,
    _gstrip,
)


class TestOneDGrid(TestCase):
    """OneDGrid test class."""

    def setUp(self):
        """Test setup function."""
        ...

    def test_gausslaguerre(self):
        """Test Guass Laguerre polynomial grid."""
        points, weights = np.polynomial.laguerre.laggauss(10)
        weights = weights * np.exp(points) * np.power(points, 0)
        grid = GaussLaguerre(10)
        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_gausslegendre(self):
        """Test Guass Legendre polynomial grid."""
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
        grid = UniformInteger(10)
        assert_allclose(grid.points, np.arange(10))
        assert_allclose(grid.weights, np.ones(10))

    def test_gausschebyshev2(self):
        """Test Gauss Chebyshev type 2 polynomial grid."""
        points, weights = roots_chebyu(10)
        grid = GaussChebyshevType2(10)
        weights /= np.sqrt(1 - np.power(points, 2))
        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_gausschebyshevlobatto(self):
        """Test Gauss Chebyshev Lobatto grid."""
        grid = GaussChebyshevLobatto(10)

        idx = np.arange(10)
        weights = np.ones(10)
        idx = (idx * np.pi) / 9

        points = np.cos(idx)
        points = np.sort(points)

        weights *= np.pi / 9
        weights *= np.sqrt(1 - np.power(points, 2))
        weights[0] /= 2
        weights[9] /= 2

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_trapezoidal(self):
        """Test for Trapezoidal rule."""
        grid = Trapezoidal(10)

        idx = np.arange(10)
        points = -1 + (2 * idx / 9)

        weights = 2 * np.ones(10) / 9
        weights[0] /= 2
        weights[9] = weights[0]

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    # def test_rectanglesineendpoints(self):
    #     """Test for rectangle rule for sine series with endpoints."""
    #     grid = RectangleRuleSineEndPoints(10)
    #
    #     idx = np.arange(10) + 1
    #     points = idx / 11
    #
    #     weights = np.zeros(10)
    #
    #     index_m = np.arange(10) + 1
    #
    #     for i in range(0, 10):
    #         elements = np.zeros(10)
    #         elements = np.sin(index_m * np.pi * points[i])
    #         elements *= (1 - np.cos(index_m * np.pi)) / (index_m * np.pi)
    #
    #         weights[i] = (2 / (11)) * np.sum(elements)
    #
    #     points = 2 * points - 1
    #     weights *= 2
    #
    #     assert_allclose(grid.points, points)
    #     assert_allclose(grid.weights, weights)

    # def test_rectanglesine(self):
    #     """Test for rectangle rule for sine series without endpoint."""
    #     grid = RectangleRuleSine(10)
    #
    #     idx = np.arange(10) + 1
    #     points = (2 * idx - 1) / 20
    #
    #     weights = np.zeros(10)
    #
    #     index_m = np.arange(9) + 1
    #
    #     weights = (
    #         (2 / (10 * np.pi**2))
    #         * np.sin(10 * np.pi * points)
    #         * np.sin(10 * np.pi / 2) ** 2
    #     )
    #
    #     for i in range(0, 10):
    #         elements = np.zeros(9)
    #         elements = np.sin(index_m * np.pi * points[i])
    #         elements *= np.sin(index_m * np.pi / 2) ** 2
    #         elements /= index_m
    #         weights[i] += (4 / (10 * np.pi)) * np.sum(elements)
    #
    #     points = 2 * points - 1
    #     weights *= 2
    #
    #     assert_allclose(grid.points, points)
    #     assert_allclose(grid.weights, weights)

    def test_tanhsinh(self):
        """Test for Tanh - Sinh rule."""
        delta = 0.1 * np.pi / np.sqrt(11)
        grid = TanhSinh(11, delta)

        jmin = -5
        points = np.zeros(11)
        weights = np.zeros(11)

        for i in range(0, 11):
            j = jmin + i
            arg = np.pi * np.sinh(j * delta) / 2

            points[i] = np.tanh(arg)

            weights[i] = np.pi * delta * np.cosh(j * delta) * 0.5
            weights[i] /= np.cosh(arg) ** 2

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_errors_raise(self):
        """Test errors raise."""
        with self.assertRaises(ValueError):
            GaussLaguerre(10, -1)
        with self.assertRaises(ValueError):
            GaussLaguerre(0, 1)
        with self.assertRaises(ValueError):
            GaussLegendre(-10)
        with self.assertRaises(ValueError):
            GaussChebyshev(-10)
        with self.assertRaises(ValueError):
            UniformInteger(-10)
        with self.assertRaises(ValueError):
            TanhSinh(10, 1)
        with self.assertRaises(ValueError):
            Simpson(4)
        with self.assertRaises(ValueError):
            GaussChebyshevType2(-10)
        with self.assertRaises(ValueError):
            GaussChebyshevLobatto(-10)
        with self.assertRaises(ValueError):
            Trapezoidal(-10)
        with self.assertRaises(ValueError):
            TanhSinh(-11, 1)
        with self.assertRaises(ValueError):
            Simpson(-11)
        with self.assertRaises(ValueError):
            MidPoint(-10)
        with self.assertRaises(ValueError):
            ClenshawCurtis(-10)
        with self.assertRaises(ValueError):
            FejerFirst(-10)
        with self.assertRaises(ValueError):
            FejerSecond(-10)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                ExpSinh(11, -0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                ExpSinh(-11, 0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                ExpSinh(10, 0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                LogExpSinh(11, -0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                LogExpSinh(-11, 0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                LogExpSinh(10, 0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                ExpExp(11, -0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                ExpExp(-11, 0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                ExpExp(10, 0.1)
        with self.assertRaises(ValueError):
            SingleTanh(11, -0.1)
        with self.assertRaises(ValueError):
            SingleTanh(-11, 0.1)
        with self.assertRaises(ValueError):
            SingleTanh(10, 0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                SingleExp(11, -0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                SingleExp(-11, 0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                SingleExp(10, 0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                SingleArcSinhExp(11, -0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                SingleArcSinhExp(-11, 0.1)
        with self.assertRaises(ValueError):
            with pytest.warns(UserWarning, match="Using this quadrature require *"):
                SingleArcSinhExp(10, 0.1)

    @staticmethod
    def helper_gaussian(x):
        """Compute gauss function for integral between [-1, 1]."""
        # integrate (exp(-x^2)) x=[-1, 1], result = 1.49365
        return np.exp(-(x**2))

    @staticmethod
    def helper_quadratic(x):
        """Compute quadratic function for integral between [-1, 1]."""
        # integrate (-x^2 + 1) x=[-1, 1], result = 1.33333
        return -(x**2) + 1

    def test_oned_integral(self):
        """A simple integral tests for basic oned grid."""
        # create candidate function
        # add more ``quadratures: npoints`` if needed
        candidates_quadratures = {
            GaussChebyshev: 15,
            GaussLegendre: 15,
            GaussChebyshevType2: 25,
            GaussChebyshevLobatto: 25,
            Trapezoidal: 35,
            RectangleRuleSineEndPoints: 375,
            Simpson: 11,
            TanhSinh: 35,
            MidPoint: 25,
            ClenshawCurtis: 10,
            FejerFirst: 10,
            FejerSecond: 10,
            TrefethenCC: 15,
            TrefethenGC2: 30,
            TrefethenStripCC: 20,
            TrefethenStripGC2: 70,
            # SingleTanh: 75,           # TODO: The following grids don't work
            # ExpSinh: 11,              # TODO: The following grids don't work
            # LogExpSinh: 75,           # TODO: The following grids don't work
            # ExpExp: 75,               # TODO: The following grids don't work
            SingleTanh: 75,
            # SingleExp: 75,            # TODO: The following grids don't work
            # SingleArcSinhExp: 75,     # TODO: The following grids don't work
        }
        # loop each pair to create pts instance
        for quadrature, n_points in candidates_quadratures.items():
            grid = quadrature(n_points)
            # compute gauss numpymerical integral value
            f1_value = np.sum(self.helper_gaussian(grid.points) * grid.weights)
            # ref value
            ref_value = 1.49365
            assert_almost_equal(f1_value, ref_value, decimal=3)

            # compute quadratic integral value
            f2_value = np.sum(self.helper_quadratic(grid.points) * grid.weights)
            # ref value
            ref_value = 1.33333
            assert_almost_equal(f2_value, ref_value, decimal=3)
            print(f"{quadrature.__name__} passed the tests.")

    def test_Simpson(self):
        """Test for Simpson rule."""
        grid = Simpson(11)
        idx = np.arange(11)
        points = -1 + (2 * idx / 10)
        weights = 2 * np.ones(11) / 30

        for i in range(1, 10):
            if i % 2 == 0:
                weights[i] *= 2
            else:
                weights[i] *= 4
        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_MidPoint(self):
        """Test for midpoint rule."""
        grid = MidPoint(10)
        points = np.zeros(10)
        weights = np.ones(10)

        idx = np.arange(10)

        weights *= 2 / 10
        points = -1 + (2 * idx + 1) / 10

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_ClenshawCurtis(self):
        """Test for ClenshawCurtis."""
        grid = ClenshawCurtis(10)
        points = np.zeros(10)
        weights = 2 * np.ones(10)

        theta = np.zeros(10)

        for i in range(0, 10):
            theta[i] = (9 - i) * np.pi / 9

        points = np.cos(theta)
        weights = np.zeros(10)

        jmed = 9 // 2

        for i in range(0, 10):
            weights[i] = 1

            for j in range(0, jmed):
                if (2 * (j + 1)) == 9:
                    b = 1
                else:
                    b = 2

                weights[i] = weights[i] - b * np.cos(2 * (j + 1) * theta[i]) / (4 * j * (j + 2) + 3)

        for i in range(1, 9):
            weights[i] = 2 * weights[i] / 9

        weights[0] /= 9
        weights[9] /= 9

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_FejerFirst(self):
        """Test for Fejer first rule."""
        grid = FejerFirst(10)

        theta = np.pi * (2 * np.arange(10) + 1) / 20

        points = np.cos(theta)
        weights = np.zeros(10)

        nsum = 10 // 2

        for k in range(0, 10):
            serie = 0

            for m in range(1, nsum):
                serie += np.cos(2 * m * theta[k]) / (4 * m**2 - 1)

            serie = 1 - 2 * serie

            weights[k] = (2 / 10) * serie

        points = points[::-1]
        weights = weights[::-1]

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_FejerSecond(self):
        """Test for Fejer second rule."""
        grid = FejerSecond(10)

        theta = np.pi * (np.arange(10) + 1) / 11

        points = np.cos(theta)
        weights = np.zeros(10)

        nsum = 11 // 2

        for k in range(0, 10):
            serie = 0

            for m in range(1, nsum):
                serie += np.sin((2 * m - 1) * theta[k]) / (2 * m - 1)

            weights[k] = (4 * np.sin(theta[k]) / 11) * serie

        points = points[::-1]
        weights = weights[::-1]

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_AuxiliarTrefethenSausage(self):
        """Test for Auxiliary functions using in Trefethen Sausage."""
        xref = np.array([-1.0, 0.0, 1.0])
        d2ref = np.array([1.5100671140939597, 0.8053691275167785, 1.5100671140939597])
        d3ref = np.array([1.869031249411366, 0.7594793648401741, 1.869031249411366])
        newxg2 = _g2(xref)
        newxg3 = _g3(xref)
        newd2 = _derg2(xref)
        newd3 = _derg3(xref)

        assert_allclose(xref, newxg2)
        assert_allclose(xref, newxg3)
        assert_allclose(d2ref, newd2)
        assert_allclose(d3ref, newd3)

    def test_TrefethenCC_d2(self):
        """Test for Trefethen - Sausage Clenshaw Curtis and parameter d=5."""
        grid = TrefethenCC(10, 5)

        tmp = ClenshawCurtis(10)

        new_points = _g2(tmp.points)
        new_weights = _derg2(tmp.points) * tmp.weights

        assert_allclose(grid.points, new_points)
        assert_allclose(grid.weights, new_weights)

    def test_TrefethenCC_d3(self):
        """Test for Trefethen - Sausage Clenshaw Curtis and parameter d=9."""
        grid = TrefethenCC(10, 9)

        tmp = ClenshawCurtis(10)

        new_points = _g3(tmp.points)
        new_weights = _derg3(tmp.points) * tmp.weights

        assert_allclose(grid.points, new_points)
        assert_allclose(grid.weights, new_weights)

    def test_TrefethenCC_d0(self):
        """Test for Trefethen - Sausage Clenshaw Curtis and parameter d=1."""
        grid = TrefethenCC(10, 1)

        tmp = ClenshawCurtis(10)

        assert_allclose(grid.points, tmp.points)
        assert_allclose(grid.weights, tmp.weights)

    def test_TrefethenGC2_d2(self):
        """Test for Trefethen - Sausage GaussChebyshev2 and parameter d=5."""
        grid = TrefethenGC2(10, 5)

        tmp = GaussChebyshevType2(10)

        new_points = _g2(tmp.points)
        new_weights = _derg2(tmp.points) * tmp.weights

        assert_allclose(grid.points, new_points)
        assert_allclose(grid.weights, new_weights)

    def test_TrefethenGC2_d3(self):
        """Test for Trefethen - Sausage GaussChebyshev2 and parameter d=9."""
        grid = TrefethenGC2(10, 9)

        tmp = GaussChebyshevType2(10)

        new_points = _g3(tmp.points)
        new_weights = _derg3(tmp.points) * tmp.weights

        assert_allclose(grid.points, new_points)
        assert_allclose(grid.weights, new_weights)

    def test_TrefethenGC2_d0(self):
        """Test for Trefethen - Sausage GaussChebyshev2 and parameter d=1."""
        grid = TrefethenGC2(10, 1)

        tmp = GaussChebyshevType2(10)

        assert_allclose(grid.points, tmp.points)
        assert_allclose(grid.weights, tmp.weights)

    def test_TrefethenGeneral_d2(self):
        """Test for Trefethen - Sausage General and parameter d=5."""
        grid = TrefethenGeneral(10, ClenshawCurtis, 5)
        new = TrefethenCC(10, 5)

        assert_allclose(grid.points, new.points)
        assert_allclose(grid.weights, new.weights)

    def test_TrefethenGeneral_d3(self):
        """Test for Trefethen - Sausage General and parameter d=9."""
        grid = TrefethenGeneral(10, ClenshawCurtis, 9)
        new = TrefethenCC(10, 9)

        assert_allclose(grid.points, new.points)
        assert_allclose(grid.weights, new.weights)

    def test_TrefethenGeneral_d0(self):
        """Test for Trefethen - Sausage General and parameter d=1."""
        grid = TrefethenGeneral(10, ClenshawCurtis, 1)
        new = TrefethenCC(10, 1)

        assert_allclose(grid.points, new.points)
        assert_allclose(grid.weights, new.weights)

    def test_AuxiliarTrefethenStrip(self):
        """Test for Auxiliary functions using in Trefethen Strip."""
        xref = np.array([-1.0, 0.0, 1.0])
        dref = np.array([10.7807092, 0.65413403, 10.7807092])
        newx = _gstrip(1.1, xref)
        newd = _dergstrip(1.1, xref)

        assert_allclose(xref, newx)
        assert_allclose(dref, newd)

    def test_TrefethenStripCC(self):
        """Test for Trefethen - Strip Clenshaw Curtis."""
        grid = TrefethenStripCC(10, 1.1)
        tmp = ClenshawCurtis(10)

        new_points = _gstrip(1.1, tmp.points)
        new_weights = _dergstrip(1.1, tmp.points) * tmp.weights

        assert_allclose(grid.points, new_points)
        assert_allclose(grid.weights, new_weights)

    def test_TrefethenStripGC2(self):
        """Test for Trefethen - Strip Gauss Chebyshev 2."""
        grid = TrefethenStripGC2(10, 1.1)

        tmp = GaussChebyshevType2(10)

        new_points = _gstrip(1.1, tmp.points)
        new_weights = _dergstrip(1.1, tmp.points) * tmp.weights

        assert_allclose(grid.points, new_points)
        assert_allclose(grid.weights, new_weights)

    def test_TrefethenStripGeneral(self):
        """Test for Trefethen - Strip General."""
        grid = TrefethenStripGeneral(10, ClenshawCurtis, 3)
        new = TrefethenStripCC(10, 3)

        assert_allclose(grid.points, new.points)
        assert_allclose(grid.weights, new.weights)

    def test_ExpSinh(self):
        """Test for ExpSinh rule."""
        with pytest.warns(UserWarning, match="Using this quadrature require *"):
            grid = ExpSinh(11, 0.1)

        k = np.arange(-5, 6)
        points = np.exp(np.pi * np.sinh(k * 0.1) / 2)
        weights = points * np.pi * 0.1 * np.cosh(k * 0.1) / 2

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_LogExpSinh(self):
        """Test for LogExpSinh rule."""
        with pytest.warns(UserWarning, match="Using this quadrature require *"):
            grid = LogExpSinh(11, 0.1)

        k = np.arange(-5, 6)
        points = np.log(np.exp(np.pi * np.sinh(k * 0.1) / 2) + 1)
        weights = np.exp(np.pi * np.sinh(k * 0.1) / 2) * np.pi * 0.1 * np.cosh(k * 0.1) / 2
        weights /= np.exp(np.pi * np.sinh(k * 0.1) / 2) + 1

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_ExpExp(self):
        """Test for ExpExp rule."""
        # Test that Userwarning was raised when using ExpExp
        with pytest.warns(UserWarning, match="Using this quadrature require *"):
            grid = ExpExp(11, 0.1)

        k = np.arange(-5, 6)
        points = np.exp(k * 0.1) * np.exp(-np.exp(-k * 0.1))
        weights = 0.1 * np.exp(-np.exp(-k * 0.1)) * (np.exp(k * 0.1) + 1)

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_SingleTanh(self):
        """Test for singleTanh rule."""
        grid = SingleTanh(11, 0.1)

        k = np.arange(-5, 6)
        points = np.tanh(k * 0.1)
        weights = 0.1 / np.cosh(k * 0.1) ** 2

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_SingleExp(self):
        """Test for Single Exp rule."""
        with pytest.warns(UserWarning, match="Using this quadrature require *"):
            grid = SingleExp(11, 0.1)

        k = np.arange(-5, 6)
        points = np.exp(k * 0.1)
        weights = 0.1 * points

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)

    def test_SingleArcSinhExp(self):
        """Test for SingleArcSinhExp rule."""
        with pytest.warns(UserWarning, match="Using this quadrature require *"):
            grid = SingleArcSinhExp(11, 0.1)

        k = np.arange(-5, 6)
        points = np.arcsinh(np.exp(k * 0.1))
        weights = 0.1 * np.exp(k * 0.1) / np.sqrt(np.exp(2 * 0.1 * k) + 1)

        assert_allclose(grid.points, points)
        assert_allclose(grid.weights, weights)
