# -*- coding: utf-8 -*-
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
"""Transformation tests file."""

from unittest import TestCase

from grid.onedgrid import GaussChebyshev, GaussLegendre
from grid.rtransform import (
    BeckeTF,
    HandyModTF,
    HandyTF,
    InverseTF,
    KnowlesTF,
    LinearTF,
    MultiExpTF,
)

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal


class TestTransform(TestCase):
    """Transform testcase class."""

    def setUp(self):
        """Test setup function."""
        self.array = np.linspace(-0.9, 0.9, 19)
        self.array_2 = np.linspace(-0.9, 0.9, 10)
        self.num = -0.9
        self.num_2 = 0.9

    def _deriv_finite_diff(self, rmin, rmax, tf):
        """General function to test analytic deriv and finite difference."""
        # 1st derivative analytic and finite diff
        array1 = np.sort(np.random.uniform(rmin, rmax, 1))
        array2 = array1 - 1e-6
        # analytic
        a_d1 = tf.deriv(array1)
        # finit diff
        df_d1 = (tf.transform(array1) - tf.transform(array2)) / 1e-6
        assert_allclose(a_d1, df_d1, rtol=1e-5)

        # 2nd derivative analytic and finite diff
        a_d2 = tf.deriv2(array1)
        df_d2 = (tf.deriv(array1) - tf.deriv(array2)) / 1e-6
        assert_allclose(a_d2, df_d2, rtol=1e-4)

        # 3rd derivative analytic and finite diff
        a_d3 = tf.deriv3(array1)
        df_d3 = (tf.deriv2(array1) - tf.deriv2(array2)) / 1e-6
        assert_allclose(a_d3, df_d3, rtol=1e-4)

        # finite diff for num
        for _ in range(50):
            num1 = np.random.uniform(rmin, rmax, 1)[0]
            num2 = num1 - 1e-6
            # d1
            a_d1 = tf.deriv(num1)
            df_d1 = (tf.transform(num1) - tf.transform(num2)) / 1e-6
            assert_allclose(a_d1, df_d1, rtol=1e-5)
            # d2
            a_d2 = tf.deriv2(num1)
            df_d2 = (tf.deriv(num1) - tf.deriv(num2)) / 1e-6
            assert_allclose(a_d2, df_d2, rtol=1e-4)
            # d3
            a_d3 = tf.deriv3(num1)
            df_d3 = (tf.deriv2(num1) - tf.deriv2(num2)) / 1e-6
            assert_allclose(a_d3, df_d3, rtol=1e-4)

    def _transform_and_inverse(self, rmin, rmax, tf):
        """General purpose function for test transform and its inverse."""
        array = np.sort(np.random.uniform(rmin, rmax, 50))
        tf_array = tf.transform(array)
        new_array = tf.inverse(tf_array)
        assert_allclose(new_array, array)

        for _ in range(50):
            num = np.random.uniform(rmin, rmax, 1)[0]
            tf_num = tf.transform(num)
            new_num = tf.inverse(tf_num)
            assert_almost_equal(new_num, num)

    def test_becke_tf(self):
        """Test Becke initializaiton."""
        btf = BeckeTF(0.1, 1.2)
        assert btf.R == 1.2
        assert btf.rmin == 0.1

    def test_becke_parameter_calc(self):
        """Test parameter function."""
        R = BeckeTF.find_parameter(self.array, 0.1, 1.2)
        # R = 1.1
        assert np.isclose(R, 1.1)
        btf = BeckeTF(0.1, R)
        tf_array = btf.transform(self.array)
        assert tf_array[9] == 1.2
        # for even number of grid
        R = BeckeTF.find_parameter(self.array_2, 0.2, 1.3)
        btf_2 = BeckeTF(0.2, R)
        tf_elemt = btf_2.transform(np.array([(self.array_2[4] + self.array_2[5]) / 2]))
        assert_allclose(tf_elemt, 1.3)

    def test_becke_transform(self):
        """Test becke transformation."""
        btf = BeckeTF(0.1, 1.1)
        tf_array = btf.transform(self.array)
        single_v = btf.transform(self.num)
        single_v2 = btf.transform(self.num_2)
        new_array = btf.inverse(tf_array)
        assert_allclose(new_array, self.array)
        assert_allclose(tf_array[0], single_v)
        assert_allclose(tf_array[-1], single_v2)
        # test tf and inverse
        self._transform_and_inverse(-1, 1, btf)

    def test_becke_infinite(self):
        """Test becke transformation when inf generated."""
        inf_array = np.linspace(-1, 1, 21)
        R = BeckeTF.find_parameter(inf_array, 0.1, 1.2)
        btf = BeckeTF(0.1, R, trim_inf=True)
        tf_array = btf.transform(inf_array)
        inv_array = btf.inverse(tf_array)
        assert_allclose(inv_array, inf_array)
        # extra test for neg inf
        # test for number
        result = btf._convert_inf(-np.inf)
        assert_almost_equal(result, -1e16)
        result = btf._convert_inf(np.inf)
        assert_almost_equal(result, 1e16)
        # test for array
        test_array = np.random.rand(5)
        test_array[3] = -np.inf
        result = btf._convert_inf(test_array)
        assert_almost_equal(result[3], -1e16)
        test_array[3] = np.inf
        result = btf._convert_inf(test_array)
        assert_almost_equal(result[3], 1e16)

    def test_becke_deriv(self):
        """Test becke transform derivatives with finite diff."""
        btf = BeckeTF(0.1, 1.1)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-1, 0.90, btf)

    def test_becke_inverse(self):
        """Test inverse transform basic function."""
        btf = BeckeTF(0.1, 1.1)
        inv = InverseTF(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_becke_inverse_deriv(self):
        """Test inverse transformation derivatives with finite diff."""
        btf = BeckeTF(0.1, 1.1)
        inv = InverseTF(btf)
        self._deriv_finite_diff(0, 20, inv)

    def test_becke_inverse_inverse(self):
        """Test inverse of inverse of Becke transformation."""
        btf = BeckeTF(0.1, 1.1)
        inv = InverseTF(btf)
        inv_inv = inv.inverse(inv.transform(self.array))
        assert_allclose(inv_inv, self.array, atol=1e-7)

    def test_becke_integral(self):
        """Test transform integral."""
        oned = GaussLegendre(20)
        btf = BeckeTF(0.00001, 1.0)
        rad = btf.transform_1d_grid(oned)

        def gauss(x):
            return np.exp(-(x ** 2))

        result = rad.integrate(gauss(rad.points))
        ref_result = np.sqrt(np.pi) / 2
        assert_almost_equal(result, ref_result, decimal=5)

        oned = GaussChebyshev(20)
        rad = btf.transform_1d_grid(oned)
        result = rad.integrate(gauss(rad.points))
        assert_almost_equal(result, ref_result, decimal=3)

    def test_linear_transform(self):
        """Test linear transformation."""
        ltf = LinearTF(0.1, 10)
        self._transform_and_inverse(-1, 1, ltf)

    def test_linear_finite_diff(self):
        """Test finite diff for linear derivs."""
        ltf = LinearTF(0.1, 10)
        self._deriv_finite_diff(-1, 1, ltf)

    def test_linear_inverse(self):
        """Test inverse transform and derivs function."""
        ltf = LinearTF(0.1, 10)
        iltf = InverseTF(ltf)
        # transform & inverse
        self._transform_and_inverse(0, 20, iltf)
        # finite diff for derivs
        self._deriv_finite_diff(0, 20, iltf)

    def test_errors_assert(self):
        """Test errors raise."""
        # parameter error
        with self.assertRaises(ValueError):
            BeckeTF.find_parameter(np.arange(5), 0.5, 0.1)
        # transform non array type
        with self.assertRaises(TypeError):
            btf = BeckeTF(0.1, 1.1)
            btf.transform("dafasdf")
        # inverse init error
        with self.assertRaises(TypeError):
            InverseTF(0.5)
        # type error for transform_1d_grid
        with self.assertRaises(TypeError):
            btf = BeckeTF(0.1, 1.1)
            btf.transform_1d_grid(np.arange(3))
        with self.assertRaises(ZeroDivisionError):
            btf = BeckeTF(0.1, 0)
            itf = InverseTF(btf)
            itf._d1(0.5)
        with self.assertRaises(ZeroDivisionError):
            btf = BeckeTF(0.1, 0)
            itf = InverseTF(btf)
            itf._d1(np.array([0.1, 0.2, 0.3]))

    def test_multiexp_tf(self):
        """Test MultiExp initializaiton."""
        btf = MultiExpTF(0.1, 1.2)
        assert btf.R == 1.2
        assert btf.rmin == 0.1

    def test_multiexp_transform(self):
        """Test MultiExp transformation."""
        btf = MultiExpTF(0.1, 1.1)
        tf_array = btf.transform(self.array)
        single_v = btf.transform(self.num)
        single_v2 = btf.transform(self.num_2)
        new_array = btf.inverse(tf_array)
        assert_allclose(new_array, self.array)
        assert_allclose(tf_array[0], single_v)
        assert_allclose(tf_array[-1], single_v2)
        # test tf and inverse
        self._transform_and_inverse(-1, 1, btf)

    def test_multiexp_deriv(self):
        """Test MultiExp transform derivatives with finite diff."""
        btf = MultiExpTF(0.1, 1.1)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-0.9, 0.90, btf)

    def test_multiexp_inverse(self):
        """Test inverse transform basic function."""
        btf = MultiExpTF(0.1, 1.1)
        inv = InverseTF(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_knowles_tf(self):
        """Test Knowles initializaiton."""
        btf = KnowlesTF(0.1, 1.2, 2)
        assert btf.R == 1.2
        assert btf.rmin == 0.1
        assert btf.k == 2

    def test_knowles_transform(self):
        """Test knowles transformation."""
        btf = KnowlesTF(0.1, 1.1, 2)
        tf_array = btf.transform(self.array)
        single_v = btf.transform(self.num)
        single_v2 = btf.transform(self.num_2)
        new_array = btf.inverse(tf_array)
        assert_allclose(new_array, self.array)
        assert_allclose(tf_array[0], single_v)
        assert_allclose(tf_array[-1], single_v2)
        # test tf and inverse
        self._transform_and_inverse(-1, 1, btf)

    def test_knowles_deriv(self):
        """Test Knowles transform derivatives with finite diff."""
        btf = KnowlesTF(0.1, 1.1, 2)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-0.9, 0.90, btf)

    def test_knowles_inverse(self):
        """Test inverse transform basic function."""
        btf = KnowlesTF(0.1, 1.1, 2)
        inv = InverseTF(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_handy_tf(self):
        """Test Handy initializaiton."""
        btf = HandyTF(0.1, 1.2, 2)
        assert btf.R == 1.2
        assert btf.m == 2
        assert btf.rmin == 0.1

    def test_handy_transform(self):
        """Test Handy transformation."""
        btf = HandyTF(0.1, 1.1, 2)
        tf_array = btf.transform(self.array)
        single_v = btf.transform(self.num)
        single_v2 = btf.transform(self.num_2)
        new_array = btf.inverse(tf_array)
        assert_allclose(new_array, self.array)
        assert_allclose(tf_array[0], single_v)
        assert_allclose(tf_array[-1], single_v2)
        # test tf and inverse
        self._transform_and_inverse(-1, 1, btf)

    def test_handy_deriv(self):
        """Test Handy transform derivatives with finite diff."""
        btf = HandyTF(0.1, 1.2, 2)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-0.8, 0.8, btf)

    def test_handy_inverse(self):
        """Test inverse transform basic function."""
        btf = HandyTF(0.1, 1.1, 2)
        inv = InverseTF(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_handymod_tf(self):
        """Test Handy Mod initializaiton."""
        btf = HandyModTF(0.1, 10.0, 2)
        assert btf.m == 2
        assert btf.rmin == 0.1
        assert btf.rmax == 10.0

    def test_handymod_transform(self):
        """Test Handy Mod transformation."""
        btf = HandyModTF(0.1, 10.0, 2)
        tf_array = btf.transform(self.array)
        new_array = btf.inverse(tf_array)
        assert_allclose(new_array, self.array)
        # test tf and inverse
        self._transform_and_inverse(-0.9, 0.9, btf)

    def test_handymod_deriv(self):
        """Test Handy Mod transform derivatives with finite diff."""
        btf = HandyModTF(0.1, 10.0, 2)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-0.9, 0.90, btf)

    def test_handymod_inverse(self):
        """Test inverse transform basic function."""
        btf = HandyModTF(0.1, 10.0, 2)
        inv = InverseTF(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_errors_raises(self):
        """Test errors raise."""
        with self.assertRaises(ValueError):
            KnowlesTF(0.1, 10.0, 0)
        with self.assertRaises(ValueError):
            HandyTF(0.1, 10.0, 0)
        with self.assertRaises(ValueError):
            HandyModTF(0.1, 10.0, 0)
        with self.assertRaises(ValueError):
            HandyModTF(10.0, 1.0, 2)
