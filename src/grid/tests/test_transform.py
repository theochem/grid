# GRID is a numerical integration library for quantum chemistry.
#
# Copyright (C) 2011-2017 The GRID Development Team
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
# along with this program; if not, see <http:#www.gnu.org/licenses/>
#
# --
"""Transformation tests file."""

from unittest import TestCase

from grid.rtransform import BeckeTF, HandyTF, InverseTF, KnowlesTF, LinearTF, MultiExpTF

import numpy as np


class TestTransform(TestCase):
    """Transform testcase class."""

    def setUp(self):
        """Test setup function."""
        self.array = np.linspace(-0.9, 0.9, 19)
        self.array_2 = np.linspace(-0.9, 0.9, 10)

    def _deriv_finite_diff(self, tf, array1):
        """General function to test analytic deriv and finite difference."""
        # 1st derivative analytic and finite diff
        array2 = array1 - 1e-7
        # analytic
        a_d1 = tf.deriv(array1)
        # finit diff
        df_d1 = (tf.transform(array1) - tf.transform(array2)) / 1e-7
        assert np.allclose(a_d1, df_d1, rtol=1e-5)

        # 2nd derivative analytic and finite diff
        a_d2 = tf.deriv2(array1)
        df_d2 = (tf.deriv(array1) - tf.deriv(array2)) / 1e-7
        assert np.allclose(a_d2, df_d2, rtol=1e-4)

        # 3rd derivative analytic and finite diff
        a_d3 = tf.deriv3(array1)
        df_d3 = (tf.deriv2(array1) - tf.deriv2(array2)) / 1e-7
        assert np.allclose(a_d3, df_d3, rtol=1e-4)

    def test_becke_tf(self):
        """Test Becke initializaiton."""
        btf = BeckeTF(0.1, 1.2)
        assert btf.R == 1.2
        assert btf.r0 == 0.1

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
        assert np.allclose(tf_elemt, 1.3)

    def test_becke_transform(self):
        """Test becke transformation."""
        btf = BeckeTF(0.1, 1.1)
        tf_array = btf.transform(self.array)
        new_array = btf.inverse(tf_array)
        assert np.allclose(new_array, self.array)

    def test_becke_infinite(self):
        """Test becke transformation when inf generated."""
        inf_array = np.linspace(-1, 1, 21)
        R = BeckeTF.find_parameter(inf_array, 0.1, 1.2)
        btf = BeckeTF(0.1, R)
        tf_array = btf.transform(inf_array, trim_inf=True)
        inv_array = btf.inverse(tf_array)
        assert np.allclose(inv_array, inf_array)

    def test_becke_deriv(self):
        """Test becke transform derivatives with finite diff."""
        btf = BeckeTF(0.1, 1.1)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(btf, self.array)

    def test_becke_inverse(self):
        """Test inverse transform basic function."""
        btf = BeckeTF(0.1, 1.1)
        inv = InverseTF(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert np.allclose(new_array, self.array)

    def test_becke_inverse_deriv(self):
        """Test inverse transformation derivatives with finite diff."""
        btf = BeckeTF(0.1, 1.1)
        inv = InverseTF(btf)
        r_array = 2 ** (np.arange(-1, 8, dtype=float))
        self._deriv_finite_diff(inv, r_array)

    def test_becke_inverse_inverse(self):
        """Test inverse of inverse of Becke transformation."""
        btf = BeckeTF(0.1, 1.1)
        inv = InverseTF(btf)
        inv_inv = inv.inverse(inv.transform(self.array))
        assert np.allclose(inv_inv, self.array)

    def test_errors_assert(self):
        """Test errors raise."""
        # parameter error
        with self.assertRaises(ValueError):
            BeckeTF.find_parameter(np.arange(5), 0.5, 0.1)
        # transform non array type
        with self.assertRaises(TypeError):
            btf = BeckeTF(0.1, 1.1)
            btf.transform(0.5)
        # inverse init error
        with self.assertRaises(TypeError):
            InverseTF(0.5)

    # MultiExp
    def test_multiexp_tf(self):
        """Test MultiExp initialization."""
        metf = MultiExpTF(0.1, 1.1)
        assert metf.R == 1.1
        assert metf.r0 == 0.1

    def test_multiexp_transform(self):
        """Test MultiExp transformation."""
        metf = MultiExpTF(0.1, 1.1)
        tf_array = metf.transform(self.array)
        new_array = metf.inverse(tf_array)
        assert np.allclose(new_array, self.array)

    def test_multiexp_infinite(self):
        """Test Multiexp transformation when inf generated."""
        inf_array = np.linspace(-1, 1, 21)
        metf = MultiExpTF(0.1, 1.1)
        tf_array = metf.transform(inf_array, trim_inf=True)
        inv_array = metf.inverse(tf_array)
        assert np.allclose(inv_array, inf_array)

    def test_multiexp_deriv(self):
        """Test Multiexp transform derivatives with finite diff."""
        metf = MultiExpTF(0.1, 1.1)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(metf, self.array)

    def test_multiexp_inverse(self):
        """Test inverse transform."""
        metf = MultiExpTF(0.1, 1.1)
        inv = InverseTF(metf)
        new_array = inv.transform(metf.transform(self.array))
        assert np.allclose(new_array, self.array)

    def test_multiexp_inverse_inverse(self):
        """Test inverse of inverse of MultiExp transformation."""
        metf = MultiExpTF(0.1, 1.1)
        inv = InverseTF(metf)
        inv_inv = inv.inverse(inv.transform(self.array))
        assert np.allclose(inv_inv, self.array)

    # KnowlesTF
    def test_knowles_tf(self):
        """Test Knowles initialization."""
        ktf = KnowlesTF(0.1, 1.1, 2)
        assert ktf.R == 1.1
        assert ktf.r0 == 0.1
        assert ktf.k == 2

    def test_knowles_transform(self):
        """Test Knowles transformation."""
        ktf = KnowlesTF(0.1, 1.1, 2)
        tf_array = ktf.transform(self.array)
        new_array = ktf.inverse(tf_array)
        assert np.allclose(new_array, self.array)

    def test_knowles_infinite(self):
        """Test Knowles transformation when inf generated."""
        inf_array = np.linspace(-1, 1, 21)
        ktf = KnowlesTF(0.1, 1.1, 2)
        tf_array = ktf.transform(inf_array, trim_inf=True)
        inv_array = ktf.inverse(tf_array)
        assert np.allclose(inv_array, inf_array)

    def test_knowles_deriv(self):
        """Test Knowles transform derivatives with finite diff."""
        ktf = KnowlesTF(0.1, 1.1, 2)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(ktf, self.array)

    def test_knowles_inverse(self):
        """Test inverse transform."""
        ktf = KnowlesTF(0.1, 1.1, 2)
        inv = InverseTF(ktf)
        new_array = inv.transform(ktf.transform(self.array))
        assert np.allclose(new_array, self.array)

    def test_knowles_inverse_inverse(self):
        """Test inverse of inverse of Knowles transformation."""
        ktf = KnowlesTF(0.1, 1.1, 1)
        inv = InverseTF(ktf)
        inv_inv = inv.inverse(inv.transform(self.array))
        assert np.allclose(inv_inv, self.array)

    # HandyTF
    def test_handy_tf(self):
        """Test Handy initialization."""
        htf = HandyTF(0.1, 1.1, 2)
        assert htf.R == 1.1
        assert htf.r0 == 0.1
        assert htf.m == 2

    def test_handy_transform(self):
        """Test Handy transformation."""
        htf = HandyTF(0.1, 1.1, 2)
        tf_array = htf.transform(self.array)
        new_array = htf.inverse(tf_array)
        assert np.allclose(new_array, self.array)

    def test_handy_infinite(self):
        """Test Handy transformation when inf generated."""
        inf_array = np.linspace(-1, 1, 21)
        htf = HandyTF(0.1, 1.1, 2)
        tf_array = htf.transform(inf_array, trim_inf=True)
        inv_array = htf.inverse(tf_array)
        assert np.allclose(inv_array, inf_array)

    def test_handy_deriv(self):
        """Test Handy transform derivatives with finite diff."""
        htf = HandyTF(0.1, 1.1, 2)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(htf, self.array)

    def test_handy_inverse(self):
        """Test inverse transform."""
        htf = HandyTF(0.1, 1.1, 2)
        inv = InverseTF(htf)
        new_array = inv.transform(htf.transform(self.array))
        assert np.allclose(new_array, self.array)

    def test_handy_inverse_deriv(self):
        """Test inverse transformation derivatives with finite diff."""
        htf = HandyTF(0.1, 1.1, 2)
        inv = InverseTF(htf)
        r_array = 2 ** (np.arange(-1, 8, dtype=float))
        self._deriv_finite_diff(inv, r_array)

    def test_handy_inverse_inverse(self):
        """Test inverse of inverse of Handy transformation."""
        htf = HandyTF(0.1, 1.1, 1)
        inv = InverseTF(htf)
        inv_inv = inv.inverse(inv.transform(self.array))
        assert np.allclose(inv_inv, self.array)

    # LinearTF
    def test_linear_tf(self):
        """Test Linear initialization."""
        ltf = LinearTF(0.1, 1.1)
        assert ltf.r0 == 0.1
        assert ltf.rmax == 1.1

    def test_linear_transform(self):
        """Test Linear transformation."""
        ltf = LinearTF(0.1, 1.1)
        tf_array = ltf.transform(self.array)
        new_array = ltf.inverse(tf_array)
        assert np.allclose(new_array, self.array)

    def test_linear_infinite(self):
        """Test Linear transformation when inf generated."""
        inf_array = np.linspace(-1, 1, 21)
        ltf = LinearTF(0.1, 1.1)
        tf_array = ltf.transform(inf_array, trim_inf=True)
        inv_array = ltf.inverse(tf_array)
        assert np.allclose(inv_array, inf_array)

    def test_linear_deriv(self):
        """Test linear transform derivatives with finite diff."""
        ltf = LinearTF(0.1, 1.1)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(ltf, self.array)

    def test_linear_inverse(self):
        """Test inverse transform."""
        ltf = LinearTF(0.1, 1.1)
        inv = InverseTF(ltf)
        new_array = inv.transform(ltf.transform(self.array))
        assert np.allclose(new_array, self.array)

    def test_linear_inverse_deriv(self):
        """Test inverse transformation derivatives with finite diff."""
        ltf = LinearTF(0.1, 1.1)
        inv = InverseTF(ltf)
        r_array = 2 ** (np.arange(-1, 8, dtype=float))
        self._deriv_finite_diff(inv, r_array)

    def test_linear_inverse_inverse(self):
        """Test inverse of inverse of MultiExp transformation."""
        ltf = LinearTF(0.1, 1.1)
        inv = InverseTF(ltf)
        inv_inv = inv.inverse(inv.transform(self.array))
        assert np.allclose(inv_inv, self.array)
