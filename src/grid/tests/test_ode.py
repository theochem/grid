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
"""ODE test module."""
from numbers import Number
from unittest import TestCase

from grid.ode import ODE
from grid.onedgrid import GaussLaguerre
from grid.rtransform import (
    BaseTransform,
    BeckeTF,
    IdentityRTransform,
    InverseTF,
    LinearTF,
)

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from scipy.integrate import solve_bvp


class TestODE(TestCase):
    """ODE solver test class."""

    def test_transform_coeff_with_x_and_r(self):
        """Test coefficient transform between x and r."""
        coeff = np.array([2, 3, 4])
        ltf = LinearTF(1, 10)  # (-1, 1) -> (r0, rmax)
        inv_tf = InverseTF(ltf)  # (r0, rmax) -> (-1, 1)
        x = np.linspace(-1, 1, 20)
        r = ltf.transform(x)
        assert r[0] == 1
        assert r[-1] == 10
        coeff_b = ODE._transformed_coeff_ode(coeff, inv_tf, x)
        derivs_fun = [inv_tf.deriv, inv_tf.deriv2, inv_tf.deriv3]
        coeff_b_ref = ODE._transformed_coeff_ode_with_r(coeff, derivs_fun, r)
        assert_allclose(coeff_b, coeff_b_ref)

    def test_transform_coeff(self):
        """Test coefficient transform with r."""
        # d^2y / dx^2 = 1
        itf = IdentityRTransform()
        inv_tf = InverseTF(itf)
        derivs_fun = [inv_tf.deriv, inv_tf.deriv2, inv_tf.deriv3]
        coeff = np.array([0, 0, 1])
        x = np.linspace(0, 1, 10)
        coeff_b = ODE._transformed_coeff_ode_with_r(coeff, derivs_fun, x)
        # compute transformed coeffs
        assert_allclose(coeff_b, np.zeros((3, 10), dtype=float) + coeff[:, None])
        # f_x = 0 * x + 1  # 1 for every

    def test_linear_transform_coeff(self):
        """Test coefficient with linear transformation."""
        x = GaussLaguerre(10).points
        ltf = LinearTF(1, 10)
        inv_ltf = InverseTF(ltf)
        derivs_fun = [inv_ltf.deriv, inv_ltf.deriv2, inv_ltf.deriv3]
        coeff = np.array([2, 3, 4])
        coeff_b = ODE._transformed_coeff_ode_with_r(coeff, derivs_fun, x)
        # assert values
        assert_allclose(coeff_b[0], np.ones(len(x)) * coeff[0])
        assert_allclose(coeff_b[1], 1 / 4.5 * coeff[1])
        assert_allclose(coeff_b[2], (1 / 4.5) ** 2 * coeff[2])

    def test_rearange_ode_coeff(self):
        """Test rearange ode coeff and solver result."""
        coeff_b = [0, 0, 1]
        x = np.linspace(0, 2, 20)
        y = np.zeros((2, x.size))

        def fx(x):
            return 1 if isinstance(x, Number) else np.ones(x.size)

        def func(x, y):
            dy_dx = ODE._rearrange_ode(x, y, coeff_b, fx(x))
            return np.vstack((*y[1:], dy_dx))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        res = solve_bvp(func, bc, x, y)
        # res = 0.5 * x**2 - x
        assert_almost_equal(res.sol(0)[0], 0)
        assert_almost_equal(res.sol(1)[0], -0.5)
        assert_almost_equal(res.sol(2)[0], 0)
        assert_almost_equal(res.sol(0)[1], -1)
        assert_almost_equal(res.sol(1)[1], 0)
        assert_almost_equal(res.sol(2)[1], 1)

        # 2nd example
        coeff_b_2 = [-3, 2, 1]

        def fx2(x):
            return -6 * x ** 2 - x + 10

        def func2(x, y):
            dy_dx = ODE._rearrange_ode(x, y, coeff_b_2, fx2(x))
            return np.vstack((*y[1:], dy_dx))

        def bc2(ya, yb):
            return np.array([ya[0], yb[0] - 14])

        res2 = solve_bvp(func2, bc2, x, y)
        # res2 = 2 * x**2 + 3x
        assert_almost_equal(res2.sol(0)[0], 0)
        assert_almost_equal(res2.sol(1)[0], 5)
        assert_almost_equal(res2.sol(2)[0], 14)
        assert_almost_equal(res2.sol(0)[1], 3)
        assert_almost_equal(res2.sol(1)[1], 7)
        assert_almost_equal(res2.sol(2)[1], 11)

    def test_second_order_ode(self):
        """Test same result for 2nd order ode."""
        stf = SqTF()
        coeff = np.array([0, 0, 1])
        # transform
        r = np.linspace(1, 2, 10)  # r
        y = np.zeros((2, r.size))
        x = stf.transform(r)  # transformed x
        assert_almost_equal(x, r ** 2)

        def fx(x):
            return 1 if isinstance(x, Number) else np.ones(x.size)

        def func(x, y):
            dy_dx = ODE._rearrange_trans_ode(x, y, coeff, stf, fx)
            return np.vstack((*y[1:], dy_dx))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        res = solve_bvp(func, bc, x, y)
        # print(res.sol(x))
        ref_y = np.zeros((2, r.size))

        def func_ref(x, y):
            return np.vstack((y[1:], np.ones(y[0].size)))

        res_ref = solve_bvp(func_ref, bc, r, ref_y)
        assert_allclose(res.sol(x)[0], res_ref.sol(r)[0], atol=1e-4)

    def test_second_order_ode_with_diff_coeff(self):
        """Test same result for 2nd order with diff TF and Coeffs."""
        stf = SqTF()
        stf = SqTF(3, 1)
        coeff = np.array([2, 3, 2])
        # transform
        r = np.linspace(1, 2, 10)  # r
        y = np.zeros((2, r.size))
        x = stf.transform(r)  # transformed x
        # assert_almost_equal(x, r ** 2)

        def fx(x):
            return 1 / x ** 3 + 3 * x ** 2 + x

        def func(x, y):
            dy_dx = ODE._rearrange_trans_ode(x, y, coeff, stf, fx)
            return np.vstack((*y[1:], dy_dx))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        res = solve_bvp(func, bc, x, y)
        # print(res.sol(x))
        ref_y = np.zeros((2, r.size))

        def func_ref(x, y):
            return np.vstack((y[1], (fx(x) - 2 * y[0] - 3 * y[1]) / 2))

        res_ref = solve_bvp(func_ref, bc, r, ref_y)
        assert_allclose(res.sol(x)[0], res_ref.sol(r)[0], atol=1e-4)

    def test_second_order_ode_with_fx(self):
        """Test same result for 2nd order with non-homo term."""
        stf = SqTF()
        coeff = np.array([2, 2, 3])
        # transform
        r = np.linspace(1, 2, 10)  # r
        y = np.zeros((2, r.size))
        x = stf.transform(r)  # transformed x
        assert_almost_equal(x, r ** 2)

        def fx(x):
            return x ** 2 + 3 * x - 5

        def func(x, y):
            dy_dx = ODE._rearrange_trans_ode(x, y, coeff, stf, fx)
            return np.vstack((*y[1:], dy_dx))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        res = solve_bvp(func, bc, x, y)
        # print(res.sol(x))
        ref_y = np.zeros((2, r.size))

        def func_ref(x, y):
            dy_dx = ODE._rearrange_ode(x, y, coeff, fx(x))
            return np.vstack((*y[1:], dy_dx))

        res_ref = solve_bvp(func_ref, bc, r, ref_y)
        assert_allclose(res.sol(x)[0], res_ref.sol(r)[0], atol=1e-4)

    def test_becke_transform_2nd_order_ode(self):
        """Test same result for 2nd order ode with becke tf."""
        btf = BeckeTF(0.1, 2)
        ibtf = InverseTF(btf)
        coeff = np.array([0, 1, 1])
        # transform
        # r = np.linspace(1, 2, 10)  # r
        # x = ibtf.transform(r)  # transformed x
        x = np.linspace(-0.9, 0.9, 20)
        r = ibtf.inverse(x)
        y = np.zeros((2, r.size))

        def fx(x):
            return 1 / x ** 2

        def func(x, y):
            dy_dx = ODE._rearrange_trans_ode(x, y, coeff, ibtf, fx)
            return np.vstack((*y[1:], dy_dx))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        res = solve_bvp(func, bc, x, y)
        print(res.sol(x)[0])

        def func_ref(x, y):
            dy_dx = ODE._rearrange_ode(x, y, coeff, fx(x))
            return np.vstack((*y[1:], dy_dx))

        res_ref = solve_bvp(func_ref, bc, r, y)
        print(res_ref.sol(r)[0])

        assert_allclose(res.sol(x)[0], res_ref.sol(r)[0], atol=5e-3)

    def test_becke_transform_3nd_order_ode(self):
        """Test same result for 3rd order ode with becke tf."""
        btf = BeckeTF(0.1, 10)
        ibtf = InverseTF(btf)
        coeff = np.array([0, 2, 3, 3])
        # transform
        # r = np.linspace(1, 2, 10)  # r
        # x = ibtf.transform(r)  # transformed x
        x = np.linspace(-0.9, 0.9, 20)
        r = ibtf.inverse(x)
        y = np.random.rand(3, r.size)

        def fx(x):
            return 1 / x ** 4

        def func(x, y):
            dy_dx = ODE._rearrange_trans_ode(x, y, coeff, ibtf, fx)
            return np.vstack((*y[1:], dy_dx))

        def bc(ya, yb):
            return np.array([ya[0], yb[0], ya[1]])

        res = solve_bvp(func, bc, x, y)
        # print(res.sol(x)[0])

        def func_ref(x, y):
            dy_dx = ODE._rearrange_ode(x, y, coeff, fx(x))
            return np.vstack((*y[1:], dy_dx))

        res_ref = solve_bvp(func_ref, bc, r, y)
        # print(res_ref.sol(r)[0])
        assert_allclose(res.sol(x)[0], res_ref.sol(r)[0], atol=1e-4)

    def test_becke_transform_f0_ode(self):
        """Test same result for 3rd order ode with becke tf and fx term."""
        btf = BeckeTF(0.1, 10)
        x = np.linspace(-0.9, 0.9, 20)
        btf = BeckeTF(0.1, 5)
        ibtf = InverseTF(btf)
        r = btf.transform(x)
        y = np.random.rand(2, x.size)
        coeff = [-1, -1, 2]

        def fx(x):
            return -1 / x ** 2

        def func(x, y):
            dy_dx = ODE._rearrange_trans_ode(x, y, coeff, ibtf, fx)
            return np.vstack((*y[1:], dy_dx))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        res = solve_bvp(func, bc, x, y)

        def func_ref(x, y):
            dy_dx = ODE._rearrange_ode(x, y, coeff, fx(x))
            return np.vstack((*y[1:], dy_dx))

        res_ref = solve_bvp(func_ref, bc, r, y)
        assert_allclose(res.sol(x)[0], res_ref.sol(r)[0], atol=1e-4)

    def test_solve_ode_bvp(self):
        """Test result for high level api solve_ode."""
        x = np.linspace(0, 2, 10)

        def fx(x):
            return 1 if isinstance(x, Number) else np.ones(x.size)

        coeffs = [0, 0, 1]
        bd_cond = [[0, 0, 0], [1, 0, 0]]

        res = ODE.solve_ode(x, fx, coeffs, bd_cond)

        assert_almost_equal(res(0)[0], 0)
        assert_almost_equal(res(1)[0], -0.5)
        assert_almost_equal(res(2)[0], 0)
        assert_almost_equal(res(0)[1], -1)
        assert_almost_equal(res(1)[1], 0)
        assert_almost_equal(res(2)[1], 1)

    def test_solver_ode_bvp_with_tf(self):
        """Test result for high level api solve_ode with fx term."""
        x = np.linspace(-0.999, 0.999, 20)
        btf = BeckeTF(0.1, 5)
        r = btf.transform(x)
        ibtf = InverseTF(btf)

        def fx(x):
            return 1 / x ** 2

        coeffs = [-1, 1, 1]
        bd_cond = [(0, 0, 0), (1, 0, 0)]
        # calculate diff equation wt/w tf.
        res = ODE.solve_ode(x, fx, coeffs, bd_cond, ibtf)
        res_ref = ODE.solve_ode(r, fx, coeffs, bd_cond)
        assert_allclose(res(x)[0], res_ref(r)[0], atol=1e-4)

    def test_construct_coeffs(self):
        """Test construct coefficients."""
        # first test
        x = np.linspace(-0.9, 0.9, 20)
        coeff = [2, 1.5, lambda x: x ** 2]
        coeff_a = ODE._construct_coeff_array(x, coeff)
        assert_allclose(coeff_a[0], np.ones(20) * 2)
        assert_allclose(coeff_a[1], np.ones(20) * 1.5)
        assert_allclose(coeff_a[2], x ** 2)
        # second test
        coeff = [lambda x: 1 / x, 2, lambda x: x ** 3, lambda x: np.exp(x)]
        coeff_a = ODE._construct_coeff_array(x, coeff)
        assert_allclose(coeff_a[0], 1 / x)
        assert_allclose(coeff_a[1], np.ones(20) * 2)
        assert_allclose(coeff_a[2], x ** 3)
        assert_allclose(coeff_a[3], np.exp(x))

    def test_solver_ode_coeff_a_f_x_with_tf(self):
        """Test ode with a(x) and f(x) involved."""
        x = np.linspace(-0.999, 0.999, 20)
        btf = BeckeTF(0.1, 5)
        r = btf.transform(x)
        ibtf = InverseTF(btf)

        def fx(x):
            return 0 * x

        coeffs = [lambda x: x ** 2, lambda x: 1 / x ** 2, 0.5]
        bd_cond = [(0, 0, 0), (1, 0, 0)]
        # calculate diff equation wt/w tf.
        res = ODE.solve_ode(x, fx, coeffs, bd_cond, ibtf)
        res_ref = ODE.solve_ode(r, fx, coeffs, bd_cond)
        assert_allclose(res(x)[0], res_ref(r)[0], atol=1e-4)

    def test_error_raises(self):
        """Test proper error raises."""
        x = np.linspace(-0.999, 0.999, 20)
        # r = btf.transform(x)

        def fx(x):
            return 1 / x ** 2

        coeffs = [-1, -2, 1]
        bd_cond = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        with self.assertRaises(NotImplementedError):
            ODE.solve_ode(x, fx, coeffs, bd_cond[3:])
        with self.assertRaises(NotImplementedError):
            ODE.solve_ode(x, fx, coeffs, bd_cond)
        with self.assertRaises(NotImplementedError):
            test_coeff = [1, 2, 3, 4, 5]
            ODE.solve_ode(x, fx, test_coeff, bd_cond)
        with self.assertRaises(ValueError):
            test_coeff = [1, 2, 3, 3]
            tf = BeckeTF(0.1, 1)

            def fx(x):
                return x

            ODE.solve_ode(x, fx, test_coeff, bd_cond[:3], tf)


class SqTF(BaseTransform):
    """Test power transformation class."""

    def __init__(self, exp=2, extra=0):
        """Initialize power transform instance."""
        self._exp = exp
        self._extra = extra

    def transform(self, x):
        """Transform given array."""
        return x ** self._exp + self._extra

    def inverse(self, r):
        """Inverse transformed array."""
        return (r - self._extra) ** (1 / self._exp)

    def deriv(self, x):
        """Compute 1st order deriv of TF."""
        return self._exp * x ** (self._exp - 1)

    def deriv2(self, x):
        """Compute 2nd order deriv of TF."""
        return (self._exp - 1) * (self._exp) * x ** (self._exp - 2)

    def deriv3(self, x):
        """Compute 3rd order deriv of TF."""
        return (self._exp - 2) * (self._exp - 1) * (self._exp) * x ** (self._exp - 3)
