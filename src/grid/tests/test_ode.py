from unittest import TestCase
from numbers import Number

from grid.ode import ODE
from grid.rtransform import (
    IdentityRTransform,
    InverseTF,
    LinearTF,
    BaseTransform,
    BeckeTF,
)
from grid.onedgrid import GaussLaguerre

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.integrate import solve_bvp


class TestODE(TestCase):
    def test_transform_coeff_with_x_and_r(self):
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
        def fx(x):
            return 1 if isinstance(x, Number) else np.ones(x.size)

        coeff_b = [0, 0, 1]
        x = np.linspace(0, 2, 20)
        y = np.zeros((2, x.size))

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
        print(res.sol(x)[0])

        def func_ref(x, y):
            dy_dx = ODE._rearrange_ode(x, y, coeff, fx(x))
            return np.vstack((*y[1:], dy_dx))

        res_ref = solve_bvp(func_ref, bc, r, y)
        print(res_ref.sol(r)[0])

        assert_allclose(res.sol(x)[0], res_ref.sol(r)[0], atol=3e-4)

    def test_solve_ode_bvp(self):
        x = np.linspace(0, 2, 10)

        def fx(x):
            return 1 if isinstance(x, Number) else np.ones(x.size)

        coeffs = [0, 0, 1]
        bd_cond = [[0, 0, 0], [1, 0, 0]]

        res = ODE.solve_bvp(x, fx, coeffs, bd_cond)

        assert_almost_equal(res.sol(0)[0], 0)
        assert_almost_equal(res.sol(1)[0], -0.5)
        assert_almost_equal(res.sol(2)[0], 0)
        assert_almost_equal(res.sol(0)[1], -1)
        assert_almost_equal(res.sol(1)[1], 0)
        assert_almost_equal(res.sol(2)[1], 1)

    def test_solver_ode_bvp_with_tf(self):
        x = np.linspace(-1, 1, 20)
        # btf = BeckeTF(0.1, 1.5)
        # r = btf.transform(x)


class SqTF(BaseTransform):
    def __init__(self, exp=2, extra=0):
        self._exp = exp
        self._extra = extra

    def transform(self, x):
        return x ** self._exp + self._extra

    def inverse(self, r):
        return (r - self._extra) ** (1 / self._exp)

    def deriv(self, x):
        return self._exp * x ** (self._exp - 1)

    def deriv2(self, x):
        return (self._exp - 1) * (self._exp) * x ** (self._exp - 2)

    def deriv3(self, x):
        return (self._exp - 2) * (self._exp - 1) * (self._exp) * x ** (self._exp - 3)
