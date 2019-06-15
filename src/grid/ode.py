"""Generic ode solver module."""
from numbers import Number

from grid.rtransform import InverseTF

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_bvp
from sympy import bell


class ODE:

    @staticmethod
    def solve_bvp(x, fx, coeffs, bd_cond, transform=None):
        order = len(coeffs) - 1
        if len(bd_cond) != order:
            raise NotImplementedError(
                "# of boundary condition need to be the same as ODE order."
                f"Expect: {order}, got: {len(bd_cond)}."
            )
        if order > 3:
            raise NotImplementedError("Only support 3rd order ODE or less.")

        # define 1st order ODE for solver
        def func(x, y):
            if transform:
                dy_dx = ODE._rearrange_trans_ode(x, y, coeffs, transform, fx)
            else:
                dy_dx = ODE._rearrange_ode(x, y, coeffs, fx(x))
            return np.vstack((*y[1:], dy_dx))

        # define boundary condition
        def bc(ya, yb):
            bonds = [ya, yb]
            conds = []
            for i, deriv, value in bd_cond:
                conds.append(bonds[i][deriv] - value)
            return np.array(conds)

        y = np.random.rand(order, x.size)
        sol = solve_bvp(func, bc, x, y)
        return sol

    @staticmethod
    def _transformed_coeff_ode(coeff_a, tf, x):
        deriv_func = [tf.deriv, tf.deriv2, tf.deriv3]
        r = tf.inverse(x)
        return ODE._transformed_coeff_ode_with_r(coeff_a, deriv_func, r)

    @staticmethod
    def _transformed_coeff_ode_with_r(coeff_a, deriv_func_list, r):
        derivs = np.array([dev(r) for dev in deriv_func_list])
        total = len(coeff_a)
        coeff_b = np.zeros((total, r.size), dtype=float)
        # constrcut 1 - 3 directly
        coeff_b[0] += coeff_a[0]
        if total > 1:
            coeff_b[1] += coeff_a[1] * derivs[0]
        if total > 2:
            coeff_b[1] += coeff_a[2] * derivs[1]
            coeff_b[2] += coeff_a[2] * derivs[0] ** 2
        if total > 3:
            coeff_b[1] += coeff_a[3] * derivs[2]
            coeff_b[2] += coeff_a[3] * 3 * derivs[0] * derivs[1]
            coeff_b[3] += coeff_a[3] * derivs[0] ** 3

        # construct 4th order and onwards
        # if total >= 4:
        #     for i, j in enumerate(r):
        #         for coeff_index in range(4, total):
        #             for dev_order in range(coeff_index, total):  # efficiency
        #                 coeff_b[coeff_index, i] += (
        #                     float(bell(dev_order, coeff_index, derivs[:, i]))
        #                     * coeff_a[dev_order]
        #                 )
        return coeff_b

    @staticmethod
    def _rearrange_trans_ode(x, y, coeff_a, tf, fx):
        coeff_b = ODE._transformed_coeff_ode(coeff_a, tf, x)
        result = ODE._rearrange_ode(x, y, coeff_b, fx(tf.inverse(x)))
        return result

    @staticmethod
    def _rearrange_ode(x, y, coeff_b, fx):
        result = fx
        for i, b in enumerate(coeff_b[:-1]):
            # if isinstance(b, Number):
            result -= b * y[i]
        return result / coeff_b[-1]
