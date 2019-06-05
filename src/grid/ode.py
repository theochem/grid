"""Generic ode solver module."""
from grid.rtransform import InverseTF
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.integrate import solve_bvp, solve_ivp
from sympy import bell


class ODE:
    def __init__(self, radial_points, radial_value, order=3):
        self._r_pts = radial_points
        self._order = order
        self.spl = UnivariateSpline(self._r_pts, radial_value, k=self._order)

    def solve_ode_ivp(self, x, fx, coeffs, transform, bc):
        if not isinstance(transform, InverseTF):
            raise NotImplementedError  # should be a warning here.
        order = len(coeffs) - 1
        if order > 3:
            raise NotImplementedError  # transform only support update to 3rd order derivative
        derivs = [transform.deriv, transform.deriv2, transform.deriv3][:order]
        coeff_b = ODE._transformed_coeff_ode(coeffs, derivs, x)
        y = np.zeros((order, x.size))

        # define 1st order ODE for solver
        def func(x, y):
            dy_dx = ODE._rearrange_ode(coeff_b, fx, y)
            return np.vstack((*y[1:], dy_dx))

        sol = solve_bvp(func, bc, x, y)
        return sol

    @staticmethod
    def _transformed_coeff_ode(coeff_a, deriv_func_list, x):
        derivs = np.array([dev(x) for dev in deriv_func_list])
        total = len(coeff_a)
        coeff_b = np.zeros((total, x.size), dtype=float)
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
        # construct 4 and onwards
        if total >= 4:
            for i, j in enumerate(x):
                for coeff_index in range(4, total):
                    for dev_order in range(coeff_index, total):  # efficiency
                        coeff_b[coeff_index, i] += (
                            float(bell(dev_order, coeff_index, derivs[:, i]))
                            * coeff_a[dev_order]
                        )
        return coeff_b

    @staticmethod
    def _rearrange_ode(coeff_b, f_x, ys):
        result = f_x
        for i, b in enumerate(coeff_b[:-1]):
            result -= b * ys[i]
        return result / coeff_b[-1]
