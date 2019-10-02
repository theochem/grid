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
"""Generic ode solver module."""
# from numbers import Number

# from grid.rtransform import InverseTF
from numbers import Number

import numpy as np

from scipy.integrate import solve_bvp

# from sympy import bell


class ODE:
    """General ordinary differential equation solver."""

    @staticmethod
    def solve_ode(x, fx, coeffs, bd_cond, transform=None):
        """Solve generic boundary condition ODE.

        .. math::
            ...

        Parameters
        ----------
        x : np.ndarray(N,)
            Points from domain for solver ODE
        fx : callable
            Non homogeneous term in ODE
        coeffs : list[int or callable] or np.ndarray(K,)
            Coefficients of each differential term
        bd_cond : iterable
            Boundary condition for specific solution
        transform : BaseTransform, optional
            Transformation instance r -> x

        Returns
        -------
        PPoly
            scipy.interpolate.PPoly instance for interpolating new values
            and its derivative

        Raises
        ------
        NotImplementedError
            ODE over 3rd order is not supported at this stage.
        """
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
                coeffs_mt = ODE._construct_coeff_array(x, coeffs)
                dy_dx = ODE._rearrange_ode(x, y, coeffs_mt, fx(x))
            return np.vstack((*y[1:], dy_dx))

        # define boundary condition
        def bc(ya, yb):
            bonds = [ya, yb]
            conds = []
            for i, deriv, value in bd_cond:
                conds.append(bonds[i][deriv] - value)
            return np.array(conds)

        y = np.random.rand(order, x.size)
        res = solve_bvp(func, bc, x, y, tol=1e-4)
        # raise error if didn't converge
        if res.status != 0:
            raise ValueError(
                f"The ode solver didn't converge, got status: {res.status}"
            )
        return res.sol

    @staticmethod
    def _transformed_coeff_ode(coeff_a, tf, x):
        """Compute coeff for transformed domain.

        Parameters
        ----------
        coeff_a : list[number or callable] or np.ndarray
            Coefficients for normal ODE
        tf : BaseTransform
            Transform instance form r -> x
        x : np.ndarray(N,)
            Points in the transformed domain

        Returns
        -------
        np.ndarray
            Coefficients for transformed ODE
        """
        deriv_func = [tf.deriv, tf.deriv2, tf.deriv3]
        r = tf.inverse(x)
        return ODE._transformed_coeff_ode_with_r(coeff_a, deriv_func, r)

    @staticmethod
    def _transformed_coeff_ode_with_r(coeff_a, deriv_func_list, r):
        """Convert higher order ODE into 1st order ODE with original domain r.

        Parameters
        ----------
        coeff_a : list[callable or numnber] or np.ndarray
            Coefficients for each differential part
        deriv_func_list : list[Callable]
            A list of functions for compute transformation derivatives
        r : np.ndarray
            Points from the non-transformed domain

        Returns
        -------
        np.ndarray
            Coefficients for transformed ODE
        """
        derivs = np.array([dev(r) for dev in deriv_func_list])
        total = len(coeff_a)
        # coeff_a = np.zeros((total, r.size), dtype=float)
        # # construct coeff matrix
        # for i, val in enumerate(coeff_a_ori):
        #     if isinstance(val, Number):
        #         coeff_a[i] += val
        #     else:
        #         coeff_a[i] += val(r)
        coeff_a_mtr = ODE._construct_coeff_array(r, coeff_a)
        coeff_b = np.zeros((total, r.size), dtype=float)
        # constrcut 1 - 3 directly
        coeff_b[0] += coeff_a_mtr[0]
        if total > 1:
            coeff_b[1] += coeff_a_mtr[1] * derivs[0]
        if total > 2:
            coeff_b[1] += coeff_a_mtr[2] * derivs[1]
            coeff_b[2] += coeff_a_mtr[2] * derivs[0] ** 2
        if total > 3:
            coeff_b[1] += coeff_a_mtr[3] * derivs[2]
            coeff_b[2] += coeff_a_mtr[3] * 3 * derivs[0] * derivs[1]
            coeff_b[3] += coeff_a_mtr[3] * derivs[0] ** 3

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
    def _rearrange_trans_ode(x, y, coeff_a, tf, fx_func):
        """Rearrange coefficients in transformed domain.

        Parameters
        ----------
        x : np.ndarray(n,)
            points from desired domain
        y : np.ndarray(order, n)
            initial guess for the function values and its derivatives
        coeff_a : list[number or callable] or np.ndarray(Order + 1)
            Coefficients for each differential from non-transformed part on ODE
        tf: BaseTransform
            transform instance r -> x
        fx_func : Callable
            Non-homogeneous term at given x

        Returns
        -------
        np.ndarray(N,)
            proper expr for the right side of the transformed ODE equation
        """
        coeff_b = ODE._transformed_coeff_ode(coeff_a, tf, x)
        result = ODE._rearrange_ode(x, y, coeff_b, fx_func(tf.inverse(x)))
        return result

    @staticmethod
    def _construct_coeff_array(x, coeff):
        """Construct coefficient matrix for given points.

        Parameters
        ----------
        x : np.ndarray(K,)
            Points on the mesh grid
        coeff : list[number or callable] or np.ndarray, length N
            Coefficient for each derivatives in the ode

        Returns
        -------
        np.ndarray(N, K)
            Numerical coefficient value for ODE
        """
        coeff_mtr = np.zeros((len(coeff), x.size), dtype=float)
        for i, val in enumerate(coeff):
            if isinstance(val, Number):
                coeff_mtr[i] += val
            else:
                coeff_mtr[i] += val(x)
        return coeff_mtr

    @staticmethod
    def _rearrange_ode(x, y, coeff_b, fx):
        """Rearrange coefficients for scipy solver.

        Parameters
        ----------
        x : np.ndarray(N,)
            Points from desired domain
        y : np.ndarray(order, N)
            Initial guess for the function values and its derivatives
        coeff_b : list[number] or np.ndarray(Order + 1)
            Coefficients for each differential part on ODE
        fx : np.ndarray(N,)
            Non-homogeneous term at given x

        Returns
        -------
        np.ndarray(N,)
            proper expr for the right side of the ODE equation
        """
        result = fx
        for i, b in enumerate(coeff_b[:-1]):
            # if isinstance(b, Number):
            result -= b * y[i]
        return result / coeff_b[-1]
