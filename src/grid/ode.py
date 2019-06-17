"""Generic ode solver module."""
# from numbers import Number

# from grid.rtransform import InverseTF
import numpy as np

from scipy.integrate import solve_bvp

# from sympy import bell


class ODE:
    """General ordinary differential equation solver."""

    @staticmethod
    def bvp_solver(x, fx, coeffs, bd_cond, transform=None):
        """Solve generic boundary condition ODE.

        .. math::
            ...

        Parameters
        ----------
        x : np.ndarray(N,)
            Points from domain for solver ODE
        fx : callable
            Non homogeneous term in ODE
        coeffs : list or np.ndarray(K,)
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
        res = solve_bvp(func, bc, x, y)
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
        coeff_a : list or np.ndarray
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
        coeff_a : list or np.ndarray
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
        """Rearrange coefficients in transformed domain.

        Parameters
        ----------
        x : np.ndarray(n,)
            points from desired domain
        y : np.ndarray(order, n)
            initial guess for the function values and its derivatives
        coeff_a : list or np.ndarray(Order + 1)
            Coefficients for each differential from non-transformed part on ODE
        tf: BaseTransform
            transform instance r -> x
        fx : Callable
            Non-homogeneous term at given x

        Returns
        -------
        np.ndarray(N,)
            proper expr for the right side of the transformed ODE equation
        """
        coeff_b = ODE._transformed_coeff_ode(coeff_a, tf, x)
        result = ODE._rearrange_ode(x, y, coeff_b, fx(tf.inverse(x)))
        return result

    @staticmethod
    def _rearrange_ode(x, y, coeff_b, fx):
        """Rearrange coefficients for scipy solver.

        Parameters
        ----------
        x : np.ndarray(N,)
            Points from desired domain
        y : np.ndarray(order, N)
            Initial guess for the function values and its derivatives
        coeff_b : list or np.ndarray(Order + 1)
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
