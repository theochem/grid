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
r"""Linear ordinary differential equation solver.

Solves a linear ordinary differential equation of order :math:`K` of the form

.. math::
    \sum_{k=0}^{K} a_k(x) \frac{d^k y(x)}{dx^k} = f(x)

with either boundary conditions on the two end-points for some unknown function
:math:`y(x)` with independent variable :math:`x` or as an initial value
problem on a interval.

It also supports the ability to transform the independent variable :math:`x`
to another domain :math:`g(x)` for some :math:`K`-th differentiable transformation
:math:`g(x)`.  This is particularly useful to convert infinite domains, e.g.
:math:`[0, \infty)`, to a finite interval.
"""

from __future__ import annotations

import warnings
from numbers import Number, Real

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.linalg import solve
from sympy import bell

from grid.rtransform import BaseTransform

__all__ = ["solve_ode_bvp", "solve_ode_ivp"]


def solve_ode_ivp(
    x_span: tuple,
    fx: callable,
    coeffs: list | np.ndarray,
    y0: list | np.ndarray,
    transform: BaseTransform = None,
    method: str = "DOP853",
    no_derivatives: bool = False,
    rtol: float = 1e-8,
    atol: float = 1e-6,
):
    r"""
    Solve a linear ODE as an initial value problem.

    Parameters
    ----------
    x_span : (int, int)
        The interval of integration :math:`(t_0, t_1)` from the first point to the second point.
    fx : callable
        Right-hand function :math:`f(x)`.
    coeffs : list[callable or float] or ndarray(K + 1,)
        Coefficients :math:`a_k` of each term :math:`\frac{d^k y(x)}{d x^k}`
        ordered from 0 to K. Either a list of callable functions :math:`a_k(x)` that depends
        on :math:`x` or array of constants :math:`\{a_k\}_{k=0}^{K}`.
    y0 : list[K] or ndarray(K)
        The initial value conditions :math:`\frac{d^k y(t_0)}{d x^k} = c_k` at the initial point
        :math:`t_0` from :math:`k=0,\cdots,K-1`.
    transform : BaseTransform, optional
        Transformation from one domain :math:`x` to another :math:`r := g(x)`.
    method : str
        The method used to solve the ode by scipy.
        See `scipy.integrate.solve_ivp` function for more info.
    no_derivatives : bool, optional
        If true, when transform is used then it only returns the solution :math:`y(x)` rather
        than its derivative. If false, it includes the derivatives up to :math:`P-1`.
    rtol, atol : (float, float), optional
        The relative and absolute tolerance. See `scipy.integrate.solve_ivp` for more info.

    Returns
    -------
    callable :
        Interpolate function (scipy.interpolate.PPoly) instance whose input is the
        original domain :math:`x` and output is an array of the function :math:`y(x)` evaluated
        on the points and its derivatives wrt to :math:`x` up to :math:`K - 1`.

    """
    order = len(coeffs) - 1
    if len(y0) != order:
        raise ValueError(
            "Number of boundary condition need to be the same as ODE order."
            f"Expect: {order}, got: {len(y0)}."
        )

    if transform is not None and order > 3:
        raise NotImplementedError("Only support 3rd order ODE or less when using `transform`.")

    def func(x, y):
        # x has shape (1,) and y has shape (K+1,1), output has shape (K+1,1)
        x = np.array([x])
        if transform:
            # Transform the points back to the original domain.
            orig_dom = transform.inverse(x)
            dy_dx = _transform_and_rearrange_to_explicit_ode(orig_dom, y, coeffs, transform, fx)
        else:
            coeffs_mt = _evaluate_coeffs_on_points(x, coeffs)
            dy_dx = _rearrange_to_explicit_ode(y, coeffs_mt, fx(x))
        # (*y[1:, :],) returns a tuple of all rows excluding the first row.
        #    This is due to conversion to system of first-order ODE form.
        return np.vstack((*y[1:, :], dy_dx))

    if transform:
        # first check if the bounds are in the domain
        if min(x_span) < transform.domain[0] or max(x_span) > transform.domain[1]:
            raise ValueError(
                f"The x_span {min(x_span), max(x_span)} is not within the transform "
                f"domain {transform.domain}."
            )
        # Convert the initial value problem to the new derivative space, only transform up to K-1
        # e.g. the first derivative dV/dx = dV/dr * dr/dx = dV/dr / (dx/dr)
        deriv = _derivative_transformation_matrix(
            [transform.deriv, transform.deriv2, transform.deriv3],
            x_span[0],
            order - 1,  # Only need derivatives up to K-1.
        )
        # If transform is used, then transform (x_0, x_1) that it integrates up to.
        x_span = transform.transform(np.array(list(x_span)))
        # Solve for derivatives in original domain by solving A(original derivs) = new derivs
        y_derivs = solve(deriv, np.array(y0[1:]))
        if np.any(np.isinf(y_derivs)):
            raise ValueError(
                f"The initial value of the derivative {y_derivs} "
                f"when using transform is infinity."
            )
        y0 = np.hstack(([y0[0]], y_derivs))

    res = solve_ivp(
        func,
        x_span,
        y0=y0,
        dense_output=True,
        vectorized=True,
        rtol=rtol,
        atol=atol,
        method=method,
    )

    # raise error if didn't converge
    if res.status != 0:
        raise ValueError(f"The ode solver didn't converge, got status: {res.status}")

    if transform is not None:
        # Transform the function so that it's input is the original variable and
        #   derivative is with respect to the original variable as well.
        return _transform_solution_to_original_domain(res, transform, no_derivatives, order)

    return res.sol


def solve_ode_bvp(
    x: np.ndarray,
    fx: callable,
    coeffs: list | np.ndarray,
    bd_cond: list,
    transform: BaseTransform = None,
    tol: float = 1e-4,
    max_nodes: int = 5000,
    initial_guess_y: np.ndarray = None,
    no_derivatives: bool = True,
):
    r"""Solve a linear ODE with boundary conditions.

    Parameters
    ----------
    x : np.ndarray(N,)
        Points of the independent variable/domain.  If `transform` is provided, then
        these are the points that are to be transformed.
    fx : callable
        Right-hand function :math:`f(x)`.
    coeffs : list[callable or float] or ndarray(K + 1,)
        Coefficients :math:`a_k` of each term :math:`\frac{d^k y(x)}{d x^k}`
        ordered from 0 to K. Either a list of callable functions :math:`a_k(x)` that depends
        on :math:`x` or array of constants :math:`\{a_k\}_{k=0}^{K}`.
    bd_cond : list[list[int, int, float]]
        Boundary condition specified by list of size :math:`K` of three entries [i, j, C],
        where :math:`i \in \{0, 1\}` determines whether the lower or upper bound is being
        constrained, :math:`j \in \{0, \cdots, K\}` determines which derivative
        :math:`\frac{d^j y}{dx^j}` is being constrained, and :math:`C`
        determines the boundary constraint value, i.e. :math:`\frac{d^j y}{dx^j} = C`.
        Size needs to be the same as the order :math:`K` of the ODE.
        If `transform` is given, then the constraint of the derivatives is assumed
        to be with respect to the new coordinate i.e. :math:`\frac{d^j y}{dr^j} = C`.
    transform : BaseTransform, optional
        Transformation from one domain :math:`x` to another :math:`r := g(x)`.
    tol : float, optional
        Tolerance of the ODE solver and for the boundary condition residuals.
        See `scipy.integrate.solve_bvp` for more info.
    max_nodes : int, optional
        The maximum number of mesh nodes that determine termination.
        See `scipy.integrate.solve_bvp` function for more info.
    initial_guess_y : ndarray(K, N), optional
        Initial guess for :math:`y(x), \cdots, \frac{d^{K} y}{d x^{K}}` at the points `x`.
        If not provided, then random set of points from 0 to 1.
    no_derivatives : bool, optional
        If true, when transform is used then it only returns the solution :math:`y(x)` rather
        than its derivative. If false, it includes the derivatives up to :math:`P-1`.

    Returns
    -------
    callable :
        Interpolate function (scipy.interpolate.PPoly) instance whose input is the
        original domain :math:`x` and output is an array of the function :math:`y(x)` evaluated
        on the points and its derivatives wrt to :math:`x` up to :math:`K - 1`.

    """
    order = len(coeffs) - 1
    if len(bd_cond) != order:
        raise ValueError(
            "Number of boundary condition need to be the same as ODE order."
            f"Expect: {order}, got: {len(bd_cond)}."
        )
    if transform is not None and order > 3:
        raise NotImplementedError("Only support 3rd order ODE or less when using `transform`.")

    # define first order ODE for solver, needs to be in explicit form for scipy solver.
    def func(x, y):
        # x has shape (N,) and y has shape (K+1, N), output has shape (K+1, N)
        if transform:
            # Transform the points back to the original domain.
            orig_dom = transform.inverse(x)
            dy_dx = _transform_and_rearrange_to_explicit_ode(orig_dom, y, coeffs, transform, fx)
        else:
            coeffs_mt = _evaluate_coeffs_on_points(x, coeffs)
            dy_dx = _rearrange_to_explicit_ode(y, coeffs_mt, fx(x))
        # (*y[1:, :],) returns a tuple of all rows excluding the first row.
        #    This is due to conversion to first-order ODE form.
        return np.vstack((*y[1:, :], dy_dx))

    # define boundary condition
    def bc(ya, yb):
        # a denotes the lower bound, b denotes the upper bound
        # Input of ya, yb should have shape (K+1,) , output should be shape (K+1,)
        bonds = [ya, yb]
        conds = []
        for i, deriv, value in bd_cond:
            conds.append(bonds[i][deriv] - value)
        return np.array(conds)

    # Generate random initial guess if not provided.
    if initial_guess_y is None:
        initial_guess_y = np.random.rand(order, x.size)

    # Solve the ODE
    if transform:
        pts_tf = transform.transform(x)
        res = solve_bvp(func, bc, pts_tf, y=initial_guess_y, tol=tol, max_nodes=max_nodes)
    else:
        res = solve_bvp(func, bc, x, y=initial_guess_y, tol=tol, max_nodes=max_nodes)

    # raise error if didn't converge
    if res.status != 0:
        raise ValueError(f"The ode solver didn't converge, got status: {res.status}")

    if transform is not None:
        # Transform the function so that it's input is the original variable and
        #   derivative is with respect to the original variable as well.
        return _transform_solution_to_original_domain(res, transform, no_derivatives, order)

    return res.sol


def _transform_solution_to_original_domain(result, tf, no_derivs, order):
    r"""Transform interpolate solution to the original domains and its derivatives."""

    # Note this is it's own function becuase it is used twice for solve_ode_ivp and bv.
    def interpolate_wrt_original_var(pt):
        transf_pts = tf.transform(pt)
        # Row is which func/deriv and Col is points.
        interpolated = result.sol(transf_pts)
        # If derivatives are not wanted then only return y(x).
        if no_derivs:
            if interpolated.ndim == 1:
                return interpolated
            return interpolated[0, :]
        deriv_funcs = [tf.deriv, tf.deriv2, tf.deriv3]
        new_interpolate = np.zeros(interpolated.shape)
        new_interpolate[0, :] = interpolated[0, :]
        for i in range(interpolated.shape[1]):
            # Calculate the jacobian dr/dx of the original domain.
            deriv = _derivative_transformation_matrix(deriv_funcs, pt[i], order - 1)
            new_interpolate[1:, i] = deriv.dot(interpolated[1:, i])
        return new_interpolate

    return interpolate_wrt_original_var


def _transform_ode_from_derivs(
    coeffs: list | np.ndarray, deriv_transformation: list, x: np.ndarray
):
    r"""
    Transform the coefficients of ODE from one variable to another evaluated on mesh points.

    Given a :math:`K`-th differentiable transformation function :math:`g(x)`
    and a linear ODE of :math:`K`-th order on independent variable :math:`x`

    .. math::
        \sum_{k=1}^{K} a_k(x) \frac{d^k y(x)}{d x^k}.

    This transforms it into a new coordinate system :math:`g(x) =: r`

    .. math::
        \sum_{j=1}^K b_j(r) \frac{d^j y(r)}{d r^j},

    where :math:`b_j(r) = \sum_{k=1}^K a_k(g^{-1}(r)) B_{k, j}({g^{-1}}^\prime(g^{-1}(r)),
    \cdots, {g^{-1}}^{k - j + 1}(g^{-1}(r)))`.

    Parameters
    ----------
    coeffs : list[callable or number] or ndarray
        Coefficients :math:`a_k` of each term :math:`\frac{d^k y(x}{d x^k}`
        ordered from 0 to K. Either a list of callable functions :math:`a_k(x)` that depends
        on :math:`x` or array of constants :math:`\{a_k\}_{k=0}^{K}`.
    deriv_transformation : list[callable]
        List of functions for compute transformation derivatives from 1 to :math:`K`.
    x : ndarray
        Points from the original domain that is to be transformed.

    Returns
    -------
    ndarray((K+1, N))
        Coefficients :math:`b_j(r)` of the new ODE with respect to transformed variable :math:`r`.

    """
    # `derivs` has shape (K, N), calculate d^j g/ dr^j
    derivs = np.array([dev(x) for dev in deriv_transformation], dtype=float)
    total = len(coeffs)  # Should be K+1, K is the order of the ODE
    # coeff_a_mtr has shape (K+1, N)
    coeff_a_mtr = _evaluate_coeffs_on_points(x, coeffs)  # a_k(x)
    coeff_b = np.zeros((total, x.size), dtype=float)
    # The first term doesn't contain a derivative so no transformation is required:
    coeff_b[0] += coeff_a_mtr[0]
    # construct 1 - 3 directly with vectorization (faster)
    if total > 1:
        coeff_b[1] += coeff_a_mtr[1] * derivs[0]
    if total > 2:
        coeff_b[1] += coeff_a_mtr[2] * derivs[1]
        coeff_b[2] += coeff_a_mtr[2] * derivs[0] ** 2
    if total > 3:
        coeff_b[1] += coeff_a_mtr[3] * derivs[2]
        coeff_b[2] += coeff_a_mtr[3] * 3 * derivs[0] * derivs[1]
        coeff_b[3] += coeff_a_mtr[3] * derivs[0] ** 3

    # construct 4th order and onwards without vectorization (slower)
    # formula is: d^k f / dr^k = \sum_{j=1}^{k} Bell_{k, j}(...)  (d^jf / dx^j)
    if total > 4:
        # Go Through each Pt
        for i_pt in range(len(x)):
            # Go through each order from 4 to K + 1
            for j in range(4, total):
                # Go through the sum to calculate Bell's polynomial
                for k in range(j, total):
                    all_derivs_at_pt = derivs[:, i_pt]
                    coeff_b[j, i_pt] += float(bell(k, j, all_derivs_at_pt)) * coeff_a_mtr[k, i_pt]
    return coeff_b


def _transform_ode_from_rtransform(coeff_a: list | np.ndarray, tf: BaseTransform, x: np.ndarray):
    r"""
    Transform the coefficients of ODE from one variable to another based on Transform object.

    Given a :math:`K`-th differentiable transformation function :math:`g(x)`
    and a linear ODE of :math:`K`-th order

    .. math::
        \sum_{k=1}^{K} a_k(x) \frac{d^k y(x)}{d x^k} = f(x).

    This transforms it into a new coordinate system with variable :math:`g(x) =: r` via

    .. math::
        \sum_{j=1}^K b_j(r) \frac{d^j y(r)}{d r^j} = f(r),

    where :math:`b_j(r) = \sum_{k=1}^K a_k(g^{-1}(r)) B_{k, j}(g^\prime(g^{-1}(r)),
    \cdots, g^{k - j + 1}(g^{-1}(r)))`.

    Parameters
    ----------
    coeff_a : list[number or callable] or np.ndarray
        Coefficients for normal ODE
    tf : BaseTransform
        Transform instance of :math:`g(x)`
    x : np.ndarray(N,)
        Points of the independent variable/domain.

    Returns
    -------
    ndarray((K+1, N))
        Coefficients :math:`b_j(r)` of the new ODE with respect to transformed variable :math:`r`.

    """
    deriv_func = [tf.deriv, tf.deriv2, tf.deriv3]
    return _transform_ode_from_derivs(coeff_a, deriv_func, x)


def _transform_and_rearrange_to_explicit_ode(
    x: np.ndarray,
    y: np.ndarray,
    coeff_a: list | np.ndarray,
    tf: BaseTransform,
    fx_func: callable,
):
    r"""Rearrange coefficients in transformed domain.

    Parameters
    ----------
    x : np.ndarray(N,)
        Points of the independent variable/domain.
    y : np.ndarray(K + 1, N)
        Initial guess for the function values and its derivatives
    coeff_a : list[number or callable] or np.ndarray(K + 1)
        Coefficients :math:`a_k` of each term :math:`\frac{d^k y(x)}{d x^k}`
        ordered from 0 to K.  Each coefficient can either be a callable function
        :math:`a_k(x)` or a constant number :math:`a_k`.
    tf: BaseTransform
        Transform instance of :math:`g(x)`.
    fx_func : Callable
        Right-hand function of the ODE.

    Returns
    -------
    np.ndarray(N,)
        Expression for the right side of the ODE equation after converting it
        to explicit form evaluated on all N points.

    """
    coeff_b = _transform_ode_from_rtransform(coeff_a, tf, x)
    result = _rearrange_to_explicit_ode(y, coeff_b, fx_func(x))
    return result


def _derivative_transformation_matrix(deriv_func_list: list, point: float, order: int):
    r"""Compute transformation from one variable to another.

    Suppose there is a function :math:`f(x)` and a :math:`N`-th differentiable
    transformation :math:`g(x) = r` from variable :math:`x` to variable :math:`r`.
    This function returns a matrix that converts the derivatives
    :math:`[\frac{df}{dr}, \cdots, \frac{d^N f}{dr^N}]^T` by matrix multiplication
    to the derivatives of the old domain
    :math:`[\frac{df}{dx}, \cdots, \frac{d^Nf}{dx^N}]^T`.

    It is a matrix with (i,j)-th entries:

    .. math::
        m_{i, j} = \begin{cases}
           B_{i,j}\bigg(\frac{dg}{dx}, \cdots, \frac{d^{k-j+1} g}{dx^{k-j+1}} \bigg) &
           i \leq j \\
           0 & \text{else}
        \end{cases}

    Parameters
    ----------
    deriv_func_list : list[K]
        List of size :math:`K` callable functions representing the derivatives of
        :math:`g(x)`, i.e. :math:`[\frac{dg}{dx}, \cdots, \frac{d^K g}{dx^K}]`.
    point : float
        The point in :math:`r`, in which the derivatives are being calculated at.
    order : int
        The number of derivatives from 1 to `order` that are going to be transformed.

    Returns
    -------
    ndarray((order, order))
        Returns the matrix that converts derivatives of one domain to another.

    """
    if not isinstance(point, (Real, float)):
        raise TypeError(f"Point {type(point)} should be of type float.")
    numb_derivs = len(deriv_func_list)
    if order > numb_derivs:
        raise ValueError(
            f"Order {order} should not be greater than number of derivatives"
            f"functions {len(deriv_func_list)} provided."
        )
    # Calculate derivatives of transformation evaluated at the point
    derivs_at_pt = np.array([dev(point) for dev in deriv_func_list], dtype=float)
    deriv_transf = np.zeros((order, order))
    for i in range(0, order):
        for j in range(0, i + 1):
            deriv_transf[i, j] = float(bell(i + 1, j + 1, derivs_at_pt))
    return deriv_transf


def _evaluate_coeffs_on_points(x: np.ndarray, coeff: list | np.ndarray):
    r"""Construct coefficients of the ODE evaluated on all points.

    Explicitly, constructs a matrix with :math:`(n, k)`-th entry
    :math:`c_{nk} = a_k(x_n)`,

    where :math:`k`-th coefficient :math:`a_k` from the derivative
    :math:`\frac{d^k y}{d x^k}` evaluated on the :math:`n`-th term :math:`x_n`.

    Parameters
    ----------
    x : ndarray(N,)
        Points of the independent variable/domain.
    coeffs : list[callable or float] or ndarray(K + 1,)
        Coefficients :math:`a_k` of each term :math:`\frac{d^k y(x)}{d x^k}`
        ordered from 0 to K.  Each coefficient can either be a callable function
        :math:`a_k(x)` or a constant number :math:`a_k`.

    Returns
    -------
    np.ndarray(K + 1, N)
        Coefficients :math:`a_k(x_n)` from the ODE.

    """
    coeff_mtr = np.zeros((len(coeff), x.size), dtype=float)
    for i, val in enumerate(coeff):
        if isinstance(val, Number):
            coeff_mtr[i] += val
        elif callable(val):
            coeff_mtr[i] += val(x)
        else:
            raise TypeError(f"Coefficient value {type(val)} is either a number or a function.")
    return coeff_mtr


def _rearrange_to_explicit_ode(y: np.ndarray, coeff_b: np.ndarray, fx: np.ndarray):
    r"""Rearrange ODE into explicit form for conversion to first-order ODE (SciPy solver).

    Returns array with :math:`n`-th entry

    .. math::
        \frac{f(x_n) -\sum_{k=0}^{K - 1} a_k(x_n) \frac{d^k y(x_n)}{d x^k}}{a_{K}(x_n)},

    of the :math:`n`-th point :math:`x` such that :math:`a_K \neq 0`.
    This arises from re-arranging the ODE of :math:`K`-th order into explicit form.
    It is possible that :math:`a_K(x_n) = 0`, it is recommended to split the points
    into separate intervals and solve them separately.

    Parameters
    ----------
    y : np.ndarray(K + 1, N)
        Initial guess for the function values and its :math:`k`-th order derivatives
        from 1 to :math:`K`, evaluated on the :math:`n`-th point :math:`x_n`.
    coeff_b : ndarray(K + 1, N)
        Coefficients :math:`a_k(x_n)` of each term :math:`\frac{d^k y(x)}{d x^k}`
        ordered from 0 to K evaluated on the :math:`n`-th point :math:`x_n`.
    fx : np.ndarray(N,)
        Right hand-side of the ODE equation, function :math:`f(x_n)` evaluated on the
        :math:`n`-th point :math:`x_n`.

    Returns
    -------
    np.ndarray(N,)
        Expression for the right side of the ODE equation after converting it
        to explicit form evaluated on N-th points.

    """
    if np.any(np.abs(coeff_b[-1]) < 1e-10):
        warnings.warn(
            "The coefficient of the leading Kth term is zero at some point."
            "It is recommended to split into intervals and solve separately.",
            stacklevel=2,
        )

    result = fx
    # Go through all rows except the last-element.
    for i, b in enumerate(coeff_b[:-1]):
        # array of size N: a_k(x_n) * (d^k y(x_n) / d x^k)
        result -= b * y[i]

    return result / coeff_b[-1]
