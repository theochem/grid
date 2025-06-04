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
import warnings
from numbers import Number

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_raises
from scipy.integrate import solve_bvp

from grid.ode import (
    _derivative_transformation_matrix,
    _evaluate_coeffs_on_points,
    _rearrange_to_explicit_ode,
    _transform_and_rearrange_to_explicit_ode,
    _transform_ode_from_derivs,
    _transform_ode_from_rtransform,
    solve_ode_bvp,
    solve_ode_ivp,
)
from grid.onedgrid import GaussLaguerre
from grid.rtransform import (
    BaseTransform,
    BeckeRTransform,
    IdentityRTransform,
    InverseRTransform,
    KnowlesRTransform,
    LinearFiniteRTransform,
)


# List of constant right-hand side terms
def fx_ones(x):
    """Constant all ones."""
    return 1 if isinstance(x, Number) else np.ones(x.size)


def fx_complicated_example(x):
    """Reciprocal of x cube with polynomial terms."""
    return 1 / x**3 + 3 * x**2 + x


def fx_complicated_example2(x):
    """Reciprocal of x to the power of four."""
    return 1 / x**4


def fx_complicated_example3(x):
    """Reciprocal of x to the power of two."""
    return 1 / x**2


def fx_quadratic(x):
    """Quadratic polynomial example."""
    return x**2 + 3 * x - 5


class SqTF(BaseTransform):
    """Test power transformation class."""

    def __init__(self, exp=2, extra=0):
        """Initialize power transform instance."""
        self._exp = exp
        self._extra = extra

    @property
    def domain(self):
        return (-1.0, 1.0)

    def transform(self, x):
        """Transform given array."""
        return x**self._exp + self._extra

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


@pytest.mark.parametrize(
    "transform, fx, coeffs",
    [
        [IdentityRTransform(), fx_ones, [-1, 1, 1]],
        [SqTF(), fx_ones, np.random.uniform(-100, 100, (3,))],
        [KnowlesRTransform(0.01, 1, 5), fx_ones, np.random.uniform(-10, 10, (3,))],
        [BeckeRTransform(0.1, 100.0), fx_ones, np.random.uniform(0, 100, (3,))],
        [SqTF(1, 3), fx_complicated_example, np.random.uniform(0, 100, (3,))],
        [SqTF(3, 1), fx_complicated_example, [2, 3, 2]],
        [SqTF(), fx_quadratic, np.random.uniform(-100, 100, (3,))],
        [SqTF(1, 4), fx_complicated_example, np.random.uniform(-10, 10, (3,))],
        [SqTF(1, 4), fx_complicated_example2, [1, -1, 1]],
    ],
)
def test_transform_and_rearrange_to_explicit_ode_with_simple_boundary(transform, fx, coeffs):
    r"""Test transforming second-order ode with simple boundary conditions."""
    x = np.arange(0.1, 0.99, 0.01)
    transform_pts = transform.transform(x)

    def bc(ya, yb):
        # Boundary of y is zero on endpoints
        return np.array([ya[0], yb[0]])

    # Run with transformation
    def func_with_transform(r, y):
        # Transform back to original domain x
        original = transform.inverse(r)
        # Apply the ode transfomration
        dy_dx = _transform_and_rearrange_to_explicit_ode(original, y, coeffs, transform, fx)
        return np.vstack((*y[1:], dy_dx))

    init_guess = np.zeros((2, x.size))
    solution_with_transf = solve_bvp(
        func_with_transform, bc, transform_pts, init_guess, tol=1e-6, max_nodes=10000
    )

    # Run without transformation
    def func_without_transform(original_pts, y):
        coeffs_mt = _evaluate_coeffs_on_points(original_pts, coeffs)
        dy_dx = _rearrange_to_explicit_ode(y, coeffs_mt, fx(original_pts))
        return np.vstack((y[1:], dy_dx))

    # Run without transformation
    solution_without_transf = solve_bvp(
        func_without_transform,
        bc,
        x,
        solution_with_transf.sol(transform_pts),
        tol=1e-6,
        max_nodes=100000,
    )
    # Check if the solution at x is the same as the solution (with transform) at r=g(x)
    assert_allclose(
        solution_with_transf.sol(transform_pts)[0],
        solution_without_transf.sol(x)[0],
        atol=1e-4,
    )
    # Check if they're similar at random points on interval with lower accuracy tolerance
    random_pts = np.random.uniform(np.min(x), np.max(x), transform_pts.shape)
    transf_pts = transform.transform(random_pts)
    assert_allclose(
        solution_with_transf.sol(transf_pts)[0],
        solution_without_transf.sol(random_pts)[0],
        atol=1e-4,
    )
    # Test the derivative
    assert_allclose(
        solution_with_transf.sol(transform_pts)[1],
        solution_without_transf.sol(x)[1] / transform.deriv(x),
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "transform, fx, coeffs, bd_cond",
    [
        # Test with ode -y + y` + y``=1/x^2
        [
            BeckeRTransform(1.0, 5.0),
            fx_complicated_example3,
            [-1, 1, 1],
            [(0, 0, 3), (1, 0, 3)],
        ],
        [
            InverseRTransform(BeckeRTransform(1.0, 5.0)),
            fx_complicated_example3,
            [-1, 1, 1],
            [(0, 0, 3), (1, 0, 3)],
        ],
        [
            BeckeRTransform(1.0, 5.0),
            fx_complicated_example3,
            [-1, 1, 1],
            [(0, 0, 3), (1, 0, 3)],
        ],
        # Test one with boundary conditions on the derivatives
        # [
        #     SqTF(1, 3),
        #     fx_complicated_example,
        #     np.random.uniform(-50, 50, (4,)),
        #     [(0, 0, 0), (0, 1, 3), (1, 1, 3)],
        # ],
    ],
)
def test_solve_ode_bvp_with_and_without_transormation(transform, fx, coeffs, bd_cond):
    r"""Test solve_ode with and without transformation with different bd conditions."""
    x = np.linspace(0.01, 0.999, 20)
    sol_with_transform = solve_ode_bvp(
        x,
        fx,
        coeffs,
        bd_cond,
        transform,
        tol=1e-8,
        max_nodes=20000,
        no_derivatives=False,
    )
    init_guess = sol_with_transform(x)

    sol_normal = solve_ode_bvp(
        x, fx, coeffs, bd_cond, tol=1e-8, max_nodes=20000, initial_guess_y=init_guess
    )
    # Test the function values
    assert_allclose(sol_with_transform(x)[0], sol_normal(x)[0], atol=1e-5)
    # Test the boundary condition
    for bd in bd_cond:
        bnd, deriv, val = bd
        assert_allclose(sol_with_transform(x)[deriv][-bnd], val, atol=1e-5)

    if len(coeffs) >= 3:
        # Test the first derivative of y.
        assert_allclose(sol_with_transform(x)[1], sol_normal(x)[1], atol=1e-3)
        if len(coeffs) >= 4:
            assert_allclose(sol_with_transform(x)[2], sol_normal(x)[2], atol=1e-3)


@pytest.mark.parametrize(
    "transform, fx, coeffs, ivp",
    [
        # Test with ode -y + y` + y`` =1/x^2
        [
            BeckeRTransform(0.01, 5.0),
            fx_complicated_example3,
            [-1, 1, 1],
            [3.0, 0.0],
        ],
        [
            BeckeRTransform(0.01, 5.0),
            fx_complicated_example3,
            [-1, 1, 1],
            [3.0, 6.0],
        ],
        [
            BeckeRTransform(0.01, 5.0),
            fx_complicated_example3,
            [-1, 1, 1],
            [0.0, 0.0],
        ],
        # Test one with boundary conditions on the derivatives
        [
            SqTF(1, 3),
            fx_complicated_example,
            np.random.uniform(-100, 100, (4,)),
            [0.0, 3.0, 6.0],
        ],
        [
            KnowlesRTransform(0.01, 1.5, 3),
            fx_complicated_example,
            np.random.uniform(-100, 100, (4,)),
            [0.0, 3.0, 6.0],
        ],
    ],
)
def test_solve_ode_ivp_with_and_without_transformation(transform, fx, coeffs, ivp):
    r"""Test solve_ode_ivp with and without transformation with different initial guesses."""
    sol_with_transform = solve_ode_ivp(
        (0.01, 0.999),
        fx,
        coeffs,
        ivp,
        transform,
    )

    sol_normal = solve_ode_ivp((0.01, 0.999), fx, coeffs, ivp)

    # Test the initial value problem
    for i, ivp_val in enumerate(ivp):
        assert_allclose(sol_normal(np.array([0.01]))[i], ivp_val, atol=1e-5)
        assert_allclose(sol_with_transform(np.array([0.01]))[i], ivp_val, atol=1e-5)

    # Test the function values
    x = np.arange(0.01, 0.999, 0.1)
    assert_allclose(sol_with_transform(x)[0], sol_normal(x)[0], atol=1e-5, rtol=1e-5)
    if len(coeffs) >= 3:
        # Test the first derivative of y.
        assert_allclose(sol_with_transform(x)[1], sol_normal(x)[1], atol=1e-3, rtol=1e-5)
        if len(coeffs) >= 4:
            assert_allclose(sol_with_transform(x)[2], sol_normal(x)[2], atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize(
    "fx, coeffs, bvp, solutions",
    [
        # Test ode  y^`` = 1 with y(0) = 0, y`(0) = -1
        [
            lambda x: 1 if isinstance(x, Number) else np.ones(x.size),
            [0, 0, 1],
            [[0, 0, 0.0], [0, 1, -1.0]],
            lambda x: (x**2.0 / 2.0 - x, x - 1.0),
        ],
        # Test ode y`+ y cos(t) =0,
        [
            lambda x: 0 if isinstance(x, Number) else np.zeros(x.size),
            [lambda x: np.cos(x), 1],
            [[0, 0, 0.5]],
            lambda x: np.array([np.exp(-np.sin(x)) / 2.0]),
        ],
        # # Test ode y`` - y` = 2 sin(x) with y(0) = 1 and y`(0)=-1
        [
            lambda x: 2.0 * np.sin(x),
            [0, -1, 1],
            [[0, 0, 1.0], [0, 1, -1.0]],
            lambda x: (np.cos(x) - np.sin(x), -np.sin(x) - np.cos(x)),
        ],
    ],
)
def test_solve_ode_bvp_against_analytic_example(fx, coeffs, bvp, solutions):
    """Test solve_ode_ivp against analytic solution."""
    # res = solve_ode_ivp((0, 2), fx, coeffs, ivp)
    x = np.arange(0.0, 2.0, 0.1)
    res = solve_ode_bvp(x, fx, coeffs, bvp)

    # Test on random points.
    # rand_pts = np.random.uniform(0.0, 2.0, size=100)
    rand_pts = np.arange(0, 2, 0.5)
    assert_allclose(res(rand_pts), solutions(rand_pts), atol=1e-5)


@pytest.mark.parametrize(
    "fx, coeffs, ivp, solutions",
    [
        # Test ode  y^`` = 1 with y(0) = 0, y`(0) = -1
        [
            lambda x: 1 if isinstance(x, Number) else np.ones(x.size),
            [0, 0, 1],
            [0.0, -1.0],
            lambda x: (x**2.0 / 2.0 - x, x - 1.0),
        ],
        # Test ode y`+ y cos(t) =0,
        [
            lambda x: 0 if isinstance(x, Number) else np.zeros(x.size),
            [lambda x: np.cos(x), 1],
            [0.5],
            lambda x: np.array([np.exp(-np.sin(x)) / 2.0]),
        ],
        # Test ode y`` - y` = 2 sin(x) with y(0) = 1 and y`(0)=-1
        [
            lambda x: 2.0 * np.sin(x),
            [0, -1, 1],
            [1.0, -1.0],
            lambda x: (np.cos(x) - np.sin(x), -np.sin(x) - np.cos(x)),
        ],
    ],
)
def test_solve_ode_ivp_against_analytic_example(fx, coeffs, ivp, solutions):
    """Test solve_ode_ivp against analytic solution."""
    res = solve_ode_ivp((0, 2), fx, coeffs, ivp)

    # Test on random points.
    # rand_pts = np.random.uniform(0.0, 2.0, size=100)
    rand_pts = np.arange(0, 2, 0.5)
    assert_allclose(res(rand_pts), solutions(rand_pts), atol=1e-5)


def test_error_raises_for_solve_ode_bvp():
    """Test proper error raises for solve_ode_bvp."""
    x = np.linspace(-0.999, 0.999, 20)

    def fx(x):
        # Ignore any warnings from divide by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return 1 / x**2

    coeffs = [-1, -2, 1]
    bd_cond = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]

    # Ignore the SciPy warning from divide by zero, since this is an assertion test-case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Test the error that the number of boundary conditions should be equal to order.
        assert_raises(ValueError, solve_ode_bvp, x, fx, coeffs, bd_cond[3:])
        assert_raises(ValueError, solve_ode_bvp, x, fx, coeffs, bd_cond)
        test_coeff = [1, 2, 3, 4, 5]
        assert_raises(ValueError, solve_ode_bvp, x, fx, test_coeff, bd_cond)

        test_coeff = [1, 2, 3, 3]
        tf = BeckeRTransform(0.1, 1)
        assert_raises(ValueError, solve_ode_bvp, x, fx, test_coeff, bd_cond[:3], tf)


def test_construct_coeffs_of_ode_over_mesh():
    """Test construct coefficients over a mesh."""
    # first test
    x = np.linspace(-0.9, 0.9, 20)
    coeff = [2, 1.5, lambda x: x**2]
    coeff_a = _evaluate_coeffs_on_points(x, coeff)
    assert_allclose(coeff_a[0], np.ones(20) * 2)
    assert_allclose(coeff_a[1], np.ones(20) * 1.5)
    assert_allclose(coeff_a[2], x**2)
    # second test
    coeff = [lambda x: 1 / x, 2, lambda x: x**3, lambda x: np.exp(x)]
    coeff_a = _evaluate_coeffs_on_points(x, coeff)
    assert_allclose(coeff_a[0], 1 / x)
    assert_allclose(coeff_a[1], np.ones(20) * 2)
    assert_allclose(coeff_a[2], x**3)
    assert_allclose(coeff_a[3], np.exp(x))


def test_transform_coeff_with_x_and_r():
    """Test coefficient transform between x and r."""
    coeff = np.array([2, 3, 4])
    ltf = LinearFiniteRTransform(1, 10)  # (-1, 1) -> (r0, rmax)
    inv_tf = InverseRTransform(ltf)  # (r0, rmax) -> (-1, 1)
    x = np.linspace(-1, 1, 20)
    r = ltf.transform(x)
    assert r[0] == 1
    assert r[-1] == 10
    # Transform ODE from [1, 10) to (-1, 1)
    coeff_transform = _transform_ode_from_rtransform(coeff, inv_tf, x)
    derivs_fun = [inv_tf.deriv, inv_tf.deriv2, inv_tf.deriv3]
    coeff_transform_all_pts = _transform_ode_from_derivs(coeff, derivs_fun, r)
    assert_allclose(coeff_transform, coeff_transform_all_pts)


def test_transformation_of_ode_with_identity_transform():
    """Test transformation of ODE with identity transform."""
    # Checks that the identity transform x -> x results in the same answer.
    # Obtain identity trasnform and derivatives.
    itf = IdentityRTransform()
    inv_tf = InverseRTransform(itf)
    derivs_fun = [inv_tf.deriv, inv_tf.deriv2, inv_tf.deriv3]
    # d^2y / dx^2 = 1
    coeff = np.array([0, 0, 1])
    x = np.linspace(0, 1, 10)
    # compute transformed coeffs
    coeff_b = _transform_ode_from_derivs(coeff, derivs_fun, x)
    # f_x = 0 * x + 1  # 1 for every
    assert_allclose(coeff_b, np.zeros((3, 10), dtype=float) + coeff[:, None])


def test_transformation_of_ode_with_linear_transform():
    """Test transformation of ODE with linear transformation."""
    x = GaussLaguerre(10).points
    # Obtain linear transformation with rmin = 1 and rmax = 10.
    ltf = LinearFiniteRTransform(1, 10)
    # The inverse is x_i = \frac{r_i - r_{min} - R} {r_i - r_{min} + R}
    inv_ltf = InverseRTransform(ltf)
    derivs_fun = [inv_ltf.deriv, inv_ltf.deriv2, inv_ltf.deriv3]
    # Test with 2y + 3y` + 4y``
    coeff = np.array([2, 3, 4])
    coeff_b = _transform_ode_from_derivs(coeff, derivs_fun, x)
    # assert values
    assert_allclose(coeff_b[0], np.ones(len(x)) * coeff[0])
    assert_allclose(coeff_b[1], 1 / 4.5 * coeff[1])
    assert_allclose(coeff_b[2], (1 / 4.5) ** 2 * coeff[2])


def test_high_order_transformations_gives_itself():
    r"""Test transforming then transforming back gives back the same result."""

    # Consider the following transformation x^4 and its derivatives
    def transf(x):
        return x**4.0

    derivs = [
        lambda x: 4.0 * x**3.0,
        lambda x: 4.0 * 3.0 * x**2.0,
        lambda x: 4.0 * 3.0 * 2.0 * x,
        lambda x: 4.0 * 3.0 * 2.0 * np.array([1.0] * len(x)),
    ]

    # Consider ODE 2y + 3y` + 4y`` + 5y``` + 6dy^4/dx^4 and transform it
    coeffs = np.array([2, 3, 4, 5, 6])
    x = np.arange(1.0, 2.0, 0.01)
    coeffs_transf = _transform_ode_from_derivs(coeffs, derivs, x)
    # Transform it back using the derivative of the inverse transformation x^4
    derivs_invs = [
        lambda r: 1.0 / (4.0 * r ** (3.0 / 4.0)),
        lambda r: -3.0 / (16.0 * r ** (7.0 / 4.0)),
        lambda r: 21.0 / (64.0 * r ** (11.0 / 4.0)),
        lambda r: -231.0 / (256.0 * r ** (15.0 / 4.0)),
    ]
    x_transformed = transf(x)
    # Go Through Each Points and grab the new coefficients and transform it back
    for i in range(0, len(x)):
        coeffs_original = _transform_ode_from_derivs(
            np.ravel(coeffs_transf[:, i]), derivs_invs, x_transformed[i : i + 1]
        )
        # Check that it is the same as hte original transformation.
        assert_almost_equal(coeffs, np.ravel(coeffs_original))


def test_rearange_ode_coeff():
    """Test rearange ode coeff and solver result."""
    coeff_b = [0, 0, 1]
    x = np.linspace(0, 2, 20)
    y = np.zeros((2, x.size))  # Initial Guess

    def fx(x):
        return 1 if isinstance(x, Number) else np.ones(x.size)

    def func(x, y):
        dy_dx = _rearrange_to_explicit_ode(y, coeff_b, fx(x))
        return np.vstack((*y[1:], dy_dx))

    def bc(ya, yb):
        # Boundary conditions: zero at endpoints of y
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
        return -6 * x**2 - x + 10

    def func2(x, y):
        dy_dx = _rearrange_to_explicit_ode(y, coeff_b_2, fx2(x))
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


def test_first_and_second_derivative_transformation_with_Becke_transform():
    r"""Test derivative transformation of cubic function with Becke transform."""
    transform = BeckeRTransform(0.0, 5.0)

    def func(x):
        return x**3.0

    origin_domain = np.arange(0.0, 10, 0.1)  # r \in [0, \infty)
    new_domain = transform.inverse(origin_domain)  # x \in [-1, 1]

    # dr/dx,  d^2 r/ dx^2,
    deriv_tranfs = [transform.deriv, transform.deriv2, transform.deriv3]

    # derivative g(r) := r^3 wrt to r in [0, \infty)
    def deriv_func_old(x):
        return 3.0 * x**2.0

    def sec_deriv_func_old(x):
        return 6.0 * x

    # derivative g(r(x)) wrt to x in [-1, 1]
    def desired_deriv_new(x):
        return 6 * 5.0**3.0 * (1 + x) ** 2.0 / (1 - x) ** 4.0

    def desired_sec_deriv_new(x):
        return -12 * 5.0**3.0 * (1 + x) * (x + 3) / (x - 1.0) ** 5.0

    # Go through each pt, calculate the jacobian, calculate the derivative g(r(x)) and compare
    for i, pt_x in enumerate(new_domain):
        pt_r = origin_domain[i]
        # derivative r^3 wrt to r in [0, \infty)
        actual_deriv_origin = np.array([deriv_func_old(pt_r), sec_deriv_func_old(pt_r)])

        # transform derivative to get  g(r(x)) wrt to x in [-1, 1]
        jacobian = _derivative_transformation_matrix(deriv_tranfs, pt_x, 2)
        actual_deriv_new = jacobian.dot(actual_deriv_origin)

        assert_allclose(actual_deriv_new, [desired_deriv_new(pt_x), desired_sec_deriv_new(pt_x)])
