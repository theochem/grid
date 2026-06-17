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

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from grid.onedgrid import GaussChebyshev, GaussLegendre, UniformInteger
from grid.rtransform import (
    BaseTransform,
    BeckeRTransform,
    HandyModRTransform,
    HandyRTransform,
    InverseRTransform,
    KnowlesRTransform,
    LinearFiniteRTransform,
    MultiExpRTransform,
)

x_points_cases = [np.linspace(-0.9, 0.9, 19), np.linspace(-0.9, 0.9, 10)]
r_val_cases = [-0.9, 0.9]


def compute_fd_deriv(fcn, x, eps, order):
    """
    6th-order accurate centered finite-difference derivatives Fornberg-type stencils.
    """
    h = eps
    x = np.asarray(x)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    # define stencil points and weights
    if order == 1:
        offsets = np.arange(7) - 3
        w = np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]) / h
    elif order == 2:
        offsets = np.arange(7) - 3
        w = np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]) / h**2
    elif order == 3:
        offsets = np.arange(9) - 4
        w = (
            np.array(
                [-7 / 240, 3 / 10, -169 / 120, 61 / 30, 0, -61 / 30, 169 / 120, -3 / 10, 7 / 240]
            )
            / h**3
        )
    else:
        raise ValueError("order must be 1, 2, or 3")

    # find f(x) at the (N) points x + (K) offsets * h (N, K)
    pts = x[:, None] + offsets[None, :] * h
    vals = fcn(pts)
    output = np.sum(vals * w[None, :], axis=1)
    return output[0] if scalar_input else output


def transformation_case(transform_class, kwargs):
    param_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
    return pytest.param(
        transform_class,
        kwargs,
        id=f"{transform_class.__name__}[{param_str}]",
    )


# cases for testing init raises on invalid initialization parameters
INVALID_INITIALIZE_CASES = [
    transformation_case(KnowlesRTransform, dict(rmin=0.1, R=1.0, k=0)),
    transformation_case(HandyModRTransform, dict(rmin=0.1, rmax=10.0, m=0)),
    transformation_case(HandyModRTransform, dict(rmin=10.0, rmax=0.1, m=2)),
]

# cases of bounded transforms with domain [-1, 1]; verify raises on out-of-domain inputs
BOUNDED_DOMAIN_CASES = [
    transformation_case(BeckeRTransform, dict(rmin=0.1, R=1.2)),
    transformation_case(KnowlesRTransform, dict(rmin=0.1, R=1.2, k=2)),
    transformation_case(MultiExpRTransform, dict(rmin=0.1, R=1.2)),
    transformation_case(HandyModRTransform, dict(rmin=0.1, rmax=10.0, m=2)),
    transformation_case(LinearFiniteRTransform, dict(rmin=0.1, rmax=10)),
]

# cases of valid transforms used in tests
VALID_TRANSFORM_CASES = [
    transformation_case(BeckeRTransform, dict(rmin=0.1, R=1.2, trim_inf=True)),
    transformation_case(BeckeRTransform, dict(rmin=0.1, R=1.2, trim_inf=False)),
    transformation_case(KnowlesRTransform, dict(rmin=0.1, R=1.2, k=2, trim_inf=True)),
    transformation_case(KnowlesRTransform, dict(rmin=0.1, R=1.2, k=2, trim_inf=False)),
    transformation_case(MultiExpRTransform, dict(rmin=0.1, R=1.1, trim_inf=True)),
    transformation_case(MultiExpRTransform, dict(rmin=0.1, R=1.1, trim_inf=False)),
    transformation_case(HandyRTransform, dict(rmin=0.1, R=1.2, m=2, trim_inf=True)),
    transformation_case(HandyRTransform, dict(rmin=0.1, R=1.2, m=2, trim_inf=False)),
    transformation_case(HandyModRTransform, dict(rmin=0.1, rmax=10.0, m=2, trim_inf=True)),
    transformation_case(HandyModRTransform, dict(rmin=0.1, rmax=10.0, m=2, trim_inf=False)),
    transformation_case(LinearFiniteRTransform, dict(rmin=0.1, rmax=10)),
    transformation_case(LinearFiniteRTransform, dict(rmin=0.1, rmax=10)),
]


def test_base_transform_convert_inf():
    """Test conversion of infinite values for both array and scalar inputs."""

    class DummyTransform(BaseTransform):
        def transform(self, x):
            pass

        def inverse(self, x):
            pass

        def deriv(self, x):
            pass

        def deriv2(self, x):
            pass

        def deriv3(self, x):
            pass

    tf = DummyTransform()

    # Array case
    input_vals = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    results = tf._convert_inf(input_vals)
    expected = np.array([-1e16, -1.0, 0.0, 1.0, 1e16])
    assert_allclose(expected, results)

    # Scalar cases
    assert tf._convert_inf(np.inf) == 1e16
    assert tf._convert_inf(-np.inf) == -1e16

    # Finite scalar passthrough
    assert tf._convert_inf(2.5) == 2.5
    assert tf._convert_inf(-3.2) == -3.2


@pytest.mark.parametrize("transform_class, kwargs", INVALID_INITIALIZE_CASES)
def test_init_raises(transform_class, kwargs):
    """Test that transform classes raise ValueError on invalid initialization parameters."""
    with pytest.raises(ValueError):
        transform_class(**kwargs)


@pytest.mark.parametrize("transform_class, kwargs", BOUNDED_DOMAIN_CASES)
def test_bounded_domain_raises(transform_class, kwargs):
    """Test that transforming grid points outside the valid transform domain raises an error."""
    rad = UniformInteger(10)
    tf = transform_class(**kwargs)

    with pytest.raises(ValueError):
        tf.transform_1d_grid(rad)


@pytest.mark.parametrize("x_points", x_points_cases)
@pytest.mark.parametrize("transform_class, kwargs", VALID_TRANSFORM_CASES)
def test_forward_inverse_consistency(x_points, transform_class, kwargs):
    """Test forward and inverse consistency for all valid transforms."""

    transformation = transform_class(**kwargs)

    transformed_points = transformation.transform(x_points)
    inverse_transformed_points = transformation.inverse(transformed_points)

    assert_allclose(inverse_transformed_points, x_points, rtol=1e-5)

    for x_scalar in x_points:
        x_scalar = np.float64(x_scalar)

        transformed_scalar = transformation.transform(x_scalar)
        inverse_transformed_scalar = transformation.inverse(transformed_scalar)

        assert_allclose(inverse_transformed_scalar, x_scalar, rtol=1e-5)


@pytest.mark.parametrize("x_points", x_points_cases)
@pytest.mark.parametrize("transform_class, kwargs", VALID_TRANSFORM_CASES)
def test_deriv_method(x_points, transform_class, kwargs):
    """Test that derivative method works correctly for different transform classes."""
    transformation = transform_class(**kwargs)
    d1 = transformation.deriv(x_points)
    d1_fd = compute_fd_deriv(transformation.transform, x_points, eps=1e-4, order=1)
    assert np.allclose(d1, d1_fd, atol=1e-6)

    for x_scalar in x_points:
        x_scalar = np.float64(x_scalar)
        d1_scalar = transformation.deriv(x_scalar)
        d1_fd_scalar = compute_fd_deriv(transformation.transform, x_scalar, eps=1e-4, order=1)
        assert_allclose(d1_scalar, d1_fd_scalar, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("x_points", x_points_cases)
@pytest.mark.parametrize("transform_class, kwargs", VALID_TRANSFORM_CASES)
def test_deriv2_method(x_points, transform_class, kwargs):
    """Test second derivative against finite-difference approximation."""

    transformation = transform_class(**kwargs)

    d2 = transformation.deriv2(x_points)
    d2_fd = compute_fd_deriv(transformation.transform, x_points, eps=1e-4, order=2)

    assert_allclose(d2, d2_fd, rtol=1e-4, atol=1e-6)

    for x_scalar in x_points:
        x_scalar = np.float64(x_scalar)

        d2_scalar = transformation.deriv2(x_scalar)
        d2_fd_scalar = compute_fd_deriv(transformation.transform, x_scalar, eps=1e-4, order=2)

        assert_allclose(d2_scalar, d2_fd_scalar, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("x_points", x_points_cases)
@pytest.mark.parametrize("transform_class, kwargs", VALID_TRANSFORM_CASES)
def test_deriv3_method(x_points, transform_class, kwargs):
    """Test third derivative against finite-difference approximation."""

    transformation = transform_class(**kwargs)

    d3 = transformation.deriv3(x_points)
    d3_fd = compute_fd_deriv(transformation.transform, x_points, eps=1e-3, order=3)

    assert_allclose(d3, d3_fd, rtol=1e-3, atol=1e-5)

    for x_scalar in x_points:
        x_scalar = np.float64(x_scalar)

        d3_scalar = transformation.deriv3(x_scalar)
        d3_fd_scalar = compute_fd_deriv(transformation.transform, x_scalar, eps=1e-3, order=3)

        assert_allclose(d3_scalar, d3_fd_scalar, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("x_points", x_points_cases)
@pytest.mark.parametrize("transform_class, kwargs", VALID_TRANSFORM_CASES)
def test_deriv_inverse_method(x_points, transform_class, kwargs):
    """Test deriv_inverse method."""

    transformation = transform_class(**kwargs)

    r = transformation.transform(x_points)
    dx_dr = transformation.deriv_inverse(r)
    reference = 1.0 / transformation.deriv(x_points)

    assert_allclose(dx_dr, reference, rtol=1e-6, atol=1e-8)

    for x_scalar in x_points:
        x_scalar = np.float64(x_scalar)

        dx_dr_scalar = transformation.deriv_inverse(r)
        reference_scalar = 1.0 / transformation.deriv(x_points)
        assert_allclose(dx_dr_scalar, reference_scalar, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("x_points", x_points_cases)
@pytest.mark.parametrize("transform_class, kwargs", VALID_TRANSFORM_CASES)
def test_deriv2_inverse_method(x_points, transform_class, kwargs):
    """Test deriv2_inverse method"""

    transformation = transform_class(**kwargs)

    x = np.asarray(x_points)
    r = transformation.transform(x_points)
    dx2_dr2 = transformation.deriv2_inverse(r)
    d1 = transformation.deriv(x_points)
    d2 = transformation.deriv2(x_points)
    reference = -d2 / (d1**3)

    assert_allclose(dx2_dr2, reference, rtol=1e-6, atol=1e-8)

    for x_scalar in x_points:
        x_scalar = np.float64(x_scalar)
        r_scalar = transformation.transform(x_scalar)
        dx2_dr2 = transformation.deriv2_inverse(r_scalar)
        d1_scalar = transformation.deriv(x_scalar)
        d2_scalar = transformation.deriv2(x_scalar)
        reference = -d2_scalar / (d1_scalar**3)

        assert_allclose(dx2_dr2, reference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("x_points", x_points_cases)
@pytest.mark.parametrize("transform_class, kwargs", VALID_TRANSFORM_CASES)
def test_deriv3_inverse_method(x_points, transform_class, kwargs):
    """Test deriv3_inverse method"""

    transformation = transform_class(**kwargs)

    x = np.asarray(x_points)
    r = transformation.transform(x)
    dx3_dr3 = transformation.deriv3_inverse(r)
    d1 = transformation.deriv(x)
    d2 = transformation.deriv2(x)
    d3 = transformation.deriv3(x)
    reference = (3 * d2**2 - d1 * d3) / (d1**5)

    assert_allclose(dx3_dr3, reference, rtol=1e-5, atol=1e-7)

    for x_scalar in x_points:
        x_scalar = np.float64(x_scalar)
        r_scalar = transformation.transform(x_scalar)
        dx3_dr3_scalar = transformation.deriv3_inverse(r_scalar)
        d1_scalar = transformation.deriv(x_scalar)
        d2_scalar = transformation.deriv2(x_scalar)
        d3_scalar = transformation.deriv3(x_scalar)

        reference_scalar = (3 * d2_scalar**2 - d1_scalar * d3_scalar) / (d1_scalar**5)

        assert_allclose(
            dx3_dr3_scalar,
            reference_scalar,
            rtol=1e-5,
            atol=1e-7,
        )


def test_becke_r_transform_init():
    """Test BeckeRTransform initialization."""
    btf = BeckeRTransform(0.1, 1.2)
    assert btf.R == 1.2
    assert btf.rmin == 0.1


@pytest.mark.parametrize(
    "x_points, r_min, R",
    [
        pytest.param(x_points_cases[0], 0.1, 1.2, id="r_min=0.1,R=1.2"),
        pytest.param(x_points_cases[1], 0.2, 1.3, id="r_min=0.2,R=1.3"),
    ],
)
def test_becke_r_transform_find_parameter(x_points, r_min, R):
    """Test BeckeRTransform find_parameter method."""

    # find r parameter, such that transformed grid center point is R_ref
    r_param = BeckeRTransform.find_parameter(x_points, r_min, R)

    becke_transform = BeckeRTransform(r_min, r_param)
    if len(x_points) % 2 == 1:
        center_point_x = x_points[len(x_points) // 2]
    else:
        center_point_x = (x_points[len(x_points) // 2 - 1] + x_points[len(x_points) // 2]) / 2

    # check that calculated r_param is close to reference value
    ref_rparam = (R - r_min) * (1 - center_point_x) / (1 + center_point_x)
    assert_allclose(r_param, ref_rparam, rtol=1e-5)

    # check that transformed center point is close to R
    transformed_center = becke_transform.transform(center_point_x)
    assert_allclose(transformed_center, R, rtol=1e-5)


def test_becke_r_transform_trimmed_infinity_roundtrip():
    """Test BeckeRTransform roundtrip when infinities are trimmed during the transform."""
    x_points = np.linspace(-1, 1, 21)
    r_param = BeckeRTransform.find_parameter(x_points, 0.1, 1.2)
    becke_transform = BeckeRTransform(0.1, r_param, trim_inf=True)

    transformed_points = becke_transform.transform(x_points)
    inverse_transformed_points = becke_transform.inverse(transformed_points)

    assert_allclose(inverse_transformed_points, x_points)


@pytest.mark.parametrize(
    "x_points, r_min, R",
    [
        pytest.param(x_points_cases[0], 0.1, 1.2, id="r_min=0.1,R=1.2"),
        pytest.param(x_points_cases[1], 0.2, 1.3, id="r_min=0.2,R=1.3"),
    ],
)
def test_becke_r_transform_inverse_derivatives(x_points, r_min, R):
    """Test BeckeRTransform inverse derivatives against finite difference."""
    transform = BeckeRTransform(r_min, R)
    r_points = transform.transform(x_points)

    first_derivative = transform.deriv_inverse(r_points)
    first_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=1)
    assert_allclose(
        first_derivative,
        first_derivative_fd,
        rtol=1e-5,
        atol=1e-8,
        err_msg="First inverse derivative mismatch",
    )

    second_derivative = transform.deriv2_inverse(r_points)
    second_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=2)
    assert_allclose(
        second_derivative,
        second_derivative_fd,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Second inverse derivative mismatch",
    )

    third_derivative = transform.deriv3_inverse(r_points)
    third_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-3, order=3)
    assert_allclose(
        third_derivative,
        third_derivative_fd,
        rtol=1e-3,
        atol=1e-5,
        err_msg="Third inverse derivative mismatch",
    )


def test_becke_integral():
    """Test transform integral."""
    btf = BeckeRTransform(0.00001, 1.0)

    def gauss(x):
        return np.exp(-(x**2))

    ref_result = np.sqrt(np.pi) / 2

    oned = GaussLegendre(20)
    rad = btf.transform_1d_grid(oned)
    result = rad.integrate(gauss(rad.points))
    assert_almost_equal(result, ref_result, decimal=5)

    oned = GaussChebyshev(20)
    rad = btf.transform_1d_grid(oned)
    result = rad.integrate(gauss(rad.points))
    assert_almost_equal(result, ref_result, decimal=3)


def test_linear_transform_inverse_derivatives():
    """Test inverse transform and derivs function."""
    ltf = LinearFiniteRTransform(0.1, 10)
    ltf = InverseRTransform(ltf)
    x_values = np.sort(np.random.uniform(-1.0, 1.0, 50))
    r_values = ltf.transform(x_values)

    first_derivative = ltf.deriv_inverse(r_values)
    first_derivative_fd = compute_fd_deriv(ltf.inverse, r_values, eps=1e-4, order=1)
    assert_allclose(first_derivative, first_derivative_fd, rtol=1e-5, atol=1e-8)

    second_derivative = ltf.deriv2_inverse(r_values)
    second_derivative_fd = compute_fd_deriv(ltf.inverse, r_values, eps=1e-4, order=2)
    assert_allclose(second_derivative, second_derivative_fd, rtol=1e-4, atol=1e-6)

    third_derivative = ltf.deriv3_inverse(r_values)
    third_derivative_fd = compute_fd_deriv(ltf.inverse, r_values, eps=1e-3, order=3)
    assert_allclose(third_derivative, third_derivative_fd, rtol=1e-3, atol=2e-3)

    for x_scalar in x_values:
        x_scalar = np.float64(x_scalar)

        first_derivative = ltf.deriv_inverse(x_scalar)
        first_derivative_fd = compute_fd_deriv(ltf.inverse, x_scalar, eps=1e-4, order=1)
        assert_allclose(first_derivative, first_derivative_fd, rtol=1e-5, atol=1e-8)

        second_derivative = ltf.deriv2_inverse(x_scalar)
        second_derivative_fd = compute_fd_deriv(ltf.inverse, x_scalar, eps=1e-4, order=2)
        assert_allclose(second_derivative, second_derivative_fd, rtol=1e-4, atol=1e-6)

        third_derivative = ltf.deriv3_inverse(x_scalar)
        third_derivative_fd = compute_fd_deriv(ltf.inverse, x_scalar, eps=1e-3, order=3)
        assert_allclose(third_derivative, third_derivative_fd, rtol=1e-3, atol=2e-3)


def test_becke_rtransform_raise_errors():
    """Test that errors are raised for invalid inputs."""
    # Parameter error (R must be greater than r_min)
    with pytest.raises(ValueError):
        BeckeRTransform.find_parameter(np.arange(5), 0.5, 0.1)

    btf = BeckeRTransform(0.1, 1.1)

    # Transform requires array-like numeric input
    with pytest.raises(TypeError):
        btf.transform("dafasdf")

    # transform_1d_grid requires a OneDGrid instance
    with pytest.raises(TypeError):
        btf.transform_1d_grid(np.arange(3))

    # Singular transform leads to division by zero in inverse derivatives
    singular_btf = BeckeRTransform(0.1, 0)
    with pytest.raises(ZeroDivisionError):
        singular_btf.deriv_inverse(0.5)


def test_multiexp_rtransform_init():
    """Test MultiExp initializaiton."""
    btf = MultiExpRTransform(0.1, 1.2)
    assert btf.R == 1.2
    assert btf.rmin == 0.1


@pytest.mark.parametrize(
    "x_points, r_min, R",
    [
        pytest.param(x_points_cases[0], 0.1, 1.2, id="r_min=0.1,R=1.2"),
        pytest.param(x_points_cases[1], 0.2, 1.3, id="r_min=0.2,R=1.3"),
    ],
)
def test_multiexp_r_transform_inverse_derivatives(x_points, r_min, R):
    """Test MultiExpRTransform inverse derivatives against finite difference."""
    transform = MultiExpRTransform(r_min, R)
    r_points = transform.transform(x_points)

    first_derivative = transform.deriv_inverse(r_points)
    first_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=1)
    assert_allclose(
        first_derivative,
        first_derivative_fd,
        rtol=1e-5,
        atol=1e-8,
        err_msg="First inverse derivative mismatch",
    )

    second_derivative = transform.deriv2_inverse(r_points)
    second_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=2)
    assert_allclose(
        second_derivative,
        second_derivative_fd,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Second inverse derivative mismatch",
    )

    third_derivative = transform.deriv3_inverse(r_points)
    third_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-3, order=3)
    assert_allclose(
        third_derivative,
        third_derivative_fd,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Third inverse derivative mismatch",
    )


def test_knowles_rtransform_init():
    """Test KnowlesRTransform initializaiton."""
    ktf = KnowlesRTransform(0.1, 1.2, 2)
    assert ktf.R == 1.2
    assert ktf.rmin == 0.1
    assert ktf.k == 2


@pytest.mark.parametrize("x_points", x_points_cases)
def test_knowles_r_transform_inverse_derivatives(x_points):
    """Test KnowlesRTransform  inverse derivatives against finite difference."""
    transform = KnowlesRTransform(0.1, 1.1, 2)
    r_points = transform.transform(x_points)

    first_derivative = transform.deriv_inverse(r_points)
    first_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=1)
    assert_allclose(
        first_derivative,
        first_derivative_fd,
        rtol=1e-5,
        atol=1e-8,
        err_msg="First inverse derivative mismatch",
    )

    second_derivative = transform.deriv2_inverse(r_points)
    second_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=2)
    assert_allclose(
        second_derivative,
        second_derivative_fd,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Second inverse derivative mismatch",
    )

    third_derivative = transform.deriv3_inverse(r_points)
    third_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=3)
    assert_allclose(
        third_derivative,
        third_derivative_fd,
        rtol=5e-3,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "rmin, rmax, m",
    [
        (0.1, 1.2, 2),
        (0.1, 1.1, 2),
    ],
)
def test_handy_rtransform_init(rmin, rmax, m):
    """Test HandyRTransform initialization."""
    btf = HandyRTransform(rmin, rmax, m)
    assert btf.R == rmax
    assert btf.m == m
    assert btf.rmin == rmin


@pytest.mark.parametrize("x_points", x_points_cases)
def test_handy_r_transform_inverse_derivatives(x_points):
    """Test HandyRTransform  inverse derivatives against finite difference."""
    transform = HandyRTransform(0.1, 1.2, 2)
    r_points = transform.transform(x_points)

    first_derivative = transform.deriv_inverse(r_points)
    first_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=1)
    assert_allclose(first_derivative, first_derivative_fd, rtol=1e-5, atol=1e-8)

    second_derivative = transform.deriv2_inverse(r_points)
    second_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=2)
    assert_allclose(second_derivative, second_derivative_fd, rtol=1e-4, atol=1e-6)

    third_derivative = transform.deriv3_inverse(r_points)
    third_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=3)
    assert_allclose(third_derivative, third_derivative_fd, rtol=5e-3, atol=1e-4)


@pytest.mark.parametrize(
    "rmin, rmax, m",
    [
        (0.1, 10.0, 2),
    ],
)
def test_handymod_rtransform_init(rmin, rmax, m):
    """Test HandyModRTransform initialization."""
    btf = HandyModRTransform(rmin, rmax, m)
    assert np.isclose(btf.rmin, rmin)
    assert np.isclose(btf.rmax, rmax)
    assert btf.m == m


@pytest.mark.parametrize("x_points", x_points_cases)
def test_handymod_r_transform_inverse_derivatives(x_points):
    """Test HandyModRTransform  inverse derivatives against finite difference."""
    transform = HandyModRTransform(0.1, 10.0, 2)
    r_points = transform.transform(x_points)

    first_derivative = transform.deriv_inverse(r_points)
    first_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=1)
    assert_allclose(first_derivative, first_derivative_fd, rtol=1e-5, atol=1e-8)

    second_derivative = transform.deriv2_inverse(r_points)
    second_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=2)
    assert_allclose(second_derivative, second_derivative_fd, rtol=1e-4, atol=1e-6)

    third_derivative = transform.deriv3_inverse(r_points)
    third_derivative_fd = compute_fd_deriv(transform.inverse, r_points, eps=1e-4, order=3)
    assert_allclose(third_derivative, third_derivative_fd, rtol=5e-3, atol=1e-4)


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
        btf = BeckeRTransform(0.1, 1.2)
        assert btf.R == 1.2
        assert btf.rmin == 0.1

    def test_becke_parameter_calc(self):
        """Test parameter function."""
        R = BeckeRTransform.find_parameter(self.array, 0.1, 1.2)
        # R = 1.1
        assert np.isclose(R, 1.1)
        btf = BeckeRTransform(0.1, R)
        tf_array = btf.transform(self.array)
        assert tf_array[9] == 1.2
        # for even number of grid
        R = BeckeRTransform.find_parameter(self.array_2, 0.2, 1.3)
        btf_2 = BeckeRTransform(0.2, R)
        tf_elemt = btf_2.transform(np.array([(self.array_2[4] + self.array_2[5]) / 2]))
        assert_allclose(tf_elemt, 1.3)

    def test_becke_transform(self):
        """Test becke transformation."""
        btf = BeckeRTransform(0.1, 1.1)
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
        R = BeckeRTransform.find_parameter(inf_array, 0.1, 1.2)
        btf = BeckeRTransform(0.1, R, trim_inf=True)
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
        btf = BeckeRTransform(0.1, 1.1)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-1, 0.90, btf)

    def test_becke_inverse(self):
        """Test inverse transform basic function."""
        btf = BeckeRTransform(0.1, 1.1)
        inv = InverseRTransform(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_becke_inverse_deriv(self):
        """Test inverse transformation derivatives with finite diff."""
        btf = BeckeRTransform(0.1, 1.1)
        inv = InverseRTransform(btf)
        self._deriv_finite_diff(0, 20, inv)

    def test_deriv_inverse(self):
        """Test first inverse-derivative method against finite difference."""
        btf = BeckeRTransform(0.1, 1.1)
        points = np.array([0.2, 1.5, 7.0])
        eps = 1e-4
        # 4th-order centered stencil for the first derivative of inverse(r)
        # f'(r) ~= [f(r-2h) - 8f(r-h) + 8f(r+h) - f(r+2h)] / (12h)
        deriv_fd = (
            btf.inverse(points - 2 * eps)
            - 8 * btf.inverse(points - eps)
            + 8 * btf.inverse(points + eps)
            - btf.inverse(points + 2 * eps)
        ) / (12 * eps)
        assert_allclose(btf.deriv_inverse(points), deriv_fd, rtol=1e-5)

        # test for scalar
        point = 0.5
        deriv_fd_scalar = (
            btf.inverse(point - 2 * eps)
            - 8 * btf.inverse(point - eps)
            + 8 * btf.inverse(point + eps)
            - btf.inverse(point + 2 * eps)
        ) / (12 * eps)
        assert_allclose(btf.deriv_inverse(point), deriv_fd_scalar, rtol=1e-5)

    def test_deriv2_inverse(self):
        """Test second inverse-derivative method against finite difference."""
        btf = BeckeRTransform(0.1, 1.1)
        points = np.array([0.2, 1.5, 7.0])
        eps = 1e-4
        # 4th-order centered stencil for the second derivative of inverse(r)
        # f''(r) ~= [-f(r+2h) + 16f(r+h) - 30f(r) + 16f(r-h) - f(r-2h)] / (12h^2)
        deriv2_fd = (
            -btf.inverse(points + 2 * eps)
            + 16 * btf.inverse(points + eps)
            - 30 * btf.inverse(points)
            + 16 * btf.inverse(points - eps)
            - btf.inverse(points - 2 * eps)
        ) / (12 * eps**2)
        assert_allclose(btf.deriv2_inverse(points), deriv2_fd, rtol=1e-4)

        # test for scalar
        point = 0.5
        deriv2_fd_scalar = (
            -btf.inverse(point + 2 * eps)
            + 16 * btf.inverse(point + eps)
            - 30 * btf.inverse(point)
            + 16 * btf.inverse(point - eps)
            - btf.inverse(point - 2 * eps)
        ) / (12 * eps**2)
        assert_allclose(btf.deriv2_inverse(point), deriv2_fd_scalar, rtol=1e-4)

    def test_deriv3_inverse(self):
        """Test third inverse-derivative method against finite difference."""
        btf = BeckeRTransform(0.1, 1.1)
        points = np.array([0.2, 1.5, 7.0])
        eps = 1e-3
        # 4-point centered stencil for the third derivative of inverse(r)
        # f'''(r) ~= [f(r+2h) - 2f(r+h) + 2f(r-h) - f(r-2h)] / (2h^3)
        deriv3_fd = (
            btf.inverse(points + 2 * eps)
            - 2 * btf.inverse(points + eps)
            + 2 * btf.inverse(points - eps)
            - btf.inverse(points - 2 * eps)
        ) / (2 * eps**3)
        assert_allclose(btf.deriv3_inverse(points), deriv3_fd, rtol=1e-4)

        # test for scalar
        point = 0.5
        deriv3_fd_scalar = (
            btf.inverse(point + 2 * eps)
            - 2 * btf.inverse(point + eps)
            + 2 * btf.inverse(point - eps)
            - btf.inverse(point - 2 * eps)
        ) / (2 * eps**3)
        assert_allclose(btf.deriv3_inverse(point), deriv3_fd_scalar, rtol=1e-4)

    def test_becke_inverse_inverse(self):
        """Test inverse of inverse of Becke transformation."""
        btf = BeckeRTransform(0.1, 1.1)
        inv = InverseRTransform(btf)
        inv_inv = inv.inverse(inv.transform(self.array))
        assert_allclose(inv_inv, self.array, atol=1e-7)

    def test_becke_integral(self):
        """Test transform integral."""
        oned = GaussLegendre(20)
        btf = BeckeRTransform(0.00001, 1.0)
        rad = btf.transform_1d_grid(oned)

        def gauss(x):
            return np.exp(-(x**2))

        result = rad.integrate(gauss(rad.points))
        ref_result = np.sqrt(np.pi) / 2
        assert_almost_equal(result, ref_result, decimal=5)

        oned = GaussChebyshev(20)
        rad = btf.transform_1d_grid(oned)
        result = rad.integrate(gauss(rad.points))
        assert_almost_equal(result, ref_result, decimal=3)

    def test_linear_transform(self):
        """Test linear transformation."""
        ltf = LinearFiniteRTransform(0.1, 10)
        self._transform_and_inverse(-1, 1, ltf)

    def test_linear_finite_diff(self):
        """Test finite diff for linear derivs."""
        ltf = LinearFiniteRTransform(0.1, 10)
        self._deriv_finite_diff(-1, 1, ltf)

    def test_linear_inverse(self):
        """Test inverse transform and derivs function."""
        ltf = LinearFiniteRTransform(0.1, 10)
        iltf = InverseRTransform(ltf)
        # transform & inverse
        self._transform_and_inverse(0, 20, iltf)
        # finite diff for derivs
        self._deriv_finite_diff(0, 20, iltf)

    def test_errors_assert(self):
        """Test errors raise."""
        # parameter error
        with self.assertRaises(ValueError):
            BeckeRTransform.find_parameter(np.arange(5), 0.5, 0.1)
        # transform non array type
        with self.assertRaises(TypeError):
            btf = BeckeRTransform(0.1, 1.1)
            btf.transform("dafasdf")
        # inverse init error
        with self.assertRaises(TypeError):
            InverseRTransform(0.5)
        # type error for transform_1d_grid
        with self.assertRaises(TypeError):
            btf = BeckeRTransform(0.1, 1.1)
            btf.transform_1d_grid(np.arange(3))
        with self.assertRaises(ZeroDivisionError):
            btf = BeckeRTransform(0.1, 0)
            itf = InverseRTransform(btf)
            itf._d1(0.5)
        with self.assertRaises(ZeroDivisionError):
            btf = BeckeRTransform(0.1, 0)
            itf = InverseRTransform(btf)
            itf._d1(np.array([0.1, 0.2, 0.3]))

    def test_multiexp_tf(self):
        """Test MultiExp initializaiton."""
        btf = MultiExpRTransform(0.1, 1.2)
        assert btf.R == 1.2
        assert btf.rmin == 0.1

    def test_multiexp_transform(self):
        """Test MultiExp transformation."""
        btf = MultiExpRTransform(0.1, 1.1)
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
        btf = MultiExpRTransform(0.1, 1.1)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-0.9, 0.90, btf)

    def test_multiexp_inverse(self):
        """Test inverse transform basic function."""
        btf = MultiExpRTransform(0.1, 1.1)
        inv = InverseRTransform(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_knowles_tf(self):
        """Test Knowles initializaiton."""
        btf = KnowlesRTransform(0.1, 1.2, 2)
        assert btf.R == 1.2
        assert btf.rmin == 0.1
        assert btf.k == 2

    def test_knowles_transform(self):
        """Test knowles transformation."""
        btf = KnowlesRTransform(0.1, 1.1, 2)
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
        btf = KnowlesRTransform(0.1, 1.1, 2)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-0.9, 0.90, btf)

    def test_knowles_inverse(self):
        """Test inverse transform basic function."""
        btf = KnowlesRTransform(0.1, 1.1, 2)
        inv = InverseRTransform(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_handy_tf(self):
        """Test Handy initializaiton."""
        btf = HandyRTransform(0.1, 1.2, 2)
        assert btf.R == 1.2
        assert btf.m == 2
        assert btf.rmin == 0.1

    def test_handy_transform(self):
        """Test Handy transformation."""
        btf = HandyRTransform(0.1, 1.1, 2)
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
        btf = HandyRTransform(0.1, 1.2, 2)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-0.8, 0.8, btf)

    def test_handy_inverse(self):
        """Test inverse transform basic function."""
        btf = HandyRTransform(0.1, 1.1, 2)
        inv = InverseRTransform(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_handymod_tf(self):
        """Test Handy Mod initializaiton."""
        btf = HandyModRTransform(0.1, 10.0, 2)
        assert btf.m == 2
        assert btf.rmin == 0.1
        assert btf.rmax == 10.0

    def test_handymod_transform(self):
        """Test Handy Mod transformation."""
        btf = HandyModRTransform(0.1, 10.0, 2)
        tf_array = btf.transform(self.array)
        new_array = btf.inverse(tf_array)
        assert_allclose(new_array, self.array)
        # test tf and inverse
        self._transform_and_inverse(-0.9, 0.9, btf)

    def test_handymod_deriv(self):
        """Test Handy Mod transform derivatives with finite diff."""
        btf = HandyModRTransform(0.1, 10.0, 2)
        # call finite diff test function with given arrays
        self._deriv_finite_diff(-0.9, 0.90, btf)

    def test_handymod_inverse(self):
        """Test inverse transform basic function."""
        btf = HandyModRTransform(0.1, 10.0, 2)
        inv = InverseRTransform(btf)
        new_array = inv.transform(btf.transform(self.array))
        assert_allclose(new_array, self.array)

    def test_errors_raises(self):
        """Test errors raise."""
        with self.assertRaises(ValueError):
            KnowlesRTransform(0.1, 10.0, 0)
        with self.assertRaises(ValueError):
            HandyRTransform(0.1, 10.0, 0)
        with self.assertRaises(ValueError):
            HandyModRTransform(0.1, 10.0, 0)
        with self.assertRaises(ValueError):
            HandyModRTransform(10.0, 1.0, 2)

    def test_domain(self):
        """Test domain errors."""
        rad = UniformInteger(10)
        with self.assertRaises(ValueError):
            tf = BeckeRTransform(0.1, 1.2)
            tf.transform_1d_grid(rad)
        with self.assertRaises(ValueError):
            tf = HandyModRTransform(0.1, 10.0, 2)
            tf.transform_1d_grid(rad)
        with self.assertRaises(ValueError):
            tf = KnowlesRTransform(0.1, 1.2, 2)
            tf.transform_1d_grid(rad)
        with self.assertRaises(ValueError):
            tf = LinearFiniteRTransform(0.1, 10)
            tf.transform_1d_grid(rad)
        with self.assertRaises(ValueError):
            tf = MultiExpRTransform(0.1, 1.2)
            tf.transform_1d_grid(rad)
