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

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from grid.onedgrid import GaussLegendre, UniformInteger
from grid.rtransform import (
    BaseTransform,
    BeckeRTransform,
    HandyModRTransform,
    HandyRTransform,
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
    # Evaluate on a flattened array to match the 1D input contract of radial transforms.
    vals = fcn(pts.reshape(-1)).reshape(pts.shape)
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


@pytest.mark.parametrize("transform_class, kwargs", VALID_TRANSFORM_CASES)
def test_transform_init(transform_class, kwargs):
    "Test transform class initialization stores constructor parameters as attributes."
    t = transform_class(**kwargs)

    for k, v in kwargs.items():
        assert getattr(t, k) == v


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


@pytest.mark.parametrize("transform_class, kwargs", VALID_TRANSFORM_CASES)
def test_transform_integral_consistency(transform_class, kwargs):
    """Test change-of-variables consistency between reference and transformed quadrature grids."""
    N = 50
    transform = transform_class(**kwargs)

    onedgrid = GaussLegendre(N)
    rgrid = transform.transform_1d_grid(onedgrid)

    def f(r):
        return np.exp(-(r**2))

    # integral in reference-space quadrature applying change-of-variables: f(r(x)) * dr/dx
    ref_int = onedgrid.integrate(f(rgrid.points) * transform.deriv(onedgrid.points))

    # integral in physical space (Jacobian already absorbed into grid weights)
    rgrid_int = rgrid.integrate(f(rgrid.points))

    assert_allclose(ref_int, rgrid_int, atol=1e-10, rtol=1e-6)


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
