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
from grid.basegrid import OneDGrid
from grid.onedgrid import GaussChebyshevLobatto
from grid.protransform import (
    CubicProTransform,
    _PromolParams,
    _pad_coeffs_exps_with_zeros,
    _transform_coordinate,
)

import numpy as np
from numpy.testing import assert_allclose

import pytest

from scipy.optimize import approx_fprime
from scipy.special import erf


class TestTwoGaussianDiffCenters:
    r"""Test a Sum of Two Gaussian function against analytic formulas and numerical procedures."""

    def setUp(self, ss=0.1, return_obj=False, add_boundary=True):
        r"""Set up a two parameter Gaussian function."""
        c = np.array([[5.0], [10.0]])
        e = np.array([[2.0], [3.0]])
        coord = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
        params = _PromolParams(c, e, coord, dim=3)
        if return_obj:
            num_pts = int(2 / ss) + 1
            weights = np.array([2.0 / (num_pts - 2)] * num_pts)
            if add_boundary:
                l_bnd = -1.0
                end_point = True
            else:
                l_bnd = -0.99
                end_point = False
            oned = OneDGrid(
                np.linspace(l_bnd, 1, num=num_pts, endpoint=end_point),
                weights,
                domain=(-1, 1),
            )

            obj = CubicProTransform(
                [oned, oned, oned],
                params.c_m,
                params.e_m,
                params.coords,
            )
            return params, obj
        return params

    def promolecular(self, x, y, z, params):
        r"""Hard-Code the promolecular density."""
        # Promolecular in UniformProTransform class uses einsum, this tests it against that.
        # Also could be used for integration tests.
        cm, em, coords = params.c_m, params.e_m, params.coords
        promol = 0.0
        for i, coeffs in enumerate(cm):
            xc, yc, zc = coords[i]
            for j, coeff in enumerate(coeffs):
                distance = (x - xc) ** 2.0 + (y - yc) ** 2.0 + (z - zc) ** 2.0
                promol += np.sum(coeff * np.exp(-em[i, j] * distance))
        return promol

    def test_promolecular_density(self):
        r"""Test Promolecular Density function against hard-coded."""
        grid = np.array(
            [
                [-1.0, 2.0, 3.0],
                [5.0, 10.0, -1.0],
                [0.0, 0.0, 0.0],
                [3.0, 2.0, 1.0],
                [10.0, 20.0, 30.0],
                [0.0, 10.0, 0.2],
            ]
        )
        params = self.setUp()

        true_ans = []
        for pt in grid:
            x, y, z = pt
            true_ans.append(self.promolecular(x, y, z, params))

        desired = params.promolecular(grid)
        assert np.all(np.abs(np.array(true_ans) - desired) < 1e-8)

    @pytest.mark.parametrize(
        "pts, add_boundary", [[np.arange(-5.0, 5.0, 0.5), False], [np.arange(-5.0, 5.0, 0.5), True]]
    )
    def test_transforming_x_against_formula(self, pts, add_boundary):
        r"""Test transformming the X-transformation against analytic formula."""
        for pt in pts:
            true_ans = _transform_coordinate([pt], 0, self.setUp(add_boundary=add_boundary))

            def formula_transforming_x(x):
                r"""Return closed form formula for transforming x coordinate."""
                first_factor = (5.0 * np.pi**1.5 / (4 * 2**0.5)) * (erf(2**0.5 * (x - 1)) + 1.0)

                sec_fac = ((10.0 * np.pi**1.5) / (6.0 * 3**0.5)) * (erf(3.0**0.5 * (x - 2)) + 1.0)

                ans = (first_factor + sec_fac) / (
                    5.0 * (np.pi / 2) ** 1.5 + 10.0 * (np.pi / 3.0) ** 1.5
                )
                return -1.0 + 2.0 * ans

            assert np.abs(true_ans - formula_transforming_x(pt)) < 1e-8

    @pytest.mark.parametrize(
        "pts_xy, add_boundary",
        [
            [np.random.uniform(-10.0, 10.0, size=(100, 2)), False],
            [np.random.uniform(-10.0, 10.0, size=(100, 2)), True],
        ],
    )
    def test_transforming_y_against_formula(self, pts_xy, add_boundary):
        r"""Test transforming the Y-transformation against analytic formula."""
        for x, y in pts_xy:
            true_ans = _transform_coordinate([x, y], 1, self.setUp(add_boundary=add_boundary))

            def formula_transforming_y(x, y):
                r"""Return closed form formula for transforming y coordinate."""
                fac1 = 5.0 * np.sqrt(np.pi / 2.0) * np.exp(-2.0 * (x - 1) ** 2)
                fac1 *= np.sqrt(np.pi) * (erf(2.0**0.5 * (y - 2)) + 1.0) / (2.0 * np.sqrt(2.0))
                fac2 = 10.0 * np.sqrt(np.pi / 3.0) * np.exp(-3.0 * (x - 2.0) ** 2.0)
                fac2 *= np.sqrt(np.pi) * (erf(3.0**0.5 * (y - 2)) + 1.0) / (2.0 * np.sqrt(3.0))
                num = fac1 + fac2

                dac1 = 5.0 * (np.pi / 2.0) * np.exp(-2.0 * (x - 1.0) ** 2.0)
                dac2 = 10.0 * (np.pi / 3.0) * np.exp(-3.0 * (x - 2.0) ** 2.0)
                den = dac1 + dac2
                return -1.0 + 2.0 * (num / den)

            assert np.abs(true_ans - formula_transforming_y(x, y)) < 1e-8

    @pytest.mark.parametrize(
        "pts, add_boundary",
        [
            [np.random.uniform(-10.0, 10.0, size=(100, 3)), False],
            [np.random.uniform(-10.0, 10.0, size=(100, 3)), True],
        ],
    )
    def test_transforming_z_against_formula(self, pts, add_boundary):
        r"""Test transforming the Z-transformation against analytic formula."""
        params, obj = self.setUp(ss=0.5, return_obj=True, add_boundary=add_boundary)

        def formula_transforming_z(x, y, z):
            r"""Return closed form formula for transforming z coordinate."""
            a1, a2, a3 = (x - 1.0), (y - 2.0), (z - 3.0)
            erfx = erf(2.0**0.5 * a3) + 1.0
            fac1 = 5.0 * np.exp(-2.0 * (a1**2.0 + a2**2.0)) * erfx * np.pi**0.5 / (2.0 * 2.0**0.5)
            b1, b2, b3 = (x - 2.0), (y - 2.0), (z - 2.0)
            erfy = erf(3.0**0.5 * b3) + 1.0
            fac2 = 10.0 * np.exp(-3.0 * (b1**2.0 + b2**2.0)) * erfy * np.pi**0.5 / (2.0 * 3.0**0.5)
            den = 5.0 * (np.pi / 2.0) ** 0.5 * np.exp(-2.0 * (a1**2.0 + a2**2.0))
            den += 10.0 * (np.pi / 3.0) ** 0.5 * np.exp(-3.0 * (b1**2.0 + b2**2.0))
            return -1.0 + 2.0 * (fac1 + fac2) / den

        for x, y, z in pts:
            true_ans = formula_transforming_z(x, y, z)
            # Test function
            actual = _transform_coordinate([x, y, z], 2, params)
            assert np.abs(true_ans - actual) < 1e-8

            # Test Method
            actual = obj.transform(np.array([x, y, z]))[2]
            assert np.abs(true_ans - actual) < 1e-8

    def test_transforming_simple_grid(self):
        r"""Test transforming a grid that only contains one non-boundary point."""
        ss = 1.0
        params, obj = self.setUp(ss, return_obj=True)
        num_pt = int(2.0 / ss) + 1  # number of points in one-direction.
        assert obj.points.shape == (num_pt**3, 3)
        non_boundary_pt_index = num_pt**2 + num_pt + 1
        real_pt = obj.points[non_boundary_pt_index]
        # Test that this point is not the boundary.
        assert real_pt[0] != np.inf
        assert real_pt[1] != np.inf
        assert real_pt[2] != np.inf
        # Test that converting the point back to unit cube gives [0.5, 0.5, 0.5].
        for i_var in range(0, 3):
            transf = _transform_coordinate(real_pt, i_var, obj.promol)
            assert np.abs(transf) < 1e-5
        # Test that all other points are indeed boundary points.
        all_nans = np.delete(obj.points, non_boundary_pt_index, axis=0)
        assert np.all(np.any(np.isinf(all_nans), axis=1))

    def test_transforming_with_inverse_transformation_is_identity(self):
        r"""Test transforming with inverse transformation is identity."""
        # Note that for points far away from the promolecular gets mapped to nan.
        # So this isn't really the inverse, in the mathematical sense.
        param, obj = self.setUp(0.5, return_obj=True)

        pt = np.array([1, 2, 3], dtype=np.float64)
        transf = obj.transform(pt)
        reverse = obj.inverse(transf)
        assert np.all(np.abs(reverse - pt) < 1e-10)

    def test_integrating_itself(self):
        r"""Test integrating the very same promolecular density used for transformation."""
        params, obj = self.setUp(ss=0.2, return_obj=True)
        promol = []
        for pt in obj.points:
            promol.append(self.promolecular(pt[0], pt[1], pt[2], params))
        promol = np.array(promol, dtype=np.float64)
        desired = obj.prointegral
        actual = obj.integrate(promol)
        assert np.abs(actual - desired) < 1e-8

        actual = obj.integrate(promol, trick=True)
        assert np.abs(actual - desired) < 1e-8

    @pytest.mark.parametrize("pts", [np.arange(-5.0, 5.0, 0.75)])
    def test_derivative_tranformation_x_finite_difference(self, pts):
        r"""Test the derivative of X-transformation against finite-difference."""
        params, obj = self.setUp(ss=0.2, return_obj=True)
        for pt in pts:
            pt = np.array([pt, 2.0, 3.0])

            actual = obj.jacobian(pt)

            def tranformation_x(pt):
                return _transform_coordinate(pt, 0, params)

            grad = approx_fprime([pt[0]], tranformation_x, 1e-6)
            assert np.abs(grad - actual[0, 0]) < 1e-4

    @pytest.mark.parametrize("pts_xy", [np.random.uniform(-5.0, 5.0, size=(100, 2))])
    def test_derivative_tranformation_y_finite_difference(self, pts_xy):
        r"""Test the derivative of Y-transformation against finite-difference."""
        params, obj = self.setUp(ss=0.2, return_obj=True)
        for x, y in pts_xy:
            actual = obj.jacobian(np.array([x, y, 3.0]))

            def tranformation_y(pt):
                return _transform_coordinate([x, pt[0]], 1, params)

            grad = approx_fprime([y], tranformation_y, 1e-8)
            assert np.abs(grad - actual[1, 1]) < 1e-5

            def transformation_y_wrt_x(pt):
                return _transform_coordinate([pt[0], y], 1, params)

            h = 1e-8
            deriv = np.imag(transformation_y_wrt_x([complex(x, h)])) / h
            assert np.abs(deriv - actual[1, 0]) < 1e-4

    @pytest.mark.parametrize("pts", [np.random.uniform(-3.0, 3.0, size=(100, 3))])
    def test_derivative_tranformation_z_finite_difference(self, pts):
        r"""Test the derivative of Z-transformation against finite-difference."""
        params, obj = self.setUp(ss=0.2, return_obj=True)
        for x, y, z in pts:
            actual = obj.jacobian(np.array([x, y, z]))

            def tranformation_z(pt):
                return _transform_coordinate([x, y, pt[0]], 2, params)

            grad = approx_fprime([z], tranformation_z, 1e-8)
            assert np.abs(grad - actual[2, 2]) < 1e-5

            def transformation_z_wrt_y(pt):
                return _transform_coordinate([x, pt[0], z], 2, params)

            deriv = approx_fprime([y], transformation_z_wrt_y, 1e-8)
            assert np.abs(deriv - actual[2, 1]) < 1e-4

            def transformation_z_wrt_x(pt):
                a = _transform_coordinate([pt[0], y, z], 2, params)
                return a

            h = 1e-8
            deriv = np.imag(transformation_z_wrt_x([complex(x, h)])) / h

            assert np.abs(deriv - actual[2, 0]) < 1e-4

    @pytest.mark.parametrize("pts", [np.random.uniform(-3.0, 3.0, size=(100, 3))])
    def test_steepest_ascent_direction_with_numerics(self, pts):
        r"""Test steepest-ascent direction match in real and theta space.

        The function to test is x^2 + y^2 + z^2.
        """
        params, obj = self.setUp(ss=0.2, return_obj=True)

        def grad(pt):
            # Gradient of x^2 + y^2 + z^2.
            return np.array([2.0 * pt[0], 2.0 * pt[1], 2.0 * pt[2]])

        for x, y, z in pts:
            # Take a step in real-space.
            pt = np.array([x, y, z])
            grad_pt = grad(pt)
            step = 1e-8
            pt_step = pt + grad_pt * step

            # Convert the steps in theta space and calculate finite-difference gradient.
            transf = obj.transform(pt)
            transf_step = obj.transform(pt_step)
            grad_finite = (transf_step - transf) / step

            # Test the actual actual.
            actual = obj.steepest_ascent_theta(pt, grad_pt)
            assert np.all(np.abs(actual - grad_finite) < 1e-4)

    def test_second_derivative_of_theta_x_dx(self):
        r"""Test the second derivative of d(theta_x)/d(x) ."""
        params, obj = self.setUp(ss=0.2, return_obj=True)
        x, y, z = 0.0, 0.0, 0.0
        actual = obj.hessian(np.array([x, y, z]))

        # test second derivative of theta_x wrt to dx dx.
        def tranformation_x(pt):
            return obj.jacobian([pt[0], y, z])[0, 0]

        dthetax_dx_dx = approx_fprime([x], tranformation_x, 1e-8)
        assert np.abs(dthetax_dx_dx - actual[0, 0, 0]) < 1e-5
        # Since theta_x only depends on x, then all other elements
        # are zero.
        for i in range(0, 3):
            for j in range(0, 3):
                if i != j:
                    assert np.abs(actual[0, i, j]) < 1e-10

    @pytest.mark.parametrize("pts", [np.random.uniform(-3.0, 3.0, size=(100, 3))])
    def test_second_derivative_of_theta_y(self, pts):
        r"""Test the second derivative of d(theta_y)."""
        params, obj = self.setUp(ss=0.2, return_obj=True)
        for x, y, z in pts:
            actual = obj.hessian(np.array([x, y, z]))

            # test second derivative of theta_y wrt to dy dy.
            def dtheta_y_dy(pt):
                return obj.jacobian([x, pt[0], z])[1, 1]

            dthetay_dy_dy = approx_fprime([y], dtheta_y_dy, 1e-8)
            assert np.abs(dthetay_dy_dy - actual[1, 1, 1]) < 1e-5

            # test second derivative of d(theta_y)/d(y) wrt dx.
            def dtheta_y_dy(pt):
                return obj.jacobian([pt[0], y, z])[1, 1]

            dthetay_dy_dx = approx_fprime([x], dtheta_y_dy, 1e-8)
            assert np.abs(dthetay_dy_dx - actual[1, 1, 0]) < 1e-5

            # test second derivative of d(theta_y)/(dx) wrt dy
            def dtheta_y_dx(pt):
                return obj.jacobian([x, pt[0], z])[1, 0]

            dtheta_y_dx_dy = approx_fprime([y], dtheta_y_dx, 1e-8)
            assert np.abs(dtheta_y_dx_dy - actual[1, 0, 1]) < 1e-4

            # test second derivative of d(theta_y)/(dx) wrt dx
            def dtheta_y_dx(pt):
                return obj.jacobian([pt[0], y, z])[1, 0]

            dtheta_y_dx_dx = approx_fprime([x], dtheta_y_dx, 1e-8)
            assert np.abs(dtheta_y_dx_dx - actual[1, 0, 0]) < 1e-5

            # Test other elements are all zeros.
            for i in range(0, 3):
                assert np.abs(actual[1, 2, i]) < 1e-10
            assert np.abs(actual[1, 0, 2]) < 1e-10
            assert np.abs(actual[1, 1, 2]) < 1e-10

    @pytest.mark.parametrize("pts", [np.random.uniform(-3.0, 3.0, size=(100, 3))])
    def test_second_derivative_of_theta_z(self, pts):
        r"""Test the second derivative of d(theta_z)."""
        params, obj = self.setUp(ss=0.2, return_obj=True)
        for x, y, z in pts:
            actual = obj.hessian(np.array([x, y, z]))

            # test second derivative of theta_z wrt to dz dz.
            def dtheta_z_dz(pt):
                return obj.jacobian([x, y, pt[0]])[2, 2]

            dthetaz_dz_dz = approx_fprime([z], dtheta_z_dz, 1e-8)
            assert np.abs(dthetaz_dz_dz - actual[2, 2, 2]) < 1e-5

            # test second derivative of theta_z wrt to dz dy.
            def dtheta_z_dz(pt):
                return obj.jacobian([x, pt[0], z])[2, 2]

            dthetaz_dz_dy = approx_fprime([y], dtheta_z_dz, 1e-8)
            assert np.abs(dthetaz_dz_dy - actual[2, 2, 1]) < 1e-5

            # test second derivative of theta_z wrt to dz dx.
            def dtheta_z_dz(pt):
                return obj.jacobian([pt[0], y, z])[2, 2]

            dthetaz_dz_dx = approx_fprime([x], dtheta_z_dz, 1e-8)
            assert np.abs(dthetaz_dz_dx - actual[2, 2, 0]) < 1e-5

            # test second derivative of theta_z wrt to dy dz.
            def dtheta_z_dy(pt):
                return obj.jacobian([x, y, pt[0]])[2, 1]

            dthetaz_dy_dz = approx_fprime([z], dtheta_z_dy, 1e-8)
            assert np.abs(dthetaz_dy_dz - actual[2, 1, 2]) < 1e-5

            # test second derivative of theta_z wrt to dy dy.
            def dtheta_z_dy(pt):
                return obj.jacobian([x, pt[0], z])[2, 1]

            dthetaz_dy_dy = approx_fprime([y], dtheta_z_dy, 1e-8)
            assert np.abs(dthetaz_dy_dy - actual[2, 1, 1]) < 1e-5

            # test second derivative of theta_z wrt to dy dx.
            def dtheta_z_dy(pt):
                return obj.jacobian([pt[0], y, z])[2, 1]

            dthetaz_dy_dx = approx_fprime([x], dtheta_z_dy, 1e-8)
            assert np.abs(dthetaz_dy_dx - actual[2, 1, 0]) < 1e-5

            # test second derivative of theta_z wrt to dx dz.
            def dtheta_z_dx(pt):
                return obj.jacobian([x, y, pt[0]])[2, 0]

            dthetaz_dx_dz = approx_fprime([z], dtheta_z_dx, 1e-8)
            assert np.abs(dthetaz_dx_dz - actual[2, 0, 2]) < 1e-5

            # test second derivative of theta_z wrt to dx dy.
            def dtheta_z_dx(pt):
                return obj.jacobian([x, pt[0], z])[2, 0]

            dthetaz_dx_dy = approx_fprime([y], dtheta_z_dx, 1e-8)
            assert np.abs(dthetaz_dx_dy - actual[2, 0, 1]) < 1e-5

            # test second derivative of theta_z wrt to dx dx.
            def dtheta_z_dx(pt):
                return obj.jacobian([pt[0], y, z])[2, 0]

            dthetaz_dx_dx = approx_fprime([x], dtheta_z_dx, 1e-8)
            assert np.abs(dthetaz_dx_dx - actual[2, 0, 0]) < 1e-5


class TestOneGaussianAgainstNumerics:
    r"""Tests With Numerical Integration of a One Gaussian function."""

    def setUp(self, ss=0.1, return_obj=False):
        r"""Return a one Gaussian example and its UniformProTransform object."""
        c = np.array([[5.0]])
        e = np.array([[2.0]])
        coord = np.array([[1.0, 2.0, 3.0]])
        params = _PromolParams(c, e, coord, dim=3)
        if return_obj:
            num_pts = int(1 / ss) + 1
            weights = np.array([2.0 / (num_pts - 2)] * num_pts)
            oned_x = OneDGrid(
                np.linspace(-1, 1, num=num_pts, endpoint=True), weights, domain=(-1, 1)
            )

            obj = CubicProTransform([oned_x, oned_x, oned_x], params.c_m, params.e_m, params.coords)
            return params, obj
        return params

    @pytest.mark.parametrize("pts", [np.arange(-5.0, 5.0, 0.5)])
    def test_transforming_x_against_numerics(self, pts):
        r"""Test transforming X against numerical algorithms."""
        for pt in pts:
            true_ans = _transform_coordinate([pt], 0, self.setUp())

            def promolecular_in_x(grid, every_grid):
                r"""Construct the formula of promolecular for integration."""
                promol_x = 5.0 * np.exp(-2.0 * (grid - 1.0) ** 2.0)
                promol_x_all = 5.0 * np.exp(-2.0 * (every_grid - 1.0) ** 2.0)
                return promol_x, promol_x_all

            grid = np.arange(-4.0, pt, 0.000005)  # Integration till a x point
            every_grid = np.arange(-5.0, 10.0, 0.00001)  # Full Integration
            promol_x, promol_x_all = promolecular_in_x(grid, every_grid)

            # Integration over y and z cancel out from numerator and denominator.
            actual = -1.0 + 2.0 * np.trapz(promol_x, grid) / np.trapz(promol_x_all, every_grid)
            assert np.abs(true_ans - actual) < 1e-5

    @pytest.mark.parametrize("pts_xy", [np.random.uniform(-10.0, 10.0, size=(100, 2))])
    def test_transforming_y_against_numerics(self, pts_xy):
        r"""Test transformation y against numerical algorithms."""

        def promolecular_in_y(grid, every_grid):
            r"""Construct the formula of promolecular for integration."""
            promol_y = 5.0 * np.exp(-2.0 * (grid - 2.0) ** 2.0)
            promol_y_all = 5.0 * np.exp(-2.0 * (every_grid - 2.0) ** 2.0)
            return promol_y_all, promol_y

        for x, y in pts_xy:
            true_ans = _transform_coordinate([x, y], 1, self.setUp())

            grid = np.arange(-5.0, y, 0.000001)  # Integration till a x point
            every_grid = np.arange(-5.0, 10.0, 0.00001)  # Full Integration
            promol_y_all, promol_y = promolecular_in_y(grid, every_grid)

            # Integration over z cancel out from numerator and denominator.
            # Further, gaussian at a point does too.
            actual = -1.0 + 2.0 * np.trapz(promol_y, grid) / np.trapz(promol_y_all, every_grid)
            assert np.abs(true_ans - actual) < 1e-5

    @pytest.mark.parametrize("pts", [np.random.uniform(-10.0, 10.0, size=(100, 3))])
    def test_transforming_z_against_numerics(self, pts):
        r"""Test transforming Z against numerical algorithms."""

        def promolecular_in_z(grid, every_grid):
            r"""Construct the formula of promolecular for integration."""
            promol_z = 5.0 * np.exp(-2.0 * (grid - 3.0) ** 2.0)
            promol_z_all = 5.0 * np.exp(-2.0 * (every_grid - 3.0) ** 2.0)
            return promol_z_all, promol_z

        for x, y, z in pts:
            grid = np.arange(-5.0, z, 0.00001)  # Integration till a x point

            every_grid = np.arange(-5.0, 10.0, 0.00001)  # Full Integration
            promol_z_all, promol_z = promolecular_in_z(grid, every_grid)

            actual = -1.0 + 2.0 * np.trapz(promol_z, grid) / np.trapz(promol_z_all, every_grid)
            true_ans = _transform_coordinate([x, y, z], 2, self.setUp())
            assert np.abs(true_ans - actual) < 1e-4

    @pytest.mark.parametrize("pts", [np.random.uniform(-3.0, 3.0, size=(100, 3))])
    def test_jacobian(self, pts):
        r"""Test that the jacobian of the transformation."""
        params, obj = self.setUp(ss=0.2, return_obj=True)
        for x, y, z in pts:
            actual = obj.jacobian(np.array([x, y, z]))

            # assert lower-triangular component is zero.
            assert np.abs(actual[1, 0]) < 1e-5
            assert np.abs(actual[2, 0]) < 1e-5
            assert np.abs(actual[2, 1]) < 1e-5

            # test derivative wrt to x
            def tranformation_x(pt):
                return _transform_coordinate([pt[0], y, z], 0, params)

            grad = approx_fprime([x], tranformation_x, 1e-8)
            assert np.abs(grad - actual[0, 0]) < 1e-5

            # test derivative wrt to y
            def tranformation_y(pt):
                return _transform_coordinate([x, pt[0]], 1, params)

            grad = approx_fprime([y], tranformation_y, 1e-8)
            assert np.abs(grad - actual[1, 1]) < 1e-5

            # Test derivative wrt to z
            def tranformation_z(pt):
                return _transform_coordinate([x, y, pt[0]], 2, params)

            grad = approx_fprime([z], tranformation_z, 1e-8)
            assert np.abs(grad - actual[2, 2]) < 1e-5


class TestInterpolation:
    r"""Test Interpolation Methods and Cubic Grid Utility Functions."""

    def setUp(self, ss=0.1):
        r"""Set up a simple one Gaussian function with uniform cubic grid."""
        c = np.array([[5.0]])
        e = np.array([[2.0]])
        coord = np.array([[1.0, 2.0, 3.0]])
        params = _PromolParams(c, e, coord, dim=3)

        num_pts = int(2 / ss) + 1
        weights = np.array([2.0 / (num_pts - 2)] * num_pts)
        oned_x = OneDGrid(
            np.linspace(-1.0, 1.0, num=num_pts, endpoint=True), weights, domain=(-1, 1)
        )

        obj = CubicProTransform([oned_x, oned_x, oned_x], params.c_m, params.e_m, params.coords)
        return params, obj, [oned_x, oned_x, oned_x]

    def test_interpolate_cubic_function(self):
        r"""Interpolate a cubic function."""
        param, obj, oned_grids = self.setUp(ss=0.08)

        # Function to interpolate.
        def func(pts):
            return (pts[:, 0] - 1.0) ** 3.0 + (pts[:, 1] - 2.0) ** 3.0 + (pts[:, 2] - 3.0) ** 3.0

        # Test over a grid that is very close to the nucleus.
        grid = np.vstack(
            (
                np.random.uniform(1.0, 1.5, (100,)).T,
                np.random.uniform(1.5, 2.5, (100,)).T,
                np.random.uniform(2.5, 3.5, (100,)).T,
            )
        ).T
        actuals = obj.interpolate(grid, func(obj.points), oned_grids, use_log=False)
        desired = func(grid)
        assert_allclose(desired, actuals, atol=1e-3, rtol=1e-4)

    def test_interpolate_gaussian(self):
        r"""Interpolate a Gaussian function with use_log set to True."""
        param, obj, oned_grids = self.setUp(ss=0.08)

        # Function to interpolate.
        def func(pts, alpha=2.0):
            return np.exp(-alpha * np.linalg.norm(pts - param.coords[0], axis=1) ** 2.0)

        # Test over a grid. Pytest isn't used for effiency reasons.
        # TODO: the grid points need to be close to center to achieve good accuracy.
        real_grid = np.vstack(
            (
                np.random.uniform(0.75, 1.25, (100,)).T,
                np.random.uniform(0.75, 2.25, (100,)).T,
                np.random.uniform(3.75, 3.25, (100,)).T,
            )
        ).T
        actuals = obj.interpolate(real_grid, func(obj.points), oned_grids, use_log=True)
        desired = func(real_grid)
        assert_allclose(desired, actuals, atol=1e-1)

    def test_interpolate_derivative_cubic_function(self):
        r"""Interpolate the derivative of some simple function."""
        param, obj, oned_grids = self.setUp(ss=0.08)

        # Function to interpolate.
        def func(pts):
            return (pts[:, 0] - 1.0) * (pts[:, 1] - 2.0) * (pts[:, 2] - 3.0)

        def derivative(pts):
            return np.vstack(
                [
                    (pts[:, 1] - 2.0) * (pts[:, 2] - 3.0),
                    (pts[:, 0] - 1.0) * (pts[:, 2] - 3.0),
                    (pts[:, 0] - 1.0) * (pts[:, 1] - 2.0),
                ]
            ).T

        # Test over a grid. Pytest isn't used for effiency reasons.
        grid = np.vstack(
            (
                np.random.uniform(1.0, 1.5, (100,)).T,
                np.random.uniform(1.5, 2.5, (100,)).T,
                np.random.uniform(2.5, 3.5, (100,)).T,
            )
        ).T
        actual = obj.interpolate(grid, func(obj.points), oned_grids, nu=1)
        desired = derivative(grid)
        assert_allclose(actual, desired, atol=1e-2)

    def test_interpolate_derivative_cubic_function2(self):
        r"""Interpolate the derivative of some simple function."""
        param, obj, oned_grids = self.setUp(ss=0.08)

        # Function to interpolate.
        def func(pts):
            return (pts[:, 0] - 1.0) ** 2.0 * (pts[:, 1] - 2.0) ** 2.0 * (pts[:, 2] - 3.0) ** 2.0

        def derivative(pts):
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            return np.vstack(
                [
                    2.0 * (x - 1.0) * (y - 2.0) ** 2.0 * (z - 3.0) ** 2.0,
                    2.0 * (x - 1.0) ** 2.0 * (y - 2.0) * (z - 3.0) ** 2.0,
                    2.0 * (x - 1.0) ** 2.0 * (y - 2.0) ** 2.0 * (z - 3.0),
                ]
            ).T

        # Test over a grid. Pytest isn't used for effiency reasons.
        grid = np.vstack(
            (
                np.random.uniform(1.0, 1.5, (100,)).T,
                np.random.uniform(1.5, 2.5, (100,)).T,
                np.random.uniform(2.5, 3.5, (100,)).T,
            )
        ).T
        actual = obj.interpolate(grid, func(obj.points), oned_grids, nu=1, use_log=False)
        desired = derivative(grid)
        assert_allclose(actual, desired, atol=1e-2)
        # Test with logarithm set to True
        actual = obj.interpolate(grid, func(obj.points), oned_grids, nu=1, use_log=True)
        assert_allclose(actual, desired, atol=1e-2)


class TestIntegration:
    r"""
    Only one integration test as of this moment.

    Choose to make it, it's own seperate class since many choices of oned grid is possible.

    """

    def setUp_one_gaussian(self, ss=0.03):
        r"""Return a one Gaussian example and its UniformProTransform object."""
        c = np.array([[5.0]])
        e = np.array([[2.0]])
        coord = np.array([[1.0, 2.0, 3.0]])
        params = _PromolParams(c, e, coord, dim=3)
        num_pts = int(1 / ss) + 1
        oned_x = GaussChebyshevLobatto(num_pts)
        obj = CubicProTransform([oned_x, oned_x, oned_x], params.c_m, params.e_m, params.coords)
        return params, obj

    def test_integration_perturbed_gaussian_with_promolecular_trick(self):
        r"""Test integration of a slightly perturbed function."""
        # Only Measured against one decimal place and very similar exponent.
        _, obj = self.setUp_one_gaussian(ss=0.03)

        # Gaussian exponent is slightly perturbed from 2.
        exponent = 2.01

        def gaussian(grid):
            return 5.0 * np.exp(
                -exponent * np.sum((grid - np.array([1.0, 2.0, 3.0])) ** 2.0, axis=1)
            )

        func_vals = gaussian(obj.points)
        desired = 5.0 * np.sqrt(np.pi / exponent) ** 3.0
        actual = obj.integrate(func_vals, trick=True)

        assert np.abs(actual - desired) < 1e-3


def test_padding_arrays():
    r"""Test different array sizes are correctly padded."""
    coeff = np.array([[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [5.0]], dtype=object)
    exps = np.array([[4.0, 5.0], [5.0, 6.0, 7.0, 8.0], [9.0]], dtype=object)
    coeff_pad, exps_pad = _pad_coeffs_exps_with_zeros(coeff, exps)
    coeff_desired = np.array(
        [[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [5.0, 0.0, 0.0, 0.0]], dtype=object
    )
    np.testing.assert_array_equal(coeff_desired, coeff_pad)
    exp_desired = np.array(
        [[4.0, 5.0, 0.0, 0.0], [5.0, 6.0, 7.0, 8.0], [9.0, 0.0, 0.0, 0.0]], dtype=object
    )
    np.testing.assert_array_equal(exp_desired, exps_pad)
