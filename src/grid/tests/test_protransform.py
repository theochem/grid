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
r"""
Tests for Cubic Promolecular transformation.

Tests
-----
TestTwoGaussianDiffCenters :
    Test Transformation of Two Gaussian promolecular against different methods both analytical
    and numerics.
TestOneGaussianAgainstNumerics :
    Test a single Gaussian against numerical integration/differentiation.

"""

import numpy as np
from scipy.special import erf
from scipy.optimize import approx_fprime

import pytest

from grid.protransform import (
    CubicProTransform, PromolParams, transform_coordinate, _pad_coeffs_exps_with_zeros
)


class TestTwoGaussianDiffCenters:
    r"""
    Test a Sum of Two Gaussian function against analytic formulas and numerical procedures.
    """
    def setUp(self, ss=0.1, return_obj=False):
        c = np.array([[5.], [10.]])
        e = np.array([[2.], [3.]])
        coord = np.array([[1., 2., 3.], [2., 2., 2.]])
        params = PromolParams(c, e, coord, dim=3, pi_over_exponents=np.sqrt(np.pi / e))
        if return_obj:
            num_pts = int(1 / ss) + 1
            weights = np.array([((1. / (num_pts - 2)))**3.] * num_pts**3)
            obj = CubicProTransform([ss] * 3, weights, params.c_m, params.e_m, params.coords)
            return params,  obj
        return params

    def promolecular(self, x, y, z, params):
        # Promolecular in CubicProTransform class uses einsum, this tests it against that.
        # Also could be used for integration tests.
        cm, em, coords, _, _ = params
        promol = 0.
        for i, coeffs in enumerate(cm):
            xc, yc, zc = coords[i]
            for j, coeff in enumerate(coeffs):
                distance = ((x - xc)**2. + (y - yc)**2. + (z - zc)**2.)
                promol += np.sum(coeff * np.exp(- em[i, j] * distance))
        return promol

    def test_promolecular_density(self):
        grid = np.array([[-1., 2., 3.], [5., 10., -1.], [0., 0., 0.],
                         [3., 2., 1.], [10., 20., 30.], [0., 10., 0.2]])
        params, obj = self.setUp(return_obj=True)

        true_ans = []
        for pt in grid:
            x, y, z = pt
            true_ans.append(self.promolecular(x, y, z, params))

        desired = obj._promolecular(grid)
        assert np.all(np.abs(np.array(true_ans) - desired) < 1e-8)

    @pytest.mark.parametrize("pt", np.arange(-5., 5., 0.5))
    def test_transforming_x_against_formula(self, pt):
        def formula_transforming_x(x):
            r"""Closed form formula for transforming x coordinate."""
            first_factor = (5. * np.pi ** 1.5 / (4 * 2 ** 0.5)) * (erf(2 ** 0.5 * (x - 1)) + 1.)
            sec_fac = ((10. * np.pi ** 1.5) / (6. * 3 ** 0.5)) * (erf(3. ** 0.5 * (x - 2)) + 1.)
            return (first_factor + sec_fac) / (5. * (np.pi / 2) ** 1.5 + 10. * (np.pi / 3.) ** 1.5)

        true_ans = transform_coordinate([pt], 0, self.setUp())
        assert np.abs(true_ans - formula_transforming_x(pt)) < 1e-8

    @pytest.mark.parametrize("pt", np.arange(-5., 5., 0.75))
    def test_derivative_transforming_x_with_finite_difference(self, pt):
        # Unfortunately, pnly one decimal place is obtained.
        params = self.setUp()
        _, actual = transform_coordinate([pt], 0, params, deriv=True)
        def func(x):
            return transform_coordinate([x], 0, params)
        desired = approx_fprime([pt], func, epsilon=1.49e-08)
        assert np.abs(desired - actual) < 1e-1

    @pytest.mark.parametrize("x", [-10, -2, 0, 2.2, 1.23])
    @pytest.mark.parametrize("y", [-3, 2., -10.2321, 20.232109])
    def test_transforming_y_against_formula(self, x, y):
        def formula_transforming_y(x, y):
            r"""Closed form formula for transforming y coordinate."""
            fac1 = 5. * np.sqrt(np.pi / 2.) * np.exp(-2. * (x - 1.) ** 2.)
            fac1 *= (np.sqrt(np.pi) * (erf(2. ** 0.5 * (y - 2)) + 1.) / (2. * np.sqrt(2.)))
            fac2 = 10. * np.sqrt(np.pi / 3.) * np.exp(-3. * (x - 2.) ** 2.)
            fac2 *= (np.sqrt(np.pi) * (erf(3. ** 0.5 * (y - 2)) + 1.) / (2. * np.sqrt(3.)))
            num = fac1 + fac2

            dac1 = 5. * (np.pi / 2.) * np.exp(-2. * (x - 1.) ** 2.)
            dac2 = 10. * (np.pi / 3.) * np.exp(-3. * (x - 2.) ** 2.)
            den = dac1 + dac2
            return num / den
        true_ans = transform_coordinate([x, y], 1, self.setUp())
        assert np.abs(true_ans - formula_transforming_y(x, y)) < 1e-8

    @pytest.mark.parametrize("x", [-10, -2, 0, 2.2])
    @pytest.mark.parametrize("y", [-3, 2., -10.2321])
    @pytest.mark.parametrize("z", [-10., 0., 2.343432])
    def test_transforming_z_against_formula(self, x, y, z):
        def formula_transforming_z(x, y, z):
            r"""Closed form formula for transforming z coordinate."""
            a1, a2, a3 = (x - 1.), (y - 2.), (z - 3.)
            erfx = erf(2. ** 0.5 * a3) + 1.
            fac1 = 5. * np.exp(-2. * (a1 ** 2. + a2 ** 2.)) * erfx * np.pi ** 0.5 / (2. * 2. ** 0.5)

            b1, b2, b3 = (x - 2.), (y - 2.), (z - 2.)
            erfy = erf(3. ** 0.5 * b3) + 1.
            fac2 = 10. * np.exp(-3. * (b1 ** 2. + b2 ** 2.)) * erfy * np.pi ** 0.5 / (
                    2. * 3. ** 0.5)

            den = 5. * (np.pi / 2.) ** 0.5 * np.exp(-2. * (a1 ** 2. + a2 ** 2.))
            den += 10. * (np.pi / 3.) ** 0.5 * np.exp(-3. * (b1 ** 2. + b2 ** 2.))
            return (fac1 + fac2) / den

        params, obj = self.setUp(ss=0.5, return_obj=True)
        true_ans = formula_transforming_z(x, y, z)
        # Test function
        actual = transform_coordinate([x, y, z], 2, params)
        assert np.abs(true_ans - actual) < 1e-8

        # Test Method
        actual = obj.transform(np.array([x, y, z]))[2]
        assert np.abs(true_ans - actual) < 1e-8

    def test_transforming_simple_grid(self):
        r"""Test transforming a grid that only contains one non-boundary point."""
        ss = 0.5
        params, obj = self.setUp(ss, return_obj=True)
        num_pt = int(1 / ss) + 1  # number of points in one-direction.
        assert obj.points.shape == (num_pt**3, 3)
        non_boundary_pt_index = num_pt**2 + num_pt + 1
        real_pt = obj.points[non_boundary_pt_index]
        # Test that this point is not the boundary.
        assert real_pt[0] != np.nan
        assert real_pt[1] != np.nan
        assert real_pt[2] != np.nan
        # Test that converting the point back to unit cube gives [0.5, 0.5, 0.5].
        for i_var in range(0, 3):
            transf = transform_coordinate(real_pt, i_var, obj.promol)
            assert np.abs(transf - 0.5) < 1e-5
        # Test that all other points are indeed boundary points.
        all_nans = np.delete(obj.points, non_boundary_pt_index, axis=0)
        assert np.all(np.any(np.isnan(all_nans), axis=1))

    # @pytest.mark.parametrize("x", [-2, -2, 0, 2.2])
    # @pytest.mark.parametrize("y", [-3, 2., -3.2321])
    # @pytest.mark.parametrize("z", [-2., 1.5, 2.343432])
    def test_transforming_with_inverse_transformation_is_identity(self):
        # Note that for points far away from the promolecular gets mapped to nan.
        # So this isn't really the inverse, in the mathematical sense.
        param, obj = self.setUp(0.5, return_obj=True)

        pt = np.array([1, 2, 3], dtype=np.float64)
        transf = obj.transform(pt)
        reverse = obj.inverse(transf)
        assert np.all(np.abs(reverse - pt) < 1e-10)

    def test_integrating_itself(self):
        r"""Test integrating the very same promolecular density"""
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

    @pytest.mark.parametrize("pt", np.arange(-5., 5., 0.75))
    def test_derivative_tranformation_x_finite_difference(self, pt):
        params, obj = self.setUp(ss=0.2, return_obj=True)
        pt = np.array([pt, 2., 3.])

        actual = obj.jacobian(pt)
        def tranformation_x(pt):
            return transform_coordinate(pt, 0, params)

        grad = approx_fprime([pt[0]], tranformation_x, 1e-6)
        assert np.abs(grad - actual[0, 0]) < 1e-4

    @pytest.mark.parametrize("x", np.arange(-5., 5., 0.75))
    @pytest.mark.parametrize("y", [-2.5, -1.5, 0, 1.5])
    def test_derivative_tranformation_y_finite_difference(self, x, y):
        params, obj = self.setUp(ss=0.2, return_obj=True)
        actual = obj.jacobian(np.array([x, y, 3.]))

        def tranformation_y(pt):
            return transform_coordinate([x, pt[0]], 1, params)

        grad = approx_fprime([y], tranformation_y, 1e-8)
        assert np.abs(grad - actual[1, 1]) < 1e-5

        def transformation_y_wrt_x(pt):
            return transform_coordinate([pt[0], y], 1, params)
        h = 1e-8
        deriv = np.imag(transformation_y_wrt_x([complex(x, h)])) / h
        assert np.abs(deriv - actual[1, 0]) < 1e-4

    @pytest.mark.parametrize("x", [-1.5, -0.5, 0, 2.5])
    @pytest.mark.parametrize("y", [-3, 2., -2.2321])
    @pytest.mark.parametrize("z", [-1.5, 0., 2.343432])
    def test_derivative_tranformation_z_finite_difference(self, x, y, z):
        params, obj = self.setUp(ss=0.2, return_obj=True)
        actual = obj.jacobian(np.array([x, y, z]))

        def tranformation_z(pt):
            return transform_coordinate([x, y, pt[0]], 2, params)

        grad = approx_fprime([z], tranformation_z, 1e-8)
        assert np.abs(grad - actual[2, 2]) < 1e-5

        def transformation_z_wrt_y(pt):
            return transform_coordinate([x, pt[0], z], 2, params)

        deriv = approx_fprime([y], transformation_z_wrt_y, 1e-8)
        assert np.abs(deriv - actual[2, 1]) < 1e-4

        def transformation_z_wrt_x(pt):
            a = transform_coordinate([pt[0], y, z], 2, params)
            return a

        h = 1e-8
        deriv = np.imag(transformation_z_wrt_x([complex(x, h)])) / h

        assert np.abs(deriv - actual[2, 0]) < 1e-4

    @pytest.mark.parametrize("x", [-1.5, -0.5, 0, 2.5])
    @pytest.mark.parametrize("y", [-3, 2., -2.2321])
    @pytest.mark.parametrize("z", [-1.5, 0., 2.343432])
    def test_steepest_ascent_direction_with_numerics(self, x, y, z):
        r"""Test steepest-ascent direction match in real and theta space.

        The function to test is x^2 + y^2 + z^2.
        """
        def grad(pt):
            # Gradient of x^2 + y^2 + z^2.
            return np.array([2. * pt[0], 2. * pt[1], 2. * pt[2]])

        params, obj = self.setUp(ss=0.2, return_obj=True)

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


class TestOneGaussianAgainstNumerics():
    r"""
    Tests Using Numerical Integration of a One Gaussian function to match transformation function.

    """
    def setUp(self, ss=0.1, return_obj=False):
        c = np.array([[5.]])
        e = np.array([[2.]])
        coord = np.array([[1., 2., 3.]])
        params = PromolParams(c, e, coord, dim=3, pi_over_exponents=np.sqrt(np.pi / e))
        if return_obj:
            num_pts = int(1 / ss) + 1
            weights = np.array([(1. / (num_pts - 2))**3.] * num_pts**3)
            obj = CubicProTransform([ss] * 3, weights, params.c_m, params.e_m, params.coords)
            return params,  obj
        return params

    @pytest.mark.parametrize("pt", np.arange(-5., 5., 0.5))
    def test_transforming_x_against_numerics(self, pt):
        def promolecular_in_x(grid, every_grid):
            r"""Constructs the formula of promolecular for integration."""
            promol_x = 5. * np.exp(-2. * (grid - 1.)**2.)
            promol_x_all = 5. * np.exp(-2. * (every_grid - 1.)**2.)
            return promol_x, promol_x_all

        true_ans = transform_coordinate([pt], 0, self.setUp())
        grid = np.arange(-10., pt, 0.00001)  # Integration till a x point
        every_grid = np.arange(-10., 10., 0.00001)  # Full Integration
        promol_x, promol_x_all = promolecular_in_x(grid, every_grid)

        # Integration over y and z cancel out from numerator and denominator.
        actual = np.trapz(promol_x, grid) / np.trapz(promol_x_all, every_grid)
        assert np.abs(true_ans - actual) < 1e-5

    @pytest.mark.parametrize("x", [-10, -2, 0, 2.2, 1.23])
    @pytest.mark.parametrize("y", [-3, 2., -10.2321, 20.232109])
    def test_transforming_y_against_numerics(self, x, y):
        def promolecular_in_y(grid, every_grid):
            r"""Constructs the formula of promolecular for integration."""
            promol_y = 5. * np.exp(-2. * (grid - 2.) ** 2.)
            promol_y_all = 5. * np.exp(-2. * (every_grid - 2.) ** 2.)
            return promol_y_all, promol_y

        true_ans = transform_coordinate([x, y], 1, self.setUp())
        grid = np.arange(-10., y, 0.00001)  # Integration till a x point
        every_grid = np.arange(-10., 10., 0.00001)  # Full Integration
        promol_y_all, promol_y = promolecular_in_y(grid, every_grid)

        # Integration over z cancel out from numerator and denominator.
        # Further, gaussian at a point does too.
        actual = np.trapz(promol_y, grid) / np.trapz(promol_y_all, every_grid)
        assert np.abs(true_ans - actual) < 1e-5

    @pytest.mark.parametrize("x", [-10, -2, 0, 2.2])
    @pytest.mark.parametrize("y", [-3, 2., -10.2321])
    @pytest.mark.parametrize("z", [-10., 0., 2.343432])
    def test_transforming_z_against_numerics(self, x, y, z):
        def promolecular_in_z(grid, every_grid):
            r"""Constructs the formula of promolecular for integration."""
            promol_z = 5. * np.exp(-2. * (grid - 3.) ** 2.)
            promol_z_all = 5. * np.exp(-2. * (every_grid - 3.) ** 2.)
            return promol_z_all, promol_z

        grid = np.arange(-10., z, 0.00001)  # Integration till a x point
        every_grid = np.arange(-10., 10., 0.00001)  # Full Integration
        promol_z_all, promol_z = promolecular_in_z(grid, every_grid)

        actual = np.trapz(promol_z, grid) / np.trapz(promol_z_all, every_grid)
        true_ans = transform_coordinate([x, y, z], 2, self.setUp())
        assert np.abs(true_ans - actual) < 1e-5

    @pytest.mark.parametrize("x", [0., 0.25, 1.1, 0.5, 1.5])
    @pytest.mark.parametrize("y", [0., 1.25, 2.2, 2.25, 2.5])
    @pytest.mark.parametrize("z", [0., 2.25, 3.2, 3.25, 4.5])
    def test_jacobian_is_diagonal(self, x, y, z):
        params, obj = self.setUp(ss=0.2, return_obj=True)
        actual = obj.jacobian(np.array([x, y, z]))

        # assert lower-triangular component is zero.
        assert np.abs(actual[1, 0]) < 1e-5
        assert np.abs(actual[2, 0]) < 1e-5
        assert np.abs(actual[2, 1]) < 1e-5

        # test derivative wrt to x
        def tranformation_x(pt):
            return transform_coordinate([pt[0], y, z], 0, params)

        grad = approx_fprime([x], tranformation_x, 1e-8)
        assert np.abs(grad - actual[0, 0]) < 1e-5

        # test derivative wrt to y
        def tranformation_y(pt):
            return transform_coordinate([x, pt[0]], 1, params)

        grad = approx_fprime([y], tranformation_y, 1e-8)
        assert np.abs(grad - actual[1, 1]) < 1e-5

        # Test derivative wrt to z
        def tranformation_z(pt):
            return transform_coordinate([x, y, pt[0]], 2, params)
        grad = approx_fprime([z], tranformation_z, 1e-8)
        assert np.abs(grad - actual[2, 2]) < 1e-5

    def test_integration_slightly_perturbed_gaussian(self):
        # Only Measured against one decimal place and very similar exponent.
        params, obj = self.setUp(ss=0.03, return_obj=True)

        # Gaussian exponent is slightly perturbed from 2.
        exponent = 2.001

        def gaussian(grid):
            return 5. * np.exp(-exponent * np.sum((grid - np.array([1., 2., 3.]))**2., axis=1))

        func_vals = gaussian(obj.points)
        desired = 5. * np.sqrt(np.pi / exponent)**3.
        actual = obj.integrate(func_vals, trick=True)
        assert np.abs(actual - desired) < 1e-2


def test_padding_arrays():
    r"""Test different array sizes are correctly padded."""
    coeff = np.array([[1., 2.], [1., 2., 3., 4.], [5.]])
    exps = np.array([[4., 5.], [5., 6., 7., 8.], [9.]])
    coeff_pad, exps_pad = _pad_coeffs_exps_with_zeros(coeff, exps)
    coeff_desired = np.array([[1., 2., 0., 0.], [1., 2., 3., 4.], [5., 0., 0., 0.]])
    np.testing.assert_array_equal(coeff_desired, coeff_pad)
    exp_desired = np.array([[4., 5., 0., 0.], [5., 6., 7., 8.], [9., 0., 0., 0.]])
    np.testing.assert_array_equal(exp_desired, exps_pad)
