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
r"""Promolecular Grid Transformation."""

from dataclasses import dataclass, astuple

from grid.basegrid import Grid

import numpy as np

from scipy.linalg import solve_triangular
from scipy.optimize import root_scalar
from scipy.special import erf

__all__ = ["CubicProTransform"]


@dataclass
class _PromolParams:
    r"""Private class for Promolecular Density information."""
    c_m: np.ndarray
    e_m: np.ndarray
    coords: np.ndarray
    pi_over_exponents: np.ndarray
    dim: int = 3

    def promolecular(self, grid):
        r"""
        Evaluate the promolecular density over a grid.

        Parameters
        ----------
        grid : np.ndarray(N,)
            Grid points.

        Returns
        -------
        np.ndarray(N,) :
            Promolecular density evaluated at the grid points.

        """
        # M is the number of centers/atoms.
        # D is the number of dimensions, usually 3.
        # K is maximum number of gaussian functions over all M atoms.
        cm, em, coords = self.c_m, self.e_m, self.coords
        # Shape (N, M, D), then Summing gives (N, M, 1)
        distance = np.sum((grid - coords[:, np.newaxis]) ** 2.0, axis=2, keepdims=True)
        # At each center, multiply Each Distance of a Coordinate, with its exponents.
        exponen = np.exp(-np.einsum("MND, MK-> MNK", distance, em))
        # At each center, multiply the exponential with its coefficients.
        gaussian = np.einsum("MNK, MK -> MNK", exponen, cm)
        # At each point, sum for each center, then sum all centers together.
        return np.einsum("MNK -> N", gaussian, dtype=np.float64)


class CubicProTransform(Grid):
    r"""
    Promolecular Grid Transformation of a Cubic, Uniform Grid in [0,1]^3 to Real space.

    Attributes
    ----------
    num_pts : (int, int, int)
        The number of points, including both of the end/boundary points, in x, y, and z direction.
        This is calculated as `int(1. / ss[i]) + 1`.
    ss : (float, float, float)
        The step-size in each x, y, and z direction.
    points : np.ndarray(N, 3)
        The transformed points in real space.
    prointegral : float
        The integration value of the promolecular density over Euclidean space.
    weights : np.ndarray(N,)
        The weights multiplied by `prointegral`.
    promol : namedTuple
        Data about the Promolecular density.

    Methods
    -------
    integrate(trick=False)
        Integral of a real-valued function over Euclidean space.
    jacobian()
        Jacobian of the transformation from Real space to Theta/Unit cube space.
    steepest_ascent_theta()
        Direction of steepest-ascent of a function in theta space from gradient in real space.
    transform():
        Transform Real point to theta/unit-cube point :math:`[0,1]^3`.
    inverse(bracket=(-10, 10))
        Transform theta/unit-cube point to Real space :math:`\mathbb{R}^3`.

    Examples
    --------
    Define information of the Promolecular Density.
    >> c = np.array([[5.], [10.]])
    >> e = np.array([[2.], [3.]])
    >> coord = np.array([[0., 0., 0.], [2., 2., 2.]])

    Define information of the grid and its weights.
    >> numb_x = 50
    >> weights = np.array([0.01] * numb_x**3)  # Simple Riemannian weights.
    >> promol = CubicProTransform([numb_x, numb_x, numb_x], weights, c, e, coord)

    To integrate some function f.
    >> def f(pt):
    >>    return np.exp(-0.1 * np.linalg.norm(pt, axis=1)**2.)
    >> func_values = f(promol.points)
    >> print("The integral is %.4f" % promol.integrate(func_values, trick=False)

    References
    ----------
    .. [1] J. I. Rodríguez, D. C. Thompson, P. W. Ayers, and A. M. Koster, "Numerical integration
            of exchange-correlation energies and potentials using transformed sparse grids."

    Notes
    -----
    TODO: Insert Info About Conditional Distribution Method.
    TODO: Add Infor about how boundarys on theta-space are mapped to np.nan.

    """

    def __init__(self, num_pts, weights, coeffs, exps, coords):
        # TODO: Add Types
        if not isinstance(num_pts, (tuple, list)):
            pass
        if not isinstance(coeffs, (list, np.ndarray)):
            pass
        if not isinstance(exps, (list, np.ndarray)):
            pass
        if not isinstance(coords, (list, np.ndarray)):
            pass
        self._ss = (
            1. / (num_pts[0] - 1),
            1. / (num_pts[1] - 1),
            1. / (num_pts[2] - 1),
        )
        self._num_pts = num_pts

        # pad coefficients and exponents with zeros to have the same size, easier to use numpy.
        coeffs, exps = _pad_coeffs_exps_with_zeros(coeffs, exps)
        # Rather than computing this repeatedly. It is fixed.
        with np.errstate(divide="ignore"):
            pi_over_exponents = np.sqrt(np.pi / exps)
            pi_over_exponents[exps == 0] = 0
        self._prointegral = np.sum(coeffs * pi_over_exponents ** 3.0)
        self._promol = _PromolParams(coeffs, exps, coords, pi_over_exponents)

        # initialize parent class
        empty_points = np.empty((np.prod(self._num_pts), 3), dtype=np.float64)
        super().__init__(empty_points, weights * self._prointegral)
        self._transform()

    @property
    def num_pts(self):
        r"""Return number of points in each direction."""
        return self._num_pts

    @property
    def ss(self):
        r"""Return stepsize of the cubic grid."""
        return self._ss

    @property
    def prointegral(self):
        r"""Return integration of Promolecular density."""
        return self._prointegral

    @property
    def promol(self):
        r"""Return `PromolParams` namedTuple."""
        return self._promol

    def integrate(self, *value_arrays, trick=False, tol=1e-10):
        r"""
        Integrate any function.

        Assumes integrand decays faster than the promolecular density.

        Parameters
        ----------
        *value_arrays : (np.ndarray(N, dtype=float),)
            One or multiple value array to integrate.
        trick : bool, optional
            If true, uses the promolecular trick.
        tol : float, optional
            Integrand is set to zero whenever promolecular density is less than tolerance.
            Default value is 1e-10.

        Returns
        -------
        float :
            Return the integration of the function.

        Raises
        ------
        TypeError
            Input integrand is not of type np.ndarray.
        ValueError
            Input integrand array is given or not of proper shape.

        Notes
        -----
        - TODO: Insert formula for integration.
        - This method assumes the integrand decays faster than the promolecular density.

        """
        promolecular = self.promol.promolecular(self.points)
        # Integrand is set to zero when promolecular is less than certain value and,
        # When on the boundary (hence when promolecular is nan).
        cond = (promolecular <= tol) | (np.isnan(promolecular))
        promolecular = np.ma.masked_where(cond, promolecular, copy=False)

        integrands = []
        for arr in value_arrays:
            # This is needed as it gives incorrect results when arr.dtype isn't object.
            assert arr.dtype != object, "Array dtype should not be object."
            # This may be refactored to fit in the general promolecular trick in `grid`.
            # Masked array is needed since division by promolecular contains nan.
            if trick:
                integrand = np.ma.divide(arr - promolecular, promolecular)
            else:
                integrand = np.ma.divide(arr, promolecular)
            # Function/Integrand evaluated at points on the boundary is set to zero.
            np.ma.fix_invalid(integrand, copy=False, fill_value=0)
            integrands.append(integrand)

        if trick:
            return self.prointegral + super().integrate(*integrands)
        return super().integrate(*integrands)

    def jacobian(self, real_pt):
        r"""
        Jacobian of the transformation from real space to unit-cube/theta space.

        Precisely, it is the lower-triangular matrix
        .. math::
            \begin{bmatrix}
                \frac{\partial \theta_x}{\partial X} & 0 & 0 \\
                \frac{\partial \theta_y}{\partial X} & \frac{\partial \theta_y}{\partial Y} & 0 \\
                \frac{\partial \theta_z}{\partial X} & \frac{\partial \theta_Z}{\partial Y} &
                \frac{\partial \theta_Z}{\partial Z}
            \end{bmatrix}.

        Parameters
        ----------
        real_pt : np.ndarray(3,)
            Point in :math:`\mathbb{R}^3`.

        Returns
        -------
        np.ndarray(3, 3) :
            Jacobian of transformation.

        """
        jacobian = np.zeros((3, 3), dtype=np.float64)

        c_m, e_m, coords, pi_over_exps, dim = astuple(self.promol)

        # Code is duplicated from `transform_coordinate` due to effiency reasons.
        # TODO: Reduce the number of computation with `i_var`.
        for i_var in range(0, 3):

            # Distance to centers/nuclei`s and Prefactors.
            diff_coords = real_pt[: i_var + 1] - coords[:, : i_var + 1]
            diff_squared = diff_coords ** 2.0
            distance = np.sum(diff_squared[:, :i_var], axis=1)[:, np.newaxis]
            # If i_var is zero, then distance is just all zeros.

            # Gaussian Integrals Over Entire Space For Numerator and Denomator.
            coeff_num = c_m * np.exp(-e_m * distance) * pi_over_exps ** (dim - i_var)

            # Get integral of Gaussian till a point.
            coord_ivar = diff_coords[:, i_var][:, np.newaxis]
            # (pi / exponent)^0.5 is factored and absorbed in `coeff_num`.
            integrate_till_pt_x = (erf(np.sqrt(e_m) * coord_ivar) + 1.0) / 2.0

            # Final Result.
            transf_num = np.sum(coeff_num * integrate_till_pt_x)
            transf_den = np.sum(coeff_num)

            for j_deriv in range(0, i_var + 1):
                if i_var == j_deriv:
                    # Derivative eliminates `integrate_till_pt_x`, and adds a Gaussian.
                    inner_term = coeff_num * np.exp(
                        -e_m * diff_squared[:, i_var][:, np.newaxis]
                    )
                    # Needed because coeff_num has additional (pi / exponent)^0.5 term.
                    inner_term /= pi_over_exps
                    jacobian[i_var, i_var] = np.sum(inner_term) / transf_den
                elif j_deriv < i_var:
                    # Derivative of inside of Gaussian.
                    deriv_quadratic = (
                            -e_m * 2.0 * diff_coords[:, j_deriv][:, np.newaxis]
                    )
                    deriv_num = np.sum(
                        coeff_num * integrate_till_pt_x * deriv_quadratic
                    )
                    deriv_den = np.sum(coeff_num * deriv_quadratic)
                    # Quotient Rule
                    jacobian[i_var, j_deriv] = (
                            deriv_num * transf_den - transf_num * deriv_den
                    )
                    jacobian[i_var, j_deriv] /= transf_den ** 2.0

        return jacobian

    def transform(self, real_pt):
        r"""
        Transform a real point in three-dimensional Reals to theta/unit cube.

        Parameters
        ----------
        real_pt : np.ndarray(3)
            Point in :math:`\mathbb{R}^3`

        Returns
        -------
        theta_pt : np.ndarray(3)
            Point in :math:`[0, 1]^3`.

        """
        return np.array(
            [transform_coordinate(real_pt, i, self.promol) for i in range(0, 3)]
        )

    def inverse(self, theta_pt, bracket=(-10, 10)):
        r"""
        Transform a theta/unit-cube point to three-dimensional Real space.

        Parameters
        ----------
        theta_pt : np.ndarray(3)
            Point in :math:`[0, 1]^3`
        bracket : (float, float), optional
            Interval where root is suspected to be in Reals.
            Used for "brentq" root-finding method. Default is (-10, 10).

        Returns
        -------
        real_pt : np.ndarray(3)
            Point in :math:`\mathbb{R}^3`

        Notes
        -----
        - If a point is far away from the promolecular density, then it will be mapped
            to `np.nan`.

        """
        real_pt = []
        for i in range(0, 3):
            scalar = inverse_coordinate(
                theta_pt[i], i, self.promol, real_pt[:i], bracket
            )
            real_pt.append(scalar)
        return np.array(real_pt)

    def derivative(self, real_pt, real_derivative):
        r"""
        Directional derivative in theta space.

        Parameters
        ----------
        real_pt : np.ndarray(3)
            Point in :math:`\mathbb{R}^3`
        real_derivative : np.ndarray(3)
            Derivative of a function in real space with respect to x, y, z coordinates.

        Returns
        -------
        theta_derivative : np.ndarray(3)
            Derivative of a function in theta space with respect to theta coordinates.

        Notes
        -----
        This does not preserve the direction of steepest-ascent/gradient.

        """
        jacobian = self.jacobian(real_pt)
        return solve_triangular(jacobian.T, real_derivative)

    def steepest_ascent_theta(self, real_pt, real_grad):
        r"""
        Steepest ascent direction of a function in theta/unit-cube space.

        Steepest ascent is the gradient ie direction of maximum change of a function.
        This guarantees moving in direction of steepest ascent in real-space
        corresponds to moving in the direction of the gradient in theta-space.

        Parameters
        ----------
        real_pt : np.ndarray(3)
            Point in :math:`\mathbb{R}^3`
        real_grad : np.ndarray(3)
            Gradient of a function in real space.

        Returns
        -------
        theta_grad : np.ndarray(3)
            Gradient of a function in theta/unit-cube space.

        """
        jacobian = self.jacobian(real_pt)
        return jacobian.dot(real_grad)

    def _transform(self):
        # Indices (i, j, k) start from bottom, left-most corner of the unit cube.
        counter = 0
        for ix in range(self.num_pts[0]):
            cart_pt = [None, None, None]
            unit_x = self.ss[0] * ix

            bracx = self._get_bracket((ix,), 0)
            cart_pt[0] = inverse_coordinate(unit_x, 0, self.promol, cart_pt, bracx)

            for iy in range(self.num_pts[1]):
                unit_y = self.ss[1] * iy

                bracy = self._get_bracket((ix, iy), 1)
                cart_pt[1] = inverse_coordinate(unit_y, 1, self.promol, cart_pt, bracy)

                for iz in range(self.num_pts[2]):
                    unit_z = self.ss[2] * iz

                    bracz = self._get_bracket((ix, iy, iz), 2)
                    cart_pt[2] = inverse_coordinate(
                        unit_z, 2, self.promol, cart_pt, bracz
                    )

                    self.points[counter] = cart_pt.copy()
                    counter += 1

    def _get_bracket(self, indices, i_var):
        r"""
        Obtain brackets for root-finder based on the coordinate of the point.

        Parameters
        ----------
        indices : tuple(int, int, int)
            The indices of a point, where (0, 0, 0) is the bottom, left-most, down point
            of the cube.
        i_var : int
            Index of point being transformed.

        Returns
        -------
        (float, float) :
            The bracket for the root-finder solver.

        """
        # If it is a boundary point, then return nan.
        if 0 in indices[: i_var + 1] or (self.num_pts[i_var] - 1) in indices[: i_var + 1]:
            return np.nan, np.nan
        # If it is a new point, with no nearby point, get a large initial guess.
        elif indices[i_var] == 1:
            min = (np.min(self.promol.coords[:, i_var]) - 3.0) * 20.0
            max = (np.max(self.promol.coords[:, i_var]) + 3.0) * 20.0
            return min, max
        # If the previous point has been converted, use that as a initial guess.
        if i_var == 0:
            index = (indices[0] - 1) * self.num_pts[1] * self.num_pts[2]
        elif i_var == 1:
            index = indices[0] * self.num_pts[1] * self.num_pts[2] + self.num_pts[2] * (
                    indices[1] - 1
            )
        elif i_var == 2:
            index = (
                    indices[0] * self.num_pts[1] * self.num_pts[2]
                    + self.num_pts[2] * indices[1]
                    + indices[2]
                    - 1
            )

        return self.points[index, i_var], self.points[index, i_var] + 10.0


def transform_coordinate(real_pt, i_var, promol_params, deriv=False, sderiv=False):
    r"""
    Transform the `i_var` coordinate of a real point to [0, 1] using promolecular density.

    Parameters
    ----------
    real_pt : np.ndarray(D,)
        Real point being transformed.
    i_var : int
        Index that is being tranformed. Less than D.
    promol_params : namedTuple
        Data about the Promolecular density.
    deriv : bool
        If true, return the derivative of transformation wrt `i_var` real variable.
        Default is False.
    sderiv : bool
        If true, return the second derivative of transformation wrt `i_var` real variable.
        Default is False.

    Returns
    -------
    unit_pt, deriv, sderiv : (float, float, float)
        The transformed point in [0,1]^D and its derivative with respect to real point and
        the second derivative with respect to real point are returned.

    """
    c_m, e_m, coords, pi_over_exps, dim = astuple(promol_params)

    # Distance to centers/nuclei`s and Prefactors.
    diff_coords = real_pt[: i_var + 1] - coords[:, : i_var + 1]
    diff_squared = diff_coords ** 2.0
    distance = np.sum(diff_squared[:, :i_var], axis=1)[:, np.newaxis]
    # If i_var is zero, then distance is just all zeros.

    # Gaussian Integrals Over Entire Space For Numerator and Denomator.
    gaussian_integrals = np.exp(-e_m * distance) * pi_over_exps ** (dim - i_var)
    coeff_num = c_m * gaussian_integrals

    # Get the integral of Gaussian till a point excluding a prefactor.
    # This prefactor (pi / exponents) is included in `gaussian_integrals`.
    coord_ivar = diff_coords[:, i_var][:, np.newaxis]
    integrate_till_pt_x = (erf(np.sqrt(e_m) * coord_ivar) + 1.0) / 2.0

    # Final Result.
    transf_num = np.sum(coeff_num * integrate_till_pt_x)
    transf_den = np.sum(coeff_num)
    transform_value = transf_num / transf_den

    if deriv:
        inner_term = coeff_num * np.exp(-e_m * diff_squared[:, i_var][:, np.newaxis])
        deriv = np.sum(inner_term) / transf_den

        if sderiv:
            sderiv = np.sum(inner_term * -e_m * 2.0 * coord_ivar) / transf_den
            return transform_value, deriv, sderiv
        return transform_value, deriv
    return transform_value


def _root_equation(init_guess, prev_trans_pts, theta_pt, i_var, params):
    r"""
    Equation to solve for the root to find inverse coordinate from theta space to Real space.

    Parameters
    ----------
    init_guess : float
        Initial guess of Real point that transforms to `theta_pt`.
    prev_trans_pts : list[`i_var` - 1]
        The previous points in real-space that were already transformed.
    theta_pt : float
        The point in [0, 1] being transformed to the Real space.
    i_var : int
        Index of variable being transformed.
    params : _PromolParams
        Promolecular density data class.

    Returns
    -------
    float :
        The difference between `theta_pt` and the transformed point based on
        `init_guess` and `prev_trans_pts`.

    """
    all_points = np.append(prev_trans_pts, init_guess)
    transf_pt = transform_coordinate(all_points, i_var, params)
    return theta_pt - transf_pt


def inverse_coordinate(theta_pt, i_var, params, transformed, bracket=(-10, 10)):
    r"""
    Transform a point in [0, 1] to the real space corresponding to `i_var` variable.

    Parameters
    ----------
    theta_pt : float
        Point in [0, 1].
    i_var : int
        Index that is being tranformed. Less than D.
    promol_params : namedTuple
        Data about the Promolecular density.
    transformed : list[`i_var` - 1]
        The set of previous points before index `i_var` that were transformed to real space.
    bracket : (float, float)
        Interval where root is suspected to be in Reals. Used for "brentq" root-finding method.
        Default is (-10, 10).

    Returns
    -------
    real_pt : float
        Return the transformed real point.

    Raises
    ------
    AssertionError :  If the root did not converge, or brackets did not have opposite sign.
    RuntimeError :  If dynamic bracketing reached maximum iteration.

    Notes
    -----
    - If the theta point is on the boundary or it is itself a nan, then it get's mapped to nan.
        Further, if nan is in `transformed[:i_var]` then this function will return nan.
    - If Brackets do not have the opposite sign, will change the brackets by adding/subtracting
        the value 10 to the lower or upper bound that is closest to zero.

    """
    def _dynamic_bracketing(l_bnd, u_bnd, maxiter=50):
        r"""Dynamically changes either the lower or upper bound to have different sign values."""
        def is_same_sign(x, y): return (x >= 0 and y >= 0) or (x < 0 and y < 0)

        bounds = [l_bnd, u_bnd]
        f_l_bnd = _root_equation(l_bnd, *args)
        f_u_bnd = _root_equation(u_bnd, *args)
        # Get Index of the one that is closest to zero, the one that needs to change.
        f_bnds = np.abs([f_l_bnd, f_u_bnd])
        idx = f_bnds.argmin()
        # Check if they have the same sign.
        same_sign = is_same_sign(*bounds)
        counter = 0
        while same_sign and counter < maxiter:
            # Add 10 to the upper bound or subtract 10 to the lower bound to the one that
            # is closest to zero. This is done based on the sign.
            bounds[idx] = np.sign(idx - 0.5) * 10 + bracket[idx]
            # Update info for next iteration.
            if idx == 0:
                f_l_bnd = _root_equation(bracket[0], *args)
            else:
                f_u_bnd = _root_equation(bracket[1], *args)
            same_sign = is_same_sign(f_l_bnd, f_u_bnd)
            counter += 1

        if counter == maxiter:
            raise RuntimeError("Dynamic Bracketing did not converge.")
        return tuple(bounds)

    # Check's if this is a boundary points which is mapped to np.nan
    # These two conditions are added for individual point transformation.
    if np.abs(theta_pt - 0.0) < 1e-10:
        return np.nan
    if np.abs(theta_pt - 1.0) < 1e-10:
        return np.nan
    # This condition is added for transformation of the entire grid.
    # The [:i_var] is needed because of the way I've set-up transforming points in _transform.
    # Likewise for the bracket, see the function `get_bracket`.
    if np.nan in bracket or np.nan in transformed[:i_var]:
        return np.nan

    # Set up Arguments for root_equation with dynamic bracketing.
    args = (transformed[:i_var], theta_pt, i_var, params)
    bracket = _dynamic_bracketing(bracket[0], bracket[1])
    root_result = root_scalar(
        _root_equation,
        args=args,
        method="brentq",
        bracket=[bracket[0], bracket[1]],
        maxiter=50,
        xtol=2e-15,
    )
    assert root_result.converged
    return root_result.root


def _pad_coeffs_exps_with_zeros(coeffs, exps):
    r"""Pad Promolecular coefficients and exponents with zero. Results in same size array."""
    max_numb_of_gauss = max(len(c) for c in coeffs)
    coeffs = np.array(
        [
            np.pad(a, (0, max_numb_of_gauss - len(a)), "constant", constant_values=0.0)
            for a in coeffs
        ],
        dtype=np.float64,
    )
    exps = np.array(
        [
            np.pad(a, (0, max_numb_of_gauss - len(a)), "constant", constant_values=0.0)
            for a in exps
        ],
        dtype=np.float64,
    )
    return coeffs, exps
