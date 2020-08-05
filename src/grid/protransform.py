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

from dataclasses import astuple, dataclass, field

from grid.basegrid import Grid, OneDGrid

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.linalg import solve_triangular
from scipy.optimize import root_scalar
from scipy.special import erf

__all__ = ["CubicProTransform"]


class CubicProTransform(Grid):
    r"""
    Promolecular Grid Transformation of a Cubic Grid in :math:`[-1, 1]^3`.

    Grid is three dimensional and modeled as Tensor Product of Three, one dimensional grids.
    Theta space is defined to be :math:`[-1, 1]^3`.
    Real space is defined to be :math:`\mathbb{R}^3.`

    Attributes
    ----------
    num_pts : (int, int, int)
        The number of points, including both of the end/boundary points, in x, y, and z direction.
    prointegral : float
        The integration value of the promolecular density over Euclidean space.
    promol : namedTuple
        Data about the Promolecular density.
    points : np.ndarray(N, 3)
        The transformed points in real space.
    weights : np.ndarray(N,)
        The weights multiplied by `prointegral`.

    Methods
    -------
    integrate(trick=False)
        Integral of a real-valued function over Euclidean space. Can use promolecular trick.
    jacobian()
        Jacobian of the transformation from Real space to Theta space :math:`[-1, 1]^3`.
    hessian()
        Hessian of the transformation from Real space to Theta space :math:`[-1, 1]^3`.
    steepest_ascent_theta()
        Direction of steepest-ascent of a function in theta space from gradient in real space.
    transform():
        Transform Real point to Theta space :math:`[-1, 1]^3`.
    inverse(bracket=(-10, 10))
        Transform Theta point to Real space :math:`\mathbb{R}^3`.
    interpolate_function(use_log=False, nu=0)
        Interpolate a function (or its logarithm) at a real point. Can interpolate its derivative.

    Examples
    --------
    Define information of the Promolecular Density.
    >> c = np.array([[5.], [10.]])
    >> e = np.array([[2.], [3.]])
    >> coord = np.array([[0., 0., 0.], [2., 2., 2.]])

    Define information of the grid and its weights.
    >> from grid.onedgrid import GaussChebyshev

    >> numb_x = 50
    This is a grid in :math:`[-1, 1]`.
    >> oned = GaussChebyshev(numb_x)
    One dimensional grid is the same in all x, y, z directions.
    >> promol = CubicProTransform([oned, oned, oned], params.c_m, params.e_m, params.coords)

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
    Let :math:`\rho^o(x, y, z) = \sum_{i=1}^M \sum_{j=1}^D e^{}` be the Promolecular density of a \
    linear combination of Gaussian functions.

    The conditional distribution transformation from :math:`\mathbb{R}^3` to :math:`[-1, 1]^3`
    transfers the (x, y, z) coordinates in :math:`\mathbb{R}^3` to a set of coordinates,
    denoted as :math:`(\theta_x, \theta_y, \theta_z)`, in :math:`[-1,1]^3` that are "bunched"
    up where :math:`\rho^o` is large.

    Precisely it is,

    .. math::
        \begin{eqnarray}
            \theta_x(x) :&=
            -1 + 2 \frac{\int_{-\infty}^x \int \int \rho^o(x, y, z)dx dy dz }
                        {\int \int \int \rho^o(x, y, z)dxdydz}\\
            \theta_y(x, y) :&=
            -1 + 2 \frac{\int_{-\infty}^y \int \rho^o(x, y, z)dy dz }
                        {\int \int \rho^o(x, y, z)dydz} \\
            \theta_z(x, y, z) :&=
            -1 + 2 \frac{\int_{-\infty}^z \rho^o(x, y, z)dz }
                        {\int \rho^o(x, y, z)dz}\\
        \end{eqnarray}

    Integration of a integrable function :math:`f : \mathbb{R}^3 \rightarrow \mathbb{R}` can be
    done as follows in theta space:

    .. math::
        \int \int \int f(x, y, z)dxdy dz \approx
        \frac{1}{8} N \int_{-1}^1 \int_{-1}^1 \int_{-1}^1 \frac{f(\theta_x, \theta_y, \theta_z)}
        {\rho^o(\theta_x, \theta_y, \theta_z)} d\theta_x d\theta_y d\theta_z,

        \text{where }  N = \int \int \int \rho^o(x, y, z) dx dy dz.

    Note that this class always assumed the boundary of [-1, 1]^3 is always included.

    """

    def __init__(self, oned_grids, coeffs, exps, coords):
        if not isinstance(oned_grids, list):
            raise TypeError("oned_grid should be of type list.")
        if not np.all([isinstance(grid, OneDGrid) for grid in oned_grids]):
            raise TypeError("Grid in oned_grids should be of type `OneDGrid`.")
        if not np.all([grid.domain == (-1, 1) for grid in oned_grids]):
            raise ValueError("One Dimensional grid domain should be (-1, 1).")
        if not len(oned_grids) == 3:
            raise ValueError(
                "There should be three One-Dimensional grids in `oned_grids`."
            )

        self._num_pts = tuple([grid.size for grid in oned_grids])
        self._dim = len(oned_grids)

        # pad coefficients and exponents with zeros to have the same size, easier to use numpy.
        coeffs, exps = _pad_coeffs_exps_with_zeros(coeffs, exps)
        self._promol = _PromolParams(coeffs, exps, coords, self._dim)
        self._prointegral = self._promol.integrate_all()

        empty_pts = np.empty((np.prod(self._num_pts), self._dim), dtype=np.float64)
        weights = np.kron(
            np.kron(oned_grids[0].weights, oned_grids[1].weights), oned_grids[2].weights
        )
        # The prointegral is needed because of promolecular integration.
        # Divide by 8 needed because the grid is in [-1, 1] rather than [0, 1].
        super().__init__(empty_pts, weights * self._prointegral / 2.0 ** self._dim)
        # Transform Cubic Grid in Theta-Space to Real-space.
        self._transform(oned_grids)

    @property
    def num_pts(self):
        r"""Return number of points in each direction."""
        return self._num_pts

    @property
    def prointegral(self):
        r"""Return integration of Promolecular density."""
        return self._prointegral

    @property
    def promol(self):
        r"""Return `PromolParams` data class."""
        return self._promol

    @property
    def dim(self):
        r"""Return the dimension of the cubic grid."""
        return self._dim

    def transform(self, real_pt):
        r"""
        Transform a real point in three-dimensional Reals to theta space.

        Parameters
        ----------
        real_pt : np.ndarray(3)
            Point in :math:`\mathbb{R}^3`

        Returns
        -------
        theta_pt : np.ndarray(3)
            Point in :math:`[-1, 1]^3`.

        """
        return np.array(
            [
                _transform_coordinate(real_pt, i, self.promol)
                for i in range(0, self.promol.dim)
            ]
        )

    def inverse(self, theta_pt, bracket=(-10, 10)):
        r"""
        Transform a theta space point to three-dimensional Real space.

        Parameters
        ----------
        theta_pt : np.ndarray(3)
            Point in :math:`[-1, 1]^3`
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
        for i in range(0, self.promol.dim):
            scalar = _inverse_coordinate(
                theta_pt[i], i, real_pt[:i], self.promol, bracket
            )
            real_pt.append(scalar)
        return np.array(real_pt)

    def integrate(self, *value_arrays, trick=False, tol=1e-10):
        r"""
        Integrate any real-valued function on Euclidean space.

        Assumes the function decays faster than the promolecular density.

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
        - Formula for the integration of a integrable function
          :math:`f : \mathbb{R}^3 \rightarrow \mathbb{R}` is done as follows:

        .. math::
            \int \int \int f(x, y, z)dxdy dz \approx
            \frac{1}{8} N \int_{-1}^1 \int_{-1}^1 \int_{-1}^1 \frac{f(\theta_x, \theta_y, \theta_z)}
            {\rho^o(\theta_x, \theta_y, \theta_z)} d\theta_x d\theta_y d\theta_z,

            \text{where }  N = \int \int \int \rho^o(x, y, z) dx dy dz.

        - This method assumes function f decays faster than the promolecular density.

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

    def derivative(self, real_pt, real_derivative):
        r"""
        Directional derivative in theta space.

        Parameters
        ----------
        real_pt : np.ndarray(3)
            Point in :math:`\mathbb{R}^3`.
        real_derivative : np.ndarray(3)
            Derivative of a function in real space with respect to x, y, z coordinates.

        Returns
        -------
        theta_derivative : np.ndarray(3)
            Derivative of a function in theta space with respect to theta coordinates.

        Notes
        -----
        This does not preserve the direction of steepest-ascent/gradient.

        See Also
        --------
        steepest_ascent_theta : Steepest-ascent direction.

        """
        jacobian = self.jacobian(real_pt)
        return solve_triangular(jacobian.T, real_derivative)

    def steepest_ascent_theta(self, real_pt, real_grad):
        r"""
        Steepest ascent direction of a function in theta space.

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
            Gradient of a function in theta space.

        """
        jacobian = self.jacobian(real_pt)
        return jacobian.dot(real_grad)

    def interpolate_function(
        self, real_pt, func_values, oned_grids, use_log=False, nu=0
    ):
        r"""
        Interpolate function at a point.

        Parameters
        ----------
        real_pt : np.ndarray(3,)
            Point in :math:`\mathbb{R}^3` that needs to be interpolated.
        func_values : np.ndarray(N,)
            Function values at each point of the grid `points`.
        oned_grids = list(3,)
            List Containing Three One-Dimensional grid corresponding to x, y, z direction.
        use_log : bool
            If true, then logarithm is applied before interpolating to the function values,
            including  `func_values`.
        nu : int
            If zero, then the function is interpolated.
            If one, then the derivative is interpolated.

        Returns
        -------
        float :
            If nu is 0: Returns the interpolated of a function at a real point.
            If nu is 1: Returns the interpolated derivative of a function at a real point.

        """
        # TODO: Should oned_grids be stored as class attribute when only this method requires it.
        # TODO: Ask about use_log and derivative.
        if nu not in (0, 1):
            raise ValueError("The parameter nu %d is either zero or one " % nu)
        # Map to theta space.
        theta_pt = self.transform(real_pt)

        if use_log:
            func_values = np.log(func_values)

        jac = self.jacobian(real_pt).T

        # Interpolate the Z-Axis based on x, y coordinates in grid.
        def z_spline(z, x_index, y_index):
            # x_index, y_index is assumed to be in the grid while z is not assumed.
            # Get smallest and largest index for selecting func vals on this specific z-slice.
            # The `1` and `self.num_puts[2] - 2` is needed because I don't want the boundary.
            small_index = self._indices_to_index((x_index, y_index, 1))
            large_index = self._indices_to_index(
                (x_index, y_index, self.num_pts[2] - 2)
            )
            val = CubicSpline(
                oned_grids[2].points[1 : self.num_pts[2] - 2],
                func_values[small_index:large_index],
            )(z, nu)

            if nu == 1:
                # Derivative in real-space with respect to z.
                return (jac[:, 2] * val)[2]
            return val

        # Interpolate the Y-Axis based on x coordinate in grid.
        def y_splines(y, x_index, z):
            # The `1` and `self.num_puts[1] - 2` is needed because I don't want the boundary.
            # Assumes x_index is in the grid while y, z may not be.
            val = CubicSpline(
                oned_grids[1].points[1 : self.num_pts[2] - 2],
                [
                    z_spline(z, x_index, y_index)
                    for y_index in range(1, self.num_pts[1] - 2)
                ],
            )(y, nu)
            if nu == 1:
                # Derivative in real-space with respect to y.
                return (jac[:, 1] * val)[1]
            return val

        # Interpolate the X-Axis.
        def x_spline(x, y, z):
            # x, y, z may not be in the grid.
            val = CubicSpline(
                oned_grids[0].points[1 : self.num_pts[2] - 2],
                [y_splines(y, x_index, z) for x_index in range(1, self.num_pts[0] - 2)],
            )(x, nu)
            if nu == 1:
                # Derivative in real-space with respect to x.
                return (jac[:, 0] * val)[0]
            return val

        interpolated = x_spline(theta_pt[0], theta_pt[1], theta_pt[2])
        return interpolated

    def jacobian(self, real_pt):
        r"""
        Jacobian of the transformation from real space to theta space.

        Precisely, it is the lower-triangular matrix

        .. math::
            \begin{bmatrix}
                \frac{\partial \theta_x}{\partial X} & 0 & 0 \\
                \frac{\partial \theta_y}{\partial X} & \frac{\partial \theta_y}{\partial Y} & 0 \\
                \frac{\partial \theta_z}{\partial X} & \frac{\partial \theta_z}{\partial Y} &
                \frac{\partial \theta_z}{\partial Z}
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

        # Distance to centers/nuclei`s and Prefactors.
        diff_coords = real_pt - coords
        diff_squared = diff_coords ** 2.0
        # If i_var is zero, then distance is just all zeros.
        for i_var in range(0, self.promol.dim):

            # Basic-Level arrays for integration and derivatives.
            (
                distance,
                single_gauss,
                integrate_till_pt_x,
                transf_num,
                transf_den,
            ) = self.promol.helper_for_derivatives(diff_squared, diff_coords, i_var)

            for j_deriv in range(0, i_var + 1):
                if i_var == j_deriv:
                    # Derivative eliminates `integrate_till_pt_x`, and adds a Gaussian.
                    inner_term = single_gauss * np.exp(
                        -e_m * diff_squared[:, i_var][:, np.newaxis]
                    )
                    jacobian[i_var, i_var] = np.sum(inner_term) / transf_den
                elif j_deriv < i_var:
                    # Derivative of inside of Gaussian.
                    deriv_inside = self.promol.derivative_gaussian(diff_coords, j_deriv)
                    deriv_num = np.sum(
                        single_gauss * integrate_till_pt_x * deriv_inside
                    )
                    deriv_den = np.sum(single_gauss * deriv_inside * pi_over_exps)
                    # Quotient Rule
                    jacobian[i_var, j_deriv] = (
                        deriv_num * transf_den - transf_num * deriv_den
                    )
                    jacobian[i_var, j_deriv] /= transf_den ** 2.0

        return 2.0 * jacobian

    def hessian(self, real_pt):
        r"""
        Hessian of the transformation.

        The Hessian :math:`H` is a three-dimensional array with (i, j, k)th entry:

        .. math::
            H_{i, j, k} = \frac{\partial^2 \theta_i(x_0, \cdots, x_{i-1}}{\partial x_i \partial x_j}

            \text{where } (x_0, x_1, x_2) := (x, y, z).

        Parameters
        ----------
        real_pt : np.ndarray(3,)
            Real point in :math:`\mathbb{R}^3`.

        Returns
        -------
        hessian : np.ndarray(3, 3, 3)
            The (i, j, k)th entry is the partial derivative of the ith transformation function
            with respect to the jth, kth coordinate.  e.g. when i = 0, then hessian entry at
            (i, j, k) is zero unless j = k = 0.

        """
        hessian = np.zeros((self.dim, self.dim, self.dim), dtype=np.float64)

        c_m, e_m, coords, pi_over_exps, dim = astuple(self.promol)

        # Distance to centers/nuclei`s and Prefactors.
        diff_coords = real_pt - coords
        diff_squared = diff_coords ** 2.0

        # i_var is the transformation to theta-space.
        # j_deriv is the first partial derivative wrt x, y, z.
        # k_deriv is the second partial derivative wrt x, y, z.
        for i_var in range(0, 3):

            # Basic-Level arrays for integration and derivatives.
            (
                distance,
                single_gauss,
                integrate_till_pt_x,
                transf_num,
                transf_den,
            ) = self.promol.helper_for_derivatives(diff_squared, diff_coords, i_var)

            for j_deriv in range(0, i_var + 1):
                for k_deriv in range(0, i_var + 1):
                    # num is the numerator of transformation function.
                    # den is the denominator of transformation function.
                    # dnum_dk is the derivative of numerator wrt to k_deriv.
                    # dnum_dkdj is the derivative of num wrt to j_deriv then k_deriv.
                    # The derivative will store the result and pass it to the Hessian.
                    derivative = 0.0

                    if i_var == j_deriv:
                        gauss_extra = single_gauss * np.exp(
                            -e_m * diff_squared[:, j_deriv][:, np.newaxis]
                        )
                        if j_deriv == k_deriv:
                            # Diagonal derivative e.g. d(theta_X)(dx dx)
                            gauss_extra *= self.promol.derivative_gaussian(
                                diff_coords, j_deriv
                            )
                            derivative = np.sum(gauss_extra) / transf_den
                        else:
                            # Partial derivative of diagonal derivative e.g. d^2(theta_y)(dy dx).
                            deriv_inside = self.promol.derivative_gaussian(
                                diff_coords, k_deriv
                            )
                            dnum_dkdj = np.sum(gauss_extra * deriv_inside)
                            dden_dk = np.sum(single_gauss * deriv_inside * pi_over_exps)
                            # Numerator is different from `transf_num` since Gaussian is added.
                            dnum_dj = np.sum(gauss_extra)
                            # Quotient Rule
                            derivative = dnum_dkdj * transf_den - dnum_dj * dden_dk
                            derivative /= transf_den ** 2.0

                    # Here, quotient rule all the way down.
                    elif j_deriv < i_var:
                        if k_deriv == i_var:
                            gauss_extra = single_gauss * np.exp(
                                -e_m * diff_squared[:, k_deriv][:, np.newaxis]
                            )
                            deriv_inside = self.promol.derivative_gaussian(
                                diff_coords, j_deriv
                            )
                            ddnum_djdi = np.sum(gauss_extra * deriv_inside)
                            dden_dj = np.sum(single_gauss * deriv_inside * pi_over_exps)
                            # Quotient Rule
                            dnum_dj = np.sum(gauss_extra)
                            derivative = ddnum_djdi * transf_den - dnum_dj * dden_dj
                            derivative /= transf_den ** 2.0

                        elif k_deriv == j_deriv:
                            # Double Quotient Rule.
                            # See wikipedia "Quotient Rules Higher order formulas".
                            deriv_inside = self.promol.derivative_gaussian(
                                diff_coords, k_deriv
                            )
                            dnum_dj = np.sum(
                                single_gauss * integrate_till_pt_x * deriv_inside
                            )
                            dden_dj = np.sum(single_gauss * pi_over_exps * deriv_inside)

                            prod_rule = deriv_inside ** 2.0 - 2.0 * e_m
                            sec_deriv_num = np.sum(
                                single_gauss * integrate_till_pt_x * prod_rule
                            )
                            sec_deriv_den = np.sum(
                                single_gauss * pi_over_exps * prod_rule
                            )

                            output = sec_deriv_num * transf_den - dnum_dj * dden_dj
                            output /= transf_den ** 2.0
                            quot = transf_den * (
                                dnum_dj * dden_dj + transf_num * sec_deriv_den
                            )
                            quot -= 2.0 * transf_num * dden_dj * dden_dj
                            derivative = output - quot / transf_den ** 3.0

                        elif k_deriv != j_deriv:
                            # K is i_Sec_diff and i is i_diff
                            deriv_inside = self.promol.derivative_gaussian(
                                diff_coords, j_deriv
                            )
                            deriv_inside_sec = self.promol.derivative_gaussian(
                                diff_coords, k_deriv
                            )
                            gauss_and_inte_x = single_gauss * integrate_till_pt_x
                            gauss_and_inte = single_gauss * pi_over_exps

                            dnum_di = np.sum(gauss_and_inte_x * deriv_inside)
                            dden_di = np.sum(gauss_and_inte * deriv_inside)

                            dnum_dk = np.sum(gauss_and_inte_x * deriv_inside_sec)
                            dden_dk = np.sum(gauss_and_inte * deriv_inside_sec)

                            ddnum_dkdk = np.sum(
                                gauss_and_inte_x * deriv_inside * deriv_inside_sec
                            )
                            ddden_dkdk = np.sum(
                                gauss_and_inte * deriv_inside * deriv_inside_sec
                            )

                            output = ddnum_dkdk / transf_den
                            output -= dnum_di * dden_dk / transf_den ** 2.0
                            product = dnum_dk * dden_di + transf_num * ddden_dkdk
                            derivative = output
                            derivative -= product / transf_den ** 2.0
                            derivative += (
                                2.0 * transf_num * dden_di * dden_dk / transf_den ** 3.0
                            )

                    # The 2.0 is needed because we're in [-1, 1] rather than [0, 1].
                    hessian[i_var, j_deriv, k_deriv] = 2.0 * derivative

        return hessian

    def _transform(self, oned_grids):
        # Transform the entire grid.
        # Indices (i, j, k) start from bottom, left-most corner of the [-1, 1]^3 cube.
        counter = 0
        for ix in range(self.num_pts[0]):
            cart_pt = [None, None, None]
            theta_x = oned_grids[0].points[ix]

            brack_x = self._get_bracket((ix,), 0)
            cart_pt[0] = _inverse_coordinate(theta_x, 0, cart_pt, self.promol, brack_x)

            for iy in range(self.num_pts[1]):
                theta_y = oned_grids[1].points[iy]

                brack_y = self._get_bracket((ix, iy), 1)
                cart_pt[1] = _inverse_coordinate(
                    theta_y, 1, cart_pt, self.promol, brack_y
                )

                for iz in range(self.num_pts[2]):
                    theta_z = oned_grids[2].points[iz]

                    brack_z = self._get_bracket((ix, iy, iz), 2)
                    cart_pt[2] = _inverse_coordinate(
                        theta_z, 2, cart_pt, self.promol, brack_z
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
        # If it is a boundary point, then return nan. Done by indices.
        if (
            0 in indices[: i_var + 1]
            or (self.num_pts[i_var] - 1) in indices[: i_var + 1]
        ):
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

    def _index_to_indices(self, index):
        r"""
        Convert Index to Indices, ie integer m to (i, j, k) position of the Cubic Grid.

        Cubic Grid has shape (N_x, N_y, N_z) where N_x is the number of points
        in the x-direction, etc.  Then 0 <= i <= N_x - 1, 0 <= j <= N_y - 1, etc.

        Parameters
        ----------
        index : int
            Index of the grid point.

        Returns
        -------
        indices : (int, int, int)
            The ith, jth, kth position of the grid point.

        """
        assert index >= 0, "Index should be positive. %r" % index
        n_1d, n_2d = self.num_pts[2], self.num_pts[1] * self.num_pts[2]
        i = index // n_2d
        j = (index - n_2d * i) // n_1d
        k = index - n_2d * i - n_1d * j
        return i, j, k

    def _indices_to_index(self, indices):
        r"""
        Convert Indices to Index, ie (i, j, k) to a index/integer m.

        Parameters
        ----------
        indices : (int, int, int)
            The ith, jth, kth position of the grid point.

        Returns
        -------
        index : int
            Index of the grid point.

        """
        n_1d, n_2d = self.num_pts[2], self.num_pts[1] * self.num_pts[2]
        index = n_2d * indices[0] + n_1d * indices[1] + indices[2]
        return index


@dataclass
class _PromolParams:
    r"""
    Private class for Promolecular Density information.

    Contains helper-functions for Promolecular Transformation.
    They are coded as pipe-lines for this special purpose and
    the reason why "diff_coords" is chosen as a attribute rather
    than a generic "[x, y, z]" point.

    """

    c_m: np.ndarray  # Coefficients of Promolecular.
    e_m: np.ndarray  # Exponents of Promolecular.
    coords: np.ndarray  # Centers/Coordinates of Each Gaussian.
    pi_over_exponents: np.ndarray = field(init=False)
    dim: int = 3

    def __post_init__(self):
        r"""Initialize pi_over_exponents."""
        # Rather than computing this repeatedly. It is fixed.
        with np.errstate(divide="ignore"):
            self.pi_over_exponents = np.sqrt(np.pi / self.e_m)
            self.pi_over_exponents[self.e_m == 0.0] = 0.0

    def integrate_all(self):
        r"""Integration of Gaussian over Entire Real space ie :math:`\mathbb{R}^D`."""
        return np.sum(self.c_m * self.pi_over_exponents ** self.dim)

    def derivative_gaussian(self, diff_coords, j_deriv):
        r"""Return derivative of single Gaussian but without exponential."""
        return -self.e_m * 2.0 * diff_coords[:, j_deriv][:, np.newaxis]

    def integration_gaussian_till_point(self, diff_coords, i_var, with_factor=False):
        r"""Integration of Gaussian wrt to `i_var` variable till a point (inside diff_coords)."""
        coord_ivar = diff_coords[:, i_var][:, np.newaxis]
        integration = (erf(np.sqrt(self.e_m) * coord_ivar) + 1.0) / 2.0
        if with_factor:
            # Included the (pi / exponents), this becomes the actual integral.
            # Not including the (pi / exponents) increasing computation slightly faster.
            return integration * self.pi_over_exponents
        return integration

    def single_gaussians(self, distance):
        r"""Return matrix with entries a single gaussian evaluated at the float distance."""
        return self.c_m * np.exp(-self.e_m * distance)

    def promolecular(self, points):
        r"""
        Evaluate the promolecular density over a grid.

        Parameters
        ----------
        points : np.ndarray(N, D)
            Points in :math:`\mathbb{R}^D`.

        Returns
        -------
        np.ndarray(N,) :
            Promolecular density evaluated at the points.

        """
        # M is the number of centers/atoms.
        # D is the number of dimensions, usually 3.
        # K is maximum number of gaussian functions over all M atoms.
        cm, em, coords = self.c_m, self.e_m, self.coords
        # Shape (N, M, D), then Summing gives (N, M, 1)
        distance = np.sum(
            (points - coords[:, np.newaxis]) ** 2.0, axis=2, keepdims=True
        )
        # At each center, multiply Each Distance of a Coordinate, with its exponents.
        exponen = np.exp(-np.einsum("MND, MK-> MNK", distance, em))
        # At each center, multiply the exponential with its coefficients.
        gaussian = np.einsum("MNK, MK -> MNK", exponen, cm)
        # At each point, sum for each center, then sum all centers together.
        return np.einsum("MNK -> N", gaussian, dtype=np.float64)

    def helper_for_derivatives(self, diff_squared, diff_coords, i_var):
        r"""
        Return Arrays for computing the derivative of transformation functions wrt x, y, z.

        Parameters
        ----------
        diff_squared : np.ndarray
            The squared of difference of position to the center of the Promoleculars.
        diff_coords : np.ndarray
            The difference of position to the center of the Promoleculars. This is the square
            root of `diff_squared`.
        i_var : int
            Index of one of x, y, z.

        Returns
        -------
        distance : np.ndarray
            The squared distance from the position to the center of the Promoleculars.
        single_gauss : np.ndarray
            Array with entries of a single Gaussian e^(-a distance) with factor (pi / a).
        integrate_till_pt_x  : np.ndarray
            Integration of a Gaussian from -inf to x, ie
                (pi / a)^0.5 * (erf(a^0.5 (x - center of Gaussian) + 1) / 2
        transf_num : float
            The numerator of the transformation. Mostly used for quotient rule.
        transf_den : float
            The denominator of the transformation. Mostly used for quotient rule.

        """
        distance = np.sum(diff_squared[:, :i_var], axis=1)[:, np.newaxis]

        # Gaussian Integrals Over Entire Space For Numerator and Denomator.
        single_gauss = self.single_gaussians(distance)
        single_gauss *= self.pi_over_exponents ** (self.dim - i_var - 1)

        # Get integral of Gaussian till a point.
        integrate_till_pt_x = self.integration_gaussian_till_point(
            diff_coords, i_var, with_factor=True
        )
        # Numerator and Denominator of Original Transformation.
        transf_num = np.sum(single_gauss * integrate_till_pt_x)
        transf_den = np.sum(single_gauss * self.pi_over_exponents)
        return distance, single_gauss, integrate_till_pt_x, transf_num, transf_den


def _transform_coordinate(real_pt, i_var, promol):
    r"""
    Transform the `i_var` coordinate of a real point to [-1, 1] using promolecular density.

    Parameters
    ----------
    real_pt : np.ndarray(D,)
        Real point being transformed.
    i_var : int
        Index that is being tranformed. Less than D.
    promol : _PromolParams
        Promolecular Data Class.

    Returns
    -------
    theta_pt : float
        The transformed point in :math:`[-1, 1]`.

    """
    _, _, coords, pi_over_exps, dim = astuple(promol)

    # Distance to centers/nuclei`s and Prefactors.
    diff_coords = real_pt[: i_var + 1] - coords[:, : i_var + 1]
    diff_squared = diff_coords ** 2.0
    distance = np.sum(diff_squared[:, :i_var], axis=1)[:, np.newaxis]
    # If i_var is zero, then distance is just all zeros.

    # Single Gaussians Including Integration of Exponential over `(dim - i_var)` variables.
    single_gauss = promol.single_gaussians(distance) * pi_over_exps ** (dim - i_var)

    # Get the integral of Gaussian till a point excluding a prefactor.
    # prefactor (pi / exponents) is included in `gaussian_integrals`.
    cdf_gauss = promol.integration_gaussian_till_point(
        diff_coords, i_var, with_factor=False
    )

    # Final Result.
    transf_num = np.sum(single_gauss * cdf_gauss)
    transf_den = np.sum(single_gauss)
    transform_value = transf_num / transf_den

    # -1. + 2. is needed to transform to [-1, 1], rather than [0, 1].
    return -1.0 + 2.0 * transform_value


def _root_equation(init_guess, prev_trans_pts, theta_pt, i_var, promol):
    r"""
    Equation to solve for the root to find inverse coordinate from theta space to Real space.

    Parameters
    ----------
    init_guess : float
        Initial guess of Real point that transforms to `theta_pt`.
    prev_trans_pts : list[`i_var` - 1]
        The previous points in real-space that were already transformed.
    theta_pt : float
        The point in [-1, 1] being transformed to the Real space.
    i_var : int
        Index of variable being transformed.
    promol : _PromolParams
        Promolecular Data Class.

    Returns
    -------
    float :
        The difference between `theta_pt` and the transformed point based on
        `init_guess` and `prev_trans_pts`.

    """
    all_points = np.append(prev_trans_pts, init_guess)
    transf_pt = _transform_coordinate(all_points, i_var, promol)
    return theta_pt - transf_pt


def _inverse_coordinate(theta_pt, i_var, transformed, promol, bracket=(-10, 10)):
    r"""
    Transform a point in [-1, 1] to the real space corresponding to `i_var` variable.

    Parameters
    ----------
    theta_pt : float
        Point in [-1, 1].
    i_var : int
        Index that is being tranformed. Less than D.
    transformed : list[`i_var` - 1]
        The set of previous points before index `i_var` that were transformed to real space.
    promol : _PromolParams
        Promolecular Data Class.
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
    # Check's if this is a boundary points which is mapped to np.nan
    # These two conditions are added for individual point transformation.
    if np.abs(theta_pt - -1.0) < 1e-10:
        return np.nan
    if np.abs(theta_pt - 1.0) < 1e-10:
        return np.nan
    # This condition is added for transformation of the entire grid.
    # The [:i_var] is needed because of the way I've set-up transforming points in _transform.
    # Likewise for the bracket, see the function `get_bracket`.
    if np.nan in bracket or np.nan in transformed[:i_var]:
        return np.nan

    def _dynamic_bracketing(l_bnd, u_bnd, maxiter=50):
        r"""Dynamically changes the lower (or upper bound) to have different sign values."""
        bounds = [l_bnd, u_bnd]

        def is_same_sign(x, y):
            return (x >= 0 and y >= 0) or (x < 0 and y < 0)

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

        return tuple(bounds)

    # Set up Arguments for root_equation with dynamic bracketing.
    args = (transformed[:i_var], theta_pt, i_var, promol)
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
