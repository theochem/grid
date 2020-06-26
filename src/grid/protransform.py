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
from scipy.linalg import solve_triangular
from scipy.optimize import root_scalar
from scipy.special import erf

__all__ = ["CubicProTransform"]


class CubicProTransform(Grid):
    r"""
    Promolecular Grid Transformation of a Cubic Grid.

    Grid is three dimensional and modeled as Tensor Product of Three, one dimensional grids.
    Theta space is defined to be :math:`[-1, 1]^3`.
    Real space is defined to be :math:`\mathbb{R}^3.`

    Attributes
    ----------
    num_pts : (int, int, int)
        The number of points, including both of the end/boundary points, in x, y, and z direction.
        This is calculated as `int(1. / ss[i]) + 1`.
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
        Jacobian of the transformation from Real space to Theta space :math:`[-1, 1]^3`.
    steepest_ascent_theta()
        Direction of steepest-ascent of a function in theta space from gradient in real space.
    transform():
        Transform Real point to theta point :math:`[-1, 1]^3`.
    inverse(bracket=(-10, 10))
        Transform theta point to Real space :math:`\mathbb{R}^3`.

    Examples
    --------
    Define information of the Promolecular Density.
    >> c = np.array([[5.], [10.]])
    >> e = np.array([[2.], [3.]])
    >> coord = np.array([[0., 0., 0.], [2., 2., 2.]])

    Define information of the grid and its weights.
    >> from grid.onedgrid import GaussChebyshev

    >> numb_x = 50
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
    TODO: Insert Info About Conditional Distribution Method.
    TODO: Add Infor about how boundarys on theta-space are mapped to np.nan.

    """
    def __init__(self, oned_grids, coeffs, exps, coords):
        if not isinstance(oned_grids, list):
            raise TypeError("oned_grid should be of type list.")
        if not np.all([isinstance(grid, OneDGrid) for grid in oned_grids]):
            raise TypeError("Grid in oned_grids should be of type `OneDGrid`.")
        if not np.all([grid.domain == (-1, 1.) for grid in oned_grids]):
            raise ValueError("One Dimensional grid domain should be (-1, 1).")
        if not len(oned_grids) == 3:
            raise ValueError("There should be three One-Dimensional grids in `oned_grids`.")

        self._num_pts = tuple([grid.size for grid in oned_grids])
        self._dim = len(oned_grids)

        # pad coefficients and exponents with zeros to have the same size, easier to use numpy.
        coeffs, exps = _pad_coeffs_exps_with_zeros(coeffs, exps)
        self._promol = _PromolParams(coeffs, exps, coords, self._dim)
        self._prointegral = self._promol.integrate_all()

        empty_pts = np.empty((np.prod(self._num_pts), self._dim), dtype=np.float64)
        weights = np.kron(
            np.kron(oned_grids[0].weights, oned_grids[1].weights),
            oned_grids[2].weights
        )
        # The prointegral is needed because of promolecular integration.
        # Divide by 8 needed because the grid is in [-1, 1] rather than [0, 1].
        super().__init__(empty_pts, weights * self._prointegral / 2.0**self._dim)
        # Transform Cubic Grid in Theta-Space to Real-space.
        self._transform(oned_grids)

    @property
    def dim(self):
        r"""Return the dimension of the cubic grid."""
        return self._dim

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
            [_transform_coordinate(real_pt, i, self.promol)
             for i in range(0, self.promol.dim)]
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

    def differentiation_interpolation(self, real_pt, func_val, use_log=False):
        r"""
        Differentiate a point in Real-space using interpolation.

        Parameters
        ----------
        real_pt : ndarray(3,)

        func_val : float

        use_log : bool

        """
        # Map to theta space.

        # Construct Cubic Splines and differentiate.

        # Convert back to Real-Space.
        pass

    def integration_interpolation(self):
        pass

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
            distance = np.sum(diff_squared[:, :i_var], axis=1)[:, np.newaxis]

            # Gaussian Integrals Over Entire Space For Numerator and Denomator.
            single_gauss = self.promol.single_gaussians(distance)
            single_gauss *= pi_over_exps ** (dim - i_var - 1)

            # Get integral of Gaussian till a point.
            integrate_till_pt_x = self.promol.integration_gaussian_till_point(diff_coords,
                                                                              i_var,
                                                                              with_factor=True)
            # Numerator and Denominator of Original Transformation.
            transf_num = np.sum(single_gauss * integrate_till_pt_x)
            transf_den = np.sum(single_gauss * pi_over_exps)

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
        hessian = np.zeros((self.dim, self.dim, self.dim), dtype=np.float64)

        c_m, e_m, coords, pi_over_exps, dim = astuple(self.promol)

        # Distance to centers/nuclei`s and Prefactors.
        diff_coords = real_pt - coords
        diff_squared = diff_coords ** 2.0

        # i_var is the transformation to theta-space.
        # j_deriv is the first partial derivative wrt x, y, z.
        # k_deriv is the second partial derivative wrt x, y, z.
        for i_var in range(0, 3):
            distance = np.sum(diff_squared[:, :i_var], axis=1)[:, np.newaxis]

            # Gaussian Integrals Over Entire Space For Numerator and Denomator.
            single_gauss = self.promol.single_gaussians(distance)
            single_gauss *= pi_over_exps ** (dim - i_var - 1)

            # Get integral of Gaussian till a point.
            integrate_till_pt_x = self.promol.integration_gaussian_till_point(diff_coords,
                                                                              i_var,
                                                                              with_factor=True)
            # Numerator and Denominator of Original Transformation.
            transf_num = np.sum(single_gauss * integrate_till_pt_x)
            transf_den = np.sum(single_gauss * pi_over_exps)
            for j_deriv in range(0, i_var + 1):
                for k_deriv in range(0, i_var + 1):
                    derivative = 0.

                    if i_var == j_deriv:
                        gauss_extra = single_gauss * np.exp(
                            -e_m * diff_squared[:, j_deriv][:, np.newaxis]
                        )
                        if j_deriv == k_deriv:
                            # Diagonal derivative e.g. d(theta_X)(dx dx)
                            gauss_extra *= self.promol.derivative_gaussian(diff_coords, j_deriv)
                            derivative = np.sum(gauss_extra) / transf_den
                        else:
                            # Partial derivative of diagonal derivative e.g. d^2(theta_y)(dy dx).
                            deriv_inside = self.promol.derivative_gaussian(diff_coords, k_deriv)
                            deriv_num = np.sum(
                                gauss_extra * deriv_inside
                            )
                            deriv_den = np.sum(single_gauss * deriv_inside * pi_over_exps)
                            # Numerator is different from `transf_num` since Gaussian is added.
                            new_numerator = np.sum(gauss_extra)
                            # Quotient Rule
                            derivative = (
                                    deriv_num * transf_den - new_numerator * deriv_den
                            )
                            derivative /= transf_den ** 2.0

                    # Here, quotient rule all the way down.
                    elif j_deriv < i_var:
                        if k_deriv == i_var:
                            gauss_extra = single_gauss * np.exp(
                                -e_m * diff_squared[:, k_deriv][:, np.newaxis]
                            )

                            deriv_inside = self.promol.derivative_gaussian(diff_coords, j_deriv)
                            ddnum_djdi = np.sum(
                                gauss_extra * deriv_inside
                            )

                            dden_dj = np.sum(single_gauss * deriv_inside * pi_over_exps)
                            # Quotient Rule
                            dnum_dj = np.sum(gauss_extra)
                            derivative = ddnum_djdi * transf_den - dnum_dj * dden_dj
                            derivative /= transf_den ** 2.0

                        elif k_deriv == j_deriv:
                            # Double Quotient Rule.
                            # See wikipedia "Quotient Rules Higher order formulas".
                            deriv_inside = self.promol.derivative_gaussian(diff_coords, k_deriv)
                            dnum_dj = np.sum(single_gauss * integrate_till_pt_x * deriv_inside)
                            dden_dj = np.sum(single_gauss * pi_over_exps * deriv_inside)

                            prod_rule = deriv_inside ** 2.0 - 2.0 * e_m
                            sec_deriv_num = np.sum(single_gauss * integrate_till_pt_x * prod_rule)
                            sec_deriv_den = np.sum(single_gauss * pi_over_exps * prod_rule)

                            output = (sec_deriv_num * transf_den - dnum_dj * dden_dj)
                            output /= transf_den ** 2.0
                            quot = transf_den * (dnum_dj * dden_dj + transf_num * sec_deriv_den)
                            quot -= 2.0 * transf_num * dden_dj * dden_dj
                            derivative = output - quot / transf_den ** 3.0

                        elif k_deriv != j_deriv:
                            # K is i_Sec_diff and i is i_diff
                            deriv_inside = self.promol.derivative_gaussian(diff_coords, j_deriv)
                            deriv_inside_sec = self.promol.derivative_gaussian(diff_coords, k_deriv)

                            dnum_di = np.sum(single_gauss * integrate_till_pt_x * deriv_inside)
                            dden_di = np.sum(single_gauss * pi_over_exps * deriv_inside)

                            dnum_dk = np.sum(single_gauss * integrate_till_pt_x * deriv_inside_sec)
                            dden_dk = np.sum(single_gauss * pi_over_exps * deriv_inside_sec)

                            ddnum_dkdk = np.sum(single_gauss * deriv_inside * deriv_inside_sec * integrate_till_pt_x)
                            ddden_dkdk = np.sum(single_gauss * deriv_inside * deriv_inside_sec * pi_over_exps)

                            output = ddnum_dkdk / transf_den
                            output -= (dnum_di * dden_dk / transf_den ** 2.0)
                            product = dnum_dk * dden_di + transf_num * ddden_dkdk
                            derivative = output
                            derivative -= product * transf_den / transf_den ** 3.
                            derivative += 2.0 * transf_num * dden_di * dden_dk / transf_den ** 3.

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
                cart_pt[1] = _inverse_coordinate(theta_y, 1, cart_pt, self.promol, brack_y)

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

    def integrate_all_certain_variables(self):
        r""""""
        return self.pi_over_exponents ** ()

    def derivative_gaussian(self, diff_coords, j_deriv):
        r"""Derivative of single Gaussian but without exponential."""
        return -self.e_m * 2.0 * diff_coords[:, j_deriv][:, np.newaxis]

    def integration_gaussian_till_point(self, diff_coords, i_var, with_factor=False):
        r"""Integration of Gaussian wrt to `i_var` variable till a point (inside diff_coords)."""
        coord_ivar = diff_coords[:, i_var][:, np.newaxis]
        integration = (erf(np.sqrt(self.e_m) * coord_ivar) + 1.0) / 2.0
        if with_factor:
            # Included the (pi / exponents), this is the actual integral here.
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
        distance = np.sum((points - coords[:, np.newaxis]) ** 2.0, axis=2, keepdims=True)
        # At each center, multiply Each Distance of a Coordinate, with its exponents.
        exponen = np.exp(-np.einsum("MND, MK-> MNK", distance, em))
        # At each center, multiply the exponential with its coefficients.
        gaussian = np.einsum("MNK, MK -> MNK", exponen, cm)
        # At each point, sum for each center, then sum all centers together.
        return np.einsum("MNK -> N", gaussian, dtype=np.float64)


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
    cdf_gauss = promol.integration_gaussian_till_point(diff_coords, i_var,
                                                       with_factor=False)

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

    def _dynamic_bracketing(l_bnd, u_bnd, maxiter=50):
        r"""Dynamically changes the lower (or upper bound) to have different sign values."""

        def is_same_sign(x, y):
            return (x >= 0 and y >= 0) or (x < 0 and y < 0)

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
    if np.abs(theta_pt - -1.0) < 1e-10:
        return np.nan
    if np.abs(theta_pt - 1.0) < 1e-10:
        return np.nan
    # This condition is added for transformation of the entire grid.
    # The [:i_var] is needed because of the way I've set-up transforming points in _transform.
    # Likewise for the bracket, see the function `get_bracket`.
    if np.nan in bracket or np.nan in transformed[:i_var]:
        return np.nan

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
