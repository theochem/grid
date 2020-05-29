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
r"""Promolecular Grid Transformation"""


from collections import namedtuple

from grid.basegrid import Grid

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import erf


PromolParams = namedtuple("PromolParams", ["c_m", "e_m", "coords", "dim", "pi_over_exponents"])


class ProCubicTransform(Grid):
    r"""
    Promolecular Grid Transformation of a Cubic Grid in [0,1]^3.

    Attributes
    ----------
    num_pts : (int, int, int)
        The number of points in x, y, and z direction.
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

    Examples
    --------
    Define information of the Promolecular Density.
    >> c = np.array([[5.], [10.]])
    >> e = np.array([[2.], [3.]])
    >> coord = np.array([[0., 0., 0.], [2., 2., 2.]])

    Define information of the grid and its weights.
    >> stepsize = 0.01
    >> weights = np.array([0.01] * 101**3)  # Simple Riemannian weights.
    >> promol = ProCubicTransform([ss] * 3, weights, c, e, coord)

    To integrate some function f.
    >> def func(pt):
    >>    return np.exp(-0.1 * np.linalg.norm(pt, axis=1)**2.)
    >> func_values = func(promol.points)
    >> print("The integral is %.4f" % promol.integrate(func_values, trick=False)

    References
    ----------
    .. [1] J. I. Rodríguez, D. C. Thompson, P. W. Ayers, and A. M. Koster, "Numerical integration
            of exchange-correlation energies and potentials using transformed sparse grids."

    Notes
    -----

    """
    def __init__(self, stepsize, weights, coeffs, exps, coords):
        self._ss = stepsize
        self._num_pts = (int(1 / stepsize[0]) + 1,
                         int(1. / stepsize[1]) + 1,
                         int(1. / stepsize[2]) + 1)

        # pad coefficients and exponents with zeros to have the same size.
        coeffs, exps = _pad_coeffs_exps_with_zeros(coeffs, exps)
        # Rather than computing this repeatedly. It is fixed.
        with np.errstate(divide='ignore'):
            pi_over_exponents = np.sqrt(np.pi / exps)
            pi_over_exponents[exps == 0] = 0
        self._prointegral = np.sum(coeffs * pi_over_exponents ** (1.5))
        self._promol = PromolParams(coeffs, exps, coords, 3, pi_over_exponents)

        # initialize parent class
        empty_points = np.empty((np.prod(self._num_pts), 3), dtype=np.float64)
        super().__init__(empty_points, weights * self._prointegral)
        self._transform()

    @property
    def num_pts(self):
        r"""Number of points in each direction."""
        return self._num_pts

    @property
    def ss(self):
        r"""Stepsize of the cubic grid."""
        return self._ss

    @property
    def prointegral(self):
        r"""Integration of Promolecular density."""
        return self._prointegral

    @property
    def promol(self):
        r"""PromolParams namedTuple."""
        return self._promol

    def integrate(self, *value_arrays, trick=False):
        r"""
        Integrate any function.

        Parameters
        ----------
        *value_arrays : np.ndarray(N, )
            One or multiple value array to integrate.
        trick : bool
            If true, uses the promolecular trick.

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

        """
        promolecular = self._promolecular(self.points)
        integrands = []
        with np.errstate(divide='ignore'):
            for arr in value_arrays:
                if trick:
                    integrand = (arr - promolecular) / promolecular
                else:
                    integrand = arr / promolecular
                integrand[np.isnan(self.points).any(axis=1)] = 0.
                integrands.append(arr)
        if trick:
            return self._prointegral + super().integrate(*integrands)
        return super().integrate(*integrands)

    def _promolecular(self, grid):
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
        # TODO: For Design, Store this or constantly re-evaluate it?
        # M is the number of centers/atoms.
        # D is the number of dimensions, usually 3.
        # K is maximum number of gaussian functions over all M atoms.
        cm, em, coords, _, _ = self.promol
        # Shape (N, M, D), then Summing gives (N, M, 1)
        distance = np.sum((grid - coords[:, np.newaxis])**2., axis=2, keepdims=True)
        # At each center, multiply Each Distance of a Coordinate, with its exponents.
        exponen = np.exp(-np.einsum("MND, MK-> MNK" , distance, em))
        # At each center, multiply the exponential with its coefficients.
        gaussian = np.einsum("MNK, MK -> MNK", exponen, cm)
        # At each point, summing the gaussians for each center, then summing all centers together.
        return np.einsum("MNK -> N", gaussian)

    def _transform(self):
        counter = 0
        for ix in range(self.num_pts[0]):
            cart_pt = [None, None, None]
            unit_x = self.ss[0] * ix

            initx = self._get_bracket((ix,), 0)
            transformx = inverse_coordinate(unit_x, 0, self.promol, cart_pt, initx)
            cart_pt[0] = transformx

            for iy in range(self.num_pts[1]):
                unit_y = self.ss[1] * iy

                inity = self._get_bracket((ix, iy), 1)
                transformy = inverse_coordinate(unit_y, 1, self.promol, cart_pt, inity)
                cart_pt[1] = transformy

                for iz in range(self.num_pts[2]):
                    unit_z = self.ss[2] * iz

                    initz = self._get_bracket((ix, iy, iz), 2)
                    transformz = inverse_coordinate(unit_z, 2, self.promol, cart_pt, initz)
                    cart_pt[2] = transformz
                    self.points[counter] = cart_pt.copy()
                    counter += 1

    def _get_bracket(self, coord, i_var):
        # If it is a boundary point, then return nan.
        if 0. in coord[:i_var + 1] or (self.num_pts[i_var] - 1) in coord[:i_var + 1]:
            return np.nan, np.nan
        # If it is a new point, with no nearby point, get a large initial guess.
        elif coord[i_var] == 1:
            min = (np.min(self.promol.coords[:, i_var]) - 3.) * 20.
            max = (np.max(self.promol.coords[:, i_var]) + 3.) * 20.
            return min, max
        # If the previous point has been converted, use that as a initial guess.
        if i_var == 0:
            index = (coord[0] - 1) * self.num_pts[1] * self.num_pts[2]
        elif i_var == 1:
            index = coord[0] * self.num_pts[1] * self.num_pts[2] + self.num_pts[2] * (coord[1] - 1)
        elif i_var == 2:
            index = (coord[0] * self.num_pts[1] * self.num_pts[2] +
                     self.num_pts[2] * coord[1] + coord[2] - 1)

        # FIXME : Rather than using fixed +10., use truncated taylor series.
        return self.points[index, i_var], self.points[index, i_var] + 10.


def transform_coordinate(real_pt, i_var, promol_params, deriv=False, sderiv=False):
    r"""
    Transform the `i_var` coordinate in a real point to [0, 1] using promolecular density.

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
    c_m, e_m, coords, dim, pi_over_exps = promol_params

    # Distance to centers/nuclei`s and Prefactors.
    diff_coords = real_pt[:i_var + 1] - coords[:, :i_var + 1]
    diff_squared = diff_coords**2.
    distance = np.sum(diff_squared[:, :i_var], axis=1)[:, np.newaxis]
    # If i_var is zero, then distance is just all zeros.

    # Gaussian Integrals Over Entire Space For Numerator and Denomator.
    gaussian_integrals = np.exp(-e_m * distance) * pi_over_exps**(dim - i_var)
    coeff_num = c_m * gaussian_integrals

    # Get the integral of Gaussian till a point.
    coord_ivar = diff_coords[:, i_var][:, np.newaxis]
    integrate_till_pt_x = (erf(np.sqrt(e_m) * coord_ivar) + 1.) / 2.

    # Final Result.
    transf_num = np.sum(coeff_num * integrate_till_pt_x)
    transf_den = np.sum(coeff_num)
    transform_value = transf_num / transf_den

    if deriv:
        inner_term = coeff_num * np.exp(-e_m * diff_squared[:, i_var][:, np.newaxis])
        deriv = np.sum(inner_term) / transf_den

        if sderiv:
            sderiv = np.sum(inner_term * -e_m * 2. * coord_ivar) / transf_den
            return transform_value, deriv, sderiv
        return transform_value, deriv
    return transform_value


def _root_equation(init_guess, prev_trans_pts, theta_pt, i_var, params):
    all_points = np.append(prev_trans_pts, init_guess)
    transf_pt = transform_coordinate(all_points, i_var, params)
    return theta_pt - transf_pt


def inverse_coordinate(theta_pt, i_var, params, transformed, bracket=(-10, 10)):
    r"""
    Transform a point in [0, 1] to the real space corresponding to the `i_var` variable.

    Parameters
    ----------
    theta_pt : float
        Point in [0, 1].
    i_var : int
        Index that is being tranformed. Less than D.
    promol_params : namedTuple
        Data about the Promolecular density.
    transformed : list(`i_var` - 1)
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
    AssertionError :    If the root did not converge, or brackets did not have opposite sign.

    """
    # The [:i_var] is needed because of the way I've set-up transformed attribute.
    if np.isnan(bracket[0]) or np.nan in transformed[:i_var]:
        return np.nan
    args = (transformed[:i_var], theta_pt, i_var, params)
    root_result = root_scalar(_root_equation, args=args, method="brentq",
                              bracket=[bracket[0], bracket[1]], maxiter=50, xtol=2e-15)
    assert root_result.converged
    return root_result.root


def _pad_coeffs_exps_with_zeros(coeffs, exps):
    max_numb_of_gauss = max(len(c) for c in coeffs)
    coeffs = np.array([np.pad(a, (0, max_numb_of_gauss - len(a)), 'constant',
                              constant_values=0.) for a in coeffs], dtype=np.float64)
    exps = np.array([np.pad(a, (0, max_numb_of_gauss - len(a)), 'constant',
                            constant_values=0.) for a in exps], dtype=np.float64)
    return coeffs, exps
