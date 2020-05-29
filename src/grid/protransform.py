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

from grid.rtransform import BaseTransform

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import erf


PromolParams = namedtuple("PromolParams", ["c_m", "e_m", "coords", "dim", "pi_over_exponents"])


class ProCubicTransform:
    def __init__(self, stepsize, weights, coeffs, exps, coords):
        self.stepsizes = stepsize
        self.num_pts = (int(1 / stepsize[0]) + 1,
                        int(1. / stepsize[1]) + 1,
                        int(1. / stepsize[2]) + 1)

        # pad coefficients and exponents with zeros to have the same size.
        coeffs, exps = _pad_coeffs_exps_with_zeros(coeffs, exps)

        # Rather than computing this repeatedly. It is fixed.
        with np.errstate(divide='ignore'):
            pi_over_exponents = np.sqrt(np.pi / exps)
            pi_over_exponents[exps == 0] = 0

        self.promol = PromolParams(coeffs, exps, coords, 3, pi_over_exponents)
        self.points = np.empty((np.prod(self.num_pts), 3), dtype=np.float64)
        self._transform()  # Fill out self.points.
        self.weights = weights

    def _transform(self):
        counter = 0
        for ix in range(self.num_pts[0]):
            cart_pt = [None, None, None]
            unit_x = self.stepsizes[0] * ix

            initx = self._get_bracket((ix,), 0)
            transformx = inverse_coordinate(unit_x, 0, self.promol, cart_pt, initx)
            cart_pt[0] = transformx

            for iy in range(self.num_pts[1]):
                unit_y = self.stepsizes[1] * iy

                inity = self._get_bracket((ix, iy), 1)
                transformy = inverse_coordinate(unit_y, 1, self.promol, cart_pt, inity)
                cart_pt[1] = transformy

                for iz in range(self.num_pts[2]):
                    unit_z = self.stepsizes[2] * iz

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
