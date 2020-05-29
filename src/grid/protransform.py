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


def transform_coordinate(real_pt, i_var, promol_params, deriv=False, sderiv=False):
    r"""
    Transform the `i_var` coordinate in a real point to [0, 1]^D using promolecular.

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
