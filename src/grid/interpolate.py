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
"""Interpolation module for evaluating function value at any point."""

import warnings

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.special import sph_harm


def generate_real_sph_harms(l_max, theta, phi):
    """Generate real spherical harmonics.

    Parameters
    ----------
    l_max : int
        largest angular degree
    theta : np.ndarray(N,)
        Azimuthal angles
    phi : np.ndarray(N,)
        Polar angles

    Returns
    -------
    np.ndarray(l_max * 2 + 1, l_max + 1, N)
        value of angular grid in each m, n spherical harmonics
    """
    sph_h = generate_sph_harms(l_max, theta, phi)
    return np.nan_to_num(_convert_ylm_to_zlm(sph_h))


def generate_sph_harms(l_max, theta, phi):
    """Generate complex spherical harmonics.

    Parameters
    ----------
    l_max : int
        largest angular degree
    theta : np.ndarray(N,)
        Azimuthal angles
    phi : np.ndarray(N,)
        Polar angles

    Returns
    -------
    np.ndarray(l_max * 2 + 1, l_max + 1, N)
        value of angular grid in each m, n spherical harmonics
    """
    # theta azimuthal, phi polar
    l, m = _generate_sph_paras(l_max)
    return sph_harm(m[:, None, None], l[None, :, None], theta, phi)


def _generate_sph_paras(l_max):
    """Generate proper l and m list for l.

    Parameters
    ----------
    l_max : int

    Returns
    -------
    list
        l = [0, 1, 2.., l_max], m = [0, 1, ... l_max, -l_max, ..., -1]
    """
    l_list = np.arange(l_max + 1)
    m_list_p = np.arange(l_max + 1)
    m_list_n = np.arange(-l_max, 0)
    m_list = np.append(m_list_p, m_list_n)
    return l_list, m_list


def _convert_ylm_to_zlm(sp_harm_arrs):
    """Converge complex spherical into real spherical harmonics."""
    ms, ls, arrs = sp_harm_arrs.shape  # got list of Ls, and Ms
    # ls = l_max + 1
    # ms = 2 * l_max + 1
    s_h_r = np.zeros((ms, ls, arrs))  # copy old array for construct real
    # silence cast warning for complex -> float
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_h_r[1:ls] = (sp_harm_arrs[1:ls] + np.conjugate(sp_harm_arrs[1:ls])) / np.sqrt(
            2
        )
        s_h_r[-ls + 1 :] = (
            sp_harm_arrs[-ls + 1 :] - np.conjugate(sp_harm_arrs[-ls + 1 :])
        ) / (np.sqrt(2) * 1j)
        s_h_r[0] = sp_harm_arrs[0]
    return s_h_r


def spline_with_sph_harms(sph_harm, value_arrays, weights, indices, radial):
    """Compute spline with real spherical harmonics.

    Parameters
    ----------
    sph_harm : np.ndarray(M, L, N)
        spherical harmonics values of m, l, n_points
    value_arrays : np.ndarray(N,)
        fuction values on each point
    weights : np.ndarray(N,)
        weights of each point on the grid
    indices : list[int]
        indices of each chank for each radial angular partsption
    radial : RadialGrid
        radial grid instance with coordinates and weights

    Returns
    -------
    scipy.CubicSpline
        CubicSpline object for interpolating values
    """
    ml_sph_value = compute_spline_point_value(sph_harm, value_arrays, weights, indices)
    # sin \theta d \theta d \phi = d{S_r} / (r^2)
    ml_sph_value_with_r = (
        ml_sph_value / (radial.points ** 2 * radial.weights)[:, None, None]
    )
    return CubicSpline(x=radial.points, y=ml_sph_value_with_r)


def spline_with_atomic_grid(at_grid, value_array):
    """Compute spline with real spherical hormonics for given atomic grid.

    Parameters
    ----------
    at_grid : AtomicGrid
        atomic grid for interpolation
    value_array : np.narray(N,)
        function value of each point on the atomic grid

    Returns
    -------
    scipy.CubicSpline
        CubicSpline object for interpolating values
    """
    l_max = at_grid.l_max // 2
    sph_coor = at_grid.convert_cart_to_sph()
    r_sph = generate_real_sph_harms(l_max, sph_coor[:, 1], sph_coor[:, 2])
    return spline_with_sph_harms(
        r_sph, value_array, at_grid.weights, at_grid.indices, at_grid.rad_grid
    )


def compute_spline_point_value(sph_harm, value_arrays, weights, indices):
    """Compute each note value for interpotation.

    Parameters
    ----------
    sph_harm : np.ndarray(M, L, N)
        spherical harmonics values of m, l, n_points
    value_arrays : np.ndarray(N,)
        fuction values on each point
    weights : np.ndarray(N,)
        weights of each point on the grid
    indices : list[int]
        indices of each chank for each radial angular partsption

    Returns
    -------
    np.ndarray(K, M, L)
        values on each note of K radial points
    """
    prod_value = sph_harm * value_arrays * weights
    total = len(indices) - 1
    axis_value = []
    for i in range(total):
        sum_i = np.sum(prod_value[:, :, indices[i] : indices[i + 1]], axis=-1)
        axis_value.append(sum_i)
    ml_sph_value = np.nan_to_num(np.stack(axis_value))
    return ml_sph_value


def interpolate(spline, r_points, theta, phi, deriv=0):
    """Interpolate angular points on given r value.

    Parameters
    ----------
    spline : scipy.CubicSpline
        CubicSpline object for interpolating function value
    r_points : float or list[float]
        Radial value to interpolate function values
    theta : np.ndarray(N,)
        Azimuthal angle of angular grid
    phi : np.ndarray(N,)
        Polar angle of angular grid
    deriv : int, default to 0
        0 for interpolation, 1, 2, 3 for 1st, 2nd and 3rd derivatives

    Returns
    -------
    np.ndarry(N,) or np.ndarray(n, N)
        Interpolated function value at spherical grid
    """
    if deriv == 0:
        r_value = spline(r_points)
    elif 1 <= deriv <= 3:
        r_value = spline(r_points, deriv)
    else:
        raise ValueError(f"deriv should be between [0, 3], got {deriv}")
    l_max = r_value.shape[-1] - 1
    r_sph_harm = generate_real_sph_harms(l_max, theta, phi)
    # single value interpolation
    if np.isscalar(r_points):
        return np.sum(r_sph_harm * r_value[..., None], axis=(0, 1))
    # intepolate for multiple values
    else:
        # convert to np.array if list
        r_points = np.array(r_points)
        return np.sum(r_sph_harm * r_value[..., None], axis=(-3, -2))
