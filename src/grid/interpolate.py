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
"""Interpolating real-valued functions wrt radial coordinate in spherical coordinates."""

import warnings

from grid.atomgrid import AtomGrid
from grid.basegrid import OneDGrid

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.special import sph_harm


def spline_with_sph_harms(
    sph_harm: np.ndarray,
    value_arrays: np.ndarray,
    weights: np.ndarray,
    indices: list,
    radial: OneDGrid,
):
    r"""
    Return spline to interpolate radial components wrt to expansion in real spherical harmonics.

    For fixed r, a function :math:`f(r, \theta, \phi)` is projected onto the spherical
    harmonic expansion

    .. math::
        f(r, \theta, \phi) = \sum_{l=0}^\infty \sum_{m=-l}^l \rho^{lm}(r) Y^m_l(\theta, \phi)

    The coefficients :math:`\rho^{lm}(r)` are interpolated using a cubic spline for each
    consequent :math:`r` values, where one can evaluate :math:`f` on any set of points.

    Parameters
    ----------
    sph_harm : ndarray(M, L, N)
        Spherical harmonic of order :math:`m`, degree :math:`l` evaluated on :math:`N` points,
        where order :math:`m` is ordered as follows
        :math:`[0, 1, \cdots, L, -L, \cdots, -1]`.
    value_arrays : ndarray(N,)
        Function values on each point
    weights : ndarray(N,)
        Weights of each point on the grid
    indices : list[int]
        Specify the indices of size :math:`K + 1` to calculate each chunk of the
        :math:`N` points. For example, indices=[0, 5, 10], will sum from index 0 to 4,
        and from index 5 to 9, separately. Each pair of indices should
        correspond to one radial point with different spherical angles,
        i.e. index 0 to 4 is summation for a fixed :math:`r` point with 4 spherical
        angles :math:`(\theta, \phi)`, etc.
    radial : OneDGrid
        Radial (one-dimensional) grid with :math:`K` points that includes points and weights.

    Returns
    -------
    scipy.CubicSpline
        CubicSpline object for interpolating the coefficients :math:`\rho^{lm}(r)`
        on the radial coordinate :math:`r` based on the spherical harmonic expansion
        (for a fixed :math:`r`). The input of spline is array of :math:`N` points on
        :math:`[0, \infty)` and the output is (N, M, L) matrix with (i, m, l) matrix
        entries :math:`\rho^{lm}(r_i)`.

    """
    # Has shape (K, M, L), K is number of radial points, M is order of spherical harmonics
    #       L is the angular degree.
    ml_sph_value = project_function_onto_spherical_expansion(
        sph_harm, value_arrays, weights, indices
    )
    # sin \theta d \theta d \phi = d{S_r} / (r^2)
    ml_sph_value_with_r = (
        ml_sph_value / (radial.points**2 * radial.weights)[:, None, None]
    )
    return CubicSpline(x=radial.points, y=ml_sph_value_with_r)


def spline_with_atomic_grid(atgrid: AtomGrid, func_vals: np.ndarray):
    r"""
    Return spline to interpolate radial components wrt to expansion in real spherical harmonics.

    For fixed r, a function :math:`f(r, \theta, \phi)` is projected onto the spherical
    harmonic expansion

    .. math::
        f(r, \theta, \phi) = \sum_{l=0}^\infty \sum_{m=-l}^l \rho^{lm}(r) Y^m_l(\theta, \phi)

    The coefficients :math:`\rho^{lm}(r)` are interpolated on the radial component using a cubic
    spline for each consequent :math:`r` values, where one can evaluate :math:`f` on any set
    of points.

    Parameters
    ----------
    atgrid : AtomGrid
        Atomic grid which contains a radial grid for integrating on radial axis and
        angular grid for integrating on spherical angles (the unit sphere).
    func_vals : ndarray(N,)
        Function values on each point on the atomic grid.

    Returns
    -------
    scipy.CubicSpline
        CubicSpline object for interpolating the coefficients :math:`\rho^{lm}(r)`
        on the radial coordinate :math:`r` based on the spherical harmonic expansion
        (for a fixed :math:`r`). The input of spline is array of :math:`N` points in
        :math:`[0, \infty)` and the output is (N, M, L) matrix with (i, m, l) matrix
        entries :math:`\rho^{lm}(r_i)`.

    """
    l_max = atgrid.l_max // 2
    # Convert grid points from Cartesian to Spherical coordinates
    sph_coor = atgrid.convert_cart_to_sph()
    # Construct all real spherical harmonics up to degree l_max on the grid points
    r_sph = generate_real_sph_harms(l_max, sph_coor[:, 1], sph_coor[:, 2])
    return spline_with_sph_harms(
        r_sph, func_vals, atgrid.weights, atgrid.indices, atgrid.rgrid
    )


def project_function_onto_spherical_expansion(
    sph_harm: np.ndarray, value_arrays: np.ndarray, weights: np.ndarray, indices: list
):
    r"""Project function for a fixed r to spherical harmonic expansion.

    Given a function :math:`f(r, \theta, \phi)` acting on the spherical
    coordinates, for a fixed :math:`r`, the projection onto the
    spherical harmonic expansion is given by

    .. math::
        f(r, \theta, \phi) &= \sum_{l=0}^\infty \sum_{m=-l}^l \rho^{lm}(r) Y^m_l(\theta, \phi),

    where :math:`\rho^{lm}(r)` are the coefficients corresponding to the linear expansion which
    are given by

    .. math::
        \rho^{lm}(r) = \int_0^{2\pi} \int_0^\pi f(r, \theta, \phi) Y^m_l (\theta, \phi)d\theta
        d\phi.

    The integral of the coefficients is done numerically, specified by the weights and
    indices.

    Parameters
    ----------
    sph_harm : ndarray(M, L, N)
        Spherical harmonic of order :math:`m`, degree :math:`l` evaluated on
        :math:`N` points, where order :math:`m` is ordered as follows
        :math:`[0, 1, \cdots, L, -L, \cdots, -1]`.
    value_arrays : ndarray(N,)
        Function values on each :math:`N` points.
    weights : ndarray(N,)
        Weights of each point on the grid.
    indices : list[int]
        Specify the indices of size :math:`K + 1` to calculate each chunk of the
        :math:`N` points. For example, indices=[0, 5, 10], will sum from index 0 to 4,
        and from index 5 to 9, separately. Each pair of indices should
        correspond to one radial point with different spherical angles,
        i.e. index 0 to 4 is summation for a fixed :math:`r` point with 4 spherical
        angles :math:`(\theta, \phi)`, etc.

    Returns
    -------
    ndarray(K, M, L)
        Coefficients :math:`\rho^{lm}(r)`of the expansion on each of the :math:`K` radial points
        :math:`r` up to :math:`L` angular degrees and for all :math:`M` orders.

    """
    prod_value = sph_harm * value_arrays * weights
    axis_value = []
    for i in range(len(indices) - 1):
        sum_i = np.sum(prod_value[:, :, indices[i] : indices[i + 1]], axis=-1)
        axis_value.append(sum_i)
    # Stack K list `axis_value` of arrays of size (M, L) to (K, M, L) array.
    ml_sph_value = np.nan_to_num(np.stack(axis_value))
    return ml_sph_value


def interpolate_given_splines(
    spline: CubicSpline,
    r_points: list,
    theta: np.ndarray,
    phi: np.ndarray,
    deriv: bool = 0,
):
    r"""
    Interpolate function and its derivatives in spherical coordinates given radial splines.

    Any real-valued function :math:`f(r, \theta, \phi)` can be decomposed as

    .. math::
        f(r, \theta, \phi) = \sum_l \sum_{m=-l}^l \sum_i \rho_{ilm}(r) Y_{lm}(\theta, \phi)

    The spline interpolate the radial functions :math:`\sum_i \rho_{ilm}(r)` for a given
    series of :math:`r_i \in [0, \infty)` values.  This is then multipled by the
    corresponding spherical harmonics at all :math:`(\theta_j, \phi_j)` angles
    and summed to obtain the approximation to :math:`f(r_i, \theta_j, \phi_j)` for
    all combinations of :math:`r_i` and :math:`(\theta_j, \phi_j)`,
    where :math:`i=1, \cdots, N` and :math:`j=1\cdots, K`.

    Parameters
    ----------
    spline : scipy.CubicSpline
        CubicSpline object for interpolating the radial component :math:`\rho_{ilm}`.
        The input of spline is the radial points r and the output is (N, M, L) array
        with (i, m, l) matrix entry :math:`\rho^{lm}(r_i)` used to expand in the spherical
        harmonic basis.
    r_points : float or ndarray(K,)
        Radial value to interpolate the function at. If scalar, then only interpolated
        at :math:`(r, \theta_j, \phi_j)` for all j.
    theta : ndarray(N,)
        Azimuthal angle to interpolate the function at.
    phi : ndarray(N,)
        Polar angle of angular grid to interpolate the function at.
    deriv : int, optional
        Specify whether interpolation (d=0), or :math:`d`th derivative to
        compute.  Only up to third derivative is allowed.

    Returns
    -------
    ndarray(N, K)
        Matrix with (i, j) entries of the interpolated function :math:`f(r_i, \theta_j, \phi_j`
        or its derivative  :math:`\frac{f(r_i, \theta_j, \phi_j)}{r_i}` wrt to radial component
        :math:`r` at :math:`KN` points :math:`(r_i, \theta_j, \phi_j)`, where
        :math:`i=0,\cdots, N-1`, and :math:`j=0, \cdots K-1`.

    """
    if len(theta) != len(phi):
        raise ValueError(f"Shape of theta {len(theta)} should equal shape of phi {len(phi)}.")

    if deriv == 0:
        r_value = spline(r_points)
    elif 1 <= deriv <= 3:
        r_value = spline(r_points, deriv)
    else:
        raise ValueError(f"deriv should be between [0, 3], got {deriv}.")
    l_max = r_value.shape[-1] - 1
    # Return (M, L, N)
    r_sph_harm = generate_real_sph_harms(l_max, theta, phi)
    # single value interpolation, m=number of orders, l=maximum degree,
    #    n= number of radial points to interpolate,
    #    k= number of angular points to interpolate to,
    # Interpolate for (r, theta_j, \phi_j)
    if np.isscalar(r_points):
        return np.einsum("mlk,ml->k", r_sph_harm, r_value)
    # Interpolate for multiple values (r_i, theta_j, \phi_j)
    else:
        return np.einsum("mlk,nml->nk", r_sph_harm, r_value)
