# -*- coding: utf-8 -*-
# GRID is a numerical integration library for quantum chemistry.
#
# Copyright (C) 2011-2017 The GRID Development Team
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
#
# --
"""Auxiliary routines related to multipole moments.

This module fixes all the conventions with respect to multipole moments. Some
of the code below may (in some way) reoccur in the low-level routines. In any
case, such low-level code should be consistent with the conventions in this
module. See for example, grid.gobasis.cext.cart_to_pur_low.

"""

import pkg_resources
import numpy as np


__all__ = ['get_cartesian_powers', 'get_ncart', 'get_ncart_cumul',
           'rotate_cartesian_multipole', 'rotate_cartesian_moments_all',
           'get_npure', 'get_npure_cumul', 'fill_cartesian_polynomials',
           'fill_pure_polynomials', 'fill_radial_polynomials']


def get_cartesian_powers(lmax):
    """Return an ordered list of power for x, y and z up to angular moment lmax

       **Arguments:**

       lmax
            The maximum angular momentum (0=s, 1=p, 2=d, ...)

       **Returns:** an array where each row corresponds to a multipole moment
       and each column corresponds to a power of x, y and z respectively. The
       rows are grouped per angular momentum, first s, them p, then d, and so
       on. Within one angular momentum the rows are sorted 'alphabetically',
       e.g. for l=2: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz.
    """
    cartesian_powers = np.zeros((get_ncart_cumul(lmax), 3), dtype=int)
    counter = 0
    for l in range(0, lmax + 1):
        for nx in range(l + 1, -1, -1):
            for ny in range(l - nx, -1, -1):
                nz = l - ny - nx
                cartesian_powers[counter] = [nx, ny, nz]
                counter += 1
    return cartesian_powers


def get_ncart(l):
    """The number of cartesian powers for a given angular momentum, l"""
    return int(((l + 2) * (l + 1)) / 2)


def get_ncart_cumul(lmax):
    """The number of cartesian powers up to a given angular momentum, lmax."""
    return int(((lmax + 1) * (lmax + 2) * (lmax + 3)) / 6)


def rotate_cartesian_multipole(rmat, moments, mode):
    """Compute rotated Cartesian multipole moment/expansion.

       **Arguments:**

       rmat
            A (3,3) rotation matrix.

       moments
            A multipole moment/coeffs. The angular momentum is derived from the
            length of this vector.

       mode
            A string containing either 'moments' or 'coeffs'. In case if
            'moments', a Cartesian multipole moment rotation is carried out. In
            case of 'coeffs', the coefficients of a Cartesian multipole basis
            are rotated.

       **Returns:** rotated multipole.
    """
    l = ((9 + 8 * (len(moments) - 1)) ** 0.5 - 3) / 2
    if l - np.round(l) > 1e-10:
        raise ValueError('Could not determine l from number of moments.')
    l = int(np.round(l))

    if mode == 'coeffs':
        rcoeffs = rmat.T.ravel()
    elif mode == 'moments':
        rcoeffs = rmat.ravel()
    else:
        raise NotImplementedError
    result = np.zeros(len(moments))
    npy_file = pkg_resources.resource_filename("grid.data", "cart_tf.npy")
    cartesian_transforms = np.load(npy_file)
    for i0 in range(len(moments)):
        rules = cartesian_transforms[l][i0]
        for rule in rules:
            i1 = rule[0]
            factor = rule[1]
            for j in rule[2:]:
                factor *= rcoeffs[j]
            if mode == 'coeffs':
                result[i1] += moments[i0] * factor
            elif mode == 'moments':
                result[i0] += moments[i1] * factor
            else:
                raise NotImplementedError
    return result


def rotate_cartesian_moments_all(rmat, moments):
    """Rotate cartesian moments

       **Arguments:**

       rmat
            A (3,3) rotation matrix.

       moments
            A row vector with a series of cartesian multipole moments, starting
            from l=0 up to l=lmax. Items in this vector should follow the same
            order as defined by the function ``get_cartesian_powers``.

       **Returns:** A similar vector with rotated multipole moments
    """
    ncart = moments.shape[0]
    result = np.zeros(ncart)

    icart = 0
    nshell = 1
    ishell = 0
    while icart < ncart:
        result[icart:icart + nshell] = rotate_cartesian_multipole(rmat,
                                                                  moments[icart:icart + nshell],
                                                                  'moments')
        icart += nshell
        ishell += 1
        nshell += ishell + 1
    return result


def get_npure(l):
    """The number of pure functions for a given angular momentum, l"""
    return 2 * l + 1


def get_npure_cumul(lmax):
    """The number of pure functions up to a given angular momentum, lmax."""
    return (lmax + 1) ** 2


def fill_cartesian_polynomials(output: np.ndarray, lmax: int):
    # py parts
    if output.shape[0] < ((lmax + 1) * (lmax + 2) * (lmax + 3)) / 6 - 1:
        raise ValueError(
            "The size of the output array is not sufficient to store the polynomials."
        )
    return _fill_cartesian_polynomials(output, lmax)


def fill_pure_polynomials(output: np.ndarray, lmax: int):
    if output.ndim == 1:
        if output.shape[0] < (lmax + 1) ** 2 - 1:
            raise ValueError(
                "The size of the output array is not sufficient to store the polynomials."
            )
        return _fill_pure_polynomials(output, lmax)
    if output.ndim == 2:
        if output.shape[1] < (lmax + 1) ** 2 - 1:
            raise ValueError(
                "The size of the output array is not sufficient to store the polynomials."
            )
        return _fill_pure_polynomials_array(output, lmax, output.shape[0])
    raise NotImplementedError


def fill_radial_polynomials(output: np.ndarray, lmax: int):
    if output.shape[0] < lmax:
        raise ValueError(
            "The size of the output array is not sufficient to store the polynomials."
        )
    _fill_radial_polynomials(output, lmax)


def _fill_cartesian_polynomials(output: np.ndarray, lmax: int):
    # translated part
    # shell l=0
    if lmax <= 0:
        return -1
    # shell l=1
    if lmax <= 1:
        return 0
    # shell l> 1
    old_offset = 0
    old_ncart = 3
    for l in range(2, lmax + 1):
        new_ncart = old_ncart + l + 1
        new_offset = old_offset + old_ncart

        # for i in range(old_ncart):
        #     output[new_offset + i] = output[0] * output[old_offset + i]
        # vecterized function for commented loop
        output[new_offset : new_offset + old_ncart] = (
            output[0] * output[old_offset : old_offset + old_ncart]
        )

        # for i in range(l):
        #     output[new_offset + old_ncart + i] = output[1] * output[new_offset - l + i]
        # vecterized function for commented loop
        output[new_offset + old_ncart : new_offset + old_ncart + l] = (
            output[1] * output[new_offset - l : new_offset]
        )
        output[new_offset + new_ncart - 1] = output[2] * output[new_offset - 1]
        old_ncart = new_ncart
        old_offset = new_offset
    return old_offset


def _fill_pure_polynomials(output: np.ndarray, lmax: int):
    if lmax <= 0:
        return -1
    if lmax <= 1:
        return 0
    z = output[0]
    x = output[1]
    y = output[2]
    r2 = x ** 2 + y ** 2 + z ** 2
    pi_old = np.zeros(lmax + 1)
    pi_new = np.zeros(lmax + 1)
    a = np.zeros(lmax + 1)
    b = np.zeros(lmax + 1)

    pi_old[0] = 1
    pi_new[0] = z
    pi_new[1] = 1
    a[1] = x
    b[1] = y

    old_offset = 0
    old_npure = 3

    for l in range(2, lmax + 1):
        new_npure = old_npure + 2
        new_offset = old_offset + old_npure

        factor = 2.0 * l - 1

        # for m in range(l - 1):
        #     tmp = pi_old[m]
        #     pi_old[m] = pi_new[m]
        #     pi_new[m] = (z * factor * pi_old[m] - r2 * (l + m - 1) * tmp) / (l - m)
        # translate loop with vectorization
        tmp = pi_old[: l - 1].copy()
        pi_old[: l - 1] = pi_new[: l - 1]
        pi_new[: l - 1] = (
            z * factor * pi_old[: l - 1]
            - r2 * np.arange(l - 1, 2 * l - 2) * tmp[: l - 1]
        ) / np.arange(l, 1, -1)

        pi_old[l - 1] = pi_new[l - 1]
        pi_new[l] = factor * pi_old[l - 1]
        pi_new[l - 1] = z * pi_new[l]

        a[l] = x * a[l - 1] - y * b[l - 1]
        b[l] = x * b[l - 1] + y * a[l - 1]

        output[new_offset] = pi_new[0]
        factor = np.sqrt(2)

        # factor is dependent to each other, hard to vectorize
        for m in range(1, l + 1):
            factor /= np.sqrt((l + m) * (l - m + 1))
            output[new_offset + 2 * m - 1] = factor * a[m] * pi_new[m]
            output[new_offset + 2 * m] = factor * b[m] * pi_new[m]

        old_npure = new_npure
        old_offset = new_offset
    return old_offset


def _fill_pure_polynomials_array(output: np.ndarray, lmax: int, nrep: int):
    for irep in range(0, nrep):
        result = _fill_pure_polynomials(output[irep], lmax)
    return result


def _fill_radial_polynomials(output: np.ndarray, lmax: int):
    if lmax <= 1:
        return
    output[1:] = output[0] * output[:-1]
    for index in range(lmax - 1):
        output[index + 1] = output[0] * output[index]
