# -*- coding: utf-8 -*-
# OLDGRIDS: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The OLDGRIDS Development Team
#
# This file is part of OLDGRIDS.
#
# OLDGRIDS is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# OLDGRIDS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Auxiliary routines related to multipole moments

   This module fixes all the conventions with respect to multipole moments. Some
   of the code below may (in some way) reoccur in the low-level routines. In any
   case, such low-level code should be consistent with the conventions in this
   module. See for example, grid.gobasis.cext.cart_to_pur_low.
"""

import numpy as np

__all__ = ['get_cartesian_powers', 'get_ncart', 'get_ncart_cumul',
           'rotate_cartesian_multipole', 'rotate_cartesian_moments_all',
           'get_npure', 'get_npure_cumul']


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


cartesian_transforms = [
    [
        [[0, 1]],
    ],
    [
        [[0, 1, 0],
         [1, 1, 1],
         [2, 1, 2]],
        [[0, 1, 3],
         [1, 1, 4],
         [2, 1, 5]],
        [[0, 1, 6],
         [1, 1, 7],
         [2, 1, 8]],
    ],
    [
        [[0, 1, 0, 0],
         [1, 2, 0, 1],
         [2, 2, 0, 2],
         [3, 1, 1, 1],
         [4, 2, 1, 2],
         [5, 1, 2, 2]],
        [[0, 1, 0, 3],
         [1, 1, 1, 3],
         [1, 1, 0, 4],
         [2, 1, 2, 3],
         [2, 1, 0, 5],
         [3, 1, 1, 4],
         [4, 1, 2, 4],
         [4, 1, 1, 5],
         [5, 1, 2, 5]],
        [[0, 1, 0, 6],
         [1, 1, 1, 6],
         [1, 1, 0, 7],
         [2, 1, 2, 6],
         [2, 1, 0, 8],
         [3, 1, 1, 7],
         [4, 1, 2, 7],
         [4, 1, 1, 8],
         [5, 1, 2, 8]],
        [[0, 1, 3, 3],
         [1, 2, 3, 4],
         [2, 2, 3, 5],
         [3, 1, 4, 4],
         [4, 2, 4, 5],
         [5, 1, 5, 5]],
        [[0, 1, 3, 6],
         [1, 1, 4, 6],
         [1, 1, 3, 7],
         [2, 1, 5, 6],
         [2, 1, 3, 8],
         [3, 1, 4, 7],
         [4, 1, 5, 7],
         [4, 1, 4, 8],
         [5, 1, 5, 8]],
        [[0, 1, 6, 6],
         [1, 2, 6, 7],
         [2, 2, 6, 8],
         [3, 1, 7, 7],
         [4, 2, 7, 8],
         [5, 1, 8, 8]],
    ],
    [
        [[0, 1, 0, 0, 0],
         [1, 3, 0, 0, 1],
         [2, 3, 0, 0, 2],
         [3, 3, 0, 1, 1],
         [4, 6, 0, 1, 2],
         [5, 3, 0, 2, 2],
         [6, 1, 1, 1, 1],
         [7, 3, 1, 1, 2],
         [8, 3, 1, 2, 2],
         [9, 1, 2, 2, 2]],
        [[0, 1, 0, 0, 3],
         [1, 1, 0, 0, 4],
         [1, 2, 0, 1, 3],
         [2, 1, 0, 0, 5],
         [2, 2, 0, 2, 3],
         [3, 1, 1, 1, 3],
         [3, 2, 0, 1, 4],
         [4, 2, 1, 2, 3],
         [4, 2, 0, 2, 4],
         [4, 2, 0, 1, 5],
         [5, 1, 2, 2, 3],
         [5, 2, 0, 2, 5],
         [6, 1, 1, 1, 4],
         [7, 1, 1, 1, 5],
         [7, 2, 1, 2, 4],
         [8, 1, 2, 2, 4],
         [8, 2, 1, 2, 5],
         [9, 1, 2, 2, 5]],
        [[0, 1, 0, 0, 6],
         [1, 1, 0, 0, 7],
         [1, 2, 0, 1, 6],
         [2, 1, 0, 0, 8],
         [2, 2, 0, 2, 6],
         [3, 1, 1, 1, 6],
         [3, 2, 0, 1, 7],
         [4, 2, 1, 2, 6],
         [4, 2, 0, 2, 7],
         [4, 2, 0, 1, 8],
         [5, 1, 2, 2, 6],
         [5, 2, 0, 2, 8],
         [6, 1, 1, 1, 7],
         [7, 1, 1, 1, 8],
         [7, 2, 1, 2, 7],
         [8, 1, 2, 2, 7],
         [8, 2, 1, 2, 8],
         [9, 1, 2, 2, 8]],
        [[0, 1, 0, 3, 3],
         [1, 1, 1, 3, 3],
         [1, 2, 0, 3, 4],
         [2, 1, 2, 3, 3],
         [2, 2, 0, 3, 5],
         [3, 1, 0, 4, 4],
         [3, 2, 1, 3, 4],
         [4, 2, 2, 3, 4],
         [4, 2, 1, 3, 5],
         [4, 2, 0, 4, 5],
         [5, 1, 0, 5, 5],
         [5, 2, 2, 3, 5],
         [6, 1, 1, 4, 4],
         [7, 1, 2, 4, 4],
         [7, 2, 1, 4, 5],
         [8, 1, 1, 5, 5],
         [8, 2, 2, 4, 5],
         [9, 1, 2, 5, 5]],
        [[0, 1, 0, 3, 6],
         [1, 1, 1, 3, 6],
         [1, 1, 0, 4, 6],
         [1, 1, 0, 3, 7],
         [2, 1, 2, 3, 6],
         [2, 1, 0, 5, 6],
         [2, 1, 0, 3, 8],
         [3, 1, 1, 4, 6],
         [3, 1, 1, 3, 7],
         [3, 1, 0, 4, 7],
         [4, 1, 2, 4, 6],
         [4, 1, 2, 3, 7],
         [4, 1, 1, 5, 6],
         [4, 1, 1, 3, 8],
         [4, 1, 0, 5, 7],
         [4, 1, 0, 4, 8],
         [5, 1, 2, 5, 6],
         [5, 1, 2, 3, 8],
         [5, 1, 0, 5, 8],
         [6, 1, 1, 4, 7],
         [7, 1, 2, 4, 7],
         [7, 1, 1, 5, 7],
         [7, 1, 1, 4, 8],
         [8, 1, 2, 5, 7],
         [8, 1, 2, 4, 8],
         [8, 1, 1, 5, 8],
         [9, 1, 2, 5, 8]],
        [[0, 1, 0, 6, 6],
         [1, 1, 1, 6, 6],
         [1, 2, 0, 6, 7],
         [2, 1, 2, 6, 6],
         [2, 2, 0, 6, 8],
         [3, 1, 0, 7, 7],
         [3, 2, 1, 6, 7],
         [4, 2, 2, 6, 7],
         [4, 2, 1, 6, 8],
         [4, 2, 0, 7, 8],
         [5, 1, 0, 8, 8],
         [5, 2, 2, 6, 8],
         [6, 1, 1, 7, 7],
         [7, 1, 2, 7, 7],
         [7, 2, 1, 7, 8],
         [8, 1, 1, 8, 8],
         [8, 2, 2, 7, 8],
         [9, 1, 2, 8, 8]],
        [[0, 1, 3, 3, 3],
         [1, 3, 3, 3, 4],
         [2, 3, 3, 3, 5],
         [3, 3, 3, 4, 4],
         [4, 6, 3, 4, 5],
         [5, 3, 3, 5, 5],
         [6, 1, 4, 4, 4],
         [7, 3, 4, 4, 5],
         [8, 3, 4, 5, 5],
         [9, 1, 5, 5, 5]],
        [[0, 1, 3, 3, 6],
         [1, 1, 3, 3, 7],
         [1, 2, 3, 4, 6],
         [2, 1, 3, 3, 8],
         [2, 2, 3, 5, 6],
         [3, 1, 4, 4, 6],
         [3, 2, 3, 4, 7],
         [4, 2, 4, 5, 6],
         [4, 2, 3, 5, 7],
         [4, 2, 3, 4, 8],
         [5, 1, 5, 5, 6],
         [5, 2, 3, 5, 8],
         [6, 1, 4, 4, 7],
         [7, 1, 4, 4, 8],
         [7, 2, 4, 5, 7],
         [8, 1, 5, 5, 7],
         [8, 2, 4, 5, 8],
         [9, 1, 5, 5, 8]],
        [[0, 1, 3, 6, 6],
         [1, 1, 4, 6, 6],
         [1, 2, 3, 6, 7],
         [2, 1, 5, 6, 6],
         [2, 2, 3, 6, 8],
         [3, 1, 3, 7, 7],
         [3, 2, 4, 6, 7],
         [4, 2, 5, 6, 7],
         [4, 2, 4, 6, 8],
         [4, 2, 3, 7, 8],
         [5, 1, 3, 8, 8],
         [5, 2, 5, 6, 8],
         [6, 1, 4, 7, 7],
         [7, 1, 5, 7, 7],
         [7, 2, 4, 7, 8],
         [8, 1, 4, 8, 8],
         [8, 2, 5, 7, 8],
         [9, 1, 5, 8, 8]],
        [[0, 1, 6, 6, 6],
         [1, 3, 6, 6, 7],
         [2, 3, 6, 6, 8],
         [3, 3, 6, 7, 7],
         [4, 6, 6, 7, 8],
         [5, 3, 6, 8, 8],
         [6, 1, 7, 7, 7],
         [7, 3, 7, 7, 8],
         [8, 3, 7, 8, 8],
         [9, 1, 8, 8, 8]],
    ],
    [
        [[0, 1, 0, 0, 0, 0],
         [1, 4, 0, 0, 0, 1],
         [2, 4, 0, 0, 0, 2],
         [3, 6, 0, 0, 1, 1],
         [4, 12, 0, 0, 1, 2],
         [5, 6, 0, 0, 2, 2],
         [6, 4, 0, 1, 1, 1],
         [7, 12, 0, 1, 1, 2],
         [8, 12, 0, 1, 2, 2],
         [9, 4, 0, 2, 2, 2],
         [10, 1, 1, 1, 1, 1],
         [11, 4, 1, 1, 1, 2],
         [12, 6, 1, 1, 2, 2],
         [13, 4, 1, 2, 2, 2],
         [14, 1, 2, 2, 2, 2]],
        [[0, 1, 0, 0, 0, 3],
         [1, 1, 0, 0, 0, 4],
         [1, 3, 0, 0, 1, 3],
         [2, 1, 0, 0, 0, 5],
         [2, 3, 0, 0, 2, 3],
         [3, 3, 0, 1, 1, 3],
         [3, 3, 0, 0, 1, 4],
         [4, 3, 0, 0, 2, 4],
         [4, 3, 0, 0, 1, 5],
         [4, 6, 0, 1, 2, 3],
         [5, 3, 0, 2, 2, 3],
         [5, 3, 0, 0, 2, 5],
         [6, 1, 1, 1, 1, 3],
         [6, 3, 0, 1, 1, 4],
         [7, 3, 1, 1, 2, 3],
         [7, 3, 0, 1, 1, 5],
         [7, 6, 0, 1, 2, 4],
         [8, 3, 1, 2, 2, 3],
         [8, 3, 0, 2, 2, 4],
         [8, 6, 0, 1, 2, 5],
         [9, 1, 2, 2, 2, 3],
         [9, 3, 0, 2, 2, 5],
         [10, 1, 1, 1, 1, 4],
         [11, 1, 1, 1, 1, 5],
         [11, 3, 1, 1, 2, 4],
         [12, 3, 1, 2, 2, 4],
         [12, 3, 1, 1, 2, 5],
         [13, 1, 2, 2, 2, 4],
         [13, 3, 1, 2, 2, 5],
         [14, 1, 2, 2, 2, 5]],
        [[0, 1, 0, 0, 0, 6],
         [1, 1, 0, 0, 0, 7],
         [1, 3, 0, 0, 1, 6],
         [2, 1, 0, 0, 0, 8],
         [2, 3, 0, 0, 2, 6],
         [3, 3, 0, 1, 1, 6],
         [3, 3, 0, 0, 1, 7],
         [4, 3, 0, 0, 2, 7],
         [4, 3, 0, 0, 1, 8],
         [4, 6, 0, 1, 2, 6],
         [5, 3, 0, 2, 2, 6],
         [5, 3, 0, 0, 2, 8],
         [6, 1, 1, 1, 1, 6],
         [6, 3, 0, 1, 1, 7],
         [7, 3, 1, 1, 2, 6],
         [7, 3, 0, 1, 1, 8],
         [7, 6, 0, 1, 2, 7],
         [8, 3, 1, 2, 2, 6],
         [8, 3, 0, 2, 2, 7],
         [8, 6, 0, 1, 2, 8],
         [9, 1, 2, 2, 2, 6],
         [9, 3, 0, 2, 2, 8],
         [10, 1, 1, 1, 1, 7],
         [11, 1, 1, 1, 1, 8],
         [11, 3, 1, 1, 2, 7],
         [12, 3, 1, 2, 2, 7],
         [12, 3, 1, 1, 2, 8],
         [13, 1, 2, 2, 2, 7],
         [13, 3, 1, 2, 2, 8],
         [14, 1, 2, 2, 2, 8]],
        [[0, 1, 0, 0, 3, 3],
         [1, 2, 0, 1, 3, 3],
         [1, 2, 0, 0, 3, 4],
         [2, 2, 0, 2, 3, 3],
         [2, 2, 0, 0, 3, 5],
         [3, 1, 1, 1, 3, 3],
         [3, 1, 0, 0, 4, 4],
         [3, 4, 0, 1, 3, 4],
         [4, 2, 1, 2, 3, 3],
         [4, 2, 0, 0, 4, 5],
         [4, 4, 0, 2, 3, 4],
         [4, 4, 0, 1, 3, 5],
         [5, 1, 2, 2, 3, 3],
         [5, 1, 0, 0, 5, 5],
         [5, 4, 0, 2, 3, 5],
         [6, 2, 1, 1, 3, 4],
         [6, 2, 0, 1, 4, 4],
         [7, 2, 1, 1, 3, 5],
         [7, 2, 0, 2, 4, 4],
         [7, 4, 1, 2, 3, 4],
         [7, 4, 0, 1, 4, 5],
         [8, 2, 2, 2, 3, 4],
         [8, 2, 0, 1, 5, 5],
         [8, 4, 1, 2, 3, 5],
         [8, 4, 0, 2, 4, 5],
         [9, 2, 2, 2, 3, 5],
         [9, 2, 0, 2, 5, 5],
         [10, 1, 1, 1, 4, 4],
         [11, 2, 1, 2, 4, 4],
         [11, 2, 1, 1, 4, 5],
         [12, 1, 2, 2, 4, 4],
         [12, 1, 1, 1, 5, 5],
         [12, 4, 1, 2, 4, 5],
         [13, 2, 2, 2, 4, 5],
         [13, 2, 1, 2, 5, 5],
         [14, 1, 2, 2, 5, 5]],
        [[0, 1, 0, 0, 3, 6],
         [1, 1, 0, 0, 4, 6],
         [1, 1, 0, 0, 3, 7],
         [1, 2, 0, 1, 3, 6],
         [2, 1, 0, 0, 5, 6],
         [2, 1, 0, 0, 3, 8],
         [2, 2, 0, 2, 3, 6],
         [3, 1, 1, 1, 3, 6],
         [3, 1, 0, 0, 4, 7],
         [3, 2, 0, 1, 4, 6],
         [3, 2, 0, 1, 3, 7],
         [4, 1, 0, 0, 5, 7],
         [4, 1, 0, 0, 4, 8],
         [4, 2, 1, 2, 3, 6],
         [4, 2, 0, 2, 4, 6],
         [4, 2, 0, 2, 3, 7],
         [4, 2, 0, 1, 5, 6],
         [4, 2, 0, 1, 3, 8],
         [5, 1, 2, 2, 3, 6],
         [5, 1, 0, 0, 5, 8],
         [5, 2, 0, 2, 5, 6],
         [5, 2, 0, 2, 3, 8],
         [6, 1, 1, 1, 4, 6],
         [6, 1, 1, 1, 3, 7],
         [6, 2, 0, 1, 4, 7],
         [7, 1, 1, 1, 5, 6],
         [7, 1, 1, 1, 3, 8],
         [7, 2, 1, 2, 4, 6],
         [7, 2, 1, 2, 3, 7],
         [7, 2, 0, 2, 4, 7],
         [7, 2, 0, 1, 5, 7],
         [7, 2, 0, 1, 4, 8],
         [8, 1, 2, 2, 4, 6],
         [8, 1, 2, 2, 3, 7],
         [8, 2, 1, 2, 5, 6],
         [8, 2, 1, 2, 3, 8],
         [8, 2, 0, 2, 5, 7],
         [8, 2, 0, 2, 4, 8],
         [8, 2, 0, 1, 5, 8],
         [9, 1, 2, 2, 5, 6],
         [9, 1, 2, 2, 3, 8],
         [9, 2, 0, 2, 5, 8],
         [10, 1, 1, 1, 4, 7],
         [11, 1, 1, 1, 5, 7],
         [11, 1, 1, 1, 4, 8],
         [11, 2, 1, 2, 4, 7],
         [12, 1, 2, 2, 4, 7],
         [12, 1, 1, 1, 5, 8],
         [12, 2, 1, 2, 5, 7],
         [12, 2, 1, 2, 4, 8],
         [13, 1, 2, 2, 5, 7],
         [13, 1, 2, 2, 4, 8],
         [13, 2, 1, 2, 5, 8],
         [14, 1, 2, 2, 5, 8]],
        [[0, 1, 0, 0, 6, 6],
         [1, 2, 0, 1, 6, 6],
         [1, 2, 0, 0, 6, 7],
         [2, 2, 0, 2, 6, 6],
         [2, 2, 0, 0, 6, 8],
         [3, 1, 1, 1, 6, 6],
         [3, 1, 0, 0, 7, 7],
         [3, 4, 0, 1, 6, 7],
         [4, 2, 1, 2, 6, 6],
         [4, 2, 0, 0, 7, 8],
         [4, 4, 0, 2, 6, 7],
         [4, 4, 0, 1, 6, 8],
         [5, 1, 2, 2, 6, 6],
         [5, 1, 0, 0, 8, 8],
         [5, 4, 0, 2, 6, 8],
         [6, 2, 1, 1, 6, 7],
         [6, 2, 0, 1, 7, 7],
         [7, 2, 1, 1, 6, 8],
         [7, 2, 0, 2, 7, 7],
         [7, 4, 1, 2, 6, 7],
         [7, 4, 0, 1, 7, 8],
         [8, 2, 2, 2, 6, 7],
         [8, 2, 0, 1, 8, 8],
         [8, 4, 1, 2, 6, 8],
         [8, 4, 0, 2, 7, 8],
         [9, 2, 2, 2, 6, 8],
         [9, 2, 0, 2, 8, 8],
         [10, 1, 1, 1, 7, 7],
         [11, 2, 1, 2, 7, 7],
         [11, 2, 1, 1, 7, 8],
         [12, 1, 2, 2, 7, 7],
         [12, 1, 1, 1, 8, 8],
         [12, 4, 1, 2, 7, 8],
         [13, 2, 2, 2, 7, 8],
         [13, 2, 1, 2, 8, 8],
         [14, 1, 2, 2, 8, 8]],
        [[0, 1, 0, 3, 3, 3],
         [1, 1, 1, 3, 3, 3],
         [1, 3, 0, 3, 3, 4],
         [2, 1, 2, 3, 3, 3],
         [2, 3, 0, 3, 3, 5],
         [3, 3, 1, 3, 3, 4],
         [3, 3, 0, 3, 4, 4],
         [4, 3, 2, 3, 3, 4],
         [4, 3, 1, 3, 3, 5],
         [4, 6, 0, 3, 4, 5],
         [5, 3, 2, 3, 3, 5],
         [5, 3, 0, 3, 5, 5],
         [6, 1, 0, 4, 4, 4],
         [6, 3, 1, 3, 4, 4],
         [7, 3, 2, 3, 4, 4],
         [7, 3, 0, 4, 4, 5],
         [7, 6, 1, 3, 4, 5],
         [8, 3, 1, 3, 5, 5],
         [8, 3, 0, 4, 5, 5],
         [8, 6, 2, 3, 4, 5],
         [9, 1, 0, 5, 5, 5],
         [9, 3, 2, 3, 5, 5],
         [10, 1, 1, 4, 4, 4],
         [11, 1, 2, 4, 4, 4],
         [11, 3, 1, 4, 4, 5],
         [12, 3, 2, 4, 4, 5],
         [12, 3, 1, 4, 5, 5],
         [13, 1, 1, 5, 5, 5],
         [13, 3, 2, 4, 5, 5],
         [14, 1, 2, 5, 5, 5]],
        [[0, 1, 0, 3, 3, 6],
         [1, 1, 1, 3, 3, 6],
         [1, 1, 0, 3, 3, 7],
         [1, 2, 0, 3, 4, 6],
         [2, 1, 2, 3, 3, 6],
         [2, 1, 0, 3, 3, 8],
         [2, 2, 0, 3, 5, 6],
         [3, 1, 1, 3, 3, 7],
         [3, 1, 0, 4, 4, 6],
         [3, 2, 1, 3, 4, 6],
         [3, 2, 0, 3, 4, 7],
         [4, 1, 2, 3, 3, 7],
         [4, 1, 1, 3, 3, 8],
         [4, 2, 2, 3, 4, 6],
         [4, 2, 1, 3, 5, 6],
         [4, 2, 0, 4, 5, 6],
         [4, 2, 0, 3, 5, 7],
         [4, 2, 0, 3, 4, 8],
         [5, 1, 2, 3, 3, 8],
         [5, 1, 0, 5, 5, 6],
         [5, 2, 2, 3, 5, 6],
         [5, 2, 0, 3, 5, 8],
         [6, 1, 1, 4, 4, 6],
         [6, 1, 0, 4, 4, 7],
         [6, 2, 1, 3, 4, 7],
         [7, 1, 2, 4, 4, 6],
         [7, 1, 0, 4, 4, 8],
         [7, 2, 2, 3, 4, 7],
         [7, 2, 1, 4, 5, 6],
         [7, 2, 1, 3, 5, 7],
         [7, 2, 1, 3, 4, 8],
         [7, 2, 0, 4, 5, 7],
         [8, 1, 1, 5, 5, 6],
         [8, 1, 0, 5, 5, 7],
         [8, 2, 2, 4, 5, 6],
         [8, 2, 2, 3, 5, 7],
         [8, 2, 2, 3, 4, 8],
         [8, 2, 1, 3, 5, 8],
         [8, 2, 0, 4, 5, 8],
         [9, 1, 2, 5, 5, 6],
         [9, 1, 0, 5, 5, 8],
         [9, 2, 2, 3, 5, 8],
         [10, 1, 1, 4, 4, 7],
         [11, 1, 2, 4, 4, 7],
         [11, 1, 1, 4, 4, 8],
         [11, 2, 1, 4, 5, 7],
         [12, 1, 2, 4, 4, 8],
         [12, 1, 1, 5, 5, 7],
         [12, 2, 2, 4, 5, 7],
         [12, 2, 1, 4, 5, 8],
         [13, 1, 2, 5, 5, 7],
         [13, 1, 1, 5, 5, 8],
         [13, 2, 2, 4, 5, 8],
         [14, 1, 2, 5, 5, 8]],
        [[0, 1, 0, 3, 6, 6],
         [1, 1, 1, 3, 6, 6],
         [1, 1, 0, 4, 6, 6],
         [1, 2, 0, 3, 6, 7],
         [2, 1, 2, 3, 6, 6],
         [2, 1, 0, 5, 6, 6],
         [2, 2, 0, 3, 6, 8],
         [3, 1, 1, 4, 6, 6],
         [3, 1, 0, 3, 7, 7],
         [3, 2, 1, 3, 6, 7],
         [3, 2, 0, 4, 6, 7],
         [4, 1, 2, 4, 6, 6],
         [4, 1, 1, 5, 6, 6],
         [4, 2, 2, 3, 6, 7],
         [4, 2, 1, 3, 6, 8],
         [4, 2, 0, 5, 6, 7],
         [4, 2, 0, 4, 6, 8],
         [4, 2, 0, 3, 7, 8],
         [5, 1, 2, 5, 6, 6],
         [5, 1, 0, 3, 8, 8],
         [5, 2, 2, 3, 6, 8],
         [5, 2, 0, 5, 6, 8],
         [6, 1, 1, 3, 7, 7],
         [6, 1, 0, 4, 7, 7],
         [6, 2, 1, 4, 6, 7],
         [7, 1, 2, 3, 7, 7],
         [7, 1, 0, 5, 7, 7],
         [7, 2, 2, 4, 6, 7],
         [7, 2, 1, 5, 6, 7],
         [7, 2, 1, 4, 6, 8],
         [7, 2, 1, 3, 7, 8],
         [7, 2, 0, 4, 7, 8],
         [8, 1, 1, 3, 8, 8],
         [8, 1, 0, 4, 8, 8],
         [8, 2, 2, 5, 6, 7],
         [8, 2, 2, 4, 6, 8],
         [8, 2, 2, 3, 7, 8],
         [8, 2, 1, 5, 6, 8],
         [8, 2, 0, 5, 7, 8],
         [9, 1, 2, 3, 8, 8],
         [9, 1, 0, 5, 8, 8],
         [9, 2, 2, 5, 6, 8],
         [10, 1, 1, 4, 7, 7],
         [11, 1, 2, 4, 7, 7],
         [11, 1, 1, 5, 7, 7],
         [11, 2, 1, 4, 7, 8],
         [12, 1, 2, 5, 7, 7],
         [12, 1, 1, 4, 8, 8],
         [12, 2, 2, 4, 7, 8],
         [12, 2, 1, 5, 7, 8],
         [13, 1, 2, 4, 8, 8],
         [13, 1, 1, 5, 8, 8],
         [13, 2, 2, 5, 7, 8],
         [14, 1, 2, 5, 8, 8]],
        [[0, 1, 0, 6, 6, 6],
         [1, 1, 1, 6, 6, 6],
         [1, 3, 0, 6, 6, 7],
         [2, 1, 2, 6, 6, 6],
         [2, 3, 0, 6, 6, 8],
         [3, 3, 1, 6, 6, 7],
         [3, 3, 0, 6, 7, 7],
         [4, 3, 2, 6, 6, 7],
         [4, 3, 1, 6, 6, 8],
         [4, 6, 0, 6, 7, 8],
         [5, 3, 2, 6, 6, 8],
         [5, 3, 0, 6, 8, 8],
         [6, 1, 0, 7, 7, 7],
         [6, 3, 1, 6, 7, 7],
         [7, 3, 2, 6, 7, 7],
         [7, 3, 0, 7, 7, 8],
         [7, 6, 1, 6, 7, 8],
         [8, 3, 1, 6, 8, 8],
         [8, 3, 0, 7, 8, 8],
         [8, 6, 2, 6, 7, 8],
         [9, 1, 0, 8, 8, 8],
         [9, 3, 2, 6, 8, 8],
         [10, 1, 1, 7, 7, 7],
         [11, 1, 2, 7, 7, 7],
         [11, 3, 1, 7, 7, 8],
         [12, 3, 2, 7, 7, 8],
         [12, 3, 1, 7, 8, 8],
         [13, 1, 1, 8, 8, 8],
         [13, 3, 2, 7, 8, 8],
         [14, 1, 2, 8, 8, 8]],
        [[0, 1, 3, 3, 3, 3],
         [1, 4, 3, 3, 3, 4],
         [2, 4, 3, 3, 3, 5],
         [3, 6, 3, 3, 4, 4],
         [4, 12, 3, 3, 4, 5],
         [5, 6, 3, 3, 5, 5],
         [6, 4, 3, 4, 4, 4],
         [7, 12, 3, 4, 4, 5],
         [8, 12, 3, 4, 5, 5],
         [9, 4, 3, 5, 5, 5],
         [10, 1, 4, 4, 4, 4],
         [11, 4, 4, 4, 4, 5],
         [12, 6, 4, 4, 5, 5],
         [13, 4, 4, 5, 5, 5],
         [14, 1, 5, 5, 5, 5]],
        [[0, 1, 3, 3, 3, 6],
         [1, 1, 3, 3, 3, 7],
         [1, 3, 3, 3, 4, 6],
         [2, 1, 3, 3, 3, 8],
         [2, 3, 3, 3, 5, 6],
         [3, 3, 3, 4, 4, 6],
         [3, 3, 3, 3, 4, 7],
         [4, 3, 3, 3, 5, 7],
         [4, 3, 3, 3, 4, 8],
         [4, 6, 3, 4, 5, 6],
         [5, 3, 3, 5, 5, 6],
         [5, 3, 3, 3, 5, 8],
         [6, 1, 4, 4, 4, 6],
         [6, 3, 3, 4, 4, 7],
         [7, 3, 4, 4, 5, 6],
         [7, 3, 3, 4, 4, 8],
         [7, 6, 3, 4, 5, 7],
         [8, 3, 4, 5, 5, 6],
         [8, 3, 3, 5, 5, 7],
         [8, 6, 3, 4, 5, 8],
         [9, 1, 5, 5, 5, 6],
         [9, 3, 3, 5, 5, 8],
         [10, 1, 4, 4, 4, 7],
         [11, 1, 4, 4, 4, 8],
         [11, 3, 4, 4, 5, 7],
         [12, 3, 4, 5, 5, 7],
         [12, 3, 4, 4, 5, 8],
         [13, 1, 5, 5, 5, 7],
         [13, 3, 4, 5, 5, 8],
         [14, 1, 5, 5, 5, 8]],
        [[0, 1, 3, 3, 6, 6],
         [1, 2, 3, 4, 6, 6],
         [1, 2, 3, 3, 6, 7],
         [2, 2, 3, 5, 6, 6],
         [2, 2, 3, 3, 6, 8],
         [3, 1, 4, 4, 6, 6],
         [3, 1, 3, 3, 7, 7],
         [3, 4, 3, 4, 6, 7],
         [4, 2, 4, 5, 6, 6],
         [4, 2, 3, 3, 7, 8],
         [4, 4, 3, 5, 6, 7],
         [4, 4, 3, 4, 6, 8],
         [5, 1, 5, 5, 6, 6],
         [5, 1, 3, 3, 8, 8],
         [5, 4, 3, 5, 6, 8],
         [6, 2, 4, 4, 6, 7],
         [6, 2, 3, 4, 7, 7],
         [7, 2, 4, 4, 6, 8],
         [7, 2, 3, 5, 7, 7],
         [7, 4, 4, 5, 6, 7],
         [7, 4, 3, 4, 7, 8],
         [8, 2, 5, 5, 6, 7],
         [8, 2, 3, 4, 8, 8],
         [8, 4, 4, 5, 6, 8],
         [8, 4, 3, 5, 7, 8],
         [9, 2, 5, 5, 6, 8],
         [9, 2, 3, 5, 8, 8],
         [10, 1, 4, 4, 7, 7],
         [11, 2, 4, 5, 7, 7],
         [11, 2, 4, 4, 7, 8],
         [12, 1, 5, 5, 7, 7],
         [12, 1, 4, 4, 8, 8],
         [12, 4, 4, 5, 7, 8],
         [13, 2, 5, 5, 7, 8],
         [13, 2, 4, 5, 8, 8],
         [14, 1, 5, 5, 8, 8]],
        [[0, 1, 3, 6, 6, 6],
         [1, 1, 4, 6, 6, 6],
         [1, 3, 3, 6, 6, 7],
         [2, 1, 5, 6, 6, 6],
         [2, 3, 3, 6, 6, 8],
         [3, 3, 4, 6, 6, 7],
         [3, 3, 3, 6, 7, 7],
         [4, 3, 5, 6, 6, 7],
         [4, 3, 4, 6, 6, 8],
         [4, 6, 3, 6, 7, 8],
         [5, 3, 5, 6, 6, 8],
         [5, 3, 3, 6, 8, 8],
         [6, 1, 3, 7, 7, 7],
         [6, 3, 4, 6, 7, 7],
         [7, 3, 5, 6, 7, 7],
         [7, 3, 3, 7, 7, 8],
         [7, 6, 4, 6, 7, 8],
         [8, 3, 4, 6, 8, 8],
         [8, 3, 3, 7, 8, 8],
         [8, 6, 5, 6, 7, 8],
         [9, 1, 3, 8, 8, 8],
         [9, 3, 5, 6, 8, 8],
         [10, 1, 4, 7, 7, 7],
         [11, 1, 5, 7, 7, 7],
         [11, 3, 4, 7, 7, 8],
         [12, 3, 5, 7, 7, 8],
         [12, 3, 4, 7, 8, 8],
         [13, 1, 4, 8, 8, 8],
         [13, 3, 5, 7, 8, 8],
         [14, 1, 5, 8, 8, 8]],
        [[0, 1, 6, 6, 6, 6],
         [1, 4, 6, 6, 6, 7],
         [2, 4, 6, 6, 6, 8],
         [3, 6, 6, 6, 7, 7],
         [4, 12, 6, 6, 7, 8],
         [5, 6, 6, 6, 8, 8],
         [6, 4, 6, 7, 7, 7],
         [7, 12, 6, 7, 7, 8],
         [8, 12, 6, 7, 8, 8],
         [9, 4, 6, 8, 8, 8],
         [10, 1, 7, 7, 7, 7],
         [11, 4, 7, 7, 7, 8],
         [12, 6, 7, 7, 8, 8],
         [13, 4, 7, 8, 8, 8],
         [14, 1, 8, 8, 8, 8]],
    ],
]


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
