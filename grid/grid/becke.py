# -*- coding: utf-8 -*-
# GRID is a numerical integration library for quantum chemistry.
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
#
# --
"""Becke Weights Module."""


import numpy as np


__all__ = ['becke_helper_atom']


def distance(point0, point1):
    """Compute the Euclidean distance between two points.

    Arguments
    ---------
    point0 : np.ndarray, shape=(3,)
        Cartesian coordinates  of the first point.
    point1 : np.ndarray, shape=(3,)
        Cartesian coordinates of the second point.
    """
    return np.sqrt(np.sum((point0 - point1)**2))


def becke_helper_atom(points, weights, radii, centers, select, order):
    """Compute Becke weights on the grid points for a given atom.

    Arguments
    ---------
    points : np.ndarray, shape=(npoints, 3)
        Cartesian coordinates of the grid points.
    weights : np.ndarray, shape=(npoints,)
        The output array where the Becke partitioning weights are
        written. Becke weights are **multiplied** with the original
        contents of the array!
    radii : np.ndarray, shape=(natom,)
        Covalent radii used to shrink/enlarge basins in the Becke scheme.
    centers : np.ndarray, shape=(natom, 3)
        Cartesian coordinates of the nuclei.
    select : np.ndarray
        Index of atoms for which the Becke weights are computed.
    order : np.ndarray
        Order of switching function in the Becke scheme.

    Notes
    -----
    See A. D. Becke, The Journal of Chemical Physics 88, 2547 (1988).
    """

    npoint = points.shape[0]
    natom = radii.shape[0]

    assert points.shape[1] == 3, "points does not have the right size!"
    assert weights.shape[0] == npoint, "weights does not have the right size!"
    assert centers.shape[0] == natom, "centers does not have the right size!"
    assert centers.shape[1] == 3, "centers does not have the right size!"
    assert 0 <= select < natom, "select must be in the range [0, natom)"
    assert order > 0, "order must be greater than zero"

    # Precompute the alpha parameters for each atom pair
    alphas = np.zeros(int(natom * (natom + 1) / 2))
    offset = 0
    for iatom in range(natom):
        for jatom in range(iatom + 1):
            alpha = (radii[iatom] - radii[jatom]) / (radii[iatom] + radii[jatom])
            alpha = alpha / (alpha**2 - 1)  # Eq. (A5)
            # Eq. (A3), except that we use some safe margin
            # (0.45 instead of 0.5) to stay way from a ridiculous imbalance.
            if alpha > 0.45:
                alpha = 0.45
            elif alpha < -0.45:
                alpha = -0.45
            alphas[offset] = alpha
            offset += 1

    # Precompute interatomic distances
    atomic_dists = np.zeros(int(natom * (natom + 1) / 2))
    offset = 0

    for iatom in range(natom):
        for jatom in range(iatom + 1):
            atomic_dists[offset] = distance(centers[iatom], centers[jatom])
            offset += 1

    # Calculate the Becke Weights
    for ipoint in range(npoint - 1, -1, -1):
        itmp = npoint - ipoint - 1
        num = 0
        den = 0
        for iatom in range(natom):
            p = 1
            for jatom in range(natom):
                if iatom == jatom:
                    continue
                # Compute offset for alpha and interatomic distance
                if iatom < jatom:
                    offset = int((jatom * (jatom + 1) / 2)) + iatom
                    term = 1
                else:
                    offset = int((iatom * (iatom + 1) / 2)) + jatom
                    term = 0

                # Diatomic switching function
                s = distance(points[itmp], centers[iatom]) - \
                    distance(points[itmp], centers[jatom])
                s = s / atomic_dists[offset]  # Eq. (11)
                s = s + alphas[offset] * (1 - 2 * term) * (1 - s**2)  # Eq. (A2)

                for k in range(1, order + 1, 1):  # Eq. (19) and (20)
                    s = 0.5 * s * (3 - s**2)

                s = 0.5 * (1 - s)  # Eq. (18)

                p *= s  # Eq. (13)

            if iatom == select:
                nom = p

            den += p  # Eq. (22)

        # Weight function at this grid point
        weights[itmp] *= (nom / den)  # Eq. (22)
