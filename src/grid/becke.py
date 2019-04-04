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

import warnings

import numpy as np


class BeckeWeights:
    """Beckec weights functions holder class."""

    @staticmethod
    def _calculate_alpha(radii, cutoff=0.45):
        r"""Calculate parameter alpha to tune the size of the basins.

        .. math::
            u_{AB} &= \frac{R_A - R_B}{R_A + R_B} \\
            a_{AB} &= \frac{u_{AB}}{u_{AB}^2 - 1}

        Parameters
        ----------
        radii : np.array(N,)
            Covalent radii of each atoms in the molecule
        cutoff : float, default 0.45
            Cutoff need to be smaller than 0.5 to ganrantee monotonous
            transformation.

        Returns
        -------
        np.ndarray(N, N)
            alpha value for each pair of atoms
        """
        u_ab = (radii[:, None] - radii) / (radii[:, None] + radii)
        alpha = u_ab / (u_ab ** 2 - 1)
        alpha[alpha > cutoff] = cutoff
        alpha[alpha < -cutoff] = -cutoff
        return alpha

    @staticmethod
    def _atomic_dists(coors):
        """Calculate atomic distance between each atoms.

        Parameters
        ----------
        coors : np.ndarray(N, 3)
            Cartesian coordinates of each atom

        Returns
        -------
        np.ndarray(N, N)
            Atomic distance between each pair atoms
        """
        return np.linalg.norm(coors[:, None] - coors, axis=-1)

    @staticmethod
    def _switch_func(x, order=3):
        r"""Switching function that gradient at nuclei become zero.

        .. math::
            f_1(x) = \frac{x}{2}(3 - x^2)
            f_k(x) = f_1(f_{k-1}(x))

        Parameters
        ----------
        x : float or np.ndarray
            Input variable
        order : int, default to 3
            Order of iteration for switching function

        Returns
        -------
        float or np.ndarray
            result of switching function
        """
        for i in range(order):
            x = 1.5 * x - 0.5 * x ** 3
        return x

    @staticmethod
    def generate_becke_weights(points, radii, atom_coors, select_index, order=3):
        """Calculate becke weights of points for select atom.

        Parameters
        ----------
        points : np.ndarray(M, 3)
            Coordinates for each grid point
        radii : np.ndarray(N,)
            Covalent radiis for each atom in molecule
        atom_coors : np.ndarray(N, 3)
            Coordinates for each atom in molecule
        select_index : int, np.int
            Index of atom for calculate becke weights
        order : int, default to 3
            Order of iteration for switching function

        Returns
        -------
        np.ndarray(M,)
            Becke weights for each grid point
        """
        # select_index could be an array for more complicated case
        # |r_A - r| for each points, nucleus pair
        n_p = np.linalg.norm(atom_coors[:, None] - points, axis=-1)
        # |r_A - r| - |r_B - r| for each points with pair(A, B) nucleus
        p_p_n = n_p[:, None] - n_p
        atomic_dist = BeckeWeights._atomic_dists(atom_coors)
        # ignore 0 / 0 runtime warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_n_p_p = p_p_n.transpose([2, 0, 1]) / atomic_dist
        alpha = BeckeWeights._calculate_alpha(radii)
        v_pp = mu_n_p_p + alpha * (1 - mu_n_p_p ** 2)
        s_ab = 0.5 * (1 - BeckeWeights._switch_func(v_pp, order=order))
        # convert nan to 1
        s_ab[np.isnan(s_ab)] = 1
        # product up A_B, A_C, A_D ... along rows
        s_ab = np.prod(s_ab, axis=-1)
        # calculate weight for each point for select_index
        weights = s_ab[:, select_index] / np.sum(s_ab, axis=-1)
        return weights


# def distance(point0, point1):
#     """Compute the Euclidean distance between two points.

#     Arguments
#     ---------
#     point0 : np.ndarray, shape=(3,)
#         Cartesian coordinates  of the first point.
#     point1 : np.ndarray, shape=(3,)
#         Cartesian coordinates of the second point.
#     """
#     return np.sqrt(np.sum((point0 - point1) ** 2))


# def becke_helper_atom(points, weights, radii, centers, select, order=3):
#     """Compute Becke weights on the grid points for a given atom.

#     Arguments
#     ---------
#     points : np.ndarray, shape=(npoints, 3)
#         Cartesian coordinates of the grid points.
#     weights : np.ndarray, shape=(npoints,)
#         The output array where the Becke partitioning weights are
#         written. Becke weights are **multiplied** with the original
#         contents of the array!
#     radii : np.ndarray, shape=(natom,)
#         Covalent radii used to shrink/enlarge basins in the Becke scheme.
#     centers : np.ndarray, shape=(natom, 3)
#         Cartesian coordinates of the nuclei.
#     select : np.ndarray
#         Index of atoms for which the Becke weights are computed.
#     order : np.ndarray
#         Order of switching function in the Becke scheme.

#     Notes
#     -----
#     See A. D. Becke, The Journal of Chemical Physics 88, 2547 (1988).
#     """

#     npoint = points.shape[0]
#     natom = radii.shape[0]

#     assert points.shape[1] == 3, "points does not have the right size!"
#     assert weights.shape[0] == npoint, "weights does not have the right size!"
#     assert centers.shape[0] == natom, "centers does not have the right size!"
#     assert centers.shape[1] == 3, "centers does not have the right size!"
#     assert 0 <= select < natom, "select must be in the range [0, natom)"
#     assert order > 0, "order must be greater than zero"

#     # Precompute the alpha parameters for each atom pair
#     alphas = np.zeros(int(natom * (natom + 1) / 2))
#     offset = 0
#     for iatom in range(natom):
#         for jatom in range(iatom + 1):
#             alpha = (radii[iatom] - radii[jatom]) / (radii[iatom] + radii[jatom])
#             alpha = alpha / (alpha ** 2 - 1)  # Eq. (A5)
#             # Eq. (A3), except that we use some safe margin
#             # (0.45 instead of 0.5) to stay way from a ridiculous imbalance.
#             if alpha > 0.45:
#                 alpha = 0.45
#             elif alpha < -0.45:
#                 alpha = -0.45
#             alphas[offset] = alpha
#             offset += 1

#     # Precompute interatomic distances
#     atomic_dists = np.zeros(int(natom * (natom + 1) / 2))
#     offset = 0

#     for iatom in range(natom):
#         for jatom in range(iatom + 1):
#             atomic_dists[offset] = distance(centers[iatom], centers[jatom])
#             offset += 1

#     # Calculate the Becke Weights
#     for ipoint in range(npoint - 1, -1, -1):
#         itmp = npoint - ipoint - 1
#         den = 0
#         for iatom in range(natom):
#             p = 1
#             for jatom in range(natom):
#                 if iatom == jatom:
#                     continue
#                 # Compute offset for alpha and interatomic distance
#                 if iatom < jatom:
#                     offset = int((jatom * (jatom + 1) / 2)) + iatom
#                     term = 1
#                 else:
#                     offset = int((iatom * (iatom + 1) / 2)) + jatom
#                     term = 0

#                 # Diatomic switching function
#                 s = distance(points[itmp], centers[iatom]) - distance(
#                     points[itmp], centers[jatom]
#                 )
#                 s = s / atomic_dists[offset]  # Eq. (11)
#                 s = s + alphas[offset] * (1 - 2 * term) * (1 - s ** 2)  # Eq. (A2)

#                 for k in range(1, order + 1, 1):  # Eq. (19) and (20)
#                     s = 0.5 * s * (3 - s ** 2)

#                 s = 0.5 * (1 - s)  # Eq. (18)

#                 p *= s  # Eq. (13)

#             if iatom == select:
#                 nom = p

#             den += p  # Eq. (22)

#         # Weight function at this grid point
#         weights[itmp] *= nom / den  # Eq. (22)
