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
    """Becke weights functions holder class."""

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
        cutoff : float, optional
            Cutoff need to be smaller than 0.5 to guarantee monotonous
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
        order : int, optional
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
    def generate_becke_weights(
        points, radii, atom_coors, *, select=None, pt_ind=None, order=3
    ):
        """Calculate Becke weights of points for select atom.

        Parameters
        ----------
        points : np.ndarray(M, 3)
            Coordinates for each grid point
        radii : np.ndarray(N,)
            Covalent radii for each atom in molecule
        atom_coors : np.ndarray(N, 3)
            Coordinates for each atom in molecule
        select : list or integer, optional
            Index of atom index to calculate Becke weights
        pt_ind : list, optional
            Index of points for splitting sectors
        order : int, optional
            Order of iteration for switching function

        Returned
        ------------------
        np.ndarray(M, )
            Becke weights for each grid point
        """
        # select could be an array for more complicated case
        # |r_A - r| for each points, nucleus pair
        # check ``select``
        if select is None:
            select = np.arange(len(atom_coors))
        elif isinstance(select, (np.integer, int)):
            select = [select]
        # check ``pt_ind`` points index
        if pt_ind is None:
            pt_ind = []
        elif len(pt_ind) == 1:
            raise ValueError("pt_ind need include the ends of each section")
        # check how many sectors for becke weights need to be calculated
        sectors = max(len(pt_ind) - 1, 1)  # total sectors
        if atom_coors.ndim != 2:
            raise ValueError(
                f"Atom coors need to be in shape (N, 3), got {atom_coors.shape}"
            )
        if sectors != len(select):
            raise ValueError("# of select does not equal to # of indices.")
        weights = np.zeros(len(points))
        n_p = np.linalg.norm(atom_coors[:, None] - points, axis=-1)
        # |r_A - r| - |r_B - r| for each points with pair(A, B) nucleus
        p_p_n = n_p[:, None] - n_p
        atomic_dist = BeckeWeights._atomic_dists(atom_coors)
        # ignore 0 / 0 runtime warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_n_p_p = p_p_n.transpose([2, 0, 1]) / atomic_dist
        del p_p_n
        alpha = BeckeWeights._calculate_alpha(radii)
        v_pp = mu_n_p_p + alpha * (1 - mu_n_p_p ** 2)
        del mu_n_p_p
        s_ab = 0.5 * (1 - BeckeWeights._switch_func(v_pp, order=order))
        del v_pp
        # convert nan to 1
        s_ab[np.isnan(s_ab)] = 1
        # product up A_B, A_C, A_D ... along rows
        s_ab = np.prod(s_ab, axis=-1)
        # calculate weight for each point in select
        if sectors == 1:
            weights += s_ab[:, select[0]] / np.sum(s_ab, axis=-1)
        else:
            for i in range(sectors):
                sub_s_ab = s_ab[pt_ind[i] : pt_ind[i + 1]]
                weights[pt_ind[i] : pt_ind[i + 1]] += sub_s_ab[:, select[i]] / np.sum(
                    sub_s_ab, axis=-1
                )
        return weights
