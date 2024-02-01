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
"""Becke Weights Module."""


import warnings

import numpy as np

from grid.utils import get_cov_radii


class BeckeWeights:
    """Becke weights functions holder class."""

    def __init__(self, radii=None, order=3):
        r"""Initialize class.

        Parameters
        ----------
        radii : dict, optional
            Dictionary of atomic number and corresponding atomic radius.
            If None, Bragg-Slater empirically measured covalent radii
            (in atomic units) are used.
        order : int, optional
            Order of iteration for switching function.

        """
        if not isinstance(order, int):
            raise ValueError(f"order should be an integer, got {type(order)}")
        self._order = order
        # make dictionary of covalent radius for elements up to Z=86
        data = get_cov_radii(np.arange(1, 87, 1), "bragg")
        self._radii = dict([(i + 1, radius) for i, radius in enumerate(data)])
        # update given covalent radii
        if radii is not None:
            if not isinstance(radii, dict):
                raise TypeError(f"radii should be a dictionary, got {type(radii)}")
            if not np.all([isinstance(k, int) for k in radii.keys()]):
                raise TypeError("radii keys have non-integers value")
            self._radii.update(radii)

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
        alpha = u_ab / (u_ab**2 - 1)
        alpha[alpha > cutoff] = cutoff
        alpha[alpha < -cutoff] = -cutoff
        return alpha

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
        for _i in range(order):
            x = 1.5 * x - 0.5 * x**3
        return x

    def generate_weights(self, points, atcoords, atnums, *, select=None, pt_ind=None):
        r"""Calculate Becke integration weights of points for select atom.

        The units of the points and coordinates should match `radii` attribute.

        Parameters
        ----------
        points : np.ndarray(N, 3)
            Cartesian coordinates of :math:`N` grid points.
        atcoords : np.ndarray(M, 3)
            Cartesian coordinates of :math:`M` atoms in molecule.
        atnums : np.ndarray(M,)
            Atomic number of :math:`M` atoms in molecule.
        select : list or integer, optional
            Index of atom index to calculate Becke weights
        pt_ind : list, optional
            Index of points for splitting sectors

        Return
        ------
        np.ndarray(N, )
            Becke integration weights of :math:`N` grid points.

        """
        # select could be an array for more complicated case
        # |r_A - r| for each points, nucleus pair
        # check ``select``
        if select is None:
            select = np.arange(len(atcoords))
        elif isinstance(select, (np.integer, int)):
            select = [select]
        # check ``pt_ind`` points index
        if pt_ind is None:
            pt_ind = []
        elif len(pt_ind) == 1:
            raise ValueError("pt_ind need include the ends of each section")
        # check how many sectors for becke weights need to be calculated
        sectors = max(len(pt_ind) - 1, 1)  # total sectors
        if sectors != len(select):
            raise ValueError("# of select does not equal to # of indices.")
        weights = np.zeros(len(points))
        n_p = np.linalg.norm(atcoords[:, None] - points, axis=-1)
        # shape of n_p is (#nucs, #points)
        # |r_A - r| - |r_B - r| for each points with pair(A, B) nucleus
        # (#nucs, None, #points) - (#nucs, #points) -> (#nucs, #nucs, #points)
        n_n_p = n_p[:, None] - n_p
        atomic_dist = np.linalg.norm(atcoords[:, None] - atcoords, axis=-1)
        # ignore 0 / 0 runtime warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # mu_p_n_n shape (#points, #uncs, #uncs)
            mu_p_n_n = n_n_p.transpose([2, 0, 1]) / atomic_dist
        del n_n_p
        # if the radii of an atom is np.nan, use the radii with 1 less atomic number
        specified_radius = [self._radii[num] for num in atnums]
        indices = np.where(np.isnan(specified_radius))[0]
        if len(indices) != 0:
            warnings.warn(
                f"Covalent radii for the following atom numbers {atnums[indices]} is nan."
                f" Instead the radii with 1 less the atomic number is used.",
                stacklevel=2,
            )
        radii = np.array(
            [
                (
                    self._radii[num]
                    if not np.isnan(self._radii[num])
                    # if n-1 radii is nan, use the n-2 instead
                    else np.nan_to_num(self._radii[num - 1]) or np.nan_to_num(self._radii[num - 2])
                )
                for num in atnums
            ]
        )
        alpha = BeckeWeights._calculate_alpha(radii)
        v_pp = mu_p_n_n + alpha * (1 - mu_p_n_n**2)
        del mu_p_n_n
        s_ab = 0.5 * (1 - BeckeWeights._switch_func(v_pp, order=self._order))
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

    def compute_atom_weight(self, points, atcoords, atnums, select, cutoff=0.45):
        """Compute Becke weights for given atomic grid points.

        Parameters
        ----------
        points : np.ndarray(N, 3)
            Coordinates of points from given atomic grid
        atcoords : np.ndarray(M, 3)
            Coordinates of nucleis
        atnums : np.ndarray(M,)
            Atomic number for each nuclei
        select : int
            Index of atom A for computing the weights
        cutoff : float, optional
            Cufoff for a_AB if the value is too big or too small

        Returns
        -------
        np.ndarray(N,)
            Becke weights of points for selected atom
        """
        # #Nucs: M, #Pts: N
        # initialize points array (N,)
        weights = np.zeros(len(points))
        # shape of n_p is (M, N)
        n_p = np.linalg.norm(atcoords[:, None] - points, axis=-1)
        # |r_A - r| - |r_B - r| for each points with pair(A, B) nucleus
        # (M, None, N) - (M, N) -> (M, M, N)
        n_n_p = n_p[:, None] - n_p
        atomic_dist = np.linalg.norm(atcoords[:, None] - atcoords, axis=-1)
        # ignore 0 / 0 runtime warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # mu_p_n_n shape (N, M, M)
            mu_p_n_n = n_n_p.transpose([2, 0, 1]) / atomic_dist
        del n_n_p
        # if the radii of an atom is np.nan, use the radii with 1 less atomic number
        specified_radius = [self._radii[num] for num in atnums]
        indices = np.where(np.isnan(specified_radius))[0]
        if len(indices) != 0:
            warnings.warn(
                f"Covalent radii for the following atom numbers {atnums[indices]} is nan."
                f" Instead the radii with 1 less the atomic number is used.",
                stacklevel=2,
            )
        radii = np.array(
            [
                (
                    self._radii[num]
                    if not np.isnan(self._radii[num])
                    # if n-1 radii is nan, use the n-2 instead
                    else np.nan_to_num(self._radii[num - 1]) or np.nan_to_num(self._radii[num - 2])
                )
                for num in atnums
            ]
        )
        alpha = BeckeWeights._calculate_alpha(radii)
        v_pp = mu_p_n_n + alpha * (1 - mu_p_n_n**2)
        del mu_p_n_n
        s_ab = 0.5 * (1 - BeckeWeights._switch_func(v_pp, order=self._order))
        del v_pp
        # convert nan to 1
        s_ab[np.isnan(s_ab)] = 1
        # product up A_B, A_C, A_D ... along rows
        s_ab = np.prod(s_ab, axis=-1)
        # calculate weight for each point in select
        weights += s_ab[:, select] / np.sum(s_ab, axis=-1)
        return weights

    def compute_weights(self, points, atcoords, atnums, *, select=None, pt_ind=None):
        """Compute becke weights for given points and select atoms.

        Parameters
        ----------
        points : np.ndarray(N, 3)
            Cartesian coordinates of :math:`N` grid points.
        atcoords : np.ndarray(M, 3)
            Cartesian coordinates of :math:`M` atoms in molecule.
        atnums : np.ndarray(M,)
            Atomic number of :math:`M` atoms in molecule.
        select : list or integer, optional
            Index of atom index to calculate Becke weights
        pt_ind : list, optional
            Index of points for splitting sectors

        Return
        ------
        np.ndarray(N, )
            Becke integration weights of :math:`N` grid points.
        """
        if select is None:
            select = np.arange(len(atcoords))
        elif isinstance(select, (np.integer, int)):
            select = [select]
        # check ``pt_ind`` points index
        if pt_ind is None:
            pt_ind = []
        elif len(pt_ind) == 1:
            raise ValueError("pt_ind need include the ends of each section")
        # check how many sectors for becke weights need to be calculated
        sectors = max(len(pt_ind) - 1, 1)  # total sectors
        if sectors != len(select):
            raise ValueError("# of select does not equal to # of indices.")
        weights = np.zeros(len(points))
        # only weight for one atom
        if sectors == 1:
            weights += self.compute_atom_weight(points, atcoords, atnums, select[0])
        else:
            for i in select:
                ind_start = pt_ind[i]
                ind_end = pt_ind[i + 1]
                weights[ind_start:ind_end] += self.compute_atom_weight(
                    points[ind_start:ind_end], atcoords, atnums, i
                )
        return weights

    def __call__(self, points, atcoords, atnums, indices):
        r"""Evaluate integration weights on the given grid points.

        The units of the points and coordinates should match `radii` attribute.

        Parameters
        ----------
        points : np.ndarray(N, 3)
            Cartesian coordinates of :math:`N` grid points.
        atcoords : np.ndarray(M, 3)
            Cartesian coordinates of :math:`M` atoms in molecule.
        atnums : np.ndarray(M, 3)
            Atomic number of :math:`M` atoms in molecule.
        indices : np.ndarray(M+1,)
            Indices of atomic grid points for each :math:`M` atoms in molecule.

        Returns
        -------
        np.ndarray(N,)
            Becke integration weights of :math:`N` grid points.

        """
        # Becke weights are computed for "chunks" of grid points
        # to counteract the scaling of the memory usage of the
        # vectorized implementation of the Becke partitioning.
        npoints = points.shape[0]
        chunk_size = max(1, (10 * npoints) // atcoords.shape[0] ** 2)
        aim_weights = np.concatenate(
            [
                self.generate_weights(
                    points[ibegin : ibegin + chunk_size],
                    atcoords,
                    atnums,
                    pt_ind=(indices - ibegin).clip(min=0),
                )
                for ibegin in range(0, npoints, chunk_size)
            ]
        )
        return aim_weights
