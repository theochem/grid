# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2020 The GRID Development Team.
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
"""Hirshfeld Weights Module."""


import numpy as np
from importlib_resources import files
from scipy.interpolate import CubicSpline


class HirshfeldWeights:
    """Hirshfeld weights functions holder class."""

    def __init__(self):
        """Initialize class."""

    @staticmethod
    def _load_npz_proatom(num):
        """Return radial grid points and neutral density for a given atomic number."""
        data = np.load(files("grid.data.proatoms").joinpath(f"a{num:03d}.npz"))
        return data["r"], data["dn"]

    @staticmethod
    def _get_proatom_density(num, coords_radial):
        """Evaluate density of pro-atom on the given points.

        Parameters
        ----------
        num: int
            Atomic number of pro-atom.
        coords_radial: np.ndarray(K,)
            Radial grid points.

        Returns
        -------
        np.ndarray(K,)
            Pro-atom densities evaluated on :math:`K` radial grid points.
        """
        # get pre-computed radial grid points & density of pro-atom
        rad, rho = HirshfeldWeights._load_npz_proatom(num)
        # interpolate pro-atom density
        cspline = CubicSpline(rad, rho, bc_type="natural", extrapolate=True)
        # evaluate pro-atom density on the given points
        out = cspline(coords_radial).flatten()
        return out

    @staticmethod
    def generate_proatom(points, coord, num):
        """Evaluate pro-atom densities on the given grid points.

        Parameters
        ----------
        points: np.ndarray(N, 3)
            Cartesian coordinates of :math:`N` grid points.
        coord: np.ndarray(3,)
            Cartesian coordinates of the atom.
        num: int
            Atomic number of the atom.

        Returns
        -------
        np.ndarray(N,)
            Pro-atom densities evaluated on :math:`N` grid points.
        """
        # compute distance of grid points from atom
        dist = np.linalg.norm(points[:, None] - coord, axis=-1)
        # compute pro-atom density
        return HirshfeldWeights._get_proatom_density(num, dist)

    def __call__(self, points, atcoords, atnums, indices):
        """Evaluate integration weights on the given grid points.

        Parameters
        ----------
        points: np.ndarray(N,3)
            Cartesian coordinates of :math:`N` grid points.
        atcoords: np.ndarray(M,3)
            Cartesian coordinates of the :math:`M` atoms in molecule.
        atnums: np.ndarray(M,)
            Atomic number of :math:`M` atoms in molecule.
        indices : np.ndarray(M+1,)
            Indices of atomic grid points for each :math:`M` atoms in molecule.

        Return
        ------
        np.ndarray(N,)
            Hirshfeld integration weights evaluated on :math:`N` grid points.
        """
        if atnums.dtype != int:
            raise TypeError(f"Parameter atnums dtype {atnums.dtype} should be int.")
        aim_weights = np.zeros(len(points))
        promolecule = np.zeros(len(points))
        # evaluate (neutral) pro-atom densities & pro-molecule density
        for index, atnum in enumerate(atnums):
            proatom = HirshfeldWeights.generate_proatom(points, atcoords[index], atnum)
            promolecule += proatom
            start, end = indices[index], indices[index + 1]
            aim_weights[start:end] = proatom[start:end]
        # evaluate weights
        aim_weights /= promolecule
        # check weights are all positive
        # assert np.all(aim_weights >= 0), np.min(aim_weights)
        return aim_weights
