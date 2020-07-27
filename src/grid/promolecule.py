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
"""Promolecular Module."""

from importlib_resources import path

import numpy as np

from scipy.interpolate import CubicSpline


class Promolecule:
    """Promolecule functions holder class."""

    def __init__(self):
        """Initialize class."""

    @staticmethod
    def _load_npz_radial(zatom):
        """Load the radial coordinate for an atom."""
        filename = "a" + str(zatom)
        with path("grid.data.promolecule.radial", filename) as npz_file:
            data = np.load(npz_file)
        return data["points"]

    @staticmethod
    def _load_npz_log_densities(zatom, charge):
        """Load the log of density for an atom."""
        filename = "a" + str(zatom)
        with path("grid.data.promolecule.density", filename) as npz_file:
            data = np.load(npz_file)
        return data["density"]

    @staticmethod
    def _get_proatomic_density(rad, zatom, charge):
        """Get the density of a spherical atom."""
        rad_0 = Promolecule._load_npz_radial(zatom)
        log_rho_0 = Promolecule._load_npz_log_densities(zatom, charge)
        cspline = CubicSpline(rad_0, log_rho_0, bc_type="natural", extrapolate=True)

        return np.exp(cspline(rad))

    def generate_promolecule(self, points, atom_coords, atom_nums, atom_charges):
        """Calculate the promolecular density.

        Parameters
        ----------
        points: np.ndarray(N,3)
            Cartesian coordinates of N grid points.
        atom_coords: np.ndarray(M,3)
            Cartesian coordinates of the M atoms in molecule.
        atom_nums: np.ndarray(M,)
            Atomic number of M atoms in molecule.
        atom_charges: np.ndarray(M,)
            Explicit charge in the atom.

        Return
        ------
        np.ndarray(N,)
        """
        promolecule = np.zeros(len(points))

        for atom in len(atom_nums):
            rad_coord = np.linalg.norm(points[:, None] - atom_coords[atom], axis=-1)

            promolecule += Promolecule._get_proatomic_density(
                rad_coord, atom_nums[atom], atom_charges[atom]
            )

        return promolecule

    def __call__(self, points, atom_coords, atom_nums, atom_charges):
        """Evaluate the promolecular density on the given grid points.

        Parameters
        ----------
        points: np.ndarray(N,3)
            Cartesian coordinates of N grid points.
        atom_coords: np.ndarray(M,3)
            Cartesian coordinates of the M atoms in molecule.
        atom_nums: np.ndarray(M,)
            Atomic number of M atoms in molecule.
        atom_charges: np.ndarray(M,)
            Explicit charge in the atom.

        Return
        ------
        np.ndarray(N,)
            The value of the promolecular density.
        """
        pass
