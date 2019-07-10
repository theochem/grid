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
"""Molecular grid class."""


from grid.atomic_grid import AtomicGrid
from grid.basegrid import Grid, SimpleAtomicGrid

import numpy as np


class MolGrid(Grid):
    """Molecular Grid for integration."""

    def __init__(self, atomic_grids, aim_weights, store=False):
        r"""Initialize class.

        Parameters
        ----------
        atomic_grids : list[AtomicGrid]
            list of atomic grid
        aim_weights : Callable or np.ndarray(K,)
            Atoms in molecule weights.

        """
        # initialize these attributes
        self._coors = np.zeros((len(atomic_grids), 3))
        self._indices = np.zeros(len(atomic_grids) + 1, dtype=int)
        self._size = np.sum([atomgrid.size for atomgrid in atomic_grids])
        self._points = np.zeros((self._size, 3))
        self._weights = np.zeros(self._size)
        self._atomic_grids = atomic_grids if store else None

        for i, atom_grid in enumerate(atomic_grids):
            self._coors[i] = atom_grid.center
            self._indices[i + 1] += self._indices[i] + atom_grid.size
            self._points[self._indices[i] : self._indices[i + 1]] = atom_grid.points
            self._weights[self._indices[i] : self._indices[i + 1]] = atom_grid.weights

        if callable(aim_weights):
            self._aim_weights = aim_weights(self._points, self._coors, self._indices)

        elif isinstance(aim_weights, np.ndarray):
            if aim_weights.size != self.size:
                raise ValueError(
                    "aim_weights is not the same size as grid.\n"
                    f"aim_weights.size: {aim_weights.size}, grid.size: {self.size}."
                )
            self._aim_weights = aim_weights

        else:
            raise TypeError(f"Not supported aim_weights type, got {type(aim_weights)}.")

    @classmethod
    def horton_molgrid(
        cls, coors, radial, points_of_angular, aim_weights, store=False, rotate=False
    ):
        """Initialize a MolGrid instance with Horton Style input.

        Example
        -------
        >>> onedg = HortonLinear(100) # number of points, oned grid before TF.
        >>> rgrid = ExpRTransform(1e-5, 2e1).generate_radial(onedg) # radial grid
        >>> becke = BeckeWeights(coords, numbers)
        >>> molgrid = MolGrid.horton_molgrid(coors, rgrid, 110, becke)

        Parameters
        ----------
        coors : np.ndarray(N, 3)
            Cartesian coordinates for each atoms
        points_of_angular : int
            Num of points on each shell of angular grid
        aim_weights : Callable or np.ndarray(K,)
            Atoms in molecule weights.
        store : bool, optional
            Flag to store each original atomic grid information
        rotate : bool or int , optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.

        Returns
        -------
        MolGrid
            MolGrid instance with specified grid property
        """
        at_grids = []
        for i in range(len(coors)):
            at_grids.append(
                AtomicGrid(
                    radial, nums=[points_of_angular], center=coors[i], rotate=rotate
                )
            )
        return cls(at_grids, aim_weights, store=store)

    def get_atomic_grid(self, index):
        """Get the stored atomic grid with all information.

        Parameters
        ----------
        index : int
            index of atomic grid for constructing molecular grid.
            index starts from 0 to n-1

        Returns
        -------
        AtomicGrid
            AtomicGrid for n_th atom in the molecular grid

        Raises
        ------
        NotImplementedError
            If the atomic grid information is not store
        ValueError
            The input index is negative
        """
        if self._atomic_grids is None:
            raise NotImplementedError(
                "Atomic Grid info is not stored during initialization."
            )
        if index < 0:
            raise ValueError(f"Invalid negative index value, got {index}")
        return self._atomic_grids[index]

    def get_simple_atomic_grid(self, index, with_aim_wts=True):
        r"""Get a simple atomic grid with points, weights, and center.

        Parameters
        ----------
        index : int
            index of atomic grid for constructing molecular grid.
            index starts from 0 to n-1
        with_aim_wts : bool, default to True
            The flag for pre-multiply molecular weights
            if True, the weights \*= aim_weights

        Returns
        -------
        SimpleAtomicGrid
            A SimpleAtomicGrid instance for local integral
        """
        s_ind = self._indices[index]
        f_ind = self._indices[index + 1]
        # coors
        pts = self.points[s_ind:f_ind]
        # wts
        wts = self.weights[s_ind:f_ind]
        if with_aim_wts:
            wts *= self.get_aim_weights(index)
        # generate simple atomic grid
        return SimpleAtomicGrid(pts, wts, self._coors[index])

    @property
    def aim_weights(self):
        """np.ndarray(N,): atom in molecular weights for all points in grid."""
        return self._aim_weights

    def get_aim_weights(self, index):
        """Get aim weights value for given atoms in the molecule.

        Parameters
        ----------
        index : int
            index of atomic grid for constructing molecular grid.
            index starts from 0 to n-1

        Returns
        -------
        np.ndarray(K,)
            The aim_weights for points in the given atomic grid

        Raises
        ------
        ValueError
            The input index is negative
        """
        if index >= 0:
            return self._aim_weights[self._indices[index] : self._indices[index + 1]]
        else:
            raise ValueError(f"Invalid negative index value, got {index}")

    def integrate(self, *value_arrays):
        """Integrate given value_arrays on molecular grid.

        Parameters
        ----------
        *value_arrays, np.ndarray
            Evaluated integrand on the grid

        Returns
        -------
        float
            The integral of the desired integrand(s)

        Raises
        ------
        TypeError
            Given value_arrays is not np.ndarray
        ValueError
            The size of the value_arrays does not match with grid size.
        """
        if len(value_arrays) < 1:
            raise ValueError(f"No array is given to integrate.")
        for i, array in enumerate(value_arrays):
            if not isinstance(array, np.ndarray):
                raise TypeError(f"Arg {i} is {type(i)}, Need Numpy Array.")
            if array.size != self.size:
                raise ValueError(f"Arg {i} need to be of shape {self.size}.")
        return np.einsum(
            "i, i" + ",i" * len(value_arrays),
            self.weights,
            self.aim_weights,
            *(np.ravel(i) for i in value_arrays),
        )

    def __getitem__(self, index):
        """Get separate atomic grid in molecules.

        Same function as get_simple_atomic_grid. May be removed in the future.

        Parameters
        ----------
        index : int
            Index of atom in the molecule

        Returns
        -------
        AtomicGrid
            AtomicGrid of desired atom with aim weights integrated
        """
        if self._atomic_grids is None:
            s_ind = self._indices[index]
            f_ind = self._indices[index + 1]
            return SimpleAtomicGrid(
                self.points[s_ind:f_ind],
                self.weights[s_ind:f_ind] * self.aim_weights[s_ind:f_ind],
                self._coors[index],
            )
        return self._atomic_grids[index]
