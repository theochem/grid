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


from grid.atomgrid import AtomGrid
from grid.basegrid import Grid, LocalGrid, OneDGrid

import numpy as np


class MolGrid(Grid):
    """Molecular Grid for integration."""

    def __init__(self, atgrids, aim_weights, atnums, store=False):
        r"""Initialize class.

        Parameters
        ----------
        atgrids : list[AtomGrid]
            list of atomic grid
        aim_weights : Callable or np.ndarray(K,)
            Atoms in molecule weights.
        atnums : np.ndarray(M, 3)
            Atomic number of :math:`M` atoms in molecule.

        """
        # initialize these attributes
        self._atcoords = np.zeros((len(atgrids), 3))
        self._indices = np.zeros(len(atgrids) + 1, dtype=int)
        size = np.sum([atomgrid.size for atomgrid in atgrids])
        self._points = np.zeros((size, 3))
        self._atweights = np.zeros(size)
        self._atgrids = atgrids if store else None

        for i, atom_grid in enumerate(atgrids):
            self._atcoords[i] = atom_grid.center
            self._indices[i + 1] += self._indices[i] + atom_grid.size
            start, end = self._indices[i], self._indices[i + 1]
            self._points[start:end] = atom_grid.points
            self._atweights[start:end] = atom_grid.weights

        if callable(aim_weights):
            self._aim_weights = aim_weights(
                self._points, self._atcoords, atnums, self._indices
            )

        elif isinstance(aim_weights, np.ndarray):
            if aim_weights.size != size:
                raise ValueError(
                    "aim_weights is not the same size as grid.\n"
                    f"aim_weights.size: {aim_weights.size}, grid.size: {size}."
                )
            self._aim_weights = aim_weights

        else:
            raise TypeError(f"Not supported aim_weights type, got {type(aim_weights)}.")

        # initialize parent class
        super().__init__(self.points, self._atweights * self._aim_weights)

    @classmethod
    def make_grid(
        cls,
        atnums,
        atcoords,
        rgrid,
        grid_type,
        aim_weights,
        *_,
        rotate=False,
        store=False,
    ):
        """Contruct molecular grid wih preset parameters.

        Parameters
        ----------
        atnums : np.ndarray(N,)
            array of atomic number
        atcoords : np.ndarray(N, 3)
            atomic coordinates of atoms
        rgrid : OneDGrid
            one dimension grid  to construct spherical grid
        grid_type : str
            preset grid accuracy scheme, support "coarse", "medium", "fine",
            "veryfine", "ultrafine", "insane"
        aim_weights : Callable or np.ndarray(K,)
            Atoms in molecule weights.
        rotate : bool, optional
            Random rotate for each shell of atomic grid
        store : bool, optional
            Store atomic grid separately
        """
        # construct for a atom molecule
        if atcoords.ndim != 2:
            raise ValueError(
                "The dimension of coordinates need to be 2\n"
                f"got shape: {atcoords.ndim}"
            )
        if len(atnums) != atcoords.shape[0]:
            raise ValueError(
                "shape of atomic nums does not match with coordinates\n"
                f"atomic numbers: {atnums.shape}, coordinates: {atcoords.shape}"
            )
        total_atm = len(atnums)
        atomic_grids = []
        for i in range(total_atm):
            # get proper radial grid
            if isinstance(rgrid, OneDGrid):
                rad = rgrid
            elif isinstance(rgrid, list):
                rad = rgrid[i]
            elif isinstance(rgrid, dict):
                rad = rgrid[atnums[i]]
            else:
                raise TypeError(
                    f"not supported radial grid input; got input type: {type(rgrid)}"
                )
            # get proper grid type
            if isinstance(grid_type, str):
                gd_type = grid_type
            elif isinstance(grid_type, list):
                gd_type = grid_type[i]
            elif isinstance(grid_type, dict):
                gd_type = grid_type[atnums[i]]
            else:
                raise TypeError(
                    "not supported grid_type input\n"
                    f"got input type: {type(grid_type)}"
                )
            at_grid = AtomGrid.from_predefined(
                rad, atnums[i], gd_type, center=atcoords[i], rotate=rotate
            )
            atomic_grids.append(at_grid)
        return cls(atomic_grids, aim_weights, atnums, store=store)

    @classmethod
    def horton_molgrid(
        cls,
        atcoords,
        atnums,
        radial,
        points_of_angular,
        aim_weights,
        store=False,
        rotate=False,
    ):
        """Initialize a MolGrid instance with Horton Style input.

        Example
        -------
        >>> onedg = HortonLinear(100) # number of points, oned grid before TF.
        >>> rgrid = ExpRTransform(1e-5, 2e1).generate_radial(onedg) # radial grid
        >>> becke = BeckeWeights(order=3)
        >>> molgrid = MolGrid.horton_molgrid(atcoords, atnums,rgrid,110,becke)

        Parameters
        ----------
        atcoords : np.ndarray(N, 3)
            Cartesian coordinates for each atoms
        atnums : np.ndarray(M, 3)
            Atomic number of :math:`M` atoms in molecule.
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
        for i in range(len(atcoords)):
            at_grids.append(
                AtomGrid(
                    radial, sizes=[points_of_angular], center=atcoords[i], rotate=rotate
                )
            )
        return cls(at_grids, aim_weights, atnums, store=store)

    def get_atomic_grid(self, index):
        r"""Get atomic grid corresponding to the given atomic index.

        Parameters
        ----------
        index : int
            index of atom starting from 0 to :math:`N-1` where :math:`N` is the
            number of atoms in molecule.

        Returns
        -------
        AtomGrid or LocalGrid
            If store=True, the AtomGrid instance used is returned.
            If store=False, the LocalGrid containing points and weights of AtomGrid
            is returned.

        """
        if index < 0:
            raise ValueError(f"index should be non-negative, got {index}")
        # get atomic grid if stored
        if self._atgrids is not None:
            return self._atgrids[index]
        # make a local grid
        pts = self.points[self._indices[index] : self._indices[index + 1]]
        wts = self._atweights[self._indices[index] : self._indices[index + 1]]
        return LocalGrid(pts, wts, self._atcoords[index])

    @property
    def aim_weights(self):
        """np.ndarray(N,): atom in molecular weights for all points in grid."""
        return self._aim_weights

    def __getitem__(self, index):
        """Get separate atomic grid in molecules.

        Same function as get_simple_atomic_grid. May be removed in the future.

        Parameters
        ----------
        index : int
            Index of atom in the molecule

        Returns
        -------
        AtomGrid
            AtomGrid of desired atom with aim weights integrated
        """
        if self._atgrids is None:
            s_ind = self._indices[index]
            f_ind = self._indices[index + 1]
            return LocalGrid(
                self.points[s_ind:f_ind],
                self.weights[s_ind:f_ind],
                self._atcoords[index],
            )
        return self._atgrids[index]
