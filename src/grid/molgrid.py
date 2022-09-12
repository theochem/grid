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

from typing import Union

class MolGrid(Grid):
    """
    Molecular grid class for integration of three-dimensional functions.

    Molecular grid is defined here to be a weighted average of :math:`M` atomic grids
    (see AtomGrid). This is defined by a atom in molecule weights (or nuclear weight functions)
    :math:`w_n(r)` for each center n such that :math:`\sum^M_n w_n(r) = 1` for all points
    :math:`r\in\mathbb{R}^3.`

    Examples
    --------
    There are multiple methods of specifiying molecular grids.
    This example chooses Becke weights as the atom in molecule/nuclear weights and the
    radial grid is the same for all atoms.  Two atoms are considered with charges [1, 2],
    respectively.
    >>> from grid.becke BeckeWeights
    >>> from grid.onedgrid import GaussLaguerre
    >>> becke = BeckeWeights(order=3)
    >>> rgrid = GaussLaguerre(100)
    >>> charges = [1, 2]
    >>> coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    The default method is based on explicitly specifing the atomic grids (AtomGrids) for each atom.
    >>> from grid.atomgrid import AtomGrid
    >>> atgrid1 = AtomGrid(rgrid, degrees=5, center=coords[0])
    >>> atgrid2 = AtomGrid(rgrid, degrees=10, center=coords[1])
    >>> molgrid = MolGrid(charges, [atgrid1, atgrid2], aim_weights=becke)
    The `from_size` method constructs AtomGrids with degree_size specified from integer size.
    >>> size = 100  # Number of angular points used in each shell in the atomic grid.
    >>> molgrid = MolGrid.from_size(charges, coords, rgrid, size=5, aim_weights=becke)
    The `from_pruned` method is based on `AtomGrid.from_pruned` method on the idea
    of spliting radial grid points into sectors that have the same angular degrees.
    >>> sectors_r = [[0.5, 1., 1.5], [0.25, 0.5]]
    >>> sectors_deg = [[3, 7, 5, 3], [3, 2, 2]]
    >>> radius = 1.0
    >>> mol_grid = MolGrid.from_pruned(charges, coords, rgrid, radius, becke,
    >>>                                sectors_r=sectors_r, sectors_degree=sectors_deg)
    The `from_preset` method is based on `AtomGrid.from_preset` method based on a string
    specifying the size of each Levedev grid at each radial points.
    >>> preset = "fine"  # Many choices available.
    >>> molgrid = MolGrid.from_preset(charges, coords, rgrid, preset, aim_weights=becke)

    The general way to integrate is the following.
    >>> integrand = integrand_func(molgrid.points)
    >>> integrate = molgrid.integrate(integrand)

    References
    ----------
    .. [1] Becke, Axel D. "A multicenter numerical integration scheme for polyatomic molecules."
       The Journal of chemical physics 88.4 (1988): 2547-2553.

    """

    def __init__(
        self,
        atnums: np.ndarray,
        atgrids: list,
        aim_weights: Union[callable, np.ndarray],
        store: bool = False
    ):
        r"""Initialize class.

        Parameters
        ----------
        atnums : np.ndarray(M, 3)
            Atomic number of :math:`M` atoms in molecule.
        atgrids : list[AtomGrid]
            List of atomic grids of size :math:`M` for each atom in molecule.
        aim_weights : Callable or np.ndarray(\sum^M_n N_n,)
            Atoms in molecule weights :math:`{ {w_n(r_k)}_k^{N_i}}_n^{M}`, where
            :math:`N_i` is the number of points in the ith atomic grid.
        store: bool
            If true, then the atomic grids `atgrids` are stored as attribute `atgrids`.

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

    @property
    def atgrids(self):
        r"""List[AtomGrid] : List of atomic grids for each center. Returns None if `store` false."""
        return self._atgrids

    @property
    def indices(self):
        r"""
        ndarray(M + 1,) :
            List of indices :math:`[i_0, i_1, \cdots]` that whose indices range [i_k, i_{k+1}]
            specify which points in `points` correspond to kth atomic grid.

        """
        return self._indices

    @property
    def aim_weights(self):
        r"""
        ndarray(K,):
            List of atomic weights where :math:`K = \sum_n N_i` and :math:`N_i` is the number
            of points in the ith atomic grid.
        """
        return self._aim_weights
    @property
    def atcoords(self):
        r"""ndarray(M, 3) : Center/Atomic coordinates."""
        return self._atcoords

    @property
    def atweights(self):
        r"""
        ndarray(K,):
            List of weight for each point,where :math:`K = \sum_n N_i` and :math:`N_i` is the
            number of points in the ith atomic grid.
        """
        return self._atweights

    def save(self, filename):
        r"""
        Save molecular grid attributes as a npz file.

        Parameters
        ----------
        filename: str
           The path/name of the .npz file.

        """
        dict_save = {
            "points": self.points,
            "weights": self.weights,
            "atweights": self.atweights,
            "atcoords": self.atcoords,
            "aim_weights": self.aim_weights,
            "indices": self.indices,
        }
        # Save each attribute of the atomic grid.
        for i, atomgrid in enumerate(self.atgrids):
            dict_save["atgrid_" + str(i) + "_points"] = atomgrid.points,
            dict_save["atgrid_" + str(i) + "_weights"] = atomgrid.weights,
            dict_save["atgrid_" + str(i) + "_center"] = atomgrid.center,
            dict_save["atgrid_" + str(i) + "_degrees"] = atomgrid.degrees,
            dict_save["atgrid_" + str(i) + "_indices"] = atomgrid.indices,
            dict_save["atgrid_" + str(i) + "_rgrid_pts"] = atomgrid.rgrid.points,
            dict_save["atgrid_" + str(i) + "_rgrid_weights"] = atomgrid.rgrid.weights,
        np.savez(filename, **save_dict)

    @classmethod
    def from_preset(
        cls,
        atnums: np.ndarray,
        atcoords: np.ndarray,
        rgrid: Union[OneDGrid, list, dict],
        preset: Union[str, list, dict],
        aim_weights: Union[callable, np.ndarray],
        *_,
        rotate: int = 37,
        store: bool = False,
    ):
        """Construct molecular grid wih preset parameters.

        Parameters
        ----------
        atnums : np.ndarray(M,)
            Array of atomic numbers.
        atcoords : np.ndarray(M, 3)
            Atomic coordinates of atoms.
        rgrid : (OneDGrid, list[OneDGrid], dict[int: OneDGrid])
            One dimensional radial grid. If of type `OneDGrid` then this radial grid is used for
            all atoms. If a list is provided,then ith grid correspond to the ith atom.  If
            dictionary is provided, then the keys correspond to the `atnums[i]`attribute.
        preset : (str, list[str], dict[int: str])
            Preset grid accuracy scheme, support "coarse", "medium", "fine",
            "veryfine", "ultrafine", "insane".  If string is provided ,then preset is used
            for all atoms, either it is specified by a list, or a dictionary whose keys
            are from `atnums`.
        aim_weights : Callable or np.ndarray(K,)
            Atoms in molecule weights.
        rotate : bool or int, optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.
        store : bool, optional
            Store atomic grid as a class attribute.

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
            if isinstance(preset, str):
                gd_type = preset
            elif isinstance(preset, list):
                gd_type = preset[i]
            elif isinstance(preset, dict):
                gd_type = preset[atnums[i]]
            else:
                raise TypeError(
                    f"Not supported preset type; got preset {preset} with type {type(preset)}"
                )
            at_grid = AtomGrid.from_preset(
                rad, atnum=atnums[i], preset=gd_type, center=atcoords[i], rotate=rotate
            )
            atomic_grids.append(at_grid)
        return cls(atnums, atomic_grids, aim_weights, store=store)

    @classmethod
    def from_size(
        cls,
        atnums: np.ndarray,
        atcoords: np.ndarray,
        rgrid: OneDGrid,
        size: int,
        aim_weights: Union[callable, np.ndarray],
        rotate: int = 37,
        store: bool = False,
    ):
        """
        Initialize a MolGrid instance with Horton Style input.

        Example
        -------
        >>> onedg = UniformInteger(100) # number of points, oned grid before TF.
        >>> rgrid = ExpRTransform(1e-5, 2e1).generate_radial(onedg) # radial grid
        >>> becke = BeckeWeights(order=3)
        >>> molgrid = MolGrid.from_size(atnums, atcoords, rgrid, 110, becke)

        Parameters
        ----------
        atnums : np.ndarray(M, 3)
            Atomic number of :math:`M` atoms in molecule.
        atcoords : np.ndarray(N, 3)
            Cartesian coordinates for each atoms
        rgrid : OneDGrid
            one dimension grid  to construct spherical grid
        size : int
            Num of points on each shell of angular grid
        aim_weights : Callable or np.ndarray(K,)
            Atoms in molecule weights.
        rotate : bool or int , optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.
        store : bool, optional
            Flag to store each original atomic grid information

        Returns
        -------
        MolGrid
            MolGrid instance with specified grid property

        """
        at_grids = []
        for i in range(len(atcoords)):
            at_grids.append(
                AtomGrid(rgrid, sizes=[size], center=atcoords[i], rotate=rotate)
            )
        return cls(atnums, at_grids, aim_weights, store=store)

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

    def __getitem__(self, index: int):
        """Get separate atomic grid in molecules.

        Parameters
        ----------
        index : int
            Index of the atom in the molecule.

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
