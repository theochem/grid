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
from grid.utils import get_cov_radii

import numpy as np

import importlib as importlib
from importlib_resources import path


class MolGrid(Grid):
    """Molecular Grid for integration."""

    def __init__(self, atnums, atgrids, aim_weights, store=False):
        r"""Initialize class.

        Parameters
        ----------
        atnums : np.ndarray(M, 3)
            Atomic number of :math:`M` atoms in molecule.
        atgrids : list[AtomGrid]
            list of atomic grid
        aim_weights : Callable or np.ndarray(K,)
            Atoms in molecule weights.

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
    def from_preset(
        cls,
        atnums,
        atcoords,
        rgrid,
        preset,
        aim_weights,
        *_,
        rotate=37,
        store=False,
    ):
        """Construct molecular grid wih preset parameters.

        Parameters
        ----------
        atnums : np.ndarray(N,)
            array of atomic number
        atcoords : np.ndarray(N, 3)
            atomic coordinates of atoms
        rgrid : OneDGrid
            one dimension grid  to construct spherical grid
        preset : str
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
        atnums,
        atcoords,
        rgrid,
        size,
        aim_weights,
        rotate=37,
        store=False,
    ):
        """Initialize a MolGrid instance with Horton Style input.

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
                AtomGrid(rgrid, sizes=[size], center=atcoords[i], rotate=rotate)
            )
        return cls(atnums, at_grids, aim_weights, store=store)

    @classmethod
    def pruned_grid(
            cls,
            atnums,
            atcoords,
            rgrid_type,
            rtransform_type,
            type_pruned,
            aim_weights,
            rotate=False,
            store=False,
    ):
        """Construct molecular pruned grid.

        Parameters
        ----------
        atnums : np.ndarray(N,)
            array of atomic number
        atcoords : np.ndarray(N, 3)
            atomic coordinates of atoms
        rgrid_type: str
            type or radial grid to use. Options available in grid.onedgrid
        rtransform_type: str
            type of radial transformation. Should include name of transformation and options.
            e.g. 'ExpRTransform, 1e-5, 2e1'. Options available in grid.rtransform
        type_pruned : str
            Type of pruned grid to use. Options are:
            - 'sg_0', 'sg_1','sg_2' and 'sg_3'.
            - Ochsenfeld improved becke grids 'g_1', 'g_2','g_3', 'g_4', 'g_5', 'g_6', 'g_7'
            references:
                - sg_0: Chien, S.-H. and Gill, P.M.W. (2006), J. Comput. Chem., 27: 730-739.
                        doi: rm10.1002/jcc.20383.
                - sg_1: P. M. W. Gill, B. G. Johnson, and J. A. Pople. Chem. Phys. Lett., 209:0 506,
                        1993.
                        doi: rm10.1002/qua.560400604.
                - sg_2 and sg_3: Dasgupta, S., Herbert, J. M. J. Comput. Chem. 2017, 38, 869– 882.
                                 doi: 10.1002/jcc.24761
                - Ochsenfeld improved becke grids: Laqua, H.,Kussmann, J. and Ochsenfeld, C.
                                                   J. Chem. Phys. 149, 204111 (2018);
                                                    doi:10.1063/1.5049435

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
        if not isinstance(type_pruned, str):
            raise ValueError('Pruned grid type must be a string')

        elif type_pruned not in ['sg_0', 'sg_1', 'sg_2', 'sg_3',
                                 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7']:
            raise ValueError(f"type_pruned {type_pruned} not recognized as a valid pruned grid")

        with path("grid.data.prune_grid", f"prune_grid_{type_pruned}.npz") as npz_file:
            data = np.load(npz_file)
        at_grids = []
        for idx_1, at_num in enumerate(atnums):
            # load predefined_radial sectors and num of points in each sector
            if type_pruned == 'sg_1':
                n_rad = data['r_points']
            rad = data[f"{at_num}_rad"]
            npt = data[f"{at_num}_npt"]
            try:
                # Initialize oned grid with predefined number of radial points
                onedg_type = importlib.import_module("grid.onedgrid").__getattribute__(rgrid_type)
                if type_pruned == 'sg_1':
                    onedg = onedg_type(n_rad)
                else:
                    onedg = onedg_type(sum(rad))
            except AttributeError:
                raise ValueError('This one dimensional grid is not implemented in Grid package')
            try:
                # Unpack type of transformation and its options
                rtransform_type_list = rtransform_type.split(",")
                radial_transform_opt = [float(arg) for arg in rtransform_type_list[1:]]
                onedg_transf_type = importlib.import_module("grid.rtransform").__getattribute__(
                    rtransform_type_list[0])
                rgrid = onedg_transf_type(*radial_transform_opt).transform_1d_grid(onedg)
            except AttributeError:
                raise ValueError('This radial grid transform is not implemented in Grid package')

            sizes = []
            # Generate elements with the number of angular points for all the radial
            # points included in that radial sector
            if type_pruned == 'sg_1':
                # Sectors obtained with alpha parameter
                data_radii = get_cov_radii(np.arange(1, 87, 1), "bragg")
                bragg_radii = dict([(i + 1, radius) for i, radius in enumerate(data_radii)])
                sphere_radii = [bragg_radii[at_num] * r for r in rad]
                sphere_radii.insert(0, 0)
                points_sector = []
                for idx, r in enumerate(sphere_radii):
                    if idx == len(sphere_radii)-1:
                        raw_points_sector = rgrid.points[rgrid.points > r]
                    else:
                        raw_points_sector = rgrid.points[rgrid.points > r]
                        raw_points_sector = raw_points_sector[
                                                    raw_points_sector <= sphere_radii[idx + 1]]
                    points_sector.append(raw_points_sector)
                sizes = []
                for idx_2 in range(len(points_sector)):
                    sector_sizes = [npt[idx_2] for r_p in range(len(points_sector[idx_2]))]
                    sizes.extend(sector_sizes)
            else:
                for idx_2 in range(len(rad)):
                    sector_sizes = [npt[idx_2] for r_p in range(rad[idx_2])]
                    sizes.extend(sector_sizes)
            at_grids.append(AtomGrid(rgrid, sizes=sizes, center=atcoords[idx_1], rotate=rotate))
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
