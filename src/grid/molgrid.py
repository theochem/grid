"""Molecular grid class."""
# from grid.atomic_grid import AtomicGrid
from grid.atomic_grid import AtomicGrid
from grid.basegrid import Grid, SimpleAtomicGrid
from grid.becke import BeckeWeights
from grid.utils import get_cov_radii

import numpy as np


class MolGrid(Grid):
    """Molecular Grid for integration."""

    def __init__(self, atomic_grids, numbers, aim_weights="becke", store=False):
        """Initialize molgrid class.

        Parameters
        ----------
        atomic_grids : list[AtomicGrid]
            list of atomic grid
        radii : np.ndarray(N,)
            Radii for each atom in the molecular grid
        aim_weights : str or np.ndarray(K,), default to "becke"
            Atoms in molecule weights. If str, certain function will be called
            to compute aim_weights, if np.ndarray, it will be treated as the
            aim_weights
        """
        # initialize these attributes
        numbers = np.array(numbers)
        radii = get_cov_radii(numbers)
        self._coors = np.zeros((len(radii), 3))
        self._indices = np.zeros(len(radii) + 1, dtype=int)
        self._size = np.sum([atomgrid.size for atomgrid in atomic_grids])
        self._points = np.zeros((self._size, 3))
        self._weights = np.zeros(self._size)
        self._atomic_grids = atomic_grids if store else None

        for i, atom_grid in enumerate(atomic_grids):
            self._coors[i] = atom_grid.center
            self._indices[i + 1] += self._indices[i] + atom_grid.size
            self._points[self._indices[i] : self._indices[i + 1]] = atom_grid.points
            self._weights[self._indices[i] : self._indices[i + 1]] = atom_grid.weights

        if isinstance(aim_weights, str):
            if aim_weights == "becke":
                self._aim_weights = BeckeWeights.generate_becke_weights(
                    self._points, radii, self._coors, pt_ind=self._indices
                )
            else:
                raise NotImplementedError(
                    f"Given aim_weights is not supported, got {aim_weights}"
                )
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
    def horton_molgrid(cls, coors, numbers, radial, points_of_angular, store=False):
        """Initialize a MolGrid instance with Horton Style input.

        Parameters
        ----------
        coors : np.ndarray(N, 3)
            Cartesian coordinates for each atoms
        numbers : np.ndarray(N,)
            Atomic number for each atoms
        radial : RadialGrid
            RadialGrid instance for constructing atomic grid for each atom
        points_of_angular : TYPE
            Num of points on each shell of angular grid
        store : bool, default to False
            Flag to store each original atomic grid information

        Returns
        -------
        MolGrid
            MolGrid instance with specified grid property
        """
        at_grids = []
        for i, _ in enumerate(numbers):
            at_grids.append(
                AtomicGrid(radial, nums=[points_of_angular], center=coors[i])
            )
        return cls(at_grids, numbers, store=store)

    def get_atomic_grid(self, index):
        """Get the stored atomic grid with all infromation.

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
        """Get a simple atomic grid with points, weights, and center.

        Parameters
        ----------
        index : int
            index of atomic grid for constructing molecular grid.
            index starts from 0 to n-1
        with_aim_wts : bool, default to True
            The flag for pre-multiply molecular weights
            if True, the weights *= aim_weights

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
            The intergral of the desired integrand(s)

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
