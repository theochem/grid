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

from typing import Union

from grid.atomgrid import AtomGrid
from grid.basegrid import Grid, LocalGrid, OneDGrid

import numpy as np


class MolGrid(Grid):
    r"""
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
        store: bool = False,
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
            self._points[start:end] = atom_grid.points  # centers it at the atomic grid.
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
        Get the indices of the molecualr grid.

        Returns
        -------
        ndarray(M + 1,) :
            List of indices :math:`[i_0, i_1, \cdots]` that whose indices range [i_k, i_{k+1}]
            specify which points in `points` correspond to kth atomic grid.

        """
        return self._indices

    @property
    def aim_weights(self):
        r"""
        Get the atom in molecules/nuclear weight function of the molecular grid.

        Returns
        -------
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
        Get the atomic grid weights for all points, i.e. without atom in molecule weights.

        Returns
        -------
        ndarray(K,):
            List of weights of atomic grid for each point,where :math:`K = \sum_n N_i` and
            :math:`N_i` is the number of points in the ith atomic grid.

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
            dict_save["atgrid_" + str(i) + "_points"] = atomgrid.points
            dict_save["atgrid_" + str(i) + "_weights"] = atomgrid.weights
            dict_save["atgrid_" + str(i) + "_center"] = atomgrid.center
            dict_save["atgrid_" + str(i) + "_degrees"] = atomgrid.degrees
            dict_save["atgrid_" + str(i) + "_indices"] = atomgrid.indices
            dict_save["atgrid_" + str(i) + "_rgrid_pts"] = atomgrid.rgrid.points
            dict_save["atgrid_" + str(i) + "_rgrid_weights"] = atomgrid.rgrid.weights
        np.savez(filename, **dict_save)

    def interpolate(self, func_vals: np.ndarray):
        r"""
        Return function that interpolates (and its derivatives) from function values.

        Consider a real-valued function :math:`f(r, \theta, \phi) = \sum_A f_A(r, \theta, \phi)`
        written as a sum of atomic centers :math:`f_A(r, \theta, \phi)`.  Each of these
        functions can be further decomposed based on the atom grid centered at :math:`A`:

        .. math::
            f_A(r, \theta, \phi) = \sum_l \sum_{m=-l}^l \sum_i \rho^A_{ilm}(r) Y_{lm}(\theta, \phi)

        A cubic spline is used to interpolate the radial functions :math:`\sum_i \rho^A_{ilm}(r)`
        based on the atomic grid centered at :math:`A`.
        This is then multipled by the corresponding spherical harmonics at all
        :math:`(\theta_j, \phi_j)` angles and summed to obtain approximation to :math:`f_A`. This
        is then further summed over all centers to get :math:`f`.

        Parameters
        ----------
        func_vals: ndarray(\sum_i N_i,)
            The function values evaluated on all :math:`N_i` points on the :math:`i`th atomic grid.

        Returns
        -------
        Callable[[ndarray(M, 3), int] -> ndarray(M)]:
            Callable function that interpolates the function and its derivative provided.
            The function takes the following attributes:

                points : ndarray(N, 3)
                    Cartesian coordinates of :math:`N` points to evaluate the splines on.
                deriv : int, optional
                    If deriv is zero, then only returns function values. If it is one, then
                    returns the first derivative of the interpolated function with respect to either
                    Cartesian or spherical coordinates. Only higher-order derivatives
                    (`deriv`=2,3) are supported for the derivatives wrt to radial components.
                deriv_spherical : bool
                    If True, then returns the derivatives with respect to spherical coordinates
                    :math:`(r, \theta, \phi)`. Default False.
                only_radial_deriv : bool
                    If true, then the derivative wrt to radius :math:`r` is returned.

            This function returns the following.

                ndarray(M,...):
                    The interpolated function values or its derivatives with respect to Cartesian
                    :math:`(x,y,z)` or if `deriv_spherical` then :math:`(r, \theta, \phi)` or
                    if `only_radial_derivs` then derivative wrt to :math:`r` is only returned.

        Examples
        --------
        >>> # Consider the function (3x^2 + 4y^2 + 5z^2)
        >>> def polynomial_func(pts) :
        >>>     return 3.0 * points[:, 0]**2.0 + 4.0 * points[:, 1]**2.0 + 5.0 * points[:, 2]**2.0
        >>> # Evaluate the polynomial over the molecular grid points and interpolate
        >>> polynomial_vals = polynomial_func(molgrid.points)
        >>> interpolate_func = molgrid.interpolate(polynomial_vals)
        >>> # Use it to interpolate at new points.
        >>> interpolate_vals = interpolate_func(new_pts)
        >>> # Can calculate first derivative wrt to Cartesian or spherical
        >>> interpolate_derivs = interpolate_func(new_pts, deriv=1)
        >>> interpolate_derivs_sph = interpolate_func(new_pts, deriv=1, deriv_spherical=True)
        >>> # Only higher-order derivatives are supported for the radius coordinate r.
        >>> interpolated_derivs_radial = interpolate_func(new_pts, deriv=2, only_radial_derivs=True)

        """
        if self.atgrids is None:
            raise ValueError(
                "Atomic grids need to be stored in molecular grid for this to work. "
                "Turn `store` attribute to true."
            )
        # Multiply f by the nuclear weight function w_n(r) for each atom grid segment.
        func_vals_atom = func_vals * self.aim_weights
        # Go through each atomic grid and construct interpolation of f*w_n.
        intepolate_funcs = []
        for i in range(len(self.atcoords)):
            start_index = self.indices[i]
            final_index = self.indices[i + 1]
            atom_grid = self[i]
            intepolate_funcs.append(
                atom_grid.interpolate(func_vals_atom[start_index:final_index])
            )

        def interpolate_low(
            points, deriv=0, deriv_spherical=False, only_radial_derivs=False
        ):
            r"""Construct a spline like callable for intepolation.

            Parameters
            ----------
            points : ndarray(N, 3)
                Cartesian coordinates of :math:`N` points to evaluate the splines on.
            deriv : int, optional
                If deriv is zero, then only returns function values. If it is one, then returns
                the first derivative of the interpolated function with respect to either Cartesian
                or spherical coordinates. Only higher-order derivatives (`deriv` =2,3) are supported
                for the derivatives wrt to radial components. `deriv=3` only returns a constant.
            deriv_spherical : bool
                If True, then returns the derivatives with respect to spherical coordinates
                :math:`(r, \theta, \phi)`. Default False.
            only_radial_derivs : bool
                If true, then the derivative wrt to radius :math:`r` is returned.

            Returns
            -------
            ndarray(M,...) :
                The interpolated function values or its derivatives with respect to Cartesian
                :math:`(x,y,z)` or if `deriv_spherical` then :math:`(r, \theta, \phi)` or
                if `only_radial_derivs` then derivative wrt to :math:`r` is only returned.

            """
            output = intepolate_funcs[0](
                points, deriv, deriv_spherical, only_radial_derivs
            )
            for interpolate in intepolate_funcs[1:]:
                output += interpolate(
                    points, deriv, deriv_spherical, only_radial_derivs
                )
            return output

        return interpolate_low

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

    @classmethod
    def from_pruned(
        cls,
        atnums: np.ndarray,
        atcoords: np.ndarray,
        rgrid: Union[OneDGrid, list],
        radius: Union[float, list],
        aim_weights: Union[callable, np.ndarray],
        sectors_r: np.ndarray,
        sectors_degree: np.ndarray = None,
        sectors_size: np.ndarray = None,
        rotate: int = 37,
        store: bool = False,
    ):
        r"""
        Initialize a MolGrid instance with pruned method from AtomGrid.

        Parameters
        ----------
        atnums: ndarray(M,)
            List of atomic numbers for each atom.
        atcoords: np.ndarray(M, 3)
            Cartesian coordinates for each atoms
        rgrid : OneDGrid or List[OneDGrid] or Dict[int: OneDGrid]
            One dimensional grid for the radial component.  If a list is provided,then ith
            grid correspond to the ith atom.  If dictionary is provided, then the keys are
            correspond to the `atnums[i]` attribute.
        radius: float, List[float]
            The atomic radius to be multiplied with `r_sectors` (to make them atom specific).
            If float, then the same atomic radius is used for all atoms, else a list specifies
            it for each atom.
        aim_weights: Callable or np.ndarray(\sum^M_n N_n,)
            Atoms in molecule/nuclear weights :math:`{ {w_n(r_k)}_k^{N_i}}_n^{M}`, where
            :math:`N_i` is the number of points in the ith atomic grid.
        sectors_r: List[List], keyword-only argument
            Each row is a sequence of boundary points specifying radial sectors of the pruned grid
            for the `m`th atom. The first sector is ``[0, radius*sectors_r[0]]``, then
            ``[radius*sectors_r[0], radius*sectors_r[1]]``, and so on.
        sectors_degree: List[List], keyword-only argument
            Each row is a sequence of Lebedev/angular degrees for each radial sector of the pruned
            grid for the `m`th atom. If both `sectors_degree` and `sectors_size` are given,
            `sectors_degree` is used.
        sectors_size: List[List], keyword-only argument
            Each row is a sequence of Lebedev sizes for each radial sector of the pruned grid
            for the `m`th atom. If both `sectors_degree` and `sectors_size` are given,
            `sectors_degree` is used.
        rotate : bool or int , optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.
        store : bool, optional
            Flag to store each original atomic grid information.

        Returns
        -------
        MolGrid:
            Molecular grid class for integration.

        """
        if atcoords.ndim != 2:
            raise ValueError(
                "The dimension of coordinates need to be 2\n"
                f"got shape: {atcoords.ndim}"
            )

        at_grids = []
        num_atoms = len(atcoords)
        # List of None is created, so that indexing is possible in the for-loop.
        sectors_degree = (
            [None] * num_atoms if sectors_degree is None else sectors_degree
        )
        sectors_size = [None] * num_atoms if sectors_size is None else sectors_size
        radius_atom = (
            [radius] * num_atoms if isinstance(radius, (float, np.float64)) else radius
        )
        for i in range(num_atoms):
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

            at_grids.append(
                AtomGrid.from_pruned(
                    rad,
                    radius_atom[i],
                    sectors_r=sectors_r[i],
                    sectors_degree=sectors_degree[i],
                    sectors_size=sectors_size[i],
                    center=atcoords[i],
                    rotate=rotate,
                )
            )
        return cls(atnums, at_grids, aim_weights, store=store)

    def get_atomic_grid(self, index: int):
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
