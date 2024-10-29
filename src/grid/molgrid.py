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

from __future__ import annotations

import numpy as np
import scipy.constants

from grid.atomgrid import AtomGrid
from grid.basegrid import Grid, LocalGrid, OneDGrid
from grid.becke import BeckeWeights
from grid.onedgrid import UniformInteger
from grid.rtransform import PowerRTransform
from grid.utils import _DEFAULT_POWER_RTRANSFORM_PARAMS


class MolGrid(Grid):
    r"""
    Molecular grid class for integration of three-dimensional functions.

    Molecular grid is defined here to be a weighted average of :math:`M` atomic grids
    (see AtomGrid). This is defined by a atom in molecule weights (or nuclear weight functions)
    :math:`w_n(r)` for each center n such that :math:`\sum^M_n w_n(r) = 1` for all points
    :math:`r\in\mathbb{R}^3.`

    References
    ----------
    .. [1] Becke, Axel D. "A multicenter numerical integration scheme for polyatomic molecules."
       The Journal of chemical physics 88.4 (1988): 2547-2553.

    """

    def __init__(
        self,
        atnums: np.ndarray,
        atgrids: list,
        aim_weights: callable | np.ndarray,
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
            self._aim_weights = aim_weights(self._points, self._atcoords, atnums, self._indices)

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
        Get the indices of the molecular grid.

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
            intepolate_funcs.append(atom_grid.interpolate(func_vals_atom[start_index:final_index]))

        def interpolate_low(points, deriv=0, deriv_spherical=False, only_radial_derivs=False):
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
            output = intepolate_funcs[0](points, deriv, deriv_spherical, only_radial_derivs)
            for interpolate in intepolate_funcs[1:]:
                output += interpolate(points, deriv, deriv_spherical, only_radial_derivs)
            return output

        return interpolate_low

    @classmethod
    def from_preset(
        cls,
        atnums: np.ndarray,
        atcoords: np.ndarray,
        preset: str | list | dict,
        rgrid: OneDGrid | list | dict | None = None,
        aim_weights: callable | np.ndarray = None,
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
        preset : (str, list[str], dict[int: str])
            Preset grid accuracy scheme. If string is provided, then preset is used
            for all atoms, either it is specified by a list, or a dictionary whose keys
            are from `atnums`. These predefined grid specify the radial sectors and
            their corresponding number of Lebedev grid points. Supported preset options include:
            'coarse', 'medium', 'fine', 'veryfine', 'ultrafine', and 'insane'.
            Other options include the "standard grids":
            'sg_0', 'sg_1', 'sg_2', and 'sg_3', and the Ochsenfeld grids:
            'g1', 'g2', 'g3', 'g4', 'g5', 'g6', and 'g7', with higher number indicating
            greater accuracy but denser grid. See `Notes` for more information.
        rgrid : (OneDGrid, list[OneDGrid], dict[int: OneDGrid], None), optional
            One dimensional radial grid. If of type `OneDGrid` then this radial grid is used for
            all atoms. If a list is provided, then ith grid correspond to the ith atom.  If
            dictionary is provided, then the keys correspond to the `atnums[i]`attribute.
            If None, a default radial grid (PowerRTransform of UniformInteger grid) is constructed
            based on the given atomic numbers.
        aim_weights : Callable or np.ndarray(K,), optional
            Atoms in molecule weights. If None, then aim_weights is Becke weights with order=3.
        rotate : bool or int, optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.
        store : bool, optional
            Store atomic grid as a class attribute.

        Notes
        -----
        - The standard and Ochsenfeld presets were not designed with symmetric spherical t-design
          in mind.
        - The "standard grids" [1]_ "SG-0" and "SG-1" are designed for large molecules with LDA
          (GGA) functionals, whereas "SG-2" and "SG-3" are designed for Meta-GGA functionals and
          B95/Minnesota functionals, respectively.
        - The Ochsenfeld pruned grids [2]_ are obtained based on the paper.

        References
        ----------
        .. [2] Y. Shao, et al. Advances in molecular quantum chemistry contained in the Q-Chem 4
               program package. Mol. Phys. 113, 184-215 (2015)
        .. [3] Laqua, H., Kussmann, J., & Ochsenfeld, C. (2018). An improved molecular partitioning
               scheme for numerical quadratures in density functional theory. The Journal of
               Chemical Physics, 149(20).

        """
        # construct for a atom molecule
        if atcoords.ndim != 2:
            raise ValueError(
                "The dimension of coordinates need to be 2\n" f"got shape: {atcoords.ndim}"
            )
        if len(atnums) != atcoords.shape[0]:
            raise ValueError(
                "shape of atomic nums does not match with coordinates\n"
                f"atomic numbers: {atnums.shape}, coordinates: {atcoords.shape}"
            )
        if aim_weights is None:
            aim_weights = BeckeWeights(order=3)
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
            elif rgrid is None:
                atnum = atnums[i]
                rad = _generate_default_rgrid(atnum)
            else:
                raise TypeError(f"not supported radial grid input; got input type: {type(rgrid)}")
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
                atnum=atnums[i], preset=gd_type, rgrid=rad, center=atcoords[i], rotate=rotate
            )
            atomic_grids.append(at_grid)
        return cls(atnums, atomic_grids, aim_weights, store=store)

    @classmethod
    def from_size(
        cls,
        atnums: np.ndarray,
        atcoords: np.ndarray,
        size: int,
        rgrid: OneDGrid | None = None,
        aim_weights: callable | np.ndarray = None,
        rotate: int = 37,
        store: bool = False,
    ):
        """
        Initialize a MolGrid instance with Horton Style input.

        Parameters
        ----------
        atnums : np.ndarray(M, 3)
            Atomic number of :math:`M` atoms in molecule.
        atcoords : np.ndarray(N, 3)
            Cartesian coordinates for each atoms
        size : int
            Number of points on each shell of angular grid.
        rgrid : OneDGrid or None, optional
            One-dimensional grid to construct the atomic grid. If none, then
            default radial grid is generated based on atomic numbers.
        aim_weights : Callable or np.ndarray(K,), optional
            Atoms in molecule weights. If None, then aim_weights is Becke weights with order=3.
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
        if aim_weights is None:
            aim_weights = BeckeWeights(order=3)
        atgrids = []
        for atnum, atcoord in zip(atnums, atcoords):
            if rgrid is None:
                rad_grid = _generate_default_rgrid(atnum)
            else:
                rad_grid = rgrid
            atgrids.append(
                AtomGrid(rad_grid, degrees=None, sizes=[size], center=atcoord, rotate=rotate)
            )
        return cls(atnums, atgrids, aim_weights, store=store)

    @classmethod
    def from_pruned(
        cls,
        atnums: np.ndarray,
        atcoords: np.ndarray,
        radius: float | list[float],
        r_sectors: float | list[float],
        d_sectors: int | list[list[int]] = 50,
        *,
        s_sectors: int | list[list[int]] | None = None,
        rgrid: OneDGrid | list | None = None,
        aim_weights: callable | np.ndarray | None = None,
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
            Three-dimensional Cartesian coordinates for each atoms in atomic units.
        radius: float, List[float]
            The atomic radius to be multiplied with `r_sectors` in atomic units (to make the
            radial sectors atom specific). If float, then the same atomic radius is used for all
            atoms, otherwise a list with :math:`M` elements is used, where :math:`M` is the number
            of atoms in the molecule. If list, then the ith element is used for the ith atom.
        r_sectors : list of List[float]
            List of sequences of the boundary radius (in atomic units) specifying sectors of
            the pruned radial grid of :math:`M` atoms. For the first atom, the first
            sector is ``(0, radius*r_sectors[0][0])``, then ``(radius*r_sectors[0][0],
            radius*r_sectors[0][1])``, and so on. See AtomGrid.from_pruned for more information.
        d_sectors : int or list of List[int], optional
            List of sequences of the angular degrees for radial sectors of :math:`M` atoms.
            If a number is given, then the same number of degrees is used for all sectors of
            all atoms.
        s_sectors : int or list of List[int] or None, optional, keyword-only
            List of sequences of angular sizes for each radial sector of of :math:`M` atoms.
            If both `d_sectors` and `s_sectors` are given, `s_sectors` is used.
        rgrid : OneDGrid or List[OneDGrid] or Dict[int: OneDGrid], optional
            One dimensional grid for the radial component.  If a list is provided,then ith
            grid correspond to the ith atom.  If dictionary is provided, then the keys are
            correspond to the `atnums[i]` attribute. If None, then using atomic numbers it will
            generate a default radial grid (PowerRTransform of UniformInteger grid).
        aim_weights: Callable or np.ndarray(\sum^M_n N_n,), optional
            Atoms in molecule/nuclear weights :math:`{ {w_n(r_k)}_k^{N_i}}_n^{M}`, where
            :math:`N_i` is the number of points in the ith atomic grid. If None, then aim_weights
            is Becke weights with order=3.
        rotate : bool or int , optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate random matrix.
        store : bool, optional
            Flag to store each original atomic grid information.

        Returns
        -------
        MolGrid:
            Molecular grid class for integration.

        """
        if atcoords.ndim != 2:
            raise ValueError(
                "The dimension of coordinates need to be 2\n" f"got shape: {atcoords.ndim}"
            )
        if atnums.size != atcoords.shape[0]:
            raise ValueError(
                "The number of atoms in atomic numbers does not match with coordinates\n"
                f"atomic numbers: {atnums.shape}, coordinates: {atcoords.shape}"
            )
        if aim_weights is None:
            aim_weights = BeckeWeights(order=3)

        at_grids = []
        natoms = len(atcoords)
        # List of int is created, so that indexing is possible in the for-loop.
        if isinstance(d_sectors, (int, np.integer)):
            d_sectors = [d_sectors] * natoms
        # If s_sectors given d_sectors is set to [None] for all atoms.
        if s_sectors is not None:
            d_sectors = [None] * natoms
        # else s_sectors is set to [None] for all atoms.
        else:
            s_sectors = [None] * natoms

        if len(d_sectors) != len(r_sectors):
            raise ValueError(
                "The number of angular sectors does not match with the number of radial sectors."
                f"Got {len(d_sectors)} angular sectors and {len(r_sectors)} radial sectors."
            )
        if len(s_sectors) != len(r_sectors):
            raise ValueError(
                "The number of angular sectors does not match with the number of radial sectors."
                f"Got {len(s_sectors)} angular sectors and {len(r_sectors)} radial sectors."
            )

        radius_atom = [radius] * natoms if isinstance(radius, (float, np.float64)) else radius
        for i, atnum in enumerate(atnums):
            # get proper radial grid
            if isinstance(rgrid, OneDGrid):
                rad = rgrid
            elif isinstance(rgrid, list):
                rad = rgrid[i]
            elif isinstance(rgrid, dict):
                rad = rgrid[atnum]
            elif rgrid is None:
                rad = _generate_default_rgrid(atnum)
            else:
                raise TypeError(f"Argument rgrid is not supported; got rgrid type: {type(rgrid)}")

            at_grids.append(
                AtomGrid.from_pruned(
                    rad,
                    radius_atom[i],
                    r_sectors=r_sectors[i],
                    d_sectors=d_sectors[i],
                    s_sectors=s_sectors[i],
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


def _generate_default_rgrid(atnum: int):
    r"""
    Generate default radial transformation grid from default Horton.

    See _DEFAULT_POWER_RTRANSFORM_PARAMS inside utils for information on how
    it was determined

    Parameters
    ----------
    atnum: int
        Atomic Number.

    Returns
    -------
    OneDGrid:
        One-dimensional grid that was transformed using PowerRTransform.

    """
    if atnum in _DEFAULT_POWER_RTRANSFORM_PARAMS:
        rmin, rmax, npt = _DEFAULT_POWER_RTRANSFORM_PARAMS[int(atnum)]
        # Convert from Angstrom to atomic units
        rmin = rmin * scipy.constants.angstrom / scipy.constants.value("atomic unit of length")
        rmax = rmax * scipy.constants.angstrom / scipy.constants.value("atomic unit of length")
        onedgrid = UniformInteger(npt)
        rgrid = PowerRTransform(rmin, rmax).transform_1d_grid(onedgrid)
        return rgrid
    else:
        raise ValueError(
            f"Default rgrid parameter is not included for the" f" atomic number {atnum}."
        )
