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
"""Module for generating AtomGrid."""


from grid.basegrid import AngularGrid, Grid, OneDGrid
from grid.lebedev import (
    _get_lebedev_size_and_degree,
    convert_lebedev_sizes_to_degrees,
    generate_lebedev_grid,
)

from importlib_resources import path

import numpy as np

from scipy.spatial.transform import Rotation as R


class AtomGrid(Grid):
    """Atomic grid construction class."""

    def __init__(
        self,
        rgrid,
        *,
        degs=None,
        size=None,
        center=None,
        rotate=False,
    ):
        """Construct atomic grid for given arguments.

        Parameters
        ----------
        rgrid : OneDGrid
            A 1D grid with positive domain representing the radial component of grid.
        degs : np.ndarray(N, dtype=int) or list, keyword-only argument
            Different degree value for each radial point
        size : np.ndarray(N, dtype=int) or list, keyword-only argument
            Different number of angular points for each radial point
        center : np.ndarray(3,), optional, keyword-only argument
            if not given, default to (0., 0., 0.)
            Central cartesian coordinates of atomic grid
        rotate : bool or int , optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.

        Raises
        ------
        TypeError
            ``rgrid`` needs to be an instance of ``Grid`` class.
        ValueError
            Length of ``degs`` should be one more than ``r_sectors``.
        """
        # check stage, if center is None, set to (0., 0., 0.)
        center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )
        self._input_type_check(rgrid, center)
        # assign & check stage
        self._center = center
        self._rgrid = rgrid
        # check rotate
        if not isinstance(rotate, (int, np.integer)):
            raise TypeError(
                f"rotate needs to be an bool or integer, got {type(rotate)}"
            )
        if (rotate is not False) and (not 0 <= rotate < 2 ** 32):
            raise ValueError(
                f"rotate need to be an integer [0, 2^32 - 1]\n"
                f"rotate is not within [0, 2^32 - 1], got {rotate}"
            )
        self._rot = rotate
        # check degs and size
        if degs is None:
            if not isinstance(size, (np.ndarray, list)):
                raise TypeError(f"size is not type: np.array or list, got {type(size)}")
            degs = convert_lebedev_sizes_to_degrees(size)
        if not isinstance(degs, (np.ndarray, list)):
            raise TypeError(f"degs is not type: np.array or list, got {type(degs)}")
        if len(degs) == 1:
            degs = np.ones(rgrid.size, dtype=int) * degs
        self._rad_degs = degs
        self._points, self._weights, self._indices = self._generate_atomic_grid(
            self._rgrid, self._rad_degs, rotate=self._rot
        )
        self._size = self._weights.size

    @classmethod
    def from_predefined(
        cls,
        atnum,
        rgrid,
        grid_type,
        *_,
        center=None,
        rotate=False,
    ):
        """High level to construct prefined atomic grid.

        Examples
        --------
        # construct an atomic grid for H with fine grid setting
        >>> atgrid = AtomGrid.from_predefined(1, rgrid, "fine")

        Parameters
        ----------
        atnum : int
            atomic number of atomic grid
        rgrid : RadialGrid
            points where sperical grids will be located
        grid_type : str
            different accuracy for atomic grid
            include: 'coarse', 'medium', 'fine', 'veryfine', 'ultrafine', 'insane'
        center : np.ndarray(3,), optional
            if not given, default (0., 0., 0.) is used
            coordinates of grid center
        rotate : bool or int , optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.
        """
        center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )
        cls._input_type_check(rgrid, center)
        # load radial points and
        with path("grid.data.prune_grid", f"prune_grid_{grid_type}.npz") as npz_file:
            data = np.load(npz_file)
            # load predefined_radial sectors and num_of_points in each sectors
            rad = data[f"{atnum}_rad"]
            npt = data[f"{atnum}_npt"]

        degs = convert_lebedev_sizes_to_degrees(npt)
        rad_degs = AtomGrid._find_l_for_rad_list(rgrid.points, rad, degs)
        return cls(rgrid, degs=rad_degs, center=center, rotate=rotate)

    @classmethod
    def from_pruned(
        cls,
        rgrid,
        radius,
        *_,
        r_sectors,
        degs=None,
        size=None,
        center=None,
        rotate=False,
    ):
        """Initialize an instance for given r_sectors of radius and degrees.

        Examples
        --------
        >>> r_sectors = [0.5, 1., 1.5]
        >>> degs = [3, 7, 5, 3]
        rad is the radius of atom
        # 0 <= r < 0.5rad, angular grid with degree 3
        # 0.5rad <= r < rad, angular grid with degree 7
        # rad <= r < 1.5rad, angular grid with degree 5
        # 1.5rad <= r, angular grid with degree 3
        >>> atgrid = AtomGrid.from_pruned(rgrid, radius, degs, r_sectors, center)

        Parameters
        ----------
        rgrid : RadialGrid
            Radial grid.
        radius : float
            Atomic radius for target atom.
        r_sectors : np.ndarray(N,), keyword-only argument
            r_sectors to define different regions on the radial axis. The first
            region is ``[0, radius*r_sectors[0]]``, then ``[radius*r_sectors[0],
            radius*r_sectors[1]]``, and so on.
        degs : np.ndarray(N + 1, dtype=int), keyword-only argument
            The degree of Lebedev-Laikov grid points for each section of atomic
            radius region.
        size : np.ndarray(N + 1, dtype=int), keyword-only argument
            The degree of Lebedev-Laikov grid points for each section of atomic
            radius region.

            Note: either degs or size is needed to construct an atomic grid

        center : np.ndarray(3, ), keyword-only argument
            if not given, (0., 0., 0.) will be used
            Cartesian coordinates of to origin of the spherical grids.
        rotate : bool or int , optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.

        Returns
        -------
        AtomGrid
            Generated AtomGrid instance for this special init method.
        """
        if degs is None:
            degs = convert_lebedev_sizes_to_degrees(size)
        center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )
        cls._input_type_check(rgrid, center)
        degs = cls._generate_degree_from_radius(rgrid, radius, r_sectors, degs)
        return cls(rgrid, degs=degs, center=center, rotate=rotate)

    @property
    def rgrid(self):
        """RadialGrid: radial points and weights in the atomic grid."""
        return self._rgrid

    @property
    def points(self):
        """np.npdarray(N, 3): cartesian coordinates of points in grid."""
        return self._points + self._center

    @property
    def indices(self):
        """np.ndarray(M+1,): Indices saved for each spherical shell."""
        # M is the number of points on radial grid.
        return self._indices

    @property
    def center(self):
        """np.ndarray(3,): Center of atomic grid."""
        return self._center

    @property
    def n_shells(self):
        """int: return the number of shells in radial points."""
        return len(self._rad_degs)

    @property
    def l_max(self):
        """int: Largest angular degree L value in angular grids."""
        return np.max(self._rad_degs)

    def get_shell_grid(self, index, r_sq=True):
        """Get the spherical integral grid at radial point {index}.

        Parameters
        ----------
        index : int
            index of radial points, start from 0
        r_sq : bool, default True
            the grid weights times r**2, total integral sum to 4 pi r**2
            if False, the total integral sum to 4 pi

        Returns
        -------
        AngularGrid
            AngularGrid at given radial index position.
        """
        ind_start = self._indices[index]
        ind_end = self._indices[index + 1]
        pts = self._points[ind_start:ind_end]
        wts = self._weights[ind_start:ind_end]
        # try not to modify wts incase some weird situation.
        if r_sq is False:
            new_wts = wts / (self._rgrid.points[index] ** 2)
        else:
            new_wts = wts
        return AngularGrid(pts, new_wts)

    def convert_cart_to_sph(self, points=None, center=None):
        """Convert a set of points from cartesian to spherical coordinates.

        Parameters
        ----------
        points : np.ndarray(n, 3), optional
            3 dimentional numpy array for points
            atomic grid points will be used if `points` is not given
        center : np.ndarray(3,), optional
            center of the spherical coordinates
            atomic center will be used if `center` is not given

        Returns
        -------
        np.ndarray(N, 3)
            Spherical coordinates of atoms respect to the center
        """
        if points is None:
            points = self.points
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        center = self.center if center is None else np.asarray(center)
        relat_pts = points - center
        # compute r
        r = np.linalg.norm(relat_pts, axis=-1)
        # polar angle: arccos(z / r)
        phi = np.arccos(relat_pts[:, 2] / r)
        # azimuthal angle arctan2(y / x)
        theta = np.arctan2(relat_pts[:, 1], relat_pts[:, 0])
        return np.vstack([r, theta, phi]).T

    @staticmethod
    def _input_type_check(rgrid, center):
        """Check input type.

        Raises
        ------
        TypeError
            ``rgrid`` needs to be an instance of ``RadialGrid`` class.
        ValueError
            ``center`` needs to be an instance of ``np.ndarray`` class.
        """
        if not isinstance(rgrid, OneDGrid):
            raise TypeError(
                f"Argument rgrid is not an instance of OneDGrid, got {type(rgrid)}."
            )
        if rgrid.domain is not None and rgrid.domain[0] < 0:
            raise TypeError(
                f"Argument rgrid should have a positive domain, got {rgrid.domain}"
            )
        elif np.min(rgrid.points) < 0.0:
            raise TypeError(
                f"Smallest rgrid.points is negative, got {np.min(rgrid.points)}"
            )
        if center.shape != (3,):
            raise ValueError(f"Center should be of shape (3,), got {center.shape}.")

    @staticmethod
    def _generate_degree_from_radius(rgrid, radius, r_sectors, degs):
        """Generate proper degrees for radius.

        Parameters
        ----------
        rgrid : RadialGrid
            A radialgrid instance
        radius : float
            radius of interested atom
        r_sectors : list or np.ndarray
            a list of r_sectors number
        degs : list or np.ndarray
            a list of degs for each radius section

        Returns
        -------
        np.ndarray
            a numpy array of L degree value for each radial point

        Raises
        ------
        ValueError
            Description
        """
        r_sectors = np.array(r_sectors)
        degs = np.array(degs)
        if len(degs) == 0:
            raise ValueError("rad_list can't be empty.")
        if len(degs) - len(r_sectors) != 1:
            raise ValueError("degs should have only one more element than r_sectors.")
        # match given degrees to the supported (i.e., pre-computed) Lebedev degrees
        matched_deg = np.array(
            [_get_lebedev_size_and_degree(degree=d)[0] for d in degs]
        )
        rad_degs = AtomGrid._find_l_for_rad_list(
            rgrid.points, radius * r_sectors, matched_deg
        )
        return rad_degs

    @staticmethod
    def _find_l_for_rad_list(radial_arrays, radius_list, degs):
        """Find proper magic L value for radial points at different radius range.

        use broadcast to compare each point with r_sectors then sum over all
        the True value, which should equal to the position of L

        Parameters
        ----------
        radial_arrays : np.ndarray(N,)
            radial grid points
        radius_list : np.ndarray(K,)
            an array of r_sectors * radius
        degs : np.ndarray(K+1,),
            an array of degs for different r_sectors

        Returns
        -------
        np.ndarray(N,), an array of magic numbers for each radial points
        """
        position = np.sum(radial_arrays[:, None] > radius_list[None, :], axis=1)
        return degs[position]

    @staticmethod
    def _generate_atomic_grid(rgrid, degs, rotate=False):
        """Generate atomic grid for each radial point with angular degree L.

        Parameters
        ----------
        rgrid : Grid, radial grid of given atomic grid.
        degs : np.ndarray(N,), an array of magic number for each radial point.
        rotate : boolean or int, whether a rotation will be applied to grid

        Returns
        -------
        tuple(np.ndarray(M,), np.ndarray(M,), np.ndarray(N,)),
            grid points, grid weights, and indices for each shell.
        """
        if len(degs) != rgrid.size:
            raise ValueError("The shape of radial grid does not match given degs.")
        all_points, all_weights = [], []
        # Preload spherical information in case of repeated IO
        unique_degs = np.unique(degs)
        sphere_grids = {i: generate_lebedev_grid(degree=i) for i in unique_degs}

        shell_pt_indices = np.zeros(len(degs) + 1, dtype=int)  # set index to int
        for i, deg_i in enumerate(degs):  # TODO: proper tests
            sphere_grid = sphere_grids[deg_i]
            if rotate is False:
                pass
            # if rotate is True, rotate each shell
            elif rotate is True:
                rot_mt = R.random().as_matrix()
                new_points = sphere_grid.points @ rot_mt
                sphere_grid = AngularGrid(new_points, sphere_grid.weights)
            # if rotate is a seed
            else:
                assert isinstance(rotate, int)  # check seed proper value
                rot_mt = R.random(random_state=rotate + i).as_matrix()
                new_points = sphere_grid.points @ rot_mt
                sphere_grid = AngularGrid(new_points, sphere_grid.weights)
            # construct atomic grid with each radial point and each spherical shell
            # compute points
            points = sphere_grid.points * rgrid[i].points
            # compute weights
            weights = sphere_grid.weights * rgrid[i].weights * rgrid[i].points ** 2
            # locate separators
            shell_pt_indices[i + 1] = shell_pt_indices[i] + len(points)
            all_points.append(points)
            all_weights.append(weights)
        indices = shell_pt_indices
        points = np.vstack(all_points)
        weights = np.hstack(all_weights)
        return points, weights, indices
