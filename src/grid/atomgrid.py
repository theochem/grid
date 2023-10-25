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


from grid.basegrid import Grid, OneDGrid
from grid.lebedev import AngularGrid
from grid.utils import convert_cart_to_sph

from importlib_resources import path

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.special import sph_harm


class AtomGrid(Grid):
    """Atomic grid construction class."""

    def __init__(self, rgrid, *, degrees=None, sizes=None, center=None, rotate=False):
        """Construct atomic grid for given arguments.

        Parameters
        ----------
        rgrid : OneDGrid
            The (1-dimensional) radial grid representing the radius of spherical grids.
        degrees : np.ndarray(N, dtype=int) or list, keyword-only argument
            Sequence of Lebedev grid degrees used for constructing spherical grids at each
            radial grid point.
            If only one degree is given, the specified degree is used for all spherical grids.
            If the given degree is not supported, the next largest degree is used.
        sizes : np.ndarray(N, dtype=int) or list, keyword-only argument
            Sequence of Lebedev grid sizes used for constructing spherical grids at each
            radial grid point.
            If only one size is given, the specified size is used for all spherical grids.
            If the given size is not supported, the next largest size is used.
            If both degrees and sizes are given, degrees is used for making the spherical grids.
        center : np.ndarray(3,), optional, keyword-only argument
            Cartesian coordinates of the grid center. If `None`, the origin is used.
        rotate : bool or int , optional
            Whether to rotate the Lebedev spherical grids at each radial grid point.
            If given an integer, it is used as a seed for generating random rotation matrices.

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
        if degrees is None:
            if not isinstance(sizes, (np.ndarray, list)):
                raise TypeError(
                    f"sizes is not type: np.array or list, got {type(sizes)}"
                )
            degrees = AngularGrid.convert_lebedev_sizes_to_degrees(sizes)
        if not isinstance(degrees, (np.ndarray, list)):
            raise TypeError(
                f"degrees is not type: np.array or list, got {type(degrees)}"
            )
        if len(degrees) == 1:
            degrees = np.ones(rgrid.size, dtype=int) * degrees
        self._degs = degrees
        self._points, self._weights, self._indices = self._generate_atomic_grid(
            self._rgrid, self._degs, rotate=self._rot
        )
        self._size = self._weights.size
        self._basis = None

    @classmethod
    def from_preset(cls, rgrid=None, *, atnum, preset, center=None, rotate=False):
        """High level api to construct an atomic grid with preset arguments.

        Examples
        --------
        # construct an atomic grid for H with fine grid setting
        >>> atgrid = AtomGrid.from_preset(rgrid, atnum=1, preset="fine")

        Parameters
        ----------
        rgrid : OneDGrid, optional
            The (1-dimensional) radial grid representing the radius of spherical grids.
        atnum : int, keyword-only argument
            The atomic number specifying the predefined grid.
        preset : str, keyword-only argument
            The name of predefined grid specifying the radial sectors and their corresponding
            number of Lebedev grid points. Supported preset options include:
            'coarse', 'medium', 'fine', 'veryfine', 'ultrafine', and 'insane'.
            Other options include the "standard grids":
            'sg_0', 'sg_1', 'sg_2', and 'sg_3', and the Ochsenfeld grids:
            'g1', 'g2', 'g3', 'g4', 'g5', 'g6', and 'g7', with higher number indicating
            greater accuracy but denser grid. See `Notes` for more information.
        center : np.ndarray(3,), optional, keyword-only argument
            Cartesian coordinates of the grid center. If `None`, the origin is used.
        rotate : bool or int , optional
            Whether to rotate the Lebedev spherical grids at each radial grid point.
            If given an integer, it is used as a seed for generating random rotation matrices.

        Notes
        -----
        - The "standard grids" [1]_ "SG-0" and "SG-1" are designed for large molecules with LDA (GGA) functionals,
          whereas "SG-2" and "SG-3" are designed for Meta-GGA functionals and B95/Minnesota functionals, respectively.
        - The Ochsenfeld pruned grids [2]_ are obtained based on the paper.

        References
        ----------
        .. [1] Y. Shao, et al. Advances in molecular quantum chemistry contained in the Q-Chem 4 program package.
               [Mol. Phys. 113, 184–215 (2015)]
        .. [2] Laqua, H., Kussmann, J., & Ochsenfeld, C. (2018). An improved molecular partitioning scheme for
               numerical quadratures in density functional theory. The Journal of Chemical Physics, 149(20).

        """
        if rgrid is None:
            # TODO: generate a default rgrid, currently raise an error instead
            raise ValueError("A default OneDGrid will be generated")
        center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )
        cls._input_type_check(rgrid, center)
        # load radial points and
        with path("grid.data.prune_grid", f"prune_grid_{preset}.npz") as npz_file:
            data = np.load(npz_file)
            # load predefined_radial sectors and num_of_points in each sectors
            rad = data[f"{atnum}_rad"]
            npt = data[f"{atnum}_npt"]
        if preset in ['sg_0', 'sg_2', 'sg_3', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7']:
            sector_sizes = [npt[idx] for idx in range(len(rad)) for r_p in range(rad[idx])]
            return cls(rgrid, sizes=sector_sizes, center=center, rotate=rotate)
        elif preset == 'sg_1' and atnum > 19:
            sector_sizes = [npt[idx] for idx in range(len(rad)) for r_p in range(rad[idx])]
            return cls(rgrid, sizes=sector_sizes, center=center, rotate=rotate)
        else:
            degs = AngularGrid.convert_lebedev_sizes_to_degrees(npt)
            rad_degs = AtomGrid._find_l_for_rad_list(rgrid.points, rad, degs)
            return cls(rgrid, degrees=rad_degs, center=center, rotate=rotate)

    @classmethod
    def from_pruned(
        cls,
        rgrid,
        radius,
        *_,
        sectors_r,
        sectors_degree=None,
        sectors_size=None,
        center=None,
        rotate=False,
    ):
        """Initialize an instance for given r_sectors of radius and degrees.

        Examples
        --------
        >>> sectors_r = [0.5, 1., 1.5]
        >>> sectors_degree = [3, 7, 5, 3]
        rad is the radius of atom
        # 0 <= r < 0.5rad, angular grid with degree 3
        # 0.5rad <= r < rad, angular grid with degree 7
        # rad <= r < 1.5rad, angular grid with degree 5
        # 1.5rad <= r, angular grid with degree 3
        >>> atgrid = AtomGrid.from_pruned(rgrid, radius, sectors_r, sectors_degree)

        Parameters
        ----------
        rgrid : OneDGrid
            The (1-dimensional) radial grid representing the radius of spherical grids.
        radius : float
            The atomic radius to be multiplied with `r_sectors` (to make them atom specific).
        sectors_r : np.ndarray(N,), keyword-only argument
            Sequence of boundary points specifying radial sectors of the pruned grid.
            The first sector is ``[0, radius*sectors_r[0]]``, then ``[radius*sectors_r[0],
            radius*sectors_r[1]]``, and so on.
        sectors_degree : np.ndarray(N + 1, dtype=int), keyword-only argument
            Sequence of Lebedev degrees for each radial sector of the pruned grid.
        sectors_size : np.ndarray(N + 1, dtype=int), keyword-only argument
            Sequence of Lebedev sizes for each radial sector of the pruned grid.
            If both sectors_degree and sectors_size are given, sectors_degree is used.
        center : np.ndarray(3,), optional, keyword-only argument
            Cartesian coordinates of the grid center. If `None`, the origin is used.
        rotate : bool or int , optional
            Whether to rotate the Lebedev spherical grids at each radial grid point.
            If given an integer, it is used as a seed for generating random rotation matrices.

        Returns
        -------
        AtomGrid
            Generated AtomGrid instance for this special init method.
        """
        if sectors_degree is None:
            sectors_degree = AngularGrid.convert_lebedev_sizes_to_degrees(sectors_size)
        center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )
        cls._input_type_check(rgrid, center)
        degrees = cls._generate_degree_from_radius(
            rgrid, radius, sectors_r, sectors_degree
        )
        return cls(rgrid, degrees=degrees, center=center, rotate=rotate)

    @property
    def rgrid(self):
        """OneDGrid: The radial grid representing the radius of spherical grids."""
        return self._rgrid

    @property
    def points(self):
        """np.npdarray(N, 3): Cartesian coordinates of the grid points."""
        return self._points + self._center

    @property
    def indices(self):
        """np.ndarray(M+1,): Indices saved for each spherical shell."""
        # M is the number of points on radial grid.
        return self._indices

    @property
    def center(self):
        """np.ndarray(3,): Cartesian coordinates of the grid center."""
        return self._center

    @property
    def n_shells(self):
        """int: Number of shells in radial points."""
        return len(self._degs)

    @property
    def l_max(self):
        """int: Largest angular degree L value in angular grids."""
        return np.max(self._degs)

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
            [radius, azumuthal, polar]
        """
        if points is None:
            points = self.points
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        center = self.center if center is None else np.asarray(center)
        return convert_cart_to_sph(points, center)

    def fit(self, values):
        """Fit given value arrays into splines that matches atomic grid.

        Parameters
        ----------
        values : np.ndarray(N,)
            a 1d-array evaluated at each atomic grid point

        Returns
        -------
        list[scipy.PPoly]
            A list of cubic spline fitted by given value arrays
        """
        if values.size != self.size:
            raise ValueError(
                "The size of values does not match with the size of grid\n"
                f"The size of value array: {values.size}\n"
                f"The size of grid: {self.size}"
            )
        if self._basis is None:
            theta, phi = self.convert_cart_to_sph().T[1:]
            self._basis = self._generate_real_sph_harm(self.l_max // 2, theta, phi)
        prod_value = self._basis * values * self.weights
        rad_values = [
            np.sum(prod_value[:, self.indices[i] : self.indices[i + 1]], axis=-1)
            for i in range(self.n_shells)
        ]
        # rad_values in shape (n_shell, n_sph_harms)

        ml_sph_values = np.array(rad_values).T  # shape(n_sph_harms, n_shell)

        # each shell can only integrate upto shell_degree // 2, so if shell_degree < l_max,
        # the f_{lm} should be set to zero for l > shell_degree // 2. Instead, one could set
        # truncate the basis of a given shell.
        for i in range(self.n_shells):
            if self._degs[i] != self.l_max:
                num_nonzero_sph = (self._degs[i] // 2 + 1) ** 2
                ml_sph_values[num_nonzero_sph:, i] = 0.0

        # compute radial value at each radial point
        # the total value of each shell is sum(r^2 * r_weight * sph_weight * points_value)
        # where sum(sph_weight) = 4 * Pi
        ml_sph_values /= self.rgrid.points ** 2 * self.rgrid.weights
        return [
            CubicSpline(x=self.rgrid.points, y=sph_val) for sph_val in ml_sph_values
        ]

    def interpolate(self, values):
        """Interpolate values at given points on the splines fitted by provided value_array.

        Parameters
        ----------
        values : np.ndarray(M,)
            a 1d-array evaluabled at each atomic grid point

        Returns
        -------
        Callable[[np.ndarray(N, 3), int] -> np.ndarray(N)]
            a callable function can be evalabled given 3D points
        """
        # compute splines for given value_array on grid points
        splines = self.fit(values)

        def interpolate_low(points, deriv=0):
            """
            Parameters
            ----------
            points : ndarray(N, 3)
                Cartesian coordinates of :math:`N` points to evaluate the splines on.
            deriv : int, optional
                order of derivative to be evaluated on the spline, by default 0
                0 : the spline value
                1 : the 1st order deriv of the spline
                2 : the 2nd order deriv of the spline
                3 : the 3nd order deriv of the spline, warning: the value is a constant
            """
            r_pts, theta, phi = self.convert_cart_to_sph(points).T
            r_values = np.array([spline(r_pts, deriv) for spline in splines])
            r_sph_harm = self._generate_real_sph_harm(self.l_max // 2, theta, phi)
            return np.einsum("ij, ij -> j", r_values, r_sph_harm)

        return interpolate_low

    @staticmethod
    def _generate_real_sph_harm(l_max, theta, phi):
        """Generate real spherical harmonics.

        Parameters
        ----------
        l_max : int
            largest angular degree
        theta : np.ndarray(N,)
            azimuthal angles
        phi : np.ndarray(N,)
            polar angles

        Returns
        -------
        np.ndarray((l_max + 1)**2, N)
            value of angular grid in each m, n spherical harmonics
        """
        if l_max < 0:
            raise ValueError(f"lmax needs to be >=0, got l_amx={l_max}")
        total_sph = np.zeros((0, len(theta)), dtype=float)
        l_list = np.arange(l_max + 1)
        for l_val in l_list:
            # generate l=0 real spheric
            zero_real_sph = sph_harm(0, l_val, theta, phi).real

            # generate l=positive real spheric
            m_list_p = np.arange(1, l_val + 1, dtype=float)
            pos_real_sph = (
                sph_harm(m_list_p[:, None], l_val, theta, phi).real
                * np.sqrt(2)
                * (-1) ** m_list_p[:, None]
            )
            # generate l=negative real spheric
            m_list_n = np.arange(-l_val, 0, dtype=float)
            neg_real_sph = (
                sph_harm(m_list_n[:, None], l_val, theta, phi).imag
                * np.sqrt(2)
                * (-1) ** m_list_n[:, None]
            )
            total_sph = np.vstack(
                (total_sph, zero_real_sph, pos_real_sph, neg_real_sph)
            )
        return total_sph

    @staticmethod
    def _input_type_check(rgrid, center):
        """Check input type.

        Parameters
        ----------
        rgrid : OneDGrid
            The (1-dimensional) radial grid representing the radius of spherical grids.
        center : np.ndarray(3,), optional
            center of the spherical coordinates
            atomic center will be used if `center` is not given

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
    def _generate_degree_from_radius(rgrid, radius, r_sectors, degrees):
        """Generate proper degrees for radius.

        Parameters
        ----------
        rgrid : RadialGrid
            A radialgrid instance
        radius : float
            radius of interested atom
        r_sectors : list or np.ndarray
            a list of r_sectors number
        degrees : list or np.ndarray
            a list of degs for each radius section

        Returns
        -------
        np.ndarray
            a numpy array of L degree value for each radial point

        """
        r_sectors = np.array(r_sectors)
        degrees = np.array(degrees)
        if len(degrees) == 0:
            raise ValueError("rad_list can't be empty.")
        if len(degrees) - len(r_sectors) != 1:
            raise ValueError("degs should have only one more element than r_sectors.")
        # match given degrees to the supported (i.e., pre-computed) Lebedev degrees
        matched_deg = np.array(
            [AngularGrid._get_lebedev_size_and_degree(degree=d)[0] for d in degrees]
        )
        rad_degs = AtomGrid._find_l_for_rad_list(
            rgrid.points, radius * r_sectors, matched_deg
        )
        return rad_degs

    @staticmethod
    def _find_l_for_rad_list(radial_arrays, radius_list, degrees):
        """Find proper magic L value for radial points at different radius range.

        use broadcast to compare each point with r_sectors then sum over all
        the True value, which should equal to the position of L

        Parameters
        ----------
        radial_arrays : np.ndarray(N,)
            radial grid points
        radius_list : np.ndarray(K,)
            an array of r_sectors * radius
        degrees : np.ndarray(K+1,),
            an array of degs for different r_sectors

        Returns
        -------
        np.ndarray(N,), an array of magic numbers for each radial points
        """
        position = np.sum(radial_arrays[:, None] > radius_list[None, :], axis=1)
        return degrees[position]

    @staticmethod
    def _generate_atomic_grid(rgrid, degrees, rotate=False):
        """Generate atomic grid for each radial point with angular degree L.

        Parameters
        ----------
        rgrid : OneDGrid
            The (1-dimensional) radial grid representing the radius of spherical grids.
        degrees : np.ndarray(N,)
            Sequence of Lebedev grid degrees used for constructing spherical grids at each
            radial grid point.
            If the given degree is not supported, the next largest degree is used.
        rotate : bool or int , optional
            Whether to rotate the Lebedev spherical grids at each radial grid point.
            If given an integer, it is used as a seed for generating random rotation matrices.

        Returns
        -------
        tuple(np.ndarray(M,), np.ndarray(M,), np.ndarray(N,)),
            grid points, grid weights, and indices for each shell.
        """
        if len(degrees) != rgrid.size:
            raise ValueError("The shape of radial grid does not match given degs.")
        all_points, all_weights = [], []

        shell_pt_indices = np.zeros(len(degrees) + 1, dtype=int)  # set index to int
        for i, deg_i in enumerate(degrees):  # TODO: proper tests
            sphere_grid = AngularGrid(degree=deg_i)
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


def get_rgrid_size(preset_grid, atnums=None):
    """ Get the predefined radial points for available pruned grids

    Parameters
    ----------
    preset_grid : str
        String specifiying type of pruned grid to access radial points data.
    atnums : int or list/array of ints
        Atomic numbers for which to retrieve number of radial points.

    """
    if preset_grid not in ['sg_0', 'sg_1', 'sg_2', 'sg_3',
                           'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7']:
        raise ValueError(f"type_pruned {preset_grid} not recognized as a valid pruned grid")
    elif atnums is None:
        raise ValueError(f"At least one atomic number must be specified. Got {atnums}")
    else:
        if isinstance(atnums, int):
            atnums = [atnums]

        radial_pts = []
        with path("grid.data.prune_grid", f"prune_grid_{preset_grid}.npz") as npz_file:
            data = np.load(npz_file)
        for idx, at_num in enumerate(atnums):
            if preset_grid == 'sg_1':
                rad = data['r_points']
                radial_pts.append(sum(rad))
            else:
                rad = data[f"{at_num}_rad"]
                radial_pts.append(sum(rad))
        return radial_pts
