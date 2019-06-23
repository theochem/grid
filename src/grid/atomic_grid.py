"""Module for generating Atomic Grid."""
from grid.basegrid import AngularGrid, Grid, RadialGrid
from grid.lebedev import generate_lebedev_grid, match_degree, size_to_degree

import numpy as np

from scipy.spatial.transform import Rotation as R


class AtomicGrid(Grid):
    """Atomic grid construction class."""

    def __init__(
        self,
        radial_grid,
        *,
        degs=None,
        nums=None,
        center=np.array([0.0, 0.0, 0.0]),
        rotate=False,
    ):
        """Construct atomic grid for given arguments.

        Parameters
        ----------
        radial_grid : RadialGrid
            Radial grid
        degs : np.ndarray(N, dtype=int) or list, keyword-only argument
            Different degree value for each radial point
        nums : np.ndarray(N, dtype=int) or list, keyword-only argument
            Different number of angular points for eah radial point
        center : np.ndarray(3,), optional, keyword-only argument
            Central cartesian coordinates of atomic grid
        rotate : bool or int , optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.

        Raises
        ------
        TypeError
            ``radial_grid`` needs to be an instance of ``Grid`` class.
        ValueError
            Length of ``degs`` should be one more than ``scales``.
        """
        # check stage
        self._input_type_check(radial_grid, center)
        # assign stage
        self._center = np.array(center)
        self._radial_grid = radial_grid
        if degs is None:
            if not isinstance(nums, (np.ndarray, list)):
                raise TypeError(f"nums is not type: np.array or list, got {type(nums)}")
            degs = size_to_degree(nums)
        if not isinstance(degs, (np.ndarray, list)):
            raise TypeError(f"degs is not type: np.array or list, got {type(degs)}")
        if len(degs) == 1:
            degs = np.ones(radial_grid.size) * degs
        self._rad_degs = degs
        self._points, self._weights, self._indices = self._generate_atomic_grid(
            self._radial_grid, self._rad_degs
        )
        self._size = self._weights.size
        # add random rotation
        if rotate is not False:
            if rotate is True:
                rot_mt = R.random().as_dcm()
                self._points = np.dot(self._points, rot_mt)
            elif isinstance(rotate, (int, np.integer)) and rotate >= 0:
                rot_mt = R.random(random_state=rotate).as_dcm()
                self._points = np.dot(self._points, rot_mt)
            else:
                raise ValueError(
                    f"rotate need to be an integer [0, 2^32 - 1]\n"
                    f"rotate is not within [0, 2^32 - 1], got {rotate}"
                )

    @classmethod
    def special_init(
        cls, radial_grid, radius, *_, degs, scales, center=np.zeros(3), rotate=False
    ):
        """Initialize an instance for given scales of radius and degrees.

        Examples
        --------
        >>> scales = [0.5, 1., 1.5]
        >>> degs = [3, 7, 5, 3]
        rad is the radius of atom
        # 0 <= r < 0.5rad, angular grid with degree 3
        # 0.5rad <= r < rad, angular grid with degree 7
        # rad <= r < 1.5rad, angular grid with degree 5
        # 1.5rad <= r, angular grid with degree 3
        >>> atgrid = AtomicGrid.special_init(radial_grid, radius, degs, scales, center)

        Parameters
        ----------
        radial_grid : RadialGrid
            Radial grid.
        radius : float
            Atomic radius for target atom.
        scales : np.ndarray(N,), keyword-only argument
            Scales to define different regions on the radial axis. The first
            region is ``[0, radius*scale[0]]``, then ``[radius*scale[0],
            radius*scale[1]]``, and so on.
        degs : np.ndarray(N + 1, dtype=int), keyword-only argument
            The number of Lebedev-Laikov grid points for each section of atomic
            radius region.
        center : np.ndarray(3, ), default to [0., 0., 0.], keyword-only argument
            Cartesian coordinates of to origin of the spherical grids.
        rotate : bool or int , optional
            Flag to set auto rotation for atomic grid, if given int, the number
            will be used as a seed to generate rantom matrix.

        Returns
        -------
        AtomicGrid
            Generated AtomicGrid instance for this special init method.
        """
        cls._input_type_check(radial_grid, center)
        degs = cls._generate_degree_from_radius(radial_grid, radius, scales, degs)
        return cls(radial_grid, degs=degs, center=center, rotate=rotate)

    @property
    def rad_grid(self):
        """RadialGrid: radial points and weights in the atomic grid."""
        return self._radial_grid

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
            if False, the total intergral sum to 4 pi

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
            new_wts = wts / (self._radial_grid.points[index] ** 2)
        else:
            new_wts = wts
        return AngularGrid(pts, new_wts)

    def convert_cart_to_sph(self):
        """Compute spherical coordinates of the grid.

        Returns
        -------
        np.ndarray(N, 3):
            [azimuthal angle(0, 2pi), polar angle(0, pi), radius]
        """
        r = np.linalg.norm(self._points, axis=1)
        # polar angle: arccos(z / r)
        phi = np.arccos(self._points[:, 2] / r)
        # azimuthal angle arctan2(y / x)
        theta = np.arctan2(self._points[:, 1], self._points[:, 0])
        return np.vstack([theta, phi, r]).T

    @staticmethod
    def _input_type_check(radial_grid, center):
        """Check input type.

        Raises
        ------
        TypeError
            ``radial_grid`` needs to be an instance of ``RadialGrid`` class.
        ValueError
            ``center`` needs to be an instance of ``np.ndarray`` class.
        """
        if not isinstance(radial_grid, RadialGrid):
            raise TypeError(
                f"Radial_grid is not an instance of RadialGrid, got {type(radial_grid)}."
            )
        if not isinstance(center, np.ndarray):
            raise TypeError(
                f"Center should be a numpy array with 3 entries, got {type(center)}."
            )
        if len(center) != 3:
            raise ValueError(f"Center should only have 3 entries, got {len(center)}.")

    @staticmethod
    def _generate_degree_from_radius(radial_grid, radius, scales, degs):
        """Generate proper degrees for radius."""
        scales = np.array(scales)
        degs = np.array(degs)
        if len(degs) == 0:
            raise ValueError("rad_list can't be empty.")
        if len(degs) - len(scales) != 1:
            raise ValueError("degs should have only one more element than scales.")
        rad_degs = AtomicGrid._find_l_for_rad_list(
            radial_grid.points, radius, scales, degs
        )
        return match_degree(rad_degs)

    @staticmethod
    def _find_l_for_rad_list(radial_arrays, radius, scales, degs):
        """Find proper magic L value for given scales.

        Parameters
        ----------
        radial_arrays : np.ndarray(N,), radial grid points
        radius : float, atomic radius of desired atom
        scales : np.ndarray(K,), an array of scales
        degs : np.ndarray(K+1,), an array of degs for different scales

        Returns
        -------
        np.ndarray(N,), an array of magic numbers for each radial points
        """
        # scale_list   R_row * a
        radius = scales * radius
        # use broadcast to compare each point with scale
        # then sum over all the True value, which should equal to the
        # position of L
        position = np.sum(radial_arrays[:, None] > radius[None, :], axis=1)
        return degs[position]

    @staticmethod
    def _preload_unit_sphere_grid(degs):
        """Preload spherical information in case of repeated IO.

        Parameters
        ----------
        degs : np.ndarray(N,), an array of preferred magic number degrees.

        Returns
        -------
        dict{degree: AngularGrid}
        """
        # if non-magic number will bring redundancy. But it only link by ref,
        # so the efficiency would be fine.
        unique_degs = np.unique(degs)
        return {i: generate_lebedev_grid(degree=i) for i in unique_degs}

    @staticmethod
    def _generate_sphere_grid(one_pt_grid, angle_grid, rad_order=2):
        """Generate spherical grid's points/coordinates and weights.

        Parameters
        ----------
        one_pt_grid : Grid
        angle_grid : AngularGrid

        Returns
        -------
        tuple(np.ndarra(N,), np.ndarray(N,)), spherical points and its weights.
        """
        return (
            angle_grid.points * one_pt_grid.points,
            angle_grid.weights * one_pt_grid.weights * one_pt_grid.points ** rad_order,
        )

    @staticmethod
    def _generate_atomic_grid(rad_grid, degs):
        """Generate atomic grid for each radial point with given magic L.

        Parameters
        ----------
        rad_grid : Grid, radial grid of given atomic grid.
        degs : np.ndarray(N,), an array of magic number for each radial point.

        Returns
        -------
        tuple(AtomicGrid, np.ndarray(N,)), atomic grid and indices for each shell.
        """
        if len(degs) != rad_grid.size:
            raise ValueError("The shape of radial grid does not match given degs.")
        all_points, all_weights = [], []
        sphere_grids = AtomicGrid._preload_unit_sphere_grid(degs)
        index_array = np.zeros(len(degs) + 1, dtype=int)  # set index to int
        for i, j in enumerate(degs):
            points, weights = AtomicGrid._generate_sphere_grid(
                rad_grid[i], sphere_grids[j]
            )
            index_array[i + 1] = index_array[i] + len(points)
            all_points.append(points)
            all_weights.append(weights)
        indices = index_array
        points = np.vstack(all_points)
        weights = np.hstack(all_weights)
        # atomic_grid = AtomicGrid(
        #     np.vstack(all_points) + center, np.hstack(all_weights), center
        # )
        return points, weights, indices
