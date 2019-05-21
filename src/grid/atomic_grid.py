"""Module for generating Atomic Grid."""
from grid.basegrid import Grid
from grid.lebedev import generate_lebedev_grid, match_degree

import numpy as np


class AtomicGrid(Grid):
    """Atomic grid construction class."""

    # construct AtomicGrid
    # coor_to_poar: np.stack(
    #     [np.arctan2(X[:, 1], X[:, 0]), np.arccos(X[:, 2], np.linalg.norm(X, axis=1))],
    #     axis=1,
    # )

    def __init__(
        self, radial_grid, atomic_rad, *, scales, degs, center=np.array([0.0, 0.0, 0.0])
    ):
        """Construct atomic grid for given arguments.

        Parameters
        ----------
        radial_grid : Grid
            Radial grid.
        atomic_rad : float
            Atomic radius for target atom.
        scales : np.ndarray(N,), keyword-only argument
            Scales to define different regions on the radial axis. The first
            region is ``[0, atomic_rad*scale[0]]``, then ``[atomic_rad*scale[0],
            atomic_rad*scale[1]]``, and so on.
        degs : np.ndarray(N + 1, dtype=int), keyword-only argument
            The number of Lebedev-Laikov grid points for each section of atomic
            radius region.
        center : np.ndarray(3, ), default to [0., 0., 0.], keyword-only argument
            Cartesian coordinates of to origin of the spherical grids.

        Raises
        ------
        TypeError
            ``radial_grid`` needs to be an instance of ``Grid`` class.
        ValueError
            Length of ``degs`` should be one more than ``scales``.
        """
        scales = np.array(scales)
        degs = np.array(degs)
        # check stage
        if not isinstance(radial_grid, Grid):
            raise TypeError(
                f"Radial_grid is not an instance of Grid, got {type(radial_grid)}."
            )
        if len(degs) == 0:
            raise ValueError("rad_list can't be empty.")
        if len(degs) - len(scales) != 1:
            raise ValueError("degs should have only one more element than scales.")
        if not isinstance(center, np.ndarray):
            raise TypeError(
                f"Center should be a numpy array with 3 entries, got {type(center)}."
            )
        if len(center) != 3:
            raise ValueError(f"Center should only have 3 entries, got {len(center)}.")
        # assign stage
        self._center = center
        self._radial_grid = radial_grid
        # initiate atomic grid property as None
        rad_degs = self._find_l_for_rad_list(
            self._radial_grid.points, atomic_rad, scales, degs
        )
        # set real degree to each rad point
        self._rad_degs = match_degree(rad_degs)
        self._points, self._weights, self._indices = self._generate_atomic_grid(
            self._radial_grid, self._rad_degs, center
        )
        self._size = len(self._weights)

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
    def l_max(self):
        """int: Largest angular degree L value in angular grids."""
        return np.max(self._rad_degs)

    def convert_cart_to_sph(self):
        """Compute spherical coordinates of the grid.

        Returns
        -------
        np.ndarray(N, 3):
            [azimuthal angle(0, 2pi), polar angle(0, pi), radii]
        """
        r = np.linalg.norm(self._points, axis=1)
        # polar angle: arccos(z / r)
        phi = np.arccos(self._points[:, 2] / r)
        # azimuthal angle arctan2(y / x)
        theta = np.arctan2(self._points[:, 1], self._points[:, 0])
        return np.vstack([theta, phi, r]).T

    @staticmethod
    def _find_l_for_rad_list(radial_arrays, atomic_rad, scales, degs):
        """Find proper magic L value for given scales.

        Parameters
        ----------
        radial_arrays : np.ndarray(N,), radial grid points
        atomic_rad : float, atomic radius of desired atom
        scales : np.ndarray(K,), an array of scales
        degs : np.ndarray(K+1,), an array of degs for different scales

        Returns
        -------
        np.ndarray(N,), an array of magic numbers for each radial points
        """
        # scale_list   R_row * a
        radii = scales * atomic_rad
        # use broadcast to compare each point with scale
        # then sum over all the True value, which should equal to the
        # position of L
        position = np.sum(radial_arrays[:, None] > radii[None, :], axis=1)
        return degs[position]

    @staticmethod
    def _preload_unit_sphere_grid(degs):
        """Preload spherical information incase repeated IO.

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
        """Generate spherical grid's points(coords) and weights.

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
    def _generate_atomic_grid(rad_grid, degs, center):
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
