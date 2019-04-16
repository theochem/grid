"""Module for generating Atomic Grid."""
from grid.grid import AtomicGrid, Grid
from grid.lebedev import generate_lebedev_grid

import numpy as np


class AtomicGridFactory:
    """Atomic grid construction class."""

    def __init__(
        self, radial_grid, atomic_rad, *, scales, degs, center=np.array([0.0, 0.0, 0.0])
    ):
        """Construct atomic grid for given arguments.

        Parameters
        ----------
        center : np.ndarray(3,)
            Central cartesian coordinates of atomic grid
        radial_grid : Grid
            Radial grid for each unit spherical shell
        atomic_rad : float
            Atomic radium for targit atom
        scales : np.ndarray(N,)
            Scales for selecting different spherical grids.
        degs : np.ndarray(N+1, dtype=int)
            Different magic number for each section of atomic radium region

        Raises
        ------
        TypeError
            Radial_grid need to ba an instance of Grid class
        ValueError
            Length of degs should be one more than scales
        """
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
        self._center = center
        self._radial_grid = radial_grid
        # initiate atomic grid property as None
        rad_degs = self._find_l_for_rad_list(
            radial_grid.points, atomic_rad, scales, degs
        )
        self._atomic_grid, self._indices = self._generate_atomic_grid(
            radial_grid, rad_degs, center
        )

    @property
    def atomic_grid(self):
        """AtomicGrid: the generate atomic grid for input atoms."""
        return self._atomic_grid

    @property
    def indices(self):
        """np.ndarray(M+1,): Indices saved for each spherical shell."""
        # M is the number of points on radial grid.
        return self._indices

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
    def _generate_sphere_grid(one_pt_grid, angle_grid):
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
            angle_grid.weights * one_pt_grid.weights,
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
        sphere_grids = AtomicGridFactory._preload_unit_sphere_grid(degs)
        index_array = np.zeros(len(degs) + 1, dtype=int)  # set index to int
        for i, j in enumerate(degs):
            points, weights = AtomicGridFactory._generate_sphere_grid(
                rad_grid[i], sphere_grids[j]
            )
            index_array[i + 1] = index_array[i] + len(points)
            all_points.append(points)
            all_weights.append(weights)
        indices = index_array
        atomic_grid = AtomicGrid(
            np.vstack(all_points) + center, np.hstack(all_weights), center
        )
        return atomic_grid, indices
