import numpy as np

from grid.lebedev import generate_lebedev_grid
from grid.grid import AtomicGrid


class AtomicGridFactory:
    def __init__(self, radial_grid, atomic_rad, scales, degs):
        if len(degs) == 0:
            raise ValueError("rad_list can't be empty.")
        if len(degs) - len(scales) != 1:
            raise ValueError("degs should have only one more element than scales.")
        self._radial_grid = radial_grid
        self._indices = None
        self._radial_grid = radial_grid
        degs = self._find_l_for_rad_list(radial_grid.points, atomic_rad, scales, degs)

    @staticmethod
    def _find_l_for_rad_list(radial_arrays, atomic_rad, scales, degs):
        # scale_list   R_row * a
        radii = scales * atomic_rad
        # use broadcast to compare each point with scale
        # then sum over all the True value, which should equal to the
        # position of L
        position = np.sum(radial_arrays[:, None] > radii[None, :], axis=1)
        return degs[position]

    @staticmethod
    def _preload_unit_sphere_grid(degs):
        # if non-magic number will lower efficiency.
        unique_degs = np.unique(degs)
        return {i: generate_lebedev_grid(degree=i) for i in unique_degs}

    @staticmethod
    def _generate_sphere_grid(one_pt_grid, angle_grid):
        return (
            angle_grid.points * one_pt_grid.points,
            angle_grid.weights * one_pt_grid.weights,
        )

    def _generate_atomic_grid(self, rad_grid, degs):
        if len(degs) != rad_grid.size:
            raise ValueError("The shape of radial grid does not match given degs.")
        all_points, all_weights = [], []
        sphere_grids = self._preload_unit_sphere_grid(degs)
        index_array = np.zeros(len(degs) + 1)
        for i, j in enumerate(degs):
            points, weights = self._generate_sphere_grid(rad_grid[i], sphere_grids[j])
            index_array[i + 1] = index_array + len(points)
            all_points.append(points)
            all_weights.append(weights)
        self._indices = index_array
        self._atomic_grid = AtomicGrid(np.vstack(all_points), np.hstack(all_weights))
