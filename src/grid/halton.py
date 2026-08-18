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
"""Halton low-discrepancy sequence grids."""

import numpy as np
from scipy.stats import qmc

from grid.basegrid import Grid


class Halton(Grid):
    """Generate a multidimensional Halton low-discrepancy sequence."""

    name = "Halton"

    def __init__(
        self,
        n_points: int,
        dimension: int,
        origin=None,
        axes=None,
        scramble=False,
        seed=None,
    ):
        """Generate ``n_points`` points in ``dimension`` dimensions.

        Parameters
        ----------
        n_points : int
            Number of points.
        dimension : int
            Number of dimensions.
        origin : np.ndarray, optional
            Origin of the parallelepiped.
        axes : np.ndarray, optional
            Axes defining the parallelepiped.
        scramble : bool, optional
            Whether to scramble the Halton sequence.
        seed : int or numpy.random.Generator, optional
            Random seed or generator used for scrambling.
        """
        if not isinstance(n_points, (int, np.integer)) or n_points < 1:
            raise ValueError(
                f"Argument n_points must be a positive integer, given {n_points}"
            )

        if not isinstance(dimension, (int, np.integer)) or dimension < 1:
            raise ValueError(
                f"Argument dimension must be a positive integer, given {dimension}"
            )

        self._n_points = int(n_points)
        self._dimension = int(dimension)

        if origin is None:
            origin = np.zeros(self._dimension)
        else:
            origin = np.asarray(origin, dtype=float)

        if axes is None:
            axes = np.eye(self._dimension)
        else:
            axes = np.asarray(axes, dtype=float)

        if origin.shape != (self._dimension,):
            raise ValueError(
                f"Argument origin should have shape ({self._dimension},), "
                f"given {origin.shape}"
            )

        if axes.shape != (self._dimension, self._dimension):
            raise ValueError(
                f"Argument axes should have shape "
                f"({self._dimension}, {self._dimension}), given {axes.shape}"
            )

        sampler = qmc.Halton(
            d=self._dimension,
            scramble=scramble,
            seed=seed,
        )
        points = sampler.random(n=self._n_points)

        # Map unit-cube points onto the parallelepiped.
        points = origin + points @ axes.T

        weights = np.full(self._n_points, 1.0 / self._n_points)

        super().__init__(points, weights)

    @property
    def n_points(self):
        """int: Number of points in the Halton design."""
        return self._n_points

    @property
    def dimension(self):
        """int: Dimension of the Halton grid."""
        return self._dimension

    def __getitem__(self, index):
        """Return a selected subset of the Halton grid."""
        if isinstance(index, (int, np.integer)):
            points = np.array([self.points[index]])
            weights = np.array([self.weights[index]])
        else:
            points = np.array(self.points[index])
            weights = np.array(self.weights[index])

        new_grid = object.__new__(Halton)
        new_grid._n_points = len(points)
        new_grid._dimension = self._dimension
        Grid.__init__(new_grid, points, weights)
        return new_grid