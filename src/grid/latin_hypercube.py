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
"""Latin Hypercube Sampling for integration on (hyper)cubic grids."""

import numpy as np

from grid.basegrid import Grid


class LatinHypercube(Grid):
    r"""Randomized Latin Hypercube Sampling for integration on a (hyper)cubic grid.

    Latin Hypercube Sampling (LHS) stratifies every one-dimensional marginal exactly:
    splitting :math:`[0,1)` into :math:`N` equal strata along *any* single coordinate
    axis places exactly one point in each stratum. For a given dimension, a point's
    coordinate is

    .. math::
        x_i = \frac{\pi(i) - U_i}{N}, \qquad i = 1, \ldots, N,

    where :math:`\pi` is an independent random permutation of :math:`1, \ldots, N` and
    :math:`U_i \sim \text{Uniform}(0,1)` i.i.d., drawn independently for every dimension.
    Setting `randomize=False` replaces :math:`U_i` with the constant :math:`0.5`, placing
    each point at the center of its stratum instead of a random position within it.

    Unlike Monte Carlo sampling, LHS is asymptotically at least as accurate as plain
    Monte Carlo for any integrand, and strictly better for integrands with a strong
    additive component (Stein, 1987).

    The integration weights are all equal to :math:`V/N`, where :math:`V` is the volume
    of the integration domain.

    References
    ----------
    - McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A Comparison of Three
      Methods for Selecting Values of Input Variables in the Analysis of Output from
      a Computer Code. Technometrics, 21(2), 239-245.
    - Stein, M. (1987). Large Sample Properties of Simulations Using Latin Hypercube
      Sampling. Technometrics, 29(2), 143-151.

    """

    def __init__(
        self,
        n_points,
        dimension,
        seed=None,
        randomize=True,
        origin=None,
        axes=None,
    ):
        r"""Construct a Latin Hypercube grid.

        Parameters
        ----------
        n_points : int
            Number of integration points :math:`N`.
        dimension : int
            Dimension :math:`d` of the integration domain.
        seed : int, optional
            Seed for the random number generator, for reproducibility.
        randomize : bool, optional
            If True (default), each point is jittered uniformly within its stratum.
            If False, each point is placed at the center of its stratum instead
            (the assignment of strata to dimensions is still drawn randomly, since
            otherwise every dimension would place its points on the same diagonal
            pattern).
        origin : np.ndarray, shape (d,), optional
            Origin of the hypercube. Defaults to zero vector.
        axes : np.ndarray, shape (d, d), optional
            Axes defining the hypercube (as row vectors). Defaults to identity matrix
            (unit hypercube). The LHS points are first generated on :math:`[0,1)^d` and
            then affine-transformed to the specified parallelepiped.

        Raises
        ------
        ValueError
            If n_points or dimension is not a positive integer, or if origin/axes have
            an incorrect shape, or if axes are not linearly independent.

        """
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")

        # Generate LHS points in the unit cube [0, 1)^d
        points_unit = self._generate_lhs_points(n_points, dimension, seed, randomize)

        if origin is None:
            origin = np.zeros(dimension)
        else:
            origin = np.asarray(origin, dtype=float)
            if origin.shape != (dimension,):
                raise ValueError(f"origin must have shape ({dimension},), got {origin.shape}")

        if axes is None:
            axes = np.eye(dimension)
        else:
            axes = np.asarray(axes, dtype=float)
            if axes.shape != (dimension, dimension):
                raise ValueError(
                    f"axes must have shape ({dimension}, {dimension}), got {axes.shape}"
                )
            if np.linalg.matrix_rank(axes) < dimension:
                raise ValueError("axes must be linearly independent")

        # Map the unit cube points to the parallelepiped defined by origin and axes
        # (affine transformation: x = origin + points_unit @ axes)
        points = origin + points_unit @ axes

        # Volume of the parallelepiped
        volume = np.abs(np.linalg.det(axes))

        # Uniform weights
        weights = np.full(n_points, volume / n_points)

        self._n_points = n_points
        self._dimension = dimension
        self._seed = seed
        self._randomize = randomize
        self._origin = origin
        self._axes = axes

        super().__init__(points, weights)

    def _generate_lhs_points(self, n_points, dimension, seed, randomize):
        r"""Generate Latin Hypercube points on the unit hypercube [0,1)^d.

        Parameters
        ----------
        n_points : int
            Number of points :math:`N`.
        dimension : int
            Dimension :math:`d`.
        seed : int or None
            Seed for the random number generator.
        randomize : bool
            If True, jitter each point uniformly within its stratum. If False,
            place each point at the center of its stratum.

        Returns
        -------
        np.ndarray, shape (N, d)
            Latin Hypercube points in the unit hypercube.

        """
        rng = np.random.default_rng(seed)
        keys = rng.random(size=(dimension, n_points))
        permutations = np.argsort(keys, axis=-1) + 1
        if randomize:
            U = rng.uniform(0, 1, size=permutations.shape)
            result = (permutations - U) / n_points
        else:
            result = (permutations - 0.5) / n_points
        return result.T

    def __getitem__(self, index):
        """Return a plain Grid for a point subset (a subset is not itself a valid LHS design)."""
        if isinstance(index, int):
            return Grid(np.array([self.points[index]]), np.array([self.weights[index]]))
        return Grid(np.array(self.points[index]), np.array(self.weights[index]))

    @property
    def n_points(self):
        """int: Number of points in the LHS design."""
        return self._n_points

    @property
    def dimension(self):
        """int: Dimension of the LHS grid."""
        return self._dimension

    @property
    def seed(self):
        """int or None: Seed used for reproducibility."""
        return self._seed

    @property
    def randomize(self):
        """bool: Whether points are jittered within strata (True) or centered (False)."""
        return self._randomize

    @property
    def origin(self):
        """np.ndarray: Origin of the integration domain."""
        return self._origin

    @property
    def axes(self):
        """np.ndarray: Axes defining the integration domain."""
        return self._axes
