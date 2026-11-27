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
r"""Sobol' Sequence for Integration on (Hyper)Cubic Grids."""

import numpy as np
from scipy.stats import qmc

from grid.basegrid import Grid


class Sobol(Grid):
    r"""Sobol' sequence for integration on a (hyper)cubic grid.

    Sobol' sequences are a digital-net construction providing low-discrepancy,
    equal-weight integration points for multidimensional integration over the
    unit hypercube :math:`[0,1)^d`. Points are generated using scipy's
    :class:`scipy.stats.qmc.Sobol`, with points generated on :math:`[0,1)^d`
    via ``random_base2`` for the balance properties associated with
    powers-of-two sample sizes.

    The integration weights are all equal to :math:`V/N`, where :math:`V` is the volume
    of the integration domain.

    References
    ----------
    - Sobol', I. M. (1967). On the distribution of points in a cube and the
      approximate evaluation of integrals. USSR Computational Mathematics and
      Mathematical Physics, 7(4), 86-112.
    - Owen, A. B. (2020). On dropping the first Sobol' point.
      arXiv:2008.08051.
    - SciPy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html

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
        r"""Construct a Sobol grid.

        Parameters
        ----------
        n_points : int
            Number of integration points :math:`N`. Must be a power of 2
            (e.g., 1024, 2048, 4096, ...), for the balance properties of the
            underlying digital net construction.
        dimension : int
            Dimension :math:`d` of the integration domain.
        seed : int, optional
            Seed for the random number generator, used only when
            ``randomize=True``.
        randomize : bool, optional
            If True (default), applies Owen scrambling to the sequence. If
            False, generates the unscrambled (deterministic) Sobol' sequence.
        origin : np.ndarray, shape (d,), optional
            Origin of the hypercube. Defaults to zero vector.
        axes : np.ndarray, shape (d, d), optional
            Axes defining the hypercube (as row vectors). Defaults to identity
            matrix (unit hypercube). The Sobol' points are first generated on
            :math:`[0,1)^d` and then affine-transformed to the specified
            parallelepiped.

        Raises
        ------
        ValueError
            If n_points is not a power of 2, or if dimension is less than 1,
            or if origin/axes have an incorrect shape, or if axes are not
            linearly independent.

        """
        if not self._is_power_of_2(n_points):
            raise ValueError(f"n_points must be a power of 2, got {n_points}")
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")

        # Generate Sobol points in the unit cube [0, 1)^d
        points_unit = self._generate_sobol_points(n_points, dimension, seed, randomize)

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

    def _generate_sobol_points(self, n_points, dimension, seed, randomize):
        r"""Generate Sobol' points on the unit hypercube [0,1)^d.

        Parameters
        ----------
        n_points : int
            Number of points :math:`N` (a power of 2).
        dimension : int
            Dimension :math:`d`.
        seed : int or None
            Seed for the random number generator.
        randomize : bool
            If True, apply Owen scrambling. If False, generate the
            unscrambled sequence.

        Returns
        -------
        np.ndarray, shape (N, d)
            Sobol' points in the unit hypercube.

        """
        rng = np.random.default_rng(seed)
        m = int(np.log2(n_points))
        return qmc.Sobol(d=dimension, scramble=randomize, rng=rng).random_base2(m=m)

    @staticmethod
    def _is_power_of_2(n):
        """Check if n is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def __getitem__(self, index):
        """Return a plain Grid for a point subset."""
        if isinstance(index, int):
            return Grid(np.array([self.points[index]]), np.array([self.weights[index]]))
        return Grid(np.array(self.points[index]), np.array(self.weights[index]))

    @property
    def n_points(self):
        """int: Number of points in the Sobol design."""
        return self._n_points

    @property
    def dimension(self):
        """int: Dimension of the Sobol grid."""
        return self._dimension

    @property
    def seed(self):
        """int or None: Seed used for reproducibility."""
        return self._seed

    @property
    def randomize(self):
        """bool: Whether Owen scrambling is applied (True) or not (False)."""
        return self._randomize

    @property
    def origin(self):
        """np.ndarray: Origin of the integration domain."""
        return self._origin

    @property
    def axes(self):
        """np.ndarray: Axes defining the integration domain."""
        return self._axes
