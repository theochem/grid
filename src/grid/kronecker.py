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
r"""Kronecker (Weyl) Sequences for Integration on (Hyper)Cubic Grids."""

import numpy as np
from sympy import sieve

from grid.basegrid import Grid


class Kronecker(Grid):
    r"""Kronecker (Weyl) sequence for integration on a (hyper)cubic grid.

    Kronecker sequences are one of the simplest low-discrepancy constructions:
    given a vector of irrational numbers :math:`\boldsymbol{\alpha}
    \in \mathbb{R}^d`, the sequence is

    .. math::
        \mathbf{x}_i = \{ i\, \boldsymbol{\alpha} \} \quad \text{for } i = 0, \ldots, N-1

    where :math:`\{y\}` denotes the fractional part of :math:`y`, applied
    componentwise. This class uses :math:`\alpha_j = \sqrt{p_j}`, the square
    roots of the first :math:`d` prime numbers, a standard choice: square
    roots of distinct primes are irrational and free of simple rational
    relations to one another, which keeps the components of the sequence
    from lining up in low-dimensional resonances.

    With ``randomize=True``, a single random shift is added to every point
    and reduced modulo 1 (Cranley & Patterson, 1976) -- a common
    randomization technique for low-discrepancy sequences that preserves
    their equidistribution properties while enabling randomized error
    estimates.

    The integration weights are all equal to :math:`V/N`, where :math:`V` is the volume
    of the integration domain.

    References
    ----------
    - Kronecker, L. (1884). Naeherungsweise ganzzahlige Aufloesung linearer
      Gleichungen. Berliner Sitzungsberichte, 1179-1193, 1271-1299.
    - Cranley, R., & Patterson, T. N. L. (1976). Randomization of number
      theoretic methods for multiple integration. SIAM Journal on Numerical
      Analysis, 13(6), 904-914.

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
        r"""Construct a Kronecker grid.

        Parameters
        ----------
        n_points : int
            Number of integration points :math:`N`.
        dimension : int
            Dimension :math:`d` of the integration domain.
        seed : int, optional
            Seed for the random number generator, used only when
            ``randomize=True``.
        randomize : bool, optional
            If True (default), applies a random Cranley-Patterson shift to
            the sequence. If False, generates the deterministic sequence
            :math:`\mathbf{x}_i = \{i\boldsymbol{\alpha}\}`, whose first
            point (:math:`i=0`) is always the origin.
        origin : np.ndarray, shape (d,), optional
            Origin of the hypercube. Defaults to zero vector.
        axes : np.ndarray, shape (d, d), optional
            Axes defining the hypercube (as row vectors). Defaults to identity
            matrix (unit hypercube). The Kronecker points are first generated
            on :math:`[0,1)^d` and then affine-transformed to the specified
            parallelepiped.

        Raises
        ------
        ValueError
            If n_points or dimension is not a positive integer, or if
            origin/axes have an incorrect shape, or if axes are not linearly
            independent.

        """
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")

        # Generate Kronecker points in the unit cube [0, 1)^d
        points_unit = self._generate_kronecker_points(n_points, dimension, seed, randomize)

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

    def _generate_kronecker_points(self, n_points, dimension, seed, randomize):
        r"""Generate Kronecker points on the unit hypercube [0,1)^d.

        Parameters
        ----------
        n_points : int
            Number of points :math:`N`.
        dimension : int
            Dimension :math:`d`.
        seed : int or None
            Seed for the random number generator.
        randomize : bool
            If True, apply a random Cranley-Patterson shift. If False,
            generate the deterministic sequence.

        Returns
        -------
        np.ndarray, shape (N, d)
            Kronecker points in the unit hypercube.

        """
        # sqrt of the first `dimension` primes, via sympy's public sieve API
        # (sieve is 1-indexed: sieve[1] is the first prime, 2).
        sieve.extend_to_no(dimension)
        primes = np.array([sieve[i] for i in range(1, dimension + 1)])
        primes_sqrt = np.sqrt(primes)

        # x_i[j] = {i * sqrt(p_j)}, via broadcasting rather than materializing
        # an explicit (n_points, dimension) tiled array of indices first.
        result = np.mod(np.arange(n_points)[:, None] * primes_sqrt[None, :], 1)

        if randomize:
            rng = np.random.default_rng(seed)
            shift = rng.uniform(0, 1, size=dimension)
            result = np.mod(result + shift, 1)

        return result

    def __getitem__(self, index):
        """Return a plain Grid for a point subset"""
        if isinstance(index, int):
            return Grid(np.array([self.points[index]]), np.array([self.weights[index]]))
        return Grid(np.array(self.points[index]), np.array(self.weights[index]))

    @property
    def n_points(self):
        """int: Number of points in the Kronecker design."""
        return self._n_points

    @property
    def dimension(self):
        """int: Dimension of the Kronecker grid."""
        return self._dimension

    @property
    def seed(self):
        """int or None: Seed used for reproducibility."""
        return self._seed

    @property
    def randomize(self):
        """bool: Whether a random Cranley-Patterson shift is applied (True) or not (False)."""
        return self._randomize

    @property
    def origin(self):
        """np.ndarray: Origin of the integration domain."""
        return self._origin

    @property
    def axes(self):
        """np.ndarray: Axes defining the integration domain."""
        return self._axes
