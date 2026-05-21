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
r"""Lattice Rules for Integration on (Hyper)Cubic Grids."""

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from grid.basegrid import Grid


# Tabulated generating vectors from UNSW (https://web.maths.unsw.edu.au/~fkuo/lattice/)
# Truncated to first 100 dimensions for practical use
# Format: dimension -> component value for N=2^20 (1048576)
# Source: lattice-32001-1024-1048576.3600 (Order 2 weights, recommended)
_GENERATING_VECTOR_ORDER2 = {
    1: 1,
    2: 182667,
    3: 469891,
    4: 498753,
    5: 110745,
    6: 446247,
    7: 250185,
    8: 118627,
    9: 245333,
    10: 283199,
    11: 408519,
    12: 391023,
    13: 246327,
    14: 126539,
    15: 399185,
    16: 461527,
    17: 300343,
    18: 69681,
    19: 516695,
    20: 436179,
    21: 106383,
    22: 238523,
    23: 413283,
    24: 70841,
    25: 47719,
    26: 300129,
    27: 113029,
    28: 123925,
    29: 410745,
    30: 211325,
    31: 17489,
    32: 511893,
    33: 40767,
    34: 186077,
    35: 519471,
    36: 255369,
    37: 101819,
    38: 243573,
    39: 66189,
    40: 152143,
    41: 503455,
    42: 113217,
    43: 132603,
    44: 463967,
    45: 297717,
    46: 157383,
    47: 224015,
    48: 502917,
    49: 36237,
    50: 94049,
}


class Lattice(Grid):
    r"""Lattice rule for integration on a (hyper)cubic grid.

    Lattice rules provide efficient equal-weight integration points for
    multidimensional integration over the unit hypercube :math:`[0,1)^d`.
    Points are generated using a generating vector :math:`\mathbf{z}`:

    .. math::
        \mathbf{x}_i = \left\{ \frac{i \mathbf{z}}{N} \right\} \quad \text{for } i = 0, \ldots, N-1

    where :math:`\{x\}` denotes the fractional part of :math:`x`.

    The integration weights are all equal to :math:`V/N`, where :math:`V` is the volume
    of the integration domain.

    References
    ----------
    - Sloan, I. H., & Joe, S. (1994). Lattice Methods for Multiple Integration.
    - Kuo, F. Y., & Nuyens, D. (2016). Application of Quasi-Monte Carlo Methods to Elliptic PDEs
      with Random Diffusion Coefficients: A Survey of Analysis and Implementation.
    - UNSW Lattice Rules: https://web.maths.unsw.edu.au/~fkuo/lattice/

    """

    def __init__(
        self,
        n_points,
        dimension,
        generating_vector=None,
        rule="order2",
        origin=None,
        axes=None,
    ):
        r"""Construct a Lattice grid.

        Parameters
        ----------
        n_points : int
            Number of integration points :math:`N`. Should be a power of 2 for embedded
            lattice rules (e.g., 1024, 2048, 4096, ..., up to 1048576).
        dimension : int
            Dimension :math:`d` of the integration domain.
        generating_vector : np.ndarray, shape (d,), optional
            Custom generating vector :math:`\mathbf{z}`. If None, uses tabulated
            vectors based on the `rule` parameter.
        rule : str, optional
            Which tabulated rule to use. Options are:
            - "order2": Order-2 weights (recommended, default)
            Currently only "order2" is implemented with a truncated table.
        origin : np.ndarray, shape (d,), optional
            Origin of the hypercube. Defaults to zero vector.
        axes : np.ndarray, shape (d, d), optional
            Axes defining the hypercube (as row vectors). Defaults to identity matrix
            (unit hypercube). The lattice points are first generated on [0,1)^d and
            then affine-transformed to the specified parallelepiped.

        Raises
        ------
        ValueError
            If n_points is not a power of 2, or if dimension exceeds available
            tabulated vectors, or if rule is not recognized.

        """
        if not self._is_power_of_2(n_points):
            raise ValueError(f"n_points must be a power of 2, got {n_points}")

        if n_points > 1048576:
            raise ValueError(
                f"n_points must be <= 1048576 (2^20) for tabulated rules, got {n_points}"
            )

        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")

        if generating_vector is None:
            if rule == "order2":
                if dimension > len(_GENERATING_VECTOR_ORDER2):
                    raise ValueError(
                        f"Tabulated order2 rule only supports up to "
                        f"{len(_GENERATING_VECTOR_ORDER2)} dimensions, got {dimension}"
                    )
                # Extract generating vector for the requested dimension
                generating_vector = np.array(
                    [_GENERATING_VECTOR_ORDER2[d] for d in range(1, dimension + 1)],
                    dtype=np.int64,
                )
            else:
                raise ValueError(f"Unknown rule: {rule}. Only 'order2' is supported.")
        else:
            generating_vector = np.asarray(generating_vector, dtype=np.int64)
            if generating_vector.shape != (dimension,):
                raise ValueError(
                    f"generating_vector must have shape ({dimension},), "
                    f"got {generating_vector.shape}"
                )

        # Set defaults for origin and axes
        if origin is None:
            origin = np.zeros(dimension)
        else:
            origin = np.asarray(origin, dtype=float)
            if origin.shape != (dimension,):
                raise ValueError(
                    f"origin must have shape ({dimension},), got {origin.shape}"
                )

        if axes is None:
            axes = np.eye(dimension)
        else:
            axes = np.asarray(axes, dtype=float)
            if axes.shape != (dimension, dimension):
                raise ValueError(
                    f"axes must have shape ({dimension}, {dimension}), got {axes.shape}"
                )
            if np.abs(np.linalg.det(axes)) < 1e-10:
                raise ValueError(
                    f"axes must be linearly independent, got det(axes)={np.linalg.det(axes)}"
                )

        self._n_points = n_points
        self._dimension = dimension
        self._generating_vector = generating_vector
        self._origin = origin
        self._axes = axes
        self._rule = rule

        # Generate lattice points on [0,1)^d
        points_unit = self._generate_lattice_points()

        # Transform to the specified parallelepiped: x = origin + points_unit @ axes
        points = origin + points_unit @ axes

        # All weights are equal: V / N
        volume = np.abs(np.linalg.det(axes))
        weights = np.full(n_points, volume / n_points)

        super().__init__(points, weights)

    @staticmethod
    def _is_power_of_2(n):
        """Check if n is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def _generate_lattice_points(self):
        r"""Generate lattice points on the unit hypercube [0,1)^d.

        Returns
        -------
        np.ndarray, shape (N, d)
            Lattice points in the unit hypercube.

        """
        # x_i = {i * z / N} for i = 0, ..., N-1
        # where {x} is the fractional part
        indices = np.arange(self._n_points, dtype=np.int64)
        # Broadcasting: (N, 1) * (1, d) -> (N, d)
        points_scaled = indices[:, None] * self._generating_vector[None, :]
        # Apply modulo N and divide by N to get fractional parts in [0, 1)
        points_unit = (points_scaled % self._n_points) / self._n_points
        return points_unit

    @property
    def dimension(self):
        """int: Dimension of the lattice grid."""
        return self._dimension

    @property
    def generating_vector(self):
        """np.ndarray: Generating vector used for the lattice rule."""
        return self._generating_vector

    @property
    def origin(self):
        """np.ndarray: Origin of the integration domain."""
        return self._origin

    @property
    def axes(self):
        """np.ndarray: Axes defining the integration domain."""
        return self._axes

    @property
    def rule(self):
        """str: Name of the lattice rule used."""
        return self._rule

    def interpolate(self, new_points, values):
        r"""Interpolate function values at new points using linear interpolation.

        Note: Lattice points are not structured as a tensor grid, so we use
        scipy's LinearNDInterpolator which constructs a Delaunay triangulation.
        This may be slow for high dimensions or large N.

        Parameters
        ----------
        new_points : np.ndarray, shape (M, d)
            Points at which to interpolate.
        values : np.ndarray, shape (N,)
            Function values at the lattice points.

        Returns
        -------
        np.ndarray, shape (M,)
            Interpolated values at new_points.

        """
        if values.shape[0] != self.size:
            raise ValueError(
                f"values must have length {self.size}, got {values.shape[0]}"
            )
        new_points = np.asarray(new_points)
        if new_points.ndim == 1:
            new_points = new_points.reshape(1, -1)
        if new_points.shape[1] != self._dimension:
            raise ValueError(
                f"new_points must have {self._dimension} columns, "
                f"got {new_points.shape[1]}"
            )

        # Use LinearNDInterpolator from scipy
        interpolator = LinearNDInterpolator(self.points, values)
        return interpolator(new_points)
