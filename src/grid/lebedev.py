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
"""
Lebedev angular grid module for constructing integration grids on the unit sphere.

The Lebedev grid points here were obtained and calculated from the theochem/HORTON package.
Their calculations were based on F77 translation by Dr. Christoph van Wuellen
and these are the comments from that translation.

This subroutine is part of a set of subroutines that generate
Lebedev grids [1-6] for integration on a sphere. The original
C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
translated into fortran by Dr. Christoph van Wuellen.
This subroutine was translated from C to fortran77 by hand.
Users of this code are asked to include reference [1] in their
publications, and in the user- and programmers-manuals
describing their codes.
This code was distributed through CCL (http://www.ccl.net/).

References
----------
.. [1] V.I. Lebedev, and D.N. Laikov
   "A quadrature formula for the sphere of the 131st algebraic order of accuracy"
   Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
.. [2] V.I. Lebedev "A quadrature formula for the sphere of 59th algebraic order of accuracy"
   Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286.
.. [3] V.I. Lebedev, and A.L. Skorokhodov "Quadrature formulas of orders 41, 47, and 53 for
   the sphere" Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592.
.. [4] V.I. Lebedev "Spherical quadrature formulas exact to orders 25-29"
   Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
.. [5] V.I. Lebedev "Quadratures on a sphere" Computational Mathematics and Mathematical
   Physics, Vol. 16, 1976, pp. 10-24.
.. [6] V.I. Lebedev "Values of the nodes and weights of ninth to seventeenth order Gauss-Markov
   quadrature formulae invariant under the octahedron group with inversion" Computational
   Mathematics and Mathematical Physics, Vol. 15, 1975, pp. 44-51.

"""

import warnings
from bisect import bisect_left

from grid.basegrid import Grid

from importlib_resources import path

import numpy as np

# Lebedev dictionary for converting number of grid points (keys) to grid's degrees (values)
LEBEDEV_NPOINTS = {
    6: 3,
    14: 5,
    26: 7,
    38: 9,
    50: 11,
    74: 13,
    86: 15,
    110: 17,
    146: 19,
    170: 21,
    194: 23,
    230: 25,
    266: 27,
    302: 29,
    350: 31,
    434: 35,
    590: 41,
    770: 47,
    974: 53,
    1202: 59,
    1454: 65,
    1730: 71,
    2030: 77,
    2354: 83,
    2702: 89,
    3074: 95,
    3470: 101,
    3890: 107,
    4334: 113,
    4802: 119,
    5294: 125,
    5810: 131,
}

# Lebedev dictionary for converting grid's degrees (keys) to number of grid points (values)
LEBEDEV_DEGREES = dict([(v, k) for k, v in LEBEDEV_NPOINTS.items()])

LEBEDEV_CACHE = {}


class AngularGrid(Grid):
    """Unit spehrical integration grid."""

    def __init__(
        self, points=None, weights=None, *, degree=None, size=None, cache=True
    ):
        r"""Generate a Lebedev angular grid for the given degree and/or size.

        Use points with weights, or degree, or size to generate Angular Grid

        Parameters
        ----------
        points : np.ndarray(N, N, N)
            The Cartesian coordinates of integral grid points
        weights : np.ndarray(N,)
            The weights of each point on the intergral quadrature grid
        degree : int, optional
            Degree of Lebedev grid. If the Lebedev grid corresponding to the given degree is not
            supported, the next largest degree is used.
        size : int, optional
            Number of Lebedev grid points. If the Lebedev grid corresponding to the given size is
            not supported, the next largest size is used. If both degree and size are given,
            degree is used for constructing the grid.
        cache : bool, optional
            Enable to store loaded lebedeve grids in cache to avoid duplicate
            file reading process. Default to `True`

        Returns
        -------
        AngularGrid
            An angular grid with Lebedev points and weights (including :math:`4\pi`) on
            a unit sphere.

        Examples
        --------
        >>> # Initialize with degree
        >>> AngularGrid(degree=3)
        >>>
        >>> # Initialize with size
        >>> AngularGrid(size=6)
        >>>
        >>> # Initialize with points and weights
        >>> pts = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
        >>> wts = np.array([0.5, 0.4, 0.3])
        >>> AngularGrid(pts, wts)
        """
        # construct grid from pts and wts given directly
        if points is not None and weights is not None:
            super().__init__(points, weights)
            if degree or size:
                warnings.warn(
                    "degree or size are not used for generating grids "
                    "because points and weights are provided",
                    RuntimeWarning,
                )
            return
        # map degree and size to the supported (i.e., pre-computed) degree and size
        degree, size = self._get_lebedev_size_and_degree(degree=degree, size=size)
        # load pre-computed Lebedev points & weights and make angular grid
        if degree not in LEBEDEV_CACHE:
            points, weights = self._load_lebedev_grid(degree, size)
            if cache:
                LEBEDEV_CACHE[degree] = points, weights
        else:
            points, weights = LEBEDEV_CACHE[degree]
        super().__init__(points, weights * 4 * np.pi)

    @staticmethod
    def convert_lebedev_sizes_to_degrees(sizes):
        """Convert given Lebedev grid sizes to degrees.

        Parameters
        ----------
        sizes : array_like
            Sequence of Lebedev grid sizes (e.g., number of points for each atomic shell).

        Returns
        -------
        array_like
            Sequence of the corresponding Lebedev grid degrees.

        """
        degrees = np.zeros(len(sizes), dtype=int)
        for size in np.unique(sizes):
            # get the degree corresponding to the given (unique) size
            deg = AngularGrid._get_lebedev_size_and_degree(size=size)[0]
            # set value of degree to corresponding to the given size equal to deg
            degrees[np.where(sizes == size)] = deg
        return degrees

    @staticmethod
    def _get_lebedev_size_and_degree(*, degree=None, size=None):
        """Map the given degree and/or size to the degree and size of a supported Lebedev grid.

        Parameters
        ----------
        degree : int, optional
            Degree of Lebedev grid. If the Lebedev grid corresponding to the given degree is not
            supported, the next largest degree is used.
        size : int, optional
            Number of Lebedev grid points. If the Lebedev grid corresponding to the given size is
            not supported, the next largest size is used. If both degree and size are given,
            degree is used for constructing the grid.

        Returns
        -------
        int, int
            Degree and size of a supported Lebedev grid (equal to or larger than the
            requested grid).

        """
        if degree and size:
            warnings.warn(
                "Both degree and size arguments are given, so only degree is used!",
                RuntimeWarning,
            )
        if degree:
            leb_degs = list(LEBEDEV_DEGREES.keys())
            max_degree = max(leb_degs)
            if degree < 0 or degree > max_degree:
                raise ValueError(
                    f"Argument degree should be a positive integer <= {max_degree}, got {degree}"
                )
            # match the given degree to the existing Lebedev degree or the next largest degree
            degree = (
                degree
                if degree in LEBEDEV_DEGREES
                else leb_degs[bisect_left(leb_degs, degree)]
            )
            return degree, LEBEDEV_DEGREES[degree]
        elif size:
            leb_npts = list(LEBEDEV_NPOINTS.keys())
            max_size = max(leb_npts)
            if size < 0 or size > max_size:
                raise ValueError(
                    f"Argument size should be a positive integer <= {max_size}, got {size}"
                )
            # match the given size to the existing Lebedev size or the next largest size
            size = (
                size
                if size in LEBEDEV_NPOINTS
                else leb_npts[bisect_left(leb_npts, size)]
            )
            return LEBEDEV_NPOINTS[size], size
        else:
            raise ValueError("Provide degree and/or size arguments!")

    @staticmethod
    def _load_lebedev_grid(degree: int, size: int):
        """Load the .npz file containing the pre-computed Lebedev grid points and weights.

        Parameters
        ----------
        degree : int, optional
            Degree of Lebedev grid.
        size : int, optional
            Number of Lebedev grid points.

        Returns
        -------
        np.ndarray(N, 3), np.ndarray(N,)
            The 3-dimensional Cartesian coordinates & weights of :math:`N` grid
            points on a unit sphere.

        """
        # check given degree & size
        if degree not in LEBEDEV_DEGREES:
            raise ValueError(
                f"Given degree={degree} is not supported, choose from {LEBEDEV_DEGREES}"
            )
        if size not in LEBEDEV_NPOINTS:
            raise ValueError(
                f"Given size={size} is not supported, choose from {LEBEDEV_NPOINTS}"
            )
        # load npz file corresponding to the given degree & size
        filename = f"lebedev_{degree}_{size}.npz"
        with path("grid.data.lebedev", filename) as npz_file:
            data = np.load(npz_file)
        return data["points"], data["weights"]
