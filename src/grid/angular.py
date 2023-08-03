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
Angular grid module for constructing integration grids on the unit sphere.

The Lebedev grid points here were obtained and calculated from the theochem/HORTON package.
Their calculations were based on F77 translation by Dr. Christoph van Wuellen
and these are the comments from that translation.

This subroutine is part of a set of subroutines that generate
Lebedev grids [1-6]_ for integration on a sphere. The original
C-code [1]_ was kindly provided by Dr. Dmitri N. Laikov and
translated into fortran by Dr. Christoph van Wuellen.
This subroutine was translated from C to fortran77 by hand.
Users of this code are asked to include reference [1]_ in their
publications, and in the user- and programmers-manuals
describing their codes.
This code was distributed through CCL (http://www.ccl.net/).

The symmetric spherical t-design were obtained from reference [7]_.

References
----------
The following references are for the Lebedev grid points.

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

The following is references for the symmetric spherical t-design points:

.. [7] R. S. Womersley, Efficient Spherical Designs with Good Geometric Properties.
   In: Dick J., Kuo F., Wozniakowski H. (eds) Contemporary Computational Mathematics -
   A Celebration of the 80th Birthday of Ian Sloan. Springer (2018) pp. 1243-1285
   https://doi.org/10.1007/978-3-319-72456-0_57

"""

import warnings
from bisect import bisect_left

import numpy as np
from importlib_resources import files

from grid.basegrid import Grid

# Lebedev dictionary for converting number of grid points (keys) to grid's degrees (values)
LEBEDEV_NPOINTS = {
    6: 3,
    18: 5,
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
SPHERICAL_NPOINTS = {
    2: 1,
    6: 3,
    12: 5,
    32: 7,
    48: 9,
    70: 11,
    94: 13,
    120: 15,
    156: 17,
    192: 19,
    234: 21,
    278: 23,
    328: 25,
    380: 27,
    438: 29,
    498: 31,
    564: 33,
    632: 35,
    706: 37,
    782: 39,
    864: 41,
    948: 43,
    1038: 45,
    1130: 47,
    1228: 49,
    1328: 51,
    1434: 53,
    1542: 55,
    1656: 57,
    1772: 59,
    1894: 61,
    2018: 63,
    2148: 65,
    2280: 67,
    2418: 69,
    2558: 71,
    2704: 73,
    2852: 75,
    3006: 77,
    3162: 79,
    3324: 81,
    3488: 83,
    3658: 85,
    3830: 87,
    4008: 89,
    4188: 91,
    4374: 93,
    4562: 95,
    4756: 97,
    4952: 99,
    5154: 101,
    5358: 103,
    5568: 105,
    5780: 107,
    5998: 109,
    6218: 111,
    6444: 113,
    6672: 115,
    6906: 117,
    7142: 119,
    7384: 121,
    7628: 123,
    7878: 125,
    8130: 127,
    8388: 129,
    8648: 131,
    8914: 133,
    9182: 135,
    9456: 137,
    9732: 139,
    10014: 141,
    10298: 143,
    10588: 145,
    10880: 147,
    11178: 149,
    11478: 151,
    11784: 153,
    12092: 155,
    12406: 157,
    12722: 159,
    13044: 161,
    13368: 163,
    13698: 165,
    14030: 167,
    14368: 169,
    14708: 171,
    15054: 173,
    15402: 175,
    15756: 177,
    16112: 179,
    16474: 181,
    16838: 183,
    17208: 185,
    17580: 187,
    17958: 189,
    18338: 191,
    18724: 193,
    19112: 195,
    19506: 197,
    19902: 199,
    20304: 201,
    20708: 203,
    21118: 205,
    21530: 207,
    21948: 209,
    22368: 211,
    22794: 213,
    23222: 215,
    23656: 217,
    24092: 219,
    24534: 221,
    24978: 223,
    25428: 225,
    25880: 227,
    26338: 229,
    26798: 231,
    27264: 233,
    27732: 235,
    28206: 237,
    28682: 239,
    29164: 241,
    29648: 243,
    30138: 245,
    30630: 247,
    31128: 249,
    31628: 251,
    32134: 253,
    32642: 255,
    33156: 257,
    33672: 259,
    34194: 261,
    34718: 263,
    35248: 265,
    35780: 267,
    36318: 269,
    36858: 271,
    37404: 273,
    37952: 275,
    38506: 277,
    39062: 279,
    39624: 281,
    40188: 283,
    40758: 285,
    41330: 287,
    41908: 289,
    42488: 291,
    43074: 293,
    43662: 295,
    44256: 297,
    44852: 299,
    45454: 301,
    46058: 303,
    46668: 305,
    47280: 307,
    47898: 309,
    48518: 311,
    49144: 313,
    49772: 315,
    50406: 317,
    51042: 319,
    51684: 321,
    52328: 323,
    52978: 325,
}

# Lebedev/Spherical dictionary for converting grid's degrees (keys) to numb of grid points (values)
LEBEDEV_DEGREES = dict([(v, k) for k, v in LEBEDEV_NPOINTS.items()])
SPHERICAL_DEGREES = dict([(v, k) for k, v in SPHERICAL_NPOINTS.items()])

LEBEDEV_CACHE = {}
SPHERICAL_CACHE = {}


class AngularGrid(Grid):
    r"""Angular grid for integrating functions on the unit sphere.

    This class numerically evaluates the integral of a function
    :math:`f: S^2 \rightarrow \mathbb{R}` on the unit-sphere as follows:

    .. math::
        \int_{S^2} f   = \int_0^{2\pi} \int_0^\pi f(\theta, \phi) \sin(\phi)
         d\theta d\phi \approx \sum w_i f(\phi_i, \theta_i),

    where :math:`S^2` is the unit-sphere, :math:`\theta \in [0, 2\pi]` and :math:`\phi \in [0, \pi)`
    and :math:`w^{ang}_i` are the quadrature weights.

    Two types of angular grids are provided: Lebedev-Laikov grid (default) or the
    symmetric spherical t-design. Specifically, for spherical t-design, the weights are constant
    value of :math:`4 \pi / N`, where :math:`N` is the number of points in the grid. The weights
    are chosen so that the spherical harmonics are normalized.

    """

    def __init__(
        self,
        points: np.ndarray = None,
        weights: np.ndarray = None,
        *,
        degree: int = None,
        size: int = None,
        cache: bool = True,
        use_spherical: bool = False,
    ):
        r"""Generate angular grid for a given degree and/or size.

        Three choices to construct this grid:

        1. Add your own points and weights.
        2. Specify maximum degree of spherical harmonics that the angular grid can
           integrate accurately on a unit sphere.
        3. The number of points (size) wanted in the grid. The size corresponds
           to a maximum degree.

        Parameters
        ----------
        points : ndarray(N, 3), optional
            The Cartesian coordinates of integral grid points.  The attribute
            `weights` should also be provided.
        weights : ndarray(N,), optional
            The weights of each point on the integral quadrature grid. The attribute
            `points` should also be applied.
        degree : int, optional
            Maximum angular degree :math:`l` of spherical harmonics that the grid
            can integrate accurately. If the grid corresponding to the given angular
            degree is not supported, the next largest degree is used.
        size : int, optional
            Number of grid points. If the grid corresponding to the given size is
            not supported, the next largest size is used. If both degree and size are given,
            degree is used for constructing the grid.
        cache : bool, optional
            If true, then store the points and weights of the AngularGrid in cache
            to avoid duplicate grids that have the same `degree`. Default to `True`.
        use_spherical : bool, optional
            If true, then it loads/uses the symmetric spherical t-design, rather than
            Lebedev-Laikov grid.

        Returns
        -------
        AngularGrid
            An angular grid with points and weights on a unit sphere.

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

        Notes
        -----
        - Sometimes the weights for Lebedev-Laikov grids can be negative. Choosing degrees that have
          positive weights can mitigate round-off errors. Degrees equal to 13, 25 or 27 have
          negative weights.

        """
        if not isinstance(use_spherical, bool):
            raise TypeError(
                f"use_spherical {use_spherical, type(use_spherical)} should be of type " f"boolean."
            )
        # construct grid from pts and wts given directly
        if points is not None and weights is not None:
            super().__init__(points, weights)
            if degree or size:
                warnings.warn(
                    "degree or size are not used for generating grids "
                    "because points and weights are provided",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            # map degree and size to the supported (i.e., pre-computed) degree and size
            degree, size = self._get_size_and_degree(
                degree=degree, size=size, use_spherical=use_spherical
            )
            # load pre-computed angular points & weights and make angular grid
            cache_dict = SPHERICAL_CACHE if use_spherical else LEBEDEV_CACHE
            if degree not in cache_dict:
                points, weights = self._load_precomputed_angular_grid(degree, size, use_spherical)
                if cache:
                    cache_dict[degree] = points, weights
            else:
                points, weights = cache_dict[degree]
            self._degree = degree
            # Multiply weights by 4 pi, so that the spherical harmonics are orthonormalized,
            #   etc. \int Y_l1 Y_l2 = \delta_{l1, l2}
            super().__init__(points, weights * 4 * np.pi)

        if not use_spherical and np.any(weights < 0.0):
            # Lebedev degrees 13, 25, 27 have negative weights. Symmetric spherical t-design
            # have positive weights.
            warnings.warn(
                "Lebedev weights are negative which can introduce round-off errors.", stacklevel=2
            )

        self._use_spherical = use_spherical

    @property
    def degree(self):
        r"""int: The degree of spherical harmonics that this angular grid can integrate exactly."""
        return self._degree

    @property
    def use_spherical(self):
        r"""bool: If true, then symmetric spherical t-design was used else Lebedev-Laikov grid."""
        return self._use_spherical

    @staticmethod
    def convert_angular_sizes_to_degrees(sizes: np.ndarray, use_spherical: bool = False):
        """
        Convert given Lebedev/Spherical design grid sizes to degrees.

        Parameters
        ----------
        sizes : ndarray[int]
            Sequence of angular grid sizes (e.g., number of points for each atomic shell).
        use_spherical: bool, optional
            If true, loads the symmetric spherical t-design grid rather than the Lebedev-Laikov
            grid.


        Returns
        -------
        ndarray[int]
            Sequence of the corresponding angular degree of the angular grid corresponding to
            its size.

        """
        degrees = np.zeros(len(sizes), dtype=int)
        for size in np.unique(sizes):
            # get the degree corresponding to the given (unique) size
            deg = AngularGrid._get_size_and_degree(
                degree=None, size=size, use_spherical=use_spherical
            )[0]
            # set value of degree to corresponding to the given size equal to deg
            degrees[np.where(sizes == size)] = deg
        return degrees

    @staticmethod
    def _get_size_and_degree(*, degree: int = None, size: int = None, use_spherical: bool = False):
        """
        Map the given degree and/or size to the degree and size of a supported angular grid.

        Parameters
        ----------
        degree : int, optional
            Maximum angular degree :math:`l` of spherical harmonics that the angular grid
            can integrate accurately. If the angular grid corresponding to the given angular
            degree is not supported, the next largest degree is used.
        size : int, optional
            Number of angular grid points. If the angular grid corresponding to the given size is
            not supported, the next largest size is used. If both degree and size are given,
            degree is used for constructing the grid.
        use_spherical: bool, optional
            If true, loads the symmetric spherical t-design grid rather than the Lebedev-Laikov
            grid.

        Returns
        -------
        (int, int)
            Degree and size of a supported angular grid (equal to or larger than the
            requested grid).

        """
        if isinstance(size, bool):
            raise TypeError(
                f"size {size} should be of type int, not boolean. May be confused"
                f"with use_spherical."
            )

        degrees = SPHERICAL_DEGREES if use_spherical else LEBEDEV_DEGREES
        npoints = SPHERICAL_NPOINTS if use_spherical else LEBEDEV_NPOINTS
        if degree and size:
            warnings.warn(
                "Both degree and size arguments are given, so only degree is used!",
                RuntimeWarning,
                stacklevel=2,
            )
        if degree is not None:
            ang_degs = list(degrees.keys())
            max_degree = max(ang_degs)
            if degree < 0 or degree > max_degree:
                raise ValueError(
                    f"Argument degree should be a positive integer <= {max_degree}, got {degree}"
                )
            # match the given degree to the existing angular degree or the next largest degree
            degree = degree if degree in degrees else ang_degs[bisect_left(ang_degs, degree)]
            return degree, degrees[degree]
        elif size:
            ang_npts = list(npoints.keys())
            max_size = max(ang_npts)
            if size < 0 or size > max_size:
                raise ValueError(
                    f"Argument size should be a positive integer <= {max_size}, got {size}"
                )
            # match the given size to the existing angular size or the next largest size
            size = size if size in npoints else ang_npts[bisect_left(ang_npts, size)]
            return npoints[size], size
        else:
            raise ValueError("Provide degree and/or size arguments!")

    @staticmethod
    def _load_precomputed_angular_grid(degree: int, size: int, use_spherical: bool):
        """
        Load the .npz file containing the pre-computed angular grid points and weights.

        Either loads Lebedev-Laikov grid or the symmetric spherical t-design grid.

        Parameters
        ----------
        degree : int
            Maximum angular degree :math:`l` of spherical harmonics that the angular grid
            can integrate accurately.
        size : int
            Number of angular grid points. If the angular grid corresponding to the given size is
            not supported, the next largest size is used.
        use_spherical: bool, optional
            If true, loads the symmetric spherical t-design grid rather than the Lebedev-Laikov
            grid.

        Returns
        -------
        (ndarray(N, 3), ndarray(N,))
            The three-dimensional Cartesian coordinates & weights of :math:`N` grid
            points on a unit sphere.

        """
        if not isinstance(use_spherical, bool):
            raise TypeError(f"use_spherical {use_spherical} should be of type boolean.")
        degrees = SPHERICAL_DEGREES if use_spherical else LEBEDEV_DEGREES
        npoints = SPHERICAL_NPOINTS if use_spherical else LEBEDEV_NPOINTS
        type = "spherical" if use_spherical else "lebedev"
        file_path = "grid.data.spherical_design" if use_spherical else "grid.data.lebedev"

        # check given degree & size
        if degree not in degrees:
            raise ValueError(f"Given degree={degree} is not supported, choose from {degrees}")
        if size not in npoints:
            raise ValueError(f"Given size={size} is not supported, choose from {npoints}")
        # load npz file corresponding to the given degree & size
        filename = f"{type}_{degree}_{size}.npz"
        data = np.load(files(file_path).joinpath(filename))
        if len(data["weights"]) == 1:
            return data["points"], np.ones(len(data["points"])) * data["weights"]
        return data["points"], data["weights"]
