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
"""Generate Lebedev grid."""

import warnings

from grid.basegrid import AngularGrid

from importlib_resources import path

import numpy as np


lebedev_npoints = [
    6,
    14,
    26,
    38,
    50,
    74,
    86,
    110,
    146,
    170,
    194,
    230,
    266,
    302,
    350,
    434,
    590,
    770,
    974,
    1202,
    1454,
    1730,
    2030,
    2354,
    2702,
    3074,
    3470,
    3890,
    4334,
    4802,
    5294,
    5810,
]

lebedev_degrees = [
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21,
    23,
    25,
    27,
    29,
    31,
    35,
    41,
    47,
    53,
    59,
    65,
    71,
    77,
    83,
    89,
    95,
    101,
    107,
    113,
    119,
    125,
    131,
]


def generate_lebedev_grid(*, degree=None, size=None):
    r"""Generate a Lebedev angular grid for the given degree and/or size.

    Parameters
    ----------
    degree : int, optional
        Degree of Lebedev grid. If the Lebedev grid corresponding to the given degree is not
        supported, the next largest degree is used.
    size : int, optional
        Number of Lebedev grid points. If the Lebedev grid corresponding to the given size is not
        supported, the next largest size is used. If both degree and size are given, degree is
        used for constructing the grid.

    Returns
    -------
    AngularGrid
        An angular grid with Lebedev points and weights (including :math:`4\pi`) on a unit sphere.

    """
    # map degree and size to the supported (i.e., pre-computed) degree and size
    degree, size = _select_grid_type(degree=degree, size=size)
    # load pre-computed Lebedev points & weights and make angular grid
    points, weights = _load_grid_filename(degree, size)
    return AngularGrid(points, weights * 4 * np.pi)


def match_degree(degree_nums):
    """Generate proper angular degree for given arbitrary degree list.

    Parameters
    ----------
    degree_nums : list[int]
        a list of arbitrary degree nums

    Returns
    -------
    np.ndarray[int]
        An array of proper angular degree values
    """
    return np.array([_select_grid_type(degree=i)[0] for i in degree_nums], dtype=int)


def size_to_degree(num_array):
    """Generate degs given nums.

    Parameters
    ----------
    num_array : np.ndarray(N,)
        Numpy array with # of points for each shell

    Returns
    -------
    np.ndarray(N,)
        Numpy array with L value for each shell
    """
    num_array = np.asarray(num_array)
    unik_arr = np.unique(num_array)
    degs = np.zeros(num_array.size)
    for i in unik_arr:
        deg = _select_grid_type(size=i)[0]
        degs[np.where(num_array == i)] = deg
    return degs


def _select_grid_type(*, degree=None, size=None):
    """Map the given degree and/or size to the degree and size of a supported Lebedev grid.

    Parameters
    ----------
    degree : int, optional
        Degree of Lebedev grid. If the Lebedev grid corresponding to the given degree is not
        supported, the next largest degree is used.
    size : int, optional
        Number of Lebedev grid points. If the Lebedev grid corresponding to the given size is not
        supported, the next largest size is used. If both degree and size are given, degree is
        used for constructing the grid.

    Returns
    -------
    int, int
        Degree and size of a supported Lebedev grid (equal to or larger than the requested grid).

    """
    if degree and size:
        warnings.warn(
            "Both degree and size arguments are given, so only degree is used!",
            RuntimeWarning,
        )
    if degree:
        max_degree = np.max(lebedev_degrees)
        if degree < 0 or degree > max_degree:
            raise ValueError(
                f"Argument degree should be a positive integer <= {max_degree}, got {degree}"
            )
        # match the given degree to the existing Lebedev degree or the next largest degree
        for index, pre_degree in enumerate(lebedev_degrees):
            if degree <= pre_degree:
                return lebedev_degrees[index], lebedev_npoints[index]
    elif size:
        max_size = np.max(lebedev_npoints)
        if size < 0 or size > max_size:
            raise ValueError(
                f"Argument size should be a positive integer <= {max_size}, got {size}"
            )
        # match the given size to the existing Lebedev size or the next largest size
        for index, pre_point in enumerate(lebedev_npoints):
            if size <= pre_point:
                return lebedev_degrees[index], lebedev_npoints[index]
    else:
        raise ValueError("Provide degree and/or size arguments!")


def _load_grid_filename(degree: int, size: int):
    """Load saved .npz file to construct Lebedev grid.

    Parameters
    ----------
    degree : int
    size : int

    Returns
    -------
    tuple(np.ndarray(N,), np.ndarray(N,)), the coordinates and weights of grid.
    """
    filename = f"lebedev_{degree}_{size}.npz"
    with path("grid.data.lebedev", filename) as npz_file:
        data = np.load(npz_file)
    return data["points"], data["weights"]
