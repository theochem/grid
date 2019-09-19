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

n_points = [
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

n_degree = [
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
    """Generate Lebedev grid for given degree or size.

    Either degree or size is needed to generate proper grid. If both provided,
    degree will be used instead of size.

    Parameters
    ----------
    degree : None, optional
        Degree L for Lebedev grid
    size : None, optional
        Number of preferred points on Lebedev grid

    Returns
    -------
    AngularGrid
        An AngularGrid instance with points and weights.
    """
    degree, size = _select_grid_type(degree=degree, size=size)
    points, weights = _load_grid_arrays(_load_grid_filename(degree, size))
    # set weights to 4\pi
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
    num_array = np.array(num_array)
    unik_arr = np.unique(num_array)
    degs = np.zeros(num_array.size)
    for i in unik_arr:
        deg = _select_grid_type(size=i)[0]
        degs[np.where(num_array == i)] = deg
    return degs


def _select_grid_type(*, degree=None, size=None):
    """Select proper Lebedev grid scheme for given degree or size.

    Parameters
    ----------
    degree : int, the magic number for spherical grid
    size : int, the number of points for spherical grid

    Returns
    -------
    tuple(int, int), proper magic number and its corresponding number of points.
    """
    if degree and size:
        warnings.warn(
            "Both degree and size are provided, will use degree only", RuntimeWarning
        )
    if degree:
        if degree < 0 or degree > 131:
            raise ValueError(
                f"'degree' needs to be an positive integer <= 131, got {degree}"
            )
        for index, pre_degree in enumerate(n_degree):
            if degree <= pre_degree:
                return n_degree[index], n_points[index]
    elif size:
        if size < 0 or size > 5810:
            raise ValueError(f"'size' needs to be an integer <= 5810, got {degree}")
        for index, pre_point in enumerate(n_points):
            if size <= pre_point:
                return n_degree[index], n_points[index]
    else:
        raise ValueError(
            "Please provide 'degree' or 'size' to define a grid type is provided in arguments"
        )


def _load_grid_filename(degree: int, size: int):
    """Construct Lebedev file name for given degree and size.

    Parameters
    ----------
    degree : int
    size : int

    Returns
    -------
    str, file name for given type of Lebedev grid
    """
    return f"lebedev_{degree}_{size}.npz"


def _load_grid_arrays(filename):
    """Load saved .npz file to generate Lebedev points.

    Parameters
    ----------
    filename : str or Path

    Returns
    -------
    tuple(np.ndarray(N,), np.ndarray(N,)), the coordinates and weights of grid.
    """
    with path("grid.data.lebedev", filename) as npz_file:
        data = np.load(npz_file)
    return data["points"], data["weights"]
