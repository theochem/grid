"""Generate Lebedev grid."""

import warnings

from grid.grid import AngularGrid

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
    """Generate lebedev grid for given degree or size.

    Either degree or size is needed to generate proper grid. If both provided,
    degree will be used instead of size.

    Parameters
    ----------
    degree : None, optional
        Degree L for lebedev grid
    size : None, optional
        Number of preferred points on lebedev grid

    Returns
    -------
    AngularGrid
        An AngularGrid instance with points and weights.
    """
    degree, size = _select_grid_type(degree=degree, size=size)
    points, weights = _load_grid_arrays(_load_grid_filename(degree, size))
    return AngularGrid(points, weights)


def _select_grid_type(*, degree=None, size=None):
    """Select proper lebedev grid scheme for given degree or size."""
    if degree and size:
        warnings.warn(
            "Both degree and size are provided, will use degree only", RuntimeWarning
        )
    if degree:
        if not isinstance(degree, int) or degree < 0 or degree > 131:
            raise ValueError(
                f"'degree' needs to be an positive integer < 131, got {degree}"
            )
        for index, pre_degree in enumerate(n_degree):
            if degree <= pre_degree:
                return n_degree[index], n_points[index]
    elif size:
        if not isinstance(size, int) or size < 0 or size > 5810:
            raise ValueError(f"'size' needs to be an integer < 5810, got {degree}")
        for index, pre_point in enumerate(n_points):
            if size <= pre_point:
                return n_degree[index], n_points[index]
    else:
        raise ValueError(
            "Please provide 'degree' or 'size' to define a grid type is provided in arguments"
        )


def _load_grid_filename(degree: int, size: int):
    """Construct lebedev file name for given degree and size."""
    return f"lebedev_{degree}_{size}.npz"


def _load_grid_arrays(filename):
    """Load .npz presaved file to generate lebedev points."""
    with path("grid.data.lebedev", filename) as npz_file:
        data = np.load(npz_file)
    return data["points"], data["weights"]
