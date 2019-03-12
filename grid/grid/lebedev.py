import warnings
import numpy as np
from importlib_resources import path

n_points = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350,
            434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470,
            3890, 4334, 4802, 5294, 5810]

n_degree = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47,
            53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]


lebedev_laikov_npoints = {
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

lebedev_laikov_lmaxs = {
    0: 6, 1: 6, 2: 6, 3: 6, 4: 14, 5: 14, 6: 26, 7: 26, 8: 38, 9: 38, 10: 50,
    11: 50, 12: 74, 13: 74, 14: 86, 15: 86, 16: 110, 17: 110, 18: 146, 19: 146,
    20: 170, 21: 170, 22: 194, 23: 194, 24: 230, 25: 230, 26: 266, 27: 266, 28:
    302, 29: 302, 30: 350, 31: 350, 32: 434, 33: 434, 34: 434, 35: 434, 36: 590,
    37: 590, 38: 590, 39: 590, 40: 590, 41: 590, 42: 770, 43: 770, 44: 770, 45:
    770, 46: 770, 47: 770, 48: 974, 49: 974, 50: 974, 51: 974, 52: 974, 53: 974,
    54: 1202, 55: 1202, 56: 1202, 57: 1202, 58: 1202, 59: 1202, 60: 1454, 61:
    1454, 62: 1454, 63: 1454, 64: 1454, 65: 1454, 66: 1730, 67: 1730, 68: 1730,
    69: 1730, 70: 1730, 71: 1730, 72: 2030, 73: 2030, 74: 2030, 75: 2030, 76:
    2030, 77: 2030, 78: 2354, 79: 2354, 80: 2354, 81: 2354, 82: 2354, 83: 2354,
    84: 2702, 85: 2702, 86: 2702, 87: 2702, 88: 2702, 89: 2702, 90: 3074, 91:
    3074, 92: 3074, 93: 3074, 94: 3074, 95: 3074, 96: 3470, 97: 3470, 98: 3470,
    99: 3470, 100: 3470, 101: 3470, 102: 3890, 103: 3890, 104: 3890, 105: 3890,
    106: 3890, 107: 3890, 108: 4334, 109: 4334, 110: 4334, 111: 4334, 112: 4334,
    113: 4334, 114: 4802, 115: 4802, 116: 4802, 117: 4802, 118: 4802, 119: 4802,
    120: 5294, 121: 5294, 122: 5294, 123: 5294, 124: 5294, 125: 5294, 126: 5810,
    127: 5810, 128: 5810, 129: 5810, 130: 5810, 131: 5810,
}


def lebedev_laikov_sphere(npoints):
    degree, size = _select_grid_type(size=npoints)
    points, weights = _load_grid_arrays(_load_grid_filename(degree, size))
    return points, weights


def _select_grid_type(degree=None, size=None):
    if degree and size:
        warnings.warn("Both degree and size are provided, will use degree only",RuntimeWarning)
    if degree:
        if not isinstance(degree, int) or degree < 0 or degree > 131:
            raise ValueError(
                f"'degree' needs to be an positive integer < 131, got {degree}")
        for index, pre_degree in enumerate(n_degree):
            if degree <= pre_degree:
                return n_degree[index], n_points[index]
    elif size:
        if not isinstance(size, int) or size < 0 or size > 5810:
            raise ValueError(
                f"'size' needs to be an integer < 5810, got {degree}")
        for index, pre_point in enumerate(n_points):
            if size <= pre_point:
                return n_degree[index], n_points[index]
    else:
        raise ValueError(
            "Please provide 'degree' or 'size' to define a grid type is provided in arguments"
        )


def _load_grid_filename(degree: int, size: int):
    return f"lebedev_{degree}_{size}.npz"


def _load_grid_arrays(filename):
    with path("grid.grid.data.lebgrid", filename) as npz_file:
        data = np.load(npz_file)
    return data["points"], data["weights"]
