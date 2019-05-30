"""Functions for integrating grids."""
from grid.basegrid import Grid

import numpy as np


def two_grid_integral(grid1, grid2, value1, value2, *, func_rad, func_val):
    """Generalized two body integral.

    Parameters
    ----------
    grid1 : Grid
        Grid instance for first integral
    grid2 : Grid
        Grid instance for second integral
    value1 : np.ndarray(N,)
        Value array for the first grid
    value2 : np.ndarray(M,)
        Value array for the first grid
    func_rad : callable[np.ndarray, np.ndarray] -> np.ndarray
        Callable function for f(r1, r2)
    func_val : callable[np.ndarray, np.ndarray] -> np.ndarray
        Callable function for f(v1, v2)

    Returns
    -------
    float
        integral result of two body.
        .. math::

    Raises
    ------
    TypeError
        Description
    """
    if not isinstance(grid1, Grid):
        raise TypeError(f"grid1 is not type Grid, got {type(grid1)}")
    if not isinstance(grid2, Grid):
        raise TypeError(f"grid2 is not type Grid, got {type(grid2)}")
    if not isinstance(value1, np.ndarray):
        raise TypeError(f"value1 is not type np.ndarray, got {type(value1)}")
    if not isinstance(value2, np.ndarray):
        raise TypeError(f"value2 is not type np.ndarray, got {type(value2)}")
    if not callable(func_rad):
        raise TypeError(f"func_rad is not type function, got {type(func_rad)}")
    if not callable(func_val):
        raise TypeError(f"func_val is not type function, got {type(func_val)}")

    weights = grid1.weights[:, None] * grid2.weights
    r12_value = _v1_v2_low(grid1.points, grid2.points, func_rad)
    v12_value = _v1_v2_low(value1, value2, func_val)
    return np.sum(weights * r12_value * v12_value)


def elec_elec_integral(grid1, grid2, value1, value2):
    """Electron electron interaction integral.

    Parameters
    ----------
    grid1 : Grid
        Grid instance for first integral
    grid2 : Grid
        Grid instance for second integral
    value1 : np.ndarray(N,)
        Value array for the first grid
    value2 : np.ndarray(M,)
        Value array for the first grid

    Returns
    -------
    float
        integral of two electron interaction
    """
    if grid1.points.ndim != 2 or grid1.points.shape[-1] != 3:
        raise ValueError(f"Input grid1 is not 3d grid, got {grid1.points.shape}")
    if grid2.points.ndim != 2 or grid2.points.shape[-1] != 3:
        raise ValueError(f"Input grid2 is not 3d grid, got {grid2.points.shape}")
    return two_grid_integral(
        grid1,
        grid2,
        value1,
        value2,
        func_rad=lambda x, y: 1 / np.linalg.norm(x - y, axis=-1),
        func_val=lambda x, y: x * y,
    )


def _v1_v2_low(v1, v2, v12_func):
    """Low lever helpfer function for v1, v2 relation.

    Parameters
    ----------
    v1 : np.ndarray(N,)
        First array with values
    v2 : np.ndarray(M,)
        First array with values
    v12_func : callable[np.ndarray, np.ndarray] -> np.ndarray
        function used for f(v1, v2)

    Returns
    -------
    np.ndarray(N, M)
        f(v1, v2) value for each v1, v2 paris
    """
    return v12_func(v1[:, None], v2)
