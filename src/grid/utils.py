"""Utils function module."""
from importlib_resources import path

import numpy as np


def get_cov_radii(numbers, type="bragg"):
    """Get the covalent radii for given atomic number(s).

    Parameters
    ----------
    numbers : int or np.ndarray
        atomic number of interested
    type : str, default to bragg
        types of covalent radii for elements.
        "bragg": Bragg-Slater empirically measured covalent radii
        "cambridge": Covalent radii from analysis of the Cambridge Database"

    Returns
    -------
    np.ndarray
        covalent radii of desired atom(s)

    Raises
    ------
    ValueError
        Invalid covalent type
    """
    with path("grid.data", "cov_radii.npz") as npz_file:
        data = np.load(npz_file)
    bragg, cambridge = data["bragg"], data["cambridge"]
    if type == "bragg":
        return bragg[numbers]
    elif type == "cambridge":
        return cambridge[numbers]
    else:
        raise ValueError(f"Not supported radii type, got {type}")
