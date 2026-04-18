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
"""Maximum determinant spherical grids."""

from __future__ import annotations

import numpy as np
import warnings
from importlib_resources import files
from grid.basegrid import Grid

__all__ = ["MaxDeterminantGrid"]

MAXDET_CACHE = {}

class MaxDeterminantGrid(Grid):
    r"""Spherical integration grid using maximum determinant points.
    
    Maximum determinant points (also known as Fekete or extremal points) 
    provide stable integration on the sphere with non-negative weights.
    For degree :math:`t`, the number of points is :math:`N = (t+1)^2`.
    """

    def __init__(self, degree: int | None = 10, *, size: int | None = None, cache: bool = True):
        r"""Generate maximum determinant spherical grid.

        Parameters
        ----------
        degree : int or None, optional
            Maximum angular degree :math:`t` that the grid can integrate accurately. 
            If the grid for the given degree is not supported, the next largest degree is used.
            If `size` is provided, `degree` is ignored.
        size : int or None, optional, keyword-only
            Number of angular grid points. If provided, the grid with at least this many
            points is used.
        cache : bool, optional, keyword-only
            If True, store the grid in a global cache.
        """
        if size is not None:
            # Map size back to degree: N = (t+1)^2 -> t = sqrt(N) - 1
            t = int(np.ceil(np.sqrt(size))) - 1
            if degree is not None:
                warnings.warn(f"Size is used, degree={degree} is ignored!", RuntimeWarning, stacklevel=2)
            degree = t

        # Support mapping and loading
        points, weights, actual_degree = self._load_maxdet(degree, cache)
        
        # Store raw unnormalized weights
        super().__init__(points, weights)
        self._degree = actual_degree

    @staticmethod
    def _get_available_degrees():
        if not hasattr(MaxDeterminantGrid, "_available_degrees"):
            supported = []
            data_path = files("grid.data.max_det")
            for f in data_path.iterdir():
                if f.name.startswith("maxdet_") and f.name.endswith(".npz"):
                    parts = f.name.replace(".npz", "").split("_")
                    supported.append(int(parts[1]))
            supported.sort()
            MaxDeterminantGrid._available_degrees = supported
        return MaxDeterminantGrid._available_degrees

    @classmethod
    def _get_degree_and_size(cls, degree: int | None, size: int | None):
        """Map the given degree and/or size to the degree and size of a supported maxdet grid."""
        if degree is None and size is None:
            raise ValueError("Both degree and size cannot be None.")
        
        if size is not None:
            degree = int(np.ceil(np.sqrt(size))) - 1
            
        supported_degrees = cls._get_available_degrees()
        if not supported_degrees:
            raise FileNotFoundError("No precomputed maximum determinant points found in grid.data.max_det")
            
        idx = np.searchsorted(supported_degrees, degree)
        if idx >= len(supported_degrees):
            idx = len(supported_degrees) - 1
            
        actual_degree = supported_degrees[idx]
        return actual_degree, (actual_degree + 1)**2

    @property
    def degree(self):
        """int: The degree of the maximum determinant point system."""
        return self._degree

    def integrate(self, *args):
        r"""Integrate a function or arrays over the sphere.
        
        Parameters
        ----------
        *args : callable or np.ndarray
            If one argument is given and it is callable, it is evaluated on the grid 
            points and integrated. Otherwise, it behaves like Grid.integrate,
            integrating the product of the given value arrays.
            
        Returns
        -------
        float:
            The integral of the function/arrays over the unit sphere.
        """
        if len(args) == 1 and callable(args[0]):
            return 4 * np.pi * np.sum(args[0](self.points) * self.weights)
        return 4 * np.pi * super().integrate(*args)

    def _load_maxdet(self, degree, cache):
        """Internal loader for precomputed max-det points."""
        actual_degree, size = self._get_degree_and_size(degree, None)
        supported_degrees = self._get_available_degrees()
        
        if degree is not None and degree > supported_degrees[-1]:
            warnings.warn(f"Degree {degree} is larger than maximum supported {supported_degrees[-1]}. Using {supported_degrees[-1]}.", RuntimeWarning)
        
        if cache and actual_degree in MAXDET_CACHE:
            return MAXDET_CACHE[actual_degree] + (actual_degree,)

        data_path = files("grid.data.max_det")
        filename = f"maxdet_{actual_degree}_{size}.npz"
        filepath = data_path.joinpath(filename)
        
        if not filepath.exists():
             raise FileNotFoundError(f"Missing data for degree {actual_degree}")

        data = np.load(filepath)
        points, weights = data["points"], data["weights"]
        
        if cache:
            MAXDET_CACHE[actual_degree] = (points, weights)
            
        return points, weights, actual_degree
