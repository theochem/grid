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
"""Construct basic grid data structure."""
import numpy as np


class Grid:
    """Basic Grid class for grid information storage."""

    def __init__(self, points, weights):
        """Construct Grid instance.

        Parameters
        ----------
        points : np.ndarray(N,)
            An array with coordinates as each entry.

        weights : np.ndarray(N,)
            An array of weights associated with each point on the grid.

        Raises
        ------
        ValueError
            Shape of points and weights does not match.
        """
        if len(points) != len(weights):
            raise ValueError(
                "Shape of points and weight does not match. \n"
                "shape of points: {len(poitns)}, shape of weights: {len(weights)}."
            )
        if weights.ndim != 1:
            raise ValueError(
                f"Argument weights should be a 1-D array. weights.ndim={weights.ndim}"
            )
        self._points = points
        self._weights = weights
        self._size = self._weights.size

    @property
    def points(self):
        """np.ndarray(N,): the coordinates of each grid point."""
        return self._points

    @property
    def weights(self):
        """np.ndarray(N,): the weights of each grid point."""
        return self._weights

    @property
    def size(self):
        """int: the total number of points on the grid."""
        return self._size

    def __getitem__(self, index):
        """Dunder method for index grid object and slicing.

        Parameters
        ----------
        index : int or slice
            index of slice object for selecting certain part of grid

        Returns
        -------
        Grid
            Return a new Grid object with selected points
        """
        if isinstance(index, int):
            return self.__class__(
                np.array([self.points[index]]), np.array([self.weights[index]])
            )
        else:
            return self.__class__(
                np.array(self.points[index]), np.array(self.weights[index])
            )

    def integrate(self, *value_arrays):
        """Integrate over the whole grid for given multiple value arrays.

        Parameters
        ----------
        *value_arrays : np.ndarray(N, )
            One or multiple value array to integrate.

        Returns
        -------
        float
            The calculated integral over given integrand or function

        Raises
        ------
        TypeError
            Input integrand is not of type np.ndarray.
        ValueError
            Input integrand array is given or not of proper shape.
        """
        if len(value_arrays) < 1:
            raise ValueError(f"No array is given to integrate.")
        for i, array in enumerate(value_arrays):
            if not isinstance(array, np.ndarray):
                raise TypeError(f"Arg {i} is {type(i)}, Need Numpy Array.")
            if array.size != self.size:
                raise ValueError(f"Arg {i} need to be of shape {self.size}.")
        # return np.einsum("i, ..., i", a, ..., z)
        return np.einsum(
            "i" + ",i" * len(value_arrays),
            self.weights,
            *(np.ravel(i) for i in value_arrays),
        )


class AngularGrid(Grid):
    """Angular lebedev grid."""


class SimpleAtomicGrid(Grid):
    """Simplified Atomic grid."""

    def __init__(self, points, weights, center):
        """Initialize an atomic grid.

        Parameters
        ----------
        points : np.ndarray(N, 3)
            Points on 3d space
        weights : np.ndarray(N)
            Weights for each points on the grid
        center : np.ndarray(3,)
            The center of the atomic grid
        """
        super().__init__(points, weights)
        self._center = center

    @property
    def center(self):
        """np.ndarray(3,): return the coordinates of the atomic grid center."""
        return self._center


class OneDGrid(Grid):
    """One-Dimensional Grid."""

    def __init__(self, points, weights, domain=None):
        r"""Construct grid.

        Parameters
        ----------
        points : np.ndarray(N,)
            A 1-D array of coordinates of :math:`N` points in one-dimension.
        weights : np.ndarray(N,)
            A 1-D array of integration weights of :math:`N` points in one-dimension.
        domain : tuple(float, float), optional
            Lower and upper bounds for which the grid can carry out numerical
            integration. This does not always coincide with the positions of the first
            and last grid point. For example, in case of the Gauss-Chebyshev quadrature
            the domain is [-1,1] but all grid points lie in (-1, 1).

        """
        # check points & weights
        if points.ndim != 1:
            raise ValueError(
                f"Argument points should be a 1-D array. points.ndim={points.ndim}"
            )

        # check domain
        if domain is not None:
            if len(domain) != 2 or domain[0] > domain[1]:
                raise ValueError(
                    f"domain should be an ascending tuple of length 2. domain={domain}"
                )
            min_p = np.min(points)
            if domain[0] - 1e-7 >= min_p:
                raise ValueError(
                    f"point coordinates should not be below domain! {min_p < domain[0]}"
                )
            max_p = np.max(points)
            if domain[1] + 1e-7 <= max_p:
                raise ValueError(
                    f"point coordinates should not be above domain! {domain[1] < max_p}"
                )
        super().__init__(points, weights)
        self._domain = domain

    @property
    def domain(self):
        """(float, float): the range of grid points."""
        return self._domain

    def __getitem__(self, index):
        """Dunder method for index grid object and slicing.

        Parameters
        ----------
        index : int or slice
            index of slice object for selecting certain part of grid

        Returns
        -------
        OneDGrid
            Return a new grid instance with a subset of points.
        """
        if isinstance(index, int):
            return self.__class__(
                np.array([self.points[index]]),
                np.array([self.weights[index]]),
                self._domain,
            )
        else:
            return self.__class__(
                np.array(self.points[index]),
                np.array(self.weights[index]),
                self._domain,
            )
