"""Comstruct basic grid data structure."""
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


class AtomicGrid(Grid):
    """Atomic grid."""

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
    """PlaceHolder for 1dGrid object."""


class RadialGrid(Grid):
    """PlaceHolder for Radial Grid."""
