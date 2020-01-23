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

from scipy.spatial import cKDTree


class Grid:
    """Basic Grid class for grid information storage."""

    def __init__(self, points, weights):
        """Construct Grid instance.

        Parameters
        ----------
        points : np.ndarray(N,) or np.ndarray(N, M)
            An array with positions of the grid points.
        weights : np.ndarray(N,)
            An array of weights associated with each point on the grid.

        Raises
        ------
        ValueError
            Shape of points and weights does not match.
        """
        if len(points) != len(weights):
            raise ValueError(
                "Number of points and weights does not match. \n"
                f"Number of points: {len(points)}, Number of weights: {len(weights)}."
            )
        if weights.ndim != 1:
            raise ValueError(
                f"Argument weights should be a 1-D array. weights.ndim={weights.ndim}"
            )
        if points.ndim not in [1, 2]:
            raise ValueError(
                f"Argument points should be a 1D or 2D array. points.ndim={points.ndim}"
            )
        self._points = points
        self._weights = weights
        self._kdtree = None

    @property
    def points(self):
        """np.ndarray(N,) or np.ndarray(N, M): Positions of the grid points."""
        return self._points

    @property
    def weights(self):
        """np.ndarray(N,): the weights of each grid point."""
        return self._weights

    @property
    def size(self):
        """int: the total number of points on the grid."""
        return self._weights.size

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
        r"""Integrate over the whole grid for given multiple value arrays.

        Product of all value_arrays will be computed element-wise then
        integrated on the grid with its weights.
        .. math::
            Integral = \int w(x) \prod_i f_i(x) dx

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
            if array.shape != (self.size,):
                raise ValueError(f"Arg {i} need to be of shape ({self.size},).")
        # return np.einsum("i, ..., i", a, ..., z)
        return np.einsum(
            "i" + ",i" * len(value_arrays),
            self.weights,
            *(array for array in value_arrays),
        )

    def get_subgrid(self, center, radius):
        """Create a grid from subset of points within the given radius of center.

        Parameters
        ----------
        center : float or np.array(M,)
            Cartesian coordinates of subgrid center.
        radius : float
            Radius of sphere around the center. When equal to np.inf, the
            subgrid coincides with the whole grid, which can be useful for
            debugging.

        Returns
        -------
        SubGrid
            Instance of SubGrid.

        """
        center = np.asarray(center)
        if center.shape != self._points.shape[1:]:
            raise ValueError(
                "Argument center has the wrong shape \n"
                f"center.shape: {center.shape}, points.shape: {self._points.shape}"
            )
        if radius < 0:
            raise ValueError(f"Negative radius: {radius}")
        if not (np.isfinite(radius) or radius == np.inf):
            raise ValueError(f"Invalid radius: {radius}")
        if radius == np.inf:
            return SubGrid(self._points, self._weights, center, np.arange(self.size))
        else:
            # When points.ndim == 1, we have to reshape a few things to
            # make the input compatible with cKDTree
            _points = self._points.reshape(self.size, -1)
            _center = np.array([center]) if center.ndim == 0 else center
            if self._kdtree is None:
                self._kdtree = cKDTree(_points)
            indices = np.array(self._kdtree.query_ball_point(_center, radius, p=2.0))
            return SubGrid(
                self._points[indices], self._weights[indices], center, indices
            )


class AngularGrid(Grid):
    """Angular lebedev grid."""


class SubGrid(Grid):
    """Subset of grid surrounding a center."""

    def __init__(self, points, weights, center, indices=None):
        r"""Initialize a sub-grid.

        Parameters
        ----------
        points : np.ndarray(N,) or np.ndarray(N,M)
            Cartesian coordinates of :math:`N` grid points in 1D or M-D space.
        weights : np.ndarray(N)
            Integration weight of :math:`N` grid points
        center : float or np.ndarray(M,)
            Cartesian coordinates of sub-grid center in 3D space.
        indices : np.ndarray(N,), optional
            Indices of :math:`N` grid points and weights in the parent grid.

        """
        if indices is not None:
            if len(points) != len(indices):
                raise ValueError(
                    "Number of points and indices does not match. \n"
                    f"number of points: {len(points)}, number of indices: {len(indices)}."
                )
            if indices.ndim != 1:
                raise ValueError(
                    f"Argument indices should be a 1-D array. indices.ndim={indices.ndim}"
                )
        super().__init__(points, weights)
        self._center = center
        self._indices = indices

    @property
    def center(self):
        """np.ndarray(3,): Cartesian coordinates of sub-grid center."""
        return self._center

    @property
    def indices(self):
        """np.ndarray(N,): Indices of grid points and weights in the parent grid."""
        return self._indices


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
