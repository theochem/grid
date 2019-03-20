# -*- coding: utf-8 -*-
# GRID is a numerical integration library for quantum chemistry.
#
# Copyright (C) 2011-2017 The GRID Development Team
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
#
# --
"""Uniform Grid Module."""


import numpy as np


__all__ = ['UniformGrid']


class UniformGrid(object):
    """Class of uniform grid."""

    def __init__(self, origin, rvecs, shape):
        """Initialize the uniform grid.

        Parameters
        ----------
        origin : np.ndarray
            Cartesian coordinates of grid origin, i.e. coordinates of first grid point.
        rvecs : np.ndarray
            Real-space basis vectors defining the spacings between the grids.
        shape : np.ndarray
            Shape of the grid, i.e. number of points along each basis.

        """
        assert origin.shape[0] == 3
        assert rvecs.shape[0] == 3
        assert rvecs.shape[1] == 3
        assert shape.shape[0] == 3

        self._origin = origin
        self._rvecs = rvecs
        self._shape = shape

    @property
    def origin(self):
        """Cartesian coordinates of grid origin."""
        return self._origin

    @property
    def rvecs(self):
        """Real-space basis vectors of grid."""
        return self._rvecs

    @property
    def shape(self):
        """Number of grid points along each axis."""
        return self._shape

    @property
    def size(self):
        """Total number of grid points."""
        return np.product(self.shape)

    def delta_grid_point(self, center, index):
        """Compute the vector **from** a center **to** a grid point.

        Parameters
        ----------
        center : np.ndarray
            Cartesian coordinates of center points.
        index : np.ndarray
            Integer indexes of the grid point (may fall outside of shape).

        """
        delta = self.origin + np.dot(index, self.rvecs) - center
        return delta

    def dist_grid_point(self, center, index):
        """Compute the distance between a center and a grid point.

        Parameters
        ----------
        center : np.ndarray
            Cartesian coordinates of center points.
        index : np.ndarray
            Integer indexes of the grid point (may fall outside of shape).

        """
        return np.linalg.norm(self.delta_grid_point(center, index))

    # def get_ranges_rcut(self, center, rcut):
    #     """Return the ranges of indexes that lie within the cutoff sphere.
    #
    #     Parameters
    #     ----------
    #     center : np.ndarray
    #         Cartesian coordinates of center points.
    #     rcut : float
    #         Radius of cutoff sphere.
    #     """
    #     assert center.size == 3
    #     assert rcut >= 0
    #     # compute spacing between grid points along each axis
    #     spacing = np.linalg.norm(self.rvecs, axis=1)

    def integrate(self, *arrays):
        """Integrate dot product of all arrays.

        Parameters
        ----------
        arrays : sequence of np.ndarray
            All arguments must be arrays with the same size as the number
            of grid points. The arrays contain the functions, evaluated
            at the grid points, that must be multiplied and integrated.

        """
        args = [array.ravel() for array in arrays]
        # dot product of all arrays
        if len(args) == 1:
            integrand = np.sum(arrays)
        else:
            integrand = np.linalg.multi_dot(arrays)
        return integrand * np.abs(np.linalg.det(self.rvecs))
