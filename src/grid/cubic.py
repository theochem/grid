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
r"""3D Integration Grid"""


import numpy as np
from scipy.interpolate import CubicSpline
from sympy import symbols
from sympy.functions.combinatorial.numbers import bell

from grid.basegrid import Grid, OneDGrid


class _CubicGrid(Grid):
    def __init__(self, points, weights, shape):
        r"""
        Construct the CubicGrid class, i.e. each point is specified by (i, j, k) indices.

        Parameters
        ----------
        points : np.ndarray(M, 3)
            The 3D points of the cubic grid.
        weights : np.ndarray(M*3)
            The weights corresponding to each cubic grid point.  It moves in the following
            order (x, y, z).
        shape : np.ndarray(3)
            The number of points in each of the three (x, y, z) directions.

        """
        if len(shape) != 3:
            raise ValueError("Shape (length {0}) should have length three.".format(len(shape)))
        self._shape = shape
        super().__init__(points, weights)

    @property
    def shape(self):
        r"""Return number of points in each direction."""
        return self._shape

    def interpolate_function(
            self, real_pt, func_values, use_log=False, nu_x=0, nu_y=0, nu_z=0,
    ):
        r"""
        Interpolate function at a point.

        Parameters
        ----------
        real_pt : np.ndarray(3,)
            Point in :math:`\mathbb{R}^3` that needs to be interpolated.
        func_values : np.ndarray(N,)
            Function values at each point of the grid `points`.
        use_log : bool
            If true, then logarithm is applied before interpolating to the function values.
            Can only be used for interpolating derivatives when the derivative is not a
            mixed derivative.
        nu_x : int
            If zero, then the function in x-direction is interpolated.
            If greater than zero, then the "nu_x"th-order derivative in the x-direction is
            interpolated.
        nu_y : int
            If zero, then the function in x-direction is interpolated.
            If greater than zero, then the "nu_x"th-order derivative in the y-direction is
            interpolated.
        nu_z : int
            If zero, then the function in x-direction is interpolated.
            If greater than zero, then the "nu_x"th-order derivative in the z-direction is
            interpolated.

        Returns
        -------
        float :
            Returns the interpolation of a function (or of it's derivatives) at a point.

        """
        if use_log:
            func_values = np.log(func_values)

        # Interpolate the Z-Axis.
        def z_spline(z, x_index, y_index, nu_z=nu_z):
            # x_index, y_index is assumed to be in the grid while z is not assumed.
            # Get smallest and largest index for selecting func vals on this specific z-slice.
            # The `1` and `self.num_puts[2] - 2` is needed because I don't want the boundary.
            small_index = self.coordinates_to_index((x_index, y_index, 1))
            large_index = self.coordinates_to_index(
                (x_index, y_index, self.shape[2] - 2)
            )
            val = CubicSpline(
                self.points[small_index:large_index, 2],
                func_values[small_index:large_index],
            )(z, nu_z)

            return val

        # Interpolate the Y-Axis from a list of interpolated points on the z-axis.
        def y_splines(y, x_index, z, nu_y=nu_y):
            # The `1` and `self.num_puts[1] - 2` is needed because I don't want the boundary.
            # Assumes x_index is in the grid while y, z may not be.
            val = CubicSpline(
                self.points[np.arange(1, self.shape[1] - 2) * self.shape[2], 1],
                [
                    z_spline(z, x_index, y_index, nu_z)
                    for y_index in range(1, self.shape[1] - 2)
                ],
            )(y, nu_y)

            return val

        # Interpolate the point (x, y, z) from a list of interpolated points on x,y-axis.
        def x_spline(x, y, z, nu_x):
            val = CubicSpline(
                self.points[np.arange(1, self.shape[0] - 2) * self.shape[1] * self.shape[2], 0],
                [y_splines(y, x_index, z, nu_y) for x_index in range(1, self.shape[0] - 2)],
            )(x, nu_x)

            return val

        if use_log:
            # All derivatives require the interpolation of f at (x,y,z)
            interpolated = np.exp(self.interpolate_function(real_pt, func_values, use_log=False,
                                                            nu_x=0, nu_y=0, nu_z=0))
            # Only consider taking the derivative in only one direction
            one_var_deriv = sum([nu_x == 0, nu_y == 0, nu_z == 0]) == 2

            # Return the derivative of f = exp(log(f)).
            if (nu_x, nu_y, nu_z) == (0, 0, 0):
                return interpolated
            elif one_var_deriv:
                # Taking the k-th derivative wrt to only one variable (x, y, z)
                # Interpolate d^k ln(f) d"deriv_var" for all k from 1 to "deriv_var"
                if nu_x > 0:
                    derivs = [
                        self.interpolate_function(real_pt, func_values, use_log=False,
                                                  nu_x=i, nu_y=0, nu_z=0)
                        for i in range(1, nu_x + 1)
                    ]
                    deriv_var = nu_x
                elif nu_y > 0:
                    derivs = [
                        self.interpolate_function(real_pt, func_values, use_log=False,
                                                  nu_x=0, nu_y=i, nu_z=0)
                        for i in range(1, nu_y + 1)
                    ]
                    deriv_var = nu_y
                else:
                    derivs = [
                        self.interpolate_function(real_pt, func_values, use_log=False,
                                                  nu_x=0, nu_y=0, nu_z=i)
                        for i in range(1, nu_z + 1)
                    ]
                    deriv_var = nu_z
                # Sympy symbols and dictionary of symbols pointing to the derivative values
                sympy_symbols = symbols("x:" + str(deriv_var))
                symbol_values = {"x" + str(i): float(derivs[i]) for i in range(0, deriv_var)}
                return interpolated * float(sum(
                    [
                        bell(deriv_var, i, sympy_symbols).evalf(subs=symbol_values)
                        for i in range(1, deriv_var + 1)
                    ]
                )
                )
            else:
                raise NotImplementedError("Taking mixed derivative while appling the logarithm is "
                                          "not supported.")
        # Normal interpolation without logarithm.
        interpolated = x_spline(real_pt[0], real_pt[1], real_pt[2], nu_x)
        return interpolated

    def coordinates_to_index(self, indices):
        r"""
        Convert coordinates to Index, ie (i, j, k) to an integer m corresponding to grid point.

        Parameters
        ----------
        indices : (int, int, int)
            The ith, jth, kth position of the grid point.

        Returns
        -------
        index : int
            Index of the grid point.

        """
        n_1d, n_2d = self.shape[2], self.shape[1] * self.shape[2]
        index = n_2d * indices[0] + n_1d * indices[1] + indices[2]
        return index

    def index_to_coordinates(self, index):
        r"""
        Convert Index to coordinates, ie integer m to (i, j, k) position of the Cubic Grid.

        Cubic Grid has shape (N_x, N_y, N_z) where N_x is the number of points
        in the x-direction, etc.  Then 0 <= i <= N_x - 1, 0 <= j <= N_y - 1, etc.

        Parameters
        ----------
        index : int
            Index of the grid point.

        Returns
        -------
        indices : (int, int, int)
            The ith, jth, kth position of the grid point.

        """
        assert index >= 0, "Index should be positive. %r" % index
        n_1d, n_2d = self.shape[2], self.shape[1] * self.shape[2]
        i = index // n_2d
        j = (index - n_2d * i) // n_1d
        k = index - n_2d * i - n_1d * j
        return i, j, k


class Tensor1DGrids(_CubicGrid):
    def __init__(self, oned_x, oned_y, oned_z):
        r"""Construct Tensor1DGrids: Tensor product of three one-dimensional grids."""
        if not isinstance(oned_x, OneDGrid):
            raise TypeError("Argument 'oned_x' ({0}) should be of type `OneDGrid`.".
                            format(type(oned_x)))
        if not isinstance(oned_y, OneDGrid):
            raise TypeError("Argument 'oned_y' ({0}) should be of type 'OneDGrid'".
                            format(type(oned_y)))
        if not isinstance(oned_z, OneDGrid):
            raise TypeError("Argument 'oned_z' ({0}) should be of type 'OneDGrid'".
                            format(type(oned_z)))

        shape = (oned_x.size, oned_y.size, oned_z.size)

        # Construct 3D set of points
        points = np.vstack(
            np.meshgrid(oned_x.points, oned_y.points, oned_z.points,indexing="ij",)
        ).reshape(3,-1).T
        weights = np.kron(
            np.kron(oned_x.weights, oned_y.weights), oned_z.weights
        )
        super().__init__(points, weights, shape)

    @property
    def origin(self):
        r"""Return the origin of the coordinate system."""
        # Bottom, Left-Most, Down-most point of the Cubic Grid in [-1, 1]^3.
        return self.points[0]


class UniformCubicGrid(_CubicGrid):
    def __init__(self, origin, axes, shape, weight_type="Trapezoid"):
        """
        Construct the UniformCubicGrid object:

        Grid whose points in each (x, y, z) direction has a constant step-size/evenly spaced.

        Parameters
        ----------
        origin : np.ndarray(3,)
            Cartesian coordinates of the cubic grid origin.
        axes : np.ndarray(3, 3)
            The three vectors, stored as rows of axes array,
            defining the Cartesian coordinate system used to build the
            cubic grid, i.e. the directions of the "(x,y,z)"-axis
            whose norm tells us the distance between points in that direction.
        shape : np.ndarray(3,)
            Number of grid points along "axes" axis.
        weight_type : str
            The integration weighting scheme. Can be either:
            Rectangle :
                The weights are the standard Riemannian weights.
            Trapezoid :
                Equivalent to rectangle rule with the assumption function is zero on the boundaries.
            Fourier1 :
                Assumes function can be expanded in a Fourier series, and then use Gaussian
                quadrature. Assumes the function is zero at the boundary of the cube.
            Fourier2 :
                Alternative weights based on Fourier series. Assumes the function is zero at the
                boundary of the cube.
            Alternative :
                This does not assume function is zero at the boundary.

        """
        if not isinstance(origin, np.ndarray):
            raise TypeError("origin should be a numpy array.")
        if not isinstance(axes, np.ndarray):
            raise TypeError("axes should be a numpy array.")
        if not isinstance(shape, np.ndarray):
            raise TypeError("shape should be a numpy array.")
        if origin.size != 3 or shape.size != 3:
            raise ValueError("Origin and shape should have size 3.")
        if axes.shape != (3, 3):
            raise ValueError("Axes {0} should be a three by three array.".format(axes.shape))
        if np.any(shape <= 0):
            raise ValueError("In each coordinate, shape should be positive.")

        self._axes = axes
        self._origin = origin
        # Make an array to store coordinates of grid points
        self._points = np.zeros((shape[0] * shape[1] * shape[2], 3))
        coords = np.array(
            np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        )
        coords = np.swapaxes(coords, 1, 2)
        coords = coords.reshape(3, -1)
        points = coords.T.dot(self._axes) + origin

        # assign the weights
        weights = self._choose_weight_scheme(weight_type, shape)
        super().__init__(points, weights, shape)

    @property
    def axes(self):
        """Array with axes of the cube.

        These give the direction of the each of the "(x,y,z") axes
        of the cubic grid.
        """
        return self._axes

    @property
    def origin(self):
        """Cartesian coordinate of the origin of the cubic grid."""
        return self._origin

    def __len__(self):
        """Return the number of grid points."""
        return self._points.size

    def _calculate_volume(self, shape):
        r"""Returns the volume of the Uniform Cubic Grid."""
        volume = np.linalg.norm(shape[0] * self.axes[0])
        volume *= np.linalg.norm(shape[1] * self.axes[1])
        volume *= np.linalg.norm(shape[2] * self.axes[2])
        return volume

    def _calculate_alternative_volume(self, shape):
        volume = self._calculate_volume(shape)
        factor = np.prod((shape - 1) / shape)
        return volume * factor

    def _choose_weight_scheme(self, type, shape):
        # Choose different weighting schemes.
        if type == "Rectangle":
            volume = self._calculate_volume(shape)
            numpnt = 1.0 * np.prod(shape)
            weights = np.full(np.prod(shape), volume / numpnt)
            return weights
        elif type == "Trapezoid":
            volume = self._calculate_volume(shape)
            numpnt = (shape[0] + 1.0) * (shape[1] + 1.0) * (shape[2] + 1.0)
            weights = np.full(np.prod(shape), volume / numpnt)
            return weights
        elif type == "Fourier1":
            volume = self._calculate_volume(shape)
            # “Gaussian” quadrature rule for Fourier series.
            numpnt = (shape[0] + 1.0) * (shape[1] + 1.0) * (shape[2] + 1.0)
            weight = np.ones(shape) * (8 * volume / numpnt)

            # Calculate the weights in the x-direction.
            grid_x = np.arange(1, shape[0] + 1)
            grid_x_2d = np.outer(grid_x, grid_x)
            sin_x = np.sin(grid_x_2d * np.pi / (shape[0] + 1.0))
            weight_x = np.einsum("ij,j->i", sin_x, (1 - np.cos(grid_x * np.pi)) / (grid_x * np.pi))
            weight = np.einsum("ijk,i->ijk", weight, weight_x)

            # Calculate the weights in the y-direction.
            grid_y = np.arange(1, shape[1] + 1)
            grid_y_2d = np.outer(grid_y, grid_y)
            sin_y = np.sin(grid_y_2d * np.pi / (shape[1] + 1.0))
            weight_y = np.einsum("ij,j->i", sin_y, (1 - np.cos(grid_y * np.pi)) / (grid_y * np.pi))
            weight = np.einsum("ijk,j->ijk", weight, weight_y)

            # Calculate the weights in the z-direction.
            grid_z = np.arange(1, shape[2] + 1)
            grid_z_2d = np.outer(grid_z, grid_z)
            sin_z = np.sin(grid_z_2d * np.pi / (shape[2] + 1.0))
            weight_z = np.einsum("ij,j->i", sin_z, (1 - np.cos(grid_z * np.pi)) / (grid_z * np.pi))
            weight = np.einsum("ijk,k->ijk", weight, weight_z)

            return np.ravel(weight)
        elif type == "Alternative":
            alt_volume = self._calculate_alternative_volume(shape)
            return np.ones(np.prod(shape)) * alt_volume
        elif type == "Fourier2":
            alt_volume = self._calculate_alternative_volume(shape)

            # Calculate weights in x-direction
            grid_x = np.arange(1, shape[0] + 1)
            grid_p = np.arange(1, shape[0])
            grid_2d = np.outer((2.0 * grid_x - 1) / shape[0], grid_p)
            weight_x = 4.0 * np.einsum(
                "ij,j->i",
                np.sin(grid_2d * np.pi),
                np.sin(grid_p * np.pi / 2.0)**2.0 / grid_p
            ) / (np.pi * shape[0])
            weight_x += 2.0 * np.sin(np.pi * shape[0] / 2.0)**2.0 * \
                np.sin((grid_x - 0.5) * np.pi) / (shape[0]**2.0 * np.pi)

            # Calculate weights in y-direction
            grid_y = np.arange(1, shape[1] + 1)
            grid_p = np.arange(1, shape[1])
            grid_2d = np.outer((2.0 * grid_y - 1) / shape[1], grid_p)
            weight_y = 4.0 * np.einsum(
                "ij,j->i",
                np.sin(grid_2d * np.pi),
                np.sin(grid_p * np.pi / 2.0)**2.0 / grid_p
            ) / (np.pi * shape[1])
            weight_y += 2.0 * np.sin(np.pi * shape[1] / 2.0)**2.0 * \
                np.sin((grid_y - 0.5) * np.pi) / (shape[1]**2.0 * np.pi)

            # Calculate weights in z-direction
            grid_z = np.arange(1, shape[2] + 1)
            grid_p = np.arange(1, shape[2])
            grid_2d = np.outer((2.0 * grid_z - 1) / shape[2], grid_p)
            weight_z = 4.0 * np.einsum(
                "ij,j->i",
                np.sin(grid_2d * np.pi),
                np.sin(grid_p * np.pi / 2.0)**2.0 / grid_p
            ) / (np.pi * shape[2])
            weight_z += 2.0 * np.sin(np.pi * shape[2] / 2.0)**2.0 * \
                np.sin((grid_z - 0.5) * np.pi) / (shape[2]**2.0 * np.pi)

            weight = np.ones(shape)
            weight = np.einsum("ijk,i,j,k->ijk", weight, weight_x, weight_y, weight_z) * alt_volume
            return np.ravel(weight)
        else:
            raise ValueError("The weighting type {0} parameter is not known.".format(type))

    def closest_point(self, point, which="closest"):
        r"""
        Return closest index of the grid point to a point.

        Imagine a point inside a small sub-cube. If `closest` is selected, it will
        pick the corner in the sub-cube that is closest to that point.
        if `origin` is selected, it will pick the corner that is the bottom,
        left-most, down-most in the sub-cube.

        Parameters
        ----------
        point : np.ndarray(3,)
            Point in :math:`[-1,1]^3`.
        which : str
            If "closest", returns the closest index of the grid point.
            If "origin", return the left-most, down-most closest index of the grid point.

        Returns
        -------
        index : int
            Index of the point in `points` closest to the grid point.

        """
        # I'm not entirely certain that this method will work with non-orthogonal axes.
        #    Added this just in case, cause I know it will work with orthogonal axes.
        if not np.count_nonzero(self.axes - np.diag(np.diagonal(self.axes))) == 0:
            raise ValueError("Finding closest point only works when the 'axes' attribute"
                             "is a diagonal matrix.")

        # Calculate step-size of the cube.
        step_size_x = np.linalg.norm(self.axes[0] - self.origin)
        step_size_y = np.linalg.norm(self.axes[1] - self.origin)
        step_size_z = np.linalg.norm(self.axes[2] - self.origin)
        step_sizes = np.array([step_size_x, step_size_y, step_size_z])

        coord = np.array([(point[i] - self.origin[i]) / step_sizes[i] for i in range(3)])

        if which == "origin":
            # Round to smallest integer.
            coord = np.floor(coord)
        elif which == "closest":
            # Round to nearest integer.
            coord = np.rint(coord)
        else:
            raise ValueError("`which` parameter was not the standard options.")

        # Convert indices (i, j, k) into index.
        index = self.coordinates_to_index(coord)

        return index
