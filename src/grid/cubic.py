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
r"""Hyper Rectangular Grid In Either Two or Three Dimensions."""

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from sympy import symbols
from sympy.functions.combinatorial.numbers import bell

from grid.basegrid import Grid, OneDGrid


class _HyperRectangleGrid(Grid):
    def __init__(self, points, weights, shape):
        r"""Construct the _HyperRectangleGrid class.

        Hyper Rectangle grid is restricted to either two-dimensions and three-dimensions and
        is defined to be a grid whose point is specified by (i, j) or (i, j, k) integer indices.

        Parameters
        ----------
        points : np.ndarray, shape (N, D)
            The :math:`D` Cartesian coordinates of :math:`N` grid points, where :math:`D`
            is either two or three.
        weights : np.ndarray, shape (N,)
            The integration weights corresponding to each :math:`N` grid point.
        shape : np.ndarray, shape (D,)
            The number of grid points in the each of the :math:`D` directions.

        """
        if len(shape) not in [2, 3]:
            raise ValueError(
                f"Argument shape should have length two or three; got length {len(shape)}."
            )
        if np.any(np.array(shape) <= 1.0):
            raise ValueError(
                f"Argument shape should be greater than one in all directions {shape}."
            )
        if np.prod(shape) != points.shape[0]:
            raise ValueError(
                f"The product of shape elements {np.prod(shape)} should match the number of grid "
                f"points {points.shape[0]}."
            )
        if len(shape) != points.shape[1]:
            raise ValueError(
                f"The dimension of the shape/grid {len(shape)} should match the dimension of the "
                f"points {points.shape[1]}."
            )
        self._shape = shape
        super().__init__(points, weights)

    @property
    def shape(self):
        r"""Return number of grid points in the :math:`(x, y)` or :math:`(x, y, z)` directions."""
        return self._shape

    @property
    def ndim(self):
        r"""Return the dimension of the grid."""
        return len(self._shape)

    def get_points_along_axes(self):
        r"""Return the points along each axes.

        Returns
        -------
        np.ndarray(M_x,), np.ndarray(M_y,) np.ndarray(M_z) :
            The points in the x, y, if in three dimensions, z axes respectively,
            where :math:`M_i` is the number of points in the ith-direction.

        """
        # Need to obtain the points in the x-, y-, z-axis separately. In order to do so, need to
        # assume the points have a specific structure.
        if self.ndim == 3:
            z = self.points[: self.shape[2], 2]
            coords_y = [self.coordinates_to_index((0, j, 0)) for j in range(self.shape[1])]
            y = self.points[coords_y, 1]
            coords_x = [self.coordinates_to_index((j, 0, 0)) for j in range(self.shape[0])]
            x = self.points[coords_x, 0]
            return x, y, z
        # Case of two-dimensions
        y = self.points[: self.shape[1], 1]
        coords_x = [self.coordinates_to_index((j, 0)) for j in range(self.shape[0])]
        x = self.points[coords_x, 0]
        return x, y

    def interpolate(self, points, values, use_log=False, nu_x=0, nu_y=0, nu_z=0, method="cubic"):
        r"""Interpolate function and its derivatives on cubic grid.

        Only implemented in three-dimensions.

        Parameters
        ----------
        points : np.ndarray, shape (M, 3)
            The 3D Cartesian coordinates of :math:`M` points in :math:`\mathbb{R}^3` for which
            the interpolant (i.e., the interpolated function) is evaluated. If
            method="cubic", then M must be equal to one.
        values : np.ndarray, shape (N,)
            Function values at each of the :math:`N` grid points.
        use_log : bool, optional
            If True, the logarithm of the function values are interpolated.
            Can only be used for interpolating derivatives when the derivative is not a
            mixed derivative.
        nu_x : int, optional
            If zero, then the function in x-direction is interpolated.
            If greater than zero, then the "nu_x"th-order derivative in the x-direction is
            interpolated.
        nu_y : int, optional
            If zero, then the function in y-direction is interpolated.
            If greater than zero, then the "nu_y"th-order derivative in the y-direction is
            interpolated.
        nu_z : int, optional
            If zero, then the function in z-direction is interpolated.
            If greater than zero, then the "nu_z"th-order derivative in the z-direction is
            interpolated.
        method : str, optional
            The method of interpolation to perform. Supported are "cubic" (most accurate but
            computationally expensive), "linear", or "nearest" (least accurate but cheap
            computationally). The last two methods use SciPy's RegularGridInterpolator function.

        Returns
        -------
        float :
            The interpolation of a function (or of it's derivatives) at a :math:`M` point.

        """
        if method not in ["cubic", "linear", "nearest"]:
            raise ValueError(
                f"Argument method should be either cubic, linear, or nearest , got {method}"
            )
        if self.ndim != 3:
            raise NotImplementedError(
                f"Interpolation only works for three dimension, got ndim={self.ndim}"
            )
        if values.shape[0] != np.prod(self.shape):
            raise ValueError(
                f"Number of function values {values.shape[0]} does not match number of "
                f"grid points {np.prod(self.shape)}."
            )

        if use_log:
            values = np.log(values)

        # Use scipy if linear and nearest is requested and raise error if it's not cubic.
        if method in ["linear", "nearest"]:
            x, y, z = self.get_points_along_axes()
            values = values.reshape(self.shape)
            interpolate = RegularGridInterpolator((x, y, z), values, method=method)
            return interpolate(points)

        # Interpolate the Z-Axis.
        def z_spline(z, x_index, y_index, nu_z=nu_z):
            # x_index, y_index is assumed to be in the grid while z is not assumed.
            # Get smallest and largest index for selecting func vals on this specific z-slice.
            # The `1` and `self.num_puts[2] - 2` is needed because I don't want the boundary.
            small_index = self.coordinates_to_index((x_index, y_index, 1))
            large_index = self.coordinates_to_index((x_index, y_index, self.shape[2] - 2))
            val = CubicSpline(
                self.points[small_index:large_index, 2],
                values[small_index:large_index],
            )(z, nu_z)
            return val

        # Interpolate the Y-Axis from a list of interpolated points on the z-axis.
        def y_splines(y, x_index, z, nu_y=nu_y):
            # The `1` and `self.num_puts[1] - 2` is needed because I don't want the boundary.
            # Assumes x_index is in the grid while y, z may not be.
            val = CubicSpline(
                self.points[np.arange(1, self.shape[1] - 2) * self.shape[2], 1],
                [z_spline(z, x_index, y_index, nu_z) for y_index in range(1, self.shape[1] - 2)],
            )(y, nu_y)
            # Trying to vectorize over z-axis and y-axis, this computes the interpolation for every
            #      pair of (y,z) pair when we're only interested in the diagonal. This is faster
            #      than running a list comprehensive every each point in y, but memory storage
            #      is larger.
            return np.diag(val)

        # Interpolate the point (x, y, z) from a list of interpolated points on x,y-axis.
        def x_spline(x, y, z, nu_x):
            val = CubicSpline(
                self.points[np.arange(1, self.shape[0] - 2) * self.shape[1] * self.shape[2], 0],
                [y_splines(y, x_index, z, nu_y) for x_index in range(1, self.shape[0] - 2)],
            )(x, nu_x)
            # Trying to vectorize over x-axis, this computes the interpolation for every
            #      pair of (x, (y, z)) pair when we're only interested in the diagonal. This is
            #      faster than running a list comprehensive/for loop every each point in y,
            #      but the memory storage is larger.
            return np.diag(val)

        if use_log:
            # All derivatives require the interpolation of f at (x,y,z)
            interpolated = np.exp(
                self.interpolate(points, values, use_log=False, nu_x=0, nu_y=0, nu_z=0)
            )
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
                        self.interpolate(points, values, use_log=False, nu_x=i, nu_y=0, nu_z=0)
                        for i in range(1, nu_x + 1)
                    ]
                    deriv_var = nu_x
                elif nu_y > 0:
                    derivs = [
                        self.interpolate(points, values, use_log=False, nu_x=0, nu_y=i, nu_z=0)
                        for i in range(1, nu_y + 1)
                    ]
                    deriv_var = nu_y
                else:
                    derivs = [
                        self.interpolate(points, values, use_log=False, nu_x=0, nu_y=0, nu_z=i)
                        for i in range(1, nu_z + 1)
                    ]
                    deriv_var = nu_z
                # Sympy symbols and dictionary of symbols pointing to the derivative values
                sympy_symbols = symbols("x:" + str(deriv_var))
                # This interpolation only works on singular points and thus we can safely
                #   assuming derivs=[[0], [1]] is a list of list of single points.
                bell_derivs = []
                for j in range(0, len(points)):
                    symbol_values = {"x" + str(i): float(derivs[i][j]) for i in range(0, deriv_var)}
                    bell_derivs.append(
                        float(
                            sum(
                                [
                                    bell(deriv_var, i, sympy_symbols).evalf(subs=symbol_values)
                                    for i in range(1, deriv_var + 1)
                                ]
                            )
                        )
                    )
                return interpolated * np.array(bell_derivs)
            else:
                raise NotImplementedError(
                    "Taking mixed derivative while applying the logarithm is not supported."
                )
        # Normal interpolation without logarithm.
        interpolated = x_spline(points[:, 0], points[:, 1], points[:, 2], nu_x)
        return interpolated

    def coordinates_to_index(self, indices):
        r"""Convert (i, j) or (i, j, k) integer coordinates to the grid point index.

        Assumes the grid is ordered moving in the last-coordinate (with other coordinates fixed,
        e.g. k) followed by the next coordinate to the left (e.g. j), continuing to the first
        coordinate (e.g. i) (i.e., lexicographical ordering).

        Parameters
        ----------
        indices : (int, int) or (int, int, int)
            The :math:`i-th`, :math:`j-th`, (or :math:`k-th`) positions of the grid point.

        Returns
        -------
        index : int
            Index of the grid point.

        """
        if self.ndim == 3:
            n_1d, n_2d = self.shape[2], self.shape[1] * self.shape[2]
            # TODO Change this so that indices can be multi-dimensional
            index = n_2d * indices[0] + n_1d * indices[1] + indices[2]
            return index
        # Case of two-dimensions
        index = self.shape[1] * indices[0] + indices[1]
        return index

    def index_to_coordinates(self, index):
        r"""Convert grid point index to its (i, j) or (i, j, k) integer coordinates in the grid.

        For 3D grid it has a shape of :math:`(N_x, N_y, N_z)` denoting the number of points in
        :math:`x`, :math:`y`, and :math:`z` directions. So, each grid point has a :math:`(i, j, k)`
        integer coordinate where :math:`0 <= i <= N_x - 1`, :math:`0 <= j <= N_y - 1`,
        and :math:`0 <= k <= N_z - 1`.  Two-dimensional case similarly follows. Assumes
        the grid enumerates in the last coordinate first (with others fixed), following the
        next coordinate to the left (i.e., lexicographical ordering).

        Parameters
        ----------
        index : int
            Index of the grid point.

        Returns
        -------
        indices : (int, int) or (int, int, int)
            The corresponding :math:`(i, j)` or :math:`(i, j, k)` integer coordinates in the grid.

        """
        if not index >= 0:
            raise ValueError(f"Argument index should be a positive integer, got {index}")
        if self.ndim == 3:
            n_1d, n_2d = self.shape[2], self.shape[1] * self.shape[2]
            i = index // n_2d
            j = (index - n_2d * i) // n_1d
            k = index - n_2d * i - n_1d * j
            return i, j, k
        # Case of two-dimensions
        i = index // self.shape[1]
        j = index - self.shape[1] * i
        return i, j


class Tensor1DGrids(_HyperRectangleGrid):
    r"""Tensor product of two/three one-dimensional grids."""

    def __init__(self, oned_x, oned_y, oned_z=None):
        r"""Construct Tensor1DGrids by tensor product of two (or three) one-dimensional grids.

        Parameters
        ----------
        oned_x : OneDGrid
            One-dimensional grid representing the grids along x-axis.
        oned_y : OneDGrid
            One-dimensional grid representing the grids along y-axis.
        oned_z : OneDGrid, optional
            One-dimensional grid representing the grids along z-axis.

        """
        if not isinstance(oned_x, OneDGrid):
            raise TypeError(
                f"Argument oned_x should be an instance of `OneDGrid`, got {type(oned_x)}"
            )
        if not isinstance(oned_y, OneDGrid):
            raise TypeError(
                f"Argument oned_y should be an instance of `OneDGrid`, got {type(oned_y)}"
            )
        if not isinstance(oned_z, (OneDGrid, type(None))):
            raise TypeError(
                f"Argument oned_z should be an instance of `OneDGrid`, got {type(oned_z)}"
            )

        if oned_z is not None:
            # number of points in x, y, and z direction of the tensor product grid
            shape = (oned_x.size, oned_y.size, oned_z.size)
            # Construct 3D set of points and weights
            # Note: points move in z-axis first (x,y-fixed) then y-axis then x-axis
            # (i.e., lexicographical ordering)
            points = (
                np.vstack(
                    np.meshgrid(
                        oned_x.points,
                        oned_y.points,
                        oned_z.points,
                        indexing="ij",
                    )
                )
                .reshape(3, -1)
                .T
            )
            weights = np.kron(np.kron(oned_x.weights, oned_y.weights), oned_z.weights)
        else:
            # number of points in x and y directions of the two-dimensional grid
            shape = (oned_x.size, oned_y.size)
            # Construct 2D set of points and weights
            points = (
                np.vstack(
                    np.meshgrid(
                        oned_x.points,
                        oned_y.points,
                        indexing="ij",
                    )
                )
                .reshape(2, -1)
                .T
            )
            weights = np.kron(oned_x.weights, oned_y.weights)
        super().__init__(points, weights, shape)

    @property
    def origin(self):
        r"""Cartesian coordinates of the grid origin."""
        # Bottom, Left-Most, Down-most point of the hyper-rectangular grid.
        return self.points[0]

    def save(self, filename):
        r"""
        Save tensor product of three one-dimensional grids attributes as a npz file.

        Parameters
        ----------
        filename: str
           The path/name of the .npz file.

        """
        dict_save = {
            "points": self.points,
            "weights": self.weights,
            "origin": self.origin,
        }
        np.savez(filename, **dict_save)


class UniformGrid(_HyperRectangleGrid):
    r"""Uniform grid (a.k.a. rectilinear grid) with evenly-spaced points in each axes."""

    def __init__(self, origin, axes, shape, weight="Trapezoid"):
        r"""Construct the UniformGrid object in either two or three dimensions.

        Three dimensions is presented below, but similarly will work in two-dimensions.
        Grid whose points in each (x, y, z) direction has a constant step-size/evenly
        spaced. Given a origin :math:`\mathbf{o} = (o_x, o_y, o_z)` and three directions forming
        the axes :math:`\mathbf{a_1}, \mathbf{a_2}, \mathbf{a_3}` with shape
        :math:`(M_x, M_y, M_z)`, then the :math:`(i, j, k)-\text{th}` point of the grid are:

        .. math::
            x_i &= o_x + i \mathbf{a_1} \quad 0 \leq i \leq M_x \\
            y_i &= o_y + j \mathbf{a_2} \quad 0 \leq j \leq M_y \\
            z_i &= o_z + k \mathbf{a_3} \quad 0 \leq k \leq M_z

        The grid enumerates through the z-axis first, then y-axis then x-axis (i.e.,
        lexicographical ordering).

        Parameters
        ----------
        origin : np.ndarray, shape (D,)
            Cartesian coordinates of the grid origin in either two or three dimensions :math:`D`.
        axes : np.ndarray, shape (D, D)
            The :math:`D` vectors, stored as rows of axes array, whose rows
            define the Cartesian coordinate system used to build the
            cubic grid, i.e. the directions of the "(x,y)\(x,y,z)"-axis
            whose norm tells us the distance between points in that direction.
        shape : np.ndarray, shape (D,)
            Number of grid points along each axis.
        weight : str
            String indicating weighting function. This can be:

            Rectangle :
                The weights are the standard Riemannian weights,

                .. math::
                    \begin{align*}
                        w_{ij} &= \frac{V}{M_x \cdot M_y} \tag{Two-Dimensions} \\
                         w_{ijk} &= \frac{V}{M_x\cdot M_y \cdot M_z}  \tag{Three-Dimensions}
                    \end{align*}

                where :math:`V` is the volume or area of the uniform grid.

            Trapezoid :
                Equivalent to rectangle rule with the assumption function is zero on the boundaries.

                 .. math::
                    \begin{align*}
                        w_{ij} &= \frac{V}{(M_x + 1) \cdot (M_y + 1)}  \tag{Two-Dimensions} \\
                        w_{ijk} &= \frac{V}{(M_x + 1) \cdot (M_y + 1) \cdot (M_z + 1)}
                        \tag{Three-Dimensions}
                    \end{align*}
                where :math:`V` is the volume or area of the uniform grid.

            Fourier1 :
                Assumes function can be expanded in a Fourier series, and then use Gaussian
                quadrature. Assumes the function is zero at the boundary of the cube.
                In three-dimensions it is

                .. math::
                    w_{ijk} = \frac{2^3}{(M_x + 1) \cdot (M_y + 1) \cdot (M_z + 1)} \bigg[
                    \bigg(\sum_{p=1}^{M_x} \frac{\sin(ip \pi/(M_x + 1)) (1 - \cos(p\pi)}{p\pi}
                         \bigg)
                         \bigg(\sum_{p=1}^{M_y} \frac{\sin(jp \pi/(M_y + 1)) (1 - \cos(p\pi)}{p\pi}
                         \bigg)
                         \bigg(\sum_{p=1}^{M_z} \frac{\sin(kp \pi/(M_z + 1)) (1 - \cos(p\pi)}{p\pi}
                         \bigg)
                    \bigg]

                whereas in two-dimensions it is

                .. math::
                    w_{ij} = \frac{2^3}{(M_x + 1) \cdot (M_y + 1)} \bigg[
                    \bigg(\sum_{p=1}^{M_x} \frac{\sin(ip \pi/(M_x + 1)) (1 - \cos(p\pi)}{p\pi}
                         \bigg)
                         \bigg(\sum_{p=1}^{M_y} \frac{\sin(jp \pi/(M_y + 1)) (1 - \cos(p\pi)}{p\pi}
                         \bigg)
                    \bigg]

            Fourier2 :
                Alternative weights based on Fourier series. Assumes the function is zero at the
                boundary of the cube.

                .. math::
                    w_{ijk} &= V^\prime \cdot w_i w_j w_k, \\
                    w_i &= \bigg(\frac{2\sin((j - 0.5)\pi) \sin^2(M_x\pi/2)}{M_x^2 \pi} +
                     \frac{4}{M_x \pi} \sum_{p=1}^{M_x - 1}
                     \frac{\sin((2j-1)p\pi /n_x) sin^2(p \pi)}{\pi} \bigg)

            Alternative :
                This does not assume function is zero at the boundary.

            .. math::
                \begin{align*}
                    w_{ij} &= V \cdot \frac{M_x - 1}{M_x} \frac{M_y - 1}{M_y}
                    \tag{Two-Dimensions}\\
                    w_{ijk} &= V \cdot \frac{M_x - 1}{M_x} \frac{M_y - 1}{M_y} \frac{M_z - 1}{M_z}
                    \tag{Three-Dimensions}
                \end{align*}

        """
        if not isinstance(origin, np.ndarray):
            raise TypeError(f"Argument origin should be a numpy array, got {type(origin)}")
        if not isinstance(axes, np.ndarray):
            raise TypeError(f"Argument axes should be a numpy array, got {type(axes)}")
        if not isinstance(shape, np.ndarray):
            raise TypeError(f"Argument shape should be a numpy array, got {type(shape)}")
        if origin.size not in (3, 2):
            raise ValueError(f"Arguments origin should have size 2 or 3, got {origin.shape}")
        if shape.size != origin.size:
            raise ValueError(f"Shape {shape.size} should be the same size {origin.size}.")
        if axes.shape != (origin.size, origin.size):
            raise ValueError(
                f"Axes {axes.shape} should be the same shape {origin.size}, {origin.size}."
            )
        if np.abs(np.linalg.det(axes)) < 1e-10:
            raise ValueError(
                f"The axes are not linearly independent, got det(axes)={np.linalg.det(axes)}"
            )
        if np.any(shape <= 0):
            raise ValueError(
                f"Number of points in each direction should be positive, got shape={list(shape)}"
            )

        self._axes = axes
        self._origin = origin
        dim = self._origin.size
        # Make an array to store coordinates of grid points
        self._points = np.zeros((np.prod(shape), dim))
        if dim == 3:
            coords = np.array(
                np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
            )
            coords = np.swapaxes(coords, 1, 2)
            coords = coords.reshape(3, -1)
        else:
            coords = np.array(np.meshgrid(np.arange(shape[0]), np.arange(shape[1])))
            coords = coords.reshape(2, -1, order="F")
        points = coords.T.dot(self._axes) + origin
        # assign the weights
        weights = self._choose_weight_scheme(weight, shape)
        super().__init__(points, weights, shape)

    @classmethod
    def from_molecule(
        cls,
        atcorenums,
        atcoords,
        spacing=0.2,
        extension=5.0,
        rotate=True,
        weight="Trapezoid",
    ):
        r"""
        Construct a uniform grid given the molecular pseudo-numbers and coordinates.

        Parameters
        ----------
        atcorenums : np.ndarray, shape (M,)
            Pseudo-number of :math:`M` atoms in the molecule.
        atcoords : np.ndarray, shape (M, 3)
            Cartesian coordinates of :math:`M` atoms in the molecule.
        spacing : float, optional
            Increment between grid points along :math:`x`, :math:`y`, and :math:`z` direction.
        extension : float, optional
            The extension of the length of the cube on each side of the molecule.
        rotate : bool, optional
            When True, the molecule is rotated so the axes of the cube file are
            aligned with the principle axes of rotation of the molecule.
            If False, generates axes based on the x,y,z-axis and the spacing parameter, and
            the origin is defined by the maximum/minimum of the atomic coordinates.
        weight : str, optional
            String indicating weighting function. Denoting the volume/area of the uniform grid
            by :math:`V`, the weighting function can be:

            Rectangle :
                The weights are the standard Riemannian weights,

                .. math::
                    w_{ijk} = \frac{V}{M_x\cdot M_y \cdot M_z}

            Trapezoid :
                Equivalent to rectangle rule with the assumption function is zero on the boundaries.

                 .. math::
                    w_{ijk} = \frac{V}{(M_x + 1) \cdot (M_y + 1) \cdot (M_z + 1)}

            Fourier1 :
                Assumes function can be expanded in a Fourier series, and then use Gaussian
                quadrature. Assumes the function is zero at the boundary of the cube.

                .. math::
                    w_{ijk} = \frac{8}{(M_x + 1) \cdot (M_y + 1) \cdot (M_z + 1)} \bigg[
                    \bigg(\sum_{p=1}^{M_x} \frac{\sin(ip \pi/(M_x + 1)) (1 - \cos(p\pi)}{p\pi}
                         \bigg)
                         \bigg(\sum_{p=1}^{M_y} \frac{\sin(jp \pi/(M_y + 1)) (1 - \cos(p\pi)}{p\pi}
                         \bigg)
                         \bigg(\sum_{p=1}^{M_z} \frac{\sin(kp \pi/(M_z + 1)) (1 - \cos(p\pi)}{p\pi}
                         \bigg)
                    \bigg]

            Fourier2 :
                Alternative weights based on Fourier series. Assumes the function is zero at the
                boundary of the cube.

                .. math::
                    w_{ijk} &= V^\prime \cdot w_i w_j w_k,\\
                    w_i &= \bigg(\frac{2\sin((j - 0.5)\pi) \sin^2(M_x\pi/2)}{M_x^2 \pi} +
                     \frac{4}{M_x \pi} \sum_{p=1}^{M_x - 1}
                     \frac{\sin((2j-1)p\pi /n_x) sin^2(p \pi)}{\pi} \bigg)

            Alternative :
                This does not assume function is zero at the boundary.

            .. math::
                w_{ijk} = V \cdot \frac{M_x - 1}{M_x} \frac{M_y - 1}{M_y} \frac{M_z - 1}{M_z}

        """
        # calculate center of mass of the nuclear charges:
        totz = np.sum(atcorenums)
        com = np.dot(atcorenums, atcoords) / totz
        # Determine best axes and coordinates to calculate the lower and upper bound of grid.
        if rotate:
            # calculate moment of inertia tensor:
            itensor = np.zeros([3, 3])
            for i in range(atcorenums.shape[0]):
                xyz = atcoords[i] - com
                r = np.linalg.norm(xyz) ** 2.0
                tempitens = np.diag([r, r, r])
                tempitens -= np.outer(xyz.T, xyz)
                itensor += atcorenums[i] * tempitens
            # Eigenvectors define the new axes of the grid with spacing
            _, v = np.linalg.eigh(itensor)
            # Project the coordinates of atoms centered at the center of mass to the eigenvectors
            new_coordinates = np.dot((atcoords - com), v)
            axes = spacing * v
        else:
            # Just use the original coordinates
            new_coordinates = atcoords
            # Compute the unit vectors of the cubic grid's coordinate system
            axes = np.diag([spacing, spacing, spacing])

        # maximum and minimum value of x, y and z coordinates/grid.
        max_coordinate = np.amax(new_coordinates, axis=0)
        min_coordinate = np.amin(new_coordinates, axis=0)
        # Compute the required number of points along x, y, and z axis
        shape = (max_coordinate - min_coordinate + 2.0 * extension) / spacing
        # Add one to include the upper-bound as well.
        shape = np.ceil(shape)
        shape = np.array(shape, int)
        # Compute origin by taking the center of mass then subtracting the half of the number
        #    of points in the direction of the axes.
        origin = com - np.dot((0.5 * shape), axes)
        return cls(origin, axes, shape, weight)

    @classmethod
    def from_cube(cls, fname, weight="Trapezoid", return_data=False):
        r"""Initialize ``UniformGrid`` class based on the grid specifications of a cube file.

        Parameters
        ----------
        fname : str
            Cube file name with \*.cube extension.
        weight : str
            Scheme for computing the weights of the grid. See the acceptable values in the
            :func:`~UniformGrid.__init__` method.
        return_data : bool
            If False, only the grid is returned. If True a tuple with the grid and the cube data
            is returned. The cube data is a dictionary with the following keys:

            - ``atnums``: atomic numbers of the atoms in the molecule.
            - ``atcorenums``: Pseudo-number of :math:`M` atoms in the molecule.
            - ``atcoords``: Cartesian coordinates of :math:`M` atoms in the molecule.

        """
        fname = str(fname)
        if not fname.endswith(".cube"):
            raise ValueError("Argument fname should be a cube file with *.cube extension!")

        with open(fname) as f:
            # skip the title and second line
            f.readline()
            f.readline()

            def read_grid_line(line):
                """Read a number and (x, y, z) coordinate from the cube file line."""
                words = line.split()
                return (
                    int(words[0]),
                    np.array([float(words[1]), float(words[2]), float(words[3])], float),
                    # all coordinates in a cube file are in atomic units
                )

            # number of atoms and origin of the grid
            natom, origin = read_grid_line(f.readline())
            # number of grid points in A direction and step vector A, and so on
            shape0, axis0 = read_grid_line(f.readline())
            shape1, axis1 = read_grid_line(f.readline())
            shape2, axis2 = read_grid_line(f.readline())
            shape = np.array([shape0, shape1, shape2], int)
            axes = np.array([axis0, axis1, axis2])

            # if return_data=False, only grid is returned
            if not return_data:
                return cls(origin, axes, shape, weight)

            # otherwise, return the atomic numbers, coordinates, and the grid data as well
            def read_coordinate_line(line):
                """Read atomic number and (x, y, z) coordinate from the cube file line."""
                words = line.split()
                return (
                    int(words[0]),
                    float(words[1]),
                    np.array([float(words[2]), float(words[3]), float(words[4])], float),
                    # all coordinates in a cube file are in atomic units
                )

            numbers = np.zeros(natom, int)
            # get the core charges
            pseudo_numbers = np.zeros(natom, float)
            coordinates = np.zeros((natom, 3), float)
            for i in range(natom):
                numbers[i], pseudo_numbers[i], coordinates[i] = read_coordinate_line(f.readline())
                # If the pseudo_number field is zero, we assume that no effective core
                # potentials were used.
                if pseudo_numbers[i] == 0.0:
                    pseudo_numbers[i] = numbers[i]

            # load data stored in the cube file
            data = np.zeros(tuple(shape), float).ravel()
            counter = 0
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                words = np.array(line.split(), float)
                data[counter : counter + len(words)] = words
                counter += len(words)
            # store information from the cube file in a dictionary
            cube_data = {
                "atnums": numbers,
                "atcorenums": pseudo_numbers,
                "atcoords": coordinates,
                "data": data,
            }
        return cls(origin, axes, shape, weight), cube_data

    @property
    def axes(self):
        """Return the axes of the uniform grid."""
        return self._axes

    @property
    def origin(self):
        """Return the Cartesian coordinates of the uniform grid origin."""
        return self._origin

    def save(self, filename):
        r"""
        Save uniform cubic grid attributes as a npz file.

        Parameters
        ----------
        filename: str
           The path/name of the .npz file.

        """
        dict_save = {
            "points": self.points,
            "weights": self.weights,
            "origin": self.origin,
            "axes": self.axes,
        }
        np.savez(filename, **dict_save)

    def _calculate_volume(self, shape):
        r"""Return the volume of the Uniform Grid."""
        # Shape needs to be an argument, because I need to calculate the weights before
        #       initializing the _HyperRectangleGrid (where shape is set there).
        # Three-Dims: Volume of a parallelepiped spanned by a, b, c is  | (a x b) dot c|.
        if len(shape) == 3:
            volume = np.dot(
                np.cross(shape[0] * self.axes[0], shape[1] * self.axes[1]),
                shape[2] * self.axes[2],
            )
        else:
            # Two-Dims: Volume of a parallelogram is the absolute value of the determinant |a x b|
            volume = np.linalg.det(np.array([shape[0] * self.axes[0], shape[1] * self.axes[1]]))
        return np.abs(volume)

    def _calculate_alternative_volume(self, shape):
        # Used for calculating different integration weights.
        volume = self._calculate_volume(shape)
        factor = np.prod((shape - 1) / shape)
        return volume * factor

    def _choose_weight_scheme(self, weight, shape):
        # Choose different weighting schemes.
        if weight == "Rectangle":
            volume = self._calculate_volume(shape)
            numpnt = 1.0 * np.prod(shape)
            weights = np.full(np.prod(shape), volume / numpnt)
            return weights
        elif weight == "Trapezoid":
            volume = self._calculate_volume(shape)
            numpnt = np.prod(shape + 1.0)
            weights = np.full(np.prod(shape), volume / numpnt)
            return weights
        elif weight == "Fourier1":
            dim = len(shape)  # Dimension of the grid.
            volume = self._calculate_volume(shape)
            # “Gaussian” quadrature rule for Fourier series.
            numpnt = np.prod(shape + 1.0)
            weight = np.ones(shape) * (2**dim * volume / numpnt)

            def _fourier1(weight, shape, index, dim):
                # Return the fourier1 weights in the "index"-direction.
                grid_dir = np.arange(1, shape[index] + 1)
                grid_dir_2d = np.outer(grid_dir, grid_dir)
                sin_dir = np.sin(grid_dir_2d * np.pi / (shape[index] + 1.0))
                weight_dir = np.einsum(
                    "ij,j->i",
                    sin_dir,
                    (1 - np.cos(grid_dir * np.pi)) / (grid_dir * np.pi),
                )
                if dim == 3:
                    if index == 0:
                        return np.einsum("ijk,i->ijk", weight, weight_dir)
                    elif index == 1:
                        return np.einsum("ijk,j->ijk", weight, weight_dir)
                    elif index == 2:
                        return np.einsum("ijk,k->ijk", weight, weight_dir)
                elif dim == 2:
                    if index == 0:
                        return np.einsum("ij,i->ij", weight, weight_dir)
                    elif index == 1:
                        return np.einsum("ij,j->ij", weight, weight_dir)

            # Calculate the weights in the x-direction, y-direction and z-direction, respectively.
            if dim == 3:
                weight = _fourier1(weight, shape, 0, dim)
                weight = _fourier1(weight, shape, 1, dim)
                weight = _fourier1(weight, shape, 2, dim)
            else:
                weight = _fourier1(weight, shape, 0, dim)
                weight = _fourier1(weight, shape, 1, dim)
            return np.ravel(weight)  # Ravel it to be in the dictionary order (z, y, x).
        elif weight == "Alternative":
            alt_volume = self._calculate_alternative_volume(shape)
            return np.ones(np.prod(shape)) * alt_volume / np.prod(shape)
        elif weight == "Fourier2":
            alt_volume = self._calculate_alternative_volume(shape)

            def _fourier2(shape, index):
                # Calculate the Fourier2 weights in each of the directions.
                # Shape of the grid,  index is either 0 (x-direction), 1 (y-dir), 2 (z-dir)
                grid_dir = np.arange(1, shape[index] + 1)
                grid_p = np.arange(1, shape[index])
                grid_2d = np.outer((2.0 * grid_dir - 1) / shape[index], grid_p)
                weight = (
                    4.0
                    * np.einsum(
                        "ij,j->i",
                        np.sin(grid_2d * np.pi),
                        np.sin(grid_p * np.pi / 2.0) ** 2.0 / grid_p,
                    )
                    / (np.pi * shape[index])
                )
                weight += (
                    2.0
                    * np.sin(np.pi * shape[index] / 2.0) ** 2.0
                    * np.sin((grid_dir - 0.5) * np.pi)
                    / (shape[index] ** 2.0 * np.pi)
                )
                return weight

            # Calculate weights in x-direction, y-direction, z-direction, respectively.
            weight = np.ones(shape)
            weight_x = _fourier2(shape, 0)
            weight_y = _fourier2(shape, 1)
            weight_z = _fourier2(shape, 2)
            weight = np.einsum("ijk,i,j,k->ijk", weight, weight_x, weight_y, weight_z) * alt_volume
            return np.ravel(weight)
        else:
            raise ValueError(f"The weight type parameter is not known, got {weight}")

    def closest_point(self, point, which="closest"):
        r"""Identify the index of the closest grid point to a given point.

        Imagine a point inside a small sub-cube. If `closest` is selected, it will
        pick the corner in the sub-cube that is closest to that point.
        if `origin` is selected, it will pick the corner that is the bottom,
        left-most, down-most in the sub-cube.

        Parameters
        ----------
        point : np.ndarray, shape (3,)
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
            raise ValueError(
                "Finding closest point only works when the 'axes' attribute"
                " is a diagonal matrix."
            )

        # Calculate step-size of the cube.
        step_sizes = np.array([np.linalg.norm(axis) for axis in self.axes])
        coord = np.array([(point[i] - self.origin[i]) / step_sizes[i] for i in range(self.ndim)])

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

    def generate_cube(self, fname, data, atcoords, atnums, pseudo_numbers=None):
        r"""Write the data evaluated on grid points into a cube file.

        Parameters
        ----------
        fname : str
            Cube file name with \*.cube extension.
        data : np.ndarray, shape=(npoints,)
            An array containing the evaluated scalar property on the grid points.
        atcoords : np.ndarray, shape (M, 3)
            Cartesian coordinates of :math:`M` atoms in the molecule.
        atnums : np.ndarray, shape (M,)
            Atomic numbers of :math:`M` atoms in the molecule.
        pseudo_numbers : np.ndarray, shape (M,), optional
            Pseudo-numbers (core charges) of :math:`M` atoms in the molecule.

        """
        if not fname.endswith(".cube"):
            raise ValueError("Argument fname should be a cube file with `*.cube` extension!")
        natom = len(atnums)
        if natom != len(atcoords):
            raise ValueError("The number of atomic numbers and atom coordinates should equal.")
        if data.size != len(self.points):
            raise ValueError(
                f"`data` should have the same size as the grid {data.size}!={len(self.points)}"
            )
        if pseudo_numbers is None:
            pseudo_numbers = atnums.astype(float)
        if pseudo_numbers.size != natom:
            raise ValueError(
                "Argument pseudo_numbers should have the same size as the atomic numbers. "
                + f"{pseudo_numbers.size}!={natom}"
            )

        # Write data into the cube file
        with open(fname, "w") as f:
            # writing the cube header:
            f.write("Cubefile created with THEOCHEM Grid\n")
            f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
            x, y, z = self._origin
            f.write(f"{natom:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")
            rvecs = self._axes
            for i, (x, y, z) in zip(self._shape, rvecs):
                f.write(f"{i:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")
            for i, q, (x, y, z) in zip(atnums, pseudo_numbers, atcoords):
                f.write(f"{i:5d} {q:11.6f} {x:11.6f} {y:11.6f} {z:11.6f}\n")
            # writing the cube data:
            num_chunks = 6
            for i in range(0, data.size, num_chunks):
                row_data = data.flat[i : i + num_chunks]
                f.write((row_data.size * " {:12.5E}").format(*row_data))
                f.write("\n")
