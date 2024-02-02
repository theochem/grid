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
r"""Rectangular Grid Testing."""

from unittest import TestCase

import numpy as np
from importlib_resources import files
from numpy.testing import assert_allclose

from grid.cubic import Tensor1DGrids, UniformGrid, _HyperRectangleGrid
from grid.onedgrid import GaussLaguerre, MidPoint


class TestHyperRectangleGrid(TestCase):
    r"""Test HyperRectangleGrid class."""

    def test_get_points_along_axes(self):
        r"""Test getting the points alongside each axis."""
        # Three Dimensions
        points = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        weights = np.array([1.0] * points.shape[0])
        grid = _HyperRectangleGrid(points, weights, (2, 2, 2))
        x, y, z = grid.get_points_along_axes()
        assert_allclose(x, np.array([0, 1]))
        assert_allclose(y, np.array([0, 1]))
        assert_allclose(z, np.array([0, 1]))

        # Two Dimensions
        points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        weights = np.array([1.0] * points.shape[0])
        grid = _HyperRectangleGrid(points, weights, (2, 2))
        x, y = grid.get_points_along_axes()
        assert_allclose(x, np.array([0, 1]))
        assert_allclose(y, np.array([0, 1]))

    def test_raises_error_when_constructing_grid(self):
        r"""Test parameters of constructing a hyper-rectangular grid."""
        points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        weights = np.array([1.0] * points.shape[0])
        # Test with shape length greater than 3
        with self.assertRaises(ValueError) as err:
            _HyperRectangleGrid(points, weights, (2, 2, 2, 2))
        self.assertEqual(
            "Argument shape should have length two or three; got length 4.",
            str(err.exception),
        )
        # Test with shape with number of points is one.
        with self.assertRaises(ValueError) as err:
            _HyperRectangleGrid(points, weights, (2, 2, 1))
        self.assertEqual(
            "Argument shape should be greater than one in all directions (2, 2, 1).",
            str(err.exception),
        )
        # Test the product of shape doesn't equal the number of points
        with self.assertRaises(ValueError) as err:
            _HyperRectangleGrid(points, weights, (2, 3))
        self.assertEqual(
            "The product of shape elements 6 should match the number of grid points 4.",
            str(err.exception),
        )
        # Test the dimension specified by shape doesn't match the number of points.
        points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]])
        with self.assertRaises(ValueError) as err:
            _HyperRectangleGrid(points, weights, (2, 2))
        self.assertEqual(
            "The dimension of the shape/grid 2 should match the dimension of the points 3.",
            str(err.exception),
        )

    def test_raise_error_when_using_interpolation(self):
        r"""Test the raising of error when using interpolation function."""
        points = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        weights = np.array([1.0] * points.shape[0])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        # Test raises error if method string isn't correct
        with self.assertRaises(ValueError) as err:
            _HyperRectangleGrid(points, weights, (2, 2, 2)).interpolate(
                points, values, method="not cubic"
            )
        self.assertEqual(
            "Argument method should be either cubic, linear, or nearest , got not cubic",
            str(err.exception),
        )
        # Test raises error if dimension is two.
        with self.assertRaises(NotImplementedError) as err:
            points2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            weights2 = np.array([1.0] * points2.shape[0])
            _HyperRectangleGrid(points2, weights2, (2, 2)).interpolate(points, values)
        self.assertEqual(
            "Interpolation only works for three dimension, got ndim=2",
            str(err.exception),
        )
        # Test when number of function values isn't the same as the number of grid points
        with self.assertRaises(ValueError) as err:
            values = np.array([1.0, 2.0])
            _HyperRectangleGrid(points, weights, (2, 2, 2)).interpolate(points, values)
        self.assertEqual(
            "Number of function values 2 does not match number of grid points 8.",
            str(err.exception),
        )


class TestTensor1DGrids(TestCase):
    r"""Test Tensor Product of 1D Grids."""

    def test_raise_error_when_constructing_grid(self):
        r"""Test raising of error when constructing Tensor1DGrids."""
        oned = GaussLaguerre(10)
        # Test raises error
        with self.assertRaises(TypeError) as err:
            Tensor1DGrids(5, oned, oned)
        self.assertEqual(
            "Argument oned_x should be an instance of `OneDGrid`, got <class 'int'>",
            str(err.exception),
        )
        with self.assertRaises(TypeError) as err:
            Tensor1DGrids(oned, 5, oned)
        self.assertEqual(
            "Argument oned_y should be an instance of `OneDGrid`, got <class 'int'>",
            str(err.exception),
        )
        with self.assertRaises(TypeError) as err:
            Tensor1DGrids(oned, oned, 5)
        self.assertEqual(
            "Argument oned_z should be an instance of `OneDGrid`, got <class 'int'>",
            str(err.exception),
        )

    def test_origin_of_tensor1d_grids(self):
        r"""Test that the origin is the first point."""
        oned = GaussLaguerre(10)
        oned2 = GaussLaguerre(20)
        grid = Tensor1DGrids(oned, oned2)
        assert_allclose(np.array([oned.points[0], oned2.points[0]]), grid.origin)

    def test_point_and_weights_are_correct(self):
        r"""Test that the points and weights are correctly computed."""
        oned = GaussLaguerre(10)
        cubic = Tensor1DGrids(oned, oned, oned)

        index = 0  # Index for cubic points.
        for i in range(oned.size):
            for j in range(oned.size):
                for k in range(oned.size):
                    actual_pt = np.array([oned.points[i], oned.points[j], oned.points[k]])
                    assert_allclose(actual_pt, cubic.points[index, :])
                    actual_weight = oned.weights[i] * oned.weights[j] * oned.weights[k]
                    assert_allclose(actual_weight, cubic.weights[index])
                    index += 1

    def test_interpolation_of_gaussian_vertorized(self):
        r"""Test interpolation of a Gaussian function with vectorization."""
        oned = MidPoint(50)
        cubic = Tensor1DGrids(oned, oned, oned)

        def gaussian(points):
            return np.exp(-3 * np.linalg.norm(points, axis=1) ** 2.0)

        gaussian_pts = gaussian(cubic.points)
        num_pts = 500
        random_pts = np.random.uniform(-0.9, 0.9, (num_pts, 3))
        interpolated = cubic.interpolate(random_pts, gaussian_pts, use_log=False)
        assert_allclose(interpolated, gaussian(random_pts), rtol=1e-5, atol=1e-6)

        interpolated = cubic.interpolate(random_pts, gaussian_pts, use_log=True)
        assert_allclose(interpolated, gaussian(random_pts), rtol=1e-5, atol=1e-6)

    def test_interpolation_of_linear_function_using_scipy_linear_method(self):
        r"""Test interpolation of a linear function using scipy with linear method."""
        oned = MidPoint(50)
        cubic = Tensor1DGrids(oned, oned, oned)

        def linear_func(points):
            return np.dot(np.array([1.0, 2.0, 3.0]), points.T)

        gaussian_pts = linear_func(cubic.points)
        num_pts = 50
        random_pts = np.random.uniform(-0.9, 0.9, (num_pts, 3))
        interpolated = cubic.interpolate(random_pts, gaussian_pts, use_log=False, method="linear")
        assert_allclose(interpolated, linear_func(random_pts))

    def test_interpolation_of_constant_function_using_scipy_nearest_method(self):
        r"""Test interpolation of a constant function using scipy with nearest method."""
        oned = MidPoint(50)
        cubic = Tensor1DGrids(oned, oned, oned)

        def linear_func(points):
            return np.array([1.0] * points.shape[0]) + np.random.random(points.shape[0]) * 1.0e-6

        gaussian_pts = linear_func(cubic.points)
        num_pts = 5
        random_pts = np.random.uniform(-0.9, 0.9, (num_pts, 3))
        for pt in random_pts:
            interpolated = cubic.interpolate(pt, gaussian_pts, use_log=False, method="nearest")
            assert_allclose(interpolated, linear_func(np.array([pt]))[0], rtol=1e-6)

    def test_interpolation_of_various_derivative_gaussian_using_logarithm(self):
        r"""Test interpolation of the thederivatives of a Gaussian function."""
        oned = MidPoint(100)
        cubic = Tensor1DGrids(oned, oned, oned)

        def gaussian(points):
            return np.exp(-3 * np.linalg.norm(points, axis=1) ** 2.0)

        def derivative_wrt_one_var(points, i_var_deriv):
            return (
                np.exp(-3 * np.linalg.norm(points, axis=1) ** 2.0)
                * points[:, i_var_deriv]
                * (-3 * 2.0)
            )

        def derivative_second_x(points):
            expon = np.exp(-3 * np.linalg.norm(points, axis=1) ** 2.0)
            return expon * points[:, 0] ** 2.0 * (-3 * 2.0) ** 2.0 + expon * (-3 * 2.0)

        gaussian_pts = gaussian(cubic.points)

        pts = np.random.uniform(-0.5, 0.5, (10, 3))
        # Test taking derivative in x-direction
        interpolated = cubic.interpolate(pts, gaussian_pts, use_log=True, nu_x=1)
        assert_allclose(interpolated, derivative_wrt_one_var(pts, 0), rtol=1e-4)

        # Test taking derivative in y-direction
        interpolated = cubic.interpolate(pts, gaussian_pts, use_log=True, nu_y=1)
        assert_allclose(interpolated, derivative_wrt_one_var(pts, 1), rtol=1e-4)

        # Test taking derivative in z-direction
        interpolated = cubic.interpolate(pts, gaussian_pts, use_log=True, nu_z=1)
        assert_allclose(interpolated, derivative_wrt_one_var(pts, 2), rtol=1e-4)

        # Test taking second-derivative in x-direction
        interpolated = cubic.interpolate(pts, gaussian_pts, use_log=True, nu_x=2, nu_y=0, nu_z=0)
        assert_allclose(interpolated, derivative_second_x(pts), rtol=1e-4)

        # Test raises error
        with self.assertRaises(NotImplementedError):
            cubic.interpolate(pts, gaussian_pts, use_log=True, nu_x=2, nu_y=2)

    def test_integration_of_gaussian(self):
        r"""Test integration of a rapidly-decreasing Gaussian."""
        oned = MidPoint(250)
        cubic = Tensor1DGrids(oned, oned, oned)

        def gaussian(points):
            return np.exp(-6 * np.linalg.norm(points, axis=1) ** 2.0)

        gaussian_pts = gaussian(cubic.points)
        desired = np.sqrt(np.pi / 6) ** 3
        actual = cubic.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-3)

    def test_moving_coordinates_to_index_and_back_three_dimensions(self):
        r"""Test moving from coordinates and index and back in three_dimensions."""
        oned = MidPoint(3)
        cubic = Tensor1DGrids(oned, oned, oned)

        # Convert index to coordinate.
        index = 3
        coord = (0, 1, 0)
        assert_allclose(coord, cubic.index_to_coordinates(index))

        # Convert coordinate to index
        coord = (1, 0, 1)
        index = 10
        assert_allclose(index, cubic.coordinates_to_index(coord))

        # Convert back
        index = 9
        assert_allclose(index, cubic.coordinates_to_index(cubic.index_to_coordinates(index)))

        # Test raises error when index isn't positive
        with self.assertRaises(ValueError):
            cubic.index_to_coordinates(-1)

    def test_moving_coordinates_to_index_and_back_two_dimensions(self):
        r"""Test moving from coordinates and index and back in two dimensions."""
        oned = MidPoint(3)
        cubic = Tensor1DGrids(oned, oned)

        # Convert index to coordinate.
        index = 3
        coord = (1, 0)
        assert_allclose(coord, cubic.index_to_coordinates(index))

        # Convert coordinate to index
        coord = (1, 1)
        index = 4
        assert_allclose(index, cubic.coordinates_to_index(coord))

        # Convert back
        index = 1
        assert_allclose(index, cubic.coordinates_to_index(cubic.index_to_coordinates(index)))


class TestUniformGrid(TestCase):
    r"""Test Uniform Grid Class."""

    def test_raises_error_correctly_when_constructing_uniform_grid(self):
        r"""Test raises error when constructing uniform grid."""
        proper_origin = np.array([0.0, 0.0, 0.0])
        proper_axes = np.eye(3)
        proper_shape = np.array([5, 5, 5])
        # Test origin
        with self.assertRaises(TypeError) as err:
            UniformGrid((0, 0, 0), proper_axes, proper_shape)
        self.assertEqual(
            "Argument origin should be a numpy array, got <class 'tuple'>",
            str(err.exception),
        )
        # Test axes
        with self.assertRaises(TypeError) as err:
            UniformGrid(proper_origin, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], proper_shape)
        self.assertEqual(
            "Argument axes should be a numpy array, got <class 'list'>",
            str(err.exception),
        )
        # Test shape
        with self.assertRaises(TypeError) as err:
            UniformGrid(proper_origin, proper_axes, (5, 5, 5))
        self.assertEqual(
            "Argument shape should be a numpy array, got <class 'tuple'>",
            str(err.exception),
        )
        # Test origin has correct shape
        with self.assertRaises(ValueError) as err:
            UniformGrid(np.array([0, 0, 0, 0]), proper_axes, proper_shape)
        self.assertEqual(
            "Arguments origin should have size 2 or 3, got (4,)",
            str(err.exception),
        )
        # Test that origin and shape should have the same size
        with self.assertRaises(ValueError) as err:
            UniformGrid(proper_origin, proper_axes, np.array([5, 5, 5, 5]))
        self.assertEqual(
            "Shape 4 should be the same size 3.",
            str(err.exception),
        )
        # Test that axes has the same dimensions as the origin
        with self.assertRaises(ValueError) as err:
            UniformGrid(proper_origin, np.array([[5, 5], [5, 5]]), proper_shape)
        self.assertEqual(
            "Axes (2, 2) should be the same shape 3, 3.",
            str(err.exception),
        )
        # Test that axes is linearly independent
        with self.assertRaises(ValueError) as err:
            UniformGrid(proper_origin, np.array([[5, 5, 5], [5, 5, 5], [0, 0, 1]]), proper_shape)
        self.assertEqual(
            "The axes are not linearly independent, got det(axes)=0.0",
            str(err.exception),
        )
        # Test shape is always positive
        with self.assertRaises(ValueError) as err:
            UniformGrid(proper_origin, proper_axes, np.array([5, -1, 2]))
        self.assertEqual(
            "Number of points in each direction should be positive, got shape=[5, -1, 2]",
            str(err.exception),
        )
        # Test the weights are correct
        with self.assertRaises(ValueError) as err:
            UniformGrid(proper_origin, proper_axes, proper_shape, weight="not trapezoid")
        self.assertEqual(
            "The weight type parameter is not known, got not trapezoid",
            str(err.exception),
        )

    def test_constructing_uniform_grid_in_three_dimensions(self):
        r"""Test the construction of a uniform grid is correct in three dimensions."""
        origin = np.array([1.0, 1.0, 1.0])
        axes = np.diag([1, 2, 3])
        shape = np.array([2, 2, 2])
        uniform = UniformGrid(origin, axes, shape)
        actual = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 4.0],
                [1.0, 3.0, 1.0],
                [1.0, 3.0, 4.0],
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 4.0],
                [2.0, 3.0, 1.0],
                [2.0, 3.0, 4.0],
            ]
        )
        assert_allclose(actual, uniform.points)

    def test_constructing_uniform_grid_in_two_dimensions(self):
        r"""Test the construction of a uniform grid is correct in two dimensions."""
        origin = np.array([1.0, 1.0])
        axes = np.diag([1, 2])
        shape = np.array([2, 2])
        uniform = UniformGrid(origin, axes, shape, "Rectangle")
        actual = np.array([[1.0, 1.0], [1.0, 3.0], [2.0, 1.0], [2.0, 3.0]])
        assert_allclose(actual, uniform.points)

    def test_volume_generalizes_volume_of_cube_with_orthogonal_axes(self):
        r"""Test volume of a cube is the same as multiplying by length of axes."""
        origin = np.array([0.0, 0.0, 0.0])
        axes = np.eye(3)
        shape = np.array([5, 6, 7], dtype=int)
        uniform = UniformGrid(origin, axes, shape=shape)
        volume = 5 * 6 * 7
        assert_allclose(volume, uniform._calculate_volume(shape))

    def test_area_generalized_volume_of_rectangle_with_orthogonal_axes(self):
        r"""Test area of a rectangle is the same as multiplying the length of axes."""
        origin = np.array([0.0, 0.0])
        axes = np.eye(2)
        shape = np.array([5, 6], dtype=int)
        uniform = UniformGrid(origin, axes, shape=shape)
        volume = 5 * 6
        assert_allclose(volume, uniform._calculate_volume(shape))

    def test_fourier1_weights_are_correct(self):
        r"""Test Fourier1 weights are correct against brute force."""
        origin = np.array([0.0, 0.0, 0.0])
        axes = np.eye(3)
        shape = np.array([5, 6, 7], dtype=int)
        volume = 5 * 6 * 7  # Volume of cube centered at zero, moves in one step at a time (axes)
        uniform = UniformGrid(origin, axes, shape=shape, weight="Fourier1")

        index = 0  # Index to iterate through uniform.weights.
        for j in range(1, shape[0] + 1):
            grid_x = np.arange(1, shape[0] + 1)
            desired_x = np.sum(
                np.sin(j * np.pi * grid_x / (shape[0] + 1))
                * (1 - np.cos(grid_x * np.pi))
                / (grid_x * np.pi)
            )
            for k in range(1, shape[1] + 1):
                grid_y = np.arange(1, shape[1] + 1)
                desired_y = np.sum(
                    np.sin(k * np.pi * grid_y / (shape[1] + 1))
                    * (1 - np.cos(grid_y * np.pi))
                    / (grid_y * np.pi)
                )
                for m in range(1, shape[2] + 1):
                    grid_z = np.arange(1, shape[2] + 1)
                    desired_z = np.sum(
                        np.sin(m * np.pi * grid_z / (shape[2] + 1))
                        * (1 - np.cos(grid_z * np.pi))
                        / (grid_z * np.pi)
                    )
                    desired = 8 * desired_x * desired_y * desired_z * volume / (6 * 7 * 8)
                    assert_allclose(uniform.weights[index], desired)
                    index += 1

    def test_fourier2_weights_are_correct(self):
        r"""Test that the Fourier2 weights are correct against brute force."""
        origin = np.array([0.0, 0.0, 0.0])
        axes = np.eye(3)
        shape = np.array([5, 6, 7], dtype=int)
        volume = 5 * 6 * 7  # Volume of cube centered at zero, moves in one step at a time (axes)
        volume *= (4.0 / 5.0) * (5.0 / 6) * (6.0 / 7)  # Alternative volume is used here.
        uniform = UniformGrid(origin, axes, shape=shape, weight="Fourier2")
        index = 0  # Index to iterate through uniform.weights.
        for j in range(1, shape[0] + 1):
            # Calculate weight in the x-direction
            grid_x = np.arange(1, shape[0])
            desired_x = 2.0 * np.sin((j - 0.5) * np.pi) * np.sin(shape[0] * np.pi / 2) ** 2.0
            desired_x /= shape[0] ** 2.0 * np.pi
            desired_x += (
                4.0
                * np.sum(
                    np.sin((2.0 * j - 1.0) * grid_x * np.pi / shape[0])
                    * np.sin(grid_x * np.pi / 2) ** 2.0
                    / grid_x
                )
                / (shape[0] * np.pi)
            )
            for k in range(1, shape[1] + 1):
                # Calculate weight in the y-direction
                grid_y = np.arange(1, shape[1])
                desired_y = 2.0 * np.sin((k - 0.5) * np.pi) * np.sin(shape[1] * np.pi / 2) ** 2.0
                desired_y /= shape[1] ** 2.0 * np.pi
                desired_y += (
                    4.0
                    * np.sum(
                        np.sin((2.0 * k - 1.0) * grid_y * np.pi / shape[1])
                        * np.sin(grid_y * np.pi / 2) ** 2.0
                        / grid_y
                    )
                    / (shape[1] * np.pi)
                )
                for m in range(1, shape[2] + 1):
                    # Calculate weight in the z-direction
                    grid_z = np.arange(1, shape[2])
                    desired_z = (
                        2.0 * np.sin((m - 0.5) * np.pi) * np.sin(shape[2] * np.pi / 2) ** 2.0
                    )
                    desired_z /= shape[2] ** 2.0 * np.pi
                    desired_z += (
                        4.0
                        * np.sum(
                            np.sin((2.0 * m - 1.0) * grid_z * np.pi / shape[2])
                            * np.sin(grid_z * np.pi / 2) ** 2.0
                            / grid_z
                        )
                        / (shape[2] * np.pi)
                    )
                    desired = desired_x * desired_y * desired_z * volume
                    assert_allclose(uniform.weights[index], desired)
                    index += 1

    def test_calculating_rectangle_weights_with_orthogonal_axes(self):
        r"""Test calcualting rectangle weights with orthogonal axes."""
        # Set up the grid with easy examples but axes that form a cube.
        origin = np.array([0.0, 0.0, 0.0])
        axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        shape = np.array([3, 3, 3], dtype=int)
        uniform = UniformGrid(origin, axes, shape, weight="Rectangle")
        volume = 3 * 3 * 3  # Volume of cube.
        desired_wghts = np.ones(uniform.size) * volume / np.prod(shape)
        assert_allclose(uniform.weights, desired_wghts)

    def test_calculating_trapezoid_weights_with_orthogonal_axes(self):
        r"""Test calculating trapezoid weights with orthogonal axes."""
        # Set up the grid with easy examples but axes that form a cube.
        origin = np.array([0.0, 0.0, 0.0])
        axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        shape = np.array([3, 3, 3], dtype=int)
        uniform = UniformGrid(origin, axes, shape, weight="Trapezoid")
        volume = 3 * 3 * 3  # Volume of cube.
        desired_wghts = np.ones(uniform.size) * volume / np.prod(shape + 1)
        assert_allclose(uniform.weights, desired_wghts)

    def test_calculating_alternative_weights_with_orthogonal_axes(self):
        r"""Test calculating alternative weights with orthogonal axes."""
        # Set up the grid with easy examples but axes that form a cube.
        origin = np.array([0.0, 0.0, 0.0])
        axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        shape = np.array([3, 3, 3], dtype=int)
        uniform = UniformGrid(origin, axes, shape, weight="Alternative")
        volume = 3 * 3 * 3  # Volume of cube.
        desired_wghts = np.ones(uniform.size) * volume * np.prod(shape - 1) / np.prod(shape) ** 2.0
        assert_allclose(uniform.weights, desired_wghts)

    def test_integration_with_gaussian_with_cubic_grid(self):
        r"""Test integration against a Gaussian with a cubic grid."""
        origin = np.array([-1.0, -1.0, -1.0])
        axes = np.eye(3) * 0.01
        shape = np.array([250, 250, 250], dtype=int)

        def gaussian(points):
            return np.exp(-5 * np.linalg.norm(points, axis=1) ** 2.0)

        # Test with rectangle weights.
        uniform = UniformGrid(origin, axes, shape, weight="Rectangle")
        gaussian_pts = gaussian(uniform.points)
        desired = np.sqrt(np.pi / 5) ** 3
        actual = uniform.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-2)

        # Test with Trapezoid weights.
        uniform = UniformGrid(origin, axes, shape, weight="Trapezoid")
        gaussian_pts = gaussian(uniform.points)
        desired = np.sqrt(np.pi / 5) ** 3
        actual = uniform.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-2)

        # Test with Fourier1 weights.
        uniform = UniformGrid(origin, axes, shape, weight="Fourier1")
        gaussian_pts = gaussian(uniform.points)
        desired = np.sqrt(np.pi / 5) ** 3
        actual = uniform.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-2)

    def test_integration_with_gaussian_with_square_grid(self):
        r"""Test integration against a Gaussian with a square grid."""
        origin = np.array([-1.0, -1.0])
        axes = np.eye(2) * 0.005
        shape = np.array([500, 500], dtype=int)

        def gaussian(points):
            return np.exp(-5 * np.linalg.norm(points, axis=1) ** 2.0)

        # Test with rectangle weights.
        uniform = UniformGrid(origin, axes, shape, weight="Rectangle")
        gaussian_pts = gaussian(uniform.points)
        desired = np.pi / 5
        actual = uniform.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-2)

        # Test with Trapezoid weights.
        uniform = UniformGrid(origin, axes, shape, weight="Trapezoid")
        gaussian_pts = gaussian(uniform.points)
        desired = np.pi / 5
        actual = uniform.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-2)

        # Test with Fourier1 weights.
        uniform = UniformGrid(origin, axes, shape, weight="Fourier1")
        gaussian_pts = gaussian(uniform.points)
        desired = np.pi / 5
        actual = uniform.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-2)

        # Test with Fourier1 weights.
        uniform = UniformGrid(origin, axes, shape, weight="Alternative")
        gaussian_pts = gaussian(uniform.points)
        desired = np.pi / 5
        actual = uniform.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-2)

    def test_finding_closest_point_to_cubic_grid(self):
        r"""Test finding the closest point to a cubic grid."""
        # Set up the grid with easy examples but axes that form a cube.
        origin = np.array([0.0, 0.0, 0.0])
        axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        shape = np.array([3, 3, 3], dtype=int)
        uniform = UniformGrid(origin, axes, shape)

        # Create point close to the origin
        pt = np.array([0.1, 0.1, 0.1])
        index = uniform.closest_point(pt, "origin")
        assert index == 0
        index = uniform.closest_point(pt, "closest")
        assert index == 0

        # Create point close to the point with coordinates (1, 1, 1) but whose origin is zero.
        pt = np.array([0.75, 0.75, 0.75])
        index = uniform.closest_point(pt, "origin")
        assert index == 0
        index = uniform.closest_point(pt, "closest")
        assert index == 13

        # Test raises error
        with self.assertRaises(ValueError):
            # Test wrong attribute.
            uniform.closest_point(pt, "not origin or closest")

        # Test raises error with orthogonal axes.
        axes = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        uniform = UniformGrid(origin, axes, shape)
        with self.assertRaises(ValueError) as err:
            # Test wrong attribute.
            uniform.closest_point(pt, "origin")
        self.assertEqual(
            "Finding closest point only works when the 'axes' attribute is a diagonal matrix.",
            str(err.exception),
        )

    def test_finding_closest_point_to_square_grid(self):
        r"""Test finding the closest point to a square grid."""
        # Set up the grid with easy examples but axes that form a cube.
        origin = np.array([0.0, 0.0])
        axes = np.eye(2)
        shape = np.array([3, 3], dtype=int)
        uniform = UniformGrid(origin, axes, shape)

        # Create point close to the origin
        pt = np.array([0.1, 0.1])
        index = uniform.closest_point(pt, "origin")
        assert index == 0
        index = uniform.closest_point(pt, "closest")
        assert index == 0

        # Create point close to the point with coordinates (1, 1, 1) but whose origin is zero.
        pt = np.array([1.75, 1.75, 1.75])
        index = uniform.closest_point(pt, "origin")
        assert index == 4
        index = uniform.closest_point(pt, "closest")
        assert index == 8

    def test_uniformgrid_points_h2o(self):
        r"""Test creating uniform cubic grid from molecule against h2o example."""
        # replace this test with a better one later
        pseudo_numbers = np.array([8.0, 1.0, 1.0, 8.0, 1.0, 1.0])
        coordinates = np.array(
            [
                [-1.01306328e-03, 2.87066713e00, -1.97656750e-16],
                [1.74795831e-01, 1.04876344e00, -7.33275045e-17],
                [1.70493824e00, 3.49056186e00, -2.51209193e-16],
                [-1.01306328e-03, -2.62839035e00, 1.80987402e-16],
                [-9.31762597e-01, -3.23876982e00, 1.43814244e00],
                [-9.31762597e-01, -3.23876982e00, -1.43814244e00],
            ]
        )
        grid = UniformGrid.from_molecule(
            pseudo_numbers, coordinates, spacing=2.0, extension=0.0, rotate=True
        )
        expected = np.array(
            [
                [-2.31329824e00, -2.00000000e00, 3.82735565e00],
                [-2.31329824e00, -4.99999997e-09, 3.82735565e00],
                [-3.19696330e-01, -2.00000000e00, 3.98720381e00],
                [-3.19696330e-01, -4.99999997e-09, 3.98720381e00],
                [-2.15345008e00, -2.00000000e00, 1.83375375e00],
                [-2.15345008e00, -4.99999997e-09, 1.83375375e00],
                [-1.59848169e-01, -2.00000000e00, 1.99360191e00],
                [-1.59848169e-01, -4.99999997e-09, 1.99360191e00],
                [-1.99360191e00, -2.00000000e00, -1.59848162e-01],
                [-1.99360191e00, -4.99999997e-09, -1.59848162e-01],
                [-6.77400003e-09, -2.00000000e00, 0.00000000e00],
                [-6.77400003e-09, -4.99999997e-09, 0.00000000e00],
                [-1.83375375e00, -2.00000000e00, -2.15345007e00],
                [-1.83375375e00, -4.99999997e-09, -2.15345007e00],
                [1.59848155e-01, -2.00000000e00, -1.99360191e00],
                [1.59848155e-01, -4.99999997e-09, -1.99360191e00],
            ]
        )
        assert_allclose(grid.points, expected, rtol=1.0e-7, atol=1.0e-7)

    def test_uniformgrid_from_cube(self):
        r"""Test creating uniform cubic grid from cube example."""
        atnums = np.array([6, 1, 1, 1, 1])
        pseudo_numbers = np.array([6, 1, 1, 1, 1]).astype(float)
        atcoords = np.array(
            [
                [-0.000041, 0.000035, 0.000003],
                [1.051607, 1.517467, 0.942259],
                [1.296786, -1.543671, -0.481272],
                [-1.476881, -0.708836, 1.269987],
                [-0.871678, 0.735176, -1.730958],
            ]
        )
        axes = np.array(
            [
                [2.605101, 0.000000, 0.000000],
                [0.000000, 2.605101, 0.000000],
                [0.000000, 0.000000, 2.605101],
            ]
        )
        origin = np.array([-6.512793, -6.512718, -6.51275])
        shape = np.array([6, 6, 6])
        data_vals = np.array(
            [
                2.66011e-09,
                2.43339e-09,
                2.14921e-08,
                7.16256e-08,
                7.41998e-08,
                3.40400e-08,
                1.69019e-08,
                2.07614e-08,
                2.34886e-07,
                9.17704e-07,
                4.01950e-07,
                1.05928e-07,
                7.35455e-08,
                3.21547e-07,
                2.09291e-06,
                6.03301e-06,
                1.45834e-06,
                1.68969e-07,
                1.30633e-07,
                7.04469e-07,
                2.11049e-06,
                3.14829e-06,
                8.05222e-07,
                9.93385e-08,
                8.65880e-08,
                2.07846e-07,
                2.42194e-07,
                8.52672e-08,
                5.42543e-08,
                2.40865e-08,
                2.92011e-08,
                4.43778e-08,
                2.27907e-08,
                1.82844e-09,
                2.96839e-09,
                4.61889e-09,
                3.85685e-09,
                4.77266e-09,
                8.96467e-08,
                4.14563e-07,
                2.06334e-07,
                6.44963e-08,
                5.35725e-08,
                8.30180e-07,
                2.20099e-05,
                5.81077e-05,
                8.87121e-06,
                3.42128e-07,
                8.23328e-07,
                4.34876e-05,
                4.35873e-04,
                1.28722e-03,
                8.71430e-05,
                1.23662e-06,
                1.51554e-06,
                7.54430e-05,
                5.63293e-04,
                5.33647e-04,
                5.19831e-05,
                6.33116e-07,
                4.26008e-07,
                7.87307e-06,
                3.52842e-05,
                1.52522e-05,
                1.27574e-06,
                4.07203e-08,
                7.92854e-08,
                1.83999e-07,
                2.08056e-07,
                3.95658e-08,
                1.81406e-08,
                1.06424e-08,
                1.49034e-08,
                1.88894e-07,
                1.84617e-06,
                1.92382e-06,
                3.17458e-07,
                4.89679e-08,
                1.46657e-07,
                2.15085e-05,
                3.85288e-04,
                5.27086e-04,
                4.97767e-05,
                6.21140e-07,
                4.69357e-06,
                6.22502e-04,
                1.72733e-02,
                1.30410e-01,
                9.27459e-04,
                4.18722e-06,
                8.56323e-06,
                1.86260e-03,
                9.10047e-02,
                1.97993e-02,
                4.93451e-04,
                2.31751e-06,
                1.36703e-06,
                6.83701e-05,
                5.09508e-04,
                4.38554e-04,
                3.39089e-05,
                2.29715e-07,
                1.05285e-07,
                5.29104e-07,
                1.77812e-06,
                2.46884e-06,
                4.56541e-07,
                4.62455e-08,
                4.99681e-08,
                7.91930e-07,
                6.41396e-06,
                4.15433e-06,
                1.67544e-07,
                1.33164e-08,
                2.43791e-07,
                4.90738e-05,
                1.28808e-03,
                6.61936e-04,
                2.23468e-05,
                1.06579e-07,
                2.26723e-06,
                4.71262e-04,
                8.74264e-02,
                2.28956e-02,
                3.46961e-04,
                9.71712e-07,
                4.15301e-06,
                5.45842e-04,
                1.74207e-02,
                1.62411e-01,
                6.43468e-04,
                2.30742e-06,
                6.32463e-07,
                3.59814e-05,
                4.98093e-04,
                1.45680e-03,
                7.24936e-05,
                7.16444e-07,
                5.09114e-08,
                2.07418e-07,
                2.88089e-06,
                6.63938e-06,
                1.23315e-06,
                1.11896e-07,
                5.79635e-08,
                3.61388e-07,
                1.56116e-06,
                1.06459e-06,
                9.62388e-08,
                8.43232e-09,
                1.34509e-07,
                7.69557e-06,
                8.87826e-05,
                6.29503e-05,
                1.94672e-06,
                7.84416e-09,
                1.87110e-07,
                4.04358e-05,
                8.77030e-04,
                5.37382e-04,
                2.02792e-05,
                6.58022e-08,
                9.81608e-08,
                1.59426e-05,
                3.91357e-04,
                7.26338e-04,
                5.13141e-05,
                4.63116e-07,
                3.30760e-08,
                8.24312e-07,
                3.84964e-05,
                7.85223e-05,
                9.38048e-06,
                2.79569e-07,
                1.15275e-08,
                2.89590e-08,
                5.59929e-07,
                1.34024e-06,
                4.41195e-07,
                9.32590e-08,
                2.85841e-08,
                9.91317e-08,
                1.83985e-07,
                1.30285e-07,
                3.73820e-08,
                6.62758e-09,
                4.90894e-08,
                2.97380e-07,
                1.29174e-06,
                8.47402e-07,
                7.36705e-08,
                6.35961e-09,
                3.21197e-08,
                4.85867e-07,
                4.12370e-06,
                2.69801e-06,
                1.28003e-07,
                1.18512e-08,
                5.05474e-09,
                7.03752e-08,
                1.43865e-06,
                2.89101e-06,
                4.96815e-07,
                5.24039e-08,
                1.04634e-09,
                1.42491e-08,
                3.45934e-07,
                8.97308e-07,
                3.03767e-07,
                6.92113e-08,
                2.83805e-09,
                1.19992e-08,
                6.32416e-08,
                1.33558e-07,
                9.99063e-08,
                3.65721e-08,
            ]
        )

        ref_grid = UniformGrid(origin, axes, shape)

        cubefile = files("grid") / "data" / "tests" / "cubegen_ch4_6_gen.cube"
        grid, cube_data = UniformGrid.from_cube(cubefile, return_data=True)

        assert_allclose(grid._axes, axes)
        assert_allclose(grid._origin, origin)
        assert_allclose(grid._shape, shape)
        assert_allclose(cube_data["atnums"], atnums)
        assert_allclose(cube_data["atcoords"], atcoords)
        assert_allclose(cube_data["atcorenums"], pseudo_numbers)
        assert_allclose(cube_data["data"], data_vals)

        assert_allclose(grid.points, ref_grid.points)
        assert_allclose(grid.weights, ref_grid.weights)

    def test_uniformgrid_generate_cube(self):
        r"""Test creating uniform cubic grid from cube example."""
        # Change to better test later
        ref_cube = files("grid") / "data" / "tests" / "cubegen_ch4_6_gen.cube"
        out_cube = str(files("grid") / "data" / "tests" / "cubegen_ch4_6_gen.cube")

        grid, cube_data = UniformGrid.from_cube(ref_cube, return_data=True)
        atnums, atcoords, data = cube_data["atnums"], cube_data["atcoords"], cube_data["data"]
        grid.generate_cube(out_cube, data=data, atcoords=atcoords, atnums=atnums)

        with open(out_cube) as out_f, open(ref_cube) as ref_f:
            for _ in range(2):
                next(out_f)
                next(ref_f)
            for line_r, line_o in zip(ref_f, out_f):
                numbers_r = np.array([float(x) for x in line_r.split()])
                numbers_o = np.array([float(x) for x in line_o.split()])
                assert_allclose(numbers_r, numbers_o)

    def test_uniformgrid_points_without_rotate(self):
        r"""Test creating uniform cubic grid from simple constructed molecule."""
        # replace this test with a better one later
        pseudo_numbers = np.array([1.0, 1.0])
        coordinates = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        grid = UniformGrid.from_molecule(
            pseudo_numbers, coordinates, spacing=1.0, extension=1.0, rotate=False
        )
        # Shape should be (4, 2, 2)
        expected = np.array(
            [
                [-2, -1, -1],
                [-2, -1, 0],
                [-2, 0, -1],
                [-2, 0, 0],
                [-1, -1, -1],
                [-1, -1, 0],
                [-1, 0, -1],
                [-1, 0, 0],
                [0, -1, -1],
                [0, -1, 0],
                [0, 0, -1],
                [0, 0, 0],
                [1, -1, -1],
                [1, -1, 0],
                [1, 0, -1],
                [1, 0, 0],
            ]
        )
        assert_allclose(grid.points, expected, rtol=1.0e-7, atol=1.0e-7)
