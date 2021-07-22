r"""Cubic Grid Testing."""

from unittest import TestCase

from grid.cubic import UniformCubicGrid, Tensor1DGrids
from grid.onedgrid import GaussLaguerre, MidPoint

import numpy as np
from numpy.testing import assert_allclose


class TestTensor1DGrids(TestCase):
    r"""Test Tensor Product of 1D Grids."""
    def test_point_and_weights_are_correct(self):
        r"""Test that the points and weights are correctly computed."""
        oned = GaussLaguerre(10)
        cubic = Tensor1DGrids([oned, oned, oned])

        index = 0  # Index for cubic points.
        for i in range(oned.size):
            for j in range(oned.size):
                for k in range(oned.size):
                    actual_pt = np.array([oned.points[i], oned.points[j], oned.points[k]])
                    assert_allclose(actual_pt, cubic.points[index, :])
                    actual_weight = oned.weights[i] * oned.weights[j] * oned.weights[k]
                    assert_allclose(actual_weight, cubic.weights[index])
                    index += 1

    def test_interpolation_of_gaussian(self):
        r"""Test interpolation of a Gaussian function."""
        oned = MidPoint(50)
        cubic = Tensor1DGrids([oned, oned, oned])

        def gaussian(points):
            return np.exp(-3 * np.linalg.norm(points, axis=1)**2.0)

        gaussian_pts = gaussian(cubic.points)
        num_pts = 5
        random_pts = np.random.uniform(-0.9, 0.9, (num_pts, 3))
        for pt in random_pts:
            interpolated = cubic.interpolate_function(pt, gaussian_pts, use_log=True)
            assert_allclose(interpolated, gaussian(np.array([pt]))[0])

    def test_interpolation_of_various_derivative_polynomial(self):
        r"""Test interpolation of the derivative of a quadraticpolynomial function."""
        oned = MidPoint(200)
        cubic = Tensor1DGrids([oned, oned, oned])

        def quadratic_polynomial(points):
            return np.sum(points**4, axis=1)

        def derivative_wrt_one_var(point, i_var_deriv):
            if i_var_deriv == 0: return 4 * point[0]**3
            if i_var_deriv == 1: return 4 * point[1]**3
            if i_var_deriv == 2: return 4 * point[2]**3

        def derivative_second_x(point):
            return 4 * 3 * point[0]**2

        # Evaluate function over the grid
        gaussian_pts = quadratic_polynomial(cubic.points)
        pt = np.random.uniform(-1, 1, (3,))
        # Test taking derivative in x-direction
        interpolated = cubic.interpolate_function(pt, gaussian_pts, use_log=False, nu_x=1)
        assert_allclose(interpolated, derivative_wrt_one_var(pt, 0), rtol=1e-4, atol=1e-4)

        # Test taking derivative in y-direction
        interpolated = cubic.interpolate_function(pt, gaussian_pts, use_log=False, nu_y=1)
        assert_allclose(interpolated, derivative_wrt_one_var(pt, 1), rtol=1e-4, atol=1e-4)

        # Test taking derivative in z-direction
        interpolated = cubic.interpolate_function(pt, gaussian_pts, use_log=False, nu_z=1)
        assert_allclose(interpolated, derivative_wrt_one_var(pt, 2), rtol=1e-4, atol=1e-4)

        # Test taking derivative in x,y,z-direction
        interpolated = cubic.interpolate_function(pt, gaussian_pts, use_log=False,
                                                  nu_x=1, nu_y=1, nu_z=1)
        assert np.abs(interpolated) < 1e-8

        # Test taking second-derivative in x-direction
        interpolated = cubic.interpolate_function(pt, gaussian_pts, use_log=False,
                                                  nu_x=2, nu_y=0, nu_z=0)
        assert_allclose(interpolated, derivative_second_x(pt), rtol=1e-3)

    def test_interpolation_of_various_derivative_gaussian_using_logarithm(self):
        r"""Test interpolation of the derivatives of a Gaussian function."""
        oned = MidPoint(150)
        cubic = Tensor1DGrids([oned, oned, oned])

        def gaussian(points):
            return np.exp(-3 * np.linalg.norm(points, axis=1)**2.0)

        def derivative_wrt_one_var(point, i_var_deriv):
            return np.exp(-3 * np.linalg.norm(point)**2.0) * point[i_var_deriv] * (-3 * 2.0)

        def derivative_second_x(point):
            return np.exp(-3 * np.linalg.norm(point)**2.0) * point[0]**2.0 * (-3 * 2.0)**2.0 + \
                   np.exp(-3 * np.linalg.norm(point) ** 2.0) * (-3 * 2.0)

        gaussian_pts = gaussian(cubic.points)

        pt = np.random.uniform(-0.5, 0.5, (3,))
        # Test taking derivative in x-direction
        interpolated = cubic.interpolate_function(pt, gaussian_pts, use_log=True, nu_x=1)
        assert_allclose(interpolated, derivative_wrt_one_var(pt, 0), rtol=1e-4)

        # Test taking derivative in z-direction
        interpolated = cubic.interpolate_function(pt, gaussian_pts, use_log=True, nu_z=1)
        assert_allclose(interpolated, derivative_wrt_one_var(pt, 2), rtol=1e-4)

        # Test taking second-derivative in x-direction
        interpolated = cubic.interpolate_function(pt, gaussian_pts, use_log=True,
                                                  nu_x=2, nu_y=0, nu_z=0)
        assert_allclose(interpolated, derivative_second_x(pt), rtol=1e-4)

        # Test raises error
        with self.assertRaises(NotImplementedError):
            cubic.interpolate_function(pt, gaussian_pts, use_log=True, nu_x=2, nu_y=2)

    def test_integration_of_gaussian(self):
        r"""Test integration of a rapidly-decreasing Gaussian."""
        oned = MidPoint(250)
        cubic = Tensor1DGrids([oned, oned, oned])

        def gaussian(points):
            return np.exp(-6 * np.linalg.norm(points, axis=1)**2.0)

        gaussian_pts = gaussian(cubic.points)
        desired = np.sqrt(np.pi / 6)**3
        actual = cubic.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-3)

    def test_moving_coordinates_to_index_and_back(self):
        r"""Test moving from coordinates and index and back."""
        oned = MidPoint(3)
        cubic = Tensor1DGrids([oned, oned, oned])

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


class TestUniformCubicGrid(TestCase):
    r"""Test Uniform Cubic Grid Class."""
    def test_fourier1_weights_are_correct(self):
        r"""Test Fourier1 weights are correct against brute force."""
        origin = np.array([0., 0., 0.])
        axes = np.eye(3)
        shape = np.array([5, 6, 7], dtype=np.int)
        volume = 5 * 6 * 7  # Volume of cube centered at zero, moves in one step at a time (axes)
        uniform = UniformCubicGrid(origin, axes, shape=shape, weight_type="Fourier1")

        index = 0   # Index to iterate through uniform.weights.
        for j in range(1, shape[0] + 1):
            grid_x = np.arange(1, shape[0] + 1)
            desired_x = np.sum(
                np.sin(j * np.pi * grid_x / (shape[0] + 1)) * (1 - np.cos(grid_x * np.pi)) / (grid_x * np.pi)
            )
            for k in range(1, shape[1] + 1):

                grid_y = np.arange(1, shape[1] + 1)
                desired_y = np.sum(
                    np.sin(k * np.pi * grid_y / (shape[1] + 1)) * (
                                1 - np.cos(grid_y * np.pi)) / (grid_y * np.pi)
                )
                for l in range(1, shape[2] + 1):
                    grid_z = np.arange(1, shape[2] + 1)
                    desired_z = np.sum(
                        np.sin(l * np.pi * grid_z / (shape[2] + 1)) * (
                                1 - np.cos(grid_z * np.pi)) / (grid_z * np.pi)
                    )
                    desired = 8 * desired_x * desired_y * desired_z * volume / (6 * 7 * 8)
                    assert_allclose(uniform.weights[index], desired)
                    index += 1

    def test_fourier2_weights_are_correct(self):
        r"""Test that the Fourier2 weights are correct against brute force."""
        origin = np.array([0., 0., 0.])
        axes = np.eye(3)
        shape = np.array([5, 6, 7], dtype=np.int)
        volume = 5 * 6 * 7  # Volume of cube centered at zero, moves in one step at a time (axes)
        volume *= (4. / 5.) * (5. / 6) * (6. / 7)  # Alternative volume is used here.
        uniform = UniformCubicGrid(origin, axes, shape=shape, weight_type="Fourier2")
        index = 0  # Index to iterate through uniform.weights.
        for j in range(1, shape[0] + 1):
            # Calculate weight in the x-direction
            grid_x = np.arange(1, shape[0])
            desired_x = 2.0 * np.sin((j - 0.5) * np.pi) * np.sin(shape[0] * np.pi / 2)**2.0
            desired_x /= (shape[0]**2.0 * np.pi)
            desired_x += 4.0 * np.sum(
                np.sin((2.0 * j - 1.) * grid_x * np.pi / shape[0]) *
                np.sin(grid_x * np.pi / 2)**2.0 /
                grid_x
            ) / (shape[0] * np.pi)
            for k in range(1, shape[1] + 1):
                # Calculate weight in the y-direction
                grid_y = np.arange(1, shape[1])
                desired_y = 2.0 * np.sin((k - 0.5) * np.pi) * np.sin(shape[1] * np.pi / 2) ** 2.0
                desired_y /= (shape[1] ** 2.0 * np.pi)
                desired_y += 4.0 * np.sum(
                    np.sin((2.0 * k - 1.) * grid_y * np.pi / shape[1]) *
                    np.sin(grid_y * np.pi / 2) ** 2.0 /
                    grid_y
                ) / (shape[1] * np.pi)
                for l in range(1, shape[2] + 1):
                    # Calculate weight in the z-direction
                    grid_z = np.arange(1, shape[2])
                    desired_z = 2.0 * np.sin((l - 0.5) * np.pi) * np.sin(
                        shape[2] * np.pi / 2) ** 2.0
                    desired_z /= (shape[2] ** 2.0 * np.pi)
                    desired_z += 4.0 * np.sum(
                        np.sin((2.0 * l - 1.) * grid_z * np.pi / shape[2]) *
                        np.sin(grid_z * np.pi / 2) ** 2.0 /
                        grid_z
                    ) / (shape[2] * np.pi)
                    desired = desired_x * desired_y * desired_z * volume
                    assert_allclose(uniform.weights[index], desired)
                    index += 1

    def test_calculating_rectangle_weights(self):
        pass

    def test_calculating_trapezoid_weights(self):
        pass

    def test_calculating_alternative_weights(self):
        pass

    def test_integration_with_gaussian(self):
        origin = np.array([-1., -1., -1.])
        axes = np.eye(3) * 0.01
        shape = np.array([250, 250, 250], dtype=np.int)
        uniform = UniformCubicGrid(origin, axes, shape, weight_type="Rectangle")

        def gaussian(points):
            return np.exp(-6 * np.linalg.norm(points, axis=1) ** 2.0)

        gaussian_pts = gaussian(uniform.points)
        desired = np.sqrt(np.pi / 6) ** 3
        actual = uniform.integrate(gaussian_pts)
        assert_allclose(desired, actual, atol=1e-3)
