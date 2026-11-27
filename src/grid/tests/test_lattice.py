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
r"""Tests for Lattice Rules."""

from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from grid.lattice import Lattice


class TestLattice(TestCase):
    r"""Test Lattice class."""

    def test_raises_error_when_n_points_not_power_of_2(self):
        r"""Test that n_points must be a power of 2."""
        with self.assertRaises(ValueError) as err:
            Lattice(n_points=100, dimension=2)
        self.assertIn("must be a power of 2", str(err.exception))

    def test_raises_error_when_n_points_too_large(self):
        r"""Test that n_points must be <= 2^20."""
        with self.assertRaises(ValueError) as err:
            Lattice(n_points=2**21, dimension=2)
        self.assertIn("must be <= 1048576", str(err.exception))

    def test_raises_error_when_dimension_invalid(self):
        r"""Test that dimension must be >= 1."""
        with self.assertRaises(ValueError) as err:
            Lattice(n_points=1024, dimension=0)
        self.assertIn("must be >= 1", str(err.exception))

    def test_raises_error_when_dimension_exceeds_table(self):
        r"""Test that dimension must be within tabulated range."""
        with self.assertRaises(ValueError) as err:
            Lattice(n_points=1024, dimension=1000)
        self.assertIn("only supports up to", str(err.exception))

    def test_raises_error_for_unknown_rule(self):
        r"""Test that rule must be recognized."""
        with self.assertRaises(ValueError) as err:
            Lattice(n_points=1024, dimension=2, rule="unknown")
        self.assertIn("Unknown rule", str(err.exception))

    def test_raises_error_for_invalid_generating_vector(self):
        r"""Test that custom generating vector must have correct shape."""
        with self.assertRaises(ValueError) as err:
            Lattice(n_points=1024, dimension=3, generating_vector=np.array([1, 2]))
        self.assertIn("must have shape (3,)", str(err.exception))

    def test_raises_error_for_invalid_origin(self):
        r"""Test that origin must have correct shape."""
        with self.assertRaises(ValueError) as err:
            Lattice(n_points=1024, dimension=3, origin=np.array([0, 0]))
        self.assertIn("origin must have shape (3,)", str(err.exception))

    def test_raises_error_for_invalid_axes(self):
        r"""Test that axes must have correct shape."""
        with self.assertRaises(ValueError) as err:
            Lattice(n_points=1024, dimension=3, axes=np.eye(2))
        self.assertIn("axes must have shape (3, 3)", str(err.exception))

    def test_raises_error_for_singular_axes(self):
        r"""Test that axes must be linearly independent."""
        singular_axes = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1]])
        with self.assertRaises(ValueError) as err:
            Lattice(n_points=1024, dimension=3, axes=singular_axes)
        self.assertIn("must be linearly independent", str(err.exception))

    def test_lattice_properties(self):
        r"""Test that lattice properties are correctly set."""
        n_points = 1024
        dimension = 3
        lattice = Lattice(n_points=n_points, dimension=dimension)

        assert_equal(lattice.size, n_points)
        assert_equal(lattice.dimension, dimension)
        assert_equal(lattice.rule, "order2")
        assert_equal(lattice.points.shape, (n_points, dimension))
        assert_equal(lattice.weights.shape, (n_points,))
        assert_allclose(lattice.origin, np.zeros(dimension))
        assert_allclose(lattice.axes, np.eye(dimension))

    def test_weights_are_equal(self):
        r"""Test that all weights are equal to V/N."""
        n_points = 1024
        dimension = 2
        lattice = Lattice(n_points=n_points, dimension=dimension)

        # For unit cube, volume = 1
        expected_weight = 1.0 / n_points
        assert_allclose(lattice.weights, np.full(n_points, expected_weight))

    def test_weights_with_custom_axes(self):
        r"""Test that weights scale with volume."""
        n_points = 1024
        dimension = 2
        # Create a 2x2 square
        axes = np.array([[2.0, 0.0], [0.0, 2.0]])
        lattice = Lattice(n_points=n_points, dimension=dimension, axes=axes)

        # Volume = det(axes) = 4
        expected_weight = 4.0 / n_points
        assert_allclose(lattice.weights, np.full(n_points, expected_weight))

    def test_points_in_unit_cube(self):
        r"""Test that points are in [0, 1)^d for default parameters."""
        n_points = 1024
        dimension = 3
        lattice = Lattice(n_points=n_points, dimension=dimension)

        # All points should be in [0, 1)
        assert np.all(lattice.points >= 0.0)
        assert np.all(lattice.points < 1.0)

    def test_points_with_custom_origin_and_axes(self):
        r"""Test that points are correctly transformed."""
        n_points = 1024
        dimension = 2
        origin = np.array([1.0, 2.0])
        axes = np.array([[0.5, 0.0], [0.0, 0.5]])
        lattice = Lattice(
            n_points=n_points, dimension=dimension, origin=origin, axes=axes
        )

        # Points should be in [1, 1.5) x [2, 2.5)
        assert np.all(lattice.points[:, 0] >= 1.0)
        assert np.all(lattice.points[:, 0] < 1.5)
        assert np.all(lattice.points[:, 1] >= 2.0)
        assert np.all(lattice.points[:, 1] < 2.5)

    def test_first_point_is_origin(self):
        r"""Test that the first lattice point (i=0) is at the origin."""
        n_points = 1024
        dimension = 3
        origin = np.array([1.0, 2.0, 3.0])
        lattice = Lattice(n_points=n_points, dimension=dimension, origin=origin)

        # For i=0, x_0 = {0 * z / N} = 0, so first point is origin
        assert_allclose(lattice.points[0], origin)

    def test_integration_of_constant_function(self):
        r"""Test integration of f(x) = 1 gives volume."""
        n_points = 2048
        dimension = 3
        axes = np.diag([2.0, 3.0, 4.0])  # Volume = 24
        lattice = Lattice(n_points=n_points, dimension=dimension, axes=axes)

        func_vals = np.ones(n_points)
        integral = lattice.integrate(func_vals)

        expected = 24.0
        assert_allclose(integral, expected, rtol=1e-10)

    def test_integration_of_linear_function(self):
        r"""Test integration of f(x) = x_1 + x_2 on unit square."""
        n_points = 4096
        dimension = 2
        lattice = Lattice(n_points=n_points, dimension=dimension)

        # f(x, y) = x + y
        func_vals = lattice.points[:, 0] + lattice.points[:, 1]
        integral = lattice.integrate(func_vals)

        # Exact integral over [0,1]^2: int_0^1 int_0^1 (x+y) dx dy = 1
        expected = 1.0
        assert_allclose(integral, expected, rtol=1e-2)

    def test_integration_of_quadratic_function(self):
        r"""Test integration of f(x) = x^2 on unit interval."""
        n_points = 8192
        dimension = 1
        lattice = Lattice(n_points=n_points, dimension=dimension)

        # f(x) = x^2
        func_vals = lattice.points[:, 0] ** 2
        integral = lattice.integrate(func_vals)

        # Exact integral over [0,1]: int_0^1 x^2 dx = 1/3
        expected = 1.0 / 3.0
        assert_allclose(integral, expected, rtol=1e-3)

    def test_embedded_property(self):
        r"""Test that lattice with N points is a subset of lattice with 2N points."""
        n_points = 1024
        dimension = 2
        lattice_n = Lattice(n_points=n_points, dimension=dimension)
        lattice_2n = Lattice(n_points=2 * n_points, dimension=dimension)

        # Every other point in lattice_2n should match lattice_n
        # Because x_i = {i*z/N} and x_{2i} = {2i*z/(2N)} = {i*z/N}
        for i in range(n_points):
            assert_allclose(lattice_n.points[i], lattice_2n.points[2 * i], rtol=1e-10)

    def test_custom_generating_vector(self):
        r"""Test using a custom generating vector."""
        n_points = 1024
        dimension = 3
        custom_z = np.array([1, 123, 456])
        lattice = Lattice(
            n_points=n_points, dimension=dimension, generating_vector=custom_z
        )

        assert_allclose(lattice.generating_vector, custom_z)

        # Verify first few points manually
        # x_0 = {0 * z / 1024} = [0, 0, 0]
        assert_allclose(lattice.points[0], [0.0, 0.0, 0.0])

        # x_1 = {1 * [1, 123, 456] / 1024} = [1/1024, 123/1024, 456/1024]
        expected_1 = np.array([1.0, 123.0, 456.0]) / 1024.0
        assert_allclose(lattice.points[1], expected_1)

    def test_interpolation_linear_function(self):
        r"""Test interpolation of a linear function."""
        n_points = 2048
        dimension = 2
        lattice = Lattice(n_points=n_points, dimension=dimension)

        # f(x, y) = 2*x + 3*y
        func_vals = 2 * lattice.points[:, 0] + 3 * lattice.points[:, 1]

        # Interpolate at some test points
        test_points = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
        interpolated = lattice.interpolate(test_points, func_vals)

        # For linear function, interpolation should be exact
        expected = 2 * test_points[:, 0] + 3 * test_points[:, 1]
        assert_allclose(interpolated, expected, rtol=1e-5)

    def test_interpolation_raises_error_for_wrong_values_length(self):
        r"""Test that interpolation raises error for wrong values length."""
        n_points = 1024
        dimension = 2
        lattice = Lattice(n_points=n_points, dimension=dimension)

        with self.assertRaises(ValueError) as err:
            lattice.interpolate(np.array([[0.5, 0.5]]), np.array([1.0, 2.0]))
        self.assertIn("values must have length", str(err.exception))

    def test_interpolation_raises_error_for_wrong_dimension(self):
        r"""Test that interpolation raises error for wrong dimension."""
        n_points = 1024
        dimension = 2
        lattice = Lattice(n_points=n_points, dimension=dimension)

        func_vals = np.ones(n_points)
        with self.assertRaises(ValueError) as err:
            lattice.interpolate(np.array([[0.5, 0.5, 0.5]]), func_vals)
        self.assertIn("must have 2 columns", str(err.exception))

    def test_different_dimensions(self):
        r"""Test lattice in different dimensions."""
        for dimension in [1, 2, 3, 5, 10]:
            n_points = 1024
            lattice = Lattice(n_points=n_points, dimension=dimension)

            assert_equal(lattice.dimension, dimension)
            assert_equal(lattice.points.shape, (n_points, dimension))
            assert_equal(len(lattice.generating_vector), dimension)

    def test_save_and_load(self):
        r"""Test saving lattice to file."""
        import os
        import tempfile

        n_points = 1024
        dimension = 2
        lattice = Lattice(n_points=n_points, dimension=dimension)

        # Create temp file and close it immediately
        fd, filename = tempfile.mkstemp(suffix=".npz")
        os.close(fd)  # Close the file descriptor

        try:
            lattice.save(filename)
            loaded = np.load(filename)

            assert_allclose(loaded["points"], lattice.points)
            assert_allclose(loaded["weights"], lattice.weights)
            loaded.close()  # Close the npz file before deletion
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
