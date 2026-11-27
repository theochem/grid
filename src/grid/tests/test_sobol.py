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
r"""Tests for Sobol' Sequences."""

import os
import tempfile
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from grid.basegrid import Grid
from grid.sobol import Sobol


class TestSobol(TestCase):
    r"""Test Sobol class."""

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_raises_error_when_n_points_not_power_of_2(self):
        r"""Test that n_points must be a power of 2."""
        with self.assertRaises(ValueError) as err:
            Sobol(n_points=100, dimension=2)
        self.assertIn("must be a power of 2", str(err.exception))

    def test_raises_error_when_dimension_invalid(self):
        r"""Test that dimension must be >= 1."""
        with self.assertRaises(ValueError) as err:
            Sobol(n_points=1024, dimension=0)
        self.assertIn("must be >= 1", str(err.exception))

    def test_raises_error_for_invalid_origin(self):
        r"""Test that origin must have correct shape."""
        with self.assertRaises(ValueError) as err:
            Sobol(n_points=1024, dimension=3, origin=np.array([0, 0]))
        self.assertIn("origin must have shape (3,)", str(err.exception))

    def test_raises_error_for_invalid_axes(self):
        r"""Test that axes must have correct shape."""
        with self.assertRaises(ValueError) as err:
            Sobol(n_points=1024, dimension=3, axes=np.eye(2))
        self.assertIn("axes must have shape (3, 3)", str(err.exception))

    def test_raises_error_for_singular_axes(self):
        r"""Test that axes must be linearly independent."""
        singular_axes = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1]])
        with self.assertRaises(ValueError) as err:
            Sobol(n_points=1024, dimension=3, axes=singular_axes)
        self.assertIn("must be linearly independent", str(err.exception))

    # ------------------------------------------------------------------
    # Basic properties, weights, domain mapping
    # ------------------------------------------------------------------

    def test_properties(self):
        r"""Test that Sobol properties are correctly set."""
        n_points, dimension = 1024, 3
        sobol = Sobol(n_points=n_points, dimension=dimension, seed=0)

        assert_equal(sobol.size, n_points)
        assert_equal(sobol.n_points, n_points)
        assert_equal(sobol.dimension, dimension)
        assert_equal(sobol.randomize, True)
        assert_equal(sobol.points.shape, (n_points, dimension))
        assert_equal(sobol.weights.shape, (n_points,))
        assert_allclose(sobol.origin, np.zeros(dimension))
        assert_allclose(sobol.axes, np.eye(dimension))

    def test_weights_are_equal(self):
        r"""Test that all weights are equal to V/N."""
        n_points, dimension = 1024, 2
        sobol = Sobol(n_points=n_points, dimension=dimension, seed=0)

        expected_weight = 1.0 / n_points
        assert_allclose(sobol.weights, np.full(n_points, expected_weight))

    def test_weights_with_custom_axes(self):
        r"""Test that weights scale with volume."""
        n_points, dimension = 1024, 2
        axes = np.array([[2.0, 0.0], [0.0, 2.0]])
        sobol = Sobol(n_points=n_points, dimension=dimension, axes=axes, seed=0)

        expected_weight = 4.0 / n_points
        assert_allclose(sobol.weights, np.full(n_points, expected_weight))

    def test_points_in_unit_cube(self):
        r"""Test that points are in [0, 1)^d for default parameters."""
        sobol = Sobol(n_points=1024, dimension=3, seed=0)
        assert np.all(sobol.points >= 0.0)
        assert np.all(sobol.points < 1.0)

    def test_points_with_custom_origin_and_axes(self):
        r"""Test that points are correctly transformed."""
        n_points, dimension = 1024, 2
        origin = np.array([1.0, 2.0])
        axes = np.array([[0.5, 0.0], [0.0, 0.5]])
        sobol = Sobol(n_points=n_points, dimension=dimension, origin=origin, axes=axes, seed=0)

        assert np.all(sobol.points[:, 0] >= 1.0)
        assert np.all(sobol.points[:, 0] < 1.5)
        assert np.all(sobol.points[:, 1] >= 2.0)
        assert np.all(sobol.points[:, 1] < 2.5)

    def test_different_dimensions(self):
        r"""Test Sobol in different dimensions."""
        for dimension in [1, 2, 3, 5, 10]:
            n_points = 1024
            sobol = Sobol(n_points=n_points, dimension=dimension, seed=0)
            assert_equal(sobol.dimension, dimension)
            assert_equal(sobol.points.shape, (n_points, dimension))

    def test_save_and_load(self):
        r"""Test saving Sobol grid to file."""

        sobol = Sobol(n_points=1024, dimension=2, seed=0)

        fd, filename = tempfile.mkstemp(suffix=".npz")
        os.close(fd)

        try:
            sobol.save(filename)
            loaded = np.load(filename)
            assert_allclose(loaded["points"], sobol.points)
            assert_allclose(loaded["weights"], sobol.weights)
            loaded.close()
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    # ------------------------------------------------------------------
    # Integration accuracy
    # ------------------------------------------------------------------

    def test_integration_of_constant_function(self):
        r"""Test integration of f(x) = 1 gives volume."""
        n_points, dimension = 2048, 3
        axes = np.diag([2.0, 3.0, 4.0])  # Volume = 24
        sobol = Sobol(n_points=n_points, dimension=dimension, axes=axes, seed=0)

        func_vals = np.ones(n_points)
        integral = sobol.integrate(func_vals)
        assert_allclose(integral, 24.0, rtol=1e-10)

    def test_integration_of_linear_function(self):
        r"""Test integration of f(x) = x_1 + x_2 on unit square."""
        n_points, dimension = 4096, 2
        sobol = Sobol(n_points=n_points, dimension=dimension, seed=0)

        func_vals = sobol.points[:, 0] + sobol.points[:, 1]
        integral = sobol.integrate(func_vals)

        # Exact integral over [0,1]^2: int_0^1 int_0^1 (x+y) dx dy = 1
        assert_allclose(integral, 1.0, rtol=1e-2)

    def test_integration_of_quadratic_function(self):
        r"""Test integration of f(x) = x^2 on unit interval."""
        n_points, dimension = 8192, 1
        sobol = Sobol(n_points=n_points, dimension=dimension, seed=0)

        func_vals = sobol.points[:, 0] ** 2
        integral = sobol.integrate(func_vals)

        # Exact integral over [0,1]: int_0^1 x^2 dx = 1/3
        assert_allclose(integral, 1.0 / 3.0, rtol=1e-3)

    # ------------------------------------------------------------------
    # Properties SPECIFIC to Sobol' sequences
    # ------------------------------------------------------------------

    def test_first_point_is_origin_when_unrandomized(self):
        r"""Test that the first point (index 0) is the origin when randomize=False.

        This is a defining property of the unscrambled Sobol' construction:
        the initial point of the sequence is always the zero vector.
        """
        n_points, dimension = 1024, 3
        origin = np.array([1.0, 2.0, 3.0])
        sobol = Sobol(
            n_points=n_points,
            dimension=dimension,
            origin=origin,
            randomize=False,
            seed=0,
        )
        assert_allclose(sobol.points[0], origin)

    def test_randomize_true_and_false_differ(self):
        r"""Test that randomize=True and randomize=False give different points.

        With randomize=True, Owen scrambling is applied and even the first
        point of the sequence is no longer the origin.
        """
        n_points, dimension = 1024, 3
        sobol_scrambled = Sobol(n_points=n_points, dimension=dimension, seed=1, randomize=True)
        sobol_plain = Sobol(n_points=n_points, dimension=dimension, seed=1, randomize=False)

        assert not np.allclose(sobol_scrambled.points, sobol_plain.points)
        assert not np.allclose(sobol_scrambled.points[0], np.zeros(dimension))

    def test_reproducibility_same_seed(self):
        r"""Test that the same seed gives identical points."""
        sobol1 = Sobol(n_points=1024, dimension=2, seed=42)
        sobol2 = Sobol(n_points=1024, dimension=2, seed=42)
        assert_allclose(sobol1.points, sobol2.points)

    def test_different_seeds_give_different_points(self):
        r"""Test that different seeds give different points when randomize=True."""
        sobol1 = Sobol(n_points=1024, dimension=2, seed=42)
        sobol2 = Sobol(n_points=1024, dimension=2, seed=43)
        assert not np.allclose(sobol1.points, sobol2.points)

    def test_first_n_points_match_prefix_of_larger_sequence(self):
        r"""Test that the first N points exactly match the first N points of a 2N design.

        This nesting property lets a Sobol' design be extended to more points
        without discarding the ones already computed -- unlike a design that
        must be regenerated from scratch to add more points.
        """
        n_points, dimension = 512, 3
        sobol_n = Sobol(n_points=n_points, dimension=dimension, seed=0, randomize=False)
        sobol_2n = Sobol(n_points=2 * n_points, dimension=dimension, seed=0, randomize=False)

        assert_allclose(sobol_n.points, sobol_2n.points[:n_points])

    def test_getitem_returns_plain_grid(self):
        r"""Test that indexing returns a plain Grid, not a Sobol instance."""

        sobol = Sobol(n_points=1024, dimension=2, seed=0)

        single = sobol[3]
        assert isinstance(single, Grid)
        assert not isinstance(single, Sobol)
        assert_equal(single.points.shape, (1, 2))

        subset = sobol[5:10]
        assert isinstance(subset, Grid)
        assert not isinstance(subset, Sobol)
        assert_equal(subset.points.shape, (5, 2))
