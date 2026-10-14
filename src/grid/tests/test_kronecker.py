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
r"""Tests for Kronecker (Weyl) Sequences."""

import os
import tempfile
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from grid.basegrid import Grid
from grid.kronecker import Kronecker


class TestKronecker(TestCase):
    r"""Test Kronecker class."""

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_raises_error_when_n_points_invalid(self):
        r"""Test that n_points must be >= 1."""
        with self.assertRaises(ValueError) as err:
            Kronecker(n_points=0, dimension=2)
        self.assertIn("must be >= 1", str(err.exception))

    def test_raises_error_when_dimension_invalid(self):
        r"""Test that dimension must be >= 1."""
        with self.assertRaises(ValueError) as err:
            Kronecker(n_points=100, dimension=0)
        self.assertIn("must be >= 1", str(err.exception))

    def test_raises_error_for_invalid_origin(self):
        r"""Test that origin must have correct shape."""
        with self.assertRaises(ValueError) as err:
            Kronecker(n_points=100, dimension=3, origin=np.array([0, 0]))
        self.assertIn("origin must have shape (3,)", str(err.exception))

    def test_raises_error_for_invalid_axes(self):
        r"""Test that axes must have correct shape."""
        with self.assertRaises(ValueError) as err:
            Kronecker(n_points=100, dimension=3, axes=np.eye(2))
        self.assertIn("axes must have shape (3, 3)", str(err.exception))

    def test_raises_error_for_singular_axes(self):
        r"""Test that axes must be linearly independent."""
        singular_axes = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1]])
        with self.assertRaises(ValueError) as err:
            Kronecker(n_points=100, dimension=3, axes=singular_axes)
        self.assertIn("must be linearly independent", str(err.exception))

    # ------------------------------------------------------------------
    # Basic properties, weights, domain mapping
    # ------------------------------------------------------------------

    def test_properties(self):
        r"""Test that Kronecker properties are correctly set."""
        n_points, dimension = 100, 3
        kro = Kronecker(n_points=n_points, dimension=dimension, seed=0)

        assert_equal(kro.size, n_points)
        assert_equal(kro.n_points, n_points)
        assert_equal(kro.dimension, dimension)
        assert_equal(kro.randomize, True)
        assert_equal(kro.points.shape, (n_points, dimension))
        assert_equal(kro.weights.shape, (n_points,))
        assert_allclose(kro.origin, np.zeros(dimension))
        assert_allclose(kro.axes, np.eye(dimension))

    def test_weights_are_equal(self):
        r"""Test that all weights are equal to V/N."""
        n_points, dimension = 100, 2
        kro = Kronecker(n_points=n_points, dimension=dimension, seed=0)

        expected_weight = 1.0 / n_points
        assert_allclose(kro.weights, np.full(n_points, expected_weight))

    def test_weights_with_custom_axes(self):
        r"""Test that weights scale with volume."""
        n_points, dimension = 100, 2
        axes = np.array([[2.0, 0.0], [0.0, 2.0]])
        kro = Kronecker(n_points=n_points, dimension=dimension, axes=axes, seed=0)

        expected_weight = 4.0 / n_points
        assert_allclose(kro.weights, np.full(n_points, expected_weight))

    def test_points_in_unit_cube(self):
        r"""Test that points are in [0, 1)^d for default parameters."""
        kro = Kronecker(n_points=100, dimension=3, seed=0)
        assert np.all(kro.points >= 0.0)
        assert np.all(kro.points < 1.0)

    def test_points_with_custom_origin_and_axes(self):
        r"""Test that points are correctly transformed."""
        n_points, dimension = 100, 2
        origin = np.array([1.0, 2.0])
        axes = np.array([[0.5, 0.0], [0.0, 0.5]])
        kro = Kronecker(n_points=n_points, dimension=dimension, origin=origin, axes=axes, seed=0)

        assert np.all(kro.points[:, 0] >= 1.0)
        assert np.all(kro.points[:, 0] < 1.5)
        assert np.all(kro.points[:, 1] >= 2.0)
        assert np.all(kro.points[:, 1] < 2.5)

    def test_integration_of_constant_function(self):
        r"""Test integration of f(x) = 1 gives volume."""
        n_points, dimension = 2048, 3
        axes = np.diag([2.0, 3.0, 4.0])  # Volume = 24
        kro = Kronecker(n_points=n_points, dimension=dimension, axes=axes, seed=0)

        func_vals = np.ones(n_points)
        integral = kro.integrate(func_vals)
        assert_allclose(integral, 24.0, rtol=1e-10)

    def test_integration_of_linear_function(self):
        r"""Test integration of f(x) = x_1 + x_2 on unit square."""
        n_points, dimension = 4096, 2
        kro = Kronecker(n_points=n_points, dimension=dimension, seed=0)

        func_vals = kro.points[:, 0] + kro.points[:, 1]
        integral = kro.integrate(func_vals)

        # Exact integral over [0,1]^2: int_0^1 int_0^1 (x+y) dx dy = 1
        assert_allclose(integral, 1.0, rtol=1e-2)

    def test_save_and_load(self):
        r"""Test saving Kronecker grid to file."""
        kro = Kronecker(n_points=100, dimension=2, seed=0)

        fd, filename = tempfile.mkstemp(suffix=".npz")
        os.close(fd)

        try:
            kro.save(filename)
            loaded = np.load(filename)
            assert_allclose(loaded["points"], kro.points)
            assert_allclose(loaded["weights"], kro.weights)
            loaded.close()
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_different_dimensions(self):
        r"""Test Kronecker in different dimensions."""
        for dimension in [1, 2, 3, 5, 10]:
            n_points = 100
            kro = Kronecker(n_points=n_points, dimension=dimension, seed=0)
            assert_equal(kro.dimension, dimension)
            assert_equal(kro.points.shape, (n_points, dimension))

    def test_n_points_need_not_be_special(self):
        r"""Test that Kronecker accepts any n_points (no power-of-2 or other constraint)."""
        for n_points in [7, 100, 123, 1000]:
            kro = Kronecker(n_points=n_points, dimension=3, seed=0)
            assert_equal(kro.points.shape, (n_points, 3))

    def test_reproducibility_same_seed(self):
        r"""Test that the same seed gives identical points."""
        kro1 = Kronecker(n_points=30, dimension=2, seed=42)
        kro2 = Kronecker(n_points=30, dimension=2, seed=42)
        assert_allclose(kro1.points, kro2.points)

    def test_different_seeds_give_different_points(self):
        r"""Test that different seeds give different points when randomize=True."""
        kro1 = Kronecker(n_points=30, dimension=2, seed=42)
        kro2 = Kronecker(n_points=30, dimension=2, seed=43)
        assert not np.allclose(kro1.points, kro2.points)

    def test_getitem_returns_plain_grid(self):
        r"""Test that indexing returns a plain Grid, not a Kronecker instance."""
        kro = Kronecker(n_points=50, dimension=2, seed=0)

        single = kro[3]
        assert isinstance(single, Grid)
        assert not isinstance(single, Kronecker)
        assert_equal(single.points.shape, (1, 2))

        subset = kro[5:10]
        assert isinstance(subset, Grid)
        assert not isinstance(subset, Kronecker)
        assert_equal(subset.points.shape, (5, 2))

    # ------------------------------------------------------------------
    # Properties SPECIFIC to Kronecker sequences
    # ------------------------------------------------------------------

    def test_matches_sqrt_prime_construction(self):
        r"""Test the construction against an independent calculation using sqrt(2),sqrt(3), sqrt(5).

        This checks the defining formula directly: x_i[j] = {i * sqrt(p_j)},
        where p_j is the j-th prime, rather than only checking downstream
        properties of the generated points.
        """
        n_points, dimension = 20, 3
        kro = Kronecker(n_points=n_points, dimension=dimension, randomize=False)

        p1, p2, p3 = np.sqrt(2), np.sqrt(3), np.sqrt(5)
        expected = np.array([[(i * p1) % 1, (i * p2) % 1, (i * p3) % 1] for i in range(n_points)])
        assert_allclose(kro.points, expected)

    def test_first_point_is_origin_when_unrandomized(self):
        r"""Test that the first point (index 0) is the origin when randomize=False.

        This follows directly from the defining formula: x_0 = {0 * alpha} = 0
        for any choice of alpha, so the first point is always the origin
        before a random shift is applied.
        """
        n_points, dimension = 100, 3
        origin = np.array([1.0, 2.0, 3.0])
        kro = Kronecker(
            n_points=n_points,
            dimension=dimension,
            origin=origin,
            randomize=False,
            seed=0,
        )
        assert_allclose(kro.points[0], origin)

    def test_randomize_true_and_false_differ(self):
        r"""Test that randomize=True and randomize=False give different points.

        With randomize=True, a random Cranley-Patterson shift is applied, so
        even the first point of the sequence is no longer the origin.
        """
        n_points, dimension = 100, 3
        kro_shifted = Kronecker(n_points=n_points, dimension=dimension, seed=1, randomize=True)
        kro_plain = Kronecker(n_points=n_points, dimension=dimension, seed=1, randomize=False)

        assert not np.allclose(kro_shifted.points, kro_plain.points)
        assert not np.allclose(kro_shifted.points[0], np.zeros(dimension))

    def test_prefix_is_identical_regardless_of_total_n_points(self):
        r"""Test that the first N points do not depend on the total number requested.

        Because x_i = {i * alpha} depends only on the index i and not on the
        total sample size N, a design of size N is an exact prefix of a
        design of size M > N with the same seed and randomization -- unlike
        constructions whose point positions are defined relative to N itself.
        """
        n_points, dimension = 100, 2
        kro_n = Kronecker(n_points=n_points, dimension=dimension, seed=0, randomize=False)
        kro_2n = Kronecker(n_points=2 * n_points, dimension=dimension, seed=0, randomize=False)

        assert_allclose(kro_n.points, kro_2n.points[:n_points])

    def test_prefix_property_holds_with_randomization(self):
        r"""Test that the prefix property also holds under a Cranley-Patterson shift.

        The same random shift is applied to every point regardless of the
        total sample size, so the nesting property survives randomization
        as long as the seed is the same.
        """
        n_points, dimension = 100, 2
        kro_n = Kronecker(n_points=n_points, dimension=dimension, seed=7, randomize=True)
        kro_2n = Kronecker(n_points=2 * n_points, dimension=dimension, seed=7, randomize=True)

        assert_allclose(kro_n.points, kro_2n.points[:n_points])
