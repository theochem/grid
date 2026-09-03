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
r"""Tests for Latin Hypercube Sampling."""

import os
import tempfile
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from grid.basegrid import Grid
from grid.latin_hypercube import LatinHypercube


class TestLatinHypercube(TestCase):
    r"""Test LatinHypercube class."""

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_raises_error_when_n_points_invalid(self):
        r"""Test that n_points must be >= 1."""
        with self.assertRaises(ValueError) as err:
            LatinHypercube(n_points=0, dimension=2)
        self.assertIn("must be >= 1", str(err.exception))

    def test_raises_error_when_dimension_invalid(self):
        r"""Test that dimension must be >= 1."""
        with self.assertRaises(ValueError) as err:
            LatinHypercube(n_points=100, dimension=0)
        self.assertIn("must be >= 1", str(err.exception))

    def test_raises_error_for_invalid_origin(self):
        r"""Test that origin must have correct shape."""
        with self.assertRaises(ValueError) as err:
            LatinHypercube(n_points=100, dimension=3, origin=np.array([0, 0]))
        self.assertIn("origin must have shape (3,)", str(err.exception))

    def test_raises_error_for_invalid_axes(self):
        r"""Test that axes must have correct shape."""
        with self.assertRaises(ValueError) as err:
            LatinHypercube(n_points=100, dimension=3, axes=np.eye(2))
        self.assertIn("axes must have shape (3, 3)", str(err.exception))

    def test_raises_error_for_singular_axes(self):
        r"""Test that axes must be linearly independent."""
        singular_axes = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1]])
        with self.assertRaises(ValueError) as err:
            LatinHypercube(n_points=100, dimension=3, axes=singular_axes)
        self.assertIn("must be linearly independent", str(err.exception))

    # ------------------------------------------------------------------
    # Basic properties, weights, domain mapping
    # ------------------------------------------------------------------

    def test_properties(self):
        r"""Test that LHS properties are correctly set."""
        n_points, dimension = 100, 3
        lhs = LatinHypercube(n_points=n_points, dimension=dimension, seed=0)

        assert_equal(lhs.size, n_points)
        assert_equal(lhs.n_points, n_points)
        assert_equal(lhs.dimension, dimension)
        assert_equal(lhs.randomize, True)
        assert_equal(lhs.points.shape, (n_points, dimension))
        assert_equal(lhs.weights.shape, (n_points,))
        assert_allclose(lhs.origin, np.zeros(dimension))
        assert_allclose(lhs.axes, np.eye(dimension))

    def test_weights_are_equal(self):
        r"""Test that all weights are equal to V/N."""
        n_points, dimension = 100, 2
        lhs = LatinHypercube(n_points=n_points, dimension=dimension, seed=0)

        expected_weight = 1.0 / n_points
        assert_allclose(lhs.weights, np.full(n_points, expected_weight))

    def test_weights_with_custom_axes(self):
        r"""Test that weights scale with volume."""
        n_points, dimension = 100, 2
        axes = np.array([[2.0, 0.0], [0.0, 2.0]])
        lhs = LatinHypercube(n_points=n_points, dimension=dimension, axes=axes, seed=0)

        expected_weight = 4.0 / n_points
        assert_allclose(lhs.weights, np.full(n_points, expected_weight))

    def test_points_in_unit_cube(self):
        r"""Test that points are in [0, 1)^d for default parameters."""
        lhs = LatinHypercube(n_points=100, dimension=3, seed=0)
        assert np.all(lhs.points >= 0.0)
        assert np.all(lhs.points < 1.0)

    def test_points_with_custom_origin_and_axes(self):
        r"""Test that points are correctly transformed."""
        n_points, dimension = 100, 2
        origin = np.array([1.0, 2.0])
        axes = np.array([[0.5, 0.0], [0.0, 0.5]])
        lhs = LatinHypercube(
            n_points=n_points, dimension=dimension, origin=origin, axes=axes, seed=0
        )

        assert np.all(lhs.points[:, 0] >= 1.0)
        assert np.all(lhs.points[:, 0] < 1.5)
        assert np.all(lhs.points[:, 1] >= 2.0)
        assert np.all(lhs.points[:, 1] < 2.5)

    def test_integration_of_constant_function(self):
        r"""Test integration of f(x) = 1 gives volume."""
        n_points, dimension = 2048, 3
        axes = np.diag([2.0, 3.0, 4.0])  # Volume = 24
        lhs = LatinHypercube(n_points=n_points, dimension=dimension, axes=axes, seed=0)

        func_vals = np.ones(n_points)
        integral = lhs.integrate(func_vals)
        assert_allclose(integral, 24.0, rtol=1e-10)

    def test_integration_of_linear_function(self):
        r"""Test integration of f(x) = x_1 + x_2 on unit square."""
        n_points, dimension = 4096, 2
        lhs = LatinHypercube(n_points=n_points, dimension=dimension, seed=0)

        func_vals = lhs.points[:, 0] + lhs.points[:, 1]
        integral = lhs.integrate(func_vals)

        # Exact integral over [0,1]^2: int_0^1 int_0^1 (x+y) dx dy = 1
        assert_allclose(integral, 1.0, rtol=1e-2)

    def test_save_and_load(self):
        r"""Test saving LHS grid to file."""

        lhs = LatinHypercube(n_points=100, dimension=2, seed=0)

        fd, filename = tempfile.mkstemp(suffix=".npz")
        os.close(fd)

        try:
            lhs.save(filename)
            loaded = np.load(filename)
            assert_allclose(loaded["points"], lhs.points)
            assert_allclose(loaded["weights"], lhs.weights)
            loaded.close()
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_different_dimensions(self):
        r"""Test LHS in different dimensions."""
        for dimension in [1, 2, 3, 5, 10]:
            n_points = 100
            lhs = LatinHypercube(n_points=n_points, dimension=dimension, seed=0)
            assert_equal(lhs.dimension, dimension)
            assert_equal(lhs.points.shape, (n_points, dimension))

    # ------------------------------------------------------------------
    # Properties SPECIFIC to Latin Hypercube Sampling
    # ------------------------------------------------------------------

    def test_stratification_property(self):
        r"""Test the core LHS property: exactly one point per stratum, in every dimension."""
        n_points, dimension = 100, 4
        lhs = LatinHypercube(n_points=n_points, dimension=dimension, seed=0)

        for j in range(dimension):
            strata = np.floor(lhs.points[:, j] * n_points).astype(int)
            assert_equal(sorted(strata.tolist()), list(range(n_points)))

    def test_stratification_property_with_custom_domain(self):
        r"""Test that stratification still holds after mapping to a custom parallelepiped."""
        n_points, dimension = 100, 2
        origin = np.array([1.0, 2.0])
        axes = np.array([[3.0, 0.0], [0.0, 5.0]])
        lhs = LatinHypercube(
            n_points=n_points, dimension=dimension, origin=origin, axes=axes, seed=0
        )

        # Map back to [0, 1)^d before checking stratification
        unit_points = (lhs.points - origin) @ np.linalg.inv(axes)
        for j in range(dimension):
            strata = np.floor(unit_points[:, j] * n_points).astype(int)
            assert_equal(sorted(strata.tolist()), list(range(n_points)))

    def test_randomize_true_and_false_differ(self):
        r"""Regression test: randomize=True and randomize=False must give different points.

        This guards against a real bug found during development, where an internal
        string comparison (``randomize == "TRUE"``) silently ignored the boolean
        ``randomize`` argument, making the class always behave as if
        ``randomize=False`` regardless of what was requested.
        """
        n_points, dimension = 50, 3
        lhs_true = LatinHypercube(n_points=n_points, dimension=dimension, seed=1, randomize=True)
        lhs_false = LatinHypercube(n_points=n_points, dimension=dimension, seed=1, randomize=False)

        assert not np.allclose(lhs_true.points, lhs_false.points)

    def test_not_randomized_points_are_stratum_centers(self):
        r"""Test that randomize=False places every point exactly at its stratum center."""
        n_points, dimension = 20, 3
        lhs = LatinHypercube(n_points=n_points, dimension=dimension, seed=2, randomize=False)

        centered = lhs.points * n_points + 0.5
        assert_allclose(centered, np.round(centered), atol=1e-10)

    def test_reproducibility_same_seed(self):
        r"""Test that the same seed gives identical points."""
        lhs1 = LatinHypercube(n_points=30, dimension=2, seed=42)
        lhs2 = LatinHypercube(n_points=30, dimension=2, seed=42)
        assert_allclose(lhs1.points, lhs2.points)

    def test_different_seeds_give_different_points(self):
        r"""Test that different seeds give different points when randomize=True."""
        lhs1 = LatinHypercube(n_points=30, dimension=2, seed=42)
        lhs2 = LatinHypercube(n_points=30, dimension=2, seed=43)
        assert not np.allclose(lhs1.points, lhs2.points)

    def test_n_points_need_not_be_power_of_2(self):
        r"""Test that, unlike Lattice, LHS accepts any n_points (no power-of-2 constraint)."""
        for n_points in [7, 100, 123, 1000]:
            lhs = LatinHypercube(n_points=n_points, dimension=3, seed=0)
            assert_equal(lhs.points.shape, (n_points, 3))

    def test_not_nested_unlike_lattice(self):
        r"""Test that LHS designs of different sizes are NOT related by subsetting.

        Unlike Lattice (where a lattice of N points is exactly embedded in a lattice
        of 2N points, since x_i = {i*z/N} = {2i*z/(2N)}), LHS is not a nested/extensible
        sequence: the strata boundaries themselves depend on n_points, so a design with
        N points bears no fixed relationship to a design with 2N points.
        """
        n_points, dimension = 50, 2
        lhs_n = LatinHypercube(n_points=n_points, dimension=dimension, seed=0)
        lhs_2n = LatinHypercube(n_points=2 * n_points, dimension=dimension, seed=0)

        # No systematic relationship should hold between the two point sets.
        assert not np.allclose(lhs_n.points, lhs_2n.points[::2])

    def test_getitem_returns_plain_grid(self):
        r"""Test that indexing returns a plain Grid, not a LatinHypercube.

        A subset of an LHS design is not itself a valid LHS design (the
        stratification property no longer holds), so __getitem__ must not
        return a LatinHypercube instance.
        """

        lhs = LatinHypercube(n_points=50, dimension=2, seed=0)

        single = lhs[3]
        assert isinstance(single, Grid)
        assert not isinstance(single, LatinHypercube)
        assert_equal(single.points.shape, (1, 2))

        subset = lhs[5:10]
        assert isinstance(subset, Grid)
        assert not isinstance(subset, LatinHypercube)
        assert_equal(subset.points.shape, (5, 2))
