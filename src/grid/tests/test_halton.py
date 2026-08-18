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
"""Tests for Halton grids."""

import numpy as np
from numpy.testing import assert_allclose

from grid.halton import Halton


def test_halton_points():
    """Test the first points of a 2D Halton sequence."""
    grid = Halton(n_points=5, dimension=2)

    expected = np.array(
        [
            [0.0, 0.0],
            [0.5, 1 / 3],
            [0.25, 2 / 3],
            [0.75, 1 / 9],
            [0.125, 4 / 9],
        ]
    )

    assert_allclose(grid.points, expected)


def test_halton_weights():
    """Test uniform integration weights."""
    grid = Halton(n_points=5, dimension=2)

    assert_allclose(grid.weights, np.full(5, 0.2))


def test_halton_shape():
    """Test the shape of points and weights."""
    grid = Halton(n_points=10, dimension=3)

    assert grid.points.shape == (10, 3)
    assert grid.weights.shape == (10,)


def test_halton_properties():
    """Test read-only constructor properties."""
    grid = Halton(n_points=10, dimension=3)

    assert grid.n_points == 10
    assert grid.dimension == 3


def test_halton_indexing():
    """Test indexing and slicing."""
    grid = Halton(n_points=10, dimension=2)

    single = grid[3]
    subset = grid[2:5]

    assert single.points.shape == (1, 2)
    assert single.weights.shape == (1,)
    assert subset.points.shape == (3, 2)
    assert subset.weights.shape == (3,)


def test_halton_origin_and_axes():
    """Test mapping points onto a parallelepiped."""
    origin = np.array([1.0, 2.0])
    axes = np.array([[2.0, 0.0], [0.0, 3.0]])

    grid = Halton(
        n_points=5,
        dimension=2,
        origin=origin,
        axes=axes,
    )

    expected = origin + np.array(
        [
            [0.0, 0.0],
            [0.5, 1 / 3],
            [0.25, 2 / 3],
            [0.75, 1 / 9],
            [0.125, 4 / 9],
        ]
    ) @ axes.T

    assert_allclose(grid.points, expected)


def test_halton_scrambling():
    """Test reproducibility of scrambled Halton sequences."""
    grid1 = Halton(
        n_points=10,
        dimension=2,
        scramble=True,
        seed=42,
    )
    grid2 = Halton(
        n_points=10,
        dimension=2,
        scramble=True,
        seed=42,
    )

    assert_allclose(grid1.points, grid2.points)


def test_halton_invalid_arguments():
    """Test invalid Halton arguments."""
    for n_points in [0, -1]:
        try:
            Halton(n_points=n_points, dimension=2)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

    for dimension in [0, -1]:
        try:
            Halton(n_points=5, dimension=dimension)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

    try:
        Halton(n_points=5, dimension=2, origin=np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    try:
        Halton(n_points=5, dimension=2, axes=np.eye(3))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")