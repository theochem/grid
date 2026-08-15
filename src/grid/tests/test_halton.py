"""Tests for Halton grids."""

import numpy as np
from numpy.testing import assert_allclose

from grid.halton import Halton


def test_halton_points():
    """Test the first points of a 2D Halton sequence."""
    grid = Halton(5, 2)

    expected = np.array(
        [
            [0.5, 1 / 3],
            [0.25, 2 / 3],
            [0.75, 1 / 9],
            [0.125, 4 / 9],
            [0.625, 7 / 9],
        ]
    )

    assert_allclose(grid.points, expected)


def test_halton_weights():
    """Test uniform integration weights."""
    grid = Halton(5, 2)

    assert_allclose(grid.weights, np.full(5, 0.2))


def test_halton_shape():
    """Test the shape of points and weights."""
    grid = Halton(10, 3)

    assert grid.points.shape == (10, 3)
    assert grid.weights.shape == (10,)


def test_halton_invalid_arguments():
    """Test invalid Halton arguments."""
    for npoints in [0, -1]:
        try:
            Halton(npoints, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

    for ndim in [0, -1]:
        try:
            Halton(5, ndim)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")