#!/usr/bin/env python3
"""
Test suite for the Adaptive1DGrid implementation.
"""
import numpy as np
import pytest
import os
import sys

# Path adjustments
try:
    from grid.adaptive_oned import Adaptive1DGrid
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from src.grid.adaptive_oned import Adaptive1DGrid


# --- Test Functions and Their Analytical Solutions (defined on [-1, 1]) ---

def polynomial_func(x):
    return x ** 6 - x ** 4 + 2 * x ** 2


ANALYTICAL_POLYNOMIAL = 2 * (1 / 7 - 1 / 5 + 2 / 3)


def peak_func(x):
    return 1.0 / (1.0 + (10 * x) ** 2)


ANALYTICAL_PEAK = np.arctan(10.0) / 5.0


# --- Pytest Fixture ---
@pytest.fixture
def default_adaptive_grid():
    """Provides a default Adaptive1DGrid instance for tests."""
    return Adaptive1DGrid(n_points_per_segment=10, tolerance=1e-9, max_iterations=20)


# --- Test Cases ---

def test_polynomial_accuracy(default_adaptive_grid):
    """Verifies high accuracy for polynomial integration."""
    print("\n--- Testing Polynomial Accuracy ---")
    grid = default_adaptive_grid
    grid.max_iterations = 1

    result, _ = grid.adaptive_integrate(polynomial_func)

    print(f"Analytical Result: {ANALYTICAL_POLYNOMIAL:.15f}")
    print(f"Numerical Result:  {result:.15f}")
    assert abs(result - ANALYTICAL_POLYNOMIAL) < 1e-14


def test_adaptive_convergence_on_peak_function(default_adaptive_grid):
    """Verifies the adaptive refinement process converges on a non-trivial function."""
    print("\n--- Testing Convergence on a Peaked Function ---")
    grid = default_adaptive_grid
    result, stats = grid.adaptive_integrate(peak_func)

    print(f"Analytical Result: {ANALYTICAL_PEAK:.14f}")
    print(f"Numerical Result:  {result:.14f}")
    print(f"Converged: {stats['converged']} in {stats['iterations']} iterations")

    assert stats['converged']
    assert abs(result - ANALYTICAL_PEAK) < grid.tolerance


if __name__ == "__main__":
    pytest.main([__file__])
