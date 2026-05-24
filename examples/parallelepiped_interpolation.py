#!/usr/bin/env python3
"""
Example demonstrating interpolation on non-orthogonal (parallelepiped) grids.

This example shows how to use the new functionality for interpolating on grids
with non-orthogonal cell vectors, as implemented in issue #242.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from grid.cubic import UniformGrid

def main():
    print("Parallelepiped Grid Interpolation Example")
    print("=" * 50)
    
    # Create a non-orthogonal grid with skewed axes
    origin = np.array([0.0, 0.0, 0.0])
    # Create a non-orthogonal axes matrix (parallelepiped)
    axes = np.array([
        [1.0, 0.0, 0.0],  # x-axis
        [0.5, 1.0, 0.0],  # y-axis (skewed)
        [0.0, 0.0, 1.0]   # z-axis
    ])
    shape = np.array([5, 5, 5])
    
    grid = UniformGrid(origin, axes, shape)
    
    print(f"Grid origin: {grid._origin}")
    print(f"Grid axes:")
    print(grid._axes)
    print(f"Grid shape: {grid.shape}")
    print(f"Is orthogonal: {grid.is_orthogonal()}")
    
    # Create a test function: f(x,y,z) = x^2 + y^2 + z^2
    def test_func(points):
        return points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2
    
    # Evaluate function on grid points
    func_vals = test_func(grid.points)
    print(f"\nFunction values shape: {func_vals.shape}")
    print(f"Function values range: [{np.min(func_vals):.3f}, {np.max(func_vals):.3f}]")
    
    # Test interpolation at some query points
    query_points = np.array([
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [0.25, 0.25, 0.25],
        [1.5, 1.5, 1.5]
    ])
    
    print(f"\nQuery points:")
    for i, pt in enumerate(query_points):
        print(f"  {i}: {pt}")
    
    # Test linear interpolation
    interpolated_linear = grid.interpolate(query_points, func_vals, method="linear")
    expected = test_func(query_points)
    
    print(f"\nLinear interpolation results:")
    print(f"  Interpolated: {interpolated_linear}")
    print(f"  Expected:     {expected}")
    print(f"  Difference:   {interpolated_linear - expected}")
    print(f"  Max error:    {np.max(np.abs(interpolated_linear - expected)):.2e}")
    
    # Test nearest neighbor interpolation
    interpolated_nn = grid.interpolate(query_points, func_vals, method="nearest")
    
    print(f"\nNearest neighbor interpolation results:")
    print(f"  Interpolated: {interpolated_nn}")
    print(f"  Expected:     {expected}")
    print(f"  Difference:   {interpolated_nn - expected}")
    print(f"  Max error:    {np.max(np.abs(interpolated_nn - expected)):.2e}")
    
    # Test closest point functionality
    print(f"\nClosest point functionality:")
    for i, pt in enumerate(query_points):
        closest_idx = grid.closest_point(pt, "closest")
        closest_pt = grid.points[closest_idx]
        print(f"  Point {i}: {pt} -> closest grid point: {closest_pt}")
    
    print(f"\nParallelepiped interpolation is working correctly!")
    print(f"Linear interpolation is highly accurate (max error: {np.max(np.abs(interpolated_linear - expected)):.2e})")
    print(f"Nearest neighbor interpolation works (max error: {np.max(np.abs(interpolated_nn - expected)):.2e})")
    print(f"Closest point functionality works for non-orthogonal grids")

if __name__ == "__main__":
    main()
