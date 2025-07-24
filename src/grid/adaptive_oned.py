#!/usr/bin/env python3
"""
An adaptive one-dimensional grid for numerical integration on the [-1, 1] interval.

This module implements a 1D grid that automatically refines itself in regions
where the integrand shows complex behavior, aiming to achieve a target
accuracy with minimal function evaluations.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, List, Dict
from dataclasses import dataclass

from grid.onedgrid import GaussLegendre
from grid.basegrid import OneDGrid


@dataclass
class Segment:
    """Represents a sub-interval of the main integration domain, holding a scaled grid."""
    grid: OneDGrid

    @property
    def start(self) -> float:
        return self.grid.domain[0]

    @property
    def end(self) -> float:
        return self.grid.domain[1]

    @property
    def integral_contribution(self) -> float:
        return self._integral_contribution

    @integral_contribution.setter
    def integral_contribution(self, value: float):
        self._integral_contribution = value

    @classmethod
    def create_from_interval(cls, start: float, end: float, n_points: int) -> Segment:
        """Creates a Segment by scaling a GaussLegendre grid to the given interval."""
        base_grid = GaussLegendre(n_points)
        points = (base_grid.points + 1) * (end - start) / 2.0 + start
        weights = base_grid.weights * (end - start) / 2.0
        final_grid = OneDGrid(points, weights, (start, end))
        return cls(grid=final_grid)


class Adaptive1DGrid:
    """
    Manages a 1D adaptive grid on the fixed interval [-1, 1].
    Refinement is achieved by bisecting the segment with the largest error estimate.
    """

    def __init__(self, n_points_per_segment: int = 16, tolerance: float = 1e-8,
                 max_iterations: int = 15):
        """
        Initializes the adaptive grid. The integration domain is fixed to [-1, 1].

        Args:
            n_points_per_segment: The number of points for each base Gauss-Legendre grid.
            tolerance: The desired accuracy for the integration result.
            max_iterations: The maximum number of refinement steps.
        """
        self.n_points_per_segment = n_points_per_segment
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.domain = (-1.0, 1.0)  # Domain is now fixed.

        initial_segment = Segment.create_from_interval(self.domain[0], self.domain[1],
                                                       self.n_points_per_segment)
        self.segments: List[Segment] = [initial_segment]
        self.func_cache: Dict[float, float] = {}

    @property
    def grid(self) -> OneDGrid:
        """Returns a single OneDGrid object representing the composite grid."""
        all_points = np.concatenate([s.grid.points for s in self.segments])
        all_weights = np.concatenate([s.grid.weights for s in self.segments])
        sort_indices = np.argsort(all_points)
        return OneDGrid(all_points[sort_indices], all_weights[sort_indices], self.domain)

    def integrate(self, func: Callable[[float], float]) -> float:
        """Calculates the integral by summing contributions from all segments."""
        total_integral = 0.0
        for segment in self.segments:
            points = segment.grid.points
            values = np.zeros_like(points)
            for i, p in enumerate(points):
                if p not in self.func_cache:
                    self.func_cache[p] = func(p)
                values[i] = self.func_cache[p]

            segment.integral_contribution = segment.grid.integrate(values)
            total_integral += segment.integral_contribution
        return total_integral

    def _refine(self):
        """Finds the segment with the largest integral contribution and bisects it."""
        errors = [abs(s.integral_contribution) for s in self.segments]
        target_index = np.argmax(errors)

        target_segment = self.segments.pop(target_index)
        start, end = target_segment.start, target_segment.end
        midpoint = (start + end) / 2.0

        segment1 = Segment.create_from_interval(start, midpoint, self.n_points_per_segment)
        segment2 = Segment.create_from_interval(midpoint, end, self.n_points_per_segment)
        self.segments.extend([segment1, segment2])

    def adaptive_integrate(self, func: Callable[[float], float]) -> tuple[float, dict]:
        """Performs the full adaptive integration workflow until convergence."""
        history = []
        integral_old = self.integrate(func)
        history.append(integral_old)
        error = float('inf')

        for iteration in range(self.max_iterations):
            self._refine()
            integral_new = self.integrate(func)
            history.append(integral_new)

            error = abs(integral_new - integral_old)
            if error < self.tolerance:
                print(f"Convergence achieved after {iteration + 1} iterations.")
                break
            integral_old = integral_new
        else:
            print(
                f"Warning: Maximum iterations ({self.max_iterations}) reached without convergence.")

        stats = {
            "converged": error < self.tolerance,
            "final_result": integral_new,
            "error_estimate": error,
            "iterations": len(history) - 1,
            "num_segments": len(self.segments),
            "total_points": len(self.segments) * self.n_points_per_segment,
            "func_evaluations": len(self.func_cache),
            "history": history
        }
        return integral_new, stats
