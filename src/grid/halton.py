"""Halton low-discrepancy sequence grids."""

import numpy as np

from grid.basegrid import Grid


def _first_primes(n):
    """Return the first n prime numbers."""
    primes = []
    candidate = 2

    while len(primes) < n:
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1

    return primes


def _radical_inverse(indices, base):
    """Calculate radical inverses for a given base."""
    values = np.zeros(len(indices), dtype=float)
    factor = 1.0 / base
    indices = np.array(indices, dtype=int, copy=True)

    while np.any(indices):
        values += (indices % base) * factor
        indices //= base
        factor /= base

    return values


class Halton(Grid):
    """Generate a multidimensional Halton low-discrepancy sequence."""

    name = "Halton"

    def __init__(self, npoints: int, ndim: int):
        """Generate ``npoints`` points in ``ndim`` dimensions.

        Parameters
        ----------
        npoints : int
            Number of points.
        ndim : int
            Number of dimensions.
        """
        if not isinstance(npoints, (int, np.integer)) or npoints < 1:
            raise ValueError(f"Argument npoints must be a positive integer, given {npoints}")

        if not isinstance(ndim, (int, np.integer)) or ndim < 1:
            raise ValueError(f"Argument ndim must be a positive integer, given {ndim}")

        indices = np.arange(1, npoints + 1)
        primes = _first_primes(ndim)

        points = np.column_stack(
            [_radical_inverse(indices, prime) for prime in primes]
        )
        weights = np.full(npoints, 1.0 / npoints)

        super().__init__(points, weights)
