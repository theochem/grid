# Design: Adaptive Quadrature for `theochem/grid`

## Architecture
- **Class**: `AdaptiveOneDGrid` (inherits from `OneDGrid`).
- **Goal**: Augment an existing 1D grid with additional points in high-error regions.
- **Method**: `.refine(integrand, tolerance, max_depth)` -> Returns a new `AdaptiveOneDGrid`.

## Trade-offs
### 1. Refining Uniform Grids (e.g., Trapezoidal/Midpoint)
- **Pros**: Perfectly nested. Simpson's rule fits naturally.
- **Cons**: Requires more initial points to achieve a baseline accuracy compared to Gauss-Legendre.

### 2. Refining Gauss-Legendre Grids
- **Pros**: High initial accuracy for smooth parts of the function.
- **Cons**: Non-nested. Splitting intervals between Legendre nodes creates a "hybrid" grid. 
- **Decision**: The prototype will treat each interval $[x_i, x_{i+1}]$ as an independent domain for adaptive integration.

## Implementation Details
- Internal storage will remain as flat `points` and `weights` arrays to ensure `Grid.integrate()` works out-of-the-box.
- Recursive algorithm will follow the "point reuse" pattern developed in `test_adaptive.py`.
