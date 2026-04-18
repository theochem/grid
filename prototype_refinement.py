import numpy as np
from scipy import special
from grid.onedgrid import Trapezoidal, GaussLegendre

# Reuse the adaptive Simpson logic from our student implementation
def adaptive_simpson(f, a, b, tol, fa=None, fm=None, fb=None, depth=0, max_depth=10, stats=None):
    if stats is None:
        stats = {'evals': 0}
    
    m = (a + b) / 2
    if fa is None: fa = f(a); stats['evals'] += 1
    if fm is None: fm = f(m); stats['evals'] += 1
    if fb is None: fb = f(b); stats['evals'] += 1
    
    m_left = (a + m) / 2
    m_right = (m + b) / 2
    f_ml = f(m_left); stats['evals'] += 1
    f_mr = f(m_right); stats['evals'] += 1
    
    h = (b - a) / 2
    s_coarse = (h / 3) * (fa + 4 * fm + fb)
    
    h_small = h / 2
    s_left = (h_small / 3) * (fa + 4 * f_ml + fm)
    s_right = (h_small / 3) * (fm + 4 * f_mr + fb)
    s_refined = s_left + s_right
    
    error = abs(s_refined - s_coarse) / 15
    
    if error < tol or depth >= max_depth:
        return s_refined
    else:
        return (adaptive_simpson(f, a, m, tol/2, fa, f_ml, fm, depth+1, max_depth, stats) +
                adaptive_simpson(f, m, b, tol/2, fm, f_mr, fb, depth+1, max_depth, stats))

def refine_grid(grid, f, tol):
    """
    Take an existing grid and refine its intervals.
    Treats segments [x_i, x_{i+1}] as independent intervals.
    """
    total_integral = 0
    stats = {'evals': len(grid.points)} # count initial evaluations
    # Approximate function at grid points
    vals = f(grid.points)
    
    # Sort points to define intervals
    pts = grid.points
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i+1]
        fa, fb = vals[i], vals[i+1]
        # Recursively refine each slice
        total_integral += adaptive_simpson(f, a, b, tol / (len(pts)-1), fa=fa, fb=fb, stats=stats)
    
    return total_integral, stats

# Test Case: sharp Gaussian e^(-1000x^2)
def gaussian(x):
    return np.exp(-1000 * x**2)

exact = np.sqrt(np.pi/1000) * special.erf(np.sqrt(1000))

# 1. Starting with Uniform (Trapezoidal 10 points)
grid_unif = Trapezoidal(10)
int_unif, stats_unif = refine_grid(grid_unif, gaussian, 1e-7)

# 2. Starting with Gauss-Legendre (10 points)
grid_gl = GaussLegendre(10)
int_gl, stats_gl = refine_grid(grid_gl, gaussian, 1e-7)

print("--- Comparison of Initial Grid Choice ---")
print(f"Uniform Grid Initial (10 pts) -> Evaluations: {stats_unif['evals']}, Error: {abs(int_unif - exact):.2e}")
print(f"Gauss-Legendre Initial (10 pts) -> Evaluations: {stats_gl['evals']}, Error: {abs(int_gl - exact):.2e}")
