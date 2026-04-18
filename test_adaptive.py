import numpy as np
from scipy import special

def adaptive_simpson(f, a, b, tol, fa=None, fm=None, fb=None, depth=0, max_depth=20, stats=None):
    if stats is None:
        stats = {'evals': 0, 'max_depth': 0}
    
    m = (a + b) / 2
    # Initial evaluations if not provided
    if fa is None:
        fa = f(a); stats['evals'] += 1
    if fm is None:
        fm = f(m); stats['evals'] += 1
    if fb is None:
        fb = f(b); stats['evals'] += 1
        
    stats['max_depth'] = max(stats['max_depth'], depth)
    
    # New midpoints for refinement
    m_left = (a + m) / 2
    m_right = (m + b) / 2
    f_ml = f(m_left); stats['evals'] += 1
    f_mr = f(m_right); stats['evals'] += 1
    
    h = (b - a) / 2
    # Coarse estimate (Simpson on [a, b])
    s_coarse = (h / 3) * (fa + 4 * fm + fb)
    
    # Refined estimate (Sum of two Simpson on each half)
    # Note: h for the halves is (b-a)/4
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

# Test with sharp Gaussian
def gaussian(x):
    return np.exp(-1000 * x**2)

# Exact integral of exp(-a*x^2) from -1 to 1
a_const = 1000
exact = np.sqrt(np.pi/a_const) * special.erf(np.sqrt(a_const)) 

tol = 1e-7
stats = {'evals': 0, 'max_depth': 0}
result = adaptive_simpson(gaussian, -1, 1, tol, stats=stats)

print(f"Result: {result:.12f}")
print(f"Exact:  {exact:.12f}")
print(f"Error:  {abs(result - exact):.2e}")
print(f"Evaluations: {stats['evals']}")
print(f"Max Depth:   {stats['max_depth']}")
