import numpy as np
from scipy import special, integrate
from grid.onedgrid import GaussLegendre

# Adaptive Simpson implementation
def adaptive_simpson(f, a, b, tol, fa=None, fm=None, fb=None, depth=0, max_depth=20, stats=None):
    if stats is None: stats = {'evals': 0}
    m = (a + b) / 2
    if fa is None: fa = f(a); stats['evals'] += 1
    if fm is None: fm = f(m); stats['evals'] += 1
    if fb is None: fb = f(b); stats['evals'] += 1
    
    m_left, m_right = (a + m) / 2, (m + b) / 2
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
    return (adaptive_simpson(f, a, m, tol/2, fa, f_ml, fm, depth+1, max_depth, stats) +
            adaptive_simpson(f, m, b, tol/2, fm, f_mr, fb, depth+1, max_depth, stats))

def run_benchmark(name, f, a, b, tol=1e-7):
    # 1. Scipy Quad
    res_quad, err_quad, info_quad = integrate.quad(f, a, b, epsabs=tol, epsrel=0, full_output=True)
    evals_quad = info_quad['neval']
    
    # 2. Adaptive Simpson
    stats_simpson = {'evals': 0}
    res_simpson = adaptive_simpson(f, a, b, tol, stats=stats_simpson)
    
    print(f"--- Benchmark: {name} ---")
    print(f"Scipy Quad: Res={res_quad:.8f}, Evals={evals_quad:4d}")
    print(f"Adaptive S: Res={res_simpson:.8f}, Evals={stats_simpson['evals']:4d}")
    print(f"Rel Diff  : {abs(res_simpson - res_quad) / max(abs(res_quad), 1e-10):.2e}")

# 1. Sharp Peaks
run_benchmark("Sharp peak (exp(-1000x^2))", lambda x: np.exp(-1000*x**2), -1, 1)

# 2. Oscillatory
run_benchmark("Oscillatory (sin(100x))", lambda x: np.sin(100*x), 0, 1)

# 3. Smooth
run_benchmark("Smooth (polynomial x^4)", lambda x: x**4, 0, 1)
