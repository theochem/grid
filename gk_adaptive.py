import numpy as np
from scipy.integrate import quad

# 15-Point Gauss-Kronrod Nodes and Weights on [-1, 1]
# Sourced from standard tables (e.g., QUADPACK dqk15)
nodes_kronrod_base = np.array([
    0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
    0.7415311855993944, 0.5860872354676911, 0.4058451513773972,
    0.2077849550078985, 0.0
])

weights_kronrod_base = np.array([
    0.0229353007137257, 0.0630920926299786, 0.1047900103222502,
    0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
    0.2044329400752989, 0.2094821410847278
])

# 7-Point Gauss Weights for the corresponding Gauss nodes in the 15-point set
# These are the nodes at indices 1, 3, 5 (and center at index 7)
weights_gauss_base = np.array([
    0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694
])

import heapq

def adaptive_gk_queued(f, a, b, tol, max_evals=10000, stats=None):
    if stats is None:
        stats = {'evals': 0, 'max_depth': 0}
    
    def get_gk_estimate(a, b):
        h = (b - a) / 2
        c = (a + b) / 2
        x_pos = c + h * nodes_kronrod_base
        x_neg = c - h * nodes_kronrod_base[:-1]
        y_pos = f(x_pos)
        y_neg = f(x_neg)
        stats['evals'] += 15
        
        # Kronrod result
        res_k = np.sum(weights_kronrod_base[:-1] * (y_pos[:-1] + y_neg)) * h + weights_kronrod_base[-1] * y_pos[-1] * h
        # Gauss result
        y_gauss_pos = y_pos[[1, 3, 5]]
        y_gauss_neg = y_neg[[1, 3, 5]]
        y_gauss_mid = y_pos[-1]
        res_g = np.sum(weights_gauss_base[:-1] * (y_gauss_pos + y_gauss_neg)) * h + weights_gauss_base[-1] * y_gauss_mid * h
        
        error = abs(res_k - res_g)
        return res_k, error

    # Initial interval
    res_k, error = get_gk_estimate(a, b)
    # Heap: (-error, a, b, result, depth) -> We use negative error for max-heap
    queue = [(-error, a, b, res_k, 0)]
    total_result = res_k
    total_error = error
    
    while total_error > tol and stats['evals'] < max_evals:
        # Pop interval with largest error
        err, a_i, b_i, res_i, depth = heapq.heappop(queue)
        err = -err
        
        # Split interval
        mid = (a_i + b_i) / 2
        
        # Evaluate children
        res_l, err_l = get_gk_estimate(a_i, mid)
        res_r, err_r = get_gk_estimate(mid, b_i)
        
        # Update totals
        total_result = total_result - res_i + (res_l + res_r)
        total_error = total_error - err + (err_l + err_r)
        
        # Push children
        heapq.heappush(queue, (-err_l, a_i, mid, res_l, depth + 1))
        heapq.heappush(queue, (-err_r, mid, b_i, res_r, depth + 1))
        
        stats['max_depth'] = max(stats['max_depth'], depth + 1)
        
    return total_result

def adaptive_simpson(f, a, b, tol, fa=None, fm=None, fb=None, depth=0, max_depth=15, stats=None):
    if stats is None:
        stats = {'evals': 0, 'max_depth': 0}
    
    m = (a + b) / 2
    if fa is None: fa = f(a); stats['evals'] += 1
    if fm is None: fm = f(m); stats['evals'] += 1
    if fb is None: fb = f(b); stats['evals'] += 1
        
    stats['max_depth'] = max(stats['max_depth'], depth)
    
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

def benchmark():
    tests = [
        ("Sharp Gaussian (e^-1000x^2)", lambda x: np.exp(-1000 * x**2), -1, 1, 1e-10),
        ("Oscillatory (sin(100x))", lambda x: np.sin(100 * x), -1, 1, 1e-10),
        ("Smooth (x^4)", lambda x: x**4, -1, 1, 1e-10)
    ]
    
    for name, f, a, b, tol in tests:
        print(f"\n--- Testing: {name} (tol={tol}) ---")
        
        # Scipy Quad
        res_quad, abserr, info = quad(f, a, b, full_output=True)[:3]
        evals_quad = info['neval']
        
        # Adaptive Simpson
        stats_simpson = {'evals': 0, 'max_depth': 0}
        res_simpson = adaptive_simpson(f, a, b, tol, stats=stats_simpson)
        
        # Adaptive GK (7/15) - Queue based
        stats_gk = {'evals': 0, 'max_depth': 0}
        res_gk = adaptive_gk_queued(f, a, b, tol, stats=stats_gk)
        
        print(f"{'Method':<20} | {'Result':<18} | {'Evals':<8} | {'Error vs Scipy':<15}")
        print("-" * 75)
        print(f"{'Scipy Quad':<20} | {res_quad:<18.15f} | {evals_quad:<8} | 0.00e+00")
        print(f"{'Adaptive Simpson':<20} | {res_simpson:<18.15f} | {stats_simpson['evals']:<8} | {abs(res_simpson-res_quad):.2e}")
        print(f"{'Adaptive GK (v2)':<20} | {res_gk:<18.15f} | {stats_gk['evals']:<8} | {abs(res_gk-res_quad):.2e}")

if __name__ == "__main__":
    benchmark()
