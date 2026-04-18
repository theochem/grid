import numpy as np

def sharp_gaussian(x):
    return np.exp(-1000 * x**2)

# Nodes and Weights for 15-point Gauss-Kronrod (on [-1, 1])
# Gauss nodes are a subset (indices 1, 3, 5, 7, 9, 11, 13 in the 15-point list)
nodes_kronrod = np.array([
    0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
    0.7415311855993944, 0.5860872354676911, 0.4058451513773972,
    0.2077849550078985, 0.0
])
# Full 15 nodes (symmetric)
all_nodes = np.concatenate([-nodes_kronrod[:-1], [0.0], nodes_kronrod[:-1][::-1]])
# Wait, let's keep it simple and just use the symmetric property.

weights_kronrod = np.array([
    0.0229353007137257, 0.0630920926299786, 0.1047900103222502,
    0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
    0.2044329400752989, 0.2094821410847278
])

# Gauss nodes (n=7) nodes and weights
nodes_gauss = np.array([
    0.9491079123427585, 0.7415311855993944, 0.4058451513773972, 0.0
])
weights_gauss = np.array([
    0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694
])

def evaluate_quadrature():
    # Evaluate at Kronrod nodes (subset includes Gauss nodes)
    y_kronrod_pos = sharp_gaussian(nodes_kronrod)
    y_kronrod_neg = sharp_gaussian(-nodes_kronrod[:-1])
    
    # Kronrod estimate
    # Sum w_i * f(x_i) for positive, negative, and zero
    res_kronrod = np.sum(weights_kronrod[:-1] * y_kronrod_pos[:-1]) * 2 + weights_kronrod[-1] * y_kronrod_pos[-1]
    
    # Gauss estimate
    # Gauss nodes are index 1, 3, 5, 7 in the nodes_kronrod array
    y_gauss_pos = y_kronrod_pos[[1, 3, 5, 7]] # [0.949, 0.741, 0.405, 0.0]
    res_gauss = np.sum(weights_gauss[:-1] * y_gauss_pos[:-1]) * 2 + weights_gauss[-1] * y_gauss_pos[-1]
    
    # Reference value using scipy.integrate.quad for high precision
    from scipy.integrate import quad
    ref_val, _ = quad(sharp_gaussian, -1, 1, epsabs=1e-15)
    
    error_est = abs(res_kronrod - res_gauss)
    actual_error = abs(res_kronrod - ref_val)
    
    print(f"Results for exp(-1000x^2) on [-1, 1]:")
    print(f"Gauss (n=7) estimate  : {res_gauss:.15f}")
    print(f"Kronrod (n=15) estimate: {res_kronrod:.15f}")
    print(f"Reference value        : {ref_val:.15f}")
    print(f"-" * 40)
    print(f"Error Estimate (K-G)   : {error_est:.2e}")
    print(f"Actual Error (K-Ref)   : {actual_error:.2e}")

if __name__ == "__main__":
    evaluate_quadrature()
