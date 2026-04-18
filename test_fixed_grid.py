import numpy as np
from grid.onedgrid import GaussLegendre, Trapezoidal, MidPoint
from scipy.integrate import quad

def integrand(x, a=1000):
    return np.exp(-a * x**2)

# Analytical/Exact integral via scipy
exact, _ = quad(integrand, -1, 1)

print(f"Exact integral (scipy.integrate.quad): {exact:.12f}\n")

print(f"{'Method':<20} | {'Points':<8} | {'Result':<16} | {'Relative Error':<16}")
print("-" * 70)

for n in [10, 50, 100]:
    # Gauss-Legendre
    gl = GaussLegendre(n)
    res_gl = gl.integrate(integrand(gl.points))
    err_gl = abs(res_gl - exact) / exact if exact != 0 else abs(res_gl - exact)
    print(f"{'Gauss-Legendre':<20} | {n:<8} | {res_gl:<16.12f} | {err_gl:<16.2e}")

    # Trapezoidal
    trap = Trapezoidal(n)
    res_trap = trap.integrate(integrand(trap.points))
    err_trap = abs(res_trap - exact) / exact if exact != 0 else abs(res_trap - exact)
    print(f"{'Trapezoidal':<20} | {n:<8} | {res_trap:<16.12f} | {err_trap:<16.2e}")

    # Midpoint
    mid = MidPoint(n)
    res_mid = mid.integrate(integrand(mid.points))
    err_mid = abs(res_mid - exact) / exact if exact != 0 else abs(res_mid - exact)
    print(f"{'MidPoint':<20} | {n:<8} | {res_mid:<16.12f} | {err_mid:<16.2e}")
    print("-" * 70)
