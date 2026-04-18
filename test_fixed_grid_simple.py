import sys
import numpy as np
from grid.onedgrid import GaussLegendre, Trapezoidal, MidPoint
from scipy.integrate import quad

def integrand(x, a=1000):
    return np.exp(-a * x**2)

print("Starting test...")
sys.stdout.flush()

try:
    exact, _ = quad(integrand, -1, 1)
    print(f"Exact integral: {exact:.12f}")
    sys.stdout.flush()

    for n in [10, 50, 100]:
        gl = GaussLegendre(n)
        res = gl.integrate(integrand(gl.points))
        print(f"GL-{n}: {res:.12f}, Err: {abs(res - exact):.2e}")
        sys.stdout.flush()
except Exception as e:
    print(f"Error: {e}")
    sys.stderr.write(str(e) + "\n")
    sys.stdout.flush()
