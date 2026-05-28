import numpy as np
import pytest
from numpy.testing import assert_allclose
import warnings
from unittest.mock import patch

from grid.atomgrid import AtomGrid
from grid.onedgrid import Trapezoidal
from grid.rtransform import LinearFiniteRTransform, InverseRTransform
from grid.poisson import solve_poisson_bvp, solve_poisson_ivp, _solve_poisson_ivp_atomgrid

def gauss_density(pts, alpha=1.0):
    r = np.linalg.norm(pts, axis=1)
    return np.exp(-alpha * r**2)

def exp_density(pts, alpha=1.0):
    r = np.linalg.norm(pts, axis=1)
    return np.exp(-alpha * r)

def setup_grid():
    oned = Trapezoidal(250)
    btf = LinearFiniteRTransform(1e-3, 20.0)
    radial = btf.transform_1d_grid(oned)
    atgrid = AtomGrid(radial, center=np.array([0.0, 0.0, 0.0]), degrees=[11])
    return atgrid, btf

def test_ivp_gaussian_vs_bvp():
    atgrid, btf = setup_grid()
    density = gauss_density(atgrid.points, alpha=1.0)
    
    pot_ivp = solve_poisson_ivp(
        atgrid,
        density,
        InverseRTransform(btf),
        r_interval=(20.0, 1e-3)
    )
    
    pot_bvp = solve_poisson_bvp(
        atgrid,
        density,
        InverseRTransform(btf),
        include_origin=True
    )
    
    pts = atgrid.points
    val_ivp = pot_ivp(pts)
    val_bvp = pot_bvp(pts)
    
    # Check that they match
    assert_allclose(val_ivp, val_bvp, rtol=1e-2, atol=1e-2)

def test_ivp_fallback_warning():
    atgrid, btf = setup_grid()
    density = exp_density(atgrid.points, alpha=100.0) 
    
    import grid.ode
    original_solve_ivp = grid.ode.solve_ivp
    
    call_count = 0
    def mock_solve_ivp(fun, t_span, y0, method, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            class DummyRes:
                status = -1
                message = "Failed"
            return DummyRes()
        return original_solve_ivp(fun, t_span, y0, method=method, **kwargs)
        
    with patch("grid.ode.solve_ivp", side_effect=mock_solve_ivp):
        with pytest.warns(UserWarning, match="ODE solver method.*failed.*Trying next method"):
            pot_ivp = solve_poisson_ivp(
                atgrid,
                density,
                InverseRTransform(btf),
                r_interval=(20.0, 1e-3)
            )
    
    val = pot_ivp(atgrid.points)
    assert not np.any(np.isnan(val))

def test_adaptive_r_interval():
    atgrid, btf = setup_grid()
    
    density_compact = exp_density(atgrid.points, alpha=5.0)
    density_diffuse = exp_density(atgrid.points, alpha=0.1)
    
    used_r_intervals = []
    
    import grid.ode
    original_solve_ode_ivp = grid.ode.solve_ode_ivp
    def mock_solve_ode_ivp(r_interval, *args, **kwargs):
        used_r_intervals.append(r_interval)
        return original_solve_ode_ivp(r_interval, *args, **kwargs)

    with patch("grid.poisson.solve_ode_ivp", side_effect=mock_solve_ode_ivp):
        solve_poisson_ivp(atgrid, density_compact, InverseRTransform(btf), r_interval=(20.0, 1e-3))
        solve_poisson_ivp(atgrid, density_diffuse, InverseRTransform(btf), r_interval=(20.0, 1e-3))

    assert len(used_r_intervals) >= 2
    r_max_compact = used_r_intervals[0][0]
    r_max_diffuse = used_r_intervals[-1][0]
    assert r_max_compact < r_max_diffuse
