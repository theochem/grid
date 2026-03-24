import numpy as np
import pytest
from grid.max_det import MaxDeterminantGrid
from grid.utils import generate_real_spherical_harmonics, convert_cart_to_sph

def test_maxdet_constant():
    # Integral of constant 1 over S^2 should be 4*pi
    grid = MaxDeterminantGrid(degree=1)
    func = lambda p: np.ones(len(p))
    integral = grid.integrate(func)
    assert np.allclose(integral, 4 * np.pi)

def test_maxdet_weights_positive():
    # MaxDet weights should be all positive
    grid = MaxDeterminantGrid(degree=10)
    assert np.all(grid.weights > 0)

def test_maxdet_sum_weights():
    # Sum of weights should be 4*pi
    for d in [1, 2, 5, 10]:
        grid = MaxDeterminantGrid(degree=d)
        assert np.allclose(np.sum(grid.weights), 4 * np.pi)

@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_maxdet_integration_spherical_harmonic(degree):
    # MaxDet for degree t should integrate spherical harmonics up to degree t exactly.
    grid = MaxDeterminantGrid(degree=degree)
    
    r = np.linalg.norm(grid.points, axis=1)
    phi = np.arccos(grid.points[:, 2] / r)
    theta = np.arctan2(grid.points[:, 1], grid.points[:, 0])
    
    sph_harm = generate_real_spherical_harmonics(degree, theta, phi)
    
    for l_deg in range(degree + 1):
        # Horton 2 order indexing: l^2 to (l+1)^2
        sph_harm_l = sph_harm[l_deg**2 : (l_deg + 1)**2, :]
        for m_idx in range(2 * l_deg + 1):
            integral = np.sum(sph_harm_l[m_idx, :] * grid.weights)
            if l_deg == 0:
                # Integral of Y_00 = 1/sqrt(4pi) is sqrt(4pi)
                assert np.allclose(integral, np.sqrt(4 * np.pi))
            else:
                # Integral of Y_lm (l > 0) is 0
                assert np.abs(integral) < 1e-10

def test_maxdet_orthonormality():
    # For a grid of degree T, it can integrate products of harmonics Y_l1 * Y_l2
    # if l1 + l2 <= T.
    T = 10
    grid = MaxDeterminantGrid(degree=T)
    
    r = np.linalg.norm(grid.points, axis=1)
    phi = np.arccos(grid.points[:, 2] / r)
    theta = np.arctan2(grid.points[:, 1], grid.points[:, 0])
    
    l_max = T // 2
    sph_harm = generate_real_spherical_harmonics(l_max, theta, phi)
    num_harmonics = (l_max + 1)**2
    
    for i in range(num_harmonics):
        for j in range(num_harmonics):
            integral = np.sum(sph_harm[i, :] * sph_harm[j, :] * grid.weights)
            expected = 1.0 if i == j else 0.0
            assert np.allclose(integral, expected, atol=1e-10)

def test_integrate_method_interface():
    # Test that integrate(function) and Grid.integrate(values) are consistent
    grid = MaxDeterminantGrid(degree=2)
    # R^2 = 1 on unit sphere
    func = lambda p: p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 
    
    val1 = grid.integrate(func)
    val2 = grid.integrate(np.ones(grid.size))
    
    assert np.allclose(val1, val2)
    assert np.allclose(val1, 4 * np.pi)
