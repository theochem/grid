"""Poisson test module."""
from unittest import TestCase

from grid.atomic_grid import AtomicGrid
from grid.interpolate import interpolate, spline_with_atomic_grid
from grid.onedgrid import GaussChebyshev
from grid.poisson import Poisson
from grid.rtransform import BeckeTF, InverseTF

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal


class TestPoisson(TestCase):
    """Test poisson class."""

    def helper_func_gauss(self, coors, center=[0, 0, 0], alphas=[1, 1, 1]):
        """Compute gauss function at given center."""
        x, y, z = (coors - center).T
        a, b, c = alphas
        return a * np.exp(-(x ** 2)) * b * np.exp(-(y ** 2)) * c * np.exp(-(z ** 2))

    def test_poisson_proj(self):
        """Test the project function."""
        oned = GaussChebyshev(30)
        btf = BeckeTF(0.0001, 1.5)
        rad = btf.transform_1d_grid(oned)
        l_max = 7
        atgrid = AtomicGrid(rad, degs=[l_max])
        value_array = self.helper_func_gauss(atgrid.points)
        spl_res = spline_with_atomic_grid(atgrid, value_array)
        # test for random, r, theta, phi
        for _ in range(20):
            r = np.random.rand(1)[0] * 2
            theta = np.random.rand(10) * 3.14
            phi = np.random.rand(10) * 3.14
            result = interpolate(spl_res, r, theta, phi)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            result_ref = self.helper_func_gauss(np.array([x, y, z]).T)
            # assert similar value less than 1e-4 discrepancy
            assert_allclose(result, result_ref, atol=1e-4)

        sph_coor = atgrid.convert_cart_to_sph()[:, 1:3]
        spls_mt = Poisson._proj_sph_value(
            atgrid.rad_grid,
            sph_coor,
            l_max // 2,
            value_array,
            atgrid.weights,
            atgrid.indices,
        )
        # test each point is the same
        for _ in range(20):
            r = np.random.rand(1)[0] * 2
            # random spherical point
            int_res = np.zeros((l_max, l_max // 2 + 1), dtype=float)
            for j in range(l_max // 2 + 1):
                for i in range(-j, j + 1):
                    int_res[i, j] += spls_mt[i, j](r)
            assert_allclose(spl_res(r), int_res)

    def test_poisson_solve(self):
        """Test the poisson solve function."""
        oned = GaussChebyshev(30)
        oned = GaussChebyshev(50)
        btf = BeckeTF(1e-7, 1.5)
        rad = btf.transform_1d_grid(oned)
        l_max = 7
        atgrid = AtomicGrid(rad, degs=[l_max])
        value_array = self.helper_func_gauss(atgrid.points)
        p_0 = atgrid.integrate(value_array)

        # test density sum up to np.pi**(3 / 2)
        assert_allclose(p_0, np.pi ** 1.5, atol=1e-4)
        sph_coor = atgrid.convert_cart_to_sph()[:, 1:3]
        spls_mt = Poisson._proj_sph_value(
            atgrid.rad_grid,
            sph_coor,
            l_max // 2,
            value_array,
            atgrid.weights,
            atgrid.indices,
        )
        # test splines project fit gauss function well

        def gauss(r):
            return np.exp(-(r ** 2))

        for _ in range(20):
            coors = np.random.rand(10, 3)
            r = np.linalg.norm(coors, axis=-1)
            spl_0_0 = spls_mt[0, 0]
            interp_v = spl_0_0(r)
            ref_v = gauss(r) * np.sqrt(4 * np.pi)
            # 0.28209479 is the value in spherical harmonic Z_0_0
            assert_allclose(interp_v, ref_v, atol=1e-3)
        ibtf = InverseTF(btf)
        linsp = np.linspace(-1, 0.99, 50)
        bound = p_0 * np.sqrt(4 * np.pi)
        res_bv = Poisson.solve_poisson_bv(spls_mt[0, 0], linsp, bound, tfm=ibtf)

        near_rg_pts = np.array([1e-2, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2])
        near_tf_pts = ibtf.transform(near_rg_pts)
        long_rg_pts = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
        long_tf_pts = ibtf.transform(long_rg_pts)
        short_res = res_bv(near_tf_pts)[0] / near_rg_pts / (2 * np.sqrt(np.pi))
        long_res = res_bv(long_tf_pts)[0] / long_rg_pts / (2 * np.sqrt(np.pi))
        # ref are calculated with mathemetical
        # integrate[exp[-x^2 - y^2 - z^2] / sqrt[(x - a)^2 + y^2 +z^2], range]
        ref_short_res = [
            6.28286,  # 0.01
            6.26219,  # 0.1
            6.20029,  # 0.2
            6.09956,  # 0.3
            5.79652,  # 0.5
            5.3916,  # 0.7
            4.69236,  # 1.0
            4.22403,  # 1.2
        ]
        ref_long_res = [
            2.77108,  # 2
            1.85601,  # 3
            1.39203,  # 4
            1.11362,  # 5
            0.92802,  # 6
            0.79544,  # 7
            0.69601,  # 8
            0.61867,  # 9
            0.55680,  # 10
        ]
        assert_allclose(short_res, ref_short_res, atol=5e-4)
        assert_allclose(long_res, ref_long_res, atol=5e-4)
        # solve same poisson equation with gauss directly
        gauss_pts = btf.transform(linsp)
        res_gs = Poisson.solve_poisson_bv(gauss, gauss_pts, p_0)
        gs_int_short = res_gs(near_rg_pts)[0] / near_rg_pts
        gs_int_long = res_gs(long_rg_pts)[0] / long_rg_pts
        assert_allclose(gs_int_short, ref_short_res, 5e-4)
        assert_allclose(gs_int_long, ref_long_res, 5e-4)

    def test_poisson_solve_mtr_cmpl(self):
        """Test solve poisson equation and interpolate the result."""
        oned = GaussChebyshev(50)
        btf = BeckeTF(1e-7, 1.5)
        rad = btf.transform_1d_grid(oned)
        l_max = 7
        atgrid = AtomicGrid(rad, degs=[l_max])
        value_array = self.helper_func_gauss(atgrid.points)
        p_0 = atgrid.integrate(value_array)

        # test density sum up to np.pi**(3 / 2)
        assert_allclose(p_0, np.pi ** 1.5, atol=1e-4)
        sph_coor = atgrid.convert_cart_to_sph()[:, 1:3]
        spls_mt = Poisson._proj_sph_value(
            atgrid.rad_grid,
            sph_coor,
            l_max // 2,
            value_array,
            atgrid.weights,
            atgrid.indices,
        )
        ibtf = InverseTF(btf)
        linsp = np.linspace(-1, 0.99, 50)
        bound = p_0 * np.sqrt(4 * np.pi)
        pois_mtr = Poisson.solve_poisson(spls_mt, linsp, bound, tfm=ibtf)
        assert pois_mtr.shape == (7, 4)
        near_rg_pts = np.array([1e-2, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2])
        near_tf_pts = ibtf.transform(near_rg_pts)
        ref_short_res = [
            6.28286,  # 0.01
            6.26219,  # 0.1
            6.20029,  # 0.2
            6.09956,  # 0.3
            5.79652,  # 0.5
            5.3916,  # 0.7
            4.69236,  # 1.0
            4.22403,  # 1.2
        ]
        for i, j in enumerate(near_tf_pts):
            assert_almost_equal(
                Poisson.interpolate_radial(pois_mtr, j, 0, True) / near_rg_pts[i],
                ref_short_res[i] * np.sqrt(4 * np.pi),
                decimal=3,
            )
            matrix_result = Poisson.interpolate_radial(pois_mtr, j)
            assert_almost_equal(
                matrix_result[0, 0] / near_rg_pts[i],
                ref_short_res[i] * np.sqrt(4 * np.pi),
                decimal=3,
            )
            # test interpolate with sph
            result = Poisson.interpolate(
                pois_mtr, j, np.random.rand(5), np.random.rand(5)
            )
            assert_allclose(
                result / near_rg_pts[i] - ref_short_res[i], np.zeros(5), atol=1e-3
            )

    def test_raises_errors(self):
        """Test proper error raises."""
        oned = GaussChebyshev(50)
        btf = BeckeTF(1e-7, 1.5)
        rad = btf.transform_1d_grid(oned)
        l_max = 7
        atgrid = AtomicGrid(rad, degs=[l_max])
        value_array = self.helper_func_gauss(atgrid.points)
        p_0 = atgrid.integrate(value_array)

        # test density sum up to np.pi**(3 / 2)
        assert_allclose(p_0, np.pi ** 1.5, atol=1e-4)
        sph_coor = atgrid.convert_cart_to_sph()
        with self.assertRaises(ValueError):
            Poisson._proj_sph_value(
                atgrid.rad_grid,
                sph_coor,
                l_max // 2,
                value_array,
                atgrid.weights,
                atgrid.indices,
            )
