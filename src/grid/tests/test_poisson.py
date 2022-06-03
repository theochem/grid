# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2019 The GRID Development Team
#
# This file is part of GRID.
#
# GRID is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GRID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Poisson test module."""
from unittest import TestCase

from grid.atomgrid import AtomGrid
from grid.onedgrid import OneDGrid, GaussChebyshev
from grid.poisson import interpolate_laplacian, Poisson
from grid.rtransform import BeckeRTransform, InverseRTransform

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal


class TestPoisson(TestCase):
    """Test poisson class."""

    def helper_func_gauss(self, coors, center=[0, 0, 0], alphas=[1, 1, 1]):
        """Compute gauss function at given center."""
        x, y, z = (coors - center).T
        a, b, c = alphas
        return a * np.exp(-(x ** 2)) * b * np.exp(-(y ** 2)) * c * np.exp(-(z ** 2))

    def test_poisson_solve(self):
        """Test the poisson solve function."""
        oned = GaussChebyshev(50)
        btf = BeckeRTransform(1e-7, 1.5)
        rad = btf.transform_1d_grid(oned)
        l_max = 7
        atgrid = AtomGrid(rad, degrees=[l_max])
        value_array = self.helper_func_gauss(atgrid.points)
        p_0 = atgrid.integrate(value_array)

        # test density sum up to np.pi**(3 / 2)
        assert_allclose(p_0, np.pi ** 1.5, atol=1e-4)
        sph_coor = atgrid.convert_cartesian_to_spherical()[:, 1:3]
        spls_mt = atgrid.radial_component_splines(value_array)
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
        ibtf = InverseRTransform(btf)
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
        btf = BeckeRTransform(1e-7, 1.5)
        rad = btf.transform_1d_grid(oned)
        l_max = 7
        atgrid = AtomGrid(rad, degrees=[l_max])
        value_array = self.helper_func_gauss(atgrid.points)
        p_0 = atgrid.integrate(value_array)

        # test density sum up to np.pi**(3 / 2)
        assert_allclose(p_0, np.pi ** 1.5, atol=1e-4)
        sph_coor = atgrid.convert_cartesian_to_spherical()[:, 1:3]
        spls_mt = atgrid.radial_component_splines(value_array)
        ibtf = InverseRTransform(btf)
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
        btf = BeckeRTransform(1e-7, 1.5)
        rad = btf.transform_1d_grid(oned)
        l_max = 7
        atgrid = AtomGrid(rad, degrees=[l_max])
        value_array = self.helper_func_gauss(atgrid.points)
        p_0 = atgrid.integrate(value_array)

        # test density sum up to np.pi**(3 / 2)
        assert_allclose(p_0, np.pi ** 1.5, atol=1e-4)
        sph_coor = atgrid.convert_cartesian_to_spherical()


def test_interpolation_of_laplacian_with_spherical_harmonic(self):
    r"""Test the interpolation of Laplacian of spherical harmonic is eigenvector."""
    odg = OneDGrid(np.linspace(0.0, 10, num=2000), np.ones(2000), (0, np.inf))
    degree = 6 * 2 + 2
    atgrid = AtomGrid.from_pruned(odg, 1, sectors_r=[], sectors_degree=[degree])

    def func(sph_points):
        # Spherical harmonic of order 6 and magnetic 0
        r, phi, theta = sph_points.T
        return np.sqrt(2.0) * np.sqrt(13) / (np.sqrt(np.pi) * 32) * (
                231 * np.cos(theta) ** 6.0 - 315 * np.cos(theta) ** 4.0 + 105 * np.cos(
            theta) ** 2.0 - 5.0
        )
    # Get spherical points from atomic grid
    spherical_pts = atgrid.convert_cartesian_to_spherical(atgrid.points)
    func_values = func(spherical_pts)

    laplacian = interpolate_laplacian(atgrid, func_values)

    # Test on the same points used for interpolation and random points.
    for grid in [atgrid.points, np.random.uniform(-0.75, 0.75, (250, 3))]:
        actual = laplacian(grid)
        spherical_pts = atgrid.convert_cartesian_to_spherical(grid)
        # Eigenvector spherical harmonic times l(l + 1) / r^2
        with np.errstate(divide='ignore', invalid='ignore'):
            desired = -func(spherical_pts) * 6 * (6 + 1) / spherical_pts[:, 0]**2.0
        desired[spherical_pts[:, 0]**2.0 < 1e-10] = 0.0
        assert_almost_equal(actual, desired, decimal=3)


def test_interpolation_of_laplacian_of_exponential(self):
    r"""Test the interpolation of Laplacian of exponential."""
    odg = OneDGrid(np.linspace(0.01, 1, num=1000), np.ones(1000), (0, np.inf))
    degree = 10
    atgrid = AtomGrid.from_pruned(odg, 1, sectors_r=[], sectors_degree=[degree])

    def func(cart_pts):
        radius = np.linalg.norm(cart_pts, axis=1)
        return np.exp(-radius)

    func_values = func(atgrid.points)

    laplacian = interpolate_laplacian(atgrid, func_values)

    # Test on the same points used for interpolation and random points.
    for grid in [atgrid.points, np.random.uniform(-0.5, 0.5, (250, 3))]:
        actual = laplacian(grid)
        spherical_pts = atgrid.convert_cartesian_to_spherical(grid)
        # Laplacian of exponential is e^-x (x - 2) / x
        desired = np.exp(-spherical_pts[:, 0]) * (spherical_pts[:, 0] - 2.0) /\
                  spherical_pts[:, 0]
        assert_almost_equal(actual, desired, decimal=3)