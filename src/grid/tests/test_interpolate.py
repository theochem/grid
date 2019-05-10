"""Interpolation tests file."""
from unittest import TestCase

from numpy.testing import assert_allclose, assert_array_equal, assert_almost_equal
import numpy as np

from grid.atomic_grid import AtomicGrid
from grid.lebedev import generate_lebedev_grid
from grid.interpolate import (
    condense_values_with_sph_harms,
    interpelate,
    _generate_sph_paras,
    generate_sph_harms,
    generate_real_sph_harms,
)
from grid.onedgrid import HortonLinear


class TestInterpolate(TestCase):
    def setUp(self):
        self.ang_grid = generate_lebedev_grid(degree=7)

    def test_generate_sph_parameters(self):
        for max_l in range(20):
            l, m = _generate_sph_paras(max_l)
            assert_array_equal(l, np.arange(max_l + 1))
            # first l elements of m
            assert_array_equal(m[: max_l + 1], l)
            # last l - 1 elements of me
            assert_array_equal(m[max_l + 1 :], np.arange(-max_l, 0))

    def test_generate_sph_harms(self):
        pts = self.ang_grid.points
        wts = self.ang_grid.weights
        r = np.linalg.norm(pts, axis=1)
        # polar
        phi = np.arccos(pts[:, 2] / r)
        # azimuthal
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        # generate spherical harmonics
        sph_h = generate_sph_harms(3, theta, phi)  # l_max = 3
        assert sph_h.shape == (7, 4, 26)
        # test spherical harmonics integrated to 1 if the same index else 0.
        for _ in range(20):
            n = np.random.randint(0, 4, 2)
            m1 = np.random.randint(-n[0], n[0] + 1)
            m2 = np.random.randint(-n[1], n[1] + 1)
            re = sum(sph_h[m1, n[0]] * np.conjugate(sph_h[m2, n[1]]) * wts)
            if n[0] != n[1] or m1 != m2:
                print(n, m1, m2, re)
                assert_almost_equal(re, 0)
            else:
                print(n, m1, m2, re)
                assert_almost_equal(re, 1)

    def test_generate_real_sph_harms(self):
        pts = self.ang_grid.points
        wts = self.ang_grid.weights
        r = np.linalg.norm(pts, axis=1)
        # polar
        phi = np.arccos(pts[:, 2] / r)
        # azimuthal
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        # generate spherical harmonics
        sph_h = generate_real_sph_harms(3, theta, phi)  # l_max = 3
        assert sph_h.shape == (7, 4, 26)
        for _ in range(20):
            n = np.random.randint(0, 4, 2)
            m1 = np.random.randint(-n[0], n[0] + 1)
            m2 = np.random.randint(-n[1], n[1] + 1)
            re = sum(sph_h[m1, n[0]] * sph_h[m2, n[1]] * wts)
            if n[0] != n[1] or m1 != m2:
                print(n, m1, m2, re)
                assert_almost_equal(re, 0)
            else:
                print(n, m1, m2, re)
                assert_almost_equal(re, 1)
            # no nan in the final result
            assert np.sum(np.isnan(re)) == 0

    def helper_func_power(self, points):
        return 2 * points[:, 0] ** 2 + 3 * points[:, 1] ** 2 + 4 * points[:, 2] ** 2

    def test_condense_sph_and_interp(self):
        rad = HortonLinear(10)
        rad._points += 1
        atgrid = AtomicGrid(rad, 1, scales=[], degs=[7])
        sph_coor = atgrid.convert_cart_to_sph()
        values = self.helper_func_power(atgrid.points)
        r_sph = generate_real_sph_harms(3, sph_coor[:, 0], sph_coor[:, 1])
        result = condense_values_with_sph_harms(
            r_sph, values, atgrid.weights, atgrid.indices, rad.points
        )
        semi_sph_c = sph_coor[atgrid.indices[5] : atgrid.indices[6]]
        interp = interpelate(result, 6, semi_sph_c[:, 0], semi_sph_c[:, 1])
        # same result from points and interpolation
        assert_allclose(interp, values[atgrid.indices[5] : atgrid.indices[6]])

        # random multiple interpolation test
        for _ in range(100):
            indices = np.random.randint(1, 11, np.random.randint(1, 10))
            interp = interpelate(result, indices, semi_sph_c[:, 0], semi_sph_c[:, 1])
            for i, j in enumerate(indices):
                assert_allclose(
                    interp[i], values[atgrid.indices[j - 1] : atgrid.indices[j]]
                )
