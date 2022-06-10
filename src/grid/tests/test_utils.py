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
"""Utils function test file."""
from unittest import TestCase

from grid.lebedev import AngularGrid
from grid.utils import (
    convert_cart_to_sph,
    generate_derivative_real_spherical_harmonics,
    generate_real_spherical_harmonics,
    get_cov_radii,
)

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

import pytest


class TestUtils(TestCase):
    """Test class for functions in grid.utils."""

    def test_get_atomic_radii(self):
        """Test get_cov_radii function for all atoms."""
        # fmt: off
        Bragg_Slater = np.array([
            np.nan, 0.25, np.nan, 1.45, 1.05, 0.85, 0.7, 0.65, 0.6, 0.5, np.nan,
            1.8, 1.5, 1.25, 1.1, 1., 1., 1., np.nan, 2.2, 1.8, 1.6, 1.4, 1.35,
            1.4, 1.4, 1.4, 1.35, 1.35, 1.35, 1.35, 1.3, 1.25, 1.15, 1.15, 1.15,
            np.nan, 2.35, 2., 1.8, 1.55, 1.45, 1.45, 1.35, 1.3, 1.35, 1.4, 1.6,
            1.55, 1.55, 1.45, 1.45, 1.4, 1.4, np.nan, 2.6, 2.15, 1.95, 1.85,
            1.85, 1.85, 1.85, 1.85, 1.85, 1.8, 1.75, 1.75, 1.75, 1.75, 1.75,
            1.75, 1.75, 1.55, 1.45, 1.35, 1.35, 1.3, 1.35, 1.35, 1.35, 1.5,
            1.9, 1.8, 1.6, 1.9, np.nan, np.nan
        ])
        Cambridge = np.array([
            np.nan, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
            1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.7, 1.6,
            1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19,
            1.20, 1.20, 1.16, 2.20, 1.95, 1.9, 1.75, 1.64, 1.54, 1.47, 1.46,
            1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44,
            2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92,
            1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.7, 1.62, 1.51, 1.44, 1.41,
            1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.5, 1.5,
        ])
        # fmt: on
        all_index = np.arange(1, 87)
        bragg = get_cov_radii(all_index, type="bragg")
        assert_allclose(bragg, Bragg_Slater[1:] * 1.8897261339213)
        bragg = get_cov_radii(all_index, type="cambridge")
        assert_allclose(bragg, Cambridge[1:] * 1.8897261339213)

    def test_generate_real_spherical_is_accurate(self):
        r"""Test generated real spherical harmonic up to degree 3 is accurate."""
        numb_pts = 100
        pts = np.random.uniform(-1.0, 1.0, size=(numb_pts, 3))
        sph_pts = convert_cart_to_sph(pts)
        r, theta, phi = sph_pts[:, 0], sph_pts[:, 1], sph_pts[:, 2]
        sph_h = generate_real_spherical_harmonics(3, theta, phi)  # l_max = 3
        # Test l=0, m=0
        assert_allclose(sph_h[0, :], np.ones(len(theta)) / np.sqrt(4.0 * np.pi))
        # Test l=1, m=0, obtained from wikipedia
        assert_allclose(sph_h[1, :], np.sqrt(3.0 / (4.0 * np.pi)) * pts[:, 2] / r)
        # Test l=1, m=1, m=0
        assert_allclose(sph_h[2, :], np.sqrt(3.0 / (4.0 * np.pi)) * pts[:, 0] / r)
        assert_allclose(sph_h[3, :], np.sqrt(3.0 / (4.0 * np.pi)) * pts[:, 1] / r)

    def test_generate_real_spherical_is_orthonormal(self):
        """Test generated real spherical harmonics is an orthonormal set."""
        atgrid = AngularGrid(degree=7)
        pts = atgrid.points
        wts = atgrid.weights
        r = np.linalg.norm(pts, axis=1)
        # polar
        phi = np.arccos(pts[:, 2] / r)
        # azimuthal
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        # generate spherical harmonics
        sph_h = generate_real_spherical_harmonics(3, theta, phi)  # l_max = 3
        assert sph_h.shape == (16, 26)
        for _ in range(100):
            n1, n2 = np.random.randint(0, 16, 2)
            re = sum(sph_h[n1] * sph_h[n2] * wts)
            if n1 != n2:
                print(n1, n2, re)
                assert_almost_equal(re, 0)
            else:
                print(n1, n2, re)
                assert_almost_equal(re, 1)

        for i in range(10):
            sph_h = generate_real_spherical_harmonics(i, theta, phi)
            assert sph_h.shape == ((i + 1) ** 2, 26)

    def test_generate_real_sph_harms_integrates_correctly(self):
        """Test generated real spherical harmonics integrates correctly."""
        angular = AngularGrid(degree=7)
        pts = angular.points
        wts = angular.weights
        r = np.linalg.norm(pts, axis=1)
        # polar
        phi = np.arccos(pts[:, 2] / r)
        # azimuthal
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        # generate spherical harmonics
        lmax = 3
        sph_h = generate_real_spherical_harmonics(lmax, theta, phi)  # l_max = 3
        # Test the shape matches (order, degree, number of points)
        assert sph_h.shape == (1 + 3 + 5 + 7, 26)
        counter = 0
        for l_value in range(0, lmax + 1):
            for m in (
                [0]
                + [x for x in range(1, l_value + 1)]
                + [-x for x in range(1, l_value + 1)]
            ):
                # print(l_value, m)
                re = sum(sph_h[counter, :] * wts)
                if l_value == 0:
                    assert_almost_equal(re, np.sqrt(4.0 * np.pi))
                else:
                    assert_almost_equal(re, 0)
                # no nan in the final result
                assert np.sum(np.isnan(re)) == 0
                counter += 1

    def test_convert_cart_to_sph(self):
        """Test convert_cart_sph accuracy."""
        for _ in range(10):
            pts = np.random.rand(10, 3)
            center = np.random.rand(3)
            # given center
            sph_coor = convert_cart_to_sph(pts, center)
            assert_equal(sph_coor.shape, pts.shape)
            # Check Z
            z = sph_coor[:, 0] * np.cos(sph_coor[:, 2])
            assert_allclose(z, pts[:, 2] - center[2])
            # Check x
            xy = sph_coor[:, 0] * np.sin(sph_coor[:, 2])
            x = xy * np.cos(sph_coor[:, 1])
            assert_allclose(x, pts[:, 0] - center[0])
            # Check y
            y = xy * np.sin(sph_coor[:, 1])
            assert_allclose(y, pts[:, 1] - center[1])

            # no center
            sph_coor = convert_cart_to_sph(pts)
            assert_equal(sph_coor.shape, pts.shape)
            # Check Z
            z = sph_coor[:, 0] * np.cos(sph_coor[:, 2])
            assert_allclose(z, pts[:, 2])
            # Check x
            xy = sph_coor[:, 0] * np.sin(sph_coor[:, 2])
            x = xy * np.cos(sph_coor[:, 1])
            assert_allclose(x, pts[:, 0])
            # Check y
            y = xy * np.sin(sph_coor[:, 1])
            assert_allclose(y, pts[:, 1])

    def test_convert_cart_to_sph_origin(self):
        """Test convert_cart_sph at origin."""
        # point at origin
        pts = np.array([[0.0, 0.0, 0.0]])
        sph_coor = convert_cart_to_sph(pts, center=None)
        assert_allclose(sph_coor, 0.0)
        # point very close to origin
        pts = np.array([[1.0e-15, 1.0e-15, 1.0e-15]])
        sph_coor = convert_cart_to_sph(pts, center=None)
        assert sph_coor[0, 0] < 1.0e-12
        assert np.all(sph_coor[0, 1:] < 1.0)
        # point very very close to origin
        pts = np.array([[1.0e-100, 1.0e-100, 1.0e-100]])
        sph_coor = convert_cart_to_sph(pts, center=None)
        assert sph_coor[0, 0] < 1.0e-12
        assert np.all(sph_coor[0, 1:] < 1.0)

    def test_raise_errors(self):
        """Test raise proper errors."""
        with self.assertRaises(ValueError):
            get_cov_radii(3, type="random")
        with self.assertRaises(ValueError):
            get_cov_radii(0)
        with self.assertRaises(ValueError):
            get_cov_radii([3, 5, 0])

        with self.assertRaises(ValueError):
            pts = np.random.rand(10)
            convert_cart_to_sph(pts, np.zeros(3))
        with self.assertRaises(ValueError):
            pts = np.random.rand(10, 3, 1)
            convert_cart_to_sph(pts, np.zeros(3))
        with self.assertRaises(ValueError):
            pts = np.random.rand(10, 2)
            convert_cart_to_sph(pts, np.zeros(3))
        with self.assertRaises(ValueError):
            pts = np.random.rand(10, 3)
            convert_cart_to_sph(pts, np.zeros(2))


@pytest.mark.parametrize(
    "numb_pts, max_degree", [[1000, 10], [5000, 2], [100, 15], [10, 20]]
)
def test_derivative_of_spherical_harmonics_with_finite_difference(numb_pts, max_degree):
    """Test the derivative value from spherical harmonics with finite difference."""
    theta = np.array([1e-5])
    theta = np.hstack((theta, np.random.uniform(0.0, 2.0 * np.pi, size=(numb_pts,))))
    phi = np.array([1e-5])
    phi = np.hstack((phi, np.random.uniform(0.0, np.pi, size=(numb_pts,))))
    eps = 1e-7
    l_max = max_degree
    value = generate_real_spherical_harmonics(l_max, theta, phi)
    value_at_eps_theta = generate_real_spherical_harmonics(l_max, theta + eps, phi)
    value_at_eps_phi = generate_real_spherical_harmonics(l_max, theta, phi + eps)
    actual_answer = generate_derivative_real_spherical_harmonics(l_max, theta, phi)
    deriv_theta = (value_at_eps_theta - value) / eps
    deriv_phi = (value_at_eps_phi - value) / eps

    assert_almost_equal(actual_answer[0, :], deriv_theta, decimal=3)
    assert_almost_equal(actual_answer[1, :], deriv_phi, decimal=3)
