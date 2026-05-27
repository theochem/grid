# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2026 The GRID Development Team
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
"""Tests for analytical Coulomb potentials of Gaussians."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import erf

from grid.coulomb import coulomb_gaussian_p, coulomb_gaussian_s, coulomb_potential


def test_coulomb_gaussian_s():
    # Test r=0 limit
    assert_allclose(
        coulomb_gaussian_s(0.0, 2.0, normalized=True), 2.0 * np.sqrt(2.0) / np.sqrt(np.pi)
    )
    assert_allclose(coulomb_gaussian_s(0.0, 2.0, normalized=False), 2.0 * np.pi / 2.0)

    # Test nonzero r
    r = np.array([1.5, 3.0])
    # normalized
    assert_allclose(coulomb_gaussian_s(r, 2.0, normalized=True), erf(np.sqrt(2.0) * r) / r)
    # unnormalized
    assert_allclose(
        coulomb_gaussian_s(r, 2.0, normalized=False),
        (np.pi / 2.0) ** 1.5 * erf(np.sqrt(2.0) * r) / r,
    )


def test_coulomb_gaussian_p():
    # Test r=0 limit
    assert_allclose(
        coulomb_gaussian_p(0.0, 3.0, normalized=True),
        10.0 / 3.0 * np.sqrt(3.0) / np.sqrt(np.pi),
    )
    assert_allclose(coulomb_gaussian_p(0.0, 3.0, normalized=False), 5.0 * np.pi / 9.0)

    # Test nonzero r
    r = np.array([0.5, 2.0])
    # normalized
    exact_norm = erf(np.sqrt(3.0) * r) / r + 4.0 / 3.0 * np.sqrt(3.0 / np.pi) * np.exp(-3.0 * r**2)
    assert_allclose(coulomb_gaussian_p(r, 3.0, normalized=True), exact_norm)
    # unnormalized
    exact_unnorm = 1.5 * (np.pi**1.5 / 3.0**2.5) * erf(
        np.sqrt(3.0) * r
    ) / r + 2.0 * np.pi / 9.0 * np.exp(-3.0 * r**2)
    assert_allclose(coulomb_gaussian_p(r, 3.0, normalized=False), exact_unnorm)


def test_coulomb_potential():
    points = np.array([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]])
    centers_s = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    coeffs_s = np.array([0.5, 2.0])
    expons_s = np.array([1.5, 0.8])

    # Only s-type
    v_s = coulomb_potential(points, centers_s, coeffs_s, expons_s, normalized=True)

    # Check explicitly
    v_expected = np.zeros(len(points))
    for c, alpha, center in zip(coeffs_s, expons_s, centers_s):
        r = np.linalg.norm(points - center, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = erf(np.sqrt(alpha) * r) / r
            val[r < 1e-12] = 2.0 * np.sqrt(alpha / np.pi)
        v_expected += c * val

    assert_allclose(v_s, v_expected)
