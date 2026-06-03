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
    alpha = 2.0

    # Test r=0 limit (normalized)
    result = coulomb_gaussian_s(0.0, alpha, normalized=True)
    expected = 2.0 * np.sqrt(alpha) / np.sqrt(np.pi)
    assert_allclose(result, expected)

    # Test r=0 limit (unnormalized)
    result = coulomb_gaussian_s(0.0, alpha, normalized=False)
    expected = 2.0 * np.pi / alpha
    assert_allclose(result, expected)

    # Test nonzero r (normalized)
    r = np.array([1.5, 3.0])
    result = coulomb_gaussian_s(r, alpha, normalized=True)
    expected = erf(np.sqrt(alpha) * r) / r
    assert_allclose(result, expected)

    # Test nonzero r (unnormalized)
    result = coulomb_gaussian_s(r, alpha, normalized=False)
    expected = (np.pi / alpha) ** 1.5 * erf(np.sqrt(alpha) * r) / r
    assert_allclose(result, expected)


def test_coulomb_gaussian_p():
    beta = 3.0

    # Test r=0 limit (normalized)
    result = coulomb_gaussian_p(0.0, beta, normalized=True)
    expected = 10.0 / 3.0 * np.sqrt(beta) / np.sqrt(np.pi)
    assert_allclose(result, expected)

    # Test r=0 limit (unnormalized)
    result = coulomb_gaussian_p(0.0, beta, normalized=False)
    expected = 5.0 * np.pi / beta**2
    assert_allclose(result, expected)

    # Test nonzero r (normalized)
    r = np.array([0.5, 2.0])
    result = coulomb_gaussian_p(r, beta, normalized=True)
    expected_normalized = (
        erf(np.sqrt(beta) * r) / r
        + (4.0 / 3.0) * np.sqrt(beta / np.pi) * np.exp(-beta * r**2)
    )
    assert_allclose(result, expected_normalized)

    # Test nonzero r (unnormalized)
    result = coulomb_gaussian_p(r, beta, normalized=False)
    prefactor = (3.0 / 2.0) * (np.pi ** (3.0 / 2.0)) / (beta ** (5.0 / 2.0))
    expected = prefactor * expected_normalized
    assert_allclose(result, expected)


def test_coulomb_potential():
    points = np.array([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]])
    centers_s = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    coeffs_s = np.array([0.5, 2.0])
    alphas_s = np.array([1.5, 0.8])

    # Only s-type
    v_s = coulomb_potential(points, centers_s, coeffs_s, alphas_s, normalized=True)

    # Check explicitly
    v_expected = np.zeros(len(points))
    for c, alpha, center in zip(coeffs_s, alphas_s, centers_s):
        r = np.linalg.norm(points - center, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = erf(np.sqrt(alpha) * r) / r
            val[r < 1e-12] = 2.0 * np.sqrt(alpha / np.pi)
        v_expected += c * val

    assert_allclose(v_s, v_expected)
