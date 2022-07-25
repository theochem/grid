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
from grid.atomgrid import AtomGrid
from grid.onedgrid import GaussChebyshev, GaussLaguerre, OneDGrid, Trapezoidal
from grid.poisson import interpolate_laplacian, solve_poisson_bvp, solve_poisson_ivp
from grid.rtransform import (
    BeckeRTransform,
    IdentityRTransform,
    InverseRTransform,
    LinearFiniteRTransform,
)
from grid.utils import convert_cart_to_sph, generate_real_spherical_harmonics

import numpy as np
from numpy.testing import assert_allclose

import pytest

from scipy.special import erf


def test_interpolation_of_laplacian_with_spherical_harmonic():
    r"""Test the interpolation of Laplacian of spherical harmonic is eigenvector."""
    odg = OneDGrid(np.linspace(1e-5, 1, num=20), np.ones(20), (0, np.inf))
    degree = 6 * 2 + 2
    atgrid = AtomGrid.from_pruned(odg, 1, sectors_r=[], sectors_degree=[degree])

    def func(sph_points):
        # Spherical harmonic of degree 6 and order 0
        r, phi, theta = sph_points.T
        return (
            np.sqrt(2.0)
            * np.sqrt(13)
            / (np.sqrt(np.pi) * 32)
            * (
                231 * np.cos(theta) ** 6.0
                - 315 * np.cos(theta) ** 4.0
                + 105 * np.cos(theta) ** 2.0
                - 5.0
            )
        )

    # Get spherical points from atomic grid
    spherical_pts = atgrid.convert_cartesian_to_spherical()
    func_values = func(spherical_pts)

    laplacian = interpolate_laplacian(atgrid, func_values)

    # Test on the same points used for interpolation and random points.
    for grid in [atgrid.points, np.random.uniform(-0.75, 0.75, (250, 3))]:
        actual = laplacian(grid)
        spherical_pts = atgrid.convert_cartesian_to_spherical(grid)
        # Eigenvector spherical harmonic times l(l + 1) / r^2
        with np.errstate(divide="ignore", invalid="ignore"):
            desired = -func(spherical_pts) * 6 * (6 + 1) / spherical_pts[:, 0] ** 2.0
        # desired[spherical_pts[:, 0]**2.0 < 1e-10] = -np.inf
        np.set_printoptions(threshold=np.inf)
        assert_allclose(actual, desired)


def test_interpolation_of_laplacian_of_exponential():
    r"""Test the interpolation of Laplacian of exponential."""
    oned = Trapezoidal(5000)
    btf = BeckeRTransform(0.0, 1.5)
    radial = btf.transform_1d_grid(oned)
    degree = 10
    atgrid = AtomGrid.from_pruned(radial, 1, sectors_r=[], sectors_degree=[degree])

    def func(cart_pts):
        radius = np.linalg.norm(cart_pts, axis=1)
        return np.exp(-radius)

    func_values = func(atgrid.points)

    laplacian = interpolate_laplacian(atgrid, func_values)

    # Test on the same points used for interpolation and random points.
    for grid in [atgrid.points, np.random.uniform(-0.5, 0.5, (250, 3))]:
        actual = laplacian(grid)
        spherical_pts = atgrid.convert_cartesian_to_spherical(grid)
        spherical_pts[:, 0][spherical_pts[:, 0] < 1e-6] = 1e-6
        # Laplacian of exponential is e^-x (x - 2) / x
        desired = (
            np.exp(-spherical_pts[:, 0])
            * (spherical_pts[:, 0] - 2.0)
            / spherical_pts[:, 0]
        )
        assert_allclose(actual, desired, atol=1e-4, rtol=1e-6)


def test_interpolation_of_laplacian_with_unit_charge_distribution():
    r"""Test that the Laplacian of a potential gives back the unit-charge charge distribution."""
    # Construct Grids
    oned = Trapezoidal(5000)
    btf = BeckeRTransform(1e-3, 1.5)
    radial = btf.transform_1d_grid(oned)
    degree = 10
    atgrid = AtomGrid.from_pruned(radial, 1, sectors_r=[], sectors_degree=[degree])

    # Charge distribution
    def charge_distribution(x, alpha=0.25):
        r = np.linalg.norm(x, axis=1)
        return (alpha / np.pi) ** (3.0 / 2.0) * np.exp(-alpha * r**2.0)

    # Potential
    def potential(x, alpha=0.25):
        r_PC = np.linalg.norm(x, axis=1)
        desired = erf(np.sqrt(alpha) * r_PC) / r_PC
        desired[r_PC == 0.0] = 0.0
        return desired

    laplace = interpolate_laplacian(atgrid, potential(atgrid.points))
    true = laplace(atgrid.points)
    assert_allclose(
        -4.0 * np.pi * charge_distribution(atgrid.points), true, atol=1e-4, rtol=1e-7
    )


def zero_func(pts):
    """Zero function for test."""
    return np.array([0.0] * pts.shape[0])


def gauss(pts):
    """Gaussian function for test."""
    r = np.linalg.norm(pts, axis=1)
    return np.exp(-(r**2))


def spherical_harmonic(pts):
    r"""Y ^ 1_3, sphericla harmonic with degree 3 and order 1."""
    spherical = convert_cart_to_sph(pts)
    spherical_harmonic = generate_real_spherical_harmonics(
        4, spherical[:, 1], spherical[:, 2]  # theta, phi points
    )
    return -1 * spherical_harmonic[0, :] / (4.0 * np.pi * spherical[:, 0] ** 2.0)


@pytest.mark.parametrize(
    "oned, tf, remove_large_pts",
    [
        [Trapezoidal(10000), BeckeRTransform(0.0, 1.5, trim_inf=True), 1e6],
        [Trapezoidal(10000), BeckeRTransform(1e-6, 1.5, trim_inf=True), 1e6],
        [GaussLaguerre(100), IdentityRTransform(), None],
    ],
)
def test_poisson_bvp_on_unit_charge_distribution(oned, tf, remove_large_pts):
    r"""Test solve_poisson_bvp with unit-charge density."""
    radial = tf.transform_1d_grid(oned)
    degree = 10
    atgrid = AtomGrid(radial, degrees=[degree])

    # Func
    def charge_distribution(x, alpha=0.1):
        r = np.linalg.norm(x, axis=1)
        return (alpha / np.pi) ** (3.0 / 2.0) * np.exp(-alpha * r**2.0)

    def actual_answer(x, alpha=0.1):
        r_PC = np.linalg.norm(x, axis=1)
        desired = erf(np.sqrt(alpha) * r_PC) / r_PC
        desired[r_PC == 0.0] = 0.0
        return desired

    potential = solve_poisson_bvp(
        atgrid,
        charge_distribution(atgrid.points),
        InverseRTransform(tf),
        remove_large_pts=remove_large_pts,
    )
    actual = potential(atgrid.points)
    desired = actual_answer(atgrid.points)
    assert_allclose(actual, desired, atol=1e-4)


@pytest.mark.parametrize(
    "func, tf",
    [
        [zero_func, InverseRTransform(BeckeRTransform(0.0, 1.5))],
        [spherical_harmonic, InverseRTransform(BeckeRTransform(1e-2, 1.5))],
        [gauss, InverseRTransform(BeckeRTransform(1e-2, 1.5, trim_inf=True))],
    ],
)
def test_poisson_bvp_gives_the_correct_laplacian(func, tf):
    r"""Test poisson_bvp and the laplacian of its solution match the test function."""
    oned = GaussChebyshev(500)
    btf = BeckeRTransform(0.1, 1.5)
    radial = btf.transform_1d_grid(oned)
    # new_pts = tf.transform_1d_grid(radial)  # Points in [-1, 1]
    degree = 10
    atgrid = AtomGrid.from_pruned(radial, 1, sectors_r=[], sectors_degree=[degree])

    potential = solve_poisson_bvp(atgrid, func(atgrid.points), InverseRTransform(btf))

    # Get Laplacian of the potential
    func_vals = potential(atgrid.points)
    laplace_pot = interpolate_laplacian(atgrid, func_vals)

    desired = laplace_pot(atgrid.points)

    # Check it is the same as func on atomic grid points.
    assert_allclose(desired, -func(atgrid.points) * 4.0 * np.pi, atol=1e-2)

    # Check it is the same as func on random set of points in the interpolation regime.
    numb_pts = 500
    l_bnd = np.min(atgrid.points, axis=0)
    u_bnd = np.max(atgrid.points, axis=0)
    random_pts = np.array(
        [np.random.uniform(l_bnd[i], u_bnd[i], size=(numb_pts)) for i in range(0, 3)]
    ).T

    assert_allclose(laplace_pot(random_pts), -func(random_pts) * 4.0 * np.pi, atol=1e-2)


@pytest.mark.parametrize(
    "oned, tf",
    [
        [Trapezoidal(10000), LinearFiniteRTransform(0.0, 1000)],
        # [GaussChebyshev(1000), LinearFiniteRTransform(0.0, 1000)],
        # [Trapezoidal(10000), KnowlesRTransform(0.0, 1.5, k=1, trim_inf=True)]
    ],
)
def test_poisson_on_unit_charge_distribution_ivp(oned, tf):
    r"""Test poisson on unit-charge distribution as ivp problem, ignoring any zero radial points."""
    radial = tf.transform_1d_grid(oned)
    degree = 10
    atgrid = AtomGrid(radial, degrees=[degree])

    # Func
    def charge_distribution(x, alpha=0.1):
        r = np.linalg.norm(x, axis=1)
        return (alpha / np.pi) ** (3.0 / 2.0) * np.exp(-alpha * r**2.0)

    def actual_answer(x, alpha=0.1):
        r_PC = np.linalg.norm(x, axis=1)
        desired = erf(np.sqrt(alpha) * r_PC) / r_PC
        desired[r_PC == 0.0] = 0.0
        return desired

    potential = solve_poisson_ivp(
        atgrid,
        charge_distribution(atgrid.points),
        InverseRTransform(tf),
        r_interval=(1000, 1e-3),
    )
    # Check on the atomic grid points removing any zero due to discontinuity
    i_pts_zero = np.where(~atgrid.points.any(axis=1))[0]
    pts = atgrid.points.copy()
    pts = np.delete(pts, i_pts_zero, axis=0)  # Delete at those indices
    actual = potential(pts)
    desired = actual_answer(pts)
    assert_allclose(actual, desired, atol=1e-4)

    # Choose random points that are not zero
    num_pts = 5000
    random_pts = np.random.uniform(0, 100, size=(num_pts, 3))
    actual = potential(random_pts)
    desired = actual_answer(random_pts)
    assert_allclose(actual, desired, atol=1e-4)
