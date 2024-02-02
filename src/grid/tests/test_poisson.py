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
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import erf

from grid.atomgrid import AtomGrid
from grid.becke import BeckeWeights
from grid.molgrid import MolGrid
from grid.onedgrid import GaussLaguerre, GaussLegendre, OneDGrid, Trapezoidal
from grid.poisson import interpolate_laplacian, solve_poisson_bvp, solve_poisson_ivp
from grid.rtransform import (
    BeckeRTransform,
    IdentityRTransform,
    InverseRTransform,
    LinearFiniteRTransform,
)
from grid.utils import convert_cart_to_sph, generate_real_spherical_harmonics

pytestmark = pytest.mark.filterwarnings(
    "ignore:The coefficient of the leading Kth term is zero at some point"
)


def zero_func(pts, centers=None):
    """Zero function for test."""
    return np.array([0.0] * pts.shape[0])


def gauss(pts, centers=None, alpha=1000.0):
    """Gaussian function for test."""
    if centers is None:
        centers = np.zeros((1, 3))
    output = np.zeros(len(pts))
    for cent in centers:
        r = np.linalg.norm(pts - cent, axis=1)
        output += np.exp(-alpha * (r**2))
    return output


def spherical_harmonic(pts, centers=None):
    r"""Y ^ 1_3, spherical harmonic with degree 3 and order 1."""
    spherical = convert_cart_to_sph(pts)
    spherical_harmonic = generate_real_spherical_harmonics(
        4, spherical[:, 1], spherical[:, 2]  # theta, phi points
    )
    return -1 * spherical_harmonic[0, :] / (4.0 * np.pi * spherical[:, 0] ** 2.0)


def charge_distribution(x, alpha=0.1, centers=None):
    if centers is None:
        centers = np.array([[0.0, 0.0, 0.0]])
    result = np.zeros(len(x))
    for cent in centers:
        r = np.linalg.norm(x - cent, axis=1)
        result += (alpha / np.pi) ** (3.0 / 2.0) * np.exp(-alpha * r**2.0)
    return result


def poisson_solution_to_charge_distribution(x, alpha=0.1, centers=None):
    if centers is None:
        centers = np.array([[0.0, 0.0, 0.0]])
    result = np.zeros(len(x))
    for cent in centers:
        r_PC = np.linalg.norm(x - cent, axis=1)
        # Ignore divide by zero and nan
        with np.errstate(divide="ignore", invalid="ignore"):
            desired = erf(np.sqrt(alpha) * r_PC) / r_PC
            desired[r_PC == 0.0] = 0.0
        result += desired
    return result


def test_interpolation_of_laplacian_with_spherical_harmonic():
    r"""Test the interpolation of Laplacian of spherical harmonic is eigenvector."""
    odg = OneDGrid(np.linspace(1e-5, 1, num=20), np.ones(20), (0, np.inf))
    degree = 6 * 2 + 2
    atgrid = AtomGrid.from_pruned(odg, 1, r_sectors=[], d_sectors=[degree])

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
    atgrid = AtomGrid.from_pruned(radial, 1, r_sectors=[], d_sectors=[degree])

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
        desired = np.exp(-spherical_pts[:, 0]) * (spherical_pts[:, 0] - 2.0) / spherical_pts[:, 0]
        assert_allclose(actual, desired, atol=1e-4, rtol=1e-6)


def test_interpolation_of_laplacian_with_unit_charge_distribution():
    r"""Test that the Laplacian of a potential gives back the unit-charge charge distribution."""
    # Construct Grids
    oned = Trapezoidal(5000)
    btf = BeckeRTransform(1e-3, 1.5)
    radial = btf.transform_1d_grid(oned)
    degree = 10
    atgrid = AtomGrid.from_pruned(radial, 1, r_sectors=[], d_sectors=[degree])

    laplace = interpolate_laplacian(atgrid, poisson_solution_to_charge_distribution(atgrid.points))
    true = laplace(atgrid.points)
    assert_allclose(-4.0 * np.pi * charge_distribution(atgrid.points), true, atol=1e-4, rtol=1e-7)


# Ignore scipy/bvp warning
@pytest.mark.filterwarnings(
    "ignore:(divide by zero encountered in divide|invalid value encountered in divide)"
)
@pytest.mark.parametrize(
    "oned, tf, remove_large_pts, centers",
    [
        # Include the origin in transformation
        [
            Trapezoidal(250),
            BeckeRTransform(0.0, 1.5, trim_inf=True),
            1e6,
            np.array([[0.0, 0.0, 0.0]]),
        ],
        # Don't include the origin, instead minimum is at 1e-6
        [
            Trapezoidal(250),
            BeckeRTransform(1e-6, 1.5, trim_inf=True),
            1e6,
            np.array([[0.0, 0.0, 0.0]]),
        ],
        # Try out the Identity Transformation.
        [GaussLaguerre(100), IdentityRTransform(), None, np.array([[0.0, 0.0, 0.0]])],
        # Multi-center example
        [
            GaussLegendre(100),
            BeckeRTransform(1e-5, R=1.5),
            10.0,
            np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        ],
    ],
)
def test_poisson_bvp_on_unit_charge_distribution(oned, tf, remove_large_pts, centers):
    r"""Test solve_poisson_bvp with unit-charge density."""
    radial = tf.transform_1d_grid(oned)
    degree = 29
    atgrids = []
    for center in centers:
        atgrid = AtomGrid(radial, center=center, degrees=[degree])
        atgrids.append(atgrid)

    if len(atgrids) == 1:  # Test providing an atomic grid only
        molgrids = atgrids[0]
    else:
        becke = BeckeWeights(order=3)
        molgrids = MolGrid(
            atnums=np.array([1] * len(centers)), atgrids=atgrids, aim_weights=becke, store=True
        )

    potential = solve_poisson_bvp(
        molgrids,
        charge_distribution(molgrids.points, centers=centers),
        InverseRTransform(tf),
        remove_large_pts=remove_large_pts,
        include_origin=True,
    )
    actual = potential(molgrids.points)
    desired = poisson_solution_to_charge_distribution(molgrids.points, centers=centers)
    assert_allclose(actual, desired, atol=1e-2)


@pytest.mark.parametrize(
    "func, centers",
    [
        [zero_func, np.array([[0.0, 0.0, 0.0]])],
        [gauss, np.array([[0.0, 0.0, 0.0]])],
        [gauss, np.array([[1.0, 0.0, 0.0]])],
        # TODO: Couldn't get the following test to pass.
        # [
        #     lambda pts, centers: gauss(pts, centers),
        #     np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        # ]
    ],
)
def test_poisson_bvp_gives_the_correct_laplacian(func, centers):
    r"""Test poisson_bvp and the laplacian of its solution match the test function."""
    oned = GaussLegendre(250)
    btf = BeckeRTransform(1e-3, 1.5)
    radial = btf.transform_1d_grid(oned)
    degree = 29
    atgrids = []
    for center in centers:
        atgrid = AtomGrid(radial, degrees=[degree], center=center)
        atgrids.append(atgrid)

    if len(atgrids) == 1:  # If atomic grid
        molgrids = atgrids[0]
    else:
        becke = BeckeWeights(order=3)
        molgrids = MolGrid(
            atnums=[1] * len(centers), atgrids=atgrids, aim_weights=becke, store=True
        )

    potential = solve_poisson_bvp(
        molgrids,
        func(molgrids.points, centers),
        InverseRTransform(btf),
        include_origin=True,
        remove_large_pts=10.0,
    )

    # Get Laplacian of the potential
    func_vals = potential(molgrids.points)
    laplace_pot = interpolate_laplacian(molgrids, func_vals)

    # Check it is the same as func on atomic grid points.
    pts = molgrids.points
    desired = laplace_pot(pts)
    assert_allclose(desired, -func(pts, centers) * 4.0 * np.pi, atol=1e-2)

    # Check it is the same as func on random set of points in the interpolation regime.
    numb_pts = 500
    l_bnd = [-0.1, -0.1, -0.1]  # Avoid zero regions but it also avoids negative regions.
    u_bnd = np.max(molgrids.points, axis=0)
    random_pts = np.array(
        [np.random.uniform(l_bnd[i], u_bnd[i], size=(numb_pts)) for i in range(0, 3)]
    ).T

    assert_allclose(laplace_pot(random_pts), -func(random_pts, centers) * 4.0 * np.pi, atol=1e-2)


@pytest.mark.parametrize(
    "func, centers",
    [
        [zero_func, np.array([[0.0, 0.0, 0.0]])],
        [gauss, np.array([[0.0, 0.0, 0.0]])],
        # [gauss, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])]
    ],
)
def test_poisson_ivp_gives_the_correct_laplacian(func, centers):
    r"""Test poisson_ivp and the laplacian of its solution match the test function."""
    oned = GaussLegendre(250)
    btf = BeckeRTransform(0.01, 1.5)
    radial = btf.transform_1d_grid(oned)
    degree = 29
    atgrids = []
    for center in centers:
        atgrid = AtomGrid(radial, degrees=[degree], center=center)
        atgrids.append(atgrid)

    if len(atgrids) == 1:  # If atomic grid
        molgrids = atgrids[0]
    else:
        becke = BeckeWeights(order=3)
        molgrids = MolGrid(
            atnums=[1] * len(centers), atgrids=atgrids, aim_weights=becke, store=True
        )

    potential = solve_poisson_ivp(
        molgrids,
        func(molgrids.points, centers),
        InverseRTransform(btf),
        r_interval=(np.max(radial.points), np.min(radial.points)),
    )

    # Get Laplacian of the potential
    func_vals = potential(molgrids.points)
    laplace_pot = interpolate_laplacian(molgrids, func_vals)

    desired = laplace_pot(molgrids.points)

    # Check it is the same as func on atomic grid points.
    np.set_printoptions(threshold=np.inf)
    np.abs(desired + func(molgrids.points) * 4.0 * np.pi)
    # TODO: Improve accuracy from one decimal place
    assert_allclose(desired, -func(molgrids.points) * 4.0 * np.pi, atol=1e-1)

    # Check it is the same as func on random set of points in the interpolation regime.
    numb_pts = 500
    l_bnd = np.min(molgrids.points, axis=0)
    u_bnd = np.max(molgrids.points, axis=0)
    random_pts = np.array(
        [np.random.uniform(l_bnd[i], u_bnd[i], size=(numb_pts)) for i in range(0, 3)]
    ).T
    assert_allclose(laplace_pot(random_pts), -func(random_pts) * 4.0 * np.pi, atol=1e-2)


@pytest.mark.parametrize(
    "centers",
    [
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        # Couldn't get this test to pass
        # np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ],
)
def test_poisson_ivp_on_unit_charge_distribution(centers):
    r"""Test poisson on unit-charge distribution as ivp problem, ignoring any zero radial points."""
    oned = Trapezoidal(10000)
    btf = LinearFiniteRTransform(1e-3, 1000)
    radial = btf.transform_1d_grid(oned)
    degree = 11
    ncenters = len(centers)
    atgrids = []
    for center in centers:
        atgrid = AtomGrid(radial, center=center, degrees=[degree])
        atgrids.append(atgrid)

    if len(atgrids) == 1:  # Test providing an atomic grid only
        molgrids = atgrids[0]
    else:
        becke = BeckeWeights(order=3)
        molgrids = MolGrid(
            atnums=np.array([1] * ncenters), atgrids=atgrids, aim_weights=becke, store=True
        )
    potential = solve_poisson_ivp(
        molgrids,
        charge_distribution(molgrids.points, centers=centers),
        InverseRTransform(btf),
        r_interval=(1000.0, 1e-3),
    )

    # Check on the atomic grid points removing any zero due to discontinuity
    i_pts_zero = np.where(~molgrids.points.any(axis=1))[0]
    pts = molgrids.points.copy()
    pts = np.delete(pts, i_pts_zero, axis=0)  # Delete at those indices
    actual = potential(pts)
    desired = poisson_solution_to_charge_distribution(pts, centers=centers)
    np.abs(desired - actual)
    assert_allclose(actual, desired, atol=1e-2)

    # Choose random points that are not zero
    num_pts = 5000
    random_pts = np.random.uniform(0, 100, size=(num_pts, 3))
    actual = potential(random_pts)
    desired = poisson_solution_to_charge_distribution(random_pts, centers=centers)
    assert_allclose(actual, desired, atol=1e-2)
