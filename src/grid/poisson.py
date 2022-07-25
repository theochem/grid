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
"""Poisson solver module."""

from grid.atomgrid import AtomGrid
from grid.ode import solve_ode_bvp, solve_ode_ivp
from grid.rtransform import BaseTransform
from grid.utils import generate_real_spherical_harmonics, convert_cart_to_sph

import numpy as np

from scipy.interpolate import CubicSpline, interp1d

import warnings
from typing import Union

__all__ = ["solve_poisson_bvp", "solve_poisson_ivp"]


def solve_poisson_ivp(
        atomgrid : AtomGrid,
        func_vals : np.ndarray,
        transform : BaseTransform,
        r_interval : tuple = (1000, 1e-5),
        ode_params : Union[dict, type(None)] = None
):
    r"""
    Return interpolation of the solution to the Poisson equation solved as an initial value problem.

    The Poisson equation solves for function :math:`g` of the following:

    .. math::
        \Delta g = (-4\pi) f,

    for a fixed function :math:`f`, where :math:`\Delta` is the Laplacian.  This
    is transformed to an set of ODE problems as a initial value problem.

    Ihe initial value problem is chosen so that the boundary of :math:`g` for large r is set to
    :math:`\int \int \int f(r, \theta, \phi) / r`.  Depending on :math:`f`, this function has
    difficulty in capturing the origin :math:`r=0` region, and is recommended to keep the
    final interval :math:`a` close to zero.

    Parameters
    ----------
    atomgrid : AtomGrid
        Atomic grid that is used for integration and expanding func into real spherical
        harmonic basis.
    func_vals : ndarray(N,)
        The function values evaluated on all :math:`N` points on the atomic grid.
    transform : BaseTransform, optional
        Transformation from infinite domain :math:`r` (:math:`[0, \infty)` to another
        domain that is a finite.
    r_interval : tuple, optional
        The interval :math:`(b, a)` of :math:`r` for which the ODE solver will start from and end,
        where :math:`b>a`. The value :math:`b` should be large as it determines the asymptotic
        region of :math:`g` and value :math:`a` is recommended to be small but not zero depending
        on :math:`f`.
    ode_params : dict, optional
        The parameters for the ode solver. See `grid.ode.solve_ode_ivp` for all options.

    Returns
    -------
    callable(ndarray(N, 3) -> float) :
        The solution to Poisson equaiton/potential :math:`g : \mathbb{R}^3 \rightarrow \mathbb{R}`.

    Examples
    --------
    Set up of the radial grid
    >>> oned_grid = Trapezoidal(10000)
    >>> tf = LinearFiniteRTransform(0.0, 1000)
    >>> radial_grid = tf.transform_1d_grid(oned)
    Set up the atomic grid with degree 10 at each radial point.
    >>> atomic_grid = AtomGrid(radial_grid, degrees=[10])
    Set the charge distribution to be unit-charge density and evaluate on atomic grid points.
    >>> def charge_distribution(x, alpha=0.1):
    >>>    r = np.linalg.norm(x, axis=1)
    >>>    return (alpha / np.pi)**(3.0 / 2.0) * np.exp(-alpha * r**2.0)
    >>> func_vals = charge_distribution(atomic_grid.points)
    Solve for the potential as an initial value problem and evaluate it over the atomic grid.
    >>> potential = solve_poisson_ivp(
    >>>      atgrid, func_vals, InverseRTransform(tf), r_interval=(1000, 1e-3),
    >>>      ode_params={"method" : "DOP853", "atol": 1e-8},
    >>> )
    >>> potential_values = potential(atgrid.points)

    """
    if r_interval[0] < r_interval[1]:
        raise ValueError(
            f"Initial Radial interval {r_interval[0]} should be greater than "
            f"second entry {r_interval[1]}."
        )

    # Get the radial components from expanding func into real spherical harmonics.
    radial_components = atomgrid.radial_component_splines(func_vals)

    # Following takes the integral of f(x), to generate the bounds at very large r.
    # The calculation of the bounds is explained below.
    sph_o_l = generate_real_spherical_harmonics(0, np.array([0.1]), np.array([0.1]))
    r_max = r_interval[0]
    boundary = atomgrid.integrate(func_vals) / sph_o_l[0, 0]

    # Set up default ode parameters if it isn't set up already.
    if ode_params is None:
        ode_params = dict({})
    ode_params.setdefault("method", "DOP853")
    ode_params.setdefault("rtol", 1e-8)
    ode_params.setdefault("atol", 1e-6)

    splines = []
    i_spline = 0
    for l in range(0, atomgrid.l_max // 2 + 1):
        for m in [x for x in range(0, l + 1)] + [-x for x in range(-l, 0)]:
            def f_x(r):
                return radial_components[i_spline](r) * -4 * np.pi

            def coeff_0(r):
                a = -l * (l + 1) / r ** 2
                return a

            def coeff_1(r):
                # with np.errstate(divide='ignore', invalid='ignore'):
                return 2.0 / r

            # Set up coefficients of the ode
            coeffs = [coeff_0, coeff_1, 1]
            # initial values.
            if l == 0 and m == 0:
                # Solution to Poisson is electrostatic potential g(r) = \int f(x) /|x - r| dx,
                # for large r, we have r >> x, so |x-r| \approx |r| so that we have the following
                # bound g(large r) = \int f(x) dx / |r|,  this gives the first ivp, then
                # taking the derivative wrt to r gives the bound for the derivative at large r.
                ivp = [boundary / r_max, -boundary / r_max**2.0]
            else:
                ivp = [0.0, 0.0]
            # Solve ode
            u_lm = solve_ode_ivp(
                r_interval, f_x, coeffs, ivp, transform, no_derivatives=True, **ode_params
            )

            i_spline += 1
            splines.append(u_lm)


    def interpolate(points):
        r_pts, theta, phi = convert_cart_to_sph(points).T
        r_values = np.array([spline(r_pts) for spline in splines])
        r_sph_harm = generate_real_spherical_harmonics(atomgrid.l_max // 2, theta, phi)
        return np.einsum("ij, ij -> j", r_values, r_sph_harm)

    return interpolate


def solve_poisson_bvp(atomgrid, func_vals, transform, boundary=None, include_origin=True,
                      remove_large_pts=1e6, ode_params=None):
    r"""
    Return interpolation of the solution to the Poisson equation solved as a boundary value problem.

    The Poisson equation solves for function :math:`g` of the following:

    .. math::
        \Delta g = (-4\pi) f,

    for a fixed function :math:`f`, where :math:`\Delta` is the Laplacian.  This
    is transformed to an set of ODE problems as a boundary value problem.

    If boundary is not provided, then the boundary of :math:`g` for large r is set to
    :math:`\int \int \int f(r, \theta, \phi) / r`.

    Parameters
    ----------
    atomgrid : AtomGrid
        Atomic grid that is used for integration and expanding func into real spherical
        harmonic basis.
    func_vals : ndarray(N,)
        The function values evaluated on all :math:`N` points on the atomic grid.
    transform : BaseTransform, optional
        Transformation from infinite domain :math:`r` (:math:`[0, \infty)` to another
        domain that is a finite.
    boundary : float, optional
        The boundary value of :math:`g` in the limit of r to infinity.
    include_origin : bool, optional
        If true, will add r=0 point when solving for the ode only. If false, it is recommended
        to have many radial points near the origin.
    remove_large_pts : float, optional
        If true, will remove any points larger than `remove_large_pts` when solving for the ode
        only.
    ode_params : dict, optional
        The parameters for the ode solver. See `grid.ode.solve_ode_bvp` for all options.

    Returns
    -------
    callable(ndarray(N, 3) -> float) :
        The solution to Poisson equaiton/potential :math:`g : \mathbb{R}^3 \rightarrow \mathbb{R}`.

    Examples
    --------
    Set up of the radial grid
    >>> radial_grid = Trapezoidal(10000)
    Set up the atomic grid with degree 10 at each radial point.
    >>> degree = 10
    >>> atomic_grid = AtomGrid(radial, degrees=[degree])
    Set the charge distribution to be unit-charge density and evaluate on atomic grid points.
    >>> def charge_distribution(x, alpha=0.1):
    >>>    r = np.linalg.norm(x, axis=1)
    >>>    return (alpha / np.pi)**(3.0 / 2.0) * np.exp(-alpha * r**2.0)
    >>> func_vals = charge_distribution(atomic_grid.points)
    Solve the Poisson equation with Becke transformation
    >>> transform = BeckeRTransform(1e-6, 1.5, trim_inf=True)
    >>> potential = solve_poisson_bvp(
    >>>      atgrid, func_vals, InverseRTransform(tf), include_origin=True,
    >>>      remove_large_pts=1e6, ode_params={"tol" : 1e-6, "max_nodes": 20000},
    >>> )
    >>> actual = potential(atgrid.points)

    References
    ----------
    .. [1] Becke, A. D., & Dickson, R. M. (1988). Numerical solution of Poisson’s equation in
           polyatomic molecules. The Journal of chemical physics, 89(5), 2993-2997.

    """
    if not isinstance(boundary, (float, type(None))):
        raise TypeError(f"`boundary` {type(boundary)} should be either float or None.")
    if not isinstance(include_origin, bool):
        raise TypeError(f"`include_origin` {type(include_origin)} should be boolean.")
    if not isinstance(remove_large_pts, (float, type(None))):
        raise TypeError(f"`remove_large_pts` {type(remove_large_pts)} should be either "
                        f"float or None.")

    # If boundary is None: then bnd sets to \integral of func_vals / Spherical Harmonic at (0, 0)
    if boundary is None:
        # Since spherical harmonic at (0, 0) is a constant, then pick a random point.
        sph_o_l = generate_real_spherical_harmonics(0, np.array([0.1]), np.array([0.1]))
        boundary = atomgrid.integrate(func_vals) / sph_o_l[0, 0]

    # Check if the domain of transform is in [0, \infty)
    domain = transform.domain
    if domain[0] < 0.0:
        raise ValueError(f"The domain of the transform {domain} should be in [0, infinity).")

    # Get the radial components from expanding func into real spherical harmonics.
    radial_components = atomgrid.radial_component_splines(func_vals)

    # Include the origin r=0 point if it isn't included, this is due to potential
    # expanded with 1/r factor and so to be finite at r=0, the numerator should be zero.
    points = atomgrid.rgrid.points.copy()
    if include_origin:
        if np.all(atomgrid.rgrid.points > 0.0):
            points = np.hstack((np.array([0.0]), points))

    # Get indices where the points are large and remove them for ill-conditioning of ode-solver.
    if remove_large_pts is not None:
        indices = np.where(points > remove_large_pts)[0]
        points = np.delete(points, indices)

    # Set up default ode parameters if it isn't set up already.
    if ode_params is None:
        ode_params = dict({})
    ode_params.setdefault("tol", 1e-6)
    ode_params.setdefault("max_nodes", 50000)
    ode_params.setdefault("no_derivatives", True)

    splines = []
    i_spline = 0
    for l in range(0, atomgrid.l_max // 2 + 1):
        for m in [x for x in range(0, l + 1)] + [-x for x in range(-l, 0)]:
            def f_x(r):
                return radial_components[i_spline](r) * -4 * np.pi * r

            def coeff_0(r):
                with np.errstate(divide='ignore', invalid='ignore'):
                    a = -l * (l + 1) / r ** 2
                # Note that this assumes the boundary condition that y(0) = 0.
                a[np.abs(r) == 0.0] = 0.0
                return a

            # Set up coefficients of the ode
            coeffs = [coeff_0, 0, 1]

            # Set up boundary conditions
            if l == 0 and m == 0:
                bd_cond = [(0, 0, 0), (1, 0, boundary)]
            else:
                bd_cond = [(0, 0, 0), (1, 0, 0)]

            # Solve ode
            u_lm = solve_ode_bvp(
                points, f_x, coeffs, bd_cond, transform, **ode_params
            )
            i_spline += 1
            splines.append(u_lm)

    def interpolate(points):
        r_pts, theta, phi = convert_cart_to_sph(points).T
        r_values = np.array([spline(r_pts) / r_pts for spline in splines])
        # Since splin(r=0) = 0, then set points to zero there.
        r_values[:, np.abs(r_pts) < 1e-300] = 0.0
        r_sph_harm = generate_real_spherical_harmonics(atomgrid.l_max // 2, theta, phi)
        return np.einsum("ij, ij -> j", r_values, r_sph_harm)

    return interpolate


def interpolate_laplacian(atomgrid: AtomGrid, func_vals: np.ndarray):
    r"""
    Return a function that interpolates the Laplacian of a function.

    .. math::
        \Deltaf = \frac{1}{r}\frac{\partial^2 rf}{\partial r^2} - \frac{\hat{L}}{r^2},

    such that the angular momentum operator satisfies :math:`\hat{L}(Y_l^m) = l (l + 1) Y_l^m`.
    Expanding f in terms of spherical harmonic expansion, we get that

    .. math::
        \Deltaf = \sum \sum \bigg[\frac{\rho_{lm}^{rf}}{r} -
        \frac{l(l+1) \rho_{lm}^f}{r^2} \bigg] Y_l^m,

    where :math:`\rho_{lm}^f` is the lth, mth radial component of function f.

    Parameters
    ----------
    atomgrid : AtomGrid
        Atomic grid that can integrate spherical functions and interpolate radial components.
    func_vals : ndarray(N,)
        The function values evaluated on all :math:`N` points on the atomic grid.

    Returns
    -------
    callable[ndarray(M,3) -> ndarray(M,)] :
        Function that interpolates the Laplacian of a function whose input is
        Cartesian points.

    """
    radial, theta, phi = atomgrid.convert_cartesian_to_spherical().T
    # Multiply f by r to get rf
    func_vals_radial = radial * func_vals

    # compute spline for the radial components for rf and f
    radial_comps_rf = atomgrid.radial_component_splines(func_vals_radial)
    radial_comps_f = atomgrid.radial_component_splines(func_vals)

    def interpolate_laplacian(points):
        r_pts, theta, phi = atomgrid.convert_cartesian_to_spherical(points).T

        # Evaluate the radial components for function f
        r_values_f = np.array([spline(r_pts) for spline in radial_comps_f])
        # Evaluate the second derivatives of splines of radial component of function rf
        r_values_rf = np.array([spline(r_pts, 2) for spline in radial_comps_rf])

        r_sph_harm = generate_real_spherical_harmonics(atomgrid.l_max // 2, theta, phi)

        # Divide \rho_{lm}^{rf} by r  and \rho_{lm}^{rf} by r^2
        # l means the angular (l,m) variables and n represents points.
        with np.errstate(divide='ignore', invalid='ignore'):
            r_values_rf /= r_pts
            r_values_rf[:, r_pts < 1e-10] = 0.0
            r_values_f /= r_pts**2.0
            r_values_f[:, r_pts**2.0 < 1e-10] = 0.0

        # Multiply \rho_{lm}^f by l(l+1), note 2l+1 orders for each degree l
        degrees = np.hstack([[x * (x + 1)] * (2 * x + 1) for x in
                             np.arange(0, atomgrid.l_max // 2 + 1)])
        second_component = np.einsum("ln,l->ln", r_values_f, degrees)
        # Compute \rho_{lm}^{rf}/r - \rho_{lm}^f l(l + 1) / r^2
        component = r_values_rf - second_component
        # Multiply by spherical harmonics and take the sum
        return np.einsum("ln, ln -> n", component, r_sph_harm)

    return interpolate_laplacian