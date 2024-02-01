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
r"""
Poisson solver module.

This module solves the following Poisson equation:

.. math::
    \nabla^2 V(r) = -4\pi \rho(r),

for some Coulomb potential :math:`V(r)` and charge density :math:`\rho(r)`
over a centered atomic grid. It is recommended to use the boundary value problem
for handing singularities near the origin of the atomic grid.

"""
from typing import Union

import numpy as np

from grid.atomgrid import AtomGrid
from grid.molgrid import MolGrid
from grid.ode import solve_ode_bvp, solve_ode_ivp
from grid.rtransform import BaseTransform
from grid.utils import generate_real_spherical_harmonics

__all__ = ["solve_poisson_bvp", "solve_poisson_ivp"]


def _interpolate_molgrid_helper(molgrid, func_vals, interpolate_callable):
    # Common routine for both solve_poisson_bvp and solve_poisson_ivp
    #  the difference between the two methods lies in the interpolate_callable function
    #  which takes in two parameters, the atomic grid and function values on that atomic grid.
    if isinstance(molgrid, AtomGrid):
        # Convert AtomGrid to MolGrid, may increase computational time.
        molgrid = MolGrid(
            atnums=np.array([1.0]),
            atgrids=[molgrid],
            aim_weights=np.array([1.0] * molgrid.size),
            store=True,
        )
    if molgrid.atgrids is None:
        raise ValueError("Molecular grid (MolGrid) attribute `store` should be set to True.")

    # Multiply f by the nuclear weight function w_n(r) for each atom grid segment.
    func_vals_atom = func_vals * molgrid.aim_weights
    # Go through each atomic grid and construct interpolation of f*w_n.
    intepolate_funcs = []
    for i in range(len(molgrid.atcoords)):
        # Get the atomic grid
        start_index = molgrid.indices[i]
        final_index = molgrid.indices[i + 1]
        atom_grid = molgrid[i]

        # Add the interpolation for that atom in a list
        intepolate_funcs.append(
            interpolate_callable(atom_grid, func_vals_atom[start_index:final_index])
        )

    def sum_of_interpolation_functions(points):
        output = intepolate_funcs[0](points)
        for interpolate in intepolate_funcs[1:]:
            output += interpolate(points)
        return output

    return sum_of_interpolation_functions


def _solve_poisson_ivp_atomgrid(
    atomgrid: AtomGrid,
    func_vals: np.ndarray,
    transform: BaseTransform,
    r_interval: tuple = (1000, 1e-5),
    ode_params: Union[dict, type(None)] = None,
):
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
    for l_deg in range(0, atomgrid.l_max // 2 + 1):
        for m_ord in [x for x in range(0, l_deg + 1)] + [-x for x in range(-l_deg, 0)]:

            def f_x(r, i_spline=i_spline):
                return radial_components[i_spline](r) * -4 * np.pi

            def coeff_0(r, l_deg=l_deg):
                a = -l_deg * (l_deg + 1) / r**2
                return a

            def coeff_1(r):
                # with np.errstate(divide='ignore', invalid='ignore'):
                return 2.0 / r

            # Set up coefficients of the ode
            coeffs = [coeff_0, coeff_1, 1]
            # initial values.
            if l_deg == 0 and m_ord == 0:
                # Solution to Poisson is electrostatic potential g(r) = \int f(x) /|x - r| dx,
                # for large r, we have r >> x, so |x-r| \approx |r| so that we have the following
                # bound g(large r) = \int f(x) dx / |r|,  this gives the first ivp, then
                # taking the derivative wrt to r gives the bound for the derivative at large r.
                ivp = [boundary / r_max, -boundary / r_max**2.0]
            else:
                ivp = [0.0, 0.0]

            # Solve ode
            u_lm = solve_ode_ivp(
                r_interval,
                f_x,
                coeffs,
                ivp,
                transform,
                no_derivatives=True,
                **ode_params,
            )

            i_spline += 1
            splines.append(u_lm)

    def interpolate(points):
        # Need atomgrid to center the points to the atomic grid, then convert to spherical.
        r_pts, theta, phi = atomgrid.convert_cartesian_to_spherical(points).T
        r_values = np.array([spline(r_pts) for spline in splines])
        r_sph_harm = generate_real_spherical_harmonics(atomgrid.l_max // 2, theta, phi)
        return np.einsum("ij, ij -> j", r_values, r_sph_harm)

    return interpolate


def solve_poisson_ivp(
    molgrid: Union[MolGrid, AtomGrid],
    func_vals: np.ndarray,
    transform: BaseTransform,
    r_interval: tuple = (1000, 1e-5),
    ode_params: Union[dict, type(None)] = None,
):
    r"""
    Return interpolation of the solution to the Poisson equation solved as an initial value problem.

    The Poisson equation solves for function :math:`g` of the following:

    .. math::
        \nabla^2 g = (-4\pi) f,

    for a fixed function :math:`f`, where :math:`\nabla^2` is the Laplacian.  This
    is transformed to a set of ODE problems as an initial value problem.

    Ihe initial value problem is chosen so that the boundary of :math:`g` for large r is set to
    :math:`\int \int \int f(r, \theta, \phi) / r`.  Depending on :math:`f`, this function has
    difficulty in capturing the origin :math:`r=0` region, and is recommended to keep the
    final interval :math:`a` close to zero.

    Parameters
    ----------
    molgrid : Union[MolGrid, AtomGrid]
        Molecular or atomic grid that is used for integration and expanding func into real
        spherical harmonic basis.
    func_vals : ndarray(N,)
        The function values evaluated on all :math:`N` points on the molecular grid.
    transform : BaseTransform, optional
        Transformation from infinite domain :math:`r \in [0, \infty)` to another
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

    """
    return _interpolate_molgrid_helper(
        molgrid,
        func_vals,
        lambda atom_grid, func_vals: _solve_poisson_ivp_atomgrid(
            atom_grid, func_vals, transform=transform, r_interval=r_interval, ode_params=ode_params
        ),
    )


def _solve_poisson_bvp_atomgrid(
    atomgrid: AtomGrid,
    func_vals: np.ndarray,
    transform: BaseTransform,
    boundary: Union[float, type(None)] = None,
    include_origin: bool = True,
    remove_large_pts: float = 1e6,
    ode_params: Union[dict, type(None)] = None,
):
    if not isinstance(boundary, (float, type(None))):
        raise TypeError(f"`boundary` {type(boundary)} should be either float or None.")
    if not isinstance(include_origin, bool):
        raise TypeError(f"`include_origin` {type(include_origin)} should be boolean.")
    if not isinstance(remove_large_pts, (float, type(None))):
        raise TypeError(
            f"`remove_large_pts` {type(remove_large_pts)} should be either float or None."
        )

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
    rad_points = atomgrid.rgrid.points.copy()
    if include_origin:
        if np.all(atomgrid.rgrid.points > 0.0):
            rad_points = np.hstack((np.array([0.0]), rad_points))

    # Get indices where the points are large and remove them for ill-conditioning of ode-solver.
    if remove_large_pts is not None:
        indices = np.where(rad_points > remove_large_pts)[0]
        rad_points = np.delete(rad_points, indices)

    # Set up default ode parameters if it isn't set up already.
    if ode_params is None:
        ode_params = dict({})
    ode_params.setdefault("tol", 1e-6)
    ode_params.setdefault("max_nodes", 50000)
    ode_params.setdefault("no_derivatives", True)

    splines = []
    i_spline = 0
    for l_deg in range(0, atomgrid.l_max // 2 + 1):
        for m_ord in [x for x in range(0, l_deg + 1)] + [-x for x in range(-l_deg, 0)]:

            def f_x(r, i_spline=i_spline):
                return radial_components[i_spline](r) * -4 * np.pi * r

            def coeff_0(r, l_deg=l_deg):
                with np.errstate(divide="ignore", invalid="ignore"):
                    a = -l_deg * (l_deg + 1) / r**2
                # Note that this assumes the boundary condition that y(0) = 0.
                a[np.abs(r) == 0.0] = -l_deg * (l_deg + 1) / 1e-10**2
                return a

            # Set up coefficients of the ode
            coeffs = [coeff_0, 0, 1]

            # Set up boundary conditions
            if l_deg == 0 and m_ord == 0:
                bd_cond = [(0, 0, 0), (1, 0, boundary)]
            else:
                bd_cond = [(0, 0, 0), (1, 0, 0)]

            # Solve ode
            u_lm = solve_ode_bvp(rad_points, f_x, coeffs, bd_cond, transform, **ode_params)

            i_spline += 1
            splines.append(u_lm)

    def interpolate(points):
        # Need atomgrid to center the points to the atomic grid, then convert to spherical.
        r_pts, theta, phi = atomgrid.convert_cartesian_to_spherical(points).T
        with np.errstate(divide="ignore"):
            r_values = np.array([spline(r_pts) / r_pts for spline in splines])
            # Since spline(r=0) = 0, then set points to zero there.
            r_values[:, np.abs(r_pts) < 1e-300] = 0.0
        r_sph_harm = generate_real_spherical_harmonics(atomgrid.l_max // 2, theta, phi)
        return np.einsum("ij, ij -> j", r_values, r_sph_harm)

    return interpolate


def solve_poisson_bvp(
    molgrid: Union[MolGrid, AtomGrid],
    func_vals: np.ndarray,
    transform: BaseTransform,
    boundary: Union[float, type(None)] = None,
    include_origin: bool = True,
    remove_large_pts: float = 1e6,
    ode_params: Union[dict, type(None)] = None,
):
    r"""
    Return interpolation of the solution to the Poisson equation solved as a boundary value problem.

    The Poisson equation solves for function :math:`g` of the following:

    .. math::
        \nabla^2 g = (-4\pi) f,

    for a fixed function :math:`f`, where :math:`\nabla^2` is the Laplacian.  This
    is transformed to an set of ODE problems as a boundary value problem.

    If boundary is not provided, then the boundary of :math:`g` for large r is set to
    :math:`\int \int \int f(r, \theta, \phi) / r`.  The solution :math:`g` is assumed to be
    zero at the origin :math:`g(0, \theta, \phi) = 0`.  Use `solve_poisson_ivp` if this assumption
    isn't needed.

    Parameters
    ----------
    molgrid : Union[MolGrid, AtomGrid]
        Molecular or atomic grid that is used for integration and expanding func into real spherical
        harmonic basis.
    func_vals : ndarray(N,)
        The function values evaluated on all :math:`N` points on the molecular grid.
    transform : BaseTransform, optional
        Transformation from infinite domain :math:`r \in [0, \infty)` to another
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

    References
    ----------
    .. [1] Becke, A. D., & Dickson, R. M. (1988). Numerical solution of Poisson`s equation in
           polyatomic molecules. The Journal of chemical physics, 89(5), 2993-2997.

    """
    return _interpolate_molgrid_helper(
        molgrid,
        func_vals,
        lambda atom_grid, func_vals: _solve_poisson_bvp_atomgrid(
            atom_grid, func_vals, transform, boundary, include_origin, remove_large_pts, ode_params
        ),
    )


def interpolate_laplacian(molgrid: Union[MolGrid, AtomGrid], func_vals: np.ndarray):
    r"""
    Return a function that interpolates the Laplacian of a function.

    .. math::
        \nabla^2 f = \frac{1}{r}\frac{\partial^2 rf}{\partial r^2} - \frac{\hat{L}}{r^2},

    such that the angular momentum operator satisfies :math:`\hat{L}(Y_l^m) = l (l + 1) Y_l^m`.
    Expanding f in terms of spherical harmonic expansion, we get that

    .. math::
        \nabla^2 f = \sum_l \sum_m \bigg[ \frac{\partial^2 \rho_{lm}(r)}{\partial r^2}
        + \frac{2}{r} \frac{\partial \rho_{lm}(r)}{\partial r} - \frac{l(l+1)}{r^2}\rho_{lm}(r)
         \bigg] Y_l^m,

    where :math:`\rho_{lm}^f` is the lth, mth radial component of function f.

    Parameters
    ----------
    molgrid : Union[MolGrid, AtomGrid]
        Atomic grid that can integrate spherical functions and interpolate radial components.
    func_vals : ndarray(N,)
        The function values evaluated on all :math:`N` points on the atomic grid.

    Returns
    -------
    callable[ndarray(M,3), float -> ndarray(M,)] :
        Function that interpolates the Laplacian of a function whose input is
        Cartesian points. The float value is the cutoff where radial points
        smaller than the cutoff are replaced with the cutoff. Computing the Laplacian at r=0 can
        cause problems depending on the function provided.

    Warnings
    --------
    - Since :math:`\rho_{lm}` and its derivatives are being interpolated and due to division by
      powers of :math:`r`, it is recommended to be very careful of having values near zero.

    """
    if isinstance(molgrid, AtomGrid):
        # Convert AtomGrid to MolGrid, may increase computational time.
        molgrid = MolGrid(
            atnums=np.array([1.0]),
            atgrids=[molgrid],
            aim_weights=np.array([1.0] * molgrid.size),
            store=True,
        )
    if molgrid.atgrids is None:
        raise ValueError("Molecular grid (MolGrid) attribute `store` should be set to True.")

    # Multiply f by the nuclear weight function w_n(r) for each atom grid segment.
    func_vals_atom = func_vals * molgrid.aim_weights
    # Go through each atomic grid and construct interpolation of f*w_n.
    interpolate_funcs = []
    for i in range(len(molgrid.atcoords)):
        start_index = molgrid.indices[i]
        final_index = molgrid.indices[i + 1]
        atom_grid = molgrid[i]

        def interpolate_laplacian_atom_grid(
            points: np.ndarray,
            atom_grid: AtomGrid,
            cutoff: float = 1e-6,
            start_index=start_index,
            final_index=final_index,
        ):
            # compute spline for the radial components for f
            radial_comps_f = atom_grid.radial_component_splines(
                func_vals_atom[start_index:final_index]
            )

            r_pts, theta, phi = atom_grid.convert_cartesian_to_spherical(points).T

            if np.any(r_pts < cutoff):
                r_pts[r_pts < cutoff] = cutoff

            # Evaluate the radial components, its derivative and second deriv for function f
            r_values_f = np.array([spline(r_pts) for spline in radial_comps_f])
            dr_values = np.array([spline(r_pts, 1) for spline in radial_comps_f])
            d2r_values = np.array([spline(r_pts, 2) for spline in radial_comps_f])

            r_sph_harm = generate_real_spherical_harmonics(atom_grid.l_max // 2, theta, phi)

            # Take the sum of each component with the spherical harmonics
            # l means the angular (l,m) variables and n represents points.
            # Note this was computed on seperate due to the difficulty in handing 0/inf, 0/0, 0*inf.
            first_component = np.einsum("ln, ln -> n", d2r_values, r_sph_harm)
            second_component = np.einsum("ln, ln -> n", dr_values, r_sph_harm)

            # Compute 2 \sum \sum dr_values Y_l^m / r,
            with np.errstate(divide="ignore", invalid="ignore"):
                second_component *= 2.0 / r_pts

            # Compute l(l+1) \sum \sum r_values Y_l^m / r^2
            degrees = np.hstack(
                [[x * (x + 1)] * (2 * x + 1) for x in np.arange(0, atom_grid.l_max // 2 + 1)]
            )
            third_component = np.einsum("ln,l,ln -> n", r_values_f, degrees, r_sph_harm)
            with np.errstate(divide="ignore", invalid="ignore"):
                third_component /= r_pts**2.0

            return first_component + second_component - third_component

        interpolate_funcs.append(
            lambda points, cut_off, atom_grid=atom_grid: interpolate_laplacian_atom_grid(
                points, atom_grid, cut_off
            )
        )

    def sum_of_interpolation_funcs(points, cut_off: float = 1e-6):
        output = interpolate_funcs[0](points, cut_off)
        for interpolate in interpolate_funcs[1:]:
            output += interpolate(points, cut_off)
        return output

    return sum_of_interpolation_funcs
