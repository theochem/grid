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
"""Atomic Grid Module."""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import scipy.constants
from importlib_resources import files
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from grid.angular import AngularGrid
from grid.basegrid import Grid, OneDGrid
from grid.onedgrid import UniformInteger
from grid.rtransform import PowerRTransform
from grid.utils import (
    _DEFAULT_POWER_RTRANSFORM_PARAMS,
    convert_cart_to_sph,
    convert_derivative_from_spherical_to_cartesian,
    generate_derivative_real_spherical_harmonics,
    generate_real_spherical_harmonics,
)


class AtomGrid(Grid):
    r"""
    Atomic grid construction class for integrating three-dimensional functions.

    Atomic grid is composed of a radial grid :math:`\{(r_i, w_i)\}_{i=1}^{N}` meant to
    integrate the radius coordinate in spherical coordinates. Further, each radial point
    is associated with an Angular (Lebedev or Symmetric spherical t-design) grid
    :math:`\{(\theta^i_j, \phi^i_j, w_j^i)\}_{j=1}^{M_i}` that integrates over a sphere
    (angles in spherical coordinates).  The atomic grid points can also be centered at a given
    location.

    """

    def __init__(
        self,
        rgrid: OneDGrid,
        degrees: np.ndarray | list | None = [50],
        *,
        sizes: np.ndarray | list = None,
        center: np.ndarray = None,
        rotate: int = 0,
        method: str = "lebedev",
    ):
        """
        Construct atomic grid for given arguments.

        Parameters
        ----------
        rgrid : OneDGrid
            The (one-dimensional) radial grid representing the radius of spherical grids.
        degrees : ndarray(N, dtype=int) or list, optional
            Sequence of angular grid degrees used for constructing spherical grids at each
            radial grid point.
            If only one degree is given, the specified degree is used for all spherical grids.
            If the given degree is not supported, the next largest degree is used.
        sizes : ndarray(N, dtype=int) or list, keyword-only
            Sequence of angular grid sizes used for constructing spherical grids at each radial
            grid point. If only one size is given, the specified size is used for all spherical
            grids. If the given size is not supported, the next largest size is used. If both
            `degrees` and `sizes` are given, `size` are used for constructing the angular grid.
        center : ndarray(3,), optional, keyword-only
            Cartesian coordinates of the grid center. If `None`, the origin is used.
        rotate : int, optional, keyword-only
            Integer used as a seed for generating random rotation matrices to rotate the angular
            spherical grids at each radial grid point. If the integer is zero, then no rotate
            is used.
        method: str, optional, keyword-only
            Method for constructing the angular grid. Options are "lebedev" (for Lebedev-Laikov)
            and "spherical" (for symmetric spherical t-design).

        """
        # check stage, if center is None, set to (0., 0., 0.)
        center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)
        self._input_type_check(rgrid, center)
        # assign & check stage
        self._center = center
        self._rgrid = rgrid
        # check rotate
        if not isinstance(rotate, (int, np.integer)):
            raise TypeError(f"Rotate needs to be an integer, got {type(rotate)}")
        if (rotate is not False) and (not 0 <= rotate < 2**32 - len(rgrid.points)):
            raise ValueError(
                f"rotate need to be an integer [0, 2^32 - len(rgrid)]\n"
                f"rotate is not within [0, 2^32 - len(rgrid)], got {rotate}"
            )
        self._rot = rotate

        # allow only one of degree or size to be given
        if sizes is not None:
            warnings.warn(
                "Sizes are used for making the angular grids, degrees are ignored!",
                RuntimeWarning,
                stacklevel=2,
            )
            degree = None
            # check sizes
            if not isinstance(sizes, (np.ndarray, list)):
                raise TypeError(f"sizes is not type: np.array or list, got {type(sizes)}")
            degrees = AngularGrid.convert_angular_sizes_to_degrees(sizes, method=method)
        # check degrees
        if not isinstance(degrees, (np.ndarray, list)):
            raise TypeError(f"degrees is not type: np.array or list, got {type(degrees)}")
        if len(degrees) == 1:
            degrees = np.ones(rgrid.size, dtype=int) * degrees
        (
            self._points,
            self._weights,
            self._indices,
            self._degs,
        ) = self._generate_atomic_grid(
            self._rgrid, degrees, rotate=self._rot, method=method.lower()
        )
        self._size = self._weights.size
        self._basis = None
        self._method = method.lower()

    @classmethod
    def from_preset(
        cls,
        atnum: int,
        preset: str,
        rgrid: OneDGrid = None,
        center: np.ndarray = None,
        rotate: int = 0,
        method: str = "lebedev",
    ):
        """Construct an atomic grid with pre-defined angular grids for a given atomic number.

        Parameters
        ----------
        atnum : int
            The atomic number.
        preset : str, optional
            The name of pre-defined grid specifying the radial sectors and their corresponding
            number of angular grid points. Supported options include our custom presets:
            'coarse', 'medium', 'fine', 'veryfine', 'ultrafine', and 'insane', as well as,
            other standard pre-defined grids including:
            'sg_0', 'sg_1', 'sg_2', and 'sg_3', and the Ochsenfeld grids:
            'g1', 'g2', 'g3', 'g4', 'g5', 'g6', and 'g7', with higher number indicating
            greater accuracy but denser grid. See `Notes` for more information.
        rgrid : OneDGrid, optional
            The 1D radial grid representing the radius of spherical grids.
            If None, a default radial grid (PowerRTransform of UniformInteger grid) for the give
            `atnum` is constructed.
        center : ndarray(3,), optional
            Cartesian coordinates of the grid center. If `None`, the origin is used.
        rotate : int, optional
            Integer used as a seed for generating random rotation matrices to rotate the angular
            spherical grids at each radial grid point. If 0, then no rotate is made.
        method: str, optional
            Method for constructing the angular grid. Options are "lebedev" (for Lebedev-Laikov)
            and "spherical" (for symmetric spherical t-design).

        Notes
        -----
        - The standard and Ochsenfeld presets were not designed with symmetric spherical t-design
          in mind.
        - The "standard grids" [1]_ "SG-0" and "SG-1" are designed for large molecules with
          LDA (GGA) functionals, whereas "SG-2" and "SG-3" are designed for Meta-GGA functionals
          and B95/Minnesota functionals, respectively.
        - The Ochsenfeld pruned grids [2]_ are obtained based on the paper.

        References
        ----------
        .. [1] Y. Shao, et al. Advances in molecular quantum chemistry contained in the Q-Chem 4
               program package. Mol. Phys. 113, 184-215 (2015)
        .. [2] Laqua, H., Kussmann, J., & Ochsenfeld, C. (2018). An improved molecular partitioning
               scheme for numerical quadratures in density functional theory. The Journal of
               Chemical Physics, 149(20).

        """
        if rgrid is None:
            if atnum in _DEFAULT_POWER_RTRANSFORM_PARAMS:
                # load the default radial grid parameters for the given atomic number
                rmin, rmax, npt = _DEFAULT_POWER_RTRANSFORM_PARAMS[int(atnum)]
                # convert angstrom to atomic units
                ang2bohr = scipy.constants.angstrom / scipy.constants.value("atomic unit of length")
                rmin, rmax = rmin * ang2bohr, rmax * ang2bohr
                # construct a radial grid
                onedgrid = UniformInteger(npt)
                rgrid = PowerRTransform(rmin, rmax).transform_1d_grid(onedgrid)
            else:
                raise ValueError(
                    f"Default radial grid parameters is not included for the atomic number {atnum}."
                )
        center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)
        cls._input_type_check(rgrid, center)
        # load radial points and
        data = np.load(files("grid.data.prune_grid").joinpath(f"prune_grid_{preset}.npz"))
        # load predefined_radial sectors and num_of_points in each sectors
        rad = data[f"{atnum}_rad"]
        npt = data[f"{atnum}_npt"]

        if preset in ["sg_0", "sg_2", "sg_3", "g1", "g2", "g3", "g4", "g5", "g6", "g7"]:
            sector_sizes = [npt[idx] for idx in range(len(rad)) for _ in range(rad[idx])]
            return cls(rgrid, None, sizes=sector_sizes, center=center, rotate=rotate, method=method)
        elif preset == "sg_1" and atnum > 19:
            sector_sizes = [npt[idx] for idx in range(len(rad)) for _ in range(rad[idx])]
            return cls(rgrid, None, sizes=sector_sizes, center=center, rotate=rotate, method=method)
        else:
            degs = AngularGrid.convert_angular_sizes_to_degrees(npt, method=method)
            rad_degs = AtomGrid._find_degrees_for_radial_points(rgrid.points, rad, degs)
            return cls(rgrid, degrees=rad_degs, center=center, rotate=rotate, method=method)

    @classmethod
    def from_pruned(
        cls,
        rgrid: OneDGrid,
        radius: float,
        r_sectors: list | np.ndarray,
        d_sectors: list | np.ndarray | None = None,
        *,
        s_sectors: Union[list, np.ndarray] = None,
        center: np.ndarray = None,
        rotate: int = 0,
        method: str = "lebedev",
    ):
        r"""
        Initialize AtomGrid class that splits radial sections into sectors which specified degrees.

        Given a sequence of radial sectors :math:`\{a_i\}_{i=1}^Q`, a radius number :math:`R`
        and angular degree sectors :math:`\{L_i \}_{i=1}^{Q+1}`.  This assigned the degrees
        to the following radial points:

        .. math::
            \begin{align*}
                &L_1 \text{ when } r < R a_1 \\
                &L_2 \text{ when } R a_1 \leq r < R a_2
                \vdots \\
                &L_{Q+1} \text{ when } R a_{Q} < r.
            \end{align*}

        Parameters
        ----------
        rgrid : OneDGrid
            The (one-dimensional) radial grid representing the radius of spherical grids.
        radius : float
            The atomic radius to be multiplied with `r_sectors` in atomic units
            (to make the radial sectors atom specific).
        r_sectors : list or ndarray(S,)
            Sequence of boundary radius (in atomic units) specifying sectors of the pruned radial
            grid. The first sector is ``(0, radius*r_sectors[0])``, then ``(radius*r_sectors[0],
            radius*r_sectors[1])``, and so on.
        d_sectors : list or ndarray(S + 1, dtype=int) or None
            Sequence of angular degrees for each radial sector of `r_sectors` in the pruned grid.
            If None, then `s_sectors` should be given.
        s_sectors : list or ndarray(S + 1, dtype=int) or None, optional, keyword-only
            Sequence of angular sizes for each radial sector of `r_sectors` in the pruned grid.
            If both `d_sectors` and `s_sectors` are given, `s_sectors` is used.
        center : ndarray(3,), optional
            Three-dimensional Cartesian coordinates of the grid center in atomic units.
            If None, the origin is used.
        rotate : int, optional
            Integer used as a seed for generating random rotation matrices to rotate the angular
            spherical grids at each radial grid point. If the integer is zero, then no rotate
            is used.
        method: str, optional
            Method for constructing the angular grid. Options are "lebedev" (for Lebedev-Laikov)
            and "spherical" (for symmetric spherical t-design).

        Returns
        -------
        AtomGrid
            Generated AtomGrid instance for this special init method.

        """
        if s_sectors is not None:
            warnings.warn(
                "s_sectors are used for making the atomic grid, d_sectors is ignored!",
                RuntimeWarning,
                stacklevel=2,
            )
            d_sectors = AngularGrid.convert_angular_sizes_to_degrees(s_sectors, method)
        # else:
        #     raise ValueError("Arguments d_sectors and s_sectors cannot be both None.")
        center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)
        cls._input_type_check(rgrid, center)
        degrees = cls._generate_degree_from_radius(rgrid, radius, r_sectors, d_sectors, method)
        return cls(rgrid, degrees=degrees, center=center, rotate=rotate, method=method)

    @property
    def rgrid(self):
        """OneDGrid: The radial grid representing the radius of spherical grids."""
        return self._rgrid

    @property
    def rotate(self):
        """int: Integer representing the seed for rotating the angular grid."""
        return self._rot

    @property
    def degrees(self):
        r"""ndarray(L,): Return the degree of each angular grid at each radial point."""
        return self._degs

    @property
    def points(self):
        """ndarray(N, 3): 3D Cartesian coordinates of the grid points in atomic units."""
        return self._points + self._center

    @property
    def indices(self):
        """ndarray(M+1,): Indices saved for each spherical shell."""
        # M is the number of points on radial grid.
        return self._indices

    @property
    def center(self):
        """ndarray(3,): Cartesian coordinates of the grid center."""
        return self._center

    @property
    def n_shells(self):
        """int: Number of shells in radial points."""
        return len(self._degs)

    @property
    def l_max(self):
        """int: Largest angular degree L value in angular grids."""
        return np.max(self._degs)

    @property
    def method(self):
        r"""str: Method used for constructing an angular grid."""
        return self._method

    @property
    def basis(self):
        r"""ndarray(N, 3): Generate spherical harmonic basis evaluated on atomic grid points."""
        # Used for mostly interpolation
        return self._basis

    def save(self, filename):
        r"""
        Save atomic grid attributes as a npz file.

        Parameters
        ----------
        filename: str
           The path/name of the .npz file.

        """
        dict_save = {
            "points": self.points,
            "weights": self.weights,
            "center": self.center,
            "degrees": self.degrees,
            "indices": self.indices,
            "rgrid_pts": self.rgrid.points,
            "rgrid_weights": self.rgrid.weights,
            "method": self.method,
        }
        np.savez(filename, **dict_save)

    def get_shell_grid(self, index: int, r_sq: bool = True):
        """Get the spherical integral grid at radial point from specified index.

        The spherical integration grid has points scaled with the ith radial point
        and weights multiplied by the ith weight of the radial grid.

        Note that when :math:`r=0` then the Cartesian points are all zeros.

        Parameters
        ----------
        index : int
            Index of radial points.
        r_sq : bool, default True
            If true, multiplies the angular grid weights with r**2.

        Returns
        -------
        AngularGrid
            Instance of AngularGrid for the given radial index position and weights.

        """
        if not (0 <= index < len(self.degrees)):
            raise ValueError(
                f"Index {index} should be between 0 and less than number of "
                f"radial points {len(self.degrees)}."
            )
        degree = self.degrees[index]
        sphere_grid = AngularGrid(degree=degree, method=self.method)

        # modify points and weights of angular grid to include radial contribution
        pts = sphere_grid.points.copy()
        wts = sphere_grid.weights.copy()
        # Rotate the points
        if self.rotate != 0:
            rot_mt = R.random(random_state=self.rotate + index).as_matrix()
            pts = pts.dot(rot_mt)

        pts = pts * self.rgrid[index].points
        wts = wts * self.rgrid[index].weights
        if r_sq is True:
            wts = wts * self.rgrid[index].points ** 2

        # update points and weights of angular grid
        sphere_grid.points = pts
        sphere_grid.weights = wts
        return sphere_grid

    def convert_cartesian_to_spherical(self, points: np.ndarray = None, center: np.ndarray = None):
        r"""Convert a set of points from Cartesian to spherical coordinates.

        .. math::
            \begin{align}
                r &= \sqrt{x^2 + y^2 + z^2}\\
                \theta &= arc\tan (\frac{y}{x})\\
                \phi &= arc\cos(\frac{z}{r})
            \end{align}

        with the canonical choice :math:`r=0`, then :math:`\theta,\phi = 0`.
        If the `points` attribute is not specified, then atomic grid points are used
        and the canonical choice when :math:`r=0`, is the points
        :math:`(r=0, \theta_j, \phi_j)` where :math:`(\theta_j, \phi_j)` come
        from the Angular grid with the degree at :math:`r=0`.

        Parameters
        ----------
        points : ndarray(N, 3), optional
            Three-dimensions Cartesian coordinates of points in atomic units.
            If None, the atomic grid `points` will be used.
        center : ndarray(3,), optional
            Three-dimensional Cartesian coordinates of the center of coordinate system
            in atomic units. If None, the atomic grid `center` will be used.

        Returns
        -------
        ndarray(N, 3)
            Spherical coordinates of points respect to the center denoted as
            (radius :math:`r`, azimuthal :math:`\theta`, polar :math:`\phi`).

        """
        is_atomic = False
        if points is None:
            points = self.points
            is_atomic = True
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        center = self.center if center is None else np.asarray(center)
        spherical_points = convert_cart_to_sph(points, center)
        # If atomic grid points are being converted, then choose canonical angles (when r=0)
        # to be from the degree specified of that point.  The reasoning is to insure that
        # the integration of spherical harmonic when l=l, m=0, is zero even when r=0.
        if is_atomic:
            r_index = np.where(self.rgrid.points == 0.0)[0]
            for i in r_index:
                # build angular grid for the degree at r=0
                agrid = AngularGrid(degree=self._degs[i], method=self.method)
                i_index = self._indices[i]
                f_index = self._indices[i + 1]
                spherical_points[i_index:f_index, 1:] = convert_cart_to_sph(agrid.points)[:, 1:]
        return spherical_points

    def integrate_angular_coordinates(self, func_vals: np.ndarray):
        r"""Integrate the angular coordinates of a sequence of functions.

        Given a series of functions :math:`f_k \in L^2(\mathbb{R}^3)`, this returns the values

        .. math::
            f_k(r_i) = \int \int f(r_i, \theta, \phi) sin(\phi) d\theta d\phi

        on each radial point :math:`r_i` in the atomic grid.

        Parameters
        ----------
        func_vals : ndarray(..., N)
            The function values evaluated on all :math:`N` points on the atomic grid
            for many types of functions.  This can also be one-dimensional.

        Returns
        -------
        ndarray(..., M) :
            The function :math:`f_{...}(r_i)` on each :math:`M` radial points.

        """
        # Integrate f(r, \theta, \phi) sin(\phi) d\theta d\phi by multiplying against its weights
        prod_value = func_vals * self.weights  # Multiply weights to the last axis.
        # [..., indices] means only take the last axis, this is due func_vals being
        #  multi-dimensional, take a sum over the last axis only and swap axes so that it
        #  has shape (..., M) where ... is the number of functions and M is the number of
        #  radial points.
        radial_coefficients = np.array(
            [
                np.sum(prod_value[..., self.indices[i] : self.indices[i + 1]], axis=-1)
                for i in range(self.n_shells)
            ]
        )
        radial_coefficients = np.moveaxis(radial_coefficients, 0, -1)  # swap points axes to last

        # Remove the radial weights and r^2 values that are in self.weights
        with np.errstate(divide="ignore", invalid="ignore"):
            radial_coefficients /= self.rgrid.points**2 * self.rgrid.weights
        # For radius smaller than 1.0e-8, due to division by zero by r^2, we regenerate
        # the angular grid and calculate the integral at those points.
        r_index = np.where(self.rgrid.points < 1e-8)[0]
        for i in r_index:  # if r_index = [], then for loop doesn't occur.
            # build angular grid for i-th shell
            agrid = AngularGrid(degree=self._degs[i], method=self.method)
            values = func_vals[..., self.indices[i] : self.indices[i + 1]] * agrid.weights
            radial_coefficients[..., i] = np.sum(values, axis=-1)
        return radial_coefficients

    def spherical_average(self, func_vals: np.ndarray):
        r"""
        Return spline of the spherical average of a function.

        This function takes a function :math:`f` evaluated on the atomic grid points and returns
        the spherical average of it defined as:

        .. math::
            f_{avg}(r) := \frac{\int \int f(r, \theta, \phi) \sin(\phi) d\theta d\phi}{4 \pi}.

        The definition is chosen such that :math:`\int f_{avg}(r) 4\pi r^2 dr`
        matches the full integral :math:`\int \int \int f(x,y,z)dxdydz`.

        Parameters
        ----------
        func_vals : ndarray(N,)
            The function values evaluated on all :math:`N` points on the atomic grid.

        Returns
        -------
        CubicSpline:
            Cubic spline with input r in the positive real axis and output :math:`f_{avg}(r)`.

        """
        # Integrate f(r, theta, phi) sin(phi) d\theta d\phi
        f_radial = self.integrate_angular_coordinates(func_vals)
        f_radial /= 4.0 * np.pi
        # Construct spline of f_{avg}(r)
        spline = CubicSpline(x=self.rgrid.points, y=f_radial)
        return spline

    def radial_component_splines(self, func_vals: np.ndarray):
        r"""
        Return spline to interpolate radial components wrt to expansion in real spherical harmonics.

        For each pt :math:`r_i` of the atomic grid with associated angular degree :math:`l_i`,
        the function :math:`f(r_i, \theta, \phi)` is projected onto the spherical
        harmonic expansion:

        .. math::
            f(r_i, \theta, \phi) \approx \sum_{l=0}^{l_i} \sum_{m=-l}^l \rho^{lm}(r_i)
            Y^m_l(\theta, \phi)

        where :math:`Y^m_l` is the real Spherical harmonic of degree :math:`l` and order :math:`m`.
        The radial components :math:`\rho^{lm}(r_i)` are calculated via integration on
        the :math:`i`th Lebedev/angular grid of the atomic grid:

        .. math::
            \rho^{lm}(r_i) = \int \int f(r_i, \theta, \phi) Y^m_l(\theta, \phi) \sin(\phi)
             d\theta d\phi,

        and then interpolated using a cubic spline over all radial points of the atomic grid.

        Parameters
        ----------
        func_vals : ndarray(N,)
            The function values evaluated on all :math:`N` points on the atomic grid.

        Returns
        -------
        list[scipy.PPoly]
            A list of size :math:`(l_{max}/2 + 1)^2` of  CubicSpline object for interpolating the
            coefficients :math:`\rho^{lm}(r)`. The input of spline is array
            of :math:`N` points on :math:`[0, \infty)` and the output is :math:`\{\rho^{lm}(r_i)\}`.
            The list starts with degrees :math:`l=0,\cdots l_{max}`, and the for each degree,
            the zeroth order spline is first, followed by positive orders then negative.

        """
        if func_vals.size != self.size:
            raise ValueError(
                "The size of values does not match with the size of grid\n"
                f"The size of value array: {func_vals.size}\n"
                f"The size of grid: {self.size}"
            )
        if self._basis is None:
            theta, phi = self.convert_cartesian_to_spherical().T[1:]
            # Going up to `self.l_max // 2` is explained below.
            self._basis = generate_real_spherical_harmonics(self.l_max // 2, theta, phi)
        # Multiply spherical harmonic basis with the function values to project.
        values = np.einsum("ln,n->ln", self._basis, func_vals)
        radial_components = self.integrate_angular_coordinates(values)
        # each shell can only integrate spherical harmonics up to the shell_degree,
        # so if shell_degree < l_max, the f_{lm} should be set to zero for l > shell_degree // 2.
        # Instead, one could set truncate the basis of a given shell.
        for i in range(self.n_shells):
            if self.degrees[i] != self.l_max:
                # if self.degrees[i] > self.l_max // 2:
                num_nonzero_sph = (self.degrees[i] // 2 + 1) ** 2
                radial_components[num_nonzero_sph:, i] = 0.0

        # Return a spline for each spherical harmonic with maximum degree `self.l_max // 2`.
        return [CubicSpline(x=self.rgrid.points, y=sph_val) for sph_val in radial_components]

    def interpolate(self, func_vals: np.ndarray):
        r"""
        Return function that interpolates (and its derivatives) from function values.

        Any real-valued function :math:`f(r, \theta, \phi)` can be decomposed as

        .. math::
            f(r, \theta, \phi) = \sum_l \sum_{m=-l}^l \sum_i \rho_{ilm}(r) Y_{lm}(\theta, \phi)

        A cubic spline is used to interpolate the radial functions :math:`\sum_i \rho_{ilm}(r)`.
        This is then multiplied by the corresponding spherical harmonics at all
        :math:`(\theta_j, \phi_j)` angles and summed to obtain the equation above.

        Parameters
        ----------
        func_vals : ndarray(N,)
            The function values evaluated on all :math:`N` points on the atomic grid.

        Returns
        -------
        Callable[[ndarray(M, 3), int] -> ndarray(M)]:
            Callable function that interpolates the function and its derivative provided.
            The function takes the following attributes:

                points : ndarray(N, 3)
                    Cartesian coordinates of :math:`N` points to evaluate the splines on.
                deriv : int, optional
                    If deriv is zero, then only returns function values. If it is one, then
                    returns the first derivative of the interpolated function with respect to either
                    Cartesian or spherical coordinates. Only higher-order derivatives
                    (`deriv`=2,3) are supported for the derivatives wrt to radial components.
                deriv_spherical : bool
                    If True, then returns the derivatives with respect to spherical coordinates
                    :math:`(r, \theta, \phi)`. Default False.
                only_radial_deriv : bool
                    If true, then the derivative wrt to radius :math:`r` is returned.

            This function returns the following.

                ndarray(M,...):
                    The interpolated function values or its derivatives with respect to Cartesian
                    :math:`(x,y,z)` or if `deriv_spherical` then :math:`(r, \theta, \phi)` or
                    if `only_radial_derivs` then derivative wrt to :math:`r` is only returned.

        """
        # compute splines for given value_array on grid points
        splines = self.radial_component_splines(func_vals)

        def interpolate_low(points, deriv=0, deriv_spherical=False, only_radial_deriv=False):
            r"""Construct a spline like callable for interpolation.

            Parameters
            ----------
            points : ndarray(N, 3)
                Cartesian coordinates of :math:`N` points to evaluate the splines on.
            deriv : int, optional
                If deriv is zero, then only returns function values. If it is one, then returns
                the first derivative of the interpolated function with respect to either Cartesian
                or spherical coordinates. Only higher-order derivatives (`deriv` =2,3) are supported
                for the derivatives wrt to radial components. `deriv=3` only returns a constant.
            deriv_spherical : bool
                If True, then returns the derivatives with respect to spherical coordinates
                :math:`(r, \theta, \phi)`. Default False.
            only_radial_deriv : bool
                If true, then the derivative wrt to radius :math:`r` is returned.

            Returns
            -------
            ndarray(M,...) :
                The interpolated function values or its derivatives with respect to Cartesian
                :math:`(x,y,z)` or if `deriv_spherical` then :math:`(r, \theta, \phi)` or
                if `only_radial_derivs` then derivative wrt to :math:`r` is only returned.

            """
            if deriv_spherical and only_radial_deriv:
                warnings.warn(
                    "Since `only_radial_derivs` is true, then only the derivative wrt to"
                    "radius is returned and `deriv_spherical` value is ignored.",
                    stacklevel=2,
                )
            r_pts, theta, phi = self.convert_cartesian_to_spherical(points).T
            r_values = np.array([spline(r_pts, deriv) for spline in splines])
            r_sph_harm = generate_real_spherical_harmonics(self.l_max // 2, theta, phi)

            # If theta, phi derivaitves are wanted and the derivative is first-order.
            if not only_radial_deriv and deriv == 1:
                # Calculate derivative of f with respect to radial, theta, phi
                # Get derivative of spherical harmonics first.
                radial_components = np.array([spline(r_pts, 0) for spline in splines])
                deriv_sph_harm = generate_derivative_real_spherical_harmonics(
                    self.l_max // 2, theta, phi
                )
                deriv_r = np.einsum("ij, ij -> j", r_values, r_sph_harm)
                deriv_theta = np.einsum("ij,ij->j", radial_components, deriv_sph_harm[0, :, :])
                deriv_phi = np.einsum("ij,ij->j", radial_components, deriv_sph_harm[1, :, :])

                # If deriv spherical is wanted, then return that.
                if deriv_spherical:
                    return np.hstack((deriv_r, deriv_theta, deriv_phi))

                # Convert derivative from spherical to Cartesian:
                derivs = np.zeros((len(r_pts), 3))
                # TODO: this could be vectorized properly with memory management.
                for i_pt in range(0, len(r_pts)):
                    radial_i, theta_i, phi_i = r_pts[i_pt], theta[i_pt], phi[i_pt]
                    derivs[i_pt] = convert_derivative_from_spherical_to_cartesian(
                        deriv_r[i_pt],
                        deriv_theta[i_pt],
                        deriv_phi[i_pt],
                        radial_i,
                        theta_i,
                        phi_i,
                    )
                return derivs
            elif not only_radial_deriv and deriv != 0:
                raise ValueError(
                    f"Higher order derivatives are only supported for derivatives"
                    f"with respect to the radius. Deriv is {deriv}."
                )

            return np.einsum("ij, ij -> j", r_values, r_sph_harm)

        return interpolate_low

    @staticmethod
    def _input_type_check(rgrid: OneDGrid, center: np.ndarray):
        """Check input type.

        Parameters
        ----------
        rgrid : OneDGrid
            The (one-dimensional) radial grid representing the radius of spherical grids.
        center : ndarray(3,), optional
            Center of the spherical coordinates
            atomic center will be used if `center` is not given

        """
        if not isinstance(rgrid, OneDGrid):
            raise TypeError(f"Argument rgrid is not an instance of OneDGrid, got {type(rgrid)}.")
        if rgrid.domain is not None and rgrid.domain[0] < 0:
            raise TypeError(f"Argument rgrid should have a positive domain, got {rgrid.domain}")
        elif np.min(rgrid.points) < 0.0:
            raise TypeError(f"Smallest rgrid.points is negative, got {np.min(rgrid.points)}")
        if center.shape != (3,):
            raise ValueError(f"Center should be of shape (3,), got {center.shape}.")

    @staticmethod
    def _generate_degree_from_radius(
        rgrid: OneDGrid,
        radius: float,
        r_sectors: Union[list, np.ndarray],
        d_sectors: Union[list, np.ndarray],
        method: str = "lebedev",
    ):
        """
        Get all degrees for every radial point inside the radial grid based on the sectors.

        Parameters
        ----------
        rgrid : OneDGrid
            Radial grid with :math:`N` points.
        radius : float
            Radius of interested atom in atomic units.
        r_sectors : list or ndarray(S,)
            Sequence of boundary radius (in atomic units) specifying sectors of the pruned radial
            grid. The first sector is ``(0, radius*r_sectors[0])``, then ``(radius*r_sectors[0],
            radius*r_sectors[1])``, and so on.
        d_sectors : list or ndarray(S + 1, dtype=int)
            Sequence of angular degrees for each radial sector of `r_sectors` in the pruned grid.
        method: str, optional
            Method for constructing the angular grid. Options are "lebedev" (for Lebedev-Laikov)
            and "spherical" (for symmetric spherical t-design).

        Returns
        -------
        ndarray(N,)
            Array of degree values :math:`l` for each radial point.

        """
        r_sectors = np.array(r_sectors) * radius
        d_sectors = np.array(d_sectors)

        # check that the number of degree sectors matches the number of radial sectors
        if len(d_sectors) - len(r_sectors) != 1:
            raise ValueError(
                "d_sectors should have only one more element than r_sectors, "
                f"got len(d_sectors)={len(d_sectors)} and len(r_sectors)={len(d_sectors)}."
            )
        # match given degrees to the supported (i.e., pre-computed) angular degrees
        matched_deg = np.array(
            [
                AngularGrid._get_degree_and_size(degree=d, size=None, method=method)[0]
                for d in d_sectors
            ]
        )
        rad_degs = AtomGrid._find_degrees_for_radial_points(rgrid.points, r_sectors, matched_deg)
        return rad_degs

    @staticmethod
    def _find_degrees_for_radial_points(
        radial_points: np.ndarray, r_sectors: np.ndarray, d_sectors: np.ndarray
    ):
        r"""
        Find degrees for all radial points given radial and degree sectors.

        Parameters
        ----------
        radial_points : ndarray(N,)
            Radial grid points in angstrom.
        r_sectors : list or ndarray(S,)
            Sequence of boundary radius (in atomic units) specifying sectors of the pruned radial
            grid. The first sector is ``(0, radius*r_sectors[0])``, then ``(radius*r_sectors[0],
            radius*r_sectors[1])``, and so on.
        d_sectors : list or ndarray(S + 1, dtype=int)
            Sequence of angular degrees for each radial sector of `r_sectors` in the pruned grid.

        Returns
        -------
        ndarray(N,)
            A list of angular degrees :math:`l` for each radial grid point.

        """
        # use broadcast to compare each point with r_sectors then sum over all
        # the True value, which should equal to the position of L.
        position = np.sum(radial_points[:, None] > r_sectors[None, :], axis=1)
        return d_sectors[position]

    @staticmethod
    def _generate_atomic_grid(
        rgrid: OneDGrid,
        degrees: np.ndarray,
        rotate: int = 0,
        method: str = "lebedev",
    ):
        """Generate atomic grid for each radial point with angular degree.

        Parameters
        ----------
        rgrid : OneDGrid
            The (1-dimensional) radial grid representing the radius of spherical grids.
        degrees : ndarray(N,)
            Sequence of angular grid degrees used for constructing spherical grids at each
            radial grid point.
            If the given degree is not supported, the next largest degree is used.
        rotate : int, optional
            Integer used as a seed for generating random rotation matrices to rotate the angular
            spherical grids at each radial grid point. If the integer is zero, then no rotate
            is used.
        method: str, optional
            Method for constructing the angular grid. Options are "lebedev" (for Lebedev-Laikov)
            and "spherical" (for symmetric spherical t-design).

        Returns
        -------
        tuple(ndarray(M,), ndarray(M,), ndarray(N + 1,), ndarray(N,)),
            Atomic grid points, atomic grid weights, indices and degrees for each shell.

        """
        # check that the number of degrees matches the number of radial points
        if len(degrees) != rgrid.size:
            raise ValueError("The shape of radial grid does not match given degs.")

        # make lists to store all grid points and weights
        all_points, all_weights = [], []
        # array of integers to store the indices of points for each spherical shell of given radius
        # e.g., all_points[indices[i]:indices[i+1]] are points for the ith shell of radius rgrid[i]
        indices = np.zeros(len(degrees) + 1, dtype=int)

        # The actual degree used to construct the Angular/lebedev/spherical grid.
        actual_degrees = []

        # TODO: proper tests
        for i, deg_i in enumerate(degrees):
            # Generate Angular grid with the correct degree at the ith radial point.
            sphere_grid = AngularGrid(degree=deg_i, method=method)
            # Note that the copy is needed here.
            points, weights = sphere_grid.points.copy(), sphere_grid.weights.copy()
            actual_degrees.append(sphere_grid.degree)

            # check rotate value and randomly rotate angular grid points
            if not isinstance(rotate, int):
                raise ValueError(f"Argument rotate should be an integer, got {rotate}")
            if rotate != 0:
                rot_mt = R.random(random_state=rotate + i).as_matrix()
                points = points @ rot_mt

            # construct atomic grid with each radial point and each spherical shell
            # compute points, weights, and indices
            points = points * rgrid[i].points
            weights = weights * rgrid[i].weights * rgrid[i].points ** 2
            indices[i + 1] = indices[i] + len(points)
            all_points.append(points)
            all_weights.append(weights)

        points = np.vstack(all_points)
        weights = np.hstack(all_weights)
        return points, weights, indices, actual_degrees


def _get_rgrid_size(preset_grid, atnums=None):
    """Get the predefined radial points for available pruned grids

    Parameters
    ----------
    preset_grid : str
        String specifying type of pruned grid to access radial points data.
    atnums : int or list/array of ints
        Atomic numbers for which to retrieve number of radial points.

    """
    if preset_grid not in [
        "sg_0",
        "sg_1",
        "sg_2",
        "sg_3",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
        "g6",
        "g7",
    ]:
        raise ValueError(f"type_pruned {preset_grid} not recognized as a valid pruned grid")
    elif atnums is None:
        raise ValueError(f"At least one atomic number must be specified. Got {atnums}")
    else:
        if isinstance(atnums, int):
            atnums = [atnums]

        radial_pts = []
        data = np.load(files("grid.data.prune_grid").joinpath(f"prune_grid_{preset_grid}.npz"))
        for at_num in atnums:
            if preset_grid == "sg_1":
                rad = data["r_points"]
            else:
                rad = data[f"{at_num}_rad"]
            radial_pts.append(sum(rad))
        return radial_pts
