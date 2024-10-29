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
"""Construct basic grid data structure."""

import numpy as np
from scipy.spatial import cKDTree

from grid.utils import (
    convert_cart_to_sph,
    generate_orders_horton_order,
    solid_harmonics,
)


class Grid:
    """Basic Grid class for grid information storage."""

    def __init__(self, points, weights):
        """Construct Grid instance.

        Parameters
        ----------
        points : np.ndarray(N,) or np.ndarray(N, M)
            An array with positions of the grid points.
        weights : np.ndarray(N,)
            An array of weights associated with each point on the grid.

        """
        if len(points) != len(weights):
            raise ValueError(
                "Number of points and weights does not match. \n"
                f"Number of points: {len(points)}, Number of weights: {len(weights)}."
            )
        if weights.ndim != 1:
            raise ValueError(f"Argument weights should be a 1-D array. weights.ndim={weights.ndim}")
        if points.ndim not in [1, 2]:
            raise ValueError(
                f"Argument points should be a 1D or 2D array. points.ndim={points.ndim}"
            )
        self._points = points
        self._weights = weights
        self._kdtree = None

    @property
    def points(self):
        """np.ndarray(N,) or np.ndarray(N, M): Positions of the grid points."""
        return self._points

    @points.setter
    def points(self, value):
        """Set the points of the grid."""
        if value.shape != self._points.shape:
            raise ValueError(
                "The shape of the new points should match the shape of the old points. \n"
                f"New shape: {value.shape}, Old shape: {self._points.shape}."
            )
        self._points = value

    @property
    def weights(self):
        """np.ndarray(N,): the weights of each grid point."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Set the weights of the grid."""
        if value.shape != self._weights.shape:
            raise ValueError(
                "The shape of the new weights should match the shape of the old weights. \n"
                f"New shape: {value.shape}, Old shape: {self._weights.shape}."
            )
        self._weights = value

    @property
    def size(self):
        """int: the total number of points on the grid."""
        return self._weights.size

    def __getitem__(self, index):
        """Dunder method for index grid object and slicing.

        Parameters
        ----------
        index : int or slice
            index of slice object for selecting certain part of grid

        Returns
        -------
        Grid
            Return a new Grid object with selected points
        """
        if isinstance(index, int):
            return self.__class__(np.array([self.points[index]]), np.array([self.weights[index]]))
        else:
            return self.__class__(np.array(self.points[index]), np.array(self.weights[index]))

    def integrate(self, *value_arrays):
        r"""Integrate over the whole grid for given multiple value arrays.

        Product of all value_arrays will be computed element-wise then integrated on the grid
        with its weights:

        .. math::
            \int w(x) \prod_i f_i(x) dx.

        Parameters
        ----------
        *value_arrays : np.ndarray(N, )
            One or multiple value array to integrate.

        Returns
        -------
        float:
            The calculated integral over given integrand or function

        """
        if len(value_arrays) < 1:
            raise ValueError("No array is given to integrate.")
        for i, array in enumerate(value_arrays):
            if not isinstance(array, np.ndarray):
                raise TypeError(f"Arg {i} is {type(i)}, Need Numpy Array.")
            if array.shape != (self.size,):
                raise ValueError(f"Arg {i} need to be of shape ({self.size},).")
        # return np.einsum("i, ..., i", a, ..., z)
        return np.einsum(
            "i" + ",i" * len(value_arrays),
            self.weights,
            *(array for array in value_arrays),
        )

    def get_localgrid(self, center, radius):
        """Create a grid contain points within the given radius of center.

        Parameters
        ----------
        center : float or np.array(M,)
            Cartesian coordinates of the center of the local grid.
        radius : float
            Radius of sphere around the center. When equal to np.inf, the
            local grid coincides with the whole grid, which can be useful for
            debugging.

        Returns
        -------
        LocalGrid
            Instance of LocalGrid.

        """
        center = np.asarray(center)
        if center.shape != self._points.shape[1:]:
            raise ValueError(
                "Argument center has the wrong shape \n"
                f"center.shape: {center.shape}, points.shape: {self._points.shape}"
            )
        if radius < 0:
            raise ValueError(f"Negative radius: {radius}")
        if not (np.isfinite(radius) or radius == np.inf):
            raise ValueError(f"Invalid radius: {radius}")
        if radius == np.inf:
            return LocalGrid(self._points, self._weights, center, np.arange(self.size))
        else:
            # When points.ndim == 1, we have to reshape a few things to
            # make the input compatible with cKDTree
            _points = self._points.reshape(self.size, -1)
            _center = np.array([center]) if center.ndim == 0 else center
            if self._kdtree is None:
                self._kdtree = cKDTree(_points)
            indices = np.array(self._kdtree.query_ball_point(_center, radius, p=2.0))
            return LocalGrid(self._points[indices], self._weights[indices], center, indices)

    def moments(
        self,
        orders: int,
        centers: np.ndarray,
        func_vals: np.ndarray,
        type_mom: str = "cartesian",
        return_orders: bool = False,
    ):
        r"""
        Compute the multipole moment integral of a function over centers.

        The Cartesian type moments are:

        .. math::
            m_{n_x, n_y, n_z} = \int (x - X_c)^{n_x} (y - Y_c)^{n_y} (z - Z_c)^{n_z} f(r) dr,

        where :math:`\textbf{R}_c = (X_c, Y_c, Z_c)` is the center of the moment,
        :math:`\f(r)` is the density, and :math:`(n_x, n_y, n_z)` are the Cartesian orders.

        The spherical/pure moments with :math:`(l, m)` parameter are:

        .. math::
            m_{lm} = \int | \textbf{r} - \textbf{R}_c|^l S_l^m(\theta, \phi) f(\textbf{r})
            d\textbf{r},

        where :math:`S_l^m` is a regular, real solid harmonic.

        The radial moments with :math:`n` parameter are:

        .. math::
            m_n = \int | \textbf{r} - \textbf{R}_c|^{n} f(\textbf{r}) d\textbf{r}

        The radial combined with spherical/pure moments :math:`(n, l, m)` are:

        .. math::
            m_{nlm} = \int | \textbf{r} - \textbf{R}_c|^{n+1} S_l^m(\theta, \phi) f(\textbf{r})
            d\textbf{r}

        Parameters
        ----------
        orders : int
            Generates all orders with Horton order depending on the type of the multipole
            moment `type_mom`.
        centers : ndarray(M,  3)
            The centers :math:`\textbf{R}_c` of the moments to compute from.
        func_vals : ndarray(N,)
            The function :math:`f` values evaluated on all :math:`N` points on the integration
            grid.
        type_mom : str
            The type of multipole moments: "cartesian", "pure", "radial" and "pure-radial".
        return_orders : bool
            If true, it will also return a list of size :math:`L` of the orders
            corresponding to each integral/row of the output.

        Returns
        -------
        ndarray(L, M), or (ndarray(L, M), list)
            Computes the moment integral of the function on the `m`th center for all orders.
            If `return_orders` is true, then this also returns a list that describes what
            each row/order is, e.g. for Cartesian, [(0, 0, 0), (1, 0, 0) ,...].

        """
        if func_vals.ndim > 1:
            raise ValueError(f"`func_vals` {func_vals.ndim} should have dimension one.")
        if centers.ndim != 2:
            raise ValueError(f"`centers` {centers.ndim} should have dimension one or two.")
        if self.points.shape[1] != centers.shape[1]:
            raise ValueError(
                f"The dimension of the grid {self.points.shape[1]} should"
                f"match the dimension of the centers {centers.shape[1]}."
            )
        if len(func_vals) != self.points.shape[0]:
            raise ValueError(
                f"The length of function values {len(func_vals)} should match "
                f"the number of points in the grid {self.points.shape[0]}."
            )
        if type_mom == "pure-radial" and orders == 0:
            raise ValueError(
                f"The n/order parameter {orders} for pure-radial multipole moments"
                f"should be positive"
            )

        # Generate all orders, e.g. cartesian it is (m_x, m_y, m_z) in Horton 2 order.
        if isinstance(orders, (int, np.int32, np.int64)):
            orders = range(0, orders + 1) if type_mom != "pure-radial" else range(1, orders + 1)
        else:
            raise TypeError(f"Orders {type(orders)} should be either integer, list or numpy array.")
        dim = self.points.shape[1]
        all_orders = generate_orders_horton_order(orders[0], type_mom, dim)
        for l_ord in orders[1:]:
            all_orders = np.vstack((all_orders, generate_orders_horton_order(l_ord, type_mom, dim)))

        integrals = []
        for center in centers:
            # Calculate centered pts: [(X-c), (Y-c), (Z-c)]
            centered_pts = self.points - center

            if type_mom == "cartesian":
                # Take the powers to get [(X-c)^mx, (Y-c)^my, (Z-c)^mz]
                # This has shape [L, N, 3], where L = (l+1)^2 number of orders, N=number points
                cent_pts_with_order = centered_pts ** all_orders[:, None]
                # Take the product: [(X-c)^_mx (Y-c)^my (Z-c)^mz], has shape (L, N)
                cent_pts_with_order = np.prod(cent_pts_with_order, axis=2)
                # Calculate integral (X-c)^mx (Y-c)^my (Z-c)^mz by the function values and weights
                integral = np.einsum("ln,n,n->l", cent_pts_with_order, func_vals, self.weights)
            elif type_mom in ("radial", "pure", "pure-radial"):
                # Take the norm |r - R_c|
                cent_pts_with_order = np.linalg.norm(centered_pts, axis=1)

                if type_mom in ("pure", "pure-radial"):
                    # Calculate the spherical coordinates of the centered points and calculate
                    # the solid harmonics for all
                    sph_pts = convert_cart_to_sph(centered_pts)
                    solid_harm = solid_harmonics(orders[-1], sph_pts)

                    if type_mom == "pure":
                        # Take the integral |r - R_c|^l S_l^m(theta, phi) f(r, theta, phi) weights
                        integral = np.einsum("ln,n,n->l", solid_harm, func_vals, self.weights)
                    elif type_mom == "pure-radial":
                        # Get the correct indices in solid_harm associated to l_degree and m_orders.
                        n_princ, l_degrees, m_orders = all_orders.T
                        indices = l_degrees**2
                        indices[m_orders > 0] += 2 * m_orders[m_orders > 0] - 1
                        indices[m_orders <= 0] += 2 * np.abs(m_orders[m_orders <= 0])
                        # Take the power to get |r - R_c|^{n}
                        cent_pts_with_order = cent_pts_with_order ** n_princ[:, None]
                        integral = np.einsum(
                            "ln,ln,n,n->l",
                            cent_pts_with_order,
                            solid_harm[indices],
                            func_vals,
                            self.weights,
                        )

                elif type_mom == "radial":
                    cent_pts_with_order = cent_pts_with_order ** np.ravel(all_orders)[:, None]
                    # Take the integral |r - R_c|^l  f(r, theta, phi) weights
                    integral = np.einsum("ln,n,n->l", cent_pts_with_order, func_vals, self.weights)

            integrals.append(integral)
        if return_orders:
            return np.array(integrals).T, all_orders
        return np.array(integrals).T  # output has shape (L, Number of centers)

    def save(self, filename):
        r"""
        Save the points and weights as a npz file.

        Parameters
        ----------
        filename : str
            The path/name of the .npz file.

        """
        save_dict = {"points": self.points, "weights": self.weights}
        np.savez(filename, **save_dict)


class LocalGrid(Grid):
    """Local portion of a grid, containing all points within a sphere."""

    def __init__(self, points, weights, center, indices=None):
        r"""Initialize a local grid.

        Parameters
        ----------
        points : np.ndarray(N,) or np.ndarray(N,M)
            Cartesian coordinates of :math:`N` grid points in 1D or M-D space.
        weights : np.ndarray(N)
            Integration weight of :math:`N` grid points
        center : float or np.ndarray(M,)
            Cartesian coordinates of the center of the local grid in 3D space.
        indices : np.ndarray(N,), optional
            Indices of :math:`N` grid points and weights in the parent grid.

        """
        if indices is not None:
            if len(points) != len(indices):
                raise ValueError(
                    "Number of points and indices does not match. \n"
                    f"number of points: {len(points)}, number of indices: {len(indices)}."
                )
            if indices.ndim != 1:
                raise ValueError(
                    f"Argument indices should be a 1-D array. indices.ndim={indices.ndim}"
                )
        super().__init__(points, weights)
        self._center = center
        self._indices = indices

    @property
    def center(self):
        """np.ndarray(3,): Cartesian coordinates of the center of the local grid."""
        return self._center

    @property
    def indices(self):
        """np.ndarray(N,): Indices of grid points and weights in the parent grid."""
        return self._indices

    def save(self, filename):
        r"""
        Save the points, indices and weights as a npz file.

        Parameters
        ----------
        filename : str
           The path/name of the .npz file.

        """
        save_dict = {
            "points": self.points,
            "weights": self.weights,
            "center": self.center,
            "indices": self.indices,
        }
        np.savez(filename, **save_dict)


class OneDGrid(Grid):
    """One-Dimensional Grid."""

    def __init__(self, points, weights, domain=None):
        r"""Construct grid.

        Parameters
        ----------
        points : np.ndarray(N,)
            A 1-D array of coordinates of :math:`N` points in one-dimension.
        weights : np.ndarray(N,)
            A 1-D array of integration weights of :math:`N` points in one-dimension.
        domain : tuple(float, float), optional
            Lower and upper bounds for which the grid can carry out numerical
            integration. This does not always coincide with the positions of the first
            and last grid point. For example, in case of the Gauss-Chebyshev quadrature
            the domain is [-1,1] but all grid points lie in (-1, 1).

        """
        # check points & weights
        if points.ndim != 1:
            raise ValueError(f"Argument points should be a 1-D array. points.ndim={points.ndim}")

        # check domain
        if domain is not None:
            if len(domain) != 2 or domain[0] > domain[1]:
                raise ValueError(
                    f"domain should be an ascending tuple of length 2. domain={domain}"
                )
            min_p = np.min(points)
            if domain[0] - 1e-7 > min_p:
                raise ValueError(
                    f"point coordinates should not be below domain! {min_p} < {domain[0]}"
                )
            max_p = np.max(points)
            if domain[1] + 1e-7 < max_p:
                raise ValueError(
                    f"point coordinates should not be above domain! {domain[1]} < {max_p}"
                )
        super().__init__(points, weights)
        self._domain = domain

    @property
    def domain(self):
        """(float, float): the range of grid points."""
        return self._domain

    def __getitem__(self, index):
        """Dunder method for index grid object and slicing.

        Parameters
        ----------
        index : int or slice
            index of slice object for selecting certain part of grid

        Returns
        -------
        OneDGrid
            Return a new grid instance with a subset of points.
        """
        if isinstance(index, int):
            return OneDGrid(
                np.array([self.points[index]]),
                np.array([self.weights[index]]),
                self._domain,
            )
        else:
            return OneDGrid(
                np.array(self.points[index]),
                np.array(self.weights[index]),
                self._domain,
            )
