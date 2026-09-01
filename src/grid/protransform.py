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
r"""Promolecular Grid Transformation."""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import root_scalar
from scipy.special import erfc, erfinv, logsumexp

from grid.basegrid import OneDGrid
from grid.cubic import _HyperRectangleGrid

__all__ = ["CubicProTransform"]


class CubicProTransform(_HyperRectangleGrid):
    r"""
    Promolecular Grid Transformation of a Cubic Grid in :math:`[-1, 1]^3`.

    Grid is three-dimensional and modeled as Tensor Product of Three, one dimensional grids.
    Theta space is defined to be :math:`[-1, 1]^3`.
    Real space is defined to be :math:`\mathbb{R}^3.`

    Attributes
    ----------
    shape : (int, int, int)
        The number of points, including both of the end/boundary points, in x, y, and z direction.
    prointegral : float
        The integration value of the promolecular density over Euclidean space.
    promol : namedTuple
        Data about the Promolecular density.
    points : np.ndarray(N, 3)
        Grid points transformed to real space.
    weights : np.ndarray(N,)
        The integration weights, multiplied by `prointegral`.

    Methods
    -------
    integrate(trick=False)
        Integral of a real-valued function over Euclidean space. Can use promolecular trick.
    jacobian()
        Jacobian of the transformation from Real space to Theta space :math:`[-1, 1]^3`.
    hessian()
        Hessian of the transformation from Real space to Theta space :math:`[-1, 1]^3`.
    steepest_ascent_theta()
        Direction of steepest-ascent of a function in theta space from gradient in real space.
    transform():
        Transform Real point to Theta space :math:`[-1, 1]^3`.
    inverse(bracket=(-10, 10))
        Transform Theta point to Real space :math:`\mathbb{R}^3`.
    interpolate(use_log=False, nu=0)
        Interpolate a function (or its logarithm) at a real point. Can interpolate its derivative.

    Examples
    --------
    Define information of the Promolecular Density.
    >> c = np.array([[5.], [10.]])
    >> e = np.array([[2.], [3.]])
    >> coord = np.array([[0., 0., 0.], [2., 2., 2.]])

    Define information of the grid and its weights.
    >> from grid.onedgrid import GaussChebyshev

    >> numb_x = 50
    This is a grid in :math:`[-1, 1]`.
    >> oned = GaussChebyshev(numb_x)
    One dimensional grid is the same in all x, y, z directions.
    >> promol = CubicProTransform([oned, oned, oned], params.c_m, params.e_m, params.gauss_centers)

    To integrate some function f.
    >> def f(pt):
    >>    return np.exp(-0.1 * np.linalg.norm(pt, axis=1)**2.)
    >> func_values = f(promol.points)
    >> print("The integral is %.4f" % promol.integrate(func_values, trick=False)

    References
    ----------
    .. [1] J. I. Rodríguez, D. C. Thompson, P. W. Ayers, and A. M. Koster, "Numerical integration
            of exchange-correlation energies and potentials using transformed sparse grids."

    Notes
    -----
    Let :math:`\rho^o(x, y, z) = \sum_{i=1}^M \sum_{j=1}^D e^{}` be the Promolecular density of a \
    linear combination of Gaussian functions.

    The conditional distribution transformation from :math:`\mathbb{R}^3` to :math:`[-1, 1]^3`
    transfers the (x, y, z) coordinates in :math:`\mathbb{R}^3` to a set of coordinates,
    denoted as :math:`(\theta_x, \theta_y, \theta_z)`, in :math:`[-1,1]^3` that are "bunched"
    up where :math:`\rho^o` is large.

    Precisely it is,

    .. math::
        \begin{eqnarray}
            \theta_x(x) :&=
            -1 + 2 \frac{\int_{-\infty}^x \int \int \rho^o(x, y, z)dx dy dz }
                        {\int \int \int \rho^o(x, y, z)dxdydz}\\
            \theta_y(x, y) :&=
            -1 + 2 \frac{\int_{-\infty}^y \int \rho^o(x, y, z)dy dz }
                        {\int \int \rho^o(x, y, z)dydz} \\
            \theta_z(x, y, z) :&=
            -1 + 2 \frac{\int_{-\infty}^z \rho^o(x, y, z)dz }
                        {\int \rho^o(x, y, z)dz}\\
        \end{eqnarray}

    Integration of a integrable function :math:`f : \mathbb{R}^3 \rightarrow \mathbb{R}` can be
    done as follows in theta space:

    .. math::
        \int \int \int f(x, y, z)dxdy dz \approx
        \frac{N}{8} \int_{-1}^1 \int_{-1}^1 \int_{-1}^1 \frac{f(\theta_x, \theta_y, \theta_z)}
        {\rho^o(\theta_x, \theta_y, \theta_z)} d\theta_x d\theta_y d\theta_z,

        \text{where }  N = \int \int \int \rho^o(x, y, z) dx dy dz.

    Note that this class always assumed the boundary of [-1, 1]^3 is always included.

    """

    def __init__(self, oned_grids, coeffs, exps, coords, boundary_epsilon=1e-12):
        r"""
        Construct CubicProTransform object.

        Parameters
        ----------
        oned_grids: List[OneDGrid]
            List of three one-dimensional grid representing the grids along x-axis.
        coeffs: List[List[float]]
            Coefficients of the promolecular transformation over :math:`M` centers.
        exps: List[List[float]]
            Exponents of the promolecular transformation over :math:`M` centers.
        coords: ndarray(M, 3)
            The coordinates of the promolecular expansion.
        boundary_epsilon : float
            Small offset used to obtain finite working coordinates when a preceding real-space
            coordinate is infinite. Since :math:`-\infty` and :math:`+\infty` correspond to
            :math:`\theta=-1` and :math:`\theta=+1`, respectively, the boundary value is
            replaced by :math:`\pm(1-\varepsilon)` and transformed back to a finite real-space
            coordinate. The finite coordinate is then used when evaluating subsequent
            transformations. Default is `1e-12`.

        """
        if not isinstance(oned_grids, list):
            raise TypeError("oned_grid should be of type list.")
        if not np.all([isinstance(grid, OneDGrid) for grid in oned_grids]):
            raise TypeError("Grid in oned_grids should be of type `OneDGrid`.")
        if not np.all([grid.domain == (-1, 1) for grid in oned_grids]):
            raise ValueError("One Dimensional grid domain should be (-1, 1).")
        if not len(oned_grids) == 3:
            raise ValueError("There should be three One-Dimensional grids in `oned_grids`.")
        if not 0.0 < boundary_epsilon < 1.0:
            raise ValueError("boundary_epsilon must lie in (0, 1).")

        self._l_bnd = -1.0
        self._u_bnd = 1.0
        self._shape = tuple([grid.size for grid in oned_grids])
        self._boundary_epsilon = boundary_epsilon
        dimension = len(oned_grids)

        # pad coefficients and exponents with zeros to have the same size, easier to use numpy.
        coeffs, exps = _pad_coeffs_exps_with_zeros(coeffs, exps)
        self._promol = _PromolParams(coeffs, exps, coords, dimension)
        self._prointegral = self._promol.integrate_all()

        weights = np.kron(
            np.kron(oned_grids[0].weights, oned_grids[1].weights), oned_grids[2].weights
        )
        # Transform Cubic Grid in Theta-Space to Real-space.
        points = self._inverse_transform_grid(oned_grids)
        # The prointegral is needed because of promolecular integration.
        # Divide by 8 needed because the grid is in [-1, 1] rather than [0, 1].
        super().__init__(points, weights * self._prointegral / 2.0**dimension, self._shape)

    @property
    def l_bnd(self):
        r"""float: Lower bound in theta-space. Any point in theta-space that contains this point
        gets mapped to infinity."""
        return self._l_bnd

    @property
    def u_bnd(self):
        r"""float: Upper bound in theta-space. Any point in theta-space that contains this point
        gets mapped to infinity."""
        return self._u_bnd

    @property
    def prointegral(self):
        r"""Return integration of Promolecular density."""
        return self._prointegral

    @property
    def promol(self):
        r"""Return `PromolParams` data class."""
        return self._promol

    def transform(self, real_pt, boundary_epsilon=1e-12):
        r"""Transform a real-space point to theta space.

        Parameters
        ----------
        real_pt : array-like, shape (D,)
            Real-space point in :math:`\mathbb{R}^D`.
        boundary_epsilon : float, optional
            Distance from the theta-space endpoints used to construct regularized finite working
            coordinates for infinite preceding coordinates. The regularized endpoint is
            :math:`\pm(1-\varepsilon)`. Default is ``1e-12``.

        Returns
        -------
        theta_pt : ndarray, shape (D,)
            Transformed point in :math:`[-1,1]^D`.

        Raises
        ------
        ValueError
            If ``real_pt`` does not have shape ``(D,)`` or
            ``boundary_epsilon`` does not lie in ``(0,1)``.

        """
        real_pt = np.asarray(real_pt)
        if real_pt.shape != (self.promol.dim,):
            raise ValueError(
                f"`real_pt` must have shape ({self.promol.dim},), got {real_pt.shape}."
            )

        return np.array(
            [
                _transform_coordinate(real_pt, i_var, self.promol, boundary_epsilon)
                for i_var in range(self.promol.dim)
            ]
        )

    def inverse(self, theta_pt):
        r"""Transform a theta space point to three-dimensional Real space.

         Parameters
         ----------
         theta_pt : np.ndarray(3)
             Point in :math:`[-1, 1]^3`

         Returns
         -------
         real_pt : np.ndarray(3)
             Point in :math:`\mathbb{R}^3`

        Notes
         -----
         Theta-space boundary values :math:`-1` and :math:`1` map to
         :math:`-\infty` and :math:`+\infty`, respectively.

        """
        theta_pt = np.asarray(theta_pt)
        if theta_pt.shape != (self.promol.dim,):
            raise ValueError(
                f"`theta_pt` must have shape ({self.promol.dim},), got {theta_pt.shape}."
            )
        real_pt = []
        for i in range(0, self.promol.dim):
            scalar = _inverse_coordinate(theta_pt[i], i, real_pt[:i], self.promol)
            real_pt.append(scalar)
        return np.array(real_pt)

    def integrate(self, *value_arrays, trick=False, tol=1e-10):
        r"""Integrate any real-valued function on Euclidean space.

        For an integrable function:

        .. math::
            f : \mathbb{R}^3 \rightarrow \mathbb{R}

        The integral is approximated as follows:

        .. math::
            \int \int \int f(x, y, z)dxdy dz \approx
            \frac{1}{8} N \int_{-1}^1 \int_{-1}^1 \int_{-1}^1 \frac{f(\theta_x, \theta_y, \theta_z)}
            {\rho^o(\theta_x, \theta_y, \theta_z)} d\theta_x d\theta_y d\theta_z,

            \text{where }  N = \int \int \int \rho^o(x, y, z) dx dy dz.

        Assumes the function decays faster than the promolecular density.

        Parameters
        ----------
        *value_arrays : (np.ndarray(N, dtype=float),)
            One or multiple value array to integrate.
        trick : bool, optional
            If true, uses the promolecular trick.
        tol : float, optional
            Integrand is set to zero whenever promolecular density is less than tolerance.
            Default value is 1e-10.

        Returns
        -------
        float :
            Return the integration of the function.

        Raises
        ------
        TypeError
            Input integrand is not of type np.ndarray.
        ValueError
            Input integrand array is given or not of proper shape.

        """
        if not value_arrays:
            raise ValueError("At least one value array must be provided.")

        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a finite, non-negative number.")

        promol_vals = self.promol.promolecular(self.points)

        for arr in value_arrays:
            if not isinstance(arr, np.ndarray):
                raise TypeError("Each integrand must be a NumPy array.")
            if arr.shape != promol_vals.shape:
                raise ValueError("Each integrand must have the same shape as the grid.")
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError("Each integrand must be a numerical array.")

        # Select points where division by the promolecular density is safe
        active = np.isfinite(promol_vals) & (promol_vals > tol)
        dtype = np.result_type(np.float64, promol_vals.dtype, *(arr.dtype for arr in value_arrays))

        # Compute 1 / rho0, but only where rho0 is finite and above the tolerance
        inv_promol_vals = np.zeros(promol_vals.shape, dtype=dtype)
        np.divide(1.0, promol_vals, out=inv_promol_vals, where=active)

        # Compute the product of all value arrays
        values = np.ones(promol_vals.shape, dtype=dtype)
        for arr in value_arrays:
            # Inactive entries are never read; values remains 1 there.
            np.multiply(values, arr, out=values, where=active)

        # In trick mode, integrate f - rho0 instead of f.
        if trick:
            np.subtract(values, promol_vals, out=values, where=active)

        if not np.all(np.isfinite(values)):
            raise ValueError("Integrand contains non-finite values at active points.")

        integral = super().integrate(values, inv_promol_vals)

        # If trick is True, add back the analytic integral of the promolecular density
        if trick:
            integral += self.prointegral

        return integral

    def gradient_to_theta(self, real_pt, real_gradient):
        r"""Transform real-space gradient components to theta coordinates.

        It returns the gradient of g in theta-space, given the gradient of f in real-space.

        Given the inverse promolecular transformation from theta-space to real-space:

        .. math::
            g(\boldsymbol{\theta}) = f(\mathbf{r}(\boldsymbol{\theta}))

        then:

        .. math::
            \nabla_{\boldsymbol{\theta}} g = J^{-T}\nabla_{\mathbf r} f

        where :math:`J` is the Jacobian matrix of the forward transformation from real space to
        theta space.

        Parameters
        ----------
        real_pt : np.ndarray, shape (3,)
            Point in :math:`\mathbb{R}^3`.
        real_gradient : np.ndarray, shape (3,)
            Gradient of a function in real space with respect to x, y, z coordinates.

        Returns
        -------
        theta_gradient : np.ndarray, shape (3,)
            Gradient of a function in theta space with respect to theta coordinates.

        Notes
        -----
        This transforms gradient components and does not map the real-space steepest-ascent
        direction.

        See Also
        --------
        steepest_ascent_theta : Steepest-ascent direction.

        """
        jacobian = self.jacobian(real_pt)
        return solve_triangular(jacobian.T, real_gradient)

    def steepest_ascent_theta(self, real_pt, real_grad):
        r"""Steepest ascent direction of a function in theta space.

        Steepest ascent is the gradient ie direction of maximum change of a function.
        This guarantees moving in direction of steepest ascent in real-space
        corresponds to moving in the direction of the gradient in theta-space.

        Parameters
        ----------
        real_pt : np.ndarray(3)
            Point in :math:`\mathbb{R}^3`
        real_grad : np.ndarray(3)
            Gradient of a function in real space.

        Returns
        -------
        theta_grad : np.ndarray(3)
            Theta-space image of the real-space steepest-ascent direction.

        """
        jacobian = self.jacobian(real_pt)
        return jacobian.dot(real_grad)

    def interpolate(self, points, values, oned_grids, use_log=False, nu=0):
        r"""Interpolate a function or its gradient at real-space points.

        Parameters
        ----------
        points : np.ndarray(M, 3)
            Points in :math:`\mathbb{R}^3` at which to interpolate.
        values : np.ndarray(N,)
            Function values at the tensor-product grid defined by `oned_grids`.
        oned_grids : list[OneDGrid]
            Three one-dimensional grids corresponding to the x, y, and z directions.
        use_log : bool, optional
            If True, interpolate the logarithm of the function values.
        nu : {0, 1}, optional
            Derivative order. If zero, interpolate the function values.
            If one, interpolate the real-space gradient.

        Returns
        -------
        np.ndarray
            If ``nu == 0``, interpolated function values with shape ``(M,)``.
            If ``nu == 1``, interpolated real-space gradients with shape ``(M, 3)``.

        """
        if nu not in (0, 1):
            raise ValueError(f"The parameter nu {nu} must be either zero or one.")

        # Create a meshgrid of the points in theta-space corresponding to the one-dimensional grids
        grids_mesh = np.meshgrid(*(grid.points for grid in oned_grids), indexing="ij")
        grid_pts = np.stack(grids_mesh, axis=-1).reshape(-1, 3)

        # Transform points to interpolate to theta-space
        theta_points = np.array([self.transform(x) for x in points], dtype=float)

        # If nu is 0, interpolate the function values directly
        if nu == 0:
            return super()._interpolate(theta_points, values, use_log, 0, 0, 0, grid_pts=grid_pts)

        # compute the gradient in theta-space by interpolating each component separately
        interpolate_x = super()._interpolate(
            theta_points, values, use_log, 1, 0, 0, grid_pts=grid_pts
        )
        interpolate_y = super()._interpolate(
            theta_points, values, use_log, 0, 1, 0, grid_pts=grid_pts
        )
        interpolate_z = super()._interpolate(
            theta_points, values, use_log, 0, 0, 1, grid_pts=grid_pts
        )
        grad_theta = np.stack((interpolate_x, interpolate_y, interpolate_z), axis=-1)
        # Convert the theta-space gradient to the real-space gradient
        # grad_r(f) = J(theta <- r).T @ grad_theta(f).
        return np.array(
            [
                self.jacobian(point).T @ gradient
                for point, gradient in zip(points, grad_theta, strict=True)
            ]
        )

    def jacobian(self, real_pt):
        r"""Return the Jacobian of the transformation from real space to theta space.

        The Jacobian elements are defined as

        .. math::

            J_{ij} = \frac{\partial \Theta_i}{\partial r_j}.

        The transformation maps :math:`\mathbf{r} \in \mathbb{R}^D` to
        :math:`\boldsymbol{\theta} \in [-1, 1]^D` sequentially as

        .. math::

            \Theta_i(r_i \mid \mathbf{r}_{<i})
            =
            2\frac{N_i(r_i, \mathbf{r}_{<i})}{D_i(\mathbf{r}_{<i})} - 1,

        where

        .. math::

            N_i(r_i, \mathbf{r}_{<i})
            =
            \int_{-\infty}^{r_i}
            \int_{\mathbb{R}^{D-i-1}}
            \rho^o(\mathbf{r}_{<i}, s_i, \mathbf{s}_{>i})
            \,d\mathbf{s}_{>i}\,ds_i,

        and

        .. math::

            D_i(\mathbf{r}_{<i})
            =
            \int_{-\infty}^{\infty}
            \int_{\mathbb{R}^{D-i-1}}
            \rho^o(\mathbf{r}_{<i}, s_i, \mathbf{s}_{>i})
            \,d\mathbf{s}_{>i}\,ds_i.

        Since :math:`\Theta_i` depends only on :math:`r_0, \ldots, r_i`, :math:`J_{ij}=0` for
        :math:`j>i`. Therefore, the Jacobian is lower triangular:

        .. math::

            \mathbf{J}
            =
            \begin{bmatrix}
                J_{00} & 0 & \cdots & 0 \\
                J_{10} & J_{11} & \cdots & 0 \\
                \vdots & \vdots & \ddots & \vdots \\
                J_{D-1,0} & J_{D-1,1} & \cdots & J_{D-1,D-1}
            \end{bmatrix}.

        For the diagonal terms, :math:`D_i` is independent of :math:`r_i`, so

        .. math::

            J_{ii} = 2\frac{\partial_{r_i}N_i}{D_i}

        For the off-diagonal terms, :math:`j<i`, both :math:`N_i` and :math:`D_i` depend on
        :math:`r_j`, giving

        .. math::

            J_{ij} = 2 \frac{(\partial_{r_j}N_i)D_i - N_i(\partial_{r_j}D_i)}{D_i^2}.

        Parameters
        ----------
        real_pt : np.ndarray(D,)
            Point in :math:`\mathbb{R}^D` at which the Jacobian is evaluated.

        Returns
        -------
        np.ndarray(D, D)
            Jacobian :math:`\partial\boldsymbol{\theta}/\partial\mathbf{r}` evaluated at `real_pt`.

        """
        ndim = self.promol.dim
        jacobian = np.zeros((ndim, ndim), dtype=np.float64)

        e_m = self.promol.e_m  # shape (ncenters, ndims)
        pi_over_exps = self.promol.pi_over_exponents  # shape (ncenters, ndim)
        centers = self.promol.gauss_centers  # shape (ncenters, ndim)
        pi_over_exps = self.promol.pi_over_exponents  # shape (ncenters, ndim)

        # Coordinate differences from each Gaussian center.
        diff_coords = real_pt - centers
        diff_squared = diff_coords**2

        for i_var in range(ndim):
            _, single_gauss, integrate_till_pt_x, transf_num, transf_den = (
                self.promol.helper_for_derivatives(diff_squared, diff_coords, i_var)
            )

            # transformation is: Theta_i = 2 N_i / D_i - 1
            # only lower-triangular part of the Jacobian is non-zero, so we only compute for j <= i
            for j_deriv in range(i_var + 1):
                if j_deriv == i_var:
                    # Diagonal case J_{ii} = 2 * dN_i / D_i
                    # dN_i / dr_i = sum_k [single_gauss[k] * exp(-alpha_k * (r_i - mu_ki)^2)]

                    # Shape (ncenters, 1) to broadcast each center's distance over its exponents.
                    diff_squared_i = diff_squared[:, i_var][:, np.newaxis]
                    deriv_num = np.sum(single_gauss * np.exp(-e_m * diff_squared_i))
                    jacobian[i_var, i_var] = 2.0 * deriv_num / transf_den

                else:
                    # Off-diagonal case (j < i):
                    # r_j appears in the Gaussian factors of both N_i and D_i.
                    gauss_deriv_factor = self.promol.derivative_gaussian(diff_coords, j_deriv)
                    # dN_i / dr_j
                    deriv_num = np.sum(single_gauss * integrate_till_pt_x * gauss_deriv_factor)
                    # dD_i / dr_j
                    deriv_den = np.sum(single_gauss * gauss_deriv_factor * pi_over_exps)
                    jacobian[i_var, j_deriv] = (
                        2.0 * (deriv_num * transf_den - transf_num * deriv_den) / transf_den**2
                    )

        return jacobian

    def hessian(self, real_pt):
        r"""Return the Hessian of the transformation from real space to theta space.

        The Hessian elements are defined as

        .. math::

            H_{ijk} = \frac{\partial^2 \Theta_i} {\partial r_j \partial r_k}.

        Since :math:`\Theta_i` depends only on :math:`r_0, \ldots, r_i`, :math:`H_{ijk}=0`
        if :math:`j>i` or :math:`k>i`.

        For the smooth Gaussian transformation, mixed partial derivatives commute,

        .. math::

            H_{ijk} = H_{ikj}.


        Parameters
        ----------
        real_pt : np.ndarray(N,)
            Real point in :math:`\mathbb{R}^N`.

        Returns
        -------
        hessian : np.ndarray(N, N, N)
            Hessian tensor of the transformation at the real point. The :math:`H_{ijk}`
            entry is the second partial derivative of the :math:`i`th transformation function with
            respect to the :math:`j`th and :math:`k`th coordinates. Nonzero entries satisfy
            :math:`j \leq i` and :math:`k \leq i`; for example, when :math:`i = 0`, only
            :math:`H_{000}` can be nonzero.

        """
        hessian = np.zeros((self.ndim, self.ndim, self.ndim), dtype=np.float64)
        e_m = self.promol.e_m
        centers = self.promol.gauss_centers
        pi_over_exps = self.promol.pi_over_exponents

        # Coordinate differences to each Gaussian center and their squares
        diff_coords = real_pt - centers
        diff_squared = diff_coords**2.0

        # H[i, j, k] = \partial^2 \Theta_i / (\partial r_j \partial r_k)
        # factor 2 from Theta_i = 2 N_i / D_i - 1 is applied at the end for each entry
        for i_var in range(self.ndim):
            # Theta_i = 2 N_i / D_i - 1, with num_i = N_i and den_i = D_i.
            _, gauss_terms, integrate_till_pt_i, num_i, den_i = self.promol.helper_for_derivatives(
                diff_squared, diff_coords, i_var
            )
            # Per-Gaussian contributions to the numerator N_i and denominator D_i
            # num_i = sum(num_terms) and den_i = sum(den_terms)
            num_terms = gauss_terms * integrate_till_pt_i
            den_terms = gauss_terms * pi_over_exps
            for j_deriv in range(i_var + 1):
                # exploit symmetry of Hessian H[i, j, k] = H[i, k, j], so only compute for k <= j
                for k_deriv in range(j_deriv + 1):
                    if i_var == j_deriv:
                        # Contribution of each Gaussian component to dN_i / dr_i
                        dnum_di_terms = gauss_terms * np.exp(
                            -e_m * diff_squared[:, i_var][:, np.newaxis]
                        )
                        # H[i,i,i] = 2 * d^2(N_i)/dr_i^2 / D_i
                        if i_var == k_deriv:
                            # Gaussian derivative factor for d/dr_i
                            factor_g_di = self.promol.derivative_gaussian(diff_coords, i_var)
                            # Contributions to d^2N_i / dr_i^2
                            d2num_di2_terms = dnum_di_terms * factor_g_di
                            ratio_d2 = np.sum(d2num_di2_terms) / den_i
                        else:
                            # Gaussian derivative factor for d/dr_k
                            factor_g_dk = self.promol.derivative_gaussian(diff_coords, k_deriv)
                            # Differentiate each dN_i/dr_i Gaussian contribution with respect to r_k
                            d2num_dkdi = np.sum(dnum_di_terms * factor_g_dk)
                            # dD/d r_k and dN/d r_i as sum of Gaussian contributions
                            dden_dk = np.sum(den_terms * factor_g_dk)
                            dnum_di = np.sum(dnum_di_terms)
                            # Quotient rule for d[(dN_i/dr_i) / D_i] / dr_k:
                            # [(d2N_i/dr_k dr_i) D_i - (dN_i/dr_i)(dD_i/dr_k)] / D_i^2
                            ratio_d2 = d2num_dkdi * den_i - dnum_di * dden_dk
                            ratio_d2 /= den_i**2.0

                    elif j_deriv < i_var:
                        if k_deriv == j_deriv:
                            # gaussian derivative factor for d/dr_j and d/dr_k (j == k)
                            factor_g_dj = self.promol.derivative_gaussian(diff_coords, j_deriv)
                            # dN/dr_j and dD/dr_j as sum of Gaussian contributions
                            dnum_dj = np.sum(num_terms * factor_g_dj)
                            dden_dj = np.sum(den_terms * factor_g_dj)

                            # gaussian second-derivative (1/G) d^2G/dr_j^2 = factor_g_dj^2 - 2 alpha
                            factor_g_d2j = factor_g_dj**2.0 - 2.0 * e_m
                            # sum of gaussian contributions to d^2N_i / dr_j^2 and d^2D_i / dr_j^2
                            d2num_dj2 = np.sum(num_terms * factor_g_d2j)
                            d2den_dj2 = np.sum(den_terms * factor_g_d2j)
                            # Second derivative d^2(N_i / D_i) / dr_j^2
                            # (d^2N_i/dr_j^2) / D_i
                            # - [2 (dN_i/dr_j)(dD_i/dr_j) + N_i (d^2D_i/dr_j^2)] / D_i^2
                            # + 2 N_i (dD_i/dr_j)^2 / D_i^3
                            ratio_d2 = d2num_dj2 / den_i
                            ratio_d2 -= (2.0 * dnum_dj * dden_dj + num_i * d2den_dj2) / den_i**2.0
                            ratio_d2 += (2.0 * num_i * dden_dj**2.0) / den_i**3.0

                        else:
                            # Gaussian derivative factors for d/dr_j and d/dr_k.
                            factor_g_dj = self.promol.derivative_gaussian(diff_coords, j_deriv)
                            factor_g_dk = self.promol.derivative_gaussian(diff_coords, k_deriv)

                            # Sum Gaussian contributions to dN_i/dr_j and dD_i/dr_j.
                            dnum_dj = np.sum(num_terms * factor_g_dj)
                            dden_dj = np.sum(den_terms * factor_g_dj)

                            # dN_i / dr_k and dD_i / dr_k.
                            dnum_dk = np.sum(num_terms * factor_g_dk)
                            dden_dk = np.sum(den_terms * factor_g_dk)

                            # d^2N_i /(dr_k dr_j) and d^2D_i /(dr_k dr_j)
                            d2num_djdk = np.sum(num_terms * factor_g_dj * factor_g_dk)
                            d2den_djdk = np.sum(den_terms * factor_g_dj * factor_g_dk)

                            # d^2(N_i / D_i) / (dr_j dr_k) = (d^2N_i/dr_j dr_k) / D_i
                            # - [(dN_i/dr_j)(dD_i/dr_k) + (dN_i/dr_k)(dD_i/dr_j)
                            # + N_i(d^2D_i/dr_j dr_k)] / D_i^2 + 2 N_i(dD_i/dr_j)(dD_i/dr_k) / D_i^3
                            ratio_d2 = d2num_djdk / den_i
                            ratio_d2 -= (
                                dnum_dj * dden_dk + dnum_dk * dden_dj + num_i * d2den_djdk
                            ) / den_i**2.0
                            ratio_d2 += (2.0 * num_i * dden_dj * dden_dk) / den_i**3.0
                    # Theta_i = 2 N_i / D_i - 1, so H_ijk = 2 d^2(N_i/D_i)/(dr_j dr_k) and
                    # H[i, j, k] = H[i, k, j] by symmetry
                    hessian_entry = 2.0 * ratio_d2
                    hessian[i_var, j_deriv, k_deriv] = hessian_entry
                    hessian[i_var, k_deriv, j_deriv] = hessian_entry

        return hessian

    def _inverse_transform_grid(self, oned_grids):
        """Map the entire grid from theta-space to real-space.

        Parameters
        ----------
        oned_grids : list(OneDGrid)
            List of three one-dimensional grids in theta-space, representing x, y, z directions.

        Returns
        -------
        points : np.ndarray(N, 3)
            Points in :math:`\mathbb{R}^3` corresponding to the tensor-product grid defined by
            `oned_grids`.
        """
        counter = 0
        points = np.empty((np.prod(self.shape), len(oned_grids)), dtype=np.float64)

        for ix in range(self.shape[0]):
            cart_pt = [None, None, None]
            work_pt = [None, None, None]

            theta_x = oned_grids[0].points[ix]
            cart_pt[0] = _inverse_coordinate(theta_x, 0, work_pt, self.promol)

            # Use a finite x coordinate to condition subsequent transformations.
            if np.abs(theta_x) == 1.0:
                theta_x_reg = np.sign(theta_x) * (1.0 - self._boundary_epsilon)
                work_pt[0] = _inverse_coordinate(theta_x_reg, 0, work_pt, self.promol)
            else:
                work_pt[0] = cart_pt[0]

            for iy in range(self.shape[1]):
                theta_y = oned_grids[1].points[iy]
                cart_pt[1] = _inverse_coordinate(theta_y, 1, work_pt, self.promol)

                # Use a finite y coordinate to condition subsequent transformations.
                if np.abs(theta_y) == 1.0:
                    theta_y_reg = np.sign(theta_y) * (1.0 - self._boundary_epsilon)
                    work_pt[1] = _inverse_coordinate(theta_y_reg, 1, work_pt, self.promol)
                else:
                    work_pt[1] = cart_pt[1]

                for iz in range(self.shape[2]):
                    theta_z = oned_grids[2].points[iz]
                    cart_pt[2] = _inverse_coordinate(theta_z, 2, work_pt, self.promol)

                    points[counter] = cart_pt.copy()
                    counter += 1
        return points


@dataclass
class _PromolParams:
    r"""
    Private class for Promolecular Density information.

    Contains helper-functions for Promolecular Transformation.
    They are coded as pipe-lines for this special purpose and
    the reason why "diff_coords" is chosen as a attribute rather
    than a generic "[x, y, z]" point.

    """

    c_m: np.ndarray  # Coefficients of Promolecular.
    e_m: np.ndarray  # Exponents of Promolecular.
    gauss_centers: np.ndarray  # Centers/Coordinates of Each Gaussian.
    pi_over_exponents: np.ndarray = field(init=False)
    dim: int = 3

    def __post_init__(self):
        r"""Initialize pi_over_exponents."""
        # Rather than computing this repeatedly. It is fixed.
        with np.errstate(divide="ignore"):
            self.pi_over_exponents = np.sqrt(np.pi / self.e_m)
            self.pi_over_exponents[self.e_m == 0.0] = 0.0

    def integrate_all(self):
        r"""Integrate the Gaussian expansion over :math:`\mathbb{R}^D`."""
        active = self.c_m > 0.0

        # express integral in log space to avoid overflow/underflow issues for large/small exponents
        # and coefficients
        log_terms = np.log(self.c_m[active]) + self.dim * np.log(self.pi_over_exponents[active])
        return np.exp(logsumexp(log_terms))

    def derivative_gaussian(self, diff_coords, j_deriv):
        r"""Return the derivative factor for each Gaussian component.

        For

        .. math::

            G_{k}(\mathbf r) =
            c_k\exp\left[ -\alpha_k\lVert\mathbf r-\boldsymbol\mu_k\rVert^2 \right]

        the derivative with respect to coordinate :math:`r_j` is

        .. math::

            \frac{\partial G_k}{\partial r_j} = -2\alpha_k(r_j-\mu_{k,j})G_k(\mathbf r).

        This method returns only the factor :math:`-2\alpha_k(r_j-\mu_{k,j})`, without :math:`G_k`.

        Parameters
        ----------
        diff_coords : ndarray, shape (M, D)
            Coordinate differences between the point and each of the ``M`` Gaussian centers, defined
            by ``diff_coords[m, j] = r_j - mu[m, j]``.
        j_deriv : int
            Coordinate index with respect to which the derivative is taken.

        Returns
        -------
        derivative_factors : ndarray, shape (M, N)
            Multiplicative derivative factor for each of the ``N`` Gaussian components associated
            with each of the ``M`` centers.
        """
        return -2.0 * self.e_m * diff_coords[:, j_deriv, np.newaxis]

    def integration_gaussian_till_point(self, diff_coords, i_var, with_factor=False):
        r"""Integrate each Gaussian component along coordinate ``i_var`` up to a point.

        For Gaussian component :math:`k`, the one-dimensional integral is

        .. math::

            I_{k,i}(r_i) = \int_{-\infty}^{r_i} \exp\left[-\alpha_k(s_i-\mu_{k,i})^2\right]ds_i

        By default, the method returns the normalized integral

        .. math::

            F_{k,i}(r_i) = \frac{I_{k,i}(r_i)}{\sqrt{\pi/\alpha_k}}.

        If ``with_factor`` is ``True``, the normalization factor :math:`\sqrt{\pi/\alpha_k}` is
        included, and the method returns :math:`I_{k,i}(r_i)`. The Gaussian coefficient :math:`c_k`
        is not included.

        Parameters
        ----------
        diff_coords : ndarray, shape (M, D)
            Coordinate differences between the point and the ``M`` Gaussian centers.
        i_var : int
            Coordinate with respect to which the integration is performed.
        with_factor : bool, optional
            Whether to return the unnormalized integral. Default is ``False``.

        Returns
        -------
        integrals : ndarray, shape (M, N)
            Normalized or unnormalized one-dimensional Gaussian integrals.

        """
        coord_ivar = diff_coords[:, i_var, np.newaxis]
        scaled_coord = np.sqrt(self.e_m) * coord_ivar

        # This equals 0.5 * (1 + erf(z)) but avoids subtractive cancellation z is large and negative
        integration = 0.5 * erfc(-scaled_coord)

        if with_factor:
            # Multiply by sqrt(pi / alpha_k) to recover the full integral
            return integration * self.pi_over_exponents

        return integration

    def evaluate_gaussians(self, square_distance):
        r"""Return matrix with the value of each Gaussian component at the given squared distances.

        Parameters
        ----------
        square_distance : ndarray, shape (M, 1)
            Squared distance for each of the ``M`` centers.

        Returns
        -------
        gaussian_values : ndarray, shape (M, D)
            Gaussian values ``G_ij = c_ij * exp(-alpha_ij * squared_distances_i)``.
            Rows correspond to centers and columns to the ``D`` Gaussian components associated with
            each center.
        """
        return self.c_m * np.exp(-self.e_m * square_distance)

    def evaluate_log_gaussians(self, square_distance):
        r"""Return matrix of logarithms of Gaussian components at the given squared distances.

        Parameters
        ----------
        square_distance : ndarray, shape (M, 1)
            Squared distance for each of the ``M`` centers.

        Returns
        -------
        gaussian_values : ndarray, shape (M, D)
            Gaussian values ``G_ij = c_ij * exp(-alpha_ij * squared_distances_i)``.
            Rows correspond to centers and columns to the ``D`` Gaussian components associated with
            each center.
        """
        log_coeffs = np.full_like(self.c_m, -np.inf, dtype=float)

        # Components with c_k = 0 retain log(c_k) = -inf
        np.log(self.c_m, out=log_coeffs, where=self.c_m > 0.0)

        return log_coeffs - self.e_m * square_distance

    def promolecular(self, points):
        r"""Evaluate the promolecular density at a collection of points.

        For point :math:`\mathbf r_n`, the density is

        .. math::

            \rho^o(\mathbf r_n) = \sum_{m=0}^{M-1} \sum_{k=0}^{K-1} c_{m,k}
            \exp\left[ -\alpha_{m,k} \lVert\mathbf r_n-\boldsymbol\mu_m\rVert^2 \right]

        where :math:`M` is the number of centers and :math:`K` is the number of Gaussian
        components stored for each center.

        Parameters
        ----------
        points : ndarray, shape (N, D)
            Coordinates of the ``N`` points in :math:`\mathbb{R}^D`.

        Returns
        -------
        density : ndarray, shape (N,)
            Promolecular density evaluated at each input point.

        """
        points = np.asarray(points)

        # Broadcast points (1, N, D) vs centers (M, 1, D) summing over D
        # the size-1 axis of the points expands to M, while the size-1 axis of the centers expands
        # to N. The common D axis remains unchanged, giving coordinate differences with shape
        # (M, N, D). Squaring and summing over D gives shape (M, N, 1).
        squared_distances = np.sum(
            (points[np.newaxis, :, :] - self.gauss_centers[:, np.newaxis, :]) ** 2,
            axis=2,
            keepdims=True,
        )

        # Broadcast Gaussians coefficients and exponents (M, 1, K) vs squared distances (M, N, 1).
        # The size-1 point axis expands to N, while the size-1 component axis expands to K.
        # The common M axis remains unchanged, giving Gaussian contributions with shape(M, N, K).
        gaussian_values = self.c_m[:, np.newaxis, :] * np.exp(
            -self.e_m[:, np.newaxis, :] * squared_distances
        )

        # Sum over the M centers and K Gaussian components, retaining the N point dimension.
        return np.sum(gaussian_values, axis=(0, 2), dtype=np.float64)

    def helper_for_derivatives(self, diff_squared, diff_coords, i_var):
        r"""
        Return terms used to differentiate the transformation functions wrt x, y, z.

        Parameters
        ----------
        diff_squared : ndarray, shape (M, D)
            Elementwise squared coordinate differences, equal to ``diff_coords**2``.
        diff_coords : ndarray, shape (M, D)
            Signed coordinate differences from each Gaussian center, defined by
            ``diff_coords[m, j] = r_j - mu[m, j]``.
        i_var : int
            Index of the coordinate being transformed, e.g., ``0`` for x, ``1`` for y, and ``2``
            for z.

        Returns
        -------
        preceding_squared_distances : ndarray, shape (M, 1)
            Squared distances from each center in the preceding coordinates.
        single_gauss : np.ndarray
            Array with entries of a single Gaussian e^(-a distance) with factor (pi / a).
        integrate_till_pt : ndarray, shape (M, N)
            Unnormalized Gaussian integrals along the current coordinate from :math:`-\infty` to its
            current value:

            .. math::

                I_{k,i}(r_i) = \int_{-\infty}^{r_i} \exp\left[-\alpha_k(s_i-\mu_{k,i})^2\right] ds_i

        transf_num : float
            The numerator of the transformation. Mostly used for quotient rule.
        transf_den : float
            The denominator of the transformation. Mostly used for quotient rule.

        """
        distance = np.sum(diff_squared[:, :i_var], axis=1)[:, np.newaxis]

        # Evaluate each Gaussian using the preceding-coordinate distances
        single_gauss = self.evaluate_gaussians(distance)

        # Integrate over the D - i_var - 1 subsequent coordinates
        single_gauss *= self.pi_over_exponents ** (self.dim - i_var - 1)

        # Integrate each Gaussian along the current coordinate up to the point
        integrate_till_pt_x = self.integration_gaussian_till_point(
            diff_coords, i_var, with_factor=True
        )

        # Numerator and Denominator of Original Transformation.
        transf_num = np.sum(single_gauss * integrate_till_pt_x)
        transf_den = np.sum(single_gauss * self.pi_over_exponents)
        return distance, single_gauss, integrate_till_pt_x, transf_num, transf_den


def _transform_coordinate(real_pt, i_var, promol, boundary_epsilon=1e-12):
    r"""
    Transform the `i_var` coordinate of a real point to [-1, 1] using promolecular density.

    For :math:`\mathbf{r}=(r_0,\ldots,r_{D-1})`, coordinate :math:`i` is transformed using

    .. math::

        \theta_i(\mathbf{r}_{\leq i})
        = -1 + 2
        \frac{
            \int_{-\infty}^{r_i}
            \int_{\mathbb{R}^{D-i-1}}
            \rho^o(\mathbf{r}_{<i},s_i,\mathbf{s}_{>i})
            \,d\mathbf{s}_{>i}\,ds_i
        }{
            \int_{-\infty}^{\infty}
            \int_{\mathbb{R}^{D-i-1}}
            \rho^o(\mathbf{r}_{<i},s_i,\mathbf{s}_{>i})
            \,d\mathbf{s}_{>i}\,ds_i
        }.

    The preceding coordinates :math:`\mathbf{r}_{<i}` are held fixed, the current coordinate
    :math:`s_i` is integrated from :math:`-\infty` to :math:`r_i`, and the subsequent coordinates
    :math:`\mathbf{s}_{>i}` are integrated over their complete domains.

    Parameters
    ----------
    real_pt : np.ndarray(D,)
        Real point being transformed.
    i_var : int
        Index of the variable being transformed (0 for x, 1 for y, 2 for z).
    promol : _PromolParams
        Promolecular Data Class.
    boundary_epsilon : float, optional
        Small positive value used to regularize coordinates at the boundaries. Default is 1e-12.

    Returns
    -------
    float or complex
        Transformed coordinate. For finite real input, the result lies in ``[-1, 1]``. A complex
        input preserves the imaginary perturbation required for complex-step differentiation.

    Raises
    ------
    ValueError
        If any preceding coordinate is nonfinite.
    """
    if not 0.0 < boundary_epsilon < 1.0:
        raise ValueError("boundary_epsilon must lie in (0, 1).")

    real_pt = np.asarray(real_pt)

    # An infinite current coordinate maps directly to theta_i = +/-1.
    if np.isinf(real_pt[i_var]):
        return np.sign(np.real(real_pt[i_var]))

    # regularize real_pt at the boundaries
    reg_real_pt = real_pt.copy()
    for prev_var in range(i_var):
        if np.isinf(reg_real_pt[prev_var]):
            # find corresponding near boundary theta value
            sign = np.sign(np.real(reg_real_pt[prev_var]))
            regularized_theta = sign * (1.0 - boundary_epsilon)

            # convert back to real space the regularized theta value
            reg_real_pt[prev_var] = _inverse_coordinate(
                regularized_theta, prev_var, reg_real_pt, promol
            )

    coords = promol.gauss_centers
    pi_over_exps = promol.pi_over_exponents
    num_integrated_dims = promol.dim - i_var

    # Coordinate offsets through `i_var`; for `i_var=2`,
    # offsets[A] = [x - X_A, y - Y_A, z - Z_A] where A index corresponds to a center.
    offsets = reg_real_pt[: i_var + 1] - coords[:, : i_var + 1]

    # Shape (n_centers, 1): squared distance to each center in preceding
    # dimensions. For `i_var=2`, result[A, 0] = (x - X_A)**2 + (y - Y_A)**2.
    partial_squared_distances = np.sum(np.square(offsets[:, :i_var]), axis=1, keepdims=True)

    # Logarithm of W_{k,i} = c_k exp(-alpha_k d_{k,<i}^2) [sqrt(pi / alpha_k)]**(D - i)
    log_weights = promol.evaluate_log_gaussians(partial_squared_distances)
    log_weights += num_integrated_dims * np.log(pi_over_exps)

    # log(sum_k W_{k,i}), evaluated without forming W_{k,i}.
    log_normalization = logsumexp(log_weights)

    # For pre-evaluated coordinates at boundaries, the log_weights are all -inf.
    if not np.isfinite(log_normalization):
        raise ValueError(
            "Cannot transform a point with previously evaluated coordinates at the boundary. "
        )

    # omega_{k,i} = W_{k,i} / sum_j W_{j,i}.
    normalized_weights = np.exp(log_weights - log_normalization)

    # Individual Gaussian component integrals F_{k,i}(r_i), without their prefactors.
    component_cdfs = promol.integration_gaussian_till_point(offsets, i_var, with_factor=False)

    # theta_i = sum_k omega_{k,i} [2 F_{k,i}(r_i) - 1].
    return np.sum(normalized_weights * (2.0 * component_cdfs - 1.0))


def _root_equation(init_guess, prev_trans_pts, theta_pt, i_var, promol):
    r"""
    Return the residual used to invert one theta coordinate.

    The root finder varies the real-space coordinate :math:`r_i` until

    .. math::

        g(r_i) = \theta_i^{\mathrm{target}} - \theta_i(\mathbf{r}_{<i},r_i) = 0,

    while the preceding real-space coordinates :math:`\mathbf{r}_{<i}` remain fixed. Here,
    :math:`\theta_i^{\mathrm{target}}` is ``theta_pt`` and :math:`\theta_i(\mathbf{r}_{<i},r_i)` is
    the transformed value obtained from ``init_guess``.

    For example, if ``i_var=2``, ``previous_coords`` is ``[x, y]`` and ``trial_coord`` is the
    candidate value of ``z``.

    Parameters
    ----------
    init_guess : float
        Current real-space coordinate proposed by the root finder.
    prev_trans_pts : list[`i_var` - 1]
        Previously inverted real-space coordinates (e.g. x, y for i_var=2).
    theta_pt : float
        The target theta-space  point in [-1, 1] being transformed to the Real space.
    i_var : int
        Index of the coordinate being inverted.
    promol : _PromolParams
        Promolecular Data Class.

    Returns
    -------
    float :
        The difference between `theta_pt` and the transformed point based on
        `init_guess` and `prev_trans_pts`.

    """
    all_points = np.append(prev_trans_pts, init_guess)
    transf_pt = _transform_coordinate(all_points, i_var, promol)
    return theta_pt - transf_pt


def _inverse_coordinate(theta_pt, i_var, transformed, promol):
    r"""Invert coordinate ``i_var`` from theta space to real space.

    Find :math:`r_i` satisfying

    .. math::

        \Theta_i(r_i\mid\mathbf{r}_{<i}) = \theta_i^{\mathrm{target}},

    while keeping the preceding real-space coordinates :math:`\mathbf{r}_{<i}` fixed.

    Parameters
    ----------
    theta_pt : float
        Target coordinate in :math:`[-1, 1]`.
    i_var : int
        Index of the coordinate being inverted.
    transformed : array-like, shape (i_var,)
        Previously inverted real-space coordinates.
    promol : _PromolParams
        Promolecular density parameters.

    Returns
    -------
    float
        Inverted real-space coordinate. Targets ``-1`` and ``1`` map to :math:`-\infty` and
        :math:`+\infty`, respectively.

    Raises
    ------
    ValueError
        If ``theta_pt`` lies outside ``[-1, 1]`` or the analytical bracket does not enclose a root.
    RuntimeError
        If the root finder does not converge.

    """
    if not -1.0 <= theta_pt <= 1.0:
        raise ValueError("theta_pt must lie in [-1, 1].")

    if theta_pt == -1.0:
        return -np.inf
    if theta_pt == 1.0:
        return np.inf

    bracket = _get_inverse_bracket(theta_pt, i_var, promol)

    args = (transformed[:i_var], theta_pt, i_var, promol)
    root_result = root_scalar(
        _root_equation, args=args, method="brentq", bracket=bracket, maxiter=50, xtol=2e-15
    )

    if not root_result.converged:
        raise RuntimeError(f"Inverse transformation did not converge for coordinate {i_var}.")

    return root_result.root


def _pad_coeffs_exps_with_zeros(coeffs, exps):
    r"""Pad Promolecular coefficients and exponents with zero

    Pads the coefficients and exponents of the promolecular density with zeros to make them
    rectangular arrays.

    Parameters
    ----------
    coeffs : list of np.ndarray
        List of arrays containing the coefficients of the promolecular density.
    exps : list of np.ndarray
        List of arrays containing the exponents of the promolecular density.

    Returns
    -------
    padded_coeffs : np.ndarray
        Array containing the padded coefficients of the promolecular density.
    padded_exps : np.ndarray
        Array containing the padded exponents of the promolecular density.
    """
    max_numb_of_gauss = max(len(c) for c in coeffs)
    padded_coeffs = np.zeros((len(coeffs), max_numb_of_gauss), dtype=np.float64)
    padded_exps = np.zeros((len(exps), max_numb_of_gauss), dtype=np.float64)
    for i, c_row in enumerate(coeffs):
        padded_coeffs[i, : len(c_row)] = c_row
    for i, e_row in enumerate(exps):
        padded_exps[i, : len(e_row)] = e_row
    return padded_coeffs, padded_exps


def _get_inverse_bracket(theta_pt, i_var, promol):
    r"""Return a root bracket from the gaussian components."""

    if np.any(promol.c_m < 0.0):
        raise ValueError("Gaussian coefficients must be nonnegative.")

    active = promol.c_m > 0.0
    if not np.any(active):
        raise ValueError("At least one Gaussian coefficient must be positive.")

    active_exponents = promol.e_m[active]
    if np.any(active_exponents <= 0.0):
        raise ValueError("Active Gaussian exponents must be positive.")

    # Coordinate `i_var` of each gaussiancenter (n_centers, 1), broadcast to the shape of `c_m`
    # (n_centers, n_gaussians) for indexing.
    centers = np.broadcast_to(promol.gauss_centers[:, i_var, np.newaxis], promol.c_m.shape)
    # pick out the active Gaussian components.
    centers = centers[active]

    # Characteristic real-space width 1/sqrt(alpha) of each component.
    component_length_scales = 1.0 / np.sqrt(active_exponents)

    # Real-space coordinates with shape (n_active_gaussians,). The inverse coordinate of `theta_pt`
    # for each active Gaussian component.
    component_inverse_coords = centers + erfinv(theta_pt) * component_length_scales

    # widen the bracket by the characteristic length scale of each component to ensure that the root
    # is contained within the bracket when rounding errors are present.
    lower = np.min(component_inverse_coords - component_length_scales)
    upper = np.max(component_inverse_coords + component_length_scales)

    return lower, upper
