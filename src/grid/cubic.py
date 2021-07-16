import numpy as np
from scipy.interpolate import CubicSpline

from grid.basegrid import Grid, OneDGrid


class CubicGrid(Grid):
    r"""

    Attributes
    ----------

    Methods
    -------


    """
    def __init__(self, oned_grids):
        if not isinstance(oned_grids, list):
            raise TypeError("oned_grid should be of type list.")
        if not np.all([isinstance(grid, OneDGrid) for grid in oned_grids]):
            raise TypeError("Grid in oned_grids should be of type `OneDGrid`.")
        if not len(oned_grids) == 3:
            raise ValueError(
                "There should be three One-Dimensional grids in `oned_grids`."
            )

        self._num_pts = tuple([grid.size for grid in oned_grids])
        self._dim = len(oned_grids)

        # Construct 3D set of points
        points = np.vstack(
            np.meshgrid(
                oned_grids[0].points, oned_grids[1].points, oned_grids[2].points,
                indexing="ij",
            )
        ).reshape(3,-1).T
        weights = np.kron(
            np.kron(oned_grids[0].weights, oned_grids[1].weights), oned_grids[2].weights
        )
        super().__init__(points, weights)

    @property
    def num_pts(self):
        r"""Return number of points in each direction."""
        return self._num_pts

    @property
    def dim(self):
        r"""Return the dimension of the cubic grid."""
        return self._dim

    @property
    def origin(self):
        r"""Return the origin of the coordinate system."""
        # Bottom, Left-Most, Down-most point of the Cubic Grid in [-1, 1]^3.
        return self.points[0]

    def integrate(self, *value_arrays, is_zero_boundary=False, type="Rectangle"):
        r"""
        Integrate any real-valued function on Euclidean space.

        Parameters
        ----------
        *value_arrays : (np.ndarray(N, dtype=float),)
            One or multiple value array to integrate, these are usally the functions evaluated
            at the points of the grid.
        is_zero_boundary : bool, optional
            If the function being integrated is zero on the boundary, then this changes the
            weight used.
        type : str, optionla
            The type of weights used, either "Rectangle", or "Fourier".

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
        return super().integrate(*value_arrays)

    def interpolate_function(
            self, real_pt, func_values, use_log=False, nu_x=0, nu_y=0, nu_z=0,
    ):
        r"""
        Interpolate function at a point.

        Parameters
        ----------
        real_pt : np.ndarray(3,)
            Point in :math:`\mathbb{R}^3` that needs to be interpolated.
        func_values : np.ndarray(N,)
            Function values at each point of the grid `points`.
        use_log : bool
            If true, then logarithm is applied before interpolating to the function values,
            including  `func_values`.
        nu_x : int
            If zero, then the function in x-direction is interpolated.
            If greater than zero, then the "nu_x"th-order derivative in the x-direction is
            interpolated.
        nu_y : int
            If zero, then the function in x-direction is interpolated.
            If greater than zero, then the "nu_x"th-order derivative in the y-direction is
            interpolated.
        nu_z : int
            If zero, then the function in x-direction is interpolated.
            If greater than zero, then the "nu_x"th-order derivative in the z-direction is
            interpolated.

        Returns
        -------
        float :
            If nu is 0: Returns the interpolated of a function at a real point.
            If nu is 1: Returns the interpolated derivative of a function at a real point.

        """
        # TODO: Ask about use_log and derivative.
        if use_log:
            func_values = np.log(func_values)

        # Interpolate the Z-Axis based on x, y coordinates in grid.
        def z_spline(z, x_index, y_index, nu_z=nu_z):
            # x_index, y_index is assumed to be in the grid while z is not assumed.
            # Get smallest and largest index for selecting func vals on this specific z-slice.
            # The `1` and `self.num_puts[2] - 2` is needed because I don't want the boundary.
            small_index = self._indices_to_index((x_index, y_index, 1))
            large_index = self._indices_to_index(
                (x_index, y_index, self.num_pts[2] - 2)
            )
            val = CubicSpline(
                self.points[small_index:large_index, 2],
                func_values[small_index:large_index],
            )(z, nu_z)

            return val

        # Interpolate the Y-Axis based on x coordinate in grid.
        def y_splines(y, x_index, z, nu_y=nu_y):
            # The `1` and `self.num_puts[1] - 2` is needed because I don't want the boundary.
            # Assumes x_index is in the grid while y, z may not be.
            val = CubicSpline(
                self.points[np.arange(1, self.num_pts[1] - 2) * self.num_pts[2], 1],
                [
                    z_spline(z, x_index, y_index, nu_z)
                    for y_index in range(1, self.num_pts[1] - 2)
                ],
            )(y, nu_y)

            return val

        # Interpolate the X-Axis.
        def x_spline(x, y, z, nu_x):
            # x, y, z may not be in the grid.
            val = CubicSpline(
                self.points[np.arange(1, self.num_pts[0] - 2) * self.num_pts[1] * self.num_pts[2], 0],
                [y_splines(y, x_index, z, nu_y) for x_index in range(1, self.num_pts[0] - 2)],
            )(x, nu_x)

            return val

        print(sum([nu_x == 0, nu_y == 0, nu_z == 0]) == 2, 1 in (nu_x, nu_y, nu_z))
        # Only consider taking the derivative in only one direction.
        one_var_deriv = sum([nu_x == 0, nu_y == 0, nu_z == 0]) == 2 and 1 in (nu_x, nu_y, nu_z)
        interpolated = x_spline(real_pt[0], real_pt[1], real_pt[2], nu_x)
        if use_log and (nu_x, nu_y, nu_z) == (0, 0, 0):
            return np.exp(interpolated)
        elif use_log and one_var_deriv:
            interpolated_func = self.interpolate_function(real_pt, func_values, use_log=False)
            return interpolated * np.exp(interpolated_func)
        if use_log and sum([nu_x == 0, nu_y == 0, nu_z == 0]) == 2:
            if nu_x > 0:
                derivs = [
                    self.interpolate_function(real_pt, func_values, use_log=False,
                                              nu_x=i, nu_y=0, nu_z=0) for i in range(0, nu_x)
                ]
            elif nu_y > 0:
                derivs = [
                    self.interpolate_function(real_pt, func_values, use_log=False,
                                              nu_x=0, nu_y=i, nu_z=0) for i in range(0, nu_y)
                ]
            else:
                derivs = [
                    self.interpolate_function(real_pt, func_values, use_log=False,
                                              nu_x=0, nu_y=0, nu_z=i) for i in range(0, nu_z)
                ]
            return derivs[0] * Bell_polynomial(*derivs[1:])
        return interpolated

    def _indices_to_index(self, indices):
        r"""
        Convert Indices to Index, ie (i, j, k) to a index/integer m corresponding to grid point.

        Parameters
        ----------
        indices : (int, int, int)
            The ith, jth, kth position of the grid point.

        Returns
        -------
        index : int
            Index of the grid point.

        """
        n_1d, n_2d = self.num_pts[2], self.num_pts[1] * self.num_pts[2]
        index = n_2d * indices[0] + n_1d * indices[1] + indices[2]
        return index


class UniformCubicGrid(CubicGrid):
    def __init__(self, l_bnd, u_bnd, step_sizes):
        """
        Construct the UniformCubicGrid object.

        Parameters
        ----------
        smallest_pnt : ndarray(3,)
            The smallest point in each axis in the 3D cubic grid.
        largest_pnt : ndarray(3,)
            The largest point in each axis in the 3D cubic grid.
        step_sizes : ndarray(3,)
            The step-size between two consecutive points in each axis in the 3D cubic grid.

        """
        if not isinstance(l_bnd, np.ndarray):
            raise TypeError("Argument l_bnd should be a numpy array.")
        if not isinstance(u_bnd, np.ndarray):
            raise TypeError("Argument u_bnd should be a numpy array.")
        if not isinstance(step_size, np.ndarray):
            raise TypeError("The argument step_size should be a numpy array.")
        if l_bnd.nsize != 3 or u_bnd.nsize != 3 or step_sizes.nsize != 3:
            raise ValueError("The arguments, l_bnd, u_bnd, step_sizes should all have size 3.")
        if np.any(u_bnd >= l_bnd):
            raise ValueError("In each coordinate, u_bnd should be greater than l_bnd.")
        if np.any(step_sizes <= 0):
            raise ValueError("In each coordinate, step_sizes should be positive.")

        r"""
        Input is either
        
        (oned_Grid, oned_grid, oned_grid), Here only points on each direction is only stored.
        Have a function that returns the poitns if it is needed, i.e. utilizing vectorization
            for evaluating a function.
        
        or 
        
        oned_grid, dim=3 so that it results in (oned_grid, oned_grid, oned_grid).
        Here only points on each direction is only stored.  
        Have a function that returns the poitns if it is needed, i.e. utilizing vectorization
            for evaluating a function.
        
        or 
        
        Grid isn't stored at all, only l_bnd, u_bnd and step_size.
        Have a function that returns the poitns if it is needed, i.e. utilizing vectorization
            for evaluating a function.
        
        """

        # compute points in each direction/dimension
        points_1 = np.arange(l_bnd[0], u_bnd[0] + step_sizes[0], step_sizes[0])
        points_2 = np.arange(l_bnd[1], u_bnd[1] + step_sizes[1], step_sizes[1])
        points_3 = np.arange(l_bnd[2], u_bnd[2] + step_sizes[2], step_sizes[2])
        npts_1 = points_1.nsize
        npts_2 = points_2.nsize
        npts_3 = points_3.nsize
        self._num_pts = (npts_1, npts_2, npts_3)
        points = np.zeros((npts_1 * npts_2 * npts_3, 3))
        # assign x, y & z coordinates
        points[:, 0] = np.repeat(points_1, npts_2 * npts_3)
        points[:, 1] = np.tile(np.repeat(points_2, npts_1), npts_3)
        points[:, 2] = np.tile(points_3, npts_1 * npts_2)
        self._points = np.array(points)
        self._step_sizes = step_size

    @property
    def points(self):
        """Return cubic grid points."""
        return self._points

    @property
    def step_sizes(self):
        """Return the step size between to consecutive points."""
        return self._step_sizes

    @property
    def num_pts(self):
        r"""Return the number of points (N_x, N_y, N_z) in each direction of the Cubic Grid."""
        return self._num_pts

    @property
    def origin(self):
        r"""Return the origin of the coordinate system."""
        # Bottom, Left-Most, Down-most point of the Cubic Grid.
        return self._points[0]

    def __len__(self):
        """Return the number of grid points."""
        return self._points.nsize

    def evaluate_function_on_grid(self, f):
        r"""
        Evaluates the function f on each point of the cubic grid.

        Useful when vectorization can't be performed or memory of the grid is large.
        Slow, as each point of the grid is iterated.

        Parameters
        ----------
        f : callable(ndarray(3,))
            The function to be evaluated on the grid that takes pts as input.

        Returns
        -------
        ndarray(N,)
            Function evaluated on N points of the cubic grid.

        """
        output = []
        for pt in self._points:
            output.append(f(pt))
        return output

    def integrate(self, arr):
        r"""Compute the integral of a function evaluated on the grid points based on Riemann sums.

        .. math:: \int\int\int f(x, y, z) dx dy dz

        where :math:'f(r)' is the integrand.

        Parameters
        ----------
        arr : ndarray
            The integrand evaluated on the grid points.

        Returns
        -------
        value : float
            The value of integral.

        """
        if arr.shape != (len(self),):
            raise ValueError("Argument arr should have ({0},) shape.".format(len(self)))
        value = np.power(self._step, 3) * np.sum(arr)
        return value

    def interpolate_function(
            self, real_pt, func_values, oned_grids, use_log=False, nu=0
    ):
        r"""
        Interpolate function at a point.

        Parameters
        ----------
        real_pt : np.ndarray(3,)
            Point in :math:`\mathbb{R}^3` that needs to be interpolated.
        func_values : np.ndarray(N,)
            Function values at each point of the grid `points`.
        oned_grids = list(3,)
            List Containing Three One-Dimensional grid corresponding to x, y, z direction.
        use_log : bool
            If true, then logarithm is applied before interpolating to the function values,
            including  `func_values`.
        nu : int
            If zero, then the function is interpolated.
            If one, then the derivative is interpolated.

        Returns
        -------
        float :
            If nu is 0: Returns the interpolated of a function at a real point.
            If nu is 1: Returns the interpolated derivative of a function at a real point.

        """
        # TODO: Should oned_grids be stored as class attribute when only this method requires it.
        # TODO: Ask about use_log and derivative.
        if nu not in (0, 1):
            raise ValueError("The parameter nu %d is either zero or one " % nu)
        if use_log:
            func_values = np.log(func_values)

        # Interpolate the Z-Axis based on x, y coordinates in grid.
        def z_spline(z, x_index, y_index):
            # x_index, y_index is assumed to be in the grid while z is not assumed.
            # Get smallest and largest index for selecting func vals on this specific z-slice.
            # The `1` and `self.num_puts[2] - 2` is needed because I don't want the boundary.
            small_index = self._indices_to_index((x_index, y_index, 1))
            large_index = self._indices_to_index(
                (x_index, y_index, self.num_pts[2] - 2)
            )
            val = CubicSpline(
                oned_grids[2].points[1: self.num_pts[2] - 2],
                func_values[small_index:large_index],
            )(z, nu)

            return val

        # Interpolate the Y-Axis based on x coordinate in grid.
        def y_splines(y, x_index, z):
            # The `1` and `self.num_puts[1] - 2` is needed because I don't want the boundary.
            # Assumes x_index is in the grid while y, z may not be.
            val = CubicSpline(
                oned_grids[1].points[1: self.num_pts[2] - 2],
                [
                    z_spline(z, x_index, y_index)
                    for y_index in range(1, self.num_pts[1] - 2)
                ],
            )(y, nu)

            return val

        # Interpolate the X-Axis.
        def x_spline(x, y, z):
            # x, y, z may not be in the grid.
            val = CubicSpline(
                oned_grids[0].points[1: self.num_pts[2] - 2],
                [y_splines(y, x_index, z) for x_index in range(1, self.num_pts[0] - 2)],
            )(x, nu)

            return val

        interpolated = x_spline(theta_pt[0], theta_pt[1], theta_pt[2])
        return interpolated

    def _closest_point_theta(self, point, which="closest"):
        r"""
        Return closest index of the grid point to a theta point.

        Imagine a point inside a small sub-cube. If `closest` is selected, it will
        pick the corner in the sub-cube that is closest to that point.
        if `origin` is selected, it will pick the corner that is the bottom,
        left-most, down-most in the sub-cube.

        Parameters
        ----------
        point : np.ndarray(3,)
            Point in :math:`[-1,1]^3`.
        which : str
            If "closest", returns the closest index of the grid point.
            If "origin", return the bottom, left-most, down-most closest index of the grid point.

        Returns
        -------
        index : int
            Index of the point in `points` closest to the grid point.

        """
        # Test point is in the right domain.
        if not np.all((-1.0 <= point) & (point <= 1.0)):
            raise ValueError("Point should be in [-1, 1]^3.")
        coord = np.array([(point[i] - self.origin[i]) / self.stepsizes[i] for i in range(3)])

        print("Before coord ", coord)
        if which == "origin":
            # Round to smallest integer.
            coord = np.floor(coord)
        elif which == "closest":
            # Round to nearest integer.
            coord = np.rint(coord)
        else:
            raise TypeError("`which` parameter was not the standard options.")

        # Convert indices (i, j, k) into index.
        index = self.num_pts[2] * self.num_pts[1] * coord[0] + self.num_pts[2] * coord[1] + coord[2]

        return int(index)

    def _index_to_indices(self, index):
        r"""
        Convert Index to Indices, ie integer m to (i, j, k) position of the Cubic Grid.

        Cubic Grid has shape (N_x, N_y, N_z) where N_x is the number of points
        in the x-direction, etc.  Then 0 <= i <= N_x - 1, 0 <= j <= N_y - 1, etc.

        Parameters
        ----------
        index : int
            Index of the grid point.

        Returns
        -------
        indices : (int, int, int)
            The ith, jth, kth position of the grid point.

        """
        assert index >= 0, "Index should be positive. %r" % index
        n_1d, n_2d = self.num_pts[2], self.num_pts[1] * self.num_pts[2]
        i = index // n_2d
        j = (index - n_2d * i) // n_1d
        k = index - n_2d * i - n_1d * j
        return i, j, k

    def _indices_to_index(self, indices):
        r"""
        Convert Indices to Index, ie (i, j, k) to a index/integer m.

        Parameters
        ----------
        indices : (int, int, int)
            The ith, jth, kth position of the grid point.

        Returns
        -------
        index : int
            Index of the grid point.

        """
        n_1d, n_2d = self.num_pts[2], self.num_pts[1] * self.num_pts[2]
        index = n_2d * indices[0] + n_1d * indices[1] + indices[2]
        return index
