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
"""Grid for N particle functions."""

import numpy as np
from grid.basegrid import Grid
import itertools
from numbers import Number


class Ngrid(Grid):
    r"""
    Grid class for integration of N argument functions.

    This class is used for integrating functions of N arguments.

    ..math::
        \idotsint f(x_1, x_2, ..., x_N) dx_1 dx_2 ... dx_N

    The function to integrate must have all arguments :math:`\{x_i\}` with the same dimension as the
    points of the corresponding grids (i.e. each of the arguments corresponds to a different grid).
    For example for a function of the form f((x1,y1,z1), (x2,y2)) -> float the first argument must
    be described by a 3D grid and the second argument by a 2D grid.
    """

    def __init__(self, grid_list=None, n=None, **kwargs):
        r"""
        Initialize n particle grid.

        At least one grid must be specified. If only one grid is specified, and a value for n bigger
        than one is specified, the same grid will copied n times and one grid will used for each
        particle. If more than one grid is specified, n will be ignored. In all cases, The function
        to integrate must be a function with all arguments with the same dimension as the grid
        points and must depend on a number of particles equal to the number of grids. For example, a
        function of the form
        f((x1,y1,z1), (x2,y2,z2), ..., (xn, yn, zn)) -> float where (xi, yi, zi) are the coordinates
        of the i-th particle and n is the number of particles.

        Parameters
        ----------
        grid_list : list of Grid
            List of grids, one Grid for each particle.
        n : int
            Number of particles.
        """
        # check that grid_list is defined
        if grid_list is None:
            raise ValueError("The list must be specified")

        # check that grid_list is not empty
        if len(grid_list) == 0:
            raise ValueError("The list must contain at least one grid")

        # check that grid_list contains only Grid objects
        if not all(isinstance(grid, Grid) for grid in grid_list):
            raise ValueError("The Grid list must contain only Grid objects")

        if n is not None:
            # check that n is non negative
            if n < 0:
                raise ValueError("n must be non negative")
            # check that for n > 1, the number of grids is equal to n or 1
            if len(grid_list) > 1 and len(grid_list) != n:
                raise ValueError(
                    "Conflicting values for n and the number of grids. \n"
                    "If n is specified, the number of grids must be equal to n or 1."
                )

        self.grid_list = grid_list
        self.n = n

    def integrate(self, callable, **call_kwargs):
        r"""
        Integrate callable on the N particle grid.

        Parameters
        ----------
        callable : callable
            Callable to integrate. It must take a list of arguments (one for each particle) with
            the same dimension as the grid points and return a float (e.g. a function of the form
            f([x1,y1,z1], [x2,y2,z2]) -> float).
        call_kwargs : dict
            Keyword arguments that will be passed to callable.

        Returns
        -------
        float
            Integral of callable.
        """
        # check that grid_list is not empty
        if len(self.grid_list) == 0:
            raise ValueError("The list must contain at least one grid")

        if len(self.grid_list) == 1 and self.n is not None and self.n > 1:
            return self._n_integrate(self.grid_list * self.n, callable, **call_kwargs)
        else:
            return self._n_integrate(self.grid_list, callable, **call_kwargs)

    def _n_integrate(self, grid_list, callable, **call_kwargs):
        r"""
        Integrate callable on the space spanned domain union of the grids in grid_list.

        Parameters
        ----------
        grid_list : list of Grid
            List of grids for each particle.
        callable : callable
            Callable to integrate. It must take a list of arguments (one for each particle) with
            the same dimension as the grid points and return a float (e.g. a function of the form
            f([x1,y1,z1], [x2,y2,z2]) -> float).
        call_kwargs : dict
            Keyword arguments for callable.

        Returns
        -------
        float
            Integral of callable.
        """

        # if there is only one grid, perform the integration using the integrate method of the Grid
        if len(grid_list) == 1:
            vals = callable(grid_list[0].points, **call_kwargs)
            return grid_list[0].integrate(vals)
        else:
            # The integration is performed by integrating the function over the last grid with all
            # the other coordinates fixed for each possible combination of the other grids' points.
            #
            # Notes:
            # -----
            # - The combination of the other grids' points is generated using a generator so that
            #   the memory usage is kept low.
            # - The last grid is integrated using the integrate method of the Grid class.

            # generate all possible combinations for the first n-1 grids
            data = itertools.product(*[zip(grid.points, grid.weights) for grid in grid_list[:-1]])

            integral = 0.0
            for i in data:
                # Add a dimension to the point (two if it is a number)
                to_point = lambda x: np.array([[x]]) if isinstance(x, Number) else x[None, :]
                # extract points (convert to array, add dim) and corresponding weights combinations
                points_comb = (to_point(j[0]) for j in i)
                weights_comb = np.array([j[1] for j in i])
                # define an auxiliar function that takes a single argument (a point of the last
                # grid) but uses the other coordinates as fixed parameters i[0] and returns the
                # value of the n particle function at that point (i.e. the value of the n particle
                # function at the point defined by the last grid point and the other coordinates
                # fixed by i[0])
                aux_func = lambda x: callable(*points_comb, x, **call_kwargs)

                # calculate the value of the n particle function at each point of the last grid
                vals = aux_func(grid_list[-1].points).flatten()

                # Integrate the function over the last grid with all the other coordinates fixed.
                # The result is multiplied by the product of the weights corresponding to the other
                # grids' points (stored in i[1]).
                # This is equivalent to integrating the n particle function over the coordinates of
                # the last particle with the other coordinates fixed.
                integral += grid_list[-1].integrate(vals) * np.prod(weights_comb)
            return integral
