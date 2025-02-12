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


class MultiDomainGrid(Grid):
    r"""
    Grid class for integrating functions of multiple variables, each defined on a different grid.

    This class facilitates the numerical integration of functions with :math:`N` arguments
    over a corresponding :math:`N`-dimensional domain using grid-based methods.

    .. math::
        \int \cdots \int f(x_1, x_2, \ldots, x_N) \, dx_1 dx_2 \cdots dx_N

    The function to integrate must accept arguments :math:`\{x_i\}` that correspond to
    the dimensions (point-wise) of the respective grids. Specifically:

        - Each argument :math:`x_i` corresponds to a different grid.
        - The dimensionality of each argument must match the dimensionality of a point of its
          associated grid.

    For example:

    - For a function of the form :code:`f([x1, y1, z1], [x2, y2]) -> float`,
      the first argument corresponds to a 3-dimensional grid, and the second argument
      corresponds to a 2-dimensional grid.


    The function to integrate must have all arguments :math:`\{x_i\}` with the same dimension as the
    points of the corresponding grids (i.e. each of the arguments corresponds to a different grid).
    For example for a function of the form f((x1,y1,z1), (x2,y2)) -> float the first argument must
    be described by a 3D grid and the second argument by a 2D grid.
    """

    def __init__(self, grid_list=None, num_domains=None):
        r"""
        Initialize the MultiDomainGrid grid.

        Parameters
        ----------
        grid_list : list of Grid
            A list of Grid objects, where each Grid corresponds to a separate argument
            (integration domain) of the function to be integrated.
            - At least one grid must be specified.
            - The number of elements in `grid_list` should match the number of arguments
              in the target function.
        num_domains : int, optional
            The number of integration domains.
            - This parameter is optional and can only be specified when `grid_list` contains
              exactly one grid.
            - It must be a positive integer greater than 1.
            - If specified, the function to integrate is considered to have `num_domains` arguments,
              all defined over the same grid (i.e., the same set of points is used for each
              argument).
        """
        if not isinstance(grid_list, list):
            raise ValueError("The grid list must be defined")
        if len(grid_list) == 0:
            raise ValueError("The list must contain at least one grid")
        if not all(isinstance(grid, Grid) for grid in grid_list):
            raise ValueError("Invalid grid list. The list must contain only Grid objects")
        if num_domains is not None:
            if len(grid_list) != 1:
                raise ValueError("The number of grids must be equal to 1 if grids_num is specified")
            if not isinstance(num_domains, int) or num_domains < 1:
                raise ValueError("grids_num must be a positive integer bigger than 1")

        self.grid_list = grid_list
        self.num_domains = num_domains

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

        if len(self.grid_list) == 1 and self.num_domains is not None and self.num_domains > 1:
            return self._n_integrate(self.grid_list * self.num_domains, callable, **call_kwargs)
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
