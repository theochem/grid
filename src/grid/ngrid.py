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


class Ngrid(Grid):
    r"""
    Grid class for integration of N particle functions.

    The function to integrate must be a function of the form
    f((x1,y1,z1), (x2,y2,z2), ..., (xn, yn, zn)) -> float where (xi, yi, zi) are the coordinates
    of the i-th particle and n is the number of particles. Internally, one Grid is created for
    each particle and the integration is performed by integrating the function over the space spanned
    by the domain union of all the grids.
    """

    def __init__(self, grid_list=None, n=None, **kwargs):
        r"""
        Initialize n particle grid.

        At least one grid must be specified. If only one grid is specified, and a value for n bigger
        than one is specified, the same grid will copied n times and one grid will used for each
        particle. If more than one grid is specified, n will be ignored. In all cases, the function
        to integrate must be a function of the form f((x1,y1,z1), (x2,y2,z2), ..., (xn, yn, zn)) ->
        float where (xi, yi, zi) are the coordinates of the i-th particle and the function must
        depend on a number of particles equal to the number of grids.

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
        if not all(isinstance(grid, Grid) for grid in grid_list):
            raise ValueError("The Grid list must contain only Grid objects")

        self.grid_list = grid_list
        self.n = n

    def integrate(self, callable, **call_kwargs):
        r"""
        Integrate callable on the N particle grid.

        Parameters
        ----------
        callable : callable
            Callable to integrate. It must take a list of N three dimensional tuples as argument
            (one for each particle) and return a float (e.g. a function of the form
            f((x1,y1,z1), (x2,y2,z2)) -> float).
        call_kwargs : dict
            Keyword arguments that will be passed to callable.

        Returns
        -------
        float
            Integral of callable.
        """
        if len(self.grid_list) == 0:
            raise ValueError("The list must contain at least one grid")

        if len(self.grid_list) == 1 and self.n > 1:
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
            Callable to integrate. It must take a list of N three dimensional tuples as argument
            (one for each particle) and return a float (e.g. a function of the form
            f((x1,y1,z1), (x2,y2,z2)) -> float).
        kwcallargs : dict
            Keyword arguments for callable.

        Returns
        -------
        float
            Integral of callable.
        """

        # if there is only one grid, perform the integration using the integrate method of the Grid
        if len(grid_list) == 1:
            vals = callable(grid_list[0].points, **kwcallargs)
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
            points = itertools.product(*[grid.points for grid in grid_list[:-1]])
            # generate all corresponding weights to the generated points
            weights = itertools.product(*[grid.weights for grid in grid_list[:-1]])
            # zip the points and weights together
            data = zip(points, weights)

            integral = 0.0
            for i in data:
                # define an auxiliar function that takes a single argument (a point of the last
                # grid) but uses the other coordinates as fixed parameters i[0] and returns the
                # value of the n particle function at that point (i.e. the value of the n particle
                # function at the point defined by the last grid point and the other coordinates
                # fixed by i[0])
                aux_func = lambda x: callable(*i[0], x, **kwcallargs)

                # calculate the value of the n particle function at each point of the last grid
                vals = aux_func(grid_list[-1].points)

                # Integrate the function over the last grid with all the other coordinates fixed.
                # The result is multiplied by the product of the weights corresponding to the other
                # grids' points (stored in i[1]).
                # This is equivalent to integrating the n particle function over the coordinates of
                # the last particle with the other coordinates fixed.
                integral += grid_list[-1].integrate(vals) * np.prod(i[1])
            return integral
