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
from itertools import islice


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

        Returns
        -------
        MultiDomainGrid
            A MultiDomainGrid object.

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
        self._num_domains = num_domains

    @property
    def num_domains(self):
        """int: The number of integration domains."""
        return self._num_domains if self._num_domains is not None else len(self.grid_list)

    @property
    def size(self):
        """int: the total number of points on the grid."""
        if len(self.grid_list) == 1 and self.num_domains is not None:
            return self.grid_list[0].size ** self.num_domains
        else:
            return np.prod([grid.size for grid in self.grid_list])

    @property
    def weights(self):
        """
        Generator yielding the combined weights of the multi-dimensional grid.

        Because the multi-dimensional grid is formed combinatorially from multiple lower-dimensional
        grids, the combined weights are returned as a generator for efficiency.

        For a MultiDomainGrid formed from two grids, [(x11, y11), (x12, y12) ... (x1n, y1n)] and
        [(x21, y21), (x22, y22) ... (x2m, y2m)], the combined weights are calculated as follows:
        For each combination of points (x1, y1) from the first grid and (x2, y2) from the second
        2D grid, the combined weight `w` is:
            w = w1 * w2
        where `w1` and `w2` are the weights from the individual grids corresponding to
        (x1, y1) and (x2, y2), respectively.

        Yields
        ------
        float
            The product of the weights from each individual grid that make up a single
            point in the multi-dimensional grid.
        """

        if len(self.grid_list) == 1 and self.num_domains is not None:
            # Single grid repeated for multiple domains
            weight_combinations = itertools.product(
                self.grid_list[0].weights, repeat=self.num_domains
            )
        else:
            weight_combinations = itertools.product(*[grid.weights for grid in self.grid_list])

        # Yield the product of weights for each combination
        return (np.prod(combination) for combination in weight_combinations)

    @property
    def points(self):
        """Generator: Combined points of the multi-dimensional grid.

        Due to the combinatorial nature of the grid, the points are returned as a generator. Each
        point is a tuple of the points of the individual grids.

        For a MultiDomainGrid formed from two grids, [(x11, y11), (x12, y12) ... (x1n, y1n)] and
        [(x21, y21), (x22, y22) ... (x2m, y2m)], the combined points are calculated as follows:
        For each combination of points (x1, y1) from the first grid and (x2, y2) from the second
        2D grid, the combined point is a tuple of (x1i, y1i), (x2j, y2j) where (x1, y1) and (x2, y2)
        are the points from the individual grids respectively.
        """
        if len(self.grid_list) == 1 and self.num_domains is not None:
            # Single grid repeated for multiple domains
            points_combinations = itertools.product(
                self.grid_list[0].points, repeat=self.num_domains
            )
        else:
            points_combinations = itertools.product(*[grid.points for grid in self.grid_list])

        return points_combinations

    def integrate(self, integrand_function, non_vectorized=False, integration_chunk_size=6000):
        r"""
        Integrate callable on the N particle grid.

        Parameters
        ----------
        integrand : callable
            Integrand function to integrate. It must take a list of arguments (one for each domain)
            with the same dimension as the grid points used for the corresponding domain and return
            a float (e.g. a function of the form f([x1,y1,z1], [x2,y2,z2]) -> float).
        integration_chunk_size : int, optional
            Number of points to integrate at once. This parameter can be used to control the
            memory usage of the integration. Default is 1000.
        non_vectorized : bool, optional
            Set to True if the integrand is not vectorized. Default is False. If True, the integrand
            will be called for each point of the grid separately without vectorization. This implies
            a slower integration. Use this option if the integrand is not vectorized.
        integration_chunk_size : int, optional
            Number of points to integrate at once. This parameter can be used to control the
            memory usage of the integration. Default is 6000. Values too large may cause memory
            issues and values too small may cause accuracy issues.

        Returns
        -------
        float
            Integral of callable.
        """
        integral_value = 0.0

        if non_vectorized:
            chunked_weights = _chunked_iterator(self.weights, integration_chunk_size)
            # elementwise evaluation of the integrand for each point
            values = (integrand_function(*point) for point in self.points)
            chunked_values = _chunked_iterator(values, integration_chunk_size)

            # calculate the integral in chunks to mitigate accuracy loss of sequential summation
            for chunk_weights, chunk_values in zip(chunked_weights, chunked_values):
                weights_array = np.array(list(chunk_weights))
                values_array = np.array(list(chunk_values))
                integral_value += np.sum(values_array * weights_array)
        else:
            # trivial case of one domain and vectorized integrand (use the grid's integrate method)
            if self.num_domains == 1:
                values = integrand_function(self.grid_list[0].points)
                return self.grid_list[0].integrate(values)

            # find the possible combinations of arguments but the last one
            if len(self.grid_list) == 1:
                pre_weights_combinations = itertools.product(
                    self.grid_list[0].weights, repeat=self.num_domains - 1
                )
                pre_points_combinations = itertools.product(
                    self.grid_list[0].points, repeat=self.num_domains - 1
                )
            else:
                pre_weights_combinations = itertools.product(
                    *[grid.weights for grid in self.grid_list[:-1]]
                )
                pre_points_combinations = itertools.product(
                    *[grid.points for grid in self.grid_list[:-1]]
                )

            # collapse the weights combinations into single weights
            pre_weights = (np.prod(combination) for combination in pre_weights_combinations)

            for pre_points_combination, pre_weight in zip(pre_points_combinations, pre_weights):
                # transform the integrand to a partial integrand with the first N-1 arguments fixed
                partial_integrand = lambda x: integrand_function(*pre_points_combination, x)
                # calculate the values of the partial integrand for all points of the last grid
                values = np.array(partial_integrand(self.grid_list[-1].points))
                integral_value += pre_weight * self.grid_list[-1].integrate(np.array(values))

        return integral_value

    def get_localgrid(self, center, radius):
        raise NotImplementedError(
            "The get_local grid method is not implemented for multi-domain grids."
        )

    def moments(
        self,
        orders: int,
        centers: np.ndarray,
        func_vals: np.ndarray,
        type_mom: str = "cartesian",
        return_orders: bool = False,
    ):
        raise NotImplementedError(
            "The computation of moments is not implemented for multi-domain grids."
        )


def _chunked_iterator(iterator, size):
    """Yield chunks from an iterator."""
    iterator = iter(iterator)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk
