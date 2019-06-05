# -*- coding: utf-8 -*-
# GRID is a numerical integration library for quantum chemistry.
#
# Copyright (C) 2011-2017 The GRID Development Team
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
#
# --
"""1D Radial integration grid."""
from grid.basegrid import OneDGrid

import numpy as np

from scipy.special import roots_genlaguerre


def generate_onedgrid(npoints, *args):
    """Place holder for general api."""
    ...


def GaussLaguerre(npoints, alpha=0):
    r"""Generate a grid based on generalized Gauss-Laguerre grid.

    Generalizeed Gauss Laguerre grid is defined as:
    .. math::
        \int_{0}^{\infty} x^{\alpha}e^{-x} f(x)dx \approx
        \sum_{i=1}^n w_i f(x_i)

    This integration grid is defined as :
    .. math::
        \int_{0}^{\infty} f(x)dx \approx
        \sum_{i=1}^n \frac{w_i}{x_i^{\alpha} e^{-x_i}} f(x_i)
        = \sum_{i=1}^n  w_i' f(x_i)

    Parameters
    ----------
    npoints : int
        Number of points in the grid
    alpha : float, default to 0, required to be > -1
        parameter alpha value

    Returns
    -------
    OneDGrid
        A grid instance with points and weights, (0, inf)
    """
    if alpha <= -1:
        raise ValueError(f"Alpha need to be bigger than -1, given {alpha}")
    points, weights = roots_genlaguerre(npoints, alpha)
    weights = weights * np.exp(points) * np.power(points, -alpha)
    return OneDGrid(points, weights)


def GaussLegendre(npoints):
    """Generate Gauss-Legendre grid.

    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    OneDGrid
        A grid instance with points and weights, [-1, 1]
    """
    points, weights = np.polynomial.legendre.leggauss(npoints)
    return OneDGrid(points, weights)


def GaussChebyshev(npoints):
    r"""Generate a grid based on Gauss-Chebyshev grid.

    Gauss Chebyshev grid is defined as:
    .. math::
        \int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}}dx \approx \sum_{i=1}^n w_i f(x_i)

    This integration grid is defined as :
    .. math::
        \int_{-1}^{1}f(x)dx\approx\sum_{i=1}^n  w_i \sqrt{1-x_i^2} f(x_i)
        = \sum_{i=1}^n  w_i' f(x_i)
    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    OneDGrid
        A grid instance with points and weights, [-1, 1]
    """
    # points are generated in decreasing order
    # weights are pi/n, all weights are the same
    points, weights = np.polynomial.chebyshev.chebgauss(npoints)
    weights = weights * np.sqrt(1 - np.power(points, 2))
    return OneDGrid(points[::-1], weights)


def HortonLinear(npoints):
    """Generate even space grid.

    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    OneDGrid
        A grid instance with points and weights, [0, inf)
    """
    points = np.arange(npoints)
    weights = np.ones(npoints)
    return OneDGrid(points, weights)
