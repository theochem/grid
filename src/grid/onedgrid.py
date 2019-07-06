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
"""1D Radial integration grid."""


from grid.basegrid import Grid

import numpy as np

from scipy.special import roots_genlaguerre


def GaussLaguerre(npoints, alpha=0):
    r"""Generate 1D grid on [0, inf) interval based on Generalized Gauss-Laguerre quadrature.

    The fundamental definition of Generalized Gauss-Laguerre quadrature is:

    .. math::
        \int_{0}^{\infty} x^\alpha e^{-x} f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

    However, to integrate function :math:`g(x)` over [0, inf), this is re-written as:

    .. math::
        \int_{0}^{\infty} g(x)dx \approx
        \sum_{i=1}^n \frac{w_i}{x_i^\alpha e^{-x_i}} g(x_i) = \sum_{i=1}^n w_i' g(x_i)

    Parameters
    ----------
    npoints : int
        Number of grid points.
    alpha : float, optional
        Value of parameter :math:`alpha` which should be larger than -1.

    Returns
    -------
    Grid
        A 1D grid instance.

    """
    if alpha <= -1:
        raise ValueError(f"Alpha need to be bigger than -1, given {alpha}")
    points, weights = roots_genlaguerre(npoints, alpha)
    weights = weights * np.exp(points) * np.power(points, -alpha)
    return Grid(points, weights)


def GaussLegendre(npoints):
    r"""Generate 1D grid on [-1, 1] interval based on Gauss-Legendre quadrature.

    .. math::
        \int_{-1}^{1} f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

    Parameters
    ----------
    npoints : int
        Number of grid points.

    Returns
    -------
    Grid
        A 1D grid instance.

    """
    points, weights = np.polynomial.legendre.leggauss(npoints)
    return Grid(points, weights)


def GaussChebyshev(npoints):
    r"""Generate 1D grid on [-1, 1] interval based on Gauss-Chebyshev quadrature.

    The fundamental definition of Gauss-Chebyshev quadrature is:

    .. math::
        \int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}} dx \approx \sum_{i=1}^n w_i f(x_i)

    However, to integrate function :math:`g(x)` over [-1, 1], this is re-written as:

    .. math::
        \int_{-1}^{1}g(x) dx \approx \sum_{i=1}^n w_i \sqrt{1-x_i^2} g(x_i)
        = \sum_{i=1}^n w_i' g(x_i)

    Parameters
    ----------
    npoints : int
        Number of grid points.

    Returns
    -------
    Grid
        A 1D grid instance.

    """
    # points are generated in decreasing order
    # weights are pi/n, all weights are the same
    points, weights = np.polynomial.chebyshev.chebgauss(npoints)
    weights = weights * np.sqrt(1 - np.power(points, 2))
    return Grid(points[::-1], weights)


def HortonLinear(npoints):
    """Generate 1D grid on [0, npoints] interval using equally spaced uniform distribution.

    Parameters
    ----------
    npoints : int
        Number of grid points.

    Returns
    -------
    Grid
        A 1D grid instance.

    """
    points = np.arange(npoints)
    weights = np.ones(npoints)
    return Grid(points, weights)
