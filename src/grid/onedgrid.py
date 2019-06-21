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
from grid.grid import Grid

import numpy as np

from scipy.special import roots_chebyu, roots_genlaguerre


def generate_onedgrid(npoints, *args):
    """Place holder for general api."""
    ...


def GaussLaguerre(npoints, alpha=0):
    """Generate Gauss-Laguerre grid.

    Parameters
    ----------
    npoints : int
        Number of points in the grid
    alpha : int, default to 0, required to be > -1
        parameter alpha value

    Returns
    -------
    Grid
        A grid instance with points and weights
    """
    if alpha <= -1:
        raise ValueError(f"Alpha need to be bigger than -1, given {alpha}")
    points, weights = roots_genlaguerre(npoints, alpha)
    return Grid(points, weights)


def GaussLegendre(npoints):
    """Generate Gauss-Legendre grid.

    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    Grid
        A grid instance with points and weights
    """
    points, weights = np.polynomial.legendre.leggauss(npoints)
    return Grid(points, weights)


def GaussChebyshev(npoints):
    """Generate Gauss-Legendre grid.

    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    Grid
        A grid instance with points and weights
    """
    points, weights = np.polynomial.chebyshev.chebgauss(npoints)
    return Grid(points, weights)


def GaussChebyshevType2(npoints):
    """Generate Gauss-Chebyshev (type 2) grid.

    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    Grid
        An grid instance with points and weights
    """
    points, weights = roots_chebyu(npoints)
    return Grid(points, weights)


def GaussChebyshevLobatto(npoints):
    """Generate Gauss-Chebyshev-Lobatto grid.

    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    Grid
        An grid instance with points and weights
    """
    idx = np.arange(npoints)
    weights = np.ones(npoints)

    idx = (idx * np.pi) / (npoints - 1)

    points = np.cos(idx)
    points = np.sort(points)

    weights *= np.pi / (npoints - 1)
    weights[0] /= 2
    weights[npoints - 1] = weights[0]

    return Grid(points, weights)


def RectangleRuleSineEndPoints(npoints):
    """Generate Rectangle rule for sine series with end points.

    The original range of this rule is [0:1], the last part is 
    a linear transformation for passing from [0:1] to [-1:1]

    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    Grid
        An grid instance with points and weights
    """
    idx = np.arange(npoints) + 1
    points = idx / (npoints + 1)

    weights = np.zeros(npoints)

    index_m = np.arange(npoints) + 1

    for i in range(0, npoints):
        elements = np.zeros(npoints)
        elements = np.sin(index_m * np.pi * points[i])
        elements = elements * (1 - np.cos(index_m * np.pi)) / (index_m * np.pi)

        weights[i] = (2 / (npoints + 1)) * np.sum(elements)

    points = 2 * points - 1

    return Grid(points, weights)


def RectangleRuleSine(npoints):
    """Generate Rectangle rule for sine series without end points.

    The original range of this rule is [0:1], the last part is
    a linear transformation for passing from [0:1] to [-1:1]

    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    Grid
        An grid instance with points and weights
    """
    idx = np.arange(npoints) + 1
    points = (2 * idx - 1) / (2 * npoints)

    weights = np.zeros(npoints)

    index_m = np.arange(npoints - 1) + 1

    for i in range(0, npoints):
        elements = np.zeros(npoints - 1)
        elements = np.sin(index_m * np.pi * points[i])
        elements *= np.sin(index_m * np.pi / 2) ** 2
        elements /= index_m

        weights[i] = (4 / (npoints * np.pi)) * np.sum(elements)

        weights[i] += (
            (2 / (npoints * np.pi ** 2))
            * np.sin(npoints * np.pi * points[i])
            * np.sin(npoints * np.pi / 2) ** 2
        )

    points = 2 * points - 1

    return Grid(points, weights)


def TanhSinh(npoints, delta):
    """Generate Tanh-Sinh rule.

    The ranges is [-1:1] you need proporcionate
    a delta value for this rule.

    Parameters
    ----------
    npoints : int
        Number of points in the grid, this value must be odd.

    delta : float
        A parameter of size.

    Returns
    -------
    Grid
        An grid instance with points and weights.
    """
    if npoints % 2 == 0:
        raise ValueError("npoints must be odd, given {npoints}")

    jmin = (int)(1 - npoints) / 2

    points = np.zeros(npoints)
    weights = np.zeros(npoints)

    for i in range(0, npoints):
        j = jmin + i
        arg = np.pi * np.sinh(j * delta) / 2

        points[i] = np.tanh(arg)

        weights[i] = np.pi * delta * np.cosh(j * delta) * 0.5
        weights[i] /= np.cosh(arg) ** 2

    return Grid(points, weights)
