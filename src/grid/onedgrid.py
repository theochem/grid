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
"""1D integration grid."""


from grid.basegrid import OneDGrid

import numpy as np

from scipy.special import roots_chebyu, roots_genlaguerre


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
    OneDGrid
        A 1D grid instance.

    """
    if alpha <= -1:
        raise ValueError(f"Alpha need to be bigger than -1, given {alpha}")
    points, weights = roots_genlaguerre(npoints, alpha)
    weights = weights * np.exp(points) * np.power(points, -alpha)
    return OneDGrid(points, weights, (0, np.inf))


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
    OneDGrid
        A 1D grid instance.

    """
    points, weights = np.polynomial.legendre.leggauss(npoints)
    return OneDGrid(points, weights, (-1, 1))


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
    OneDGrid
        A 1D grid instance.

    """
    # points are generated in decreasing order
    # weights are pi/n, all weights are the same
    points, weights = np.polynomial.chebyshev.chebgauss(npoints)
    weights = weights * np.sqrt(1 - np.power(points, 2))
    return OneDGrid(points[::-1], weights, (-1, 1))


def HortonLinear(npoints):
    """Generate 1D grid on [0, npoints] interval using equally spaced uniform distribution.

    Parameters
    ----------
    npoints : int
        Number of grid points.

    Returns
    -------
    OneDGrid
        A 1D grid instance.

    """
    points = np.arange(npoints)
    weights = np.ones(npoints)
    return OneDGrid(points, weights)


def GaussChebyshevType2(npoints):
    r"""Generate 1D grid on [-1, 1] interval based on Gauss-Chebyshev 2nd kind.

    The fundamental definition of Gauss-Chebyshev quadrature is:

    .. math::
        \int_{-1}^{1} \sqrt{1-x^2} f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

    However, to integrate function :math:`g(x)` over [-1, 1], this is re-written as:

    .. math::
        \int_{-1}^{1}g(x) dx \approx \sum_{i=1}^n \frac{w_i}{\sqrt{1-x_i^2}} f(x_i)
        = \sum_{i=1}^n w_i' f(x_i)

    Where

    .. math::
        x_i = \cos\left( \frac{i}{n+1} \pi \right)

    and the weights

    .. math::
        w_i = \frac{\pi}{n+1} \sin^2 \left( \frac{i}{n+1} \pi \right)

    Parameters
    ----------
    npoints : int
        Number of grid points.

    Returns
    -------
    OneDGrid
        A 1D grid instance.

    """
    if npoints < 1:
        raise ValueError("npoints must be greater that one, given {npoints}")
    points, weights = roots_chebyu(npoints)
    weights /= np.sqrt(1 - np.power(points, 2))
    return OneDGrid(points, weights, (-1, 1))


def GaussChebyshevLobatto(npoints):
    r"""Generate 1D grid on [-1, 1] interval based on Gauss-Chebyshev-Lobatto.

    The definition of Gauss-Chebyshev-Lobato quadrature is:

    .. math::
        \int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}} dx
        \approx \sum_{i=1}^n w_i f(x_i)

    However, to integrate function :math:`g(x)` over [-1, 1], this is re-written as:

    .. math::
        \int_{-1}^{1}g(x) dx \approx \sum_{i=1}^n w_i \sqrt{1-x_i^2} f(x_i)
        = \sum_{i=1}^n w_i' f(x_i)

    Where

    .. math::
        x_i = \cos\left( \frac{(i-1)\pi}{n-1} \right)

    And the weights

    .. math::
        w_{1} = w_{n} = \frac{\pi}{2(n-1)}

    And the internal weights

    .. math::
        w_{i\neq 1,n} = \frac{\pi}{n-1}


    Parameters
    ----------
    npoints : int
        Number of points in the grid

    Returns
    -------
    OneDGrid
        A 1D grid instance.

    """
    if npoints <= 1:
        raise ValueError("npoints must be greater that one, given {npoints}")
    idx = np.arange(npoints)
    weights = np.ones(npoints)

    idx = (idx * np.pi) / (npoints - 1)

    points = np.cos(idx)
    points = points[::-1]

    weights *= np.pi / (npoints - 1)
    weights *= np.sqrt(1 - np.power(points, 2))

    weights[0] /= 2
    weights[npoints - 1] = weights[0]

    return OneDGrid(points, weights, (-1, 1))


def Trapezoidal(npoints):
    r"""Generate 1D grid on [-1:1] interval based on trapezoidal rule.

    The fundamental definition of this rule is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

    Where

    .. math::
        x_i = -1 + 2 \left(\frac{i-1}{n-1}\right)

    The weights

    .. math::
        w_{i\neq 1,n} = \frac{2}{n}

    and

    .. math::
        w_1 = w_n = \frac{1}{n}


    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.

    """
    if npoints <= 1:
        raise ValueError("npoints must be greater that one, given {npoints}")
    idx = np.arange(npoints)
    points = -1 + (2 * idx / (npoints - 1))

    weights = 2 * np.ones(npoints) / (npoints - 1)

    weights[0] = 1 / (npoints - 1)
    weights[npoints - 1] = weights[0]

    return OneDGrid(points, weights, (-1, 1))


def RectangleRuleSineEndPoints(npoints):
    r"""Generate 1D grid on [-1:1] interval based on rectangle rule.

    The fundamental definition of this quadrature is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

    The range of integration can be modified by :math: `q = 2 x - 1`.

    .. math::
        2 \int_{0}^{1} f(x) dx = \int_{-1}^{1} f(q) dq

    Where

    .. math::
        x_i = \frac{i}{n+1}

    And the weights

    .. math::
        w_i = \frac{2}{n+1} \sum_{m=1}^n
                \frac{\sin(m \pi x_i)(1-\cos(m \pi))}{m \pi}


    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.

    """
    if npoints <= 1:
        raise ValueError("npoints must be greater that one, given {npoints}")
    idx = np.arange(npoints) + 1
    points = idx / (npoints + 1)

    weights = np.zeros(npoints)

    index_m = np.arange(npoints) + 1

    for i in range(0, npoints):
        elements = np.zeros(npoints)
        elements = np.sin(index_m * np.pi * points[i])
        elements *= (1 - np.cos(index_m * np.pi)) / (index_m * np.pi)

        weights[i] = (2 / (npoints + 1)) * np.sum(elements)

    points = 2 * points - 1
    weights *= 2

    return OneDGrid(points, weights, (-1, 1))


def RectangleRuleSine(npoints):
    r"""Generate 1D grid on (0:1) interval based on rectangle rule.

    The fundamental definition of this quadrature is:

    .. math::
        \int_{0}^{1} f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

    The range of integration can be modified by :math: `q = 2 x - 1`.

    .. math::
        2 \int_{0}^{1} f(x) dx = \int_{-1}^{1} f(q) dq

    Where

    .. math::
        x_i = \frac{2 i - 1}{2 n}

    And the weights

    .. math::
        w_i = \frac{2}{n^2 \pi} \sin(n\pi x_i) \sin^2(n\pi /2)
            + \frac{4}{n \pi}\sum_{m=1}^{n-1}
            \frac{\sin(m \pi x_i)\sin^2(m\pi /2)}
                {m}

    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.

    """
    if npoints <= 1:
        raise ValueError("npoints must be greater that one, given {npoints}")
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
    weights *= 2

    return OneDGrid(points, weights, (-1, 1))


def TanhSinh(npoints, delta=0.1):
    r"""Generate 1D grid on [-1,1] interval based on Tanh-Sinh rule.

    The fundamental definition is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx
        \sum_{i=-\frac{1}{2}(n-1)}^{\frac{1}{2}(n-1)} w_i f(x_i)

    Where

    .. math::
        x_i = \tanh\left( \frac{\pi}{2} \sinh(i\delta) \right)

    And the weights

    .. math::
        w_i = \frac{\frac{\pi}{2}\delta \cosh(i\delta)}
        {\cosh^2(\frac{\pi}{2}\sinh(i\delta))}


    Parameters
    ----------
    npoints : int
        Number of points in the grid, this value must be odd.

    delta : float
        This values is a parameter :math:`\delta`, is related with the size.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    if npoints <= 1:
        raise ValueError("npoints must be greater that one, given {npoints}")
    if npoints % 2 == 0:
        raise ValueError("npoints must be odd, given {npoints}")

    jmin = int((1 - npoints) / 2)

    points = np.zeros(npoints)
    weights = np.zeros(npoints)

    for i in range(0, npoints):
        j = jmin + i
        arg = np.pi * np.sinh(j * delta) / 2

        points[i] = np.tanh(arg)

        weights[i] = np.pi * delta * np.cosh(j * delta) * 0.5
        weights[i] /= np.cosh(arg) ** 2

    return OneDGrid(points, weights, (-1, 1))


def Simpson(npoints):
    r"""Generate 1D grid on [-1:1] interval based on Simpson rule.

    The fundamental definition of this rule is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

    Where

    .. math::
        x_i = -1 + 2 \left(\frac{i-1}{n-1}\right)


    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.

    """
    if npoints <= 1:
        raise ValueError("npoints must be greater that one, given {npoints}")
    if npoints % 2 == 0:
        raise ValueError("npoints must be odd, given {npoints}")
    idx = np.arange(npoints)
    points = -1 + (2 * idx / (npoints - 1))

    weights = 2 * np.ones(npoints) / (3 * (npoints - 1))

    for i in range(1, npoints - 1):
        if i % 2 == 0:
            weights[i] *= 2
        else:
            weights[i] *= 4

    return OneDGrid(points, weights, (-1, 1))


def MidPoint(npoints):
    r"""Generate 1D grid on [-1,1] interval based on Mid Point rule.

    The fundamental definition is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx
        \sum_{i=1}^n w_i f(x_i)

    Where

    .. math::
        x_i = (2*i + 1)/n - 1

    And the weights

    .. math::
        w_i = 2/n


    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    if npoints <= 1:
        raise ValueError("npoints must be greater that one, given {npoints}")
    points = np.zeros(npoints)
    weights = np.ones(npoints)

    idx = np.arange(npoints)

    weights *= 2 / npoints

    points = -1 + (2 * idx + 1) / npoints

    return OneDGrid(points, weights, (-1, 1))


def ClenshawCurtis(npoints):
    r"""Generate 1D grid on [-1,1] interval based on Clenshaw-Curtis method.

    The fundamental definition is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx
        \sum_{i=1}^n w_i f(x_i)


    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    if npoints < 1:
        raise ValueError("npoints must be greater that one, given {npoints}")
    if npoints == 1:
        points = np.zeros(npoints)
        weights = np.zeros(npoints)

        weights[0] = 2.0
    else:
        theta = np.zeros(npoints)

        for i in range(0, npoints):
            theta[i] = (npoints - 1 - i) * np.pi / (npoints - 1)

        points = np.cos(theta)
        weights = np.zeros(npoints)

        jmed = (npoints - 1) // 2

        for i in range(0, npoints):
            weights[i] = 1

            for j in range(0, jmed):
                if (2 * (j + 1)) == (npoints - 1):
                    b = 1
                else:
                    b = 2
                weights[i] = weights[i] - b * np.cos(2 * (j + 1) * theta[i]) / (
                    4 * j * (j + 2) + 3
                )

        for i in range(1, npoints - 1):
            weights[i] = 2 * weights[i] / (npoints - 1)

        weights[0] = weights[0] / (npoints - 1)
        weights[npoints - 1] = weights[npoints - 1] / (npoints - 1)

    return OneDGrid(points, weights, (-1, 1))


def FejerFirst(npoints):
    r"""Generate 1D grid on [-1,1] interval based on Fejer-1 method.

    The fundamental definition is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx
        \sum_{i=1}^n w_i f(x_i)


    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    if npoints <= 1:
        raise ValueError("npoints must be greater that one, given {npoints}")

    theta = np.pi * (2 * np.arange(npoints) + 1) / (2 * npoints)

    points = np.cos(theta)
    weights = np.zeros(npoints)

    nsum = npoints // 2

    for k in range(0, npoints):
        serie = 0

        for m in range(1, nsum):
            serie += np.cos(2 * m * theta[k]) / (4 * m ** 2 - 1)

        serie = 1 - 2 * serie

        weights[k] = (2 / npoints) * serie

    points = points[::-1]
    weights = weights[::-1]

    return OneDGrid(points, weights, (-1, 1))


def FejerSecond(npoints):
    r"""Generate 1D grid on [-1,1] interval based on Fejer-2 method.

    The fundamental definition is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx
        \sum_{i=1}^n w_i f(x_i)


    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    if npoints <= 1:
        raise ValueError("npoints must be greater that one, given {npoints}")

    theta = np.pi * (np.arange(npoints) + 1) / (npoints + 1)

    points = np.cos(theta)
    weights = np.zeros(npoints)

    nsum = (npoints + 1) // 2

    for k in range(0, npoints):
        serie = 0

        for m in range(1, nsum):
            serie += np.sin((2 * m - 1) * theta[k]) / (2 * m - 1)

        weights[k] = (4 * np.sin(theta[k]) / (npoints + 1)) * serie

    points = points[::-1]
    weights = weights[::-1]

    return OneDGrid(points, weights, (-1, 1))


# Auxiliar functions for Trefethen "sausage" transformation
# g2 is the function and derg2 is the first derivative.
# g3 is other function with the same boundary conditions of g2 and
# derg3 is the first derivative.
# this functions work for TrefethenCC, TrefethenGC2, and TrefethenGeneral
def _g2(x):
    r"""Return an auxiliary function g2(x) for Trefethen transformation."""
    return (1 / 149) * (120 * x + 20 * x ** 3 + 9 * x ** 5)


def _derg2(x):
    r"""Return the derivative function g2(x) for Trefethen transformation."""
    return (1 / 149) * (120 + 60 * x ** 2 + 45 * x ** 4)


def _g3(x):
    r"""Return an auxiliary function g3(x) for Trefethen transformation."""
    return (1 / 53089) * (
        40320 * x + 6720 * x ** 3 + 3024 * x ** 5 + 1800 * x ** 7 + 1225 * x ** 9
    )


def _derg3(x):
    r"""Return the derivative function g3(x) for Trefethen transformation."""
    return (1 / 53089) * (
        40320 + 20160 * x ** 2 + 15120 * x ** 4 + 12600 * x ** 6 + 11025 * x ** 8
    )


def TrefethenCC(npoints, d=3):
    r"""Generate 1D grid on [-1,1] interval based on Trefethen-Clenshaw-Curtis.

    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    grid = ClenshawCurtis(npoints)

    points = np.zeros(npoints)
    weights = np.zeros(npoints)

    if d == 2:
        points = _g2(grid.points)
        weights = _derg2(grid.points) * grid.weights
    elif d == 3:
        points = _g3(grid.points)
        weights = _derg3(grid.points) * grid.weights
    else:
        points = grid.points
        weights = grid.weights

    return OneDGrid(points, weights, (-1, 1))


def TrefethenGC2(npoints, d=3):
    r"""Generate 1D grid on [-1,1] interval based on Trefethen-Gauss-Chebyshev.

    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    grid = GaussChebyshevType2(npoints)

    points = np.zeros(npoints)
    weights = np.zeros(npoints)

    if d == 2:
        points = _g2(grid.points)
        weights = _derg2(grid.points) * grid.weights
    elif d == 3:
        points = _g3(grid.points)
        weights = _derg3(grid.points) * grid.weights
    else:
        points = grid.points
        weights = grid.weights

    return OneDGrid(points, weights, (-1, 1))


def TrefethenGeneral(npoints, quadrature, d=3):
    r"""Generate 1D grid on [-1,1] interval based on Trefethen-General.

    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    grid = quadrature(npoints)

    points = np.zeros(npoints)
    weights = np.zeros(npoints)

    if d == 2:
        points = _g2(grid.points)
        weights = _derg2(grid.points) * grid.weights
    elif d == 3:
        points = _g3(grid.points)
        weights = _derg3(grid.points) * grid.weights
    else:
        points = grid.points
        weights = grid.weights

    return OneDGrid(points, weights, (-1, 1))


# Auxiliar functions for Trefethen "strip" transformation
# gstrip is the function and dergstrio is the first derivative.
# this functions work for TrefethenStripCC, TrefethenStripGC2,
# and TrefethenStripGeneral
def _gstrip(rho, s):
    r"""Auxiliary function g(x) for Trefethen strip transformation."""
    tau = np.pi / np.log(rho)
    termd = 0.5 + 1 / (np.exp(tau * np.pi) + 1)
    u = np.arcsin(s)

    cn = 1 / (np.log(1 + np.exp(-tau * np.pi)) - np.log(2) + np.pi * tau * termd / 2)

    g = cn * (
        np.log(1 + np.exp(-tau * (np.pi / 2 + u)))
        - np.log(1 + np.exp(-tau * (np.pi / 2 - u)))
        + termd * tau * u
    )

    return g


def _dergstrip(rho, s):
    r"""First derivative of g(x) for Trefethen strip transformation."""
    tau = np.pi / np.log(rho)
    termd = 0.5 + 1 / (np.exp(tau * np.pi) + 1)

    cn = 1 / (np.log(1 + np.exp(-tau * np.pi)) - np.log(2) + np.pi * tau * termd / 2)

    gp = np.zeros(len(s))

    # get true label
    mask_true = np.isclose(np.fabs(s) - 1, 0, atol=1.0e-8)
    # get false label
    mask_false = mask_true == 0
    u = np.arcsin(s)
    gp[mask_true] = cn * tau ** 2 / 4 * np.tanh(tau * np.pi / 2) ** 2
    gp[mask_false] = (
        1 / (np.exp(tau * (np.pi / 2 + u[mask_false])) + 1)
        + 1 / (np.exp(tau * (np.pi / 2 - u[mask_false])) + 1)
        - termd
    ) * (-cn * tau / np.sqrt(1 - s[mask_false] ** 2))

    return gp


def TrefethenStripCC(npoints, rho=1.1):
    r"""Generate 1D grid on [-1,1] interval based on Trefethen-Clenshaw-Curtis.

    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    grid = ClenshawCurtis(npoints)

    points = np.zeros(npoints)
    weights = np.zeros(npoints)

    points = _gstrip(rho, grid.points)
    weights = _dergstrip(rho, grid.points) * grid.weights

    return OneDGrid(points, weights, (-1, 1))


def TrefethenStripGC2(npoints, rho=1.1):
    r"""Generate 1D grid on [-1,1] interval based on Trefethen-Gauss-Chebyshev.

    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    grid = GaussChebyshevType2(npoints)

    points = np.zeros(npoints)
    weights = np.zeros(npoints)

    points = _gstrip(rho, grid.points)
    weights = _dergstrip(rho, grid.points) * grid.weights

    return OneDGrid(points, weights, (-1, 1))


def TrefethenStripGeneral(npoints, quadrature, rho=1.1):
    r"""Generate 1D grid on [-1,1] interval based on Trefethen-General.

    Parameters
    ----------
    npoints : int
        Number of points in the grid.

    Returns
    -------
    OneDGrid
        A 1D grid instance.
    """
    grid = quadrature(npoints)

    points = np.zeros(npoints)
    weights = np.zeros(npoints)

    points = _gstrip(rho, grid.points)
    weights = _dergstrip(rho, grid.points) * grid.weights

    return OneDGrid(points, weights, (-1, 1))
