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


import warnings

import numpy as np
from scipy.special import roots_chebyu, roots_genlaguerre

from grid.basegrid import OneDGrid


class GaussLaguerre(OneDGrid):
    r"""
    Gauss Laguerre integral quadrature class.

    The definition of generalized Gauss-Laguerre quadrature is:

    .. math::
        \int_{0}^{\infty} x^\alpha e^{-x} f(x) dx \approx \sum_{i=1}^n w_i f(x_i),

    where :math:`\alpha > -1`\.

    However, to integrate function :math:`g(x)` over :math:`[0, \infty)`\, this is re-written as:

    .. math::
        \int_{0}^{\infty} g(x)dx \approx
        \sum_{i=1}^n \left(\frac{w_i}{x_i^\alpha e^{-x_i}}\right) g(x_i) = \sum_{i=1}^n w_i' g(x_i)

    """

    name = "Gauss-Laguerre"

    def __init__(self, npoints: int, alpha: float = 0):
        r"""Generate grid on :math:`[0, \infty)` based on generalized Gauss-Laguerre quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.
        alpha : float, optional
            Value of the parameter :math:`alpha` which must be larger than -1.

        Returns
        -------
        OneDGrid
            A 1-D grid instance containing points and weights.

        """
        if npoints <= 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")
        if alpha <= -1:
            raise ValueError(f"Argument alpha must be larger than -1, given {alpha}")
        # compute points and weights for Generalized Gauss-Laguerre quadrature
        points, weights = roots_genlaguerre(npoints, alpha)
        if np.any(np.isnan(weights)):
            raise RuntimeError(
                "Generation of the weights for Gauss-generalized Laguerre quadrature contains "
                "nans. This issue is related to SciPy."
            )
        weights *= np.exp(points) * np.power(points, -alpha)
        super().__init__(points, weights, (0, np.inf))


class GaussLegendre(OneDGrid):
    r"""
    Gauss-Legendre integral quadrature class.

    The definition of Gauss-Legendre quadrature is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx \sum_{i=1}^n w_i f(x_i),

    where :math:`w_i` are the quadrature weights and :math:`x_i` are the
    roots of the nth Legendre polynomial.

    """

    name = "Gauss-Legendre"

    def __init__(self, npoints: int):
        r"""Generate grid on :math:`[-1, 1]` interval based on Gauss-Legendre quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.

        Returns
        -------
        OneDGrid
            A 1-D grid instance containing points and weights.

        Notes
        -----
        - Only known to be accurate up to `npoints`\=100 and may cause problems after that amount.

        """
        if npoints <= 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")
        # compute points and weights for Gauss-Legendre quadrature
        # according to numpy's leggauss, the accuracy is only known up to `npoints=100`\.
        points, weights = np.polynomial.legendre.leggauss(npoints)
        super().__init__(points, weights, (-1, 1))


class GaussChebyshev(OneDGrid):
    r"""
    Gauss-Chebyshev integral quadrature class.

    The fundamental definition of Gauss-Chebyshev quadrature is:

    .. math::
        \int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}} dx \approx& \sum_{i=1}^n w_i f(x_i) \\
        x_i =& \cos\left( \frac{2i-1}{2n}\pi \right) \\
        w_i =& \frac{\pi}{n}

    However, to integrate a given function :math:`g(x)` over :math:`[-1, 1]`\, this is re-written as:

    .. math::
        \int_{-1}^{1}g(x)dx \approx \sum_{i=1}^n \left(w_i\sqrt{1-x_i^2}\right)g(x_i) =
        \sum_{i=1}^n w_i'g(x_i)

    """

    name = "Gauss-Chebyshev"

    def __init__(self, npoints: int):
        r"""Generate grid on :math:`[-1, 1]` interval based on Gauss-Chebyshev quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.

        Returns
        -------
        OneDGrid
            A 1-D grid instance containing points and weights.

        """
        if npoints <= 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")
        # compute points and weights for Gauss-Chebyshev quadrature (Type 1)
        # points are generated in decreasing order (from +1 to -1), so the order is reversed to
        # correctly traverse [-1, 1] when making an instance of OneDGrid
        points, weights = np.polynomial.chebyshev.chebgauss(npoints)
        weights *= np.sqrt(1 - np.power(points, 2))
        super().__init__(points[::-1], weights, (-1, 1))


class UniformInteger(OneDGrid):
    r"""
    HORTON2 integral quadrature (HortonLinear) class.

    .. math::
        \int_{0}^{n} f(x) dx \approx& \sum_{i=1}^n w_i f(x_i) \\
        x_i =& i - 1 \\
        w_i =& 1.0

    """

    name = "Uniform-Integer"

    def __init__(self, npoints: int):
        r"""Generate grid on [0, npoints] interval using equally spaced uniform distribution.

        Parameters
        ----------
        npoints : int
            Number of grid points.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance containing points and weights.

        """
        if npoints <= 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")
        points = np.arange(npoints)
        weights = np.ones(npoints)
        super().__init__(points, weights, (0, np.inf))


class GaussChebyshevType2(OneDGrid):
    r"""
    Gauss Chebyshev Type2 integral quadrature class.

    The definition of the Gauss-Chebyshev of the second kind quadrature is:

    .. math::
        \int_{-1}^{1} f(x) \sqrt{1-x^2} dx \approx& \sum_{i=1}^n w_i f(x_i) \\
        x_i =& \cos\left( \frac{i}{n+1} \pi \right) \\
        w_i =& \frac{\pi}{n+1} \sin^2 \left( \frac{i}{n+1} \pi \right)

    However, to integrate a given function :math:`g(x)` over :math:`[-1, 1]`\, this is re-written as:

    .. math::
        \int_{-1}^{1} g(x) dx \approx \sum_{i=1}^n \left(\frac{w_i}{\sqrt{1-x_i^2}}\right) g(x_i) =
        \sum_{i=1}^n w_i' g(x_i)

    """

    name = "Gauss-Chebyshev-Type2"

    def __init__(self, npoints: int):
        r"""Generate grid on :math:`[-1, 1]` interval based on Gauss-Chebyshev Type 2.

        Parameters
        ----------
        npoints : int
            Number of grid points.

        Returns
        -------
        OneDGrid
            A 1-D grid instance containing points and weights.

        """
        if npoints < 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")
        # compute points and weights for Gauss-Chebyshev quadrature (Type 2)
        points, weights = roots_chebyu(npoints)
        weights /= np.sqrt(1 - np.power(points, 2))
        super().__init__(points, weights, (-1, 1))


class GaussChebyshevLobatto(OneDGrid):
    r"""
    Gauss Chebyshev Lobatto integral quadrature class.

    The definition of Gauss-Chebyshev-Lobato quadrature is:

    .. math::
        \int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}} dx \approx& \sum_{i=1}^n w_i f(x_i) \\
        x_i =& \cos\left( \frac{(i-1)}{n-1}\pi \right) \\
        w_{1} = w_{n} =& \frac{\pi}{2(n-1)} \\
        w_{i\neq 1,n} =& \frac{\pi}{n-1}

    However, to integrate a given function :math:`g(x)` over :math:`[-1, 1]`\, this is re-written as:

    .. math::
        \int_{-1}^{1}g(x) dx \approx \sum_{i=1}^n \left(w_i \sqrt{1-x_i^2}\right) g(x_i) =
        \sum_{i=1}^n w_i' g(x_i)
    """

    name = "Gauss-Chebyshev-Lobatto"

    def __init__(self, npoints: int):
        r"""Generate grid on :math:`[-1, 1]` interval based on Gauss-Chebyshev-Lobatto quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.

        Returns
        -------
        OneDGrid
            A 1-D grid instance containing points and weights.

        """
        if npoints <= 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")

        # generate points in ascending order, and then compute weights
        points = np.cos(np.arange(npoints) * np.pi / (npoints - 1))
        points = points[::-1]
        weights = np.pi * np.sqrt(1 - np.power(points, 2)) / (npoints - 1)
        weights[0] /= 2
        weights[npoints - 1] /= 2

        super().__init__(points, weights, (-1, 1))


class Trapezoidal(OneDGrid):
    r"""
    Trapezoidal Lobatto integral quadrature class.

    The fundamental definition of Trapezoidal rule is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx& \sum_{i=1}^n w_i f(x_i) \\
        x_i =& -1 + 2 \left(\frac{i-1}{n-1}\right) \\
        w_1 = w_n =& \frac{1}{n} \\
        w_{i\neq 1,n} =& \frac{2}{n}

    """

    name = "Trapezoidal-Lobatto"

    def __init__(self, npoints: int):
        r"""Generate grid on [-1, 1] interval based on Trapezoidal (Euler-Maclaurin) rule.

        Parameters
        ----------
        npoints : int
            Number of grid points.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance containing points and weights.

        """
        if npoints <= 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")

        points = -1 + (2 * np.arange(npoints) / (npoints - 1))
        weights = 2 * np.ones(npoints) / (npoints - 1)
        weights[0] /= 2
        weights[npoints - 1] /= 2

        super().__init__(points, weights, (-1, 1))


class RectangleRuleSineEndPoints(OneDGrid):
    r"""
    Rectangle-Rule Sine end points integral quadrature class. [#]_

     .. math::
        \int_{-1}^{1} f(x) dx \approx& \sum_{i=1}^n w_i f(x_i) \\
        x_i =& \frac{i}{n+1} \\
        w_i =& \frac{2}{n+1} \sum_{m=1}^n \frac{\sin(m \pi x_i)(1-\cos(m \pi))}{m \pi}

    For consistency with other 1-D grids, the integration range is modified
    by :math:`q=2x-1` to the interval :math:`[-1, 1]`\, so that

    .. math::
        2 \int_{0}^{1} f(x) dx = \int_{-1}^{1} f(q) dq

    References
    ----------
    .. [#] Boyd, John P. Chebyshev and Fourier spectral methods. Courier Corporation, 2001.

    """

    name = "Rectangle-Rule-Sine"

    def __init__(self, npoints: int):
        r"""
        Generate grid on :math:`[-1, 1]` using rectangle rule for Sine Series (with endpoints).

        Parameters
        ----------
        npoints : int
            Number of grid points.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance containing points and weights.

        """
        if npoints <= 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")

        points = np.arange(1, npoints + 1, 1) / (npoints + 1)

        # make 1-D array of m values going from 1 to n
        m = np.arange(1, npoints + 1, 1)
        bm = (1.0 - np.cos(m * np.pi)) / (m * np.pi)
        # make 2-D array of sin(pi * m * xi) where rows/columns correspond to different m/xi
        sim = np.sin(np.outer(m * np.pi, points))
        # multiply 2 matrices (bm is treated like a (1, n) matrix)
        weights = bm @ sim
        weights *= 2 / (npoints + 1)

        # change integration range using variable q = 2x - 1
        points = 2 * points - 1
        weights *= 2

        super().__init__(points, weights, (-1, 1))


# Developer Note: This is depreciated, the points/weights seems to be incorrect from the book.
# class RectangleRuleSine(OneDGrid):
#     r"""
#     Rectangle-Rule Sine integral quadrature class.
#
#     .. math::
#         \int_{-1}^{1} f(x) dx \approx& \sum_{i=1}^n w_i f(x_i) \\
#         x_i =& \frac{2 i - 1}{2 n} \\
#         w_i =& \frac{2}{n^2 \pi} \sin(n\pi x_i) \sin^2(n\pi /2) +
#                 \frac{4}{n \pi} \sum_{m=1}^{n-1} \frac{\sin(m \pi x_i)\sin^2(m\pi /2)}{m}
#
#     For consistency with other 1-D grids, the integration range is modified
#     by :math:`q=2x-1` to the interval :math:`[-1, 1]`\, such that
#
#     .. math::
#         2 \int_{0}^{1} f(x) dx = \int_{-1}^{1} f(q) dq
#
#     References
#     ----------
#     .. [#] Boyd, John P. Chebyshev and Fourier spectral methods. Courier Corporation, 2001.
#
#     """
#
#     def __init__(self, npoints: int):
#         r"""Generate grid on :math:`[-1, 1]` interval using Interior Rectangle Rule for Sines.
#
#         Parameters
#         ----------
#         npoints : int
#             Number of grid points.
#
#         Returns
#         -------
#         OneDGrid
#             One-dimensional grid instance containing points and weights.
#
#         """
#         if npoints <= 1:
#             raise ValueError(
#                 f"Argument npoints must be an integer > 1, given {npoints}"
#             )
#
#         points = (2 * np.arange(1, npoints + 1, 1) - 1) / (2 * npoints)
#
#         weights = (
#             (2 / (npoints * np.pi**2))
#             * np.sin(npoints * np.pi * points)
#             * np.sin(npoints * np.pi / 2) ** 2
#         )
#
#         m = np.arange(npoints - 1) + 1
#         bm = np.sin(m * np.pi / 2) ** 2 / m
#         sim = np.sin(np.outer(m * np.pi, points))
#         wi = bm @ sim
#         weights += (4 / (npoints * np.pi)) * wi
#
#         # change integration range using variable q = 2x - 1
#         points = 2 * points - 1
#         weights *= 2
#
#         super().__init__(points, weights, (-1, 1))


class TanhSinh(OneDGrid):
    r"""
    Tanh Sinh integral quadrature class.

    The definition of the quadrature is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx& \sum_{i=-\frac{1}{2}(n-1)}^{\frac{1}{2}(n-1)} w_i f(x_i) \\
        x_i =& \tanh\left( \frac{\pi}{2} \sinh(i\delta) \right) \\
        w_i =& \frac{\frac{\pi}{2}\delta \cosh(i\delta)}{\cosh^2(\frac{\pi}{2}\sinh(i\delta))}

    This quadrature is useful when singularities or infinite derivatives exist on the
    endpoints of :math:`[-1, 1]`\.

    """

    name = "Tanh-Sinh"

    def __init__(self, npoints: int, delta: float = 0.1):
        r"""Generate grid on :math:`[-1, 1]` interval based on Tanh-Sinh rule.

        Parameters
        ----------
        npoints : int
            Number of grid points, which should be an odd integer.
        delta : float
            The value of parameter :math:`\delta`\, which is related with the size.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance containing points and weights.

        """
        if npoints <= 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")
        if npoints % 2 == 0:
            raise ValueError(f"Argument npoints must be an odd integer, given {npoints}")

        # compute summation indices & angle values
        j = int((1 - npoints) / 2) + np.arange(npoints)
        theta = j * delta

        points = np.tanh(0.5 * np.pi * np.sinh(theta))
        weights = np.cosh(theta) / np.cosh(0.5 * np.pi * np.sinh(theta)) ** 2
        weights *= 0.5 * np.pi * delta

        super().__init__(points, weights, (-1, 1))


class Simpson(OneDGrid):
    r"""Simpson integral quadrature class.

    The definition of this quadrature is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx \sum_{i=1}^n w_i f(x_i),

    where

    .. math::
        x_i &= -1 + 2 \left(\frac{i-1}{n-1}\right),
        w_i &= \begin{cases}
            2 / (3(N - 1)) & i = 0 \\
            8 / (3(N - 1)) & i \geq 1 \text{and is odd}, \\
            4 / (3(N - 1)) & i \geq 2 \text{and is even}.
        \end{cases}

    """

    name = "Simpson"

    def __init__(self, npoints: int):
        r"""Generate grid on :math:`[-1,1]` interval based on Simpson rule.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        if npoints <= 1:
            raise ValueError("npoints must be greater that one, given {npoints}")
        if npoints % 2 == 0:
            raise ValueError("npoints must be odd, given {npoints}")
        idx = np.arange(npoints)
        points = -1 + (2 * idx / (npoints - 1))

        weights = 2 * np.ones(npoints) / (3 * (npoints - 1))

        weights[1 : npoints - 1 : 2] *= 4.0
        weights[2 : npoints - 1 : 2] *= 2.0

        super().__init__(points, weights, (-1, 1))


class MidPoint(OneDGrid):
    r"""MidPoint integral quadrature class.

    The definition of this quadrature is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx& \sum_{i=1}^n w_i f(x_i) \\
        x_i =& -1 + \frac{2i + 1}{n} \\
        w_i =& \frac{2}{n}

    """

    name = "MidPoint"

    def __init__(self, npoints: int):
        r"""Generate grid on :math:`[-1, 1]` interval based on Mid-Point rule.

        Parameters
        ----------
        npoints : int
            Number of grid points.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance containing points and weights.

        """
        if npoints <= 1:
            raise ValueError(f"Argument npoints must be an integer > 1, given {npoints}")

        points = -1 + (2 * np.arange(npoints) + 1) / npoints
        weights = 2 * np.ones(npoints) / npoints

        super().__init__(points, weights, (-1, 1))


class ClenshawCurtis(OneDGrid):
    r"""Clenshaw-Curtis integral quadrature class.

    The definition of this quadrature is:

    .. math::
        \theta_i &= \pi (i - 1) / (n - 1) \\
        x_i &= \cos (\theta_i) \\
        w_i &= \frac{c_k}{n} \bigg(1 - \sum_{j=1}^{\lfloor n/2 \rfloor}
            \frac{b_j}{4j^2 - 1} \cos(2j\theta_i) \bigg) \\
        b_j &= \begin{cases}
            1 & \text{if } j = n/2 \\
            2 & \text{if } j < n/2
        \end{cases} \\
        c_j &= \begin{cases}
            1 & \text{if } k = 0, n\\
            2 & else
        \end{cases}

    where :math:`k=0,\cdots ,n`\.

    If discontinuous, it is recommended to break the intervals at the discontinuities
    and handled separately.

    """

    name = "Clenshaw-Curtis"

    def __init__(self, npoints: int):
        r"""Generate grid on :math:`[-1,1]` interval based on Clenshaw-Curtis method.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        if npoints <= 1:
            raise ValueError("npoints must be greater that one, given {npoints}")

        theta = np.pi * np.arange(npoints) / (npoints - 1)
        theta = theta[::-1]
        points = np.cos(theta)

        jmed = (npoints - 1) // 2
        bj = 2.0 * np.ones(jmed)
        if 2 * jmed + 1 == npoints:
            bj[jmed - 1] = 1.0

        j = np.arange(jmed)
        bj /= 4 * j * (j + 2) + 3
        cij = np.cos(np.outer(2 * (j + 1), theta))
        wi = bj @ cij

        weights = 2 * (1 - wi) / (npoints - 1)
        weights[0] /= 2
        weights[npoints - 1] /= 2

        super().__init__(points, weights, (-1, 1))


class FejerFirst(OneDGrid):
    r"""
    Fejer first integral quadrature class.

    The definition of this quadrature is:

    .. math::
        \theta_i &= \frac{(2i - 1)\pi}{2n}, \\
        x_i &= \cos(\theta_i), \\
        w_i &= \frac{2}{n}\bigg(1 - 2 \sum_{j=1}^{\lfloor n/2 \rfloor}
            \frac{\cos(2j \theta_j)}{4 j^2 - 1} \bigg),

    where :math:`k=1,\cdots, n`\. It uses the zeros of the Chebyshev polynomial.
    If discontinuous, it is recommended to break the intervals at the discontinuities
    and handled separately.

    """

    name = "Fejer-First"

    def __init__(self, npoints: int):
        r"""Generate 1D grid on :math:`(-1,1)` interval based on Fejer's first rule.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        if npoints <= 1:
            raise ValueError("npoints must be greater that one, given {npoints}")

        theta = np.pi * (2 * np.arange(npoints) + 1) / (2 * npoints)
        points = np.cos(theta)

        nsum = npoints // 2
        j = np.arange(nsum - 1) + 1

        bj = 2.0 * np.ones(nsum - 1) / (4 * j**2 - 1)
        cij = np.cos(np.outer(2 * j, theta))
        di = bj @ cij
        weights = 1 - di

        points = points[::-1]
        weights = weights[::-1] * (2 / npoints)

        super().__init__(points, weights, (-1, 1))


class FejerSecond(OneDGrid):
    r"""
    Fejer Second integral quadrature class.

    The definition of this quadrature is:

    .. math::
        \theta_i &= k \pi / n \\
        x_i &= \cos(\theta_i) \\
        w_i &= \frac{4 \sin(\theta_i)}{n} \sum_{j=1}^{\lfloor n/2 \rfloor}
            \frac{\sin(2j - 1)\theta_i}{2j - 1}\\

    where :math:`k=1, \cdots n - 1` and :math:`n` is the number of points. This
    method is considered more practical than the first method.  If discontinuous, it is
    recommended to break the intervals at the discontinuities and handled separately.

    """

    name = "Fejer-Second"

    def __init__(self, npoints: int):
        r"""Generate grid on :math:`(-1,1)` interval based on Fejer's second rule.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.
        """
        if npoints <= 1:
            raise ValueError("npoints must be greater that one, given {npoints}")

        theta = np.pi * (np.arange(npoints) + 1) / (npoints + 1)

        points = np.cos(theta)

        nsum = (npoints + 1) // 2
        j = np.arange(nsum - 1) + 1

        bj = np.ones(nsum - 1) / (2 * j - 1)
        sij = np.sin(np.outer(2 * j - 1, theta))
        wi = bj @ sij
        weights = 4 * np.sin(theta) * wi

        points = points[::-1]
        weights = weights[::-1] / (npoints + 1)

        super().__init__(points, weights, (-1, 1))


# Auxiliary functions for Trefethen "sausage" transformation
# g2 is the function and derg2 is the first derivative.
# g3 is other function with the same boundary conditions of g2 and
# derg3 is the first derivative.
# this functions work for TrefethenCC, TrefethenGC2, and TrefethenGeneral
def _g2(x):
    r"""Return an auxiliary function g2(x) for Trefethen transformation."""
    return (1 / 149) * (120 * x + 20 * x**3 + 9 * x**5)


def _derg2(x):
    r"""Return the derivative function g2(x) for Trefethen transformation."""
    return (1 / 149) * (120 + 60 * x**2 + 45 * x**4)


def _g3(x):
    r"""Return an auxiliary function g3(x) for Trefethen transformation."""
    return (1 / 53089) * (40320 * x + 6720 * x**3 + 3024 * x**5 + 1800 * x**7 + 1225 * x**9)


def _derg3(x):
    r"""Return the derivative function g3(x) for Trefethen transformation."""
    return (1 / 53089) * (40320 + 20160 * x**2 + 15120 * x**4 + 12600 * x**6 + 11025 * x**8)


class TrefethenCC(OneDGrid):
    """
    Trefethen polynomial transformation of Clenshaw-Curtis integral quadrature class [#]_.

    References
    ----------
    .. [#] Hale, Nicholas, and Lloyd N. Trefethen. "New quadrature formulas from conformal maps."
       SIAM Journal on Numerical Analysis 46.2 (2008): 930-948.

    """

    name = "Trefethen-Polynomial-Transformation-Clenshaw-Curtis"

    def __init__(self, npoints: int, d: int = 9):
        r"""Generate 1D grid on :math:`[-1,1]` interval based on Trefethen-Clenshaw-Curtis.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.
        d :
            Odd degree of the Taylor series polynomial of :math:`\sin^{-1}`\.
            Only d=1,5,9 are supported.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.
        """
        grid = ClenshawCurtis(npoints)

        if d == 1:
            points = grid.points
            weights = grid.weights
        elif d == 5:
            points = _g2(grid.points)
            weights = _derg2(grid.points) * grid.weights
        elif d == 9:
            points = _g3(grid.points)
            weights = _derg3(grid.points) * grid.weights
        else:
            raise ValueError(f"Degree {d} should be either 1, 5, 9.")

        super().__init__(points, weights, (-1, 1))


class TrefethenGC2(OneDGrid):
    """
    Trefethen polynomial transformation of Gauss-Chebyshev of the second kind quadrature [#]_.

    References
    ----------
    .. [#] Hale, Nicholas, and Lloyd N. Trefethen. "New quadrature formulas from conformal maps."
       SIAM Journal on Numerical Analysis 46.2 (2008): 930-948.

    """

    name = "Trefethen-Polynomial-Transformation-Gauss-Chebyshev-Type2"

    def __init__(self, npoints: int, d: int = 9):
        r"""Generate 1D grid on [-1,1] interval based on Trefethen-Gauss-Chebyshev.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.
        d : int
            Odd degree of the Taylor series polynomial of :math:`\sin^{-1}`\.
            Only d=1,5,9 are supported.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.
        """
        grid = GaussChebyshevType2(npoints)

        if d == 1:
            points = grid.points
            weights = grid.weights
        elif d == 5:
            points = _g2(grid.points)
            weights = _derg2(grid.points) * grid.weights
        elif d == 9:
            points = _g3(grid.points)
            weights = _derg3(grid.points) * grid.weights
        else:
            raise ValueError(f"Degree {d} should be either 1, 5, 9.")

        super().__init__(points, weights, (-1, 1))


class TrefethenGeneral(OneDGrid):
    """
    Trefethen polynomial transformation of a general integral quadrature class [#]_.

    References
    ----------
    .. [#] Hale, Nicholas, and Lloyd N. Trefethen. "New quadrature formulas from conformal maps."
       SIAM Journal on Numerical Analysis 46.2 (2008): 930-948.

    """

    name = "Trefethen-Polynomial"

    def __init__(self, npoints: int, quadrature: OneDGrid, d=9):
        r"""Generate 1D grid on :math:`[-1,1]` interval based on Trefethen-General.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.
        quadrature : OneDGrid
            General one-dimensional grid.
        d :
            Odd degree of the Taylor series polynomial of :math:`\sin^{-1}`\.
            Only d=1,5,9 are supported.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.
        """
        if not issubclass(quadrature, OneDGrid):
            raise TypeError(f"Quadrature {type(OneDGrid)} should be of type OneDgrid.")

        grid = quadrature(npoints)

        if d == 1:
            points = grid.points
            weights = grid.weights
        elif d == 5:
            points = _g2(grid.points)
            weights = _derg2(grid.points) * grid.weights
        elif d == 9:
            points = _g3(grid.points)
            weights = _derg3(grid.points) * grid.weights
        else:
            raise ValueError(f"Degree {d} should be either 1, 5, 9.")

        super().__init__(points, weights, (-1, 1))


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
    gp[mask_true] = cn * tau**2 / 4 * np.tanh(tau * np.pi / 2) ** 2
    gp[mask_false] = (
        1 / (np.exp(tau * (np.pi / 2 + u[mask_false])) + 1)
        + 1 / (np.exp(tau * (np.pi / 2 - u[mask_false])) + 1)
        - termd
    ) * (-cn * tau / np.sqrt(1 - s[mask_false] ** 2))

    return gp


class TrefethenStripCC(OneDGrid):
    """Trefethen strip transformation of Clenshaw-Curtis quadrature [#]_.

    References
    ----------
    .. [#] Hale, Nicholas, and Lloyd N. Trefethen. "New quadrature formulas from conformal maps."
       SIAM Journal on Numerical Analysis 46.2 (2008): 930-948.

    """

    name = "Trefethen-Strip-Transformation-Clenshaw-Curtis"

    def __init__(self, npoints: int, rho: float = 1.1):
        r"""Generate grid on :math:`[-1,1]` interval based on Trefethen-Clenshaw-Curtis.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.
        """
        grid = ClenshawCurtis(npoints)
        points = _gstrip(rho, grid.points)
        weights = _dergstrip(rho, grid.points) * grid.weights

        super().__init__(points, weights, (-1, 1))


class TrefethenStripGC2(OneDGrid):
    """Trefethen strip transformation of the Gauss-Chebyshev of the second kind quadrature [#]_.

    References
    ----------
    .. [#] Hale, Nicholas, and Lloyd N. Trefethen. "New quadrature formulas from conformal maps."
       SIAM Journal on Numerical Analysis 46.2 (2008): 930-948.

    """

    name = "Trefethen-Strip-Transformation-Gauss-Chebyshev-Type2"

    def __init__(self, npoints: int, rho: float = 1.1):
        r"""Generate grid on :math:`[-1,1]` interval based on Trefethen-Gauss-Chebyshev.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        grid = GaussChebyshevType2(npoints)
        points = _gstrip(rho, grid.points)
        weights = _dergstrip(rho, grid.points) * grid.weights

        super().__init__(points, weights, (-1, 1))


class TrefethenStripGeneral(OneDGrid):
    """Trefethen Strip General integral quadrature class [#]_.

    References
    ----------
    .. [#] Hale, Nicholas, and Lloyd N. Trefethen. "New quadrature formulas from conformal maps."
       SIAM Journal on Numerical Analysis 46.2 (2008): 930-948.

    """

    name = "Trefethen-Strip-General"

    def __init__(self, npoints: int, quadrature, rho: float = 1.1):
        r"""Generate grid on :math:`[-1,1]` interval based on Trefethen-General.

        Parameters
        ----------
        npoints : int
            Number of points in the grid.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        grid = quadrature(npoints)
        points = _gstrip(rho, grid.points)
        weights = _dergstrip(rho, grid.points) * grid.weights

        super().__init__(points, weights, (-1, 1))


class ExpSinh(OneDGrid):
    r"""Exponential-Hyperbolic Sine quadrature class.

    The definition of this quadrature is:

    .. math::
        \int_{0}^{\infty} f(x) dx \approx
        \sum_{k=-n}^n w_k f(x_k). \\
        x_k = \exp \left(\frac{\pi}{2}\sinh(k h) \right) \\
        w_k = \exp \left(\frac{\pi}{2}\sinh(k h) \right)\left(\frac{\pi h}{2} \cosh(k h) \right)

    Warnings
    --------
    - Using this quadrature requires heavy parameter-tuning in-order to work.

    """

    name = "Exponential-Hyperbolic-Sine"

    def __init__(self, npoints: int, h: float = 1.0):
        r"""Generate 1D grid on :math:`(0, \infty)` interval based on exp-sinh quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.
        h : float
            Value of parameter :math: `h` wich control the quadrature.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        warnings.warn(
            "Using this quadrature requires heavy parameter-tuning in-order to be applicable.",
            stacklevel=2,
        )
        if h <= 0:
            raise ValueError(f"The value of h must be bigger than 0, given {h}")
        if npoints < 1:
            raise ValueError(f"npoints must be bigger than 1, given {npoints}")
        if npoints % 2 == 0:
            raise ValueError(f"npoints must be odd, given {npoints}")
        m = int((npoints - 1) / 2)
        k = np.arange(-m, m + 1)
        points = np.exp(np.pi * np.sinh(k * h) / 2)
        weights = points * np.pi * h * np.cosh(k * h) / 2
        super().__init__(points, weights, (0, np.inf))


class LogExpSinh(OneDGrid):
    r"""Logarithm-Exponential-Hyperbolic Sine quadrature class.

    The definition of this quadrature is:

    .. math::
        \int_{0}^{\infty} f(x) dx \approx
        \sum_{k=-n}^n w_k f(x_k). \\
        x_k = \log \left( \exp \left(\frac{\pi}{2}\sinh(kh) \right) + 1\right) \\
        w_k = \frac{\pi h\cosh(kh)\exp(\frac{\pi}{2}\sinh(kh))}
        {2(\exp(\frac{\pi}{2}\sinh(kh))+1)}.

    Warnings
    --------
    - Using this quadrature requires heavy parameter-tuning in-order to work.

    """

    name = "Logarithm-Exponential-Hyperbolic-Sine"

    def __init__(self, npoints: int, h: float = 0.1):
        r"""Generate 1D grid on :math:`(0, \infty)` interval based on log-exp-sinh quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.
        h : float
            Value of parameter :math: `h` wich control the quadrature.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        warnings.warn(
            "Using this quadrature requires heavy parameter-tuning in-order to be applicable.",
            stacklevel=2,
        )
        if h <= 0:
            raise ValueError(f"The value of h must be bigger than 0, given {h}")
        if npoints < 1:
            raise ValueError(f"npoints must be bigger than 1, given {npoints}")
        if npoints % 2 == 0:
            raise ValueError(f"npoints must be odd, given {npoints}")
        m = int((npoints - 1) / 2)
        k = np.arange(-m, m + 1)
        points = np.log(np.exp(np.pi * np.sinh(k * h) / 2) + 1)
        weights = np.exp(np.pi * np.sinh(k * h) / 2) * np.pi * h * np.cosh(k * h) / 2
        weights /= np.exp(np.pi * np.sinh(k * h) / 2) + 1
        super().__init__(points, weights, (0, np.inf))


class ExpExp(OneDGrid):
    r"""Exponential-Exponential quadrature class.

    The definition of this quadrature is:

    .. math::
        \int_{0}^{\infty} f(x) dx \approx
        \sum_{k=-n}^n w_k f(x_k).  \\
        x_k = e^{kh} e^{-e^{-kh}} \\
        w_k = h e^{-e^{-kh}}\left( e^{kh} + 1 \right)

    Warnings
    --------
    - Using this quadrature requires heavy parameter-tuning in-order to work.

    """

    name = "Exponential-Exponential"

    def __init__(self, npoints: int, h: float = 0.1):
        r"""Generate 1D grid on :math:`(0, \infty)` interval based on exp-exp quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.
        h : float
            Value of parameter :math: `h` which control the quadrature.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        warnings.warn(
            "Using this quadrature requires heavy parameter-tuning in-order to be applicable.",
            stacklevel=2,
        )
        if h <= 0:
            raise ValueError(f"The value of h must be bigger than 0, given {h}")
        if npoints < 1:
            raise ValueError(f"npoints must be bigger than 1, given {npoints}")
        if npoints % 2 == 0:
            raise ValueError(f"npoints must be odd, given {npoints}")
        m = int((npoints - 1) / 2)
        k = np.arange(-m, m + 1)
        points = np.exp(k * h) * np.exp(-np.exp(-k * h))
        weights = h * np.exp(-np.exp(-k * h)) * (np.exp(k * h) + 1)
        super().__init__(points, weights, (0, np.inf))


class SingleTanh(OneDGrid):
    r"""Hyperbolic Tan quadrature class.

    The definition of this quadrature is:

    .. math::
        \int_{-1}^{1} f(x) dx \approx
        \sum_{k=-n}^n w_k f(x_k). \\
        x_k = \tanh{kh}  \\
        w_k = \frac{h}{\cosh^2(kh)}

    """

    name = "Hyperbolic-Tan"

    def __init__(self, npoints: int, h: float = 0.1):
        r"""Generate 1D grid on :math:`(-1, +1)` interval based on tanh-sinh quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.
        h : float
            Value of parameter :math: `h` which control the quadrature.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        if h <= 0:
            raise ValueError(f"The value of h must be bigger than 0, given {h}")
        if npoints < 1:
            raise ValueError(f"npoints must be bigger than 1, given {npoints}")
        if npoints % 2 == 0:
            raise ValueError(f"npoints must be odd, given {npoints}")
        m = int((npoints - 1) / 2)
        k = np.arange(-m, m + 1)
        points = np.tanh(k * h)
        weights = h / np.cosh(k * h) ** 2
        super().__init__(points, weights, (-1, 1))


class SingleExp(OneDGrid):
    r"""Single exponential quadrature class.

    The definition of this quadrature is:

    .. math::
        \int_{0}^{\infty} f(x) dx \approx
        \sum_{k=-n}^n w_k f(x_k). \\
        x_k = e^{kh}  \\
        w_k = h e^{kh}.

    Warnings
    --------
    - Using this quadrature requires heavy parameter-tuning in-order to work.

    """

    name = "Single-Exponential"

    def __init__(self, npoints: int, h: float = 0.1):
        r"""Generate 1D grid on :math:`(0, \infty)` interval based on exponential quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.
        h : float
            Value of parameter :math: `h` which control the quadrature.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        warnings.warn(
            "Using this quadrature requires heavy parameter-tuning in-order to be applicable.",
            stacklevel=2,
        )
        if h <= 0:
            raise ValueError(f"The value of h must be bigger than 0, given {h}")
        if npoints < 1:
            raise ValueError(f"npoints must be bigger than 1, given {npoints}")
        if npoints % 2 == 0:
            raise ValueError(f"npoints must be odd, given {npoints}")
        m = int((npoints - 1) / 2)
        k = np.arange(-m, m + 1)
        points = np.exp(k * h)
        weights = h * np.exp(k * h)
        super().__init__(points, weights, (0, np.inf))


class SingleArcSinhExp(OneDGrid):
    r"""Single Arc Hyperbolic Sine-Exponential quadrature class.

    The definition of this quadrature is:

    .. math::
        \int_{0}^{\infty} f(x) dx \approx
        \sum_{k=-n}^n w_k f(x_k). \\
        x_k = \mbox{arcsinh}(e^{kh}) \\
        w_k = \frac{h e^{kh}}{\sqrt{e^{2kh} + 1}}

    Warnings
    --------
    - Using this quadrature requires heavy parameter-tuning in-order to work.

    """

    name = "Single-Arc-Hyperbolic-Sine-Exponential"

    def __init__(self, npoints: int, h: float = 0.1):
        r"""Generate 1D grid on :math:`(0, \infty)` interval based on tanh-sinh quadrature.

        Parameters
        ----------
        npoints : int
            Number of grid points.
        h : float
            Value of parameter :math: `h` which control the quadrature.

        Returns
        -------
        OneDGrid
            One-dimensional grid instance.

        """
        warnings.warn(
            "Using this quadrature requires heavy parameter-tuning in-order to be applicable.",
            stacklevel=2,
        )
        if h <= 0:
            raise ValueError(f"The value of h must be bigger than 0, given {h}")
        if npoints < 1:
            raise ValueError(f"npoints must be bigger than 1, given {npoints}")
        if npoints % 2 == 0:
            raise ValueError(f"npoints must be odd, given {npoints}")
        m = int((npoints - 1) / 2)
        k = np.arange(-m, m + 1)
        points = np.arcsinh(np.exp(k * h))
        weights = h * np.exp(k * h) / np.sqrt(np.exp(2 * h * k) + 1)
        super().__init__(points, weights, (0, np.inf))
