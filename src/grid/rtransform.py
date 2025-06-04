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
"""Transformation from 1D intervals [a, b] to other 1D intervals [c, d]."""


import warnings
from abc import ABC, abstractmethod
from numbers import Number

import numpy as np

from grid.basegrid import OneDGrid


class BaseTransform(ABC):
    """Abstract class for transformation."""

    @abstractmethod
    def transform(self, x):
        """Abstract method for transformation."""

    @abstractmethod
    def inverse(self, r):
        """Abstract method for inverse transformation."""

    @abstractmethod
    def deriv(self, x):
        """Abstract method for 1st derivative of transformation."""

    @abstractmethod
    def deriv2(self, x):
        """Abstract method for the second derivative of transformation."""

    @abstractmethod
    def deriv3(self, x):
        """Abstract method for the third derivative of transformation."""

    @property
    def domain(self):
        """tuple: Transformation domain."""
        return self._domain

    @property
    def codomain(self):
        """tuple: Transformation codomain."""
        return self._codomain

    def transform_1d_grid(self, oned_grid):
        r"""Generate a new integral grid by transforming the provided grid.

        .. math::
            \int^{\inf}_0 g(r) d r &= \int^{r(\infty)}_{r(0)} g(r(x)) \frac{dr}{dx} dx \\
                                   &= \int^{r(\infty)}_{r(0)} g(r(x)) \frac{dr}{dx} dx \\
                                   &\approx \sum_{i=1}^N g(r(x_i)) \frac{dr}{dx}(x_i) w_i  \\
                                   &\approx \sum_{i=1}^N g(r(x_i)) \frac{dr}{dx}(x_i) w^r_n \\
            w^r_n &= w^x_n \cdot \frac{dr}{dx}

        Parameters
        ----------
        oned_grid : OneDGrid
            An instance of one-dimensional grid.

        Returns
        -------
        OneDGrid
            Transformed one-dimensional grid spanning a different domain.

        """
        if not isinstance(oned_grid, OneDGrid):
            raise TypeError(f"Input grid is not OneDGrid, got {type(oned_grid)}")
        # check domain
        if oned_grid.domain[0] < self.domain[0] or oned_grid.domain[1] > self.domain[1]:
            raise ValueError(
                "Given 1D grid domain does not match the transformation domain.\n"
                f"grid domain: {oned_grid.domain}, tf domain: {self.domain}"
            )
        new_points = self.transform(oned_grid.points)
        new_weights = self.deriv(oned_grid.points) * oned_grid.weights
        new_domain = oned_grid.domain
        if new_domain is not None:
            # Some transformation (Issue #125) reverses the order of points i.e.
            #    [-1, 1] maps to [infinity, 0].  This sort here fixes the problem here.
            new_domain = tuple(np.sort(self.transform(np.array(oned_grid.domain))))
        return OneDGrid(new_points, new_weights, new_domain)

    def _convert_inf(self, array, replace_inf=1e16):
        """Convert np.inf to 1e16 in case of numerical failure.

        Parameters
        ----------
        array : np.ndarray(N,)
        """
        if isinstance(array, Number):  # change for number
            new_v = np.sign(array) * replace_inf if np.isinf(array) else array
        else:
            new_v = array.copy()  # change for arrays with copy
            new_v[new_v == np.inf] = replace_inf
            new_v[new_v == -np.inf] = -replace_inf
        return new_v


class BeckeRTransform(BaseTransform):
    r"""
    Becke Transformation.

    The Becke transformation transforms from :math:`[-1, 1]` to :math:`[r_{min}, \infty)`
    according to [#]_

    .. math::
        r(x) = R \frac{1 + x}{1 - x} + r_{min}.

    The inverse transformation is given by

    .. math::
        x(r) = \frac{r - r_{min} - R} {r - r_{min} + R}.

    References
    ----------
    .. [#] Becke, Axel D. "A multicenter numerical integration scheme for polyatomic molecules."
       The Journal of chemical physics 88.4 (1988): 2547-2553.

    """

    def __init__(self, rmin: float, R: float, trim_inf: bool = True):
        r"""Construct Becke transform, :math:`[-1, 1]` to :math`[r_{min}, \infty)`\.

        Parameters
        ----------
        rmin : float
            The minimum coordinate :math:`r_{min}` in the transformed interval
            :math:`[r_{min}, \infty)`\.
        R : float
            The scale factor used in the transformation.
        trim_inf : bool, optional
            Flag to trim infinite value in transformed array. If True, it will
            replace np.inf with 1e16. This may cause unexpected errors in the
            following operations.

        """
        self._rmin = rmin
        self._R = R
        self.trim_inf = trim_inf
        self._domain = (-1, 1)
        self._codomain = (rmin, np.inf)

    @property
    def rmin(self):
        """float: the minimum value for the transformed array."""
        return self._rmin

    @property
    def R(self):
        """float: the scale factor for the transformed array."""
        return self._R

    # @classmethod
    # def transform_grid(cls, oned_grid, rmin, radius):
    #     if not isinstance(oned_grid, OneDGrid):
    #         raise ValueError(f"Given grid is not OneDGrid, got {type(oned_grid)}")
    #     R = BeckeRTransform.find_parameter(oned_grid.points, rmin, radius)
    #     tfm = cls(rmin=rmin, R=R)
    #     new_points = tfm.transform(oned_grid.points)
    #     new_weights = tfm.deriv(oned_grid.points) * oned_grid.weights
    #     return RadialGrid(new_points, new_weights), tfm

    @staticmethod
    def find_parameter(array: np.ndarray, rmin: float, radius: float):
        r"""
        Compute R such that half of the points in :math:`[r_{min}, \infty)` are within radius.

        Parameters
        ----------
        array : np.ndarray(N,)
            One-dimensional array in the domain :math:`[-1, 1]`\.
        rmin : float
            Minimum value for transformed array.
        radius : float
            Atomic radius of interest.

        Returns
        -------
        float :
            The optimal value of scale factor R.

        """
        if rmin > radius:
            raise ValueError(
                f"rmin need to be smaller than radius, rmin: {rmin}, radius: {radius}."
            )
        size = array.size
        if size % 2:
            mid_value = array[size // 2]
        else:
            mid_value = (array[size // 2 - 1] + array[size // 2]) / 2
        return (radius - rmin) * (1 - mid_value) / (1 + mid_value)

    def transform(self, x: np.ndarray):
        r"""Transform from :math:`[-1, 1]` to :math:`[r_{min}, \infty)`\.

        .. math::
            r_i = R \frac{1 + x_i}{1 - x_i} + r_{min}

        Parameters
        ----------
        x : np.ndarray(N,)
            One-dimensional array in the domain :math:`[-1, 1]`\.

        Returns
        -------
        np.ndarray(N,)
            Transformed array located between :math:`[r_min, \infty)`\.

        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rf_array = self._R * (1 + x) / (1 - x) + self._rmin
        if self.trim_inf:
            rf_array = self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r: np.ndarray):
        r"""Transform :math:`[r_{mi}n, \infty)` back to original :math:`[-1, 1]`\.

        .. math::
            x_i = \frac{r_i - r_{min} - R} {r_i - r_{min} + R}

        Parameters
        ----------
        r : np.ndarray(N,)
            One-dimensional array in the codomain :math:`[r_{min}, \infty)`\.

        Returns
        -------
        np.ndarray(N,)
            One dimensional array in :math:`[-1, 1]`\.
        """
        return (r - self._rmin - self._R) / (r - self._rmin + self._R)

    def deriv(self, x: np.ndarray):
        r"""Compute the first derivative of Becke transformation.

        .. math::
            \frac{dr_i}{dx_i} = 2R \frac{1}{(1-x)^2}

        Parameters
        ----------
        x : np.array(N,)
            One-dimensional array in the domain :math:`[-1, 1]`\.

        Returns
        -------
        np.ndarray(N,)
            First derivative of Becke transformation at each point.
        """
        with np.errstate(divide="ignore"):
            deriv = 2 * self._R / ((1 - x) ** 2)
        if self.trim_inf:
            deriv = self._convert_inf(deriv)
        return deriv

    def deriv2(self, x: np.ndarray):
        r"""Compute the second derivative of Becke transformation.

        .. math::
            \frac{d^2r}{dx^2} = 4R \frac{1}{1-x^3}

        Parameters
        ----------
        x : np.array(N,)
            One-dimensional array in the domain :math:`[-1, 1]`\.

        Returns
        -------
        np.ndarray(N,)
            Second derivative of Becke transformation at each point.
        """
        with np.errstate(divide="ignore"):
            return 4 * self._R / (1 - x) ** 3

    def deriv3(self, x: np.ndarray):
        r"""Compute the third derivative of Becke transformation.

        .. math::
            \frac{d^3r}{dx^3} = 12R \frac{1}{1 - x^4}

        Parameters
        ----------
        x : np.array(N,)
            One-dimensional array in the domain :math:`[-1, 1]`\.

        Returns
        -------
        np.ndarray(N,)
            Third derivative of Becke transformation at each point.
        """
        return 12 * self._R / (1 - x) ** 4


class LinearFiniteRTransform(BaseTransform):
    r"""
    Linear finite transformation from :math:`[-1, 1]` to :math:`[r_{min}, r_{max}]`\.

    The Linear transformation from finite interval :math:`[-1, 1]` to finite interval
    :math:`[r_{min}, r_{max}]` is given by

    .. math::
        r(x) = \frac{r_{max} - r_{min}}{2} (1 + x) + r_{min}.

    The inverse transformation is given by

    .. math::
        x(r) = \frac{2 r - (r_{max} + r_{min})}{r_{max} - r_{min}}

    """

    def __init__(self, rmin: float, rmax: float):
        """Construct linear transformation instance.

        Parameters
        ----------
        rmin : float
            Minimum value for transformed interval.
        rmax : float
            Maximum value for transformed interval.
        """
        self._rmin = rmin
        self._rmax = rmax
        self._domain = (-1, 1)
        self._codomain = (rmin, rmax)

    def transform(self, x: np.ndarray):
        r"""Transform from interval :math:`[-1, 1]` to :math:`[r_{min}, r_{max}]`\.

        .. math::
            r_i = \frac{r_{max} - r_{min}}{2} (1 + x_i) + r_{min}.

        This transformation maps :math:`r_i(-1) = r_{min}` and :math:`r_i(1) = r_{max}`\.

        Parameters
        ----------
        x : ndarray
            One-dimensional array in the domain :math:`[-1, 1]`\.

        Returns
        -------
        float or np.ndarray
            Transformed points between :math:`[r_{min}, r_{max}]`
        """
        return (1 + x) * (self._rmax - self._rmin) / 2 + self._rmin

    def deriv(self, x: np.ndarray):
        r"""Compute the 1st order derivative.

        .. math::
            \frac{dr}{dx} = \frac{r_{max} - r_{min}}{2}

        Parameters
        ----------
        x : ndarray
            One-dimensional array in the domain :math:`[-1, 1]`\.

        Returns
        -------
        float or np.ndarray
            First order derivative at given points.
        """
        if isinstance(x, Number):
            return (self._rmax - self._rmin) / 2
        else:
            return np.ones(x.size) * (self._rmax - self._rmin) / 2

    def deriv2(self, x: np.ndarray):
        r"""Compute the second order derivative.

        .. math::
            \frac{d^2 r}{dx^2} = 0

        Parameters
        ----------
        x : ndarray
            One-dimensional array in the domain :math:`[-1, 1]`\.

        Returns
        -------
        float or np.ndarray
            Second order derivative at given points.
        """
        return np.array(0) if isinstance(x, Number) else np.zeros(x.size)

    def deriv3(self, x: np.ndarray):
        r"""Compute the third order derivative.

        .. math::
            \frac{d^2 r}{dx^2} = 0

        Parameters
        ----------
        x : ndarray
            One-dimensional array in the domain :math:`[-1, 1]`\.

        Returns
        -------
        ndarray
            Third order derivative at given points.
        """
        return np.array(0) if isinstance(x, Number) else np.zeros(x.size)

    def inverse(self, r: np.ndarray):
        r"""Compute the inverse of the transformation.

        .. math::
            x_i = \frac{2 r_i - (r_{max} + r_{min})}{r_{max} - r_{min}}

        Parameters
        ----------
        r : ndarray
            One-dimensional array in the co-domain :math:`[r_{min}, r_{max}]`\.

        Returns
        -------
        ndarray
            One-dimensional array in the domain :math:`[-1, 1]`\.
        """
        return (2 * r - (self._rmax + self._rmin)) / (self._rmax - self._rmin)


class InverseRTransform(BaseTransform):
    """Inverse transformation class for any general transformation."""

    def __init__(self, transform: BaseTransform):
        """Construct InverseRTransform instance.

        Parameters
        ----------
        transform : BaseTransform
            One-dimension transformation instance.

        """
        if not isinstance(transform, BaseTransform):
            raise TypeError(f"Input need to be a transform instance, got {type(transform)}.")
        self._tfm = transform
        self._domain = transform.codomain
        self._codomain = transform.domain

    def transform(self, r: np.ndarray):
        """Transform array back to the original, domain of the provided transformation.

        This transformation is equivalent to the inverse function of the original
        transformation (i.e. OriginTF.inverse).

        Parameters
        ----------
        r : np.ndarray(N,)
            One-dimensional array in the co-domain r of the original transformation.

        Returns
        -------
        np.ndarray(N,)
            Original one-dimensional array in the domain x of the original transformation.

        """
        return self._tfm.inverse(r)

    def inverse(self, x: np.ndarray):
        """Transform array to the co-domain of the provided transformation.

        This transformation is equivalent to the transformation function of the original
        transformation (i.e. OriginTF.transform).

        Parameters
        ----------
        x : np.ndarray
            One-dimension array in the domain x of the original transformation.

        Returns
        -------
        np.ndarray
            One-dimensional array in the co-domain r of the original transformation.

        """
        return self._tfm.transform(x)

    def _d1(self, r: np.ndarray):
        """Compute 1st order derivative of the original transformation.

        Parameters
        ----------
        r : np.ndarray(n)
            One-dimensional array in the co-domain r of the original transformation.

        Returns
        -------
        np.ndarray(n,)
            1st order derivative array

        """
        d1 = self._tfm.deriv(r)
        if np.any(d1 == 0):
            raise ZeroDivisionError("First derivative of original transformation has 0 value")
        return d1

    def deriv(self, r: np.ndarray):
        r"""Compute the first derivative of inverse transformation.

        .. math::
            \frac{dx}{dr} = (\frac{dr}{dx})^{-1}

        Parameters
        ----------
        r : np.array(N,)
            One-dimensional array in the co-domain r of the original transformation.

        Returns
        -------
        np.ndarray(N,)
            First derivative of the inverse transformation at each point r.

        """
        # x: inverse x array, d1: first derivative
        r = self._tfm.inverse(r)
        return 1 / self._d1(r)

    def deriv2(self, r: np.ndarray):
        r"""Compute the second derivative of inverse transformation.

        .. math::
            \frac{d^2 x}{dr^2} = - \frac{d^2 r}{dx^2} (\frac{dx}{dr})^3

        Parameters
        ----------
        r : np.array(N,)
            One-dimensional array in the co-domain r of the original transformation.

        Returns
        -------
        np.ndarray(N,)
            Second derivative of the inverse transformation at each point r.

        """
        # x: inverse x array, d1: first derivative
        # d2: second derivative d^2x / dy^2
        r = self._tfm.inverse(r)
        d2 = self._tfm.deriv2
        return -d2(r) / self._d1(r) ** 3

    def deriv3(self, r: np.ndarray):
        r"""Compute the third derivative of inverse transformation.

        .. math::
            \frac{d^3 x}{dr^3} = -\frac{d^3 r}{dx^3} (\frac{dx}{dr})^4
                               + 3 (\frac{d^2 r}{dx^2})^2 (\frac{dx}{dr})^5

        Parameters
        ----------
        r : np.array(N,)
            One-dimensional array in the co-domain r of the original transformation.

        Returns
        -------
        np.ndarray(N,)
            Third derivative of inverse transformation at each point r.

        """
        # x: inverse x array, d1: first derivative
        # d2: second derivative d^2x / dy^2
        # d3: third derivative d^3x / dy^3
        r = self._tfm.inverse(r)
        d1 = self._tfm.deriv
        d2 = self._tfm.deriv2
        d3 = self._tfm.deriv3
        return (3 * d2(r) ** 2 - d1(r) * d3(r)) / self._d1(r) ** 5


class IdentityRTransform(BaseTransform):
    r"""
    Identity Transform class.

    The identity transform class trivially transforms from :math:`[0, \infty)` to
    :math:`[0, \infty)` given by

    .. math::
        r(x) = x.

    The inverse transformation is given by

    .. math::
        x(r) = r.

    """

    def __init__(self):
        self._domain = (0, np.inf)
        self._codomain = (0, np.inf)

    def transform(self, x: np.ndarray):
        r"""
        Perform given array into itself.

        .. math::
            r_i = x_i

        Parameters
        ----------
        x : ndarray(N,)
            One dimension numpy array located in :math:`[0, \infty)`\.

        Returns
        -------
        ndarray(N,)
            Identity transformation at each point x.

        """
        return x

    def deriv(self, x: np.ndarray):
        r"""
        Compute the first derivative of identity transform.

        Parameters
        ----------
        x : ndarray(N,)
            One dimension numpy array located in :math:`[0, \infty)`\.

        Returns
        -------
        ndarray(N,)
            First derivative of identity transformation at x.

        """
        return 1 if isinstance(x, Number) else np.ones(x.size)

    def deriv2(self, x: np.ndarray):
        r"""
        Compute the second derivative of identity transform.

        Parameters
        ----------
        x : ndarray(N,)
            One dimension numpy array located in :math:`[0, \infty)`\.

        Returns
        -------
        ndarray(N,)
            Second derivative of identity transformation at x.

        """
        return 0 if isinstance(x, Number) else np.zeros(x.size)

    def deriv3(self, x: np.ndarray):
        r"""
        Compute the third derivative of identity transform.

        Parameters
        ----------
        x : ndarray(N,)
            One dimension numpy array located in :math:`[0, \infty)`\.

        Returns
        -------
        np.ndarray(N,)
            Third derivative of identity transformation at x.

        """
        return 0 if isinstance(x, Number) else np.zeros(x.size)

    def inverse(self, r: np.ndarray):
        r"""
        Compute the inverse of identity transform.

        Parameters
        ----------
        r : ndarray(N,)
            One dimension numpy array located in :math:`[0, \infty)`\.

        Returns
        -------
        np.ndarray(N,)
            Inverse transformation of the identity transformation at x.

        """
        return r


class LinearInfiniteRTransform(BaseTransform):
    r"""
    Linear transform from interval :math:`[0, \infty)` to :math:`[r_{min}, r_{max})`\.

    This transformation linearly maps the infinite interval :math:`[0, \infty)` to a finite
    interval :math:`[r_{min}, r_{max}]` given by

    .. math::
        r(x) = \frac{(r_{max} - r_{min})}{b} x + r_{min},

    where :math:`r(b) = r_{max}`\.  If None, then the :math:`b` is taken to be the maximum
    from the first grid that is being transformed. This transformation always maps zero to
    :math:`r_{min}`\.

    The original goal is to transform the `UniformGrid`\, equally-spaced integers from 0 to N-1,
    to :math:`[r_{min}, r_{max}]`\.

    The inverse is given by

    .. math::
        x(r) = (r - r_{min}) \frac{\max_i (r_i)}{r_{max} - r_{min}}

    """

    def __init__(self, rmin: float, rmax: float, b: float = None):
        r"""Initialize linear transform class.

        Parameters
        ----------
        rmin : float
            Define the lower end of the linear transform
        rmax : float
            Define the upper end of the linear transform
        b: float
            Maximum :math:`b` of a prespecified radial grid :math:`[0, b]` such that
            :math:`b` maps to `rmax`\. If None, then the maximum is taken and stored from the
            grid that is transformed initially.

        """
        if rmin >= rmax:
            raise ValueError(f"rmin need to be larger than rmax.\n  rmin: {rmin}, rmax: {rmax}")
        self._rmin = rmin
        self._rmax = rmax
        self._domain = (0, np.inf)
        self._codomain = (rmin, rmax)
        self._b = b

    @property
    def rmin(self):
        r"""float: rmin value of the tf."""
        return self._rmin

    @property
    def rmax(self):
        r"""float: rmax value of the tf."""
        return self._rmax

    @property
    def b(self):
        r"""float: Parameter such that :math:`r(b) = r_{max}`\."""
        return self._b

    def set_maximum_parameter_b(self, x):
        r"""Sets up the parameter b from taken the maximum over some grid x."""
        if self.b is None:
            self._b = np.max(x)
            if np.abs(self.b) < 1e-16:
                raise ValueError(
                    f"The parameter b {self.b} is taken from the maximum of the grid"
                    f"and can't be zero."
                )

    def transform(self, x: np.ndarray):
        r"""Transform from interval :math:`[0, \infty)` to :math:`[r_{min}, r_{max}]`\.

        .. math::
            r_i = \frac{(r_{max} - r_{min})}{\max_i x_i} x_i + r_{min},

        where :math:`N` is the number of points. The goal is to transform
        equally-spaced integers from 0 to N-1, to :math:`[r_{min}, r_{max}]`\.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain :math:`[0, \infty)`\.

        Returns
        -------
        ndarray(N,)
            Transformed points between :math:`[r_{min}, r_{max}]`\.

        """
        self.set_maximum_parameter_b(x)
        alpha = (self._rmax - self._rmin) / self.b
        return alpha * x + self._rmin

    def deriv(self, x: np.ndarray):
        r"""
        Compute the first derivative of linear transformation.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain :math:`[0, \infty)`\.

        Returns
        -------
        ndarray(N,)
            First derivative of transformation at x.

        """
        self.set_maximum_parameter_b(x)
        alpha = (self._rmax - self._rmin) / self.b
        return np.ones(x.size) * alpha

    def deriv2(self, x: np.ndarray):
        r"""Compute the second derivative of linear transformation.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain :math:`[0, \infty)`\.

        Returns
        -------
        ndarray(N,)
            Second derivative of transformation at x.

        """
        return np.zeros(x.size)

    def deriv3(self, x: np.ndarray):
        r"""Compute the third derivative of linear transformation.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain :math:`[0, \infty)`\.

        Returns
        -------
        ndarray(N,)
            Third derivative of transformation at x.

        """
        return np.zeros(x.size)

    def inverse(self, r: np.ndarray):
        r"""Compute the inverse of linear transformation.

        .. math::
            x_i = (r - r_{min}) / frac{b}{r_{max} - r_{min}}

        Parameters
        ----------
        r : ndarray(N,)
            One-dimensional array in the domain :math:`[r_{min}, r_{max}]`\.

        Returns
        -------
        ndarray(N,)
            Inverse of transformation from coordinate :math:`r` to :math:`x`\.

        """
        self.set_maximum_parameter_b(r)
        alpha = (self._rmax - self._rmin) / self.b
        return (r - self._rmin) / alpha


class ExpRTransform(BaseTransform):
    r"""
    Exponential transform from :math:`[0, \infty)` to :math:`[r_{min}, r_{max}]`\.

    This transformation is given by

    .. math::
        r(x) = r_{min} e^{x \log\bigg(\frac{r_{max}}{r_{min} / b}  \bigg)},


    where :math:`b` maps to `rmax`\. If None, then the :math:`b` is taken to be the maximum
     from the first grid that is being transformed. This transformation always maps zero to
     :math:`r_{min}`\.

    The inverse transformation is given by

    .. math::
        x(r) = \frac{\log\big(\frac{r}{r_{min}} \big) b}{\log(\frac{r_{max}}{r_{min}})}
    """

    def __init__(self, rmin: float, rmax: float, b: float = None):
        r"""Initialize exp transform instance.

        Parameters
        ----------
        rmin : float
            Minimum value for transformed points.
        rmax : float
            Maximum value for transformed points.
        b: float
            Maximum :math:`b` of a prespecified radial grid :math:`[0, b]` such that
            :math:`b` maps to `rmax`\. If None, then the maximum is taken and stored from the
            grid that is transformed initially.

        """
        if rmin < 0 or rmax < 0:
            raise ValueError(f"rmin or rmax need to be positive\n  rmin: {rmin}, rmax: {rmax}")
        if rmin >= rmax:
            raise ValueError(f"rmin need to be smaller than rmax\n  rmin: {rmin}, rmax: {rmax}")
        self._rmin = rmin
        self._rmax = rmax
        self._domain = (0, np.inf)
        self._codomain = (rmin, rmax)
        self._b = b

    @property
    def rmin(self):
        r"""float: the value of rmin."""
        return self._rmin

    @property
    def rmax(self):
        r"""float: the value of rmax."""
        return self._rmax

    @property
    def b(self):
        r"""float: Parameter :math:`b` that maps/transforms to :math:`r_{max}`\."""
        return self._b

    def set_maximum_parameter_b(self, x):
        r"""Sets up the parameter b from taken the maximum over x."""
        if self.b is None:
            self._b = np.max(x)
            if np.abs(self.b) < 1e-16:
                raise ValueError(
                    f"The parameter b {self.b} is taken from the maximum of the grid"
                    f"and can't be zero."
                )

    def transform(self, x: np.ndarray):
        r"""
        Perform exponential transform.

        .. math::
            r = r_{min} e^{x \log\bigg(\frac{r_{max}}{r_{min} / b}  \bigg)},

        where :math:`b` is a prespecified parameter that maps to :math:`r_{max}`\.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation.

        Return
        ------
        ndarray(N,)
            The transformation of x in the co-domain of the transformation.

        """
        self.set_maximum_parameter_b(x)
        alpha = np.log(self._rmax / self._rmin) / self.b
        return self._rmin * np.exp(x * alpha)

    def deriv(self, x: np.ndarray):
        r"""
        Compute the first derivative of exponential transform.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation.

        Return
        ------
        ndarray(N,)
            First derivative of transformation at x.

        """
        self.set_maximum_parameter_b(x)
        alpha = np.log(self._rmax / self._rmin) / self.b
        return self.transform(x) * alpha

    def deriv2(self, x: np.ndarray):
        r"""
        Compute the second derivative of exponential transform.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation.

        Return
        ------
        ndarray(N,)
            Second derivative of transformation at x.

        """
        self.set_maximum_parameter_b(x)
        alpha = np.log(self._rmax / self._rmin) / self.b
        return self.deriv(x) * alpha

    def deriv3(self, x: np.ndarray):
        r"""
        Compute the third derivative of exponential transform.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation.

        Return
        ------
        ndarray(N,)
            Third derivative of transformation at x.

        """
        self.set_maximum_parameter_b(x)
        alpha = np.log(self._rmax / self._rmin) / self.b
        return self.deriv2(x) * alpha

    def inverse(self, r: np.ndarray):
        r"""
        Compute the inverse of exponential transform.

        .. math::
            x(r_i) = \frac{\log\big(\frac{r}{r_{min}} \big) b}{\log(\frac{r_{max}}{r_{min}})},

        where :math:`b` is a prespecified parameter

        Parameters
        ----------
        r : ndarray(N,)
            One-dimensional array in the domain of the transformation.

        Return
        ------
        ndarray(N,)
            Inverse transformation at r.

        """
        self.set_maximum_parameter_b(r)
        alpha = np.log(self._rmax / self._rmin) / self.b
        return np.log(r / self._rmin) / alpha


class PowerRTransform(BaseTransform):
    r"""
    Power transform class from :math:`[0, \infty)` to :math:`[r_{min}, r_{max}]`\.

    This transformations is given by

    .. math::
        r(x) = r_{min}  (x + 1)^{\frac{\log(r_{max} - \log(r_{min}}{\log(b + 1)}},

    such that :math:`r(b) = r_{max}`\.

    The inverse of the transformation is given by

    .. math::
         x(r) = \frac{r}{r_{min}}^{\frac{\log(N)}{\log(r_{max}) - \log(r_{min})}} - 1.

    """

    def __init__(self, rmin: float, rmax: float, b: float = None):
        r"""Initialize power transform instance.

        Parameters
        ----------
        rmin : float
            Minimum value for transformed points
        rmax : float
            Maximum value for transformed points
        b: float
            The parameter b that maps to :math:`r_{max}`\.


        """
        if rmin >= rmax:
            raise ValueError("rmin must be smaller rmax.")
        if rmin <= 0 or rmax <= 0:
            raise ValueError("rmin and rmax must be positive.")
        self._rmin = rmin
        self._rmax = rmax
        self._domain = (0, np.inf)
        self._codomain = (rmin, rmax)
        self._b = b

    @property
    def b(self):
        r"""float: Parameter :math:`b` that maps/transforms to :math:`r_{max}`\."""
        return self._b

    def set_maximum_parameter_b(self, x):
        r"""Sets up the parameter b from taken the maximum over x."""
        if self.b is None:
            self._b = np.max(x)
            if np.abs(self.b) < 1e-16:
                raise ValueError(
                    f"The parameter b {self.b} is taken from the maximum of the grid"
                    f"and can't be zero."
                )

    @property
    def rmin(self):
        r"""float: the value of rmin."""
        return self._rmin

    @property
    def rmax(self):
        r"""float: the value of rmax."""
        return self._rmax

    def transform(self, x: np.ndarray):
        r"""
        Perform power transform.

        .. math::
            r = r_{min}  (x + 1)^{\frac{\log(r_{max} - \log(r_{min}}{b + 1}},

        such that :math:`r(b) = r_{max}`\.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation :math:`[0,\infty)`\.

        Return
        ------
        ndarray(N,)
            The transformation of x to the co-domain of the transformation.

        """
        self.set_maximum_parameter_b(x)
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(self.b + 1)
        if power < 2:
            warnings.warn(
                f"power need to be larger than 2\n  power: {power}",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._rmin * np.power(x + 1, power)

    def deriv(self, x: np.ndarray):
        r"""
        Compute first derivative of power transform.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation :math:`[0,\infty)`\.

        Return
        ------
        ndarray(N,)
            First derivative of transformation at x.

        """
        self.set_maximum_parameter_b(x)
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(self.b + 1)
        return power * self._rmin * np.power(x + 1, power - 1)

    def deriv2(self, x: np.ndarray):
        r"""
        Compute second derivative of power transform.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation :math:`[0,\infty)`\.

        Return
        ------
        ndarray(N,)
            Second derivative of transformation at x.

        """

        self.set_maximum_parameter_b(x)
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(self.b + 1)
        return power * (power - 1) * self._rmin * np.power(x + 1, power - 2)

    def deriv3(self, x: np.ndarray):
        r"""
        Compute third derivative of power transform.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation :math:`[0,\infty)`\.

        Return
        ------
        ndarray(N,)
            Third derivative of transformation at x.

        """
        self.set_maximum_parameter_b(x)
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(self.b + 1)
        return power * (power - 1) * (power - 2) * self._rmin * np.power(x + 1, power - 3)

    def inverse(self, r: np.ndarray):
        r"""
        Compute the inverse of power transform.

        .. math::
            x(r) = \frac{r}{r_{min}}^{\frac{\log(b + 1)}{\log(r_{max}) - \log(r_{min})}} - 1

        such that :math:`r(b) = r_{max}`\.


        Parameters
        ----------
        r : ndarray(N,)
            One-dimensional array in the co-domain of the transformation.

        Return
        ------
        ndarray(N,)
            Inverse of transformation at r.

        """
        self.set_maximum_parameter_b(r)
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(self.b + 1)
        return np.power(r / self._rmin, 1.0 / power) - 1


class HyperbolicRTransform(BaseTransform):
    r"""
    Hyperbolic transform from :math`[0, \infty)` to :math:`[0, \infty)`\.

    The transformation is given by

    .. math::
        r(x) = \frac{a x}{(1 - bx)},

    where :math:`b ( N - 1) \geq 1`\, and :math:`N` is the number of points in x.

    The inverse transformation is given by

    .. math::
        x(r) = \frac{r}{a + br}

    """

    def __init__(self, a: float, b: float):
        r"""Hyperbolic transform class.

        Parameters
        ----------
        a : float
            parameter a to determine hyperbolic function
        b : float
            parameter b to determine hyperbolic function

        """
        if a <= 0:
            raise ValueError(f"a must be strictly positive.\n  a: {a}")
        if b <= 0:
            raise ValueError(f"b must be strictly positive.\n  b: {b}")
        self._a = a
        self._b = b
        self._domain = (0, np.inf)
        self._codomain = (0, np.inf)

    @property
    def a(self):
        r"""float: value of parameter a."""
        return self._a

    @property
    def b(self):
        r"""float: value of parameter b."""
        return self._b

    def transform(self, x: np.ndarray):
        r"""Perform hyperbolic transformation.

        .. math::
            r_i = \frac{a x_i}{(1 - bx_i)},

        where :math:`b ( N - 1) \geq 1`\, and :math:`N` is the number of points in x.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation :math:`[0,\infty)`\.

        Return
        ------
        ndarray(N,)
            The transformation of x to the co-domain of the transformation.

        """
        if self._b * (x.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        return self._a * x / (1 - self._b * x)

    def deriv(self, x: np.ndarray):
        r"""
        Compute the first derivative of hyperbolic transformation.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation :math:`[0,\infty)`\.

        Return
        ------
        ndarray(N,)
            First derivative of transformation at x.

        """
        if self._b * (x.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        x = 1.0 / (1 - self._b * x)
        return self._a * x * x

    def deriv2(self, x: np.ndarray):
        r"""
        Compute the second derivative of hyperbolic transformation.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation :math:`[0,\infty)`\.

        Return
        ------
        ndarray(N,)
            Second derivative of transformation at x.

        """
        if self._b * (x.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        x = 1.0 / (1 - self._b * x)
        return 2.0 * self._a * self._b * x**3

    def deriv3(self, x: np.ndarray):
        r"""
        Compute the third derivative of hyperbolic transformation.

        Parameters
        ----------
        x : ndarray(N,)
            One-dimensional array in the domain of the transformation :math:`[0,\infty)`\.

        Return
        ------
        ndarray(N,)
            Third derivative of transformation at x.

        """
        if self._b * (x.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        x = 1.0 / (1 - self._b * x)
        return 6.0 * self._a * self._b * self._b * x**4

    def inverse(self, r: np.ndarray):
        r"""
        Compute the inverse of hyperbolic transformation.

        .. math::
            x(r) = \frac{r}{a + br}

        Parameters
        ----------
        r : ndarray(N,)
            One-dimensional array in the domain of the transformation :math:`[0,\infty)`\.

        Return
        ------
        ndarray(N,)
            Inverse transformation at r.

        """
        if self._b * (r.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        return r / (self._a + self._b * r)


class MultiExpRTransform(BaseTransform):
    r"""
    MultiExp Transformation class from :math:`[-1,1]` to :math:`[r_{min}, \infty)`\. [#]_

    The transformation is given by

    .. math::
        r(x) = -R \log \left( \frac{x + 1}{2} \right) + r_{min}

    The inverse of this transformation is given by

    .. math::
        x(r) = 2 \exp \left( \frac{-(r - r_{min})}{R} \right) - 1

    References
    ----------
    .. [#] Gill, Peter MW, and Siu-Hung Chien. "Radial quadrature for multiexponential integrands."
       Journal of computational chemistry 24.6 (2003): 732-740.

    """

    def __init__(self, rmin: float, R: float, trim_inf=True):
        r"""Construct MultiExp transform from :math:`[-1,1]` to :math:`[r_{min}, \infty)`\.

        Parameters
        ----------
        rmin: float
            The minimum coordinate for transformed radial array.
        R: float
            The scale factor for transformed radial array.
        trim_inf : bool, optional
            Flag to trim infinite value in transformed array. If True, it will
            replace np.inf with 1e16. This may cause unexpected errors in the
            following operations.
        """
        self._rmin = rmin
        self._R = R
        self.trim_inf = trim_inf
        self._domain = (-1, 1)
        self._codomain = (rmin, np.inf)

    @property
    def rmin(self):
        r"""float: The minimum value for the transformed radial array."""
        return self._rmin

    @property
    def R(self):
        r"""float: The scale factor for the transformed radial array."""
        return self._R

    def transform(self, x: np.ndarray):
        r"""Transform from [-1,1] to  :math:`[r_{min},\infty)`\.

        .. math::
            r_i = -R \log \left( \frac{x_i + 1}{2} \right) + r_{min}

        Parameters
        ----------
        x : ndarray(N,)
            One dimensional array with values between :math:`[-1,1]`\.

        Returns
        -------
        ndarray(N,)
            Transformed array located between :math:`[r_{min},\infty)`\.

        """
        rf_array = -self._R * np.log((x + 1) / 2) + self._rmin
        if self.trim_inf:
            rf_array = self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r: np.ndarray):
        r"""
        Inverse of transform from :math:`[r_{min},\infty)` to :math:`[-1,1]`\.

        .. math::
            x_i = 2 \exp \left( \frac{-(r_i - r_{min})}{R} \right) - 1

        Parameters
        ----------
        r : np.ndarray(N,)
            One-dimensional array in :math:`[r_{min}, \infty)`\.

        Returns
        -------
        np.ndarray(N,)
            The inverse of transformation in :math:`[-1, 1]`\.

        """
        return 2 * np.exp(-(r - self._rmin) / self._R) - 1

    def deriv(self, x: np.ndarray):
        r"""Compute the first derivative of MultiExp transformation.

        .. math::
            \frac{dr}{dx} = -\frac{R}{1+x}

        Parameters
        ----------
        x : ndarray(N,)
            One dimensional in :math:`[-1, 1]`\.

        Returns
        -------
        ndarray(N,)
            The first derivative of MultiExp transformation at each point.

        """
        return -self._R / (1 + x)

    def deriv2(self, x: np.ndarray):
        r"""Compute the second derivative of MultiExp transformation.

        .. math::
            \frac{d^2r}{dx^2} = \frac{R}{(1 + x)^2}

        Parameters
        ----------
        x : ndarray(N,)
            One dimensional in :math:`[-1,1]`\.

        Returns
        -------
        ndarray(N,)
            The second derivative of MultiExp transformation at each point.

        """
        return self._R / (1 + x) ** 2

    def deriv3(self, x: np.ndarray):
        r"""Compute the third derivative of MultiExp transformation.

        .. math::
            \frac{d^3r}{dx^3} = -\frac{2R}{(1 + x)^3}

        Parameters
        ----------
        x : ndarray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        ndarray(N,)
            The third derivative of MultiExp transformation at each point.

        """
        return -2 * self._R / (1 + x) ** 3


class KnowlesRTransform(BaseTransform):
    r"""
    Knowles Transformation from :math:`[-1, 1]` to :math:`[r_{min}, \infty)`\.

    The transformation is given by

    .. math::
       r(x) = r_{min} - R \log \left( 1 - 2^{-k} (x + 1)^k \right),

    where :math:`k > 0` and :math:`R` is the scaling parameter.

    The inverse transformation is given by

    .. math::
        x(r) = 2 \sqrt[k]{1-\exp \left( -\frac{r-r_{min}}{R}\right)}-1

    """

    def __init__(self, rmin: float, R: float, k: int, trim_inf=True):
        r"""Construct Knowles transformation class.

        Parameters
        ----------
        rmin: float
            The minimum coordinate for transformed radial array.
        R: float
            The scale factor for transformed radial array.
        k: integer k > 0
            Free parameter, k must be > 0.
        trim_inf : bool, optional
            Flag to trim infinite value in transformed array. If True, it will
            replace np.inf with 1e16. This may cause unexpected errors in the
            following operations.

        """
        if k <= 0:
            raise ValueError(f"k needs to be greater than 0, got k = {k}")
        self._rmin = rmin
        self._R = R
        self._k = k
        self.trim_inf = trim_inf
        self._domain = (-1, 1)
        self._codomain = (rmin, np.inf)

    @property
    def rmin(self):
        r"""float: The minimum value for the transformed radial array."""
        return self._rmin

    @property
    def R(self):
        r"""float: The scale factor for the transformed radial array."""
        return self._R

    @property
    def k(self):
        r"""float: Free and extra parameter, k must be > 0."""
        return self._k

    def transform(self, x: np.ndarray):
        r"""Transform from :math:`[-1,1]` to :math:`[r_{min},\infty)`\.

        .. math::
            r_i = r_{min} - R \log \left( 1 - 2^{-k} (x_i + 1)^k \right)

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        np.ndarray(N,)
            One dimensional array in :math:`[r_{min},\infty)`\.

        """
        rf_array = -self._R * np.log(1 - (2**-self._k) * (x + 1) ** self._k) + self._rmin
        if self.trim_inf:
            rf_array = self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r: np.ndarray):
        r"""Inverse of transformation from :math:`[r_{min},\infty)` to :math:`[-1,1]`\.

        .. math::
            x_i = 2 \sqrt[k]{1-\exp \left( -\frac{r_i-r_{min}}{R}\right)}-1

        Parameters
        ----------
        r: ndarray(N,)
            One-dimensional array in :math:`[r_{min},\infty)`\.

        Returns
        -------
        ndarray(N,)
            The inverse transformation in :math:`[-1,1]`\.

        """
        return -1 + 2 * (1 - np.exp((self._rmin - r) / self._R)) ** (1 / self._k)

    def deriv(self, x: np.ndarray):
        r"""Compute the first derivative of Knowles transformation.

        .. math::
            \frac{dr}{dx} = kR \frac{(1+x_i)^{k-1}}{2^k-(1+x_i)^k}

        Parameters
        ----------
        x: ndarray(N,)
            One dimensional  array with values between :math:`[-1,1]`\.

        Returns
        -------
        ndarray(N,)
            The first derivative of Knowles transformation at each point.

        """
        qi = 1 + x
        deriv = self._R * self._k * (qi ** (self._k - 1)) / (2**self._k - qi**self._k)
        if self.trim_inf:
            deriv = self._convert_inf(deriv)
        return deriv

    def deriv2(self, x: np.ndarray):
        r"""Compute the second derivative of Knowles transformation.

        .. math::
            \frac{d^2r}{dx^2} = kR \frac{(1+x_i)^{k-2}
            \left(2^k(k-1) + (1+x_i)^k \right)}{\left( 2^k-(1+x_i)^k\right)^2}

        Parameters
        ----------
        x : ndarray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        ndarray(N,)
            The second derivative of Knowles transformation at each point.

        """
        qi = 1 + x
        return (
            self._R
            * self._k
            * (qi ** (self._k - 2))
            * (2**self._k * (self._k - 1) + qi**self._k)
            / (2**self._k - qi**self._k) ** 2
        )

    def deriv3(self, x: np.ndarray):
        r"""Compute the third derivative of Knowles transformation.

        .. math::
            \frac{d^3r}{dx^3} = kR \frac{(1+x_i)^{k-3}
            \left( 4^k (k-1) (k-2) + 2^k (k-1)(k+4)(1+x_i)^k
                   +2(1+x_i)^k
            \right)}{\left( 2^k - (1+x_i)^k \right)^3}

        Parameters
        ----------
        x: ndarray(N,)
            One dimensional array with values between :math:`[-1,1]`\.

        Returns
        -------
        ndarray(N,)
            The third derivative of Knowles transformation at each point.

        """
        qi = 1 + x
        return (
            self._R
            * self._k
            * (qi ** (self._k - 3))
            * (
                4**self._k * (self._k - 2) * (self._k - 1)
                + 2**self._k * (self._k - 1) * (self._k + 4) * (qi**self._k)
                + 2 * qi ** (2 * self._k)
            )
            / (2**self._k - qi**self._k) ** 3
        )


class HandyRTransform(BaseTransform):
    r"""
    Handy Transformation class from :math:`[-1, 1]` to :math:`[r_{min}, \infty)`\.

    This transformation is given by

    .. math::
        r(x) = R \left( \frac{1+x}{1-x} \right)^m + r_{min}.

    The inverse transformations is given by

    .. math::
        x(r) = \frac{\sqrt[m]{r-r_{min}} - \sqrt[m]{R}} {\sqrt[m]{r-r_{min}} + \sqrt[m]{R}}.

    """

    def __init__(self, rmin: float, R: float, m: int, trim_inf=True):
        r"""Construct Handy transformation.

        Parameters
        ----------
        rmin: float
            The minimum coordinate for transformed radial array.
        R: float
            The scale factor for transformed radial array.
        m: integer m > 0
            Free parameter, m must be > 0.
        trim_inf : bool, optional
            Flag to trim infinite value in transformed array. If True, it will
            replace np.inf with 1e16. This may cause unexpected errors in the
            following operations.

        """
        if m <= 0:
            raise ValueError(f"m needs to be greater than 0, got m = {m}")
        self._rmin = rmin
        self._R = R
        self._m = m
        self.trim_inf = trim_inf
        self._domain = (-1, 1)
        self._codomain = (rmin, np.inf)

    @property
    def rmin(self):
        r"""float: The minimum value for the transformed radial array."""
        return self._rmin

    @property
    def R(self):
        r"""float: The scale factor for the transformed radial array."""
        return self._R

    @property
    def m(self):
        r"""integer: Free and extra parameter, m must be > 0."""
        return self._m

    def transform(self, x: np.ndarray):
        r"""Transform from :math:`[-1,1]` to :math:`[r_{min},\infty)`\.

        .. math::
            r_i = R \left( \frac{1+x_i}{1-x_i} \right)^m + r_{min}

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        np.ndarray(N,)
            One dimensional array in :math:`[r_{min},\infty)`\.

        """
        rf_array = self._R * ((1 + x) / (1 - x)) ** self._m + self._rmin
        if self.trim_inf:
            rf_array = self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r: np.ndarray):
        r"""Inverse transform from :math:`[r_{min},\infty)` to :math:`[-1,1]`\.

        .. math::
            x_i = \frac{\sqrt[m]{r_i-r_{min}} - \sqrt[m]{R}}
                       {\sqrt[m]{r_i-r_{min}} + \sqrt[m]{R}}

        Parameters
        ----------
        r : np.ndarray(N,)
            One dimensional array in :math:`[r_{min},\infty)`\.

        Returns
        -------
        np.ndarray(N,)
            One-dimensional array in :math:`[-1,1]`\.

        """
        tmp_ri = (r - self._rmin) ** (1 / self._m)
        tmp_R = self._R ** (1 / self._m)

        return (tmp_ri - tmp_R) / (tmp_ri + tmp_R)

    def deriv(self, x: np.ndarray):
        r"""Compute the first derivative of Handy transformation.

        .. math::
            \frac{dr}{dx} = 2mR \frac{(1+x)^{m-1}}{(1-x)^{m+1}}

        Parameters
        ----------
        x: ndarray(N,)
            One dimensional array with values between :math:`[-1,1]`\.

        Returns
        -------
        ndarray(N,)
            The first derivative of Handy transformation at each point.
        """
        dr = 2 * self._m * self._R * (1 + x) ** (self._m - 1) / (1 - x) ** (self._m + 1)
        if self.trim_inf:
            dr = self._convert_inf(dr)
        return dr

    def deriv2(self, x: np.ndarray):
        r"""Compute the second derivative of Handy transformation.

        .. math::
            \frac{d^2r}{dx^2} = 4mR (m + x) \frac{(1+x)^{m-2}}{(1-x)^{m+2}}

        Parameters
        ----------
        x : ndarray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        ndarray(N,)
            The second derivative of Handy transformation at each point.
        """
        dr = (
            4
            * self._m
            * self._R
            * (self._m + x)
            * (1 + x) ** (self._m - 2)
            / (1 - x) ** (self._m + 2)
        )

        if self.trim_inf:
            dr = self._convert_inf(dr)
        return dr

    def deriv3(self, x: np.ndarray):
        r"""Compute the third derivative of Handy transformation.

        .. math::
            \frac{d^3r}{dx^3} = 4mR ( 1 + 6 m x + 2 m^2 + 3 x^2)
            \frac{(1+x)^{m-3}}{(1-x)^{m+3}}

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        np.ndarray(N,)
            The third derivative of Handy transformation at each point.
        """
        dr = (
            4
            * self._m
            * self._R
            * (1 + 6 * self._m * x + 2 * self._m**2 + 3 * x**2)
            * (1 + x) ** (self._m - 3)
            / (1 - x) ** (self._m + 3)
        )

        if self.trim_inf:
            dr = self._convert_inf(dr)
        return dr


class HandyModRTransform(BaseTransform):
    r"""Modified Handy Transformation class from :math:`[-1, 1]` to :math:`[r_{min}, r_{max}]`\.

    This transformation is given by

    .. math::
        r(x) = \frac{(1+x)^m (r_{max} - r_{min})}
                { 2^m (1 - 2^m + r_{max} - r_{min})
                  - (1 + x)^m (r_{max} - r_{min} - 2^m )} + r_{min},

    where :math:`m > 0`\.

    The inverse transformation is given by

    .. math::
        x(r) =  2 \sqrt[m]{
                \frac{(r - r_{min})(r_{max} - r_{min} - 2^m + 1)}
                {(r - r_{min})(r_{max} - r_{min} - 2^m) + r_{max} - r_{min}}
                } - 1.

    """

    def __init__(self, rmin: float, rmax: float, m: int, trim_inf=True):
        r"""Construct a modified Handy transform from :math:`[-1, 1]` to :math:`[r_{min}, r_{max}]`\.

        Parameters
        ----------
        rmin: float
            The minimum coordinate for transformed radial array.
        rmax: float
            The maximum coordinate for transformed radial array.
        m: integer m > 0
            Free parameter, m must be > 0.
        trim_inf : bool, optional
            Flag to trim infinite value in transformed array. If True, it will
            replace np.inf with 1e16. This may cause unexpected errors in the
            following operations.
        """
        if m <= 0:
            raise ValueError(f"m needs to be greater than 0, got m = {m}")

        if rmax < rmin:
            raise ValueError(f"rmax needs to be greater than rmin. rmax : {rmax}, rmin : {rmin}.")
        self._rmin = rmin
        self._rmax = rmax
        self._m = m
        self.trim_inf = trim_inf
        self._domain = (-1, 1)
        self._codomain = (rmin, rmax)

    @property
    def rmin(self):
        r"""float: The minimum value for the transformed radial array."""
        return self._rmin

    @property
    def rmax(self):
        r"""float: The maximum value for the transformed radial array."""
        return self._rmax

    @property
    def m(self):
        r"""integer: Free and extra parameter, m must be > 0."""
        return self._m

    def transform(self, x: np.ndarray):
        r"""Transform given array :math:`[-1,1]` to radial array :math:`[r_{min},r_{max}]`\.

        .. math::
            r_i = \frac{(1+x_i)^m (r_{max} - r_{min})}
                { 2^m (1 - 2^m + r_{max} - r_{min})
                  - (1 + x_i)^m (r_{max} - r_{min} - 2^m )} + r_{min}

        Parameters
        ----------
        x: ndarray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        ndarray(N,)
            One dimensional array in :math:`[r_{min},r_{max}]`\.

        """
        two_m = 2**self._m
        size_r = self._rmax - self._rmin
        qi = (1 + x) ** self._m

        rf_array = qi * size_r / (two_m * (1 - two_m + size_r) - qi * (size_r - two_m))
        rf_array += self._rmin

        if self.trim_inf:
            rf_array = self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r: np.ndarray):
        r"""Inverse transform from :math:`[r_{min},r_{max}]` to :math:`[-1,1]`\.

        .. math::
            x_i = 2 \sqrt[m]{
                \frac{(r_i - r_{min})(r_{max} - r_{min} - 2^m + 1)}
                {(r_i - r_{min})(r_{max} - r_{min} - 2^m) + r_{max} - r_{min}}
                } - 1

        Parameters
        ----------
        r : ndarrray(N,)
            One dimensional array in :math:`[r_{min},\infty)`\.

        Returns
        -------
        ndarrray(N,)
            The original one dimensional array in :math:`[-1,1]`\.

        """
        two_m = 2**self._m
        size_r = self._rmax - self._rmin

        tmp_r = (
            (r - self._rmin) * (size_r - two_m + 1) / ((r - self._rmin) * (size_r - two_m) + size_r)
        )

        return 2 * (tmp_r) ** (1 / self._m) - 1

    def deriv(self, x: np.ndarray):
        r"""Compute the first derivative of modified Handy transformation.

        .. math::
            \frac{dr}{dx} = -\frac{
                2^m m (r_{max}-r_{min})(2^m-r_{max}+r_{min}-1)(1+x)^{m-1}}
                {\left( 2^m (2^m-1-r_{max}+r_{min})-(2^m-r_{max}
                + r_{min})(1 + x)^m\right)^2}

        Parameters
        ----------
        x: ndarrray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        ndarrray(N,)
            The first derivative of Handy transformation at each point.

        """
        two_m = 2**self._m
        size_r = self._rmax - self._rmin
        deriv = (
            -(self._m * two_m * (two_m - size_r - 1) * size_r * (1 + x) ** (self._m - 1))
            / (two_m * (1 - two_m + size_r) + (two_m - size_r) * (1 + x) ** self._m) ** 2
        )
        if self.trim_inf:
            deriv = self._convert_inf(deriv)
        return deriv

    def deriv2(self, x: np.ndarray):
        r"""Compute the second derivative of modified Handy transformation.

        Parameters
        ----------
        x: ndarrray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        ndarrray(N,)
            The second derivative of Handy transformation at each point.
        """
        two_m = 2**self._m
        size_r = self._rmax - self._rmin
        return (
            -(
                self._m
                * two_m
                * (two_m - size_r - 1)
                * size_r
                * (1 + x) ** (self._m - 2)
                * (
                    -two_m * (self._m - 1) * (two_m - size_r - 1)
                    - (self._m + 1) * (two_m - size_r) * (1 + x) ** (self._m)
                )
            )
            / (two_m * (1 - two_m + size_r) + (two_m - size_r) * (1 + x) ** self._m) ** 3
        )

    def deriv3(self, x: np.ndarray):
        r"""Compute the third derivative of modified Handy transformation.

        Parameters
        ----------
        x: ndarrray(N,)
            One dimensional array in :math:`[-1,1]`\.

        Returns
        -------
        ndarrray(N,)
            The third derivative of Handy transformation at each point.
        """
        two_m = 2**self._m
        size_r = self._rmax - self._rmin
        return (
            -(
                self._m
                * two_m
                * size_r
                * (two_m - size_r - 1)
                * (1 + x) ** (self._m - 3)
                * (
                    2 * two_m * (self._m - 2) * (self._m - 1) * (1 - two_m + size_r) ** 2
                    + 2 ** (self._m + 2)
                    * (self._m - 1)
                    * (self._m + 1)
                    * (two_m - 1 - size_r)
                    * (two_m - size_r)
                    * (1 + x) ** self._m
                    + (self._m + 2)
                    * (self._m + 1)
                    * (two_m - size_r) ** 2
                    * (x + 1) ** (2 * self._m)
                )
            )
            / (two_m * (1 - two_m + size_r) + (two_m - size_r) * (1 + x) ** self._m) ** 4
        )
