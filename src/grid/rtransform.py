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
"""Transformation from uniform 1D to non-uniform 1D grids."""


import warnings
from abc import ABC, abstractmethod
from numbers import Number

from grid.basegrid import OneDGrid

import numpy as np


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
        """Abstract method for 2nd derivative of transformation."""

    @abstractmethod
    def deriv3(self, x):
        """Abstract method for 3rd derivative of transformation."""

    def transform_1d_grid(self, oned_grid):
        r"""Generate a new integral grid by transforming given the OneDGrid.

        .. math::
            \int^{\inf}_0 g(r) d r &= \int^b_a g(r(x)) \frac{dr}{dx} dx \\
                                   &= \int^b_a f(x) \frac{dr}{dx} dx \\
            w^r_n = w^x_n \cdot \frac{dr}{dx} \vert_{x_n}

        Parameters
        ----------
        oned_grid : OneDGrid
            An instance of 1D grid.

        Returns
        -------
        OneDGrid
            Transformed 1D grid spanning a different domain.

        Raises
        ------
        TypeError
            Input is not a proper OneDGrid instance.
        """
        if not isinstance(oned_grid, OneDGrid):
            raise TypeError(f"Input grid is not OneDGrid, got {type(oned_grid)}")
        new_points = self.transform(oned_grid.points)
        new_weights = self.deriv(oned_grid.points) * oned_grid.weights
        new_domain = oned_grid.domain
        if new_domain is not None:
            new_domain = self.transform(np.array(oned_grid.domain))
        return OneDGrid(new_points, new_weights, new_domain)

    def _convert_inf(self, array, replace_inf=1e16):
        """Convert np.inf(float) to 1e16(float) in case of numerical failure.

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


class BeckeTF(BaseTransform):
    """Becke Transformation class."""

    def __init__(self, rmin, R, trim_inf=True):
        """Construct Becke transform, [-1, 1] -> [r_0, inf).

        Parameters
        ----------
        rmin : float
            The minimum coordinates for transformed array.
        R : float
            The scale factor for transformed array.
        trim_inf : bool, optional
            Flag to trim infinite value in transformed array. If True, it will
            replace np.inf with 1e16. This may cause unexpected errors in the
            following operations.

        """
        self._rmin = rmin
        self._R = R
        self.trim_inf = trim_inf

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
    #     R = BeckeTF.find_parameter(oned_grid.points, rmin, radius)
    #     tfm = cls(rmin=rmin, R=R)
    #     new_points = tfm.transform(oned_grid.points)
    #     new_weights = tfm.deriv(oned_grid.points) * oned_grid.weights
    #     return RadialGrid(new_points, new_weights), tfm

    @staticmethod
    def find_parameter(array, rmin, radius):
        """Compute for optimal R for certain atom, given array and rmin.

        To find the proper R value to make sure half points are within atomic
        radius r.

        Parameters
        ----------
        array : np.ndarray(N,)
            one dimention array locates within [-1, 1]
        rmin : float
            Minimum value for transformed array.
        radius : float
            Atomic radius of interest

        Returns
        -------
        float
            The optimal value of scale factor R

        Raises
        ------
        ValueError
            rmin needs to be smaller than atomic radius to compute R
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

    def transform(self, x):
        r"""Transform given array[-1, 1] to array[rmin, inf).

        .. math::
            r_i = R \frac{1 + x_i}{1 - x_i} + r_{min}

        Parameters
        ----------
        x : np.ndarray(N,)
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray(N,)
            Transformed array located between [rmin, inf)
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rf_array = self._R * (1 + x) / (1 - x) + self._rmin
        if self.trim_inf:
            rf_array = self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r):
        r"""Transform array[rmin, inf) back to original array[-1, 1].

        .. math::
            x_i = \frac{r_i - r_{min} - R} {r_i - r_{min} + R}

        Parameters
        ----------
        r : np.ndarray(N,)
            Sorted one dimension array located between [rmin, inf)

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array located between [-1, 1]
        """
        return (r - self._rmin - self._R) / (r - self._rmin + self._R)

    def deriv(self, x):
        r"""Compute the 1st derivative of Becke transformation.

        .. math::
            \frac{dr_i}{dx_i} = 2R \frac{1}{(1-x)^2}

        Parameters
        ----------
        x : np.array(N,)
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray(N,)
            1st derivative of Becke transformation at each points
        """
        return 2 * self._R / ((1 - x) ** 2)

    def deriv2(self, x):
        r"""Compute the 2nd derivative of Becke transformation.

        .. math::
            \frac{d^2r}{dx^2} = 4R \frac{1}{1-x^3}

        Parameters
        ----------
        x : np.array(N,)
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray(N,)
            2nd derivative of Becke transformation at each points
        """
        return 4 * self._R / (1 - x) ** 3

    def deriv3(self, x):
        r"""Compute the 3rd derivative of Becke transformation.

        .. math::
            \frac{d^3r}{dx^3} = 12R \frac{1}{1 - x^4}

        Parameters
        ----------
        x : np.array(N,)
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray(N,)
            3rd derivative of Becke transformation at each points
        """
        return 12 * self._R / (1 - x) ** 4


class LinearTF(BaseTransform):
    """Linear transformation class."""

    def __init__(self, rmin, rmax):
        """Construct linear transformation instance.

        Parameters
        ----------
        rmin : float
            Minimum value for transformed grid
        rmax : float
            Maximum value for transformed grid
        """
        self._rmin = rmin
        self._rmax = rmax

    def transform(self, x):
        r"""Transform onedgrid form [-1, 1] to [rmin, rmax].

        .. math::
            r_i = \frac{r_{max} - r_{min}}{2} (1 + x_i)

        Parameters
        ----------
        x : float or np.ndarray
            number or arrays to be transformed

        Returns
        -------
        float or np.ndarray
            Transformed points between [rmin, rmax]
        """
        return (self._rmax - self._rmin) / 2 * (1 + x) + self._rmin

    def deriv(self, x):
        r"""Compute the 1st order derivative.

        .. math::
            \frac{dr}{dx} = \frac{r_{max} - r_{min}}{2}

        Parameters
        ----------
        x : float or np.ndarray
            number or arrays to be transformed

        Returns
        -------
        float or np.ndarray
            1st order derivative at given points
        """
        if isinstance(x, Number):
            return (self._rmax - self._rmin) / 2
        else:
            return np.ones(x.size) * (self._rmax - self._rmin) / 2

    def deriv2(self, x):
        r"""Compute the 2nd order derivative.

        .. math::
            \frac{d^2 r}{dx^2} = 0

        Parameters
        ----------
        x : float or np.ndarray
            number or arrays to be transformed

        Returns
        -------
        float or np.ndarray
            2nd order derivative at given points
        """
        return np.array(0) if isinstance(x, Number) else np.zeros(x.size)

    def deriv3(self, x):
        r"""Compute the 3rd order derivative.

        .. math::
            \frac{d^2 r}{dx^2} = 0

        Parameters
        ----------
        x : float or np.ndarray
            number or arrays to be transformed

        Returns
        -------
        float or np.ndarray
            3rd order derivative at given points
        """
        return np.array(0) if isinstance(x, Number) else np.zeros(x.size)

    def inverse(self, r):
        r"""Compute the inverse of the transformation.

        .. math::
            x_i = \frac{2 r_i - (r_{max} + r_{min})}{r_{max} - r_{min}}

        Parameters
        ----------
        r : float or np.ndarray
            transformed number or arrays

        Returns
        -------
        float or np.ndarray
            Original number of array before the transformation
        """
        return (2 * r - (self._rmax + self._rmin)) / (self._rmax - self._rmin)


class InverseTF(BaseTransform):
    """Inverse transformation class, [rmin, rmax] or [rmin, inf) -> [-1, 1]."""

    def __init__(self, transform):
        """Construct InverseTF instance.

        Parameters
        ----------
        transform : BaseTransform
            Basic one dimension transformation instance

        Raises
        ------
        TypeError
            The input need to be a BaseTransform instance
        """
        if not isinstance(transform, BaseTransform):
            raise TypeError(
                f"Input need to be a transform instance, got {type(transform)}."
            )
        self._tfm = transform

    def transform(self, r):
        """Transform array back to original one dimension array.

        InvTF.transfrom is equivalent to OriginTF.inverse
        InvTF.inverse is equivalent to OriginTF.transform

        Parameters
        ----------
        r : np.ndarray(N,)
            Grid locates between [rmin, rmax] or [rmin, inf) depends on the
            domain of its original transformation

        Returns
        -------
        np.ndarray(N,)
            Original one dimension array locates between [-1, 1]
        """
        return self._tfm.inverse(r)

    def inverse(self, x):
        """Transform one dimension array[-1, 1] to array [rmin, rmax(inf)].

        Parameters
        ----------
        x : np.ndarray
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray
            numpy array located between [rmin, rmax(inf)]
        """
        return self._tfm.transform(x)

    def _d1(self, r):
        """Compute 1st order derivative of the original transformation.

        Parameters
        ----------
        r : np.ndarray(n)
            Transformed r value

        Returns
        -------
        np.ndarray(n,)
            1st order derivative array

        Raises
        ------
        ZeroDivisionError
            Raised when there is 0 value in returned derivative values
        """
        d1 = self._tfm.deriv(r)
        if np.any(d1 == 0):
            raise ZeroDivisionError(
                "First derivative of original transformation has 0 value"
            )
        return d1

    def deriv(self, r):
        r"""Compute the 1st derivative of inverse transformation.

        .. math::
            \frac{dx}{dr} = (\frac{dr}{dx})^{-1}

        Parameters
        ----------
        r : np.array(N,)
            One dimension numpy array located between [ro, rmax(inf)]

        Returns
        -------
        np.ndarray(N,)
            1st derivative of inverse transformation at each points
        """
        # x: inverse x array, d1: first derivative
        r = self._tfm.inverse(r)
        return 1 / self._d1(r)

    def deriv2(self, r):
        r"""Compute the 2nd derivative of inverse transformation.

        .. math::
            \frac{d^2 x}{dr^2} = - \frac{d^2 r}{dx^2} (\frac{dx}{dr})^3


        Parameters
        ----------
        r : np.array(N,)
            One dimension numpy array located between [ro, rmax(inf)]

        Returns
        -------
        np.ndarray(N,)
            2nd derivative of inverse transformation at each points
        """
        # x: inverse x array, d1: first derivative
        # d2: second derivative d^2x / dy^2
        r = self._tfm.inverse(r)
        d2 = self._tfm.deriv2
        return -d2(r) / self._d1(r) ** 3

    def deriv3(self, r):
        r"""Compute the 3rd derivative of inverse transformation.

        .. math::
            \frac{d^3 x}{dr^3} = -\frac{d^3 r}{dx^3} (\frac{dx}{dr})^4
                               + 3 (\frac{d^2 r}{dx^2})^2 (\frac{dx}{dr})^5

        Parameters
        ----------
        r : np.array(N,)
            One dimension numpy array located between [ro, rmax(inf)]

        Returns
        -------
        np.ndarray(N,)
            3rd derivative of inverse transformation at each points
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
    """Identity Transform class."""

    def transform(self, x: np.ndarray):
        """Perform given array into itself."""
        return x

    def deriv(self, x: np.ndarray):
        """Compute the first derivative of identity transform."""
        return 1 if isinstance(x, Number) else np.ones(x.size)

    def deriv2(self, x: np.ndarray):
        """Compute the second Derivative of identity transform."""
        return 0 if isinstance(x, Number) else np.zeros(x.size)

    def deriv3(self, x: np.ndarray):
        """Compute the third Derivative of identity transform."""
        return 0 if isinstance(x, Number) else np.zeros(x.size)

    def inverse(self, r: np.ndarray):
        """Compute the inverse of identity transform."""
        return r


class LinearRTransform(BaseTransform):
    """Linear transform class."""

    def __init__(self, rmin: float, rmax: float):
        """Initialize linear transform class.

        Parameters
        ----------
        rmin : float
            Define the lower end of the linear transform
        rmax : float
            Define the upper end of the linear transform

        Raises
        ------
        ValueError
            Value of rmin is larger than rmax
        """
        if rmin >= rmax:
            raise ValueError(
                f"rmin need to be larger than rmax.\n  rmin: {rmin}, rmax: {rmax}"
            )
        self._rmin = rmin
        self._rmax = rmax

    @property
    def rmin(self):
        """float: rmin value of the tf."""
        return self._rmin

    @property
    def rmax(self):
        """float: rmax value of the tf."""
        return self._rmax

    def transform(self, x: np.ndarray):
        """Perform linear transformation."""
        alpha = (self._rmax - self._rmin) / (x.size - 1)
        return alpha * x + self._rmin

    def deriv(self, x: np.ndarray):
        """Compute the first derivative of linear transformation."""
        alpha = (self._rmax - self._rmin) / (x.size - 1)
        return np.ones(x.size) * alpha

    def deriv2(self, x: np.ndarray):
        """Compute the second derivative of linear transformation."""
        return np.zeros(x.size)

    def deriv3(self, x: np.ndarray):
        """Compute the third derivative of linear transformation."""
        return np.zeros(x.size)

    def inverse(self, r: np.ndarray):
        """Compute the inverse of linear transformation."""
        r = np.array(r)
        alpha = (self._rmax - self._rmin) / (r.size - 1)
        return (r - self._rmin) / alpha


#     def to_string(self):
#         return " ".join(
#             ["LinearRTransform", repr(self.rmin), repr(self.rmax), repr(self.npoint)]
#         )

#     def chop(self, npoint):
#         rmax = self.radius(npoint - 1)
#         return LinearRTransform(self.rmin, rmax, npoint)

#     def half(self):
#         if self.npoint % 2 != 0:
#             raise ValueError(
#                 "Half method can only be called on a rtransform with an even number of points."
#             )
#         rmin = self.radius(1)
#         return LinearRTransform(rmin, self.rmax, self.npoint / 2)


class ExpRTransform(BaseTransform):
    """Exponential transform class."""

    def __init__(self, rmin: float, rmax: float):
        """Initialize exp transform instance.

        Parameters
        ----------
        rmin : float
            Min value for transformed points
        rmax : float
            Max value for transformed points


        Raises
        ------
        ValueError
            If rmin larger than rmax or one of them is negative.
        """
        if rmin < 0 or rmax < 0:
            raise ValueError(
                f"rmin or rmax need to be positive\n  rmin: {rmin}, rmax: {rmax}"
            )
        if rmin >= rmax:
            raise ValueError(
                f"rmin need to be smaller than rmax\n  rmin: {rmin}, rmax: {rmax}"
            )
        self._rmin = rmin
        self._rmax = rmax

    @property
    def rmin(self):
        """float: the value of rmin."""
        return self._rmin

    @property
    def rmax(self):
        """float: the value of rmax."""
        return self._rmax

    def transform(self, x: np.ndarray):
        """Perform exponential transform."""
        alpha = np.log(self._rmax / self._rmin) / (x.size - 1)
        return self._rmin * np.exp(x * alpha)

    def deriv(self, x: np.ndarray):
        """Compute the first derivative of exponential transform."""
        alpha = np.log(self._rmax / self._rmin) / (x.size - 1)
        return self.transform(x) * alpha

    def deriv2(self, x: np.ndarray):
        """Compute the second derivative of exponential transform."""
        alpha = np.log(self._rmax / self._rmin) / (x.size - 1)
        return self.deriv(x) * alpha

    def deriv3(self, x: np.ndarray):
        """Compute the third derivative of exponential transform."""
        alpha = np.log(self._rmax / self._rmin) / (x.size - 1)
        return self.deriv2(x) * alpha

    def inverse(self, r: np.ndarray):
        """Compute the inverse of exponential transform."""
        alpha = np.log(self._rmax / self._rmin) / (r.size - 1)
        return np.log(r / self._rmin) / alpha


#     def to_string(self):
#         return " ".join(
#             ["ExpRTransform", repr(self.rmin), repr(self.rmax), repr(self.npoint)]
#         )

#     def chop(self, npoint):
#         rmax = self.radius(npoint - 1)
#         return ExpRTransform(self.rmin, rmax, npoint)

#     def half(self):
#         if self.npoint % 2 != 0:
#             raise ValueError(
#                 "Half method can only be called on a rtransform with an even number of points."
#             )
#         rmin = self.radius(1)
#         return ExpRTransform(rmin, self.rmax, self.npoint / 2)


class PowerRTransform(BaseTransform):
    """Power transform class."""

    def __init__(self, rmin: float, rmax: float):
        """Initialize power transform instance.

        Parameters
        ----------
        rmin : float
            Min value for transformed points
        rmax : float
            Max value for transformed points

        Raises
        ------
        ValueError
            value of rmin larger than rmax or one of them is negative
        """
        if rmin >= rmax:
            raise ValueError("rmin must be smaller rmax.")
        if rmin <= 0 or rmax <= 0:
            raise ValueError("rmin and rmax must be positive.")
        self._rmin = rmin
        self._rmax = rmax

    @property
    def rmin(self):
        """float: the value of rmin."""
        return self._rmin

    @property
    def rmax(self):
        """float: the value of rmax."""
        return self._rmax

    def transform(self, x: np.ndarray):
        """Perform power transform."""
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(x.size)
        if power < 2:
            warnings.warn(
                f"power need to be larger than 2\n  power: {power}", RuntimeWarning
            )
        return self._rmin * np.power(x + 1, power)

    def deriv(self, x: np.ndarray):
        """Compute first derivative of power transform."""
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(x.size)
        return power * self._rmin * np.power(x + 1, power - 1)

    def deriv2(self, x: np.ndarray):
        """Compute second derivative of power transform."""
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(x.size)
        return power * (power - 1) * self._rmin * np.power(x + 1, power - 2)

    def deriv3(self, x: np.ndarray):
        """Compute third derivative of power transform."""
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(x.size)
        return (
            power * (power - 1) * (power - 2) * self._rmin * np.power(x + 1, power - 3)
        )

    def inverse(self, r: np.ndarray):
        """Compute the inverse of power transform."""
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(r.size)
        return np.power(r / self._rmin, 1.0 / power) - 1


#     def to_string(self):
#         return " ".join(
#             ["PowerRTransform", repr(self.rmin), repr(self.rmax), repr(self.npoint)]
#         )

#     def chop(self, npoint):
#         rmax = self.radius(npoint - 1)
#         return PowerRTransform(self.rmin, rmax, npoint)

#     def half(self):
#         if self.npoint % 2 != 0:
#             raise ValueError(
#                 "Half method can only be called on a rtransform with an even number of points."
#             )
#         rmin = self.radius(1)
#         return PowerRTransform(rmin, self.rmax, self.npoint / 2)


class HyperbolicRTransform(BaseTransform):
    """Hyperbolic transform class."""

    def __init__(self, a, b):
        """Hyperbolic transform class.

        Parameters
        ----------
        a : float
            parameter a to determine hyperbolic function
        b : float
            parameter b to determine hyperbolic function

        Raises
        ------
        ValueError
            Either a or b is negative.
        """
        if a <= 0:
            raise ValueError(f"a must be strctly positive.\n  a: {a}")
        if b <= 0:
            raise ValueError(f"b must be strctly positive.\n  b: {b}")
        self._a = a
        self._b = b

    @property
    def a(self):
        """float: value of parameter a."""
        return self._a

    @property
    def b(self):
        """float: value of parameter b."""
        return self._b

    def transform(self, x: np.ndarray):
        """Perform hyperbolic transformation."""
        if self._b * (x.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        return self._a * x / (1 - self._b * x)

    def deriv(self, x: np.ndarray):
        """Compute the first derivative of hyperbolic transformation."""
        if self._b * (x.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        x = 1.0 / (1 - self._b * x)
        return self._a * x * x

    def deriv2(self, x: np.ndarray):
        """Compute the second derivative of hyperbolic transformation."""
        if self._b * (x.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        x = 1.0 / (1 - self._b * x)
        return 2.0 * self._a * self._b * x ** 3

    def deriv3(self, x: np.ndarray):
        """Compute the third derivative of hyperbolic transformation."""
        if self._b * (x.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        x = 1.0 / (1 - self._b * x)
        return 6.0 * self._a * self._b * self._b * x ** 4

    def inverse(self, r: np.ndarray):
        """Compute the inverse of hyperbolic transformation."""
        if self._b * (r.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        return r / (self._a + self._b * r)


#     def to_string(self):
#         return " ".join(
#             ["HyperbolicRTransform", repr(self.a), repr(self.b), repr(self.npoint)]
#         )


class MultiExpTF(BaseTransform):
    """MultiExp Transformation class."""

    def __init__(self, rmin, R, trim_inf=True):
        """Construct MultiExp transform [-1,1] -> [rmin,inf).

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

    @property
    def rmin(self):
        """float: The minimum value for the transformed radial array."""
        return self._rmin

    @property
    def R(self):
        """float: The scale factor for the transformed radial array."""
        return self._R

    def transform(self, x):
        r"""Transform given array [-1,1] to radial array [rmin,inf).

        .. math::
            r_i = -R \log \left( \frac{x_i + 1}{2} \right) + r_{min}

        Parameters
        ----------
        x : np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            Transformed array located between [rmin,inf).
        """
        rf_array = -self._R * np.log((x + 1) / 2) + self._rmin
        if self.trim_inf:
            rf_array = self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r):
        r"""Transform array [rmin,inf) back to original array [-1,1].

        .. math::
            x_i = 2 \exp \left( \frac{-(r_i - r_{min})}{R} \right) - 1

        Parameters
        ----------
        r : np.ndarray(N,)
            Sorted one dimension radial array with values bewteen [rmin, inf).

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array located bewteen [-1, 1].
        """
        return 2 * np.exp(-(r - self._rmin) / self._R) - 1

    def deriv(self, x):
        r"""Compute the 1st derivative of MultiExp transformation.

        .. math::
            \frac{dr}{dx} = -\frac{R}{1+x}

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimension numpy array with values between [-1, 1].

        Returns
        -------
        np.ndarray(N,)
            The first derivative of MultiExp transformation at each point.
        """
        return -self._R / (1 + x)

    def deriv2(self, x):
        r"""Compute the 2nd derivative of MultiExp transformation.

        .. math::
            \frac{d^2r}{dx^2} = \frac{R}{(1 + x)^2}

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The second derivative of MultiExp transformation at each point.
        """
        return self._R / (1 + x) ** 2

    def deriv3(self, x):
        r"""Compute the 3rd derivative of MultiExp transformation.

        .. math::
            \frac{d^3r}{dx^3} = -\frac{2R}{(1 + x)^3}

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The third derivative of MultiExp transformation at each point.
        """
        return -2 * self._R / (1 + x) ** 3


class KnowlesTF(BaseTransform):
    """Knowles Transformation class."""

    def __init__(self, rmin, R, k, trim_inf=True):
        """Construct Knowles transform [-1,1] -> [rmin,inf).

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
            raise ValueError(f"k needs to be greater than 0.")
        self._rmin = rmin
        self._R = R
        self._k = k
        self.trim_inf = trim_inf

    @property
    def rmin(self):
        """float: The minimum value for the transformed radial array."""
        return self._rmin

    @property
    def R(self):
        """float: The scale factor for the transformed radial array."""
        return self._R

    @property
    def k(self):
        """float: Free and extra parameter, k must be > 0."""
        return self._k

    def transform(self, x):
        r"""Transform given array [-1,1] to radial array [rmin,inf).

        .. math::
            r_i = r_{min} - R \log \left( 1 - 2^{-k} (x_i + 1)^k \right)

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimension numpy array with values between [-1,1]

        Returns
        -------
        np.ndarray(N,)
            One dimension numpy array with values between [rmin,inf).
        """
        rf_array = (
            -self._R * np.log(1 - (2 ** -self._k) * (x + 1) ** self._k) + self._rmin
        )
        if self.trim_inf:
            rf_array = self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r):
        r"""Transform radiar array [rmin,inf) back to original array [-1,1].

        .. math::
            x_i = 2 \sqrt[k]{1-\exp \left( -\frac{r_i-r_{min}}{R}\right)}-1

        Parameters
        ----------
        r: np.ndarray(N,)
            Sorted one dimension radial array with values bewteen [rmin,inf).

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array with values bewteen [-1,1]
        """
        return -1 + 2 * (1 - np.exp((self._rmin - r) / self._R)) ** (1 / self._k)

    def deriv(self, x):
        r"""Compute the 1st derivative of Knowles transformation.

        .. math::
            \frac{dr}{dx} = kR \frac{(1+x_i)^{k-1}}{2^k-(1+x_i)^k}

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The first derivative of Knowles transformation at each point.
        """
        qi = 1 + x
        return (
            self._R * self._k * (qi ** (self._k - 1)) / (2 ** self._k - qi ** self._k)
        )

    def deriv2(self, x):
        r"""Compute the 2nd derivative of Knowles transformation.

        .. math::
            \frac{d^2r}{dx^2} = kR \frac{(1+x_i)^{k-2}
            \left(2^k(k-1) + (1+x_i)^k \right)}{\left( 2^k-(1+x_i)^k\right)^2}

        Parameters
        ----------
        x : np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The second derivative of Knowles transformation at each point.
        """
        qi = 1 + x
        return (
            self._R
            * self._k
            * (qi ** (self._k - 2))
            * (2 ** self._k * (self._k - 1) + qi ** self._k)
            / (2 ** self._k - qi ** self._k) ** 2
        )

    def deriv3(self, x):
        r"""Compute the 3rd derivative of Knowles transformation.

        .. math::
            \frac{d^3r}{dx^3} = kR \frac{(1+x_i)^{k-3}
            \left( 4^k (k-1) (k-2) + 2^k (k-1)(k+4)(1+x_i)^k
                   +2(1+x_i)^k
            \right)}{\left( 2^k - (1+x_i)^k \right)^3}

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The third derivative of Knowles transformation at each point.
        """
        qi = 1 + x
        return (
            self._R
            * self._k
            * (qi ** (self._k - 3))
            * (
                4 ** self._k * (self._k - 2) * (self._k - 1)
                + 2 ** self._k * (self._k - 1) * (self._k + 4) * (qi ** self._k)
                + 2 * qi ** (2 * self._k)
            )
            / (2 ** self._k - qi ** self._k) ** 3
        )


class HandyTF(BaseTransform):
    """Handy Transformation class."""

    def __init__(self, rmin, R, m, trim_inf=True):
        """Construct Handy transform [-1,1] -> [rmin,inf).

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
            raise ValueError(f"m needs to be greater than 0.")
        self._rmin = rmin
        self._R = R
        self._m = m
        self.trim_inf = trim_inf

    @property
    def rmin(self):
        """float: The minimum value for the transformed radial array."""
        return self._rmin

    @property
    def R(self):
        """float: The scale factor for the transformed radial array."""
        return self._R

    @property
    def m(self):
        """integer: Free and extra parameter, m must be > 0."""
        return self._m

    def transform(self, x):
        r"""Transform given array [-1,1] to radial array [rmin,inf).

        .. math::
            r_i = R \left( \frac{1+x_i}{1-x_i} \right)^m + r_{min}

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimension numpy array with values between [-1,1]

        Returns
        -------
        np.ndarray(N,)
            One dimension numpy array with values between [rmin,inf).
        """
        rf_array = self._R * ((1 + x) / (1 - x)) ** self._m + self._rmin
        if self.trim_inf:
            self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r):
        r"""Transform radiar array [rmin,inf) back to original array [-1,1].

        .. math::
            x_i = \frac{\sqrt[m]{r_i-r_{min}} - \sqrt[m]{R}}
                       {\sqrt[m]{r_i-r_{min}} + \sqrt[m]{R}}

        Parameters
        ----------
        r : np.ndarray(N,)
            Sorted one dimension radial array with values bewteen [rmin,inf).

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array with values bewteen [-1,1]
        """
        tmp_ri = (r - self._rmin) ** (1 / self._m)
        tmp_R = self._R ** (1 / self._m)

        return (tmp_ri - tmp_R) / (tmp_ri + tmp_R)

    def deriv(self, x):
        r"""Compute the 1st derivative of Handy transformation.

        .. math::
            \frac{dr}{dx} = 2mR \frac{(1+x)^{m-1}}{(1-x)^{m+1}}

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The first derivative of Handy transformation at each point.
        """
        dr = 2 * self._m * self._R * (1 + x) ** (self._m - 1) / (1 - x) ** (self._m + 1)
        if self.trim_inf:
            self._convert_inf(dr)
        return dr

    def deriv2(self, x):
        r"""Compute the 2nd derivative of Handy transformation.

        .. math::
            \frac{d^2r}{dx^2} = 4mR (m + x) \frac{(1+x)^{m-2}}{(1-x)^{m+2}}

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
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
            self._convert_inf(dr)
        return dr

    def deriv3(self, x):
        r"""Compute the 3rd derivative of Handy transformation.

        .. math::
            \frac{d^3r}{dx^3} = 4mR ( 1 + 6 m x + 2 m^2 + 3 x^2)
            \frac{(1+x)^{m-3}}{(1-x)^{m+3}}

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The third derivative of Handy transformation at each point.
        """
        dr = (
            4
            * self._m
            * self._R
            * (1 + 6 * self._m * x + 2 * self._m ** 2 + 3 * x ** 2)
            * (1 + x) ** (self._m - 3)
            / (1 - x) ** (self._m + 3)
        )

        if self.trim_inf:
            self._convert_inf(dr)
        return dr


class HandyModTF(BaseTransform):
    """Modified Handy Transformation class."""

    def __init__(self, rmin, rmax, m, trim_inf=True):
        """Construct a modified Handy transform [-1,1] -> [rmin,rmax].

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
            raise ValueError(f"m needs to be greater than 0.")

        if rmax < rmin:
            raise ValueError(
                f"rmax needs to be greater than rmin. rmax : {rmax}, rmin : {rmin}."
            )
        self._rmin = rmin
        self._rmax = rmax
        self._m = m
        self.trim_inf = trim_inf

    @property
    def rmin(self):
        """float: The minimum value for the transformed radial array."""
        return self._rmin

    @property
    def rmax(self):
        """float: The maximum value for the transformed radial array."""
        return self._rmax

    @property
    def m(self):
        """integer: Free and extra parameter, m must be > 0."""
        return self._m

    def transform(self, x):
        r"""Transform given array [-1,1] to radial array [rmin,rmax].

        .. math::
            r_i = \frac{(1+x_i)^m (r_{max} - r_{min})}
                { 2^m (1 - 2^m + r_{max} - r_{min})
                  - (1 + x_i)^m (r_{max} - r_{min} - 2^m )} + r_{min}

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimension numpy array with values between [-1,1]

        Returns
        -------
        np.ndarray(N,)
            One dimension numpy array with values between [rmin,rmax].
        """
        two_m = 2 ** self._m
        size_r = self._rmax - self._rmin
        qi = (1 + x) ** self._m

        rf_array = qi * size_r / (two_m * (1 - two_m + size_r) - qi * (size_r - two_m))
        rf_array += self._rmin

        if self.trim_inf:
            self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r):
        r"""Transform radiar array [rmin,rmax] back to original array [-1,1].

        .. math::
            x_i = 2 \sqrt[m]{
                \frac{(r_i - r_{min})(r_{max} - r_{min} - 2^m + 1)}
                {(r_i - r_{min})(r_{max} - r_{min} - 2^m) + r_{max} - r_{min}}
                } - 1

        Parameters
        ----------
        r : np.ndarray(N,)
            Sorted one dimension radial array with values bewteen [rmin,inf).

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array with values bewteen [-1,1]
        """
        two_m = 2 ** self._m
        size_r = self._rmax - self._rmin

        tmp_r = (
            (r - self._rmin)
            * (size_r - two_m + 1)
            / ((r - self._rmin) * (size_r - two_m) + size_r)
        )

        return 2 * (tmp_r) ** (1 / self._m) - 1

    def deriv(self, x):
        r"""Compute the 1st derivative of  modified Handy transformation.

        .. math::
            \frac{dr}{dx} = -\frac{
                2^m m (r_{max}-r_{min})(2^m-r_{max}+r_{min}-1)(1+x)^{m-1}}
                {\left( 2^m (2^m-1-r_{max}+r_{min})-(2^m-r_{max}
                + r_{min})(1 + x)^m\right)^2}

        Parameters
        ----------
        x: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The first derivative of Handy transformation at each point.
        """
        two_m = 2 ** self._m
        size_r = self._rmax - self._rmin
        return (
            -(
                self._m
                * two_m
                * (two_m - size_r - 1)
                * size_r
                * (1 + x) ** (self._m - 1)
            )
            / (two_m * (1 - two_m + size_r) + (two_m - size_r) * (1 + x) ** self._m)
            ** 2
        )

    def deriv2(self, x):
        r"""Compute the 2nd derivative of modified Handy transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The second derivative of Handy transformation at each point.
        """
        two_m = 2 ** self._m
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
            / (two_m * (1 - two_m + size_r) + (two_m - size_r) * (1 + x) ** self._m)
            ** 3
        )

    def deriv3(self, x):
        r"""Compute the 3rd derivative of modified Handy transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The third derivative of Handy transformation at each point.
        """
        two_m = 2 ** self._m
        size_r = self._rmax - self._rmin
        return (
            -(
                self._m
                * two_m
                * size_r
                * (two_m - size_r - 1)
                * (1 + x) ** (self._m - 3)
                * (
                    2
                    * two_m
                    * (self._m - 2)
                    * (self._m - 1)
                    * (1 - two_m + size_r) ** 2
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
            / (two_m * (1 - two_m + size_r) + (two_m - size_r) * (1 + x) ** self._m)
            ** 4
        )
