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

from grid.basegrid import OneDGrid, RadialGrid

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
        """Abstract method for 3nd derivative of transformation."""

    def generate_radial(self, oned_grid):
        """Generate a radial grid by transforming the OneDGrid.

        Parameters
        ----------
        oned_grid : OneDGrid
            one dimensional grid generated for integration purpose

        Returns
        -------
        RadialGrid
            one dimensional grid spanning from (0, inf(certain number))

        Raises
        ------
        TypeError
            Input is not a proper OneDGrid instance.
        """
        if not isinstance(oned_grid, OneDGrid):
            raise TypeError(f"Input grid is not OneDGrid, got {type(oned_grid)}")
        new_points = self.transform(oned_grid.points)
        new_weights = self.deriv(oned_grid.points) * oned_grid.weights
        return RadialGrid(new_points, new_weights)

    def _convert_inf(self, array, replace_inf=1e16):
        """Convert np.inf(float) to 1e16(float) in case of numerical failure.

        Parameters
        ----------
        array : np.ndarray(N,)
        """
        if isinstance(array, Number):  # change for number
            new_v = replace_inf if np.isinf(array) else array
        else:
            new_v = array.copy()  # change for arrays with copy
            new_v[new_v == np.inf] = replace_inf
        return new_v

    # def _check_inf(self, array):
    #     if np.any(array == np.inf):
    #         return True
    #     return False


class BeckeTF(BaseTransform):
    """Becke Transformation class."""

    def __init__(self, r0, R):
        """Construct Becke transform, [-1, 1] -> [r_0, inf).

        Parameters
        ----------
        r0 : float
            The minimum coordinates for transformed radial array.
        R : float
            The scale factor for transformed radial array.
        """
        self._r0 = r0
        self._R = R

    @property
    def r0(self):
        """float: the minimum value for the transformed radial array."""
        return self._r0

    @property
    def R(self):
        """float: the scale factor for the transformed radial array."""
        return self._R

    # @classmethod
    # def transform_grid(cls, oned_grid, r0, radius):
    #     if not isinstance(oned_grid, OneDGrid):
    #         raise ValueError(f"Given grid is not OneDGrid, got {type(oned_grid)}")
    #     R = BeckeTF.find_parameter(oned_grid.points, r0, radius)
    #     tfm = cls(r0=r0, R=R)
    #     new_points = tfm.transform(oned_grid.points)
    #     new_weights = tfm.deriv(oned_grid.points) * oned_grid.weights
    #     return RadialGrid(new_points, new_weights), tfm

    @staticmethod
    def find_parameter(array, r0, radius):
        """Compute for optimal R for certain atom, given array and r0.

        Parameters
        ----------
        array : np.ndarray(N,)
            one dimention array locates within [-1, 1]
        r0 : float
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
            r0 needs to be smaller than atomic radius to compute R
        """
        if r0 > radius:
            raise ValueError(
                f"r0 need to be smaller than radius, r0: {r0}, radius: {radius}."
            )
        size = array.size
        if size % 2:
            mid_value = array[size // 2]
        else:
            mid_value = (array[size // 2 - 1] + array[size // 2]) / 2
        return (radius - r0) * (1 - mid_value) / (1 + mid_value)

    def transform(self, x, trim_inf=True):
        """Transform given array[-1, 1] to radial array[r0, inf).

        Parameters
        ----------
        x : np.ndarray(N,)
            One dimension numpy array located between [-1, 1]
        trim_inf : bool, optional, default to True
            Flag to trim infinite value in transformed array. If true, will
            trim np.inf -> 1e16. If false, leave np.inf as it is. This may
            cause unexpected errors in the following operations.

        Returns
        -------
        np.ndarray(N,)
            Transformed radial array located between [r0, inf)
        """
        rf_array = self._R * (1 + x) / (1 - x) + self._r0
        if trim_inf:
            rf_array = self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r):
        """Transform radial array[r0, inf) back to original array[-1, 1].

        Parameters
        ----------
        r : np.ndarray(N,)
            Sorted one dimension radial array located between [r0, inf)

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array located between [-1, 1]
        """
        return (r - self._r0 - self._R) / (r - self._r0 + self._R)

    def deriv(self, x):
        """Compute the 1st derivative of Becke transformation.

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
        """Compute the 2nd derivative of Becke transformation.

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
        """Compute the 3rd derivative of Becke transformation.

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
        """Transform onedgrid form [-1, 1] to [rmin, rmax].

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
        """Compute the 1st order derivative.

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
        """Compute the 2nd order derivative.

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
        """Compute the 3rd order derivative.

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
        """Compute the inverse of the transformation.

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
    """Inverse transformation class, [r0, rmax] or [r0, inf) -> [-1, 1]."""

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
        """Transform radial array back to original one dimension array.

        Parameters
        ----------
        r : np.ndarray(N,)
            Radial grid locates between [r0, rmax] or [r0, inf) depends on the
            domain of its original transformation

        Returns
        -------
        np.ndarray(N,)
            Original one dimension array locates between [-1, 1]
        """
        return self._tfm.inverse(r)

    def inverse(self, x):
        """Transform one dimension array[-1, 1] to radial array [r0, rmax(inf)].

        Parameters
        ----------
        x : np.ndarray
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray
            Radial numpy array located between [r0, rmax(inf)]
        """
        return self._tfm.transform(x)

    def deriv(self, r):
        """Compute the 1st derivative of inverse transformation.

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
        d1 = self._tfm.deriv
        return 1 / d1(r)

    def deriv2(self, r):
        """Compute the 2nd derivative of inverse transformation.

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
        d1 = self._tfm.deriv
        d2 = self._tfm.deriv2
        return -d2(r) / d1(r) ** 3

    def deriv3(self, r):
        """Compute the 3rd derivative of inverse transformation.

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
        return (3 * d2(r) ** 2 - d1(r) * d3(r)) / d1(r) ** 5


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
