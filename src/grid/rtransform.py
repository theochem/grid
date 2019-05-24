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
# along with this program; if not, see <http:#www.gnu.org/licenses/>
#
# --
"""Transformation from uniform 1D to non-uniform 1D grids."""
import warnings
from abc import ABC, abstractmethod

from grid.basegrid import OneDGrid, RadialGrid

import numpy as np


class BaseTransform(ABC):
    """Abstract class for transformation."""

    @abstractmethod
    def transform(self, array):
        """Abstract method for transformation."""

    @abstractmethod
    def inverse(self, r_array):
        """Abstract method for inverse transformation."""

    @abstractmethod
    def deriv(self, array):
        """Abstract method for 1st derivative of transformation."""

    @abstractmethod
    def deriv2(self, array):
        """Abstract method for 2nd derivative of transformation."""

    @abstractmethod
    def deriv3(self, array):
        """Abstract method for 3nd derivative of transformation."""

    def transform_grid(self, oned_grid):
        """Transform given OneDGrid into a proper scaled radial grid.

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

    def _array_type_check(self, array):
        """Check input type of given array.

        Parameters
        ----------
        array : np.ndarray(N,)
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input array needs to be np.array, got: {type(array)}")

    def _convert_inf(self, array, replace_inf=1e16):
        """Convert np.inf(float) to 1e16(float) in case of numerical failure.

        Parameters
        ----------
        array : np.ndarray(N,)
        """
        array[array == np.inf] = replace_inf

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

    def transform(self, array, trim_inf=True):
        """Transform given array[-1, 1] to radial array[r0, inf).

        Parameters
        ----------
        array : np.ndarray(N,)
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
        self._array_type_check(array)
        rf_array = self._R * (1 + array) / (1 - array) + self._r0
        if trim_inf:
            self._convert_inf(rf_array)
        return rf_array

    def inverse(self, r_array):
        """Transform radial array[r0, inf) back to original array[-1, 1].

        Parameters
        ----------
        r_array : np.ndarray(N,)
            Sorted one dimension radial array located between [r0, inf)

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array located between [-1, 1]
        """
        self._array_type_check(r_array)
        return (r_array - self._r0 - self._R) / (r_array - self._r0 + self._R)

    def deriv(self, array):
        """Compute the 1st derivative of Becke transformation.

        Parameters
        ----------
        array : np.array(N,)
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray(N,)
            1st derivative of Becke transformation at each points
        """
        self._array_type_check(array)
        return 2 * self._R / ((1 - array) ** 2)

    def deriv2(self, array):
        """Compute the 2nd derivative of Becke transformation.

        Parameters
        ----------
        array : np.array(N,)
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray(N,)
            2nd derivative of Becke transformation at each points
        """
        self._array_type_check(array)
        return 4 * self._R / (1 - array) ** 3

    def deriv3(self, array):
        """Compute the 3rd derivative of Becke transformation.

        Parameters
        ----------
        array : np.array(N,)
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray(N,)
            3rd derivative of Becke transformation at each points
        """
        self._array_type_check(array)
        return 12 * self._R / (1 - array) ** 4


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

    def transform(self, r_array):
        """Transform radial array back to original one dimension array.

        Parameters
        ----------
        r_array : np.ndarray(N,)
            Radial grid locates between [r0, rmax] or [r0, inf) depends on the
            domain of its original transformation

        Returns
        -------
        np.ndarray(N,)
            Original one dimension array locates between [-1, 1]
        """
        return self._tfm.inverse(r_array)

    def inverse(self, array):
        """Transform one dimension array[-1, 1] to radial array [r0, rmax(inf)].

        Parameters
        ----------
        array : np.ndarray
            One dimension numpy array located between [-1, 1]

        Returns
        -------
        np.ndarray
            Radial numpy array located between [r0, rmax(inf)]
        """
        return self._tfm.transform(array)

    def deriv(self, r_array):
        """Compute the 1st derivative of inverse transformation.

        Parameters
        ----------
        r_array : np.array(N,)
            One dimension numpy array located between [ro, rmax(inf)]

        Returns
        -------
        np.ndarray(N,)
            1st derivative of inverse transformation at each points
        """
        # x: inverse x array, d1: first derivative
        x = self._tfm.inverse(r_array)
        d1 = self._tfm.deriv
        return 1 / d1(x)

    def deriv2(self, r_array):
        """Compute the 2nd derivative of inverse transformation.

        Parameters
        ----------
        r_array : np.array(N,)
            One dimension numpy array located between [ro, rmax(inf)]

        Returns
        -------
        np.ndarray(N,)
            2nd derivative of inverse transformation at each points
        """
        # x: inverse x array, d1: first derivative
        # d2: second derivative d^2x / dy^2
        x = self._tfm.inverse(r_array)
        d1 = self._tfm.deriv
        d2 = self._tfm.deriv2
        return -d2(x) / d1(x) ** 3

    def deriv3(self, r_array):
        """Compute the 3rd derivative of inverse transformation.

        Parameters
        ----------
        r_array : np.array(N,)
            One dimension numpy array located between [ro, rmax(inf)]

        Returns
        -------
        np.ndarray(N,)
            3rd derivative of inverse transformation at each points
        """
        # x: inverse x array, d1: first derivative
        # d2: second derivative d^2x / dy^2
        # d3: third derivative d^3x / dy^3
        x = self._tfm.inverse(r_array)
        d1 = self._tfm.deriv
        d2 = self._tfm.deriv2
        d3 = self._tfm.deriv3
        return (3 * d2(x) ** 2 - d1(x) * d3(x)) / d1(x) ** 5


class IdentityRTransform(BaseTransform):
    """Identity Transform class."""

    def transform(self, t: np.ndarray):
        """Perform given array into itself."""
        return t

    def deriv(self, t: np.ndarray):
        """Compute the first derivative of identity transform."""
        t = np.array(t)
        return np.ones(t.size)

    def deriv2(self, t: np.ndarray):
        """Compute the second Derivative of identity transform."""
        t = np.array(t)
        return np.zeros(t.size)

    def deriv3(self, t: np.ndarray):
        """Compute the third Derivative of identity transform."""
        t = np.array(t)
        return np.zeros(t.size)

    def inverse(self, r: np.ndarray):
        """Compute the inverse of identity transform."""
        r = np.array(r)
        return np.ones(r.size)


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

    def transform(self, t: np.ndarray):
        """Perform linear transformation."""
        t = np.array(t)
        alpha = (self._rmax - self._rmin) / (t.size - 1)
        return alpha * t + self._rmin

    def deriv(self, t: np.ndarray):
        """Compute the first derivative of linear transformation."""
        t = np.array(t)
        alpha = (self._rmax - self._rmin) / (t.size - 1)
        return np.ones(t.size) * alpha

    def deriv2(self, t: np.ndarray):
        """Compute the second derivative of linear transformation."""
        return np.zeros(t.size)

    def deriv3(self, t: np.ndarray):
        """Compute the third derivative of linear transformation."""
        return np.zeros(t.size)

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

    def transform(self, t: np.ndarray):
        """Perform exponential transform."""
        t = np.array(t)
        alpha = np.log(self._rmax / self._rmin) / (t.size - 1)
        return self._rmin * np.exp(t * alpha)

    def deriv(self, t: np.ndarray):
        """Compute the first derivative of exponential transform."""
        t = np.array(t)
        alpha = np.log(self._rmax / self._rmin) / (t.size - 1)
        return self.transform(t) * alpha

    def deriv2(self, t: np.ndarray):
        """Compute the second derivative of exponential transform."""
        t = np.array(t)
        alpha = np.log(self._rmax / self._rmin) / (t.size - 1)
        return self.deriv(t) * alpha

    def deriv3(self, t: np.ndarray):
        """Compute the third derivative of exponential transform."""
        t = np.array(t)
        alpha = np.log(self._rmax / self._rmin) / (t.size - 1)
        return self.deriv2(t) * alpha

    def inverse(self, r: np.ndarray):
        """Compute the inverse of exponential transform."""
        r = np.array(r)
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

    def transform(self, t: np.ndarray):
        """Perform power transform."""
        t = np.array(t)
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(t.size)
        if power < 2:
            warnings.warn(
                f"power need to be larger than 2\n  power: {power}", RuntimeWarning
            )
        return self._rmin * np.power(t + 1, power)

    def deriv(self, t: np.ndarray):
        """Compute first derivative of power transform."""
        t = np.array(t)
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(t.size)
        return power * self._rmin * np.power(t + 1, power - 1)

    def deriv2(self, t: np.ndarray):
        """Compute second derivative of power transform."""
        t = np.array(t)
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(t.size)
        return power * (power - 1) * self._rmin * np.power(t + 1, power - 2)

    def deriv3(self, t: np.ndarray):
        """Compute third derivative of power transform."""
        t = np.array(t)
        power = (np.log(self._rmax) - np.log(self._rmin)) / np.log(t.size)
        return (
            power * (power - 1) * (power - 2) * self._rmin * np.power(t + 1, power - 3)
        )

    def inverse(self, r: np.ndarray):
        """Compute the inverse of power transform."""
        r = np.array(r)
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

    def transform(self, t: np.ndarray):
        """Perform hyperbolic transformation."""
        t = np.array(t)
        if self._b * (t.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        return self._a * t / (1 - self._b * t)

    def deriv(self, t: np.ndarray):
        """Compute the first derivative of hyperbolic transformation."""
        t = np.array(t)
        if self._b * (t.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        x = 1.0 / (1 - self._b * t)
        return self._a * x * x

    def deriv2(self, t: np.ndarray):
        """Compute the second derivative of hyperbolic transformation."""
        t = np.array(t)
        if self._b * (t.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        x = 1.0 / (1 - self._b * t)
        return 2.0 * self._a * self._b * x ** 3

    def deriv3(self, t: np.ndarray):
        """Compute the third derivative of hyperbolic transformation."""
        t = np.array(t)
        if self._b * (t.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        x = 1.0 / (1 - self._b * t)
        return 6.0 * self._a * self._b * self._b * x ** 4

    def inverse(self, r: np.ndarray):
        """Compute the inverse of hyperbolic transformation."""
        r = np.array(r)
        if self._b * (r.size - 1) >= 1.0:
            raise ValueError("b*(npoint-1) must be smaller than one.")
        return r / (self._a + self._b * r)


#     def to_string(self):
#         return " ".join(
#             ["HyperbolicRTransform", repr(self.a), repr(self.b), repr(self.npoint)]
#         )
