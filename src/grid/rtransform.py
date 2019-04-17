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

from abc import ABC, abstractmethod

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

    def _array_type_check(self, array):
        """Check input type of given array.

        Parameters
        ----------
        array : np.ndarray(N,)
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input array needs to be np.array, got: {type(array)}")

    def _convert_inf(self, array, replace_inf=1e16):
        """Convert np.inf(float) to 1e16(float) incase numerical failure.

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
        """Compute the 1st derivatvie of Becke transformation.

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
        return 2 * self._R / (1 - array) ** 2

    def deriv2(self, array):
        """Compute the 2nd derivatvie of Becke transformation.

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
        """Compute the 3rd derivatvie of Becke transformation.

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
        """Compute the 1st derivatvie of inverse transformation.

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
        """Compute the 2nd derivatvie of inverse transformation.

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
        """Compute the 3rd derivatvie of inverse transformation.

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


# class RTransform:
#     def __init__(self, npoint: int):
#         if npoint < 2:
#             raise ValueError("A radial grid consists of at least two points")
#         self._npoint = npoint

#     @property
#     def npoint(self):
#         return self._npoint

#     def get_npoint(self):
#         return self.npoint

#     def radius(self, t: float):
#         if isinstance(t, Number):
#             return self._radius(t)
#         elif isinstance(t, np.ndarray):
#             return self._radius_array(t)
#         else:
#             raise NotImplementedError

#     def deriv(self, t):
#         if isinstance(t, Number):
#             return self._deriv(t)
#         elif isinstance(t, np.ndarray):
#             return self._deriv_array(t)
#         else:
#             raise NotImplementedError

#     def deriv2(self, t: float):
#         if isinstance(t, Number):
#             return self._deriv2(t)
#         elif isinstance(t, np.ndarray):
#             return self._deriv2_array(t)
#         else:
#             raise NotImplementedError

#     def deriv3(self, t: float):
#         if isinstance(t, Number):
#             return self._deriv3(t)
#         elif isinstance(t, np.ndarray):
#             return self._deriv3_array(t)
#         else:
#             raise NotImplementedError

#     def inv(self, r: float):
#         if isinstance(r, Number):
#             return self._inv(r)
#         elif isinstance(r, np.ndarray):
#             return self._inv_array(r)
#         else:
#             raise NotImplementedError

#     def _radius_array(self, t: np.ndarray):
#         vf = np.vectorize(self.radius)
#         return vf(t)

#     def _deriv_array(self, t: np.ndarray):
#         vf = np.vectorize(self.deriv)
#         return vf(t)

#     def _deriv2_array(self, t: np.ndarray):
#         vf = np.vectorize(self.deriv2)
#         return vf(t)

#     def _deriv3_array(self, t: np.ndarray):
#         vf = np.vectorize(self.deriv3)
#         return vf(t)

#     def _inv_array(self, r: np.ndarray):
#         vf = np.vectorize(self.inv)
#         return vf(r)

#     def get_radii(self):
#         """Return an array with radii"""
#         result = np.arange(self.npoint, dtype=float)
#         return self._radius_array(result)

#     def get_deriv(self):
#         """Return an array with derivatives at the grid points"""
#         result = np.arange(self.npoint, dtype=float)
#         return self._deriv_array(result)

#     def get_deriv2(self):
#         """Return an array with second derivatives at the grid points"""
#         result = np.arange(self.npoint, dtype=float)
#         return self._deriv2_array(result)

#     def get_deriv3(self):
#         """Return an array with third derivatives at the grid points"""
#         result = np.arange(self.npoint, dtype=float)
#         return self._deriv3_array(result)

#     @classmethod
#     def from_string(cls, s: str):
#         """Construct a RTransform subclass from a string."""
#         words = s.split()
#         clsname = words[0]
#         args = words[1:]
#         if clsname == "IdentityRTransform":
#             if len(args) != 1:
#                 raise ValueError(
#                     "The IdentityRTransform needs one argument, got %i." % len(words)
#                 )
#             npoint = int(args[0])
#             return IdentityRTransform(npoint)
#         elif clsname == "LinearRTransform":
#             if len(args) != 3:
#                 raise ValueError(
#                     "The LinearRTransform needs three arguments, got %i." % len(words)
#                 )
#             rmin = float(args[0])
#             rmax = float(args[1])
#             npoint = int(args[2])
#             return LinearRTransform(rmin, rmax, npoint)
#         elif clsname == "ExpRTransform":
#             if len(args) != 3:
#                 raise ValueError(
#                     "The ExpRTransform needs three arguments, got %i." % len(words)
#                 )
#             rmin = float(args[0])
#             rmax = float(args[1])
#             npoint = int(args[2])
#             return ExpRTransform(rmin, rmax, npoint)
#         elif clsname == "PowerRTransform":
#             if len(args) != 3:
#                 raise ValueError(
#                     "The PowerRTransform needs three arguments, got %i." % len(words)
#                 )
#             rmin = float(args[0])
#             rmax = float(args[1])
#             npoint = int(args[2])
#             return PowerRTransform(rmin, rmax, npoint)
#         elif clsname == "HyperbolicRTransform":
#             if len(args) != 3:
#                 raise ValueError(
#                     "The HyperbolicRTransform needs three arguments, got %i."
#                     % len(words)
#                 )
#             a = float(args[0])
#             b = float(args[1])
#             npoint = int(args[2])
#             return HyperbolicRTransform(a, b, npoint)
#         else:
#             raise TypeError("Unkown RTransform subclass: %s" % clsname)

#     def to_string(self):
#         """Represent the rtransform object as a string"""
#         raise NotImplementedError

#     def chop(self, npoint):
#         """Return an rtransform with ``npoint`` number of grid points

#            The remaining grid points are such that they coincide with those from
#            the old rtransform.
#         """
#         raise NotImplementedError

#     def half(self):
#         """Return an rtransform with half the number of grid points

#            The returned rtransform is such that old(2t+1) = new(t).
#         """
#         raise NotImplementedError


# class IdentityRTransform(RTransform):
#     def __init__(self, npoint: int):
#         super().__init__(npoint)

#     def _radius(self, t: float):
#         return t

#     def _deriv(self, t: float):
#         return 1.0

#     def _deriv2(self, t: float):
#         return 0.0

#     def _deriv3(self, t: float):
#         return 0.0

#     def _inv(self, r: float):
#         return r

#     def to_string(self):
#         return " ".join(["IdentityRTransform", repr(self.npoint)])

#     def chop(self, npoint):
#         return IdentityRTransform(npoint)


# class LinearRTransform(RTransform):
#     def __init__(self, rmin: float, rmax: float, npoint: int):
#         super().__init__(npoint)
#         if rmin >= rmax:
#             raise ValueError("rmin must be smaller rmax")
#         self._rmin = rmin
#         self._rmax = rmax
#         self._alpha: float = (rmax - rmin) / (npoint - 1)

#     @property
#     def rmin(self):
#         return self._rmin

#     @property
#     def rmax(self):
#         return self._rmax

#     @property
#     def alpha(self):
#         return self._alpha

#     def _radius(self, t: float):
#         return self._alpha * t + self._rmin

#     def _deriv(self, t: float):
#         return self._alpha

#     def _deriv2(self, t: float):
#         return 0.0

#     def _deriv3(self, t: float):
#         return 0.0

#     def _inv(self, r: float):
#         return (r - self._rmin) / self._alpha

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


# class ExpRTransform(RTransform):
#     def __init__(self, rmin: float, rmax: float, npoint: int):
#         super().__init__(npoint)
#         if rmin >= rmax:
#             raise ValueError("rmin must be smaller rmax.")
#         if (rmin <= 0) or (rmax <= 0.0):
#             raise ValueError("rmin and rmax must be positive.")
#         self._rmin = rmin
#         self._rmax = rmax
#         self._alpha = np.log(rmax / rmin) / (npoint - 1)

#     @property
#     def rmin(self):
#         return self._rmin

#     @property
#     def rmax(self):
#         return self._rmax

#     @property
#     def alpha(self):
#         return self._alpha

#     def _radius(self, t: float):
#         return self._rmin * np.exp(t * self._alpha)

#     def _deriv(self, t: float):
#         return self.radius(t) * self._alpha

#     def _deriv2(self, t: float):
#         return self.deriv(t) * self._alpha

#     def _deriv3(self, t: float):
#         return self.deriv2(t) * self._alpha

#     def _inv(self, r: float):
#         return np.log(r / self._rmin) / self._alpha

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


# class PowerRTransform(RTransform):
#     def __init__(self, rmin: float, rmax: float, npoint: float):
#         super().__init__(npoint)
#         if rmin >= rmax:
#             raise ValueError("rmin must be smaller rmax.")
#         if (rmin <= 0.0) or (rmax <= 0.0):
#             raise ValueError("rmin and rmax must be positive.")
#         self._power = (np.log(rmax) - np.log(rmin)) / np.log(npoint)
#         if self._power < 2.0:
#             raise ValueError("Power must be at least two for a decent intgration")
#         self._rmin = rmin
#         self._rmax = rmax

#     @property
#     def rmin(self):
#         return self._rmin

#     @property
#     def rmax(self):
#         return self._rmax

#     @property
#     def power(self):
#         return self._power

#     def _radius(self, t: float):
#         return self._rmin * np.power(t + 1, self._power)

#     def _deriv(self, t: float):
#         return self._power * self._rmin * np.power(t + 1, self._power - 1)

#     def _deriv2(self, t: float):
#         return (
#             self._power
#             * (self._power - 1)
#             * self._rmin
#             * np.power(t + 1, self._power - 2)
#         )

#     def _deriv3(self, t: float):
#         return (
#             self._power
#             * (self._power - 1)
#             * (self._power - 2)
#             * self._rmin
#             * np.power(t + 1, self._power - 3)
#         )

#     def _inv(self, r: float):
#         return np.power(r / self._rmin, 1.0 / self._power) - 1

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


# class HyperbolicRTransform(RTransform):
#     def __init__(self, a, b, npoint):
#         if a <= 0:
#             raise ValueError("a must be strctly positive.")
#         if b <= 0:
#             raise ValueError("b must be strctly positive.")
#         if b * (npoint - 1) >= 1.0:
#             raise ValueError("b*(npoint-1) must be smaller than one.")

#         super().__init__(npoint)
#         self._a = a
#         self._b = b

#     @property
#     def a(self):
#         return self._a

#     @property
#     def b(self):
#         return self._b

#     def _radius(self, t: float):
#         return self._a * t / (1 - self._b * t)

#     def _deriv(self, t: float):
#         x = 1.0 / (1 - self._b * t)
#         return self._a * x * x

#     def _deriv2(self, t: float):
#         x = 1.0 / (1 - self._b * t)
#         return 2.0 * self._a * self._b * x ** 3

#     def _deriv3(self, t: float):
#         x = 1.0 / (1 - self._b * t)
#         return 6.0 * self._a * self._b * self._b * x ** 4

#     def _inv(self, r: float):
#         return r / (self._a + self._b * r)

#     def to_string(self):
#         return " ".join(
#             ["HyperbolicRTransform", repr(self.a), repr(self.b), repr(self.npoint)]
#         )
