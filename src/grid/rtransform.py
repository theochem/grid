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
    @abstractmethod
    def transform(self, array):
        ...

    @abstractmethod
    def inverse(self, r_array):
        ...

    @abstractmethod
    def deriv(self, array):
        ...

    @abstractmethod
    def deriv2(self, array):
        ...

    def _type_assert(self, array):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input array needs to be np.array, got: {type(array)}")


class BeckeTF(BaseTransform):
    def __init__(self, r0, R):
        self._r0 = r0
        self._R = R

    @property
    def r0(self):
        return self._r0

    @property
    def R(self):
        return self._R

    @staticmethod
    def find_parameter(array, r0, radius):
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

    def transform(self, array):
        return self._R * (1 + array) / (1 - array) + self._r0

    def inverse(self, r_array):
        return (r_array - self._r0 - self._R) / (r_array - self._r0 + self._R)

    def deriv(self, array):
        return 2 * self._R / (1 - array) ** 2

    def deriv2(self, array):
        return 4 * self._R / (1 - array) ** 3


class InverseTF(BaseTransform):
    def __init__(self, transform):
        self._tfm = transform

    def transform(self, r_array):
        return self._tfm.inverse(r_array)

    def inverse(self, array):
        return self._tfm.transform(array)

    def deriv(self, r_array):
        return 1 / self._tfm.deriv(self._tfm.inverse(r_array))

    def deriv2(self, r_array):
        return (
            -1
            * self._tfm.deriv2(self._tfm.inverse(r_array))
            / (self._tfm.deriv(self._tfm.inverse(r_array)) ** 3)
        )


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
