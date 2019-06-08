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
        """Check input type of given array."""
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input array needs to be np.array, got: {type(array)}")

    def _convert_inf(self, array, replace_inf=1e16):
        """Convert np.inf(float) to 1e16(float) incase numerical failure."""
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


class MultiExpTF(BaseTransform):
    """MultiExp Transformation class."""

    def __init__(self, r0, R):
        """Construct MultiExp transform [-1,1] -> [r0,inf).

        Parameters
        ----------
        r0: float
            The minimum coordinate for transformed radial array.
        R: float
            The scale factor for transformed radial array.
        """
        self._r0 = r0
        self._R = R

    @property
    def r0(self):
        """float: The minimum value for the transformed radial array."""
        return self._r0

    @property
    def R(self):
        """float: The scale factor for the transformed radial array."""
        return self._R

    def transform(self, array, trim_inf=True):
        """Transform given array [-1,1] to radial array [r0,inf).

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1]
        trim_inf : bool, oprtional, default to True
            Flag to trim infinite value in transformed array. If true
            will trim np.inf -> 1E16. If false, leave np.inf as it is.
            This may cause unexpected errors in the following operations.

        Returns
        -------
        rf_array: np.ndarray(N,)
            One dimension numpy array with values between [r0,inf).
        """
        self._array_type_check(array)
        rf_array = -self._R * np.log((array + 1) / 2) + self._r0
        if trim_inf:
            self._convert_inf(rf_array)

        return rf_array

    def inverse(self, r_array):
        """Transform radiar array [r0,inf) back to original array [-1,1].

        Parameters
        ----------
        r_array: np.ndarray(N,)
            Sorted one dimension radial array with values bewteen [r0,inf).

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array with values bewteen [-1,1]
        """
        self._array_type_check(r_array)

        return 2 * np.exp(-(r_array - self._r0) / self._R) - 1

    def deriv(self, array):
        """Compute the 1st derivative of MultiExp transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The first derivative of MultiExp transformation at each point.
        """
        self._array_type_check(array)

        return -self._R / (1 + array)

    def deriv2(self, array):
        """Compute the 2nd derivative of MultiExp transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The second derivative of MultiExp transformation at each point.
        """
        self._array_type_check(array)

        return self._R / ((1 + array) ** 2)

    def deriv3(self, array):
        """Compute the 3rd derivative of MultiExp transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The third derivative of MultiExp transformation at each point.
        """
        self._array_type_check(array)

        return -2 * self._R / ((1 + array) ** 3)


class KnowlesTF(BaseTransform):
    """Knowles Transformation class."""

    def __init__(self, r0, R, k):
        """Construct Knowles transform [-1,1] -> [r0,inf).

        Parameters
        ----------
        r0: float
            The minimum coordinate for transformed radial array.
        R: float
            The scale factor for transformed radial array.
        k: float k > 0
            Free parameter, k must be > 0.
        """
        if k <= 0:
            raise ValueError(f"k need to be greater than 0.")

        self._r0 = r0
        self._R = R
        self._k = k

    @property
    def r0(self):
        """float: The minimum value for the transformed radial array."""
        return self._r0

    @property
    def R(self):
        """float: The scale factor for the transformed radial array."""
        return self._R

    @property
    def k(self):
        """float: Free and extra parameter, k must be > 0."""
        return self._k

    def transform(self, array, trim_inf=True):
        """Transform given array [-1,1] to radial array [r0,inf).

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1]
        trim_inf : bool, oprtional, default to True
            Flag to trim infinite value in transformed array. If true
            will trim np.inf -> 1E16. If false, leave np.inf as it is.
            This may cause unexpected errors in the following operations.

        Returns
        -------
        rf_array: np.ndarray(N,)
            One dimension numpy array with values between [r0,inf).
        """
        self._array_type_check(array)

        rf_array = (
            -self._R * np.log(1 - (2 ** -self._k) * (array + 1) ** self._k) + self._r0
        )

        if trim_inf:
            self._convert_inf(rf_array)

        return rf_array

    def inverse(self, r_array):
        """Transform radiar array [r0,inf) back to original array [-1,1].

        Parameters
        ----------
        r_array: np.ndarray(N,)
            Sorted one dimension radial array with values bewteen [r0,inf).

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array with values bewteen [-1,1]
        """
        self._array_type_check(r_array)

        return -1 + 2 * (1 - np.exp((self._r0 - r_array) / self._R)) ** (1 / self._k)

    def deriv(self, array):
        """Compute the 1st derivative of Knowles transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The first derivative of Knowles transformation at each point.
        """
        self._array_type_check(array)
        qi = 1 + array

        return (
            self._R * self._k * (qi ** (self._k - 1)) / (2 ** self._k - qi ** self._k)
        )

    def deriv2(self, array):
        """Compute the 2nd derivative of Knowles transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The second derivative of Knowles transformation at each point.
        """
        self._array_type_check(array)
        qi = 1 + array

        return (
            self._R
            * self._k
            * (qi ** (self._k - 2))
            * (2 ** self._k * (self._k - 1) + qi ** self._k)
            / (2 ** self._k - qi ** self._k) ** 2
        )

    def deriv3(self, array):
        """Compute the 3rd derivative of Knowles transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The third derivative of Knowles transformation at each point.
        """
        self._array_type_check(array)

        qi = 1 + array

        return (
            self._R
            * self._k
            * (qi ** (self._k - 3))
            * (
                -(4 ** self._k) * (self._k - 2) * (self._k - 1)
                - 2 ** self._k * (self._k - 1) * (self._k + 4) * (qi ** self._k)
                - 2 * qi ** (2 * self._k)
            )
            / (-2 ** self._k + qi ** self._k) ** 3
        )


class HandyTF(BaseTransform):
    """Handy Transformation class."""

    def __init__(self, r0, R, m):
        """Construct Handy transform [-1,1] -> [r0,inf).

        Parameters
        ----------
        r0: float
            The minimum coordinate for transformed radial array.
        R: float
            The scale factor for transformed radial array.
        m: float m > 0
            Free parameter, m must be > 0.
        """
        if m <= 0:
            raise ValueError(f"m need to be greater than 0.")

        self._r0 = r0
        self._R = R
        self._m = m

    @property
    def r0(self):
        """float: The minimum value for the transformed radial array."""
        return self._r0

    @property
    def R(self):
        """float: The scale factor for the transformed radial array."""
        return self._R

    @property
    def m(self):
        """float: Free and extra parameter, m must be > 0."""
        return self._m

    def transform(self, array, trim_inf=True):
        """Transform given array [-1,1] to radial array [r0,inf).

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1]
        trim_inf : bool, oprtional, default to True
            Flag to trim infinite value in transformed array. If true
            will trim np.inf -> 1E16. If false, leave np.inf as it is.
            This may cause unexpected errors in the following operations.

        Returns
        -------
        rf_array: np.ndarray(N,)
            One dimension numpy array with values between [r0,inf).
        """
        self._array_type_check(array)

        rf_array = self._R * ((1 + array) / (1 - array)) ** self._m + self._r0

        if trim_inf:
            self._convert_inf(rf_array)

        return rf_array

    def inverse(self, r_array):
        """Transform radiar array [r0,inf) back to original array [-1,1].

        Parameters
        ----------
        r_array: np.ndarray(N,)
            Sorted one dimension radial array with values bewteen [r0,inf).

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array with values bewteen [-1,1]
        """
        self._array_type_check(r_array)

        tmp_ri = (r_array - self._r0) ** (1 / self._m)
        tmp_R = self._R ** (1 / self._m)

        return (tmp_ri - tmp_R) / (tmp_ri + tmp_R)

    def deriv(self, array):
        """Compute the 1st derivative of Handy transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The first derivative of Handy transformation at each point.
        """
        self._array_type_check(array)
        q_tmp = ((1 + array) / (1 - array)) ** self._m

        return -2 * self._m * self._R * q_tmp / (array ** 2 - 1)

    def deriv2(self, array):
        """Compute the 2nd derivative of Handy transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The second derivative of Handy transformation at each point.
        """
        self._array_type_check(array)
        q_tmp = ((1 + array) / (1 - array)) ** self._m

        return (
            4 * self._m * self._R * (self._m + array) * q_tmp / ((array ** 2 - 1) ** 2)
        )

    def deriv3(self, array):
        """Compute the 3rd derivative of Handy transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The third derivative of Handy transformation at each point.
        """
        self._array_type_check(array)
        q_tmp = ((1 + array) / (1 - array)) ** self._m

        return (
            -4
            * self._m
            * self._R
            * (1 + 6 * self._m * array + 2 * self._m ** 2 + 3 * array ** 2)
            * q_tmp
            / ((array ** 2 - 1) ** 3)
        )


class LinearTF(BaseTransform):
    """Linear Transformation class."""

    def __init__(self, r0, rmax):
        """Construct Linear modifiactiontransform [-1,1] -> [r0,rmax).

        Parameters
        ----------
        r0: float
            The minimum coordinate for transformed radial array.
        R: float
            The maximum coordinate for transformed radiar array.
        """
        self._r0 = r0
        self._rmax = rmax

    @property
    def r0(self):
        """float: The minimum value for the transformed radial array."""
        return self._r0

    @property
    def rmax(self):
        """float: The maximum value for the transformed radial array."""
        return self._rmax

    def transform(self, array, trim_inf=True):
        """Transform given array [-1,1] to radial array [r0,rmax).

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1]
        trim_inf : bool, oprtional, default to True
            Flag to trim infinite value in transformed array. If true
            will trim np.inf -> 1E16. If false, leave np.inf as it is.
            This may cause unexpected errors in the following operations.

        Returns
        -------
        rf_array: np.ndarray(N,)
            One dimension numpy array with values between [r0,inf).
        """
        self._array_type_check(array)

        rf_array = 0.5 * (self._rmax - self._r0) * (1 + array) + self._r0

        if trim_inf:
            self._convert_inf(rf_array)

        return rf_array

    def inverse(self, r_array):
        """Transform radiar array [r0,inf) back to original array [-1,1].

        Parameters
        ----------
        r_array: np.ndarray(N,)
            Sorted one dimension radial array with values bewteen [r0,inf).

        Returns
        -------
        np.ndarray(N,)
            The original one dimension array with values bewteen [-1,1]
        """
        self._array_type_check(r_array)

        q_array = (2 * (r_array - self._r0) / (self._rmax - self._r0)) - 1

        return q_array

    def deriv(self, array):
        """Compute the 1st derivative of Linear transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The first derivative of Linear transformation at each point.
        """
        self._array_type_check(array)

        return 0.5 * (self._rmax - self._r0)

    def deriv2(self, array):
        """Compute the 2nd derivative of Linear transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The second derivative of Linear transformation at each point.
        """
        self._array_type_check(array)

        return 0

    def deriv3(self, array):
        """Compute the 3rd derivative of Linear transformation.

        Parameters
        ----------
        array: np.ndarray(N,)
            One dimension numpy array with values between [-1,1].

        Returns
        -------
        np.ndarray(N,)
            The third derivative of Linear transformation at each point.
        """
        self._array_type_check(array)

        return 0
