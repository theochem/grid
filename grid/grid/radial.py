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
"""1D Radial integration grid"""

import numpy as np

from old_grids.grid.cext import dot_multi

__all__ = ['RadialGrid']


class RadialGrid(object):
    """An integration grid for the radial component of a spherical coordinate system"""

    def __init__(self, rtransform):
        self._rtransform = rtransform
        self._weights = (4 * np.pi) * (
                self._rtransform.get_deriv() *
                self._rtransform.get_radii() ** 2
        )

    def __eq__(self, other):
        return self.rtransform.to_string() == other.rtransform.to_string()

    def __ne__(self, other):
        return not self.__eq__(other)

    def _get_size(self):
        """The size of the grid."""
        return self._weights.size

    size = property(_get_size)

    def _get_shape(self):
        """The shape of the grid."""
        return self._weights.shape

    shape = property(_get_shape)

    def _get_rtransform(self):
        """The RTransform object of the grid."""
        return self._rtransform

    rtransform = property(_get_rtransform)

    def _get_weights(self):
        """The grid weights."""
        return self._weights

    weights = property(_get_weights)

    def _get_radii(self):
        """The positions of the radial grid points."""
        return self._rtransform.get_radii()

    radii = property(_get_radii)

    def zeros(self):
        return np.zeros(self.shape)

    def integrate(self, *args):
        """Integrate the product of all arguments

           **Arguments:**

           data1, data2, ...
                All arguments must be arrays with the same size as the number
                of grid points. The arrays contain the functions, evaluated
                at the grid points, that must be multiplied and integrated.

        """
        args = [arg.ravel() for arg in args if arg is not None]
        args.append(self.weights)
        return dot_multi(*args)

    def chop(self, new_size):
        """Return a radial grid with a different number of points.

            **Arguments:**

            new_size
                The new number of radii.

           The corresponding radii remain the same.
        """
        rtf = self._rtransform.chop(new_size)
        return RadialGrid(rtf)
