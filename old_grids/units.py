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
# pragma pylint: disable=invalid-name
"""Conversion factors to and from atomic units."""


import scipy.constants as spc


__all__ = ["angstrom", "amu"]


# conversion factor from angstrom to atomic units
angstrom = spc.angstrom / spc.value(u'atomic unit of length')
# conversion factor from grams to atomic unit of mass per mol
amu = 1. / spc.value(u'electron mass in u')
