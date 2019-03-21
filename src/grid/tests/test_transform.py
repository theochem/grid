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
"""Transformation tests file."""

from unittest import TestCase

from grid.rtransform import BeckeTF, InverseTF

import numpy as np


class TestGrid(TestCase):
    """Grid testcase class."""

    def setUp(self):
        """Test setup function."""
        self.array = np.linspace(-0.9, 0.9, 19)

    def test_becke_tf(self):
        btf = BeckeTF(0.1, 1.2)
        assert btf.R == 1.2
        assert btf.r0 == 0.1

    def test_becke_parameter_calc(self):
        R = BeckeTF.find_parameter(self.array, 0.1, 1.2)
        # R = 1.1
        assert np.isclose(R, 1.1)
        btf = BeckeTF(0.1, R)
        tf_array = btf.transform(self.array)
        assert tf_array[9] == 1.2

    def test_becke_transform(self):
        btf = BeckeTF(0.1, 1.1)
        tf_array = btf.transform(self.array)
        new_array = btf.inverse(tf_array)
        # TODO: inf problem when reach -1
        assert np.allclose(new_array, self.array)

    def test_deriv(self):
        ...
