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
"""Utils function test file."""
from unittest import TestCase

from grid.utils import get_cov_radii

import numpy as np
from numpy.testing import assert_allclose


class TestUtils(TestCase):
    """Test class for functions in grid.utils."""

    def test_get_atomic_radii(self):
        """Test get_cov_radii function for all atoms."""
        # fmt: off
        Bragg_Slater = np.array([
            np.nan, 0.25, np.nan, 1.45, 1.05, 0.85, 0.7, 0.65, 0.6, 0.5, np.nan,
            1.8, 1.5, 1.25, 1.1, 1., 1., 1., np.nan, 2.2, 1.8, 1.6, 1.4, 1.35,
            1.4, 1.4, 1.4, 1.35, 1.35, 1.35, 1.35, 1.3, 1.25, 1.15, 1.15, 1.15,
            np.nan, 2.35, 2., 1.8, 1.55, 1.45, 1.45, 1.35, 1.3, 1.35, 1.4, 1.6,
            1.55, 1.55, 1.45, 1.45, 1.4, 1.4, np.nan, 2.6, 2.15, 1.95, 1.85,
            1.85, 1.85, 1.85, 1.85, 1.85, 1.8, 1.75, 1.75, 1.75, 1.75, 1.75,
            1.75, 1.75, 1.55, 1.45, 1.35, 1.35, 1.3, 1.35, 1.35, 1.35, 1.5,
            1.9, 1.8, 1.6, 1.9, np.nan, np.nan
        ])
        Cambridge = np.array([
            np.nan, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
            1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.7, 1.6,
            1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19,
            1.20, 1.20, 1.16, 2.20, 1.95, 1.9, 1.75, 1.64, 1.54, 1.47, 1.46,
            1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44,
            2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92,
            1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.7, 1.62, 1.51, 1.44, 1.41,
            1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.5, 1.5,
        ])
        # fmt: on
        all_index = np.arange(1, 87)
        bragg = get_cov_radii(all_index, type="bragg")
        assert_allclose(bragg, Bragg_Slater[1:] * 1.8897261339213)
        bragg = get_cov_radii(all_index, type="cambridge")
        assert_allclose(bragg, Cambridge[1:] * 1.8897261339213)

    def test_raise_errors(self):
        """Test raise proper errors."""
        with self.assertRaises(ValueError):
            get_cov_radii(3, type="random")
        with self.assertRaises(ValueError):
            get_cov_radii(0)
        with self.assertRaises(ValueError):
            get_cov_radii([3, 5, 0])
