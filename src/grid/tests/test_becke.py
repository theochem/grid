# -*- coding: utf-8 -*-
# OLDGRIDS: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The OLDGRIDS Development Team
#
# This file is part of OLDGRIDS.
#
# OLDGRIDS is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# OLDGRIDS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Becke tests files."""
from unittest import TestCase

from grid.becke import BeckeWeights

import numpy as np
from numpy.testing import assert_allclose


class TestBecke(TestCase):
    """Becke weight class tests."""

    def test_becke_sum2_one(self):
        """Test becke weights add up to one."""
        npoint = 100
        points = np.random.uniform(-5, 5, (npoint, 3))

        weights0 = np.ones(npoint, float)
        weights1 = np.ones(npoint, float)

        radii = np.array([0.5, 0.8])
        centers = np.array([[1.2, 2.3, 0.1], [-0.4, 0.0, -2.2]])
        weights0 = BeckeWeights.generate_becke_weights(
            points, radii, centers, select=[0], order=3
        )
        weights1 = BeckeWeights.generate_becke_weights(
            points, radii, centers, select=[1], order=3
        )

        assert_allclose(weights0 + weights1, np.ones(100))

    def test_becke_sum3_one(self):
        """Test becke weights add up to one with three centers."""
        npoint = 100
        points = np.random.uniform(-5, 5, (npoint, 3))

        weights0 = np.ones(npoint, float)
        weights1 = np.ones(npoint, float)
        weights2 = np.ones(npoint, float)

        radii = np.array([0.5, 0.8, 5.0])
        centers = np.array([[1.2, 2.3, 0.1], [-0.4, 0.0, -2.2], [2.2, -1.5, 0.0]])
        weights0 = BeckeWeights.generate_becke_weights(
            points, radii, centers, select=[0], order=3
        )
        weights1 = BeckeWeights.generate_becke_weights(
            points, radii, centers, select=[1], order=3
        )
        weights2 = BeckeWeights.generate_becke_weights(
            points, radii, centers, select=[2], order=3
        )

        assert_allclose(weights0 + weights1 + weights2, np.ones(100))

    def test_becke_special_points(self):
        """Test becke weights for special cases."""
        radii = np.array([0.5, 0.8, 5.0])
        centers = np.array([[1.2, 2.3, 0.1], [-0.4, 0.0, -2.2], [2.2, -1.5, 0.0]])

        weights = BeckeWeights.generate_becke_weights(
            centers, radii, centers, select=[0], order=3
        )
        assert_allclose(weights, [1, 0, 0])

        weights = BeckeWeights.generate_becke_weights(
            centers, radii, centers, select=[1], order=3
        )
        assert_allclose(weights, [0, 1, 0])

        weights = BeckeWeights.generate_becke_weights(
            centers, radii, centers, select=[2], order=3
        )
        assert_allclose(weights, [0, 0, 1])

        # each point in seperate sectors.
        weights = BeckeWeights.generate_becke_weights(
            centers, radii, centers, pt_ind=[0, 1, 2, 3]
        )
        assert_allclose(weights, [1, 1, 1])

        weights = BeckeWeights.generate_becke_weights(
            centers, radii, centers, select=[0, 1], pt_ind=[0, 1, 3]
        )
        assert_allclose(weights, [1, 1, 0])

        weights = BeckeWeights.generate_becke_weights(
            centers, radii, centers, select=[2, 0], pt_ind=[0, 2, 3]
        )
        assert_allclose(weights, [0, 0, 0])

    def test_raise_errors(self):
        """Test errors raise."""
        npoint = 100
        points = np.random.uniform(-5, 5, (npoint, 3))
        radii = np.array([0.5, 0.8])
        centers = np.array([[1.2, 2.3, 0.1], [-0.4, 0.0, -2.2]])
        with self.assertRaises(ValueError):
            BeckeWeights.generate_becke_weights(points, radii, centers, select=[])
        with self.assertRaises(ValueError):
            BeckeWeights.generate_becke_weights(points, radii, centers, pt_ind=[3])
        with self.assertRaises(ValueError):
            BeckeWeights.generate_becke_weights(points, radii, centers, pt_ind=[3, 6])
        with self.assertRaises(ValueError):
            BeckeWeights.generate_becke_weights(
                points, radii, centers, select=[], pt_ind=[]
            )
        with self.assertRaises(ValueError):
            BeckeWeights.generate_becke_weights(
                points, radii, centers, select=[0, 1], pt_ind=[0, 10, 50, 99]
            )
        with self.assertRaises(ValueError):
            BeckeWeights.generate_becke_weights(
                points, radii, centers[0], select=[0, 1], pt_ind=[0, 10, 50, 99]
            )
