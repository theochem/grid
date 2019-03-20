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


import numpy as np

from grid.grid.uniform import UniformGrid
from numpy.testing import assert_almost_equal, assert_array_almost_equal


def test_uig_integrate_gauss():
    # grid parameters
    spacing = 0.1
    naxis = 81

    # grid setup
    offset = 0.5 * spacing * (naxis - 1)
    origin = np.zeros(3, float) - offset
    rvecs = np.identity(3) * spacing
    shape = np.array([naxis, naxis, naxis])
    uig = UniformGrid(origin, rvecs, shape)

    # fill a 3D grid with a gaussian function
    x, y, z = np.indices(shape)
    dsq = (x * spacing - offset) ** 2 + (y * spacing - offset) ** 2 + (z * spacing - offset) ** 2
    data = np.exp(-0.5 * dsq) / (2 * np.pi) ** 1.5

    # compute the integral and compare with analytical result
    num_result = uig.integrate(data)
    assert_almost_equal(num_result, 1.0, decimal=3)


def get_simple_test_uig():
    origin = np.array([0.1, -2.0, 3.1])
    rvecs = np.array([
        [1.0, 0.1, 0.2],
        [0.1, 1.1, 0.2],
        [0.0, -0.1, 1.0],
    ])
    shape = np.array([40, 40, 40])
    return UniformGrid(origin, rvecs, shape)


# def test_get_ranges_rcut1():
#     uig = get_simple_test_uig()
#     center = np.array([0.1, -2.5, 3.2])
#     rb1, re1 = uig.get_ranges_rcut(center, 2.0)
#     rb2, re2 = uig.get_grid_cell().get_ranges_rcut(uig.origin - center, 2.0)
#     assert rb1[0] == rb2[0]
#     assert rb1[1] == 0
#     assert rb1[2] == rb2[2]
#     assert (re1 == re2).all()
#
#
# def test_get_ranges_rcut2():
#     uig = get_simple_test_uig()
#     center = np.array([60.0, 50.0, 60.0])
#     rb1, re1 = uig.get_ranges_rcut(center, 2.0)
#     rb2, re2 = uig.get_grid_cell().get_ranges_rcut(uig.origin - center, 2.0)
#     assert (rb1 == rb2).all()
#     assert re1[0] == re2[0]
#     assert re1[1] == 40
#     assert re1[2] == re2[2]


def test_dist_grid_point():
    uig = get_simple_test_uig()
    assert_almost_equal(uig.dist_grid_point(uig.origin, np.array([0, 0, 0])),
                        0.0, decimal=10)
    assert_almost_equal(uig.dist_grid_point(uig.origin, np.array([0, 0, 1])),
                        (0.1 ** 2 + 1.0) ** 0.5, decimal=10)
    assert_almost_equal(uig.dist_grid_point(uig.origin, np.array([0, 1, 0])),
                        (0.1 ** 2 + 1.1 ** 2 + 0.2 ** 2) ** 0.5, decimal=10)


def test_delta_grid_point():
    uig = get_simple_test_uig()
    assert_array_almost_equal(uig.delta_grid_point(uig.origin, np.array([0, 0, 0])),
                              np.array([0.0, 0.0, 0.0]), decimal=10)
    assert_array_almost_equal(uig.delta_grid_point(uig.origin, np.array([0, 0, 1])),
                              np.array([0.0, -0.1, 1.0]), decimal=10)
    assert_array_almost_equal(uig.delta_grid_point(uig.origin, np.array([0, 1, 0])),
                              np.array([0.1, 1.1, 0.2]), decimal=10)
