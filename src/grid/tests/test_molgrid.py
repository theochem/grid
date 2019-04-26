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
"""MolGrid test file."""
from unittest import TestCase

from grid.atomic_grid import AtomicGridFactory
from grid.basegrid import AtomicGrid, OneDGrid
from grid.molgrid import MolGrid
from grid.onedgrid import HortonLinear
from grid.rtransform import ExpRTransform

# from importlib_resources import path
import numpy as np
from numpy.testing import assert_almost_equal


class TestMolGrid(TestCase):
    """MolGrid test class."""

    def setUp(self):
        """Set up radial grid for integral tests."""
        pts = HortonLinear(100)
        tf = ExpRTransform(1e-3, 1e1)
        rad_pts = tf.transform(pts.points)
        rad_wts = tf.deriv(pts.points) * pts.weights
        self.rgrid = OneDGrid(rad_pts, rad_wts)

    def test_integrate_hydrogen_single_1s(self):
        """Test molecular integral in H atom."""
        # numbers = np.array([1], int)
        coordinates = np.array([0.0, 0.0, -0.5], float)
        # rgrid = BeckeTF.transform_grid(oned, 0.001, 0.5)[0]
        # rtf = ExpRTransform(1e-3, 1e1, 100)
        # rgrid = RadialGrid(rtf)
        atf = AtomicGridFactory(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates,
        )
        atgrid = atf.atomic_grid
        mg = MolGrid([atgrid], np.array([0.5]))
        # mg = BeckeMolGrid(coordinates, numbers, None, (rgrid, 110), random_rotate=False)
        dist0 = np.sqrt(((coordinates - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 1.0, decimal=6)

    def test_integrate_hydrogen_pair_1s(self):
        """Test molecular integral in H2."""
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], float)
        atg1 = AtomicGridFactory(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGridFactory(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        mg = MolGrid([atg1.atomic_grid, atg2.atomic_grid], np.array([0.5, 0.5]))
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi + np.exp(-2 * dist1) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 2.0, decimal=6)

    def test_integrate_hydrogen_trimer_1s(self):
        """Test molecular integral in H3."""
        coordinates = np.array(
            [[0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]], float
        )
        atg1 = AtomicGridFactory(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGridFactory(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        atg3 = AtomicGridFactory(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[2],
        )

        mg = MolGrid(
            [atg1.atomic_grid, atg2.atomic_grid, atg3.atomic_grid],
            np.array([0.5, 0.5, 0.5]),
        )
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        dist2 = np.sqrt(((coordinates[2] - mg.points) ** 2).sum(axis=1))
        fn = (
            np.exp(-2 * dist0) / np.pi
            + np.exp(-2 * dist1) / np.pi
            + np.exp(-2 * dist2) / np.pi
        )
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 3.0, decimal=4)

    """
    def test_all_elements():
        numbers = np.array([1, 118], int)
        coordinates = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], float)
        rtf = ExpRTransform(1e-3, 1e1, 10)
        rgrid = RadialGrid(rtf)
        while numbers[0] < numbers[1]:
            BeckeMolGrid(coordinates, numbers, None, (rgrid, 110), random_rotate=False)
            numbers[0] += 1
            numbers[1] -= 1
    """

    def test_molgrid_attrs_subgrid(self):
        """Test sub atomic grid attributes."""
        # numbers = np.array([6, 8], int)
        coordinates = np.array([[0.0, 0.2, -0.5], [0.1, 0.0, 0.5]], float)
        atg1 = AtomicGridFactory(
            self.rgrid,
            1.228,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGridFactory(
            self.rgrid,
            0.945,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        mg = MolGrid([atg1.atomic_grid, atg2.atomic_grid], np.array([1.228, 0.945]))
        # mg = BeckeMolGrid(coordinates, numbers, None, (rgrid, 110), mode='keep')

        assert mg.size == 2 * 110 * 100
        assert mg.points.shape == (mg.size, 3)
        assert mg.weights.shape == (mg.size,)
        assert mg.aim_weights.shape == (mg.size,)
        assert len(mg._indices) == 2 + 1
        # assert mg.k == 3
        # assert mg.random_rotate

        for i in range(2):
            atgrid = mg[i]
            assert isinstance(atgrid, AtomicGrid)
            assert atgrid.size == 100 * 110
            assert atgrid.points.shape == (100 * 110, 3)
            assert atgrid.weights.shape == (100 * 110,)
            # assert atgrid.subgrids is None
            # assert atgrid.number == numbers[i]
            assert (atgrid.center == coordinates[i]).all()
            # assert atgrid.rgrid.rtransform == rtf
            # assert (atgrid.nlls == [110] * 100).all()
            # assert atgrid.nsphere == 100
            # assert atgrid.random_rotate

    def test_molgrid_attrs(self):
        """Test MolGrid attributes."""
        # numbers = np.array([6, 8], int)
        coordinates = np.array([[0.0, 0.2, -0.5], [0.1, 0.0, 0.5]], float)
        atg1 = AtomicGridFactory(
            self.rgrid,
            1.228,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGridFactory(
            self.rgrid,
            0.945,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        mg = MolGrid([atg1.atomic_grid, atg2.atomic_grid], np.array([1.228, 0.945]))

        assert mg.size == 2 * 110 * 100
        assert mg.points.shape == (mg.size, 3)
        assert mg.weights.shape == (mg.size,)
        assert mg.aim_weights.shape == (mg.size,)
        # assert mg.subgrids is None
        # assert mg.k == 3
        # assert mg.random_rotate

    def test_different_aim_weights_h2(self):
        """Test different aim_weights for molgrid."""
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], float)
        atg1 = AtomicGridFactory(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGridFactory(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        aim_weights = np.ones(22000)
        mg = MolGrid(
            [atg1.atomic_grid, atg2.atomic_grid],
            np.array([0.5, 0.5]),
            aim_weights=aim_weights,
        )
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi + np.exp(-2 * dist1) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 4.0, decimal=4)

    def test_raise_errors(self):
        """Test molgrid errors raise."""
        atg = AtomicGridFactory(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=np.array([0.0, 0.0, 0.0]),
        )
        # initilize errors
        with self.assertRaises(NotImplementedError):
            MolGrid([atg.atomic_grid], np.array([1.0]), aim_weights="test")
        with self.assertRaises(ValueError):
            MolGrid([atg.atomic_grid], np.array([1.0]), aim_weights=np.array(3))
        with self.assertRaises(TypeError):
            MolGrid([atg.atomic_grid], np.array([1.0]), aim_weights=[3, 5])
        # integrate errors
        molg = MolGrid([atg.atomic_grid], np.array([1.0]))
        with self.assertRaises(ValueError):
            molg.integrate()
        with self.assertRaises(TypeError):
            molg.integrate(1)
        with self.assertRaises(ValueError):
            molg.integrate(np.array([3, 5]))


"""
def test_family():
    numbers = np.array([6, 8], int)
    coordinates = np.array([[0.0, 0.2, -0.5], [0.1, 0.0, 0.5]], float)
    grid = BeckeMolGrid(coordinates, numbers, None, 'tv-13.7-3', random_rotate=False)
    assert grid.size == 1536 + 1612
"""
