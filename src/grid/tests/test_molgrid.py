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

from grid.atomic_grid import AtomicGrid
from grid.basegrid import SimpleAtomicGrid
from grid.molgrid import MolGrid
from grid.onedgrid import HortonLinear
from grid.rtransform import ExpRTransform

# from importlib_resources import path
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal


class TestMolGrid(TestCase):
    """MolGrid test class."""

    def setUp(self):
        """Set up radial grid for integral tests."""
        pts = HortonLinear(100)
        tf = ExpRTransform(1e-3, 1e1)
        self.rgrid = tf.generate_radial(pts)

    def test_integrate_hydrogen_single_1s(self):
        """Test molecular integral in H atom."""
        # numbers = np.array([1], int)
        coordinates = np.array([0.0, 0.0, -0.5], float)
        # rgrid = BeckeTF.transform_grid(oned, 0.001, 0.5)[0]
        # rtf = ExpRTransform(1e-3, 1e1, 100)
        # rgrid = RadialGrid(rtf)
        atg1 = AtomicGrid.special_init(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates,
        )
        mg = MolGrid([atg1], np.array([1]))
        # mg = BeckeMolGrid(coordinates, numbers, None, (rgrid, 110), random_rotate=False)
        dist0 = np.sqrt(((coordinates - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 1.0, decimal=6)

    def test_integrate_hydrogen_pair_1s(self):
        """Test molecular integral in H2."""
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], float)
        atg1 = AtomicGrid.special_init(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGrid.special_init(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        mg = MolGrid([atg1, atg2], np.array([1, 1]))
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
        atg1 = AtomicGrid.special_init(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGrid.special_init(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        atg3 = AtomicGrid.special_init(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[2],
        )

        mg = MolGrid([atg1, atg2, atg3], np.array([1, 1, 1]))
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

    def test_integrate_hydrogen_8_1s(self):
        """Test molecular integral in H2."""
        x, y, z = np.meshgrid(*(3 * [[-0.5, 0.5]]))
        centers = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        atgs = [
            AtomicGrid.special_init(
                self.rgrid, 0.5, scales=np.array([]), degs=np.array([17]), center=center
            )
            for center in centers
        ]
        mg = MolGrid(atgs, np.array([1] * len(centers)))
        fn = 0
        for center in centers:
            dist = np.linalg.norm(center - mg.points, axis=1)
            fn += np.exp(-2 * dist) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, len(centers), decimal=2)

    def test_molgrid_attrs_subgrid(self):
        """Test sub atomic grid attributes."""
        # numbers = np.array([6, 8], int)
        coordinates = np.array([[0.0, 0.2, -0.5], [0.1, 0.0, 0.5]], float)
        atg1 = AtomicGrid.special_init(
            self.rgrid,
            1.228,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGrid.special_init(
            self.rgrid,
            0.945,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        mg = MolGrid([atg1, atg2], np.array([6, 8]), store=True)
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
            assert (atgrid.center == coordinates[i]).all()
        mg = MolGrid([atg1, atg2], np.array([6, 8]))
        for i in range(2):
            atgrid = mg[i]
            assert isinstance(atgrid, SimpleAtomicGrid)
            assert_allclose(atgrid.center, mg._coors[i])

    def test_molgrid_attrs(self):
        """Test MolGrid attributes."""
        # numbers = np.array([6, 8], int)
        coordinates = np.array([[0.0, 0.2, -0.5], [0.1, 0.0, 0.5]], float)
        atg1 = AtomicGrid.special_init(
            self.rgrid,
            1.228,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGrid.special_init(
            self.rgrid,
            0.945,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        mg = MolGrid([atg1, atg2], np.array([6, 8]), store=True)

        assert mg.size == 2 * 110 * 100
        assert mg.points.shape == (mg.size, 3)
        assert mg.weights.shape == (mg.size,)
        assert mg.aim_weights.shape == (mg.size,)
        assert mg.get_atomic_grid(0) is atg1
        assert mg.get_atomic_grid(1) is atg2

        simple_ag1 = mg.get_simple_atomic_grid(0)
        simple_ag2 = mg.get_simple_atomic_grid(1)
        assert_allclose(simple_ag1.points, atg1.points)
        assert_allclose(simple_ag2.points, atg2.points)

        aim_at1 = mg.get_aim_weights(0)
        aim_at2 = mg.get_aim_weights(1)
        assert_allclose(simple_ag1.weights, atg1.weights * aim_at1)
        assert_allclose(simple_ag2.weights, atg2.weights * aim_at2)

        # assert mg.subgrids is None
        # assert mg.k == 3
        # assert mg.random_rotate

    def test_different_aim_weights_h2(self):
        """Test different aim_weights for molgrid."""
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], float)
        atg1 = AtomicGrid.special_init(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomicGrid.special_init(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=coordinates[1],
        )
        aim_weights = np.ones(22000)
        mg = MolGrid([atg1, atg2], np.array([1, 1]), aim_weights=aim_weights)
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi + np.exp(-2 * dist1) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 4.0, decimal=4)

    def test_horton_molgrid(self):
        """Test horton style grid."""
        coors = np.array([[0, 0, -0.5], [0, 0, 0.5]])
        nums = [1, 1]
        mol_grid = MolGrid.horton_molgrid(coors, nums, self.rgrid, 110)
        atg1 = AtomicGrid.special_init(
            self.rgrid, 0.5, scales=np.array([]), degs=np.array([17]), center=coors[0]
        )
        atg2 = AtomicGrid.special_init(
            self.rgrid, 0.5, scales=np.array([]), degs=np.array([17]), center=coors[1]
        )
        ref_grid = MolGrid([atg1, atg2], nums, store=True)
        assert_allclose(ref_grid.points, mol_grid.points)
        assert_allclose(ref_grid.weights, mol_grid.weights)

    def test_raise_errors(self):
        """Test molgrid errors raise."""
        atg = AtomicGrid.special_init(
            self.rgrid,
            0.5,
            scales=np.array([]),
            degs=np.array([17]),
            center=np.array([0.0, 0.0, 0.0]),
        )
        # initilize errors
        with self.assertRaises(NotImplementedError):
            MolGrid([atg], np.array([1]), aim_weights="test")
        with self.assertRaises(ValueError):
            MolGrid([atg], np.array([1]), aim_weights=np.array(3))
        with self.assertRaises(TypeError):
            MolGrid([atg], np.array([1]), aim_weights=[3, 5])
        # integrate errors
        molg = MolGrid([atg], np.array([1]))
        with self.assertRaises(ValueError):
            molg.integrate()
        with self.assertRaises(TypeError):
            molg.integrate(1)
        with self.assertRaises(ValueError):
            molg.integrate(np.array([3, 5]))
        with self.assertRaises(NotImplementedError):
            molg.get_atomic_grid(0)
        with self.assertRaises(ValueError):
            molg.get_aim_weights(-3)

        molg = MolGrid([atg], np.array([1]), store=True)
        with self.assertRaises(ValueError):
            molg.get_atomic_grid(-5)

    """
    def test_family():
        numbers = np.array([6, 8], int)
        coordinates = np.array([[0.0, 0.2, -0.5], [0.1, 0.0, 0.5]], float)
        grid = BeckeMolGrid(coordinates, numbers, None, 'tv-13.7-3', random_rotate=False)
        assert grid.size == 1536 + 1612
    """
