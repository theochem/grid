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
"""MolGrid test file."""
from unittest import TestCase

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
import scipy.special
from grid.utils import solid_harmonics, convert_cart_to_sph

from grid.atomgrid import AtomGrid, _get_rgrid_size
from grid.basegrid import LocalGrid
from grid.becke import BeckeWeights
from grid.hirshfeld import HirshfeldWeights
from grid.molgrid import MolGrid
from grid.onedgrid import GaussLaguerre, Trapezoidal, UniformInteger
from grid.rtransform import ExpRTransform, LinearFiniteRTransform

# Ignore angular/Lebedev grid warnings where the weights are negative:
pytestmark = pytest.mark.filterwarnings("ignore:Lebedev weights are negative which can*")

# Helper functions for index mapping and binomial coefficient calculation
def horton_index(l, m):
    """Convert (l, m) to flat Horton order index."""
    if not isinstance(l, int) or not isinstance(m, int):
        raise TypeError("l and m must be integers")
    if abs(m) > l:
        raise ValueError("abs(m) must be <= l")
    if m == 0:
        return l * l
    elif m > 0:
        return l * l + 2 * m - 1
    else: # m < 0
        return l * l + 2 * abs(m)

def get_lm_from_horton_index(k):
    """Convert flat Horton order index k back to (l, m)."""
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a non-negative integer")
    l = int(np.floor(np.sqrt(k)))
    # Check if k corresponds to a valid index for this l
    if k >= (l + 1)**2:
        # This happens if k is not a perfect square and floor(sqrt(k)) was rounded down incorrectly
        # e.g. k=8, l=floor(sqrt(8))=2, (l+1)**2 = 9. k=8 is valid.
        # e.g. k=9, l=floor(sqrt(9))=3, (l+1)**2 = 16. k=9 is valid (l=3, m=0)
        # Need a robust way to get l
        l = int(np.ceil(np.sqrt(k+1)) - 1)

    m_abs_or_zero = (k - l*l + 1) // 2 # Corresponds to abs(m) for m != 0, or m=0
    if k == l*l:
        m = 0
    elif (k - l*l) % 2 == 1: # Positive m
        m = m_abs_or_zero
    else: # Negative m
        m = -m_abs_or_zero

    # Final check
    if abs(m) > l:
         # If k was invalid, recalculate l based on the fact that k < (l_true+1)^2
         l_check = 0
         while (l_check + 1)**2 <= k:
             l_check += 1
         l = l_check
         # Recalculate m based on the corrected l
         if k == l*l:
             m = 0
         elif (k - l*l) % 2 == 1:
              m = (k - l*l + 1) // 2
         else:
              m = -(k - l*l) // 2

    return l, m

def binomial_safe(n, k):
    """Safely compute binomial coefficient C(n, k), returning 0 if k < 0 or k > n."""
    if k < 0 or k > n:
        return 0
    # Use float conversion for safety with potentially large numbers before sqrt
    # and handle potential precision issues with exact=True for large ints
    try:
        # Attempt exact calculation first for smaller numbers if possible
        if n < 1000: # Heuristic threshold
             return float(scipy.special.comb(n, k, exact=True))
        else:
             # Use floating point for potentially large numbers
             return scipy.special.comb(n, k, exact=False)
    except OverflowError:
         # Fallback to floating point if exact calculation overflows
         return scipy.special.comb(n, k, exact=False)

class TestMolGrid(TestCase):
    """MolGrid test class."""

    def setUp(self):
        """Set up radial grid for integral tests."""
        pts = UniformInteger(100)
        tf = ExpRTransform(1e-5, 2e1)
        self.rgrid = tf.transform_1d_grid(pts)

    def test_integrate_hydrogen_single_1s(self):
        """Test molecular integral in H atom."""
        coordinates = np.array([0.0, 0.0, -0.5], float)
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates,
        )
        becke = BeckeWeights(order=3)
        mg = MolGrid(np.array([1]), [atg1], becke)
        # mg = BeckeMolGrid(coordinates, numbers, None, (rgrid, 110), random_rotate=False)
        dist0 = np.sqrt(((coordinates - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 1.0, decimal=6)

    def test_make_grid_integral(self):
        """Test molecular make_grid works as designed."""
        pts = UniformInteger(70)
        tf = ExpRTransform(1e-5, 2e1)
        rgrid = tf.transform_1d_grid(pts)
        numbers = np.array([1, 1])
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], float)
        becke = BeckeWeights(order=3)
        # construct molgrid
        for grid_type, deci in (
            ("coarse", 3),
            ("medium", 4),
            ("fine", 5),
            ("veryfine", 6),
            ("ultrafine", 6),
            ("insane", 6),
        ):
            mg = MolGrid.from_preset(numbers, coordinates, grid_type, rgrid, becke)
            dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
            dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
            fn = np.exp(-2 * dist0) / np.pi + np.exp(-2 * dist1) / np.pi
            occupation = mg.integrate(fn)
            assert_almost_equal(occupation, 2.0, decimal=deci)

    def test_make_grid_integral_with_default_rgrid(self):
        """Test molecular make_grid works as designed with default rgrid."""
        numbers = np.array([1, 1])
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], float)
        becke = BeckeWeights(order=3)
        # construct molgrid
        for grid_type, deci in (
            ("coarse", 3),
            ("medium", 4),
            ("fine", 4),
            ("veryfine", 5),
            ("ultrafine", 5),
            ("insane", 5),
        ):
            print(grid_type)
            mg = MolGrid.from_preset(numbers, coordinates, grid_type, aim_weights=becke)
            dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
            dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
            fn = np.exp(-2 * dist0) / np.pi + np.exp(-2 * dist1) / np.pi
            occupation = mg.integrate(fn)
            assert_almost_equal(occupation, 2.0, decimal=deci)

    def test_make_grid_different_grid_type(self):
        """Test different kind molgrid initizalize setting."""
        # three different radial grid
        rad2 = GaussLaguerre(50)
        rad3 = GaussLaguerre(70)
        # construct grid
        numbers = np.array([1, 8, 1])
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]], float)
        becke = BeckeWeights(order=3)

        # grid_type test with list
        mg = MolGrid.from_preset(
            numbers,
            coordinates,
            ["fine", "veryfine", "medium"],
            rad2,
            becke,
            store=True,
            rotate=False,
        )
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        dist2 = np.sqrt(((coordinates[2] - mg.points) ** 2).sum(axis=1))
        fn = (np.exp(-2 * dist0) + np.exp(-2 * dist1) + np.exp(-2 * dist2)) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 3, decimal=3)

        atgrid1 = AtomGrid.from_preset(
            atnum=numbers[0], preset="fine", center=coordinates[0], rgrid=rad2
        )
        atgrid2 = AtomGrid.from_preset(
            atnum=numbers[1], preset="veryfine", center=coordinates[1], rgrid=rad2
        )
        atgrid3 = AtomGrid.from_preset(
            atnum=numbers[2], preset="medium", center=coordinates[2], rgrid=rad2
        )
        assert_allclose(mg._atgrids[0].points, atgrid1.points)
        assert_allclose(mg._atgrids[1].points, atgrid2.points)
        assert_allclose(mg._atgrids[2].points, atgrid3.points)

        # grid type test with dict
        mg = MolGrid.from_preset(
            numbers,
            coordinates,
            {1: "fine", 8: "veryfine"},
            rad3,
            becke,
            store=True,
            rotate=False,
        )
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        dist2 = np.sqrt(((coordinates[2] - mg.points) ** 2).sum(axis=1))
        fn = (np.exp(-2 * dist0) + np.exp(-2 * dist1) + np.exp(-2 * dist2)) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 3, decimal=3)

        atgrid1 = AtomGrid.from_preset(
            atnum=numbers[0], preset="fine", center=coordinates[0], rgrid=rad3
        )
        atgrid2 = AtomGrid.from_preset(
            atnum=numbers[1], preset="veryfine", center=coordinates[1], rgrid=rad3
        )
        atgrid3 = AtomGrid.from_preset(
            atnum=numbers[2], preset="fine", center=coordinates[2], rgrid=rad3
        )
        assert_allclose(mg._atgrids[0].points, atgrid1.points)
        assert_allclose(mg._atgrids[1].points, atgrid2.points)
        assert_allclose(mg._atgrids[2].points, atgrid3.points)

    def test_make_grid_different_rad_type(self):
        """Test different radial grid input for make molgrid."""
        # radial grid test with list
        rad1 = GaussLaguerre(30)
        rad2 = GaussLaguerre(50)
        rad3 = GaussLaguerre(70)
        # construct grid
        numbers = np.array([1, 8, 1])
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]], float)
        becke = BeckeWeights(order=3)
        # construct molgrid
        mg = MolGrid.from_preset(
            numbers,
            coordinates,
            {1: "fine", 8: "veryfine"},
            [rad1, rad2, rad3],
            becke,
            store=True,
            rotate=False,
        )
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        dist2 = np.sqrt(((coordinates[2] - mg.points) ** 2).sum(axis=1))
        fn = (np.exp(-2 * dist0) + np.exp(-2 * dist1) + np.exp(-2 * dist2)) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 3, decimal=3)

        atgrid1 = AtomGrid.from_preset(
            atnum=numbers[0], preset="fine", center=coordinates[0], rgrid=rad1
        )
        atgrid2 = AtomGrid.from_preset(
            atnum=numbers[1], preset="veryfine", center=coordinates[1], rgrid=rad2
        )
        atgrid3 = AtomGrid.from_preset(
            atnum=numbers[2], preset="fine", center=coordinates[2], rgrid=rad3
        )
        assert_allclose(mg._atgrids[0].points, atgrid1.points)
        assert_allclose(mg._atgrids[1].points, atgrid2.points)
        assert_allclose(mg._atgrids[2].points, atgrid3.points)

        # radial grid test with dict
        mg = MolGrid.from_preset(
            numbers,
            coordinates,
            {1: "fine", 8: "veryfine"},
            {1: rad1, 8: rad3},
            becke,
            store=True,
            rotate=False,
        )
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        dist2 = np.sqrt(((coordinates[2] - mg.points) ** 2).sum(axis=1))
        fn = (np.exp(-2 * dist0) + np.exp(-2 * dist1) + np.exp(-2 * dist2)) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 3, decimal=3)

        atgrid1 = AtomGrid.from_preset(
            atnum=numbers[0], preset="fine", center=coordinates[0], rgrid=rad1
        )
        atgrid2 = AtomGrid.from_preset(
            atnum=numbers[1], preset="veryfine", center=coordinates[1], rgrid=rad3
        )
        atgrid3 = AtomGrid.from_preset(
            atnum=numbers[2], preset="fine", center=coordinates[2], rgrid=rad1
        )
        assert_allclose(mg._atgrids[0].points, atgrid1.points)
        assert_allclose(mg._atgrids[1].points, atgrid2.points)
        assert_allclose(mg._atgrids[2].points, atgrid3.points)

    def test_make_grid_different_grid_type_sg_0_1_2(self):
        """Test different kind molgrid initizalize setting."""
        # three different radial grid
        rad1 = GaussLaguerre(_get_rgrid_size("sg_0", atnums=1)[0])
        rad2 = GaussLaguerre(_get_rgrid_size("sg_2", atnums=8)[0])
        rad3 = GaussLaguerre(_get_rgrid_size("sg_1", atnums=1)[0])
        # construct grid
        numbers = np.array([1, 8, 1])
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]], float)
        becke = BeckeWeights(order=3)

        # grid_type test with list
        mg = MolGrid.from_preset(
            numbers,
            coordinates,
            ["sg_0", "sg_2", "sg_1"],
            [rad1, rad2, rad3],
            becke,
            store=True,
            rotate=False,
        )

        atgrid1 = AtomGrid.from_preset(
            atnum=numbers[0], preset="sg_0", center=coordinates[0], rgrid=rad1
        )
        atgrid2 = AtomGrid.from_preset(
            atnum=numbers[1], preset="sg_2", center=coordinates[1], rgrid=rad2
        )
        atgrid3 = AtomGrid.from_preset(
            atnum=numbers[2], preset="sg_1", center=coordinates[2], rgrid=rad3
        )
        assert_allclose(mg._atgrids[0].points, atgrid1.points)
        assert_allclose(mg._atgrids[1].points, atgrid2.points)
        assert_allclose(mg._atgrids[2].points, atgrid3.points)

        # three different radial grid
        rad2 = GaussLaguerre(_get_rgrid_size("sg_2", atnums=8)[0])
        rad3 = GaussLaguerre(_get_rgrid_size("sg_1", atnums=1)[0])

        # grid type test with dict
        mg = MolGrid.from_preset(
            numbers,
            coordinates,
            {1: "sg_1", 8: "sg_2"},
            [rad3, rad2, rad3],
            becke,
            store=True,
            rotate=False,
        )

        atgrid1 = AtomGrid.from_preset(
            atnum=numbers[0], preset="sg_1", center=coordinates[0], rgrid=rad3
        )
        atgrid2 = AtomGrid.from_preset(
            atnum=numbers[1], preset="sg_2", center=coordinates[1], rgrid=rad2
        )
        atgrid3 = AtomGrid.from_preset(
            atnum=numbers[2], preset="sg_1", center=coordinates[2], rgrid=rad3
        )
        assert_allclose(mg._atgrids[0].points, atgrid1.points)
        assert_allclose(mg._atgrids[1].points, atgrid2.points)
        assert_allclose(mg._atgrids[2].points, atgrid3.points)

    def test_make_grid_different_grid_type_g1_g2_g3_g4_g6_g7(self):
        """Test different kind molgrid initizalize setting."""
        # three different radial grid
        rad1 = GaussLaguerre(_get_rgrid_size("g1", atnums=1)[0])
        rad2 = GaussLaguerre(_get_rgrid_size("g2", atnums=8)[0])
        rad3 = GaussLaguerre(_get_rgrid_size("g3", atnums=1)[0])
        # construct grid
        numbers = np.array([1, 8, 1])
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]], float)
        becke = BeckeWeights(order=3)

        # grid_type test with list
        mg = MolGrid.from_preset(
            numbers,
            coordinates,
            ["g1", "g2", "g3"],
            [rad1, rad2, rad3],
            becke,
            store=True,
            rotate=False,
        )

        atgrid1 = AtomGrid.from_preset(
            atnum=numbers[0], preset="g1", center=coordinates[0], rgrid=rad1
        )
        atgrid2 = AtomGrid.from_preset(
            atnum=numbers[1], preset="g2", center=coordinates[1], rgrid=rad2
        )
        atgrid3 = AtomGrid.from_preset(
            atnum=numbers[2], preset="g3", center=coordinates[2], rgrid=rad3
        )
        assert_allclose(mg._atgrids[0].points, atgrid1.points)
        assert_allclose(mg._atgrids[1].points, atgrid2.points)
        assert_allclose(mg._atgrids[2].points, atgrid3.points)

        # three different radial grid
        rad1 = GaussLaguerre(_get_rgrid_size("g4", atnums=1)[0])
        rad2 = GaussLaguerre(_get_rgrid_size("g5", atnums=8)[0])
        rad3 = GaussLaguerre(_get_rgrid_size("g6", atnums=1)[0])

        mg = MolGrid.from_preset(
            numbers,
            coordinates,
            ["g4", "g5", "g6"],
            [rad1, rad2, rad3],
            becke,
            store=True,
            rotate=False,
        )

        atgrid1 = AtomGrid.from_preset(
            atnum=numbers[0], preset="g4", center=coordinates[0], rgrid=rad1
        )
        atgrid2 = AtomGrid.from_preset(
            atnum=numbers[1], preset="g5", center=coordinates[1], rgrid=rad2
        )
        atgrid3 = AtomGrid.from_preset(
            atnum=numbers[2], preset="g6", center=coordinates[2], rgrid=rad3
        )
        assert_allclose(mg._atgrids[0].points, atgrid1.points)
        assert_allclose(mg._atgrids[1].points, atgrid2.points)
        assert_allclose(mg._atgrids[2].points, atgrid3.points)

    def test_integrate_hydrogen_pair_1s(self):
        """Test molecular integral in H2."""
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], float)
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[1],
        )
        becke = BeckeWeights(order=3)
        mg = MolGrid(np.array([1, 1]), [atg1, atg2], becke)
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi + np.exp(-2 * dist1) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 2.0, decimal=6)

    def test_integrate_hydrogen_trimer_1s(self):
        """Test molecular integral in H3."""
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]], float)
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[1],
        )
        atg3 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[2],
        )
        becke = BeckeWeights(order=3)
        mg = MolGrid(np.array([1, 1, 1]), [atg1, atg2, atg3], becke)
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        dist2 = np.sqrt(((coordinates[2] - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi + np.exp(-2 * dist1) / np.pi + np.exp(-2 * dist2) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 3.0, decimal=4)

    def test_integrate_hydrogen_8_1s(self):
        """Test molecular integral in H2."""
        x, y, z = np.meshgrid(*(3 * [[-0.5, 0.5]]))
        centers = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        atgs = [
            AtomGrid.from_pruned(
                self.rgrid,
                0.5,
                r_sectors=np.array([]),
                d_sectors=np.array([17]),
                center=center,
            )
            for center in centers
        ]

        becke = BeckeWeights(order=3)
        mg = MolGrid(np.array([1] * len(centers)), atgs, becke)
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
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            1.228,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomGrid.from_pruned(
            self.rgrid,
            0.945,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[1],
        )
        becke = BeckeWeights(order=3)
        mg = MolGrid(np.array([6, 8]), [atg1, atg2], becke, store=True)
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
            assert isinstance(atgrid, AtomGrid)
            assert atgrid.size == 100 * 110
            assert atgrid.points.shape == (100 * 110, 3)
            assert atgrid.weights.shape == (100 * 110,)
            assert (atgrid.center == coordinates[i]).all()
        mg = MolGrid(np.array([6, 8]), [atg1, atg2], becke)
        for i in range(2):
            atgrid = mg[i]
            assert isinstance(atgrid, LocalGrid)
            assert_allclose(atgrid.center, mg._atcoords[i])

    def test_molgrid_attrs(self):
        """Test MolGrid attributes."""
        # numbers = np.array([6, 8], int)
        coordinates = np.array([[0.0, 0.2, -0.5], [0.1, 0.0, 0.5]], float)
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            1.228,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomGrid.from_pruned(
            self.rgrid,
            0.945,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[1],
        )

        becke = BeckeWeights(order=3)
        mg = MolGrid(np.array([6, 8]), [atg1, atg2], becke, store=True)

        assert mg.size == 2 * 110 * 100
        assert mg.points.shape == (mg.size, 3)
        assert mg.weights.shape == (mg.size,)
        assert mg.aim_weights.shape == (mg.size,)
        assert mg.get_atomic_grid(0) is atg1
        assert mg.get_atomic_grid(1) is atg2

        simple_ag1 = mg.get_atomic_grid(0)
        simple_ag2 = mg.get_atomic_grid(1)
        assert_allclose(simple_ag1.points, atg1.points)
        assert_allclose(simple_ag1.weights, atg1.weights)
        assert_allclose(simple_ag2.weights, atg2.weights)

        # test molgrid is not stored
        mg2 = MolGrid(np.array([6, 8]), [atg1, atg2], becke, store=False)
        assert mg2._atgrids is None
        simple2_ag1 = mg2.get_atomic_grid(0)
        simple2_ag2 = mg2.get_atomic_grid(1)
        assert isinstance(simple2_ag1, LocalGrid)
        assert isinstance(simple2_ag2, LocalGrid)
        assert_allclose(simple2_ag1.points, atg1.points)
        assert_allclose(simple2_ag1.weights, atg1.weights)
        assert_allclose(simple2_ag2.weights, atg2.weights)
        # assert mg.subgrids is None
        # assert mg.k == 3
        # assert mg.random_rotate

    def test_different_aim_weights_h2(self):
        """Test different aim_weights for molgrid."""
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], float)
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[1],
        )
        # use an array as aim_weights
        mg = MolGrid(np.array([1, 1]), [atg1, atg2], np.ones(22000))
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi + np.exp(-2 * dist1) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 4.0, decimal=4)

    def test_from_size(self):
        """Test horton style grid."""
        nums = np.array([1, 1])
        coors = np.array([[0, 0, -0.5], [0, 0, 0.5]])
        becke = BeckeWeights(order=3)
        mol_grid = MolGrid.from_size(nums, coors, 110, self.rgrid, becke, rotate=False)
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coors[0],
        )
        atg2 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coors[1],
        )
        ref_grid = MolGrid(nums, [atg1, atg2], becke, store=True)
        assert_allclose(ref_grid.points, mol_grid.points)
        assert_allclose(ref_grid.weights, mol_grid.weights)

    def test_from_pruned(self):
        r"""Test MolGrid construction via from_pruned method."""
        nums = np.array([1, 1])
        coors = np.array([[0, 0, -0.5], [0, 0, 0.5]])
        becke = BeckeWeights(order=3)
        radius = np.array([1.0, 0.5])
        r_sectors = [[0.5, 1.0, 1.5], [0.25, 0.5]]
        d_sectors = [[3, 7, 5, 3], [3, 2, 2]]
        mol_grid = MolGrid.from_pruned(
            nums,
            coors,
            radius,
            r_sectors=r_sectors,
            rgrid=self.rgrid,
            aim_weights=becke,
            d_sectors=d_sectors,
            rotate=False,
        )
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            radius[0],
            r_sectors=r_sectors[0],
            d_sectors=d_sectors[0],
            center=coors[0],
        )
        atg2 = AtomGrid.from_pruned(
            self.rgrid,
            radius[1],
            r_sectors=r_sectors[1],
            d_sectors=d_sectors[1],
            center=coors[1],
        )
        ref_grid = MolGrid(nums, [atg1, atg2], becke, store=True)
        assert_allclose(ref_grid.points, mol_grid.points)
        assert_allclose(ref_grid.weights, mol_grid.weights)

    def test_raise_errors(self):
        """Test molgrid errors raise."""
        atg = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=np.array([0.0, 0.0, 0.0]),
        )

        # errors of aim_weight
        with self.assertRaises(TypeError):
            MolGrid(atnums=np.array([1]), atgrids=[atg], aim_weights="test")
        with self.assertRaises(ValueError):
            MolGrid(atnums=np.array([1]), atgrids=[atg], aim_weights=np.array(3))
        with self.assertRaises(TypeError):
            MolGrid(atnums=np.array([1]), atgrids=[atg], aim_weights=[3, 5])

        # integrate errors
        becke = BeckeWeights({1: 0.472_431_53}, order=3)
        molg = MolGrid(np.array([1]), [atg], becke)
        with self.assertRaises(ValueError):
            molg.integrate()
        with self.assertRaises(TypeError):
            molg.integrate(1)
        with self.assertRaises(ValueError):
            molg.integrate(np.array([3, 5]))
        with self.assertRaises(ValueError):
            molg.get_atomic_grid(-3)
        molg = MolGrid(np.array([1]), [atg], becke, store=True)
        with self.assertRaises(ValueError):
            molg.get_atomic_grid(-5)

        # test make_grid error
        pts = UniformInteger(70)
        tf = ExpRTransform(1e-5, 2e1)
        rgrid = tf.transform_1d_grid(pts)
        numbers = np.array([1, 1])
        becke = BeckeWeights(order=3)
        # construct molgrid
        with self.assertRaises(ValueError):
            MolGrid.from_preset(numbers, np.array([0.0, 0.0, 0.0]), "fine", rgrid, becke)
        with self.assertRaises(ValueError):
            MolGrid.from_preset(np.array([1, 1]), np.array([[0.0, 0.0, 0.0]]), "fine", rgrid, becke)
        with self.assertRaises(ValueError):
            MolGrid.from_preset(np.array([1, 1]), np.array([[0.0, 0.0, 0.0]]), "fine", rgrid, becke)
        with self.assertRaises(TypeError):
            MolGrid.from_preset(
                np.array([1, 1]),
                np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]),
                "fine",
                becke,
                {3, 5},
            )
        with self.assertRaises(TypeError):
            MolGrid.from_preset(
                np.array([1, 1]),
                np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]),
                np.array([3, 5]),
                rgrid,
                becke,
            )

    def test_get_localgrid_1s(self):
        """Test local grid for a molecule with one atom."""
        nums = np.array([1])
        coords = np.array([0.0, 0.0, 0.0])

        # initialize MolGrid with atomic grid
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coords,
        )
        grid = MolGrid(np.array([1]), [atg1], BeckeWeights(), store=False)
        fn = np.exp(-2 * np.linalg.norm(grid.points, axis=-1))
        assert_allclose(grid.integrate(fn), np.pi)
        # conventional local grid
        localgrid = grid.get_localgrid(coords, 12.0)
        localfn = np.exp(-2 * np.linalg.norm(localgrid.points, axis=-1))
        assert localgrid.size < grid.size
        assert localgrid.size == 10560
        assert_allclose(localgrid.integrate(localfn), np.pi)
        assert_allclose(fn[localgrid.indices], localfn)
        # "whole" loal grid, useful for debugging code using local grids
        wholegrid = grid.get_localgrid(coords, np.inf)
        assert wholegrid.size == grid.size
        assert_allclose(wholegrid.points, grid.points)
        assert_allclose(wholegrid.weights, grid.weights)
        assert_allclose(wholegrid.indices, np.arange(grid.size))

        # initialize MolGrid like horton
        grid = MolGrid.from_size(nums, coords[np.newaxis, :], 110, self.rgrid, store=True)
        fn = np.exp(-4.0 * np.linalg.norm(grid.points, axis=-1))
        assert_allclose(grid.integrate(fn), np.pi / 8)
        localgrid = grid.get_localgrid(coords, 5.0)
        localfn = np.exp(-4.0 * np.linalg.norm(localgrid.points, axis=-1))
        assert localgrid.size < grid.size
        assert localgrid.size == 9900
        assert_allclose(localgrid.integrate(localfn), np.pi / 8, rtol=1e-5)
        assert_allclose(fn[localgrid.indices], localfn)

    def test_get_localgrid_1s1s(self):
        """Test local grid for a molecule with one atom."""
        nums = np.array([1, 3])
        coords = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
        grid = MolGrid.from_size(
            nums, coords, 110, self.rgrid, BeckeWeights(), store=True, rotate=False
        )
        fn0 = np.exp(-4.0 * np.linalg.norm(grid.points - coords[0], axis=-1))
        fn1 = np.exp(-8.0 * np.linalg.norm(grid.points - coords[1], axis=-1))
        assert_allclose(grid.integrate(fn0), np.pi / 8, rtol=1e-5)
        assert_allclose(grid.integrate(fn1), np.pi / 64)
        # local grid centered on atom 0 to evaluate fn0
        local0 = grid.get_localgrid(coords[0], 5.0)
        assert local0.size < grid.size
        localfn0 = np.exp(-4.0 * np.linalg.norm(local0.points - coords[0], axis=-1))
        assert_allclose(fn0[local0.indices], localfn0)
        assert_allclose(local0.integrate(localfn0), np.pi / 8, rtol=1e-5)
        # local grid centered on atom 1 to evaluate fn1
        local1 = grid.get_localgrid(coords[1], 2.5)
        assert local1.size < grid.size
        localfn1 = np.exp(-8.0 * np.linalg.norm(local1.points - coords[1], axis=-1))
        assert_allclose(local1.integrate(localfn1), np.pi / 64, rtol=1e-6)
        assert_allclose(fn1[local1.indices], localfn1)
        # approximate the sum of fn0 and fn2 by combining results from local grids.
        fnsum = np.zeros(grid.size)
        fnsum[local0.indices] += localfn0
        fnsum[local1.indices] += localfn1
        assert_allclose(grid.integrate(fnsum), np.pi * (1 / 8 + 1 / 64), rtol=1e-5)

    def test_integrate_hirshfeld_weights_single_1s(self):
        """Test molecular integral in H atom with Hirshfeld weights."""
        pts = UniformInteger(100)
        tf = ExpRTransform(1e-5, 2e1)
        rgrid = tf.transform_1d_grid(pts)
        coordinates = np.array([0.0, 0.0, -0.5])
        atg1 = AtomGrid.from_pruned(
            rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates,
        )
        mg = MolGrid(np.array([7]), [atg1], HirshfeldWeights())
        dist0 = np.sqrt(((coordinates - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 1.0, decimal=6)

    def test_integrate_hirshfeld_weights_pair_1s(self):
        """Test molecular integral in H2."""
        coordinates = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
        atg1 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[0],
        )
        atg2 = AtomGrid.from_pruned(
            self.rgrid,
            0.5,
            r_sectors=np.array([]),
            d_sectors=np.array([17]),
            center=coordinates[1],
        )
        mg = MolGrid(np.array([1, 1]), [atg1, atg2], HirshfeldWeights())
        dist0 = np.sqrt(((coordinates[0] - mg.points) ** 2).sum(axis=1))
        dist1 = np.sqrt(((coordinates[1] - mg.points) ** 2).sum(axis=1))
        fn = np.exp(-2 * dist0) / np.pi + 1.5 * np.exp(-2 * dist1) / np.pi
        occupation = mg.integrate(fn)
        assert_almost_equal(occupation, 2.5, decimal=5)

    def test_multipole_translation(self):
        """Test that pure solid harmonic moments translate correctly."""
        l_max = 3 # Maximum multipole order to test
        alpha = 2.0 # Gaussian exponent
        center_orig = np.array([0.0, 0.0, 0.0])
        shift_vec = np.array([0.1, -0.2, 0.3]) # A small shift vector
        center_shifted = center_orig + shift_vec

        # --- Grid Setup ---
        # Use a reasonable grid settings for accuracy
        # Use GaussLaguerre directly as it produces points on (0, inf)
        rgrid = GaussLaguerre(75) # More points for radial grid
        # tf = ExpRTransform(alpha=1.0, r0=1e-7) # Use ExpRTransform which maps (0, inf)
        # rgrid = tf.transform_1d_grid(pts) <-- Remove transformation step
        numbers = np.array([1]) # Single Hydrogen atom (simplest case)
        # Place atom at origin for simplicity now, grid should extend enough
        atom_coord = np.array([0.0, 0.0, 0.0])
        # Use a fine preset for better angular/pruning accuracy
        grid_preset = "fine" # "fine" or "veryfine" recommended
        atg1 = AtomGrid.from_preset(
            atnum=numbers[0], preset=grid_preset, center=atom_coord, rgrid=rgrid
        )
        becke = BeckeWeights(order=3) # Becke weights
        mg = MolGrid(numbers, [atg1], becke)
        # ------------------

        # --- Define test function (Gaussian centered at origin) ---
        def gaussian_density(points, center, exp):
            dist_sq = np.sum((points - center)**2, axis=1)
            norm = (exp / np.pi)**(1.5) # Normalization for gaussian exp(-exp*r^2)
            return norm * np.exp(-exp * dist_sq)

        # Evaluate the function centered at the global origin
        func_vals = gaussian_density(mg.points, center=np.array([0.0, 0.0, 0.0]), exp=alpha)
        # ---------------------------------------------------------

        # Calculate moments directly at original and shifted centers
        moments_orig = mg.moments(func_vals=func_vals, orders=l_max, centers=center_orig.reshape(1,3), type_mom="pure")
        moments_shifted_direct = mg.moments(func_vals=func_vals, orders=l_max, centers=center_shifted.reshape(1,3), type_mom="pure")

        # Squeeze the output from (L, 1) to (L,) to match analytical calculation shape
        moments_orig = moments_orig.squeeze()
        moments_shifted_direct = moments_shifted_direct.squeeze()

        # Calculate solid harmonics of the shift vector
        shift_vec_cart = shift_vec.reshape(1, 3)
        # Need r, theta, phi for solid_harmonics
        r_shift = np.linalg.norm(shift_vec)
        if r_shift == 0:
            # If shift is zero, R_lm is non-zero only for l=m=0, handle separately or ensure shift > 0
             R_lm_a = np.zeros(((l_max + 1)**2,))
             R_lm_a[0] = 1.0 # R_00 = 1/sqrt(4pi), but solid_harmonics includes the sqrt(4pi/(2l+1)) factor
             # Let's re-check the solid_harmonics definition R_lm = sqrt(4pi/(2l+1)) r^l Y_lm
             # For l=0, m=0: R_00 = sqrt(4pi/1) * r^0 * Y_00 = sqrt(4pi) * 1 * (1/sqrt(4pi)) = 1.0
        else:
            shift_vec_sph = convert_cart_to_sph(shift_vec_cart) # Returns (1, 3) array [r, theta, phi]
            R_lm_a = solid_harmonics(l_max, shift_vec_sph).flatten() # Get R_lm(a) in flat Horton order

        # Calculate analytically shifted moments
        num_moments = (l_max + 1)**2
        moments_shifted_analytical = np.zeros_like(moments_orig, dtype=np.float64)

        for k_lm in range(num_moments):
            l, m = get_lm_from_horton_index(k_lm)
            term_sum = 0.0
            for k_lambda_mu in range(num_moments):
                lam, mu = get_lm_from_horton_index(k_lambda_mu)
                if lam > l:
                    continue

                l_minus_lambda = l - lam
                m_minus_mu = m - mu

                if abs(m_minus_mu) > l_minus_lambda:
                    continue

                # Binomial term: sqrt(C(l+m, lam+mu) * C(l-m, lam-mu))
                bc1 = binomial_safe(l + m, lam + mu)
                bc2 = binomial_safe(l - m, lam - mu)
                binom_prod = bc1 * bc2

                if binom_prod < 0: # Should not happen with correct binomial_safe
                    binom_term = 0.0
                else:
                    binom_term = np.sqrt(binom_prod)

                if np.isnan(binom_term): # Handle NaN just in case
                    binom_term = 0.0

                # Get R_{l-lambda, m-mu}(a)
                k_R = horton_index(l_minus_lambda, m_minus_mu)
                # Ensure index is valid before accessing
                if k_R >= len(R_lm_a):
                    R_term = 0.0 # Index out of bounds implies this term is zero
                else:
                    R_term = R_lm_a[k_R]

                # Get M_{lambda, mu}(0)
                M_term = moments_orig[k_lambda_mu]

                # Accumulate sum: (-1)^(l-lambda) * binom_term * R_term * M_term
                term_sum += ((-1)**(l - lam)) * binom_term * R_term * M_term

            moments_shifted_analytical[k_lm] = term_sum

        # Compare results with appropriate tolerance
        # Tolerance might need adjustment based on grid quality and l_max
        tolerance_kwargs = {'rtol': 1e-5, 'atol': 1e-7}
        if l_max > 2: # Potentially lower tolerance for higher moments
            tolerance_kwargs = {'rtol': 1e-4, 'atol': 1e-6}

        # Provide helpful output on failure
        # print(f"Lmax={l_max}, Shift={shift_vec}")
        # print(f"Moments Original (k=0..{num_moments-1}): {moments_orig}")
        # print(f"Moments Shifted Direct: {moments_shifted_direct}")
        # print(f"Moments Shifted Analytical: {moments_shifted_analytical}")
        # diff = moments_shifted_direct - moments_shifted_analytical
        # print(f"Difference (Direct - Analytical): {diff}")
        # print(f"Max Abs Diff: {np.max(np.abs(diff))}")
        # print(f"Indices of large diff: {np.where(np.abs(diff) > tolerance_kwargs['atol'] + tolerance_kwargs['rtol'] * np.abs(moments_shifted_analytical))}")

        assert_allclose(moments_shifted_direct, moments_shifted_analytical, **tolerance_kwargs)


def test_interpolation_with_gaussian_center():
    """Test if get_multipole works with a specific center."""
    coordinates = np.array([[0.0, 0.0, -1.5], [0.0, 0.0, 1.5]])

    pts = Trapezoidal(400)
    tf = LinearFiniteRTransform(1e-8, 10.0)
    rgrid = tf.transform_1d_grid(pts)

    atg1 = AtomGrid(rgrid, degrees=[15], center=coordinates[0])
    atg2 = AtomGrid(rgrid, degrees=[17], center=coordinates[1])
    mg = MolGrid(np.array([1, 1]), [atg1, atg2], BeckeWeights(), store=True)

    def gaussian_func(pts):
        return np.exp(-5.0 * np.linalg.norm(pts - coordinates[0], axis=1) ** 2.0) + np.exp(
            -3.5 * np.linalg.norm(pts - coordinates[1], axis=1) ** 2.0
        )

    gaussians = gaussian_func(mg.points)
    interpolate_func = mg.interpolate(gaussians)

    assert_almost_equal(interpolate_func(mg.points), gaussians, decimal=3)

    # Test interpolation at new points
    new_pts = np.random.uniform(2.0, 2.0, size=(100, 3))
    assert_almost_equal(interpolate_func(new_pts), gaussian_func(new_pts), decimal=3)
