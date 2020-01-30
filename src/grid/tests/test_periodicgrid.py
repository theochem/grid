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
"""PeriodicGrid tests file."""
from grid.basegrid import Grid
from grid.periodicgrid import PeriodicGrid, PeriodicGridWarning

import numpy as np
from numpy.testing import assert_allclose

import pytest


# a small selection of displacement vectors to parametrize tests
DELTAS_1D = [0.0, 0.5, 40.0, -5.0]
DELTAS_2D = [[0.0, 0.0], [0.5, -0.2], [40.0, -60.0]]


# text formatting for the deltas
def format_1d(d):
    """Format the ID of a DELTA."""
    return "'{:.1f}'".format(d)


def format_nd(ds):
    """Format the ID of a DELTAS vec."""
    return "'{}'".format(",".join(format(d, ".1f") for d in ds))


@pytest.mark.parametrize("delta_p", DELTAS_1D, ids=format_1d)
@pytest.mark.parametrize("delta_c", DELTAS_1D, ids=format_1d)
def test_tutorial_periodic_repetition(delta_p, delta_c):
    """Test example of the periodic repetition of a local non-periodic function.

    The 1D grid has a periodicity of a=0.5. On this grid, a periodic repetition
    of a Gaussian function is evaluated and tested for consistency.
    """
    a = 0.5276
    # Our grid covers multiple primitive cells, which is not great for
    # efficiency, but it still works. Due to the efficiency issue, a warning
    # is raised. To fix this issue, any may simply add the ``wrap=True``
    # argument to the constructor. (This modifies the grid points.)
    with pytest.warns(PeriodicGridWarning):
        grid = PeriodicGrid(
            np.linspace(-1, 1, 21) + delta_p, np.full(21, 0.1), np.array([a])
        )
    # The subgrid is wider than one primitive cell, such that there will be
    # more points in the subgrid than in the periodic grid. The test is repeated
    # with two displacements of the center by an integer multiple of the
    # periodic unit.
    center = 1.8362 - delta_c
    radius = 2.1876
    fn_old = None
    for _ in range(3):
        subgrid = grid.get_subgrid(center, radius)
        assert subgrid.size > grid.size
        # Compute one Gaussian, which is to be periodically repeated.
        subfn = np.exp(-20 * (subgrid.points - center) ** 2)
        # Construct the periodic model on grid points of one primitive cell.
        # Mind the ``np.add.at`` line. This is explained in the documentation of
        # PeriodicGrid.__init__.
        fn = np.zeros(grid.size)
        np.add.at(fn, subgrid.indices, subfn)
        # Compare the periodic function to the result with the previous value
        # of center. It should be the same because the center was translated by
        # exactly one periodic unit.
        if fn_old is not None:
            assert_allclose(fn, fn_old)
        fn_old = fn
        # The integral over one primitive cell should be the same as the integral
        # over one isolated Gaussian.
        assert_allclose(subgrid.integrate(subfn), grid.integrate(fn))
        # Manually construct the periodic repetition and compare.
        fn2 = np.zeros(grid.size)
        jmin = np.ceil((grid.points.min() - center - radius) / a).astype(int)
        jmax = np.floor((grid.points.max() - center + radius) / a).astype(int)
        assert jmax >= jmin
        for j in range(jmin, jmax + 1):
            center_translated = center + a * j
            fn2 += np.exp(-20 * (grid.points - center_translated) ** 2)
        assert_allclose(fn2, fn)
        # Modify the center for the next iteration.
        center += a


def setup_equidistant_grid(origin, realvecs, npts):
    """Define a periodic grid with equally spaced grid points.

    The relative vectors between neigbouring grid points are constant and equal
    to the real-space lattice vectors divided by the number of grid points along
    one lattice vector.

    At the moment, this function is just used for testing, but this might be
    useful to generate grids for cube files and for other types of
    visualization.

    Parameters
    ----------
    origin: np.array(shape=(ndim, ), dtype=float)
        The position of the first grid point.
    realvecs: np.array(shape=(ndim, ndim), dtype=float)
        The real-space lattice vectors.
    ntps: np.array(shape=ndim, dtype=int)
        The number of grid points along each lattice vector.

    Returns
    -------
    PeriodicGrid
        The periodic grid with equidistant points.

    """
    # First build grid points in fractional coordinates and then transform
    # to real space.
    fractional_tuple = np.meshgrid(*[np.arange(npt) / npt for npt in npts])
    points = (
        sum(
            [
                np.outer(fractional.ravel(), realvec)
                for fractional, realvec in zip(fractional_tuple, realvecs)
            ]
        )
        + origin
    )
    # The weights take into account the Jacobian of the affine transformation
    # from fractional to Cartesian grid points.
    npt_total = np.product(npts)
    weights = np.full(npt_total, abs(np.linalg.det(realvecs)) / npt_total)
    return PeriodicGrid(points, weights, realvecs)


@pytest.mark.parametrize("delta_p", DELTAS_2D, ids=format_nd)
@pytest.mark.parametrize("delta_c", DELTAS_2D, ids=format_nd)
def test_tutorial_local_integral_1(delta_p, delta_c):
    """Test example of the integration of a local function on a periodic grid.

    This is a trivial example in the sense that no periodic function is included
    in the integral.
    """
    # A) Setup a periodic grid with uniformly spaced grid points.
    realvecs = np.array([[0.3, 0.4], [1.0, -0.5]])
    grid = setup_equidistant_grid(delta_p, realvecs, [40, 40])
    # B) Get a subgrid centered at some point
    center = np.array([-1.1, 4.0]) - delta_c
    cutoff = 5.4097
    subgrid = grid.get_subgrid(center, cutoff)
    # C) Evaluate a quadratic function on the local grid, with a known
    # solution for the integral.
    dists = np.linalg.norm(subgrid.points - center, axis=1)
    assert (dists <= cutoff).all()
    localfn = (dists - cutoff) ** 2
    assert_allclose(subgrid.integrate(localfn), cutoff ** 4 * np.pi / 6)
    # D) Construct a periodic repetition of the local integrand and perform the
    # same check on the integral
    periodicfn = np.zeros(grid.size)
    np.add.at(periodicfn, subgrid.indices, localfn)
    assert_allclose(grid.integrate(periodicfn), cutoff ** 4 * np.pi / 6)


@pytest.mark.parametrize("delta_p", DELTAS_2D, ids=format_nd)
@pytest.mark.parametrize("delta_c", DELTAS_2D, ids=format_nd)
def test_tutorial_local_integral_2(delta_p, delta_c):
    """Test example of the integration of a local function on a periodic grid.

    The integrand in this example is the product of a periodic function and
    a local function.

    The periodic function is a hexagonal lattice of 2D Gaussians.
    The local function is just one 2D Gaussian.
    """
    # Set up a numerical integration grid.
    alpha = np.pi / 3
    a = 4.0
    realvecs = np.array([[a, 0.0], [a * np.cos(alpha), a * np.sin(alpha)]])
    grid = setup_equidistant_grid(delta_p, realvecs, [10, 10])
    # Create a subgrid to evaluate a single Gaussian, to be periodically repeated
    cutoff = 6.0
    center1 = np.array([0.3, 0.7]) - delta_c
    subgrid1 = grid.get_subgrid(center1, cutoff)
    dists1 = np.linalg.norm(subgrid1.points - center1, axis=1)
    localfn1 = np.exp(-0.5 * dists1 ** 2)
    periodicfn = np.zeros(grid.size)
    np.add.at(periodicfn, subgrid1.indices, localfn1)
    # Check some obvious integrals:
    # - Integral over the local function within the cutoff.
    assert_allclose(subgrid1.integrate(localfn1), 2 * np.pi)
    # - Integral over the periodic function within one unit cell.
    #   (This should be the same.)
    assert_allclose(grid.integrate(periodicfn), 2 * np.pi)
    # Make sure the periodic function is not trivially constant.
    # This can easily happen if the Gaussian is wide compared to the cell size.
    # The test would not be very challenging with a nearly constant periodic
    # function.
    assert periodicfn.std() > 0.1
    # Create another subgrid for the second Gaussian, which will remain local.
    center2 = np.array([-3.3, -0.7])
    subgrid2 = grid.get_subgrid(center2, cutoff)
    dists2 = np.linalg.norm(subgrid2.points - center2, axis=1)
    localfn2 = np.exp(-0.5 * dists2 ** 2)
    # Compute the overlap integral of the periodic and the second local Gaussian
    oint_a = subgrid2.integrate(localfn2, periodicfn[subgrid2.indices])
    # Compute this overlap integral as a lattice sum of overlap integrals between
    # individual Gaussians. This is done with a fairly mindless algorithm: first
    # the relative vector between the two centers under the minimum image
    # convention (``mic``) is computed and then some neighbors are included in
    # the lattice sum.
    oint_b = 0.0
    delta = center2 - center1
    frac_delta = np.dot(grid.recivecs, delta)
    frac_delta_mic = frac_delta - np.round(frac_delta)
    assert (abs(frac_delta_mic) <= 0.5).all()
    delta_mic = np.dot(grid.realvecs.T, frac_delta_mic)
    for i0 in range(-2, 3):
        for i1 in range(-2, 3):
            delta_01 = delta_mic + grid.realvecs[0] * i0 + grid.realvecs[1] * i1
            dist_01 = np.linalg.norm(delta_01)
            # Compute the overlap integral derived with Gaussian-Product theorem
            overlap_01 = np.pi * np.exp(-0.25 * dist_01 ** 2)
            oint_b += overlap_01
    assert_allclose(oint_a, oint_b)


def assert_equal_subgrids(subgrid1, subgrid2):
    """Assert the equality of two subgrids which may differ in point order.

    This check also works for the case that a point from the parent grid appears
    multiple times (due to periodic images).
    """
    assert_allclose(subgrid1.center, subgrid2.center)
    assert sorted(subgrid1.indices) == sorted(subgrid2.indices)
    for index in np.unique(subgrid1.indices):
        # Get all points corresponding to the index in both grids
        selection1 = (subgrid1.indices == index).nonzero()[0]
        selection2 = (subgrid2.indices == index).nonzero()[0]
        points1 = subgrid1.points[selection1]
        points2 = subgrid2.points[selection2]
        # The points in set 1 and 2 should form pairs of coinciding points
        # without too much ambiguity. Either they overlap or either they
        # differ by an integer linear combination of lattice vectors.
        found = set([])
        for i1, point1 in enumerate(points1):
            if subgrid1.points.ndim == 1:
                dists = abs(points2 - point1)
            else:
                dists = np.linalg.norm(points2 - point1, axis=1)
            i2 = dists.argmin()
            assert dists[i2] < 1e-10
            assert i2 not in found
            found.add(i2)
        # All selected weights should be equal
        weights1 = subgrid1.weights[selection1]
        weights2 = subgrid2.weights[selection2]
        assert_allclose(weights1, weights1.mean())
        assert_allclose(weights2, weights2.mean())
        assert_allclose(weights1, weights2)


class PeriodicGridTester:
    """Base class for PeriodicGrid test cases.

    Subclasses should override the define_reference_data method, which sets
    the following attributes:

    - ``self._ref_points``: grid points
    - ``self._ref_weights``: grid weights
    - ``self._ref_realvecs``: cell vectors
    - ``self._ref_recivecs``: the expected reciprocal cell vectors
    - ``self._ref_spacings``: the expected spacings between crystal planes
    - ``self._ref_frac_intvls``: the expected intervals of fractional
      coordinates.

    """

    def setup(self):
        """Initialize an unwrapped and a wrapped version of the grid."""
        self.define_reference_data()
        with pytest.warns(None if self._ref_realvecs is None else PeriodicGridWarning):
            self.grid = PeriodicGrid(
                self._ref_points, self._ref_weights, self._ref_realvecs
            )
        with pytest.warns(None) as record:
            self.wrapped_grid = PeriodicGrid(
                self._ref_points, self._ref_weights, self._ref_realvecs, True
            )
            assert len(record) == 0

    def define_reference_data(self):
        """Define reference data for the test."""
        raise NotImplementedError

    def test_init_grid(self):
        """Test PeriodicGrid init."""
        assert_allclose(self.grid.points, self._ref_points, atol=1e-7)
        assert_allclose(self.grid.frac_intvls, self._ref_frac_intvls)
        assert (self.wrapped_grid.frac_intvls[:, 0] >= 0).all()
        assert (self.wrapped_grid.frac_intvls[:, 1] <= 1).all()
        if self._ref_points.ndim == 2:
            frac_points = self.wrapped_grid.points @ self.wrapped_grid.recivecs.T
        else:
            frac_points = self.wrapped_grid.points * self.wrapped_grid.recivecs
        assert (frac_points >= 0).all
        assert (frac_points <= 1).all
        for grid in self.grid, self.wrapped_grid:
            assert isinstance(grid, PeriodicGrid)
            assert_allclose(grid.weights, self._ref_weights, atol=1e-7)
            if self._ref_realvecs is None:
                assert_allclose(grid.realvecs, self._ref_recivecs, atol=1e-10)
            else:
                assert_allclose(grid.realvecs, self._ref_realvecs, atol=1e-10)
            assert_allclose(grid.recivecs, self._ref_recivecs, atol=1e-10)
            assert_allclose(grid.spacings, self._ref_spacings, atol=1e-10)
            assert grid.size == self._ref_weights.size

    def test_get_item(self):
        """Test the __getitem__ method."""
        # test index
        grid_index = self.grid[10]
        ref_grid_index = PeriodicGrid(
            self._ref_points[10:11], self._ref_weights[10:11], self._ref_realvecs
        )
        assert_allclose(grid_index.points, ref_grid_index.points)
        assert_allclose(grid_index.weights, ref_grid_index.weights)
        assert_allclose(grid_index.realvecs, self.grid.realvecs)
        assert_allclose(grid_index.recivecs, self.grid.recivecs)
        assert_allclose(grid_index.spacings, self.grid.spacings)
        assert_allclose(grid_index.frac_intvls, ref_grid_index.frac_intvls)
        assert isinstance(grid_index, PeriodicGrid)
        # test slice
        with pytest.warns(None if self._ref_realvecs is None else PeriodicGridWarning):
            ref_grid_slice = PeriodicGrid(
                self._ref_points[:15], self._ref_weights[:15], self._ref_realvecs
            )
        with pytest.warns(None if self._ref_realvecs is None else PeriodicGridWarning):
            grid_slice = self.grid[:15]
        assert_allclose(grid_slice.points, ref_grid_slice.points)
        assert_allclose(grid_slice.weights, ref_grid_slice.weights)
        assert_allclose(grid_slice.realvecs, self.grid.realvecs)
        assert_allclose(grid_slice.recivecs, self.grid.recivecs)
        assert_allclose(grid_slice.spacings, self.grid.spacings)
        assert_allclose(grid_slice.frac_intvls, ref_grid_slice.frac_intvls)
        assert isinstance(grid_slice, PeriodicGrid)
        # test take with integers
        indices = np.array([1, 3, 5])
        grid_itake = self.grid[indices]
        assert_allclose(grid_itake.points, self._ref_points[indices])
        assert_allclose(grid_itake.weights, self._ref_weights[indices])
        assert_allclose(grid_itake.realvecs, self.grid.realvecs)
        assert_allclose(grid_itake.recivecs, self.grid.recivecs)
        assert_allclose(grid_itake.spacings, self.grid.spacings)
        assert isinstance(grid_itake, PeriodicGrid)
        # test take with boolean mask
        mask = np.zeros(self.grid.size, dtype=bool)
        mask[1] = True
        mask[3] = True
        mask[5] = True
        grid_mtake = self.grid[mask]
        assert_allclose(grid_mtake.points, grid_itake.points)
        assert_allclose(grid_mtake.weights, grid_itake.weights)
        assert_allclose(grid_mtake.realvecs, self.grid.realvecs)
        assert_allclose(grid_mtake.recivecs, self.grid.recivecs)
        assert_allclose(grid_mtake.spacings, self.grid.spacings)
        assert isinstance(grid_mtake, PeriodicGrid)

    def test_get_subgrid_small_radius(self):
        """Basic checks for the get_subgrid method.

        In this unit test, the cutoff sphere fits inside a primitive cell, such
        that each grid point from the parent periodic grid will at most appear
        once in the subgrid.
        """
        center = self.grid.points[3]
        radius = 0.19475
        # Check that the sphere fits inside the primitive cell:
        assert (2 * radius < self.grid.spacings).all()
        # Build subgrids.
        subgrids = [
            self.grid.get_subgrid(center, radius),
            self.wrapped_grid.get_subgrid(center, radius),
        ]
        # When there are no lattice vectors, the subgrid from the base class
        # should also be the same.
        if self._ref_realvecs is None:
            aperiodic_grid = Grid(self._ref_points, self._ref_weights)
            subgrids.append(aperiodic_grid.get_subgrid(center, radius))
        # One should get the same local grid with or without wrapping, possibly
        # with a different ordering of the points. We can perform a relatively
        # simple check here because each point appears at most once.
        order0 = subgrids[0].indices.argsort()
        for subgrid in subgrids[1:]:
            assert_allclose(subgrids[0].center, subgrid.center)
            order = subgrid.indices.argsort()
            assert_allclose(subgrids[0].points[order0], subgrid.points[order])
            assert_allclose(subgrids[0].weights[order0], subgrid.weights[order])
        # Other sanity checks on the grids.
        for subgrid in subgrids:
            # Just make sure we are testing with an actual subgrid with at least
            # some points.
            assert subgrid.size > 0
            assert subgrid.size <= self.grid.size
            assert len(subgrid.indices) == len(set(subgrid.indices))
            # Test that the subgrid contains sensible results.
            assert_allclose(subgrid.center, center)
            assert subgrid.points.ndim == self.grid.points.ndim
            assert subgrid.weights.ndim == self.grid.weights.ndim
            if self._ref_points.ndim == 2:
                assert (np.linalg.norm(subgrid.points - center, axis=1) <= radius).all()
            else:
                assert (abs(subgrid.points - center) <= radius).all()

    def test_get_subgrid_large_radius(self):
        """Basic checks for the get_subgrid method.

        In this unit test, the cutoff sphere fits inside a primitive cell, such
        that each grid point from the parent periodic grid will at most appear
        once in the subgrid.
        """
        center = self.grid.points[3]
        radius = 2.51235
        # Check that the sphere flows over the primitive cell, when there are
        # some lattice vectors.
        if self._ref_realvecs is not None:
            assert (2 * radius > self.grid.spacings).any()
        # Build subgrids.
        subgrids = [
            self.grid.get_subgrid(center, radius),
            self.wrapped_grid.get_subgrid(center, radius),
        ]
        # When there are no lattice vectors, the subgrid from the base class
        # should also be the same.
        if self._ref_realvecs is None:
            aperiodic_grid = Grid(self._ref_points, self._ref_weights)
            subgrids.append(aperiodic_grid.get_subgrid(center, radius))
        # One should get the same local grid with or without wrapping, possibly
        # with a different ordering of the points.
        for subgrid in subgrids[1:]:
            assert_equal_subgrids(subgrids[0], subgrid)
        # Other sanity checks.
        for subgrid in subgrids:
            # With a large radius there will never be less points in the subgrid.
            if self._ref_realvecs is None:
                assert subgrid.size == self.grid.size
            else:
                assert subgrid.size > self.grid.size
            # Test that the subgrid contains sensible results.
            assert_allclose(subgrid.center, center)
            assert subgrid.points.ndim == self.grid.points.ndim
            assert subgrid.weights.ndim == self.grid.weights.ndim
            if self._ref_points.ndim == 2:
                assert (np.linalg.norm(subgrid.points - center, axis=1) <= radius).all()
            else:
                assert (abs(subgrid.points - center) <= radius).all()

    def test_exceptions(self):
        """Check exceptions, the new ones specific to PeriodicGrid."""
        # init
        with pytest.raises(ValueError):
            PeriodicGrid(self._ref_points, self._ref_weights, np.ones((1, 100)))
        with pytest.raises(ValueError):
            if self._ref_points.ndim == 1:
                PeriodicGrid(self._ref_points, self._ref_weights, np.ones(100))
            else:
                PeriodicGrid(
                    self._ref_points,
                    self._ref_weights,
                    np.ones((100, self._ref_points.shape[1])),
                )
        if self._ref_realvecs is not None and self._ref_realvecs.shape[0] > 1:
            with pytest.raises(ValueError):
                realvecs = self._ref_realvecs.copy()
                realvecs[0] = realvecs[1]
                PeriodicGrid(self._ref_points, self._ref_weights, realvecs)
        # get_subgrid
        with pytest.raises(ValueError):
            self.grid.get_subgrid(np.zeros(100), 3.0)
        with pytest.raises(ValueError):
            self.grid.get_subgrid(self.grid.points[4], np.inf)
        with pytest.raises(ValueError):
            self.grid.get_subgrid(self.grid.points[4], -np.nan)
        with pytest.raises(ValueError):
            self.grid.get_subgrid(self.grid.points[4], -1.0)


class TestPeriodicGrid1D0CV(PeriodicGridTester):
    """PeriodicGrid testcase class for 1D points without periodic boundary."""

    def define_reference_data(self):
        """Define reference data for the test."""
        self._ref_points = np.linspace(-1, 1, 21).reshape(-1, 1)
        self._ref_weights = np.linspace(0.1, 0.2, 21)
        self._ref_realvecs = None
        self._ref_recivecs = np.zeros((0, 1))
        self._ref_spacings = np.array([])
        self._ref_frac_intvls = np.zeros((0, 2))


class TestPeriodicGrid1D1CV(PeriodicGridTester):
    """PeriodicGrid testcase class for 1D points without periodic boundary."""

    def define_reference_data(self):
        """Define reference data for the test."""
        self._ref_points = np.linspace(-1, 1, 21)
        self._ref_weights = np.linspace(0.1, 0.2, 21)
        self._ref_realvecs = np.array([0.8])
        self._ref_recivecs = np.array([1.25])
        self._ref_spacings = np.array([0.8])
        self._ref_frac_intvls = np.array([[-1.25, 1.25]])


class TestPeriodicGrid1D1CVBist(PeriodicGridTester):
    """PeriodicGrid testcase class for 1D points with periodic boundary."""

    def define_reference_data(self):
        """Define reference data for the test."""
        self._ref_points = np.linspace(-1, 1, 21).reshape(-1, 1)
        self._ref_weights = np.linspace(0.1, 0.2, 21)
        self._ref_realvecs = np.array([[0.8]])
        self._ref_recivecs = np.array([[1.25]])
        self._ref_spacings = np.array([0.8])
        self._ref_frac_intvls = np.array([[-1.25, 1.25]])


class TestPeriodicGrid2D0CV(PeriodicGridTester):
    """PeriodicGrid testcase class for 2D points without periodic boundaries."""

    def define_reference_data(self):
        """Define reference data for the test."""
        self._ref_points = np.array([np.linspace(-1, 1, 21)] * 2).T
        self._ref_weights = np.linspace(0.1, 0.2, 21)
        self._ref_realvecs = None
        self._ref_recivecs = np.zeros((0, 2))
        self._ref_spacings = np.array([])
        self._ref_frac_intvls = np.zeros((0, 2))


class TestPeriodicGrid2D1CV(PeriodicGridTester):
    """PeriodicGrid testcase class for 2D points with 1 cell vector."""

    def define_reference_data(self):
        """Define reference data for the test."""
        self._ref_points = np.array([np.linspace(-1, 1, 21)] * 2).T
        self._ref_weights = np.linspace(0.1, 0.2, 21)
        self._ref_realvecs = np.array([[0.8, 0.0]])
        self._ref_recivecs = np.array([[1.25, 0.0]])
        self._ref_spacings = np.array([0.8])
        self._ref_frac_intvls = np.array([[-1.25, 1.25]])


class TestPeriodicGrid2D2CV(PeriodicGridTester):
    """PeriodicGrid testcase class for 2D points with 2 cell vectors."""

    def define_reference_data(self):
        """Define reference data for the test."""
        self._ref_points = np.array([np.linspace(-1, 1, 21)] * 2).T
        self._ref_weights = np.linspace(0.1, 0.2, 21)
        self._ref_realvecs = np.array([[0.8, 0.0], [0.0, 0.5]])
        self._ref_recivecs = np.array([[1.25, 0.0], [0.0, 2.0]])
        self._ref_spacings = np.array([0.8, 0.5])
        self._ref_frac_intvls = np.array([[-1.25, 1.25], [-2.0, 2.0]])


class TestPeriodicGrid2D2CVBis(PeriodicGridTester):
    """PeriodicGrid testcase class for 2D points with 2 cell vectors, bis."""

    def define_reference_data(self):
        """Define reference data for the test."""
        self._ref_points = np.array([np.linspace(-1, 1, 21)] * 2).T
        self._ref_weights = np.linspace(0.1, 0.2, 21)
        self._ref_realvecs = np.array([[0.3, 0.4], [1.0, 0.0]])
        self._ref_recivecs = np.array([[0.0, 2.5], [1.0, -0.75]])
        self._ref_spacings = np.array([0.4, 0.8])
        self._ref_frac_intvls = np.array([[-2.5, 2.5], [-0.25, 0.25]])


class TestPeriodicGrid3D2CV(PeriodicGridTester):
    """PeriodicGrid testcase class for 3D points with 2 cell vectors."""

    def define_reference_data(self):
        """Define reference data for the test."""
        self._ref_points = np.array([np.linspace(-1, 1, 21)] * 3).T
        self._ref_weights = np.linspace(0.1, 0.2, 21)
        self._ref_realvecs = np.array([[0.7, 0.1, -0.2], [-0.3, 0.2, 0.6]])
        self._ref_recivecs = np.linalg.pinv(self._ref_realvecs).T
        self._ref_spacings = 1 / np.sqrt((self._ref_recivecs ** 2).sum(axis=1))
        frac_edges = np.dot(self._ref_points[[0, -1]], self._ref_recivecs.T)
        self._ref_frac_intvls = np.array(
            [frac_edges.min(axis=0), frac_edges.max(axis=0)]
        ).T

    def test_init_grid(self):
        """Test the __init__ method."""
        super().test_init_grid()
        # In addition, test that the real-space lattice vectors lie in the same
        # plane as the reciprocal-space lattice cell vectors.
        ortho = np.cross(self.grid.realvecs[0], self.grid.realvecs[1])
        assert_allclose(np.dot(self.grid.recivecs[0], ortho), 0.0, atol=1e-10)
        assert_allclose(np.dot(self.grid.recivecs[1], ortho), 0.0, atol=1e-10)
