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
r"""Grids with periodic boundary conditions.

Algorithm
---------

To understand how the algorithm in ``get_subgrid`` works, it may be useful
review a few basic crystallographic concepts:

- https://en.wikipedia.org/wiki/Bravais_lattice
- https://en.wikipedia.org/wiki/Reciprocal_lattice
- https://en.wikipedia.org/wiki/Fractional_coordinates

The goal of the algorithm is to find all grid points of a periodically extended
grid, which lie in a cutoff sphere. The following ASCII art is an example of
such a problem in the 1D case:

.. code-block::

    lattice:       |          |          |          |          |          |
    periodic grid: |          |          |  1   2 3 |          |          |
    cutoff sphere:                           (             c             )

The first line represents the crystal planes of the lattice. The second line are
grid points in one primitive cell, with the lattice repeated for clarity. The
last line is the cutoff "sphere" with center ``c`` and the extent of the sphere
is shown by the parenthesis.

The conventional approach to build the subgrid considers all the relevant
periodic images of the grid points and retains only those that lie within the
cutoff sphere:

.. code-block::

    lattice:       |          |          |          |          |          |
    periodic grid: |          |          |  1   2 3 |          |          |
    image -2:      |  1   2 3 |          |          |          |          |
    image -1:      |          |  1   2 3 |          |          |          |
    image  0:      |          |          |  1   2 3 |          |          |
    image  1:      |          |          |          |  1   2 3 |          |
    image  2:      |          |          |          |          |  1   2 3 |
    cutoff sphere:                           (             c             )

For each periodic image, one can apply a nearest-neighbor lookup within the
sphere to find the relevant grid points. In this module, the cKDTree from SciPy
is used for this purpose.
See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html

Obviously, it is not necessary to consider an infinite number of periodic
images. In this example, only images 0, 1 and 2 need to be constructed, because
only these contain grid points that lie in the cutoff sphere.

After retaining only the grid points of the periodic images within the cutoff
sphere, one obtains a local grid suitable for integration or evaluation of a
function whose domain is limited (approximately) to the cutoff sphere:

.. code-block::

    lattice:       |          |          |          |          |          |
    periodic grid: |          |          |  1   2 3 |          |          |
    local grid:                                 2 3    1   2 3    1   2 3
    cutoff sphere:                           (             c             )

The algorithm in this module is slightly different. In order to build the cKDTree
instance only once for a fixed set of points, periodic images of the cutoff
sphere are constructed:

.. code-block::

    lattice:       |          |          |          |          |          |
    periodic grid: |          |          |  1   2 3 |          |          |
    c.s. image  0:                           (             c             )
    c.s. image -1:                (             c             )
    c.s. image -2:     (             c             )

In this example, only images 0, -1 and -2 of the cutoff sphere result in
non-zero "untranslated" grid points that lie within the translated sphere. For
each image, the found grid points in the periodic grid need to be translated
back, by the negative displacement of the image of the cutoff sphere, to find
the grid points in the final local grid. This translation is carried out after
the neighbor search and requires no change to the cKDTree object.

To implement the second algorithm, one needs to find all possibly relevant
displacements of the cutoff sphere. The left-most position of the center of the
cutoff sphere still containing a grid point, without worrying about periodic
boundary conditions, is:

.. math::

    c'_\text{left} = \min_i x_i - r

where :math:`x_i` is the position of grid point :math:`i` and :math:`r` is the
radius of the sphere. We only need to consider translations by multiples of the
lattice vector:

.. math::

    c_\text{left} = c + a \left\lceil \frac{\min_i x_i - c - r}{a} \right\rceil

where :math:`a` is the length of the 1D lattice vector (and also the spacing
between adjacent crystal planes). Similarly, for the right-most center, we
have:

    c_\text{right} = c + a \left\lfloor \frac{\max_i x_i - c + r}{a} \right\rfloor

The positions of all relevant centers are given by:

.. math::

    c_j = c + a j \quad \forall \, j \in \mathbb{Z} \quad \text{with} \quad
    \left\lceil \frac{\min_i x_i - c - r}{a} \right\rceil
    \le j \le
    \left\lfloor \frac{\max_i x_i - c + r}{a} \right\rfloor

Note that this algorithm also works when the points in the periodic grid are
not confined to one primitive cell (as long as no duplicates are present). This
situation will just result in more images of centers to be considered. While
this will just work, it implies more iterations at the Python level, which could
be potentially slow. To avoid such efficiency issues, the ``wrap`` option can be
set to True when creating a ``PeriodicGrid`` instance.

The generalization to higher-dimensional periodic grids is implemented with a
combination (a superposition) of all possible displacements of the center along
each lattice vector:

.. math::

    \vec{c}_{j_1, \ldots, j_K} = \vec{c} + \sum_{k=1}^K \vec{a}_k j
    \quad \forall \, j \in \mathbb{Z} \quad \text{with} \quad
    \left\lceil \min_i \tilde{x}_i - \tild{c} - \frac{r}{s_k} \right\rceil
    \le j \le
    \left\lfloor \max_i \tilde{x}_i - \tild{c} + \frac{r}{s_k} \right\rfloor

where :math:`\vec{a}_k` is lattice vector :math:`k`, :math:`\tilde{x}_i` is the
fractional coordinate of grid point :math:`i`, :math:`\tilde{c}` is the
fractional coordinate of the center and :math:`s_k` is the spacing between
adjacent crystal planes along the :math:`k`th lattice vector.
"""
import itertools
import warnings

from grid.basegrid import Grid, SubGrid

import numpy as np

from scipy.spatial import cKDTree


class PeriodicGridWarning(Warning):
    """Raised when the fractional coordinates span an interval wider than 1.1."""


class PeriodicGrid(Grid):
    """Grid with support for periodic boundary conditions.

    The purpose of this class is to support certain operations on integration
    grids with periodic boundary conditions. It does not construct such grids
    and it assumes the grid points and weights (and also the lattice vectors)
    of a single unit cell are provided when creating an instance of the class.

    The dimensionality of the grid points is not fixed but the most common use
    cases are 1D, 2D, 3D grids. The number of cell vectors can be anything from
    zero up to the dimensionality of the grid points. The cell vectors cannot
    form a singular matrix.

    The whole-grid integration works in exactly the same way as in the base
    class. Through the ``get_subgrid`` method, the following two operations can
    be carried out easily:

    1) The integration of a local function on a part of the periodic grid. The
    integrand may cover several unit cells and may contain contributions from a
    periodic function, as long as it decays sufficiently fast to zero near the
    cutoff radius.

    2) The addition of the periodic repetition of a local function to the whole
    grid.

    In both cases, one uses ``get_subgrid`` to construct an aperiodic grid,
    whose grid points lie within a given (hyper)sphere enclosing the local
    function of interest. This grid is a periodic repetition of the
    single-unit-cell grid of which only the points within the (hyper)sphere are
    retained. On this aperiodic grid, one can carry out operations as usual. The
    attribute ``subgrid.indices`` can be used for two purposes:

    1) To transfer a periodic function on the parent grid to the subgrid, one
    simply uses NumPy slicing: ``fnsub = fnperiodic[subgrid.indices]``

    2) To add a periodic repetition of a local function to the parent grid, one
    uses ``np.add.att(fnperiodic, subgrid.indices, fnsub)``. One should not
    use ``fnperiodic[subgrid.indices] += fnsub``, because this will give wrong
    results when ``subgrid.indices`` contains the same index multiple times,
    which happens when the cutoff (hyper)sphere covers multiple primitive cells.
    See https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.at.html

    The efficiency of the ``get_subgrid`` method will be optimal when all grid
    points fall within one primitive cell, e.g. the fractional coordinates
    should lie in [0, 1[ or [-0.5, 0.5[. The algorithm will also work when the
    grid points of the primitive cell accidentally overflow into neighboring
    images, but in this case it becomes less efficient.
    """

    def __init__(self, points, weights, realvecs=None, wrap=False):
        """Construct a PeriodicGrid instance.

        Parameters
        ----------
        points : np.ndarray(N,) or np.ndarray(N, M)
            An array with N positions of M-dimensional grid points.
        weights : np.ndarray(N,)
            An array of weights associated with each point on the grid.
        realvecs : np.ndarray(1,) or np.ndarray(Nlv, M,), optional
            ``Nlv`` Lattice vectors in real space defining the periodic boundary
            conditions. Each row contains one vector. These are sometimes also
            called period vectors or cell vectors. When not given, this class
            behaves identically to the Grid base class.
        wrap: boolean
            When True, the points are wrapped back into the primitive cell, by
            adding the appropriate integer linear combination of real-space
            lattice vectors to each point. (This correction is stored in a copy
            of the points array provided by the user.) As a result, all points
            will have fractional coordinates in the interval [0, 1[. When False,
            a warning is raised when the fractional coordinates span an interval
            wider than 1.1, because this implies a degradation of efficiency of
            ``get_subgrid``, which can be easily avoided.

        Raises
        ------
        ValueError
            When the shape of points, weights and realvecs are inconsistent, or
            when the lattice vectors are singular.
        PeriodicGridWarning
            When ``wrap==False`` and the fractional coordinates span an interval
            wider than 1.1.
        """
        if realvecs is None:
            realvecs = np.zeros((0,) + points.shape[1:])
        if points.ndim != realvecs.ndim:
            raise ValueError(
                "Arrays points and realvecs must have the same number of dimensions. \n"
                f"points.ndim={points.ndim} and realvecs.ndim={realvecs.ndim}"
            )
        if points.shape[1:] != realvecs.shape[1:]:
            raise ValueError(
                "Arguments points and realvecs do not have the same number of columns. \n"
                f"points.shape={points.shape} and realvecs.shape={realvecs.shape}"
            )
        ncellvec = 1 if realvecs.ndim == 1 else realvecs.shape[0]
        npointdim = 1 if points.ndim == 1 else points.shape[1]
        if ncellvec > npointdim:
            raise ValueError(
                "There cannot be more lattice vectors than the dimensionality of the points. \n"
                f"ncellvec={ncellvec} and npointdim={npointdim}"
            )
        self._realvecs = realvecs
        # Compute the reciprocal space lattice vectors: recivecs.
        if realvecs.size == 0:
            recivecs = np.zeros(realvecs.shape)
        elif realvecs.ndim == 1:
            recivecs = 1 / realvecs
        else:
            # SVD is used to construct the pseudo-inverse and to check if the
            # lattice vectors are not singular.
            rcond = np.finfo(realvecs.dtype).eps * max(realvecs.shape)
            U, S, Vt = np.linalg.svd(realvecs, full_matrices=False)
            if abs(S).max() * rcond > abs(S).min():
                raise ValueError("The cell vectors are singular.")
            recivecs = np.einsum("ij,j,jk", U, 1 / S, Vt)
        assert recivecs.shape == realvecs.shape
        self._recivecs = recivecs
        # Compute the spacings between the crystal planes. For example, the in
        # the 3D case, these are the distances between two adjacent planes
        # with Miller indices {100}, {010} and {001}, respectively.
        if points.ndim == 1:
            spacings = 1 / self._recivecs
        else:
            spacings = 1 / np.linalg.norm(self._recivecs, axis=1)
        self._spacings = spacings
        # Compute the fractional coordinates, which are only used temporarily.
        # They are not stored as an attribute.
        if points.ndim == 1:
            frac_points = points * recivecs
        else:
            frac_points = points @ recivecs.T
        # Wrap the points back into the primitive cell, in case this was asked.
        # This will not change the array given by the user.
        if wrap and realvecs.size > 0:
            frac_shift = -np.floor(frac_points)
            frac_points += frac_shift
            if points.ndim == 1:
                points = points + frac_shift * realvecs
            else:
                points = points + frac_shift @ realvecs
        # Compute the minimal and maximal values of the fractional coordinates.
        # These are the intervals spanned by the fractional coordinates along
        # each lattice vector: ``frac_intvls``.
        if points.ndim == 1:
            frac_intvls = np.array([[frac_points.min(), frac_points.max()]])
        else:
            frac_intvls = np.array([frac_points.min(axis=0), frac_points.max(axis=0)]).T
        self._frac_intvls = frac_intvls
        # If the difference between maximal and minimal fractional coordinates
        # exceeds 1.1, raise a warning. This will never happen when wrap==True.
        if len(frac_intvls) > 0:
            intvl_max = (frac_intvls[:, 1] - frac_intvls[:, 0]).max()
            if intvl_max > 1.1:
                warnings.warn(
                    f"Interval spanned by fractional coordinates: {intvl_max} > 1.1. \n"
                    " ``get_subgrid`` will be inefficient.",
                    PeriodicGridWarning,
                )
        # Call the constructor of the base class
        super().__init__(points, weights)

    @property
    def realvecs(self):
        """np.ndarray(N,) or np.ndarray(N, M): Real-space lattice vectors."""
        return self._realvecs

    @property
    def recivecs(self):
        """np.ndarray(N,) or np.ndarray(N, M): Reciprocal-space lattice vectors."""
        return self._recivecs

    @property
    def frac_intvls(self):
        """np.ndarray(Ncv, 2): Intervals containing the grid points in fractional coordinates."""
        return self._frac_intvls

    @property
    def spacings(self):
        """np.ndarray(Ncv, 2): Spacings between adjacent primitive crystal planes."""
        return self._spacings

    def __getitem__(self, index):
        """Return a part of the grid.

        Parameters
        ----------
        index : int, slice or array of integers or booleans
            Selection of the grid points to retain in the returned grid.

        Returns
        -------
        PeriodicGrid
            A new PeriodicGrid object with selected points.
        """
        if isinstance(index, int):
            return self.__class__(
                np.array([self.points[index]]),
                np.array([self.weights[index]]),
                self.realvecs,
            )
        else:
            return self.__class__(
                np.array(self.points[index]),
                np.array(self.weights[index]),
                self.realvecs,
            )

    def get_subgrid(self, center, radius):
        """Create a non-peroidic subgrid within the given distance from the center.

        Parameters
        ----------
        center : float or np.array(M,)
            Cartesian coordinates of the center of the subgrid.
        radius : float
            Radius of sphere around the center.

        Returns
        -------
        SubGrid
            Instance of SubGrid.
        """
        # A) Check arguments
        # ------------------
        center = np.asarray(center)
        if center.shape != self._points.shape[1:]:
            raise ValueError(
                "Argument center has the wrong shape \n"
                f"center.shape: {center.shape}, points.shape: {self._points.shape}"
            )
        if not np.isfinite(radius):
            raise ValueError(f"Invalid radius: {radius}")
        if radius < 0:
            raise ValueError(f"Negative radius: {radius}")

        # B) Construct all displacements of the center
        # --------------------------------------------
        # To find all periodic images of the grid points within the cutoff
        # sphere, the center is displaced by integer linear combinations of
        # lattice vectors, instead of translating the grid points. This choice
        # makes it possible to reuse the same cKDTree instance for all
        # translations. The code below constructs all possible displacements
        # of the center to be considered. Keep in mind that the center needs
        # to be displaced in the opposite direction than the grid points would
        # have been replaced, to obtain the same relative vectors.

        # Center in fractional coordinates
        if self._points.ndim == 1:
            frac_center = self._recivecs * center
        else:
            frac_center = self._recivecs @ center
        # Minimal and maximal values of the integer coefficients used in
        # to construct all relevant integer linear combinations (``ilc``) of
        # lattice vectors to translate the center.
        ilc_min = np.ceil(
            self._frac_intvls[:, 0] - frac_center - radius / self._spacings
        ).astype(int)
        ilc_max = np.floor(
            self._frac_intvls[:, 1] - frac_center + radius / self._spacings
        ).astype(int)
        assert (ilc_min <= ilc_max).all()

        # C) Loop over all possible translations of the center
        # ----------------------------------------------------
        # When points.ndim == 1, we have to reshape points and center to
        # make the input compatible with cKDTree
        if self._kdtree is None:
            self._kdtree = cKDTree(self._points.reshape(self.size, -1))
        sub_points = []
        sub_weights = []
        sub_indices = []
        # The following constructs an iterator over all possibly relevant
        # integer linear combinations of lattice vectors.
        ilc_iterator = itertools.product(
            *[range(imin, imax + 1) for imin, imax in zip(ilc_min, ilc_max)]
        )
        for ilc in ilc_iterator:
            delta = ilc @ self._realvecs
            # Translate center instead of translating the grid points
            displaced_center = center + delta
            # Fix the type of the center, to make cKDTree happy.
            _displaced_center = (
                np.array([displaced_center]) if center.ndim == 0 else displaced_center
            )
            indices = np.array(
                self._kdtree.query_ball_point(_displaced_center, radius, p=2.0)
            )
            # The following line avoids some_array[indices] when indices == [].
            if len(indices) == 0:
                continue
            sub_indices.append(indices)
            sub_weights.append(self._weights[indices])
            # Store points with the opposite displacement!!
            sub_points.append(self._points[indices] - delta)

        return SubGrid(
            np.concatenate(sub_points),
            np.concatenate(sub_weights),
            center,
            np.concatenate(sub_indices),
        )
