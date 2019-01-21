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
"""Molecular integration grids"""
import logging

import numpy as np

from grid.periodic import periodic
from grid.utils import typecheck_geo, doc_inherit

from grid.grid.atgrid import AtomicGrid, AtomicGridSpec
from grid.grid.base import IntGrid
from grid.grid.cext import becke_helper_atom

__all__ = [
    'BeckeMolGrid'
]


class BeckeMolGrid(IntGrid):
    """Molecular integration grid using Becke weights"""
    biblio = []

    def __init__(self, centers, numbers, pseudo_numbers=None, agspec='medium', k=3,
                 random_rotate=True, mode='discard'):
        """
           **Arguments:**

           centers
                An array (N, 3) with centers for the atom-centered grids.

           numbers
                An array (N,) with atomic numbers.

           **Optional arguments:**

           pseudo_numbers
                An array (N,) with effective core charges. When not given, this
                defaults to ``numbers``.

           agspec
                A specifications of the atomic grid. This can either be an
                instance of the AtomicGridSpec object, or the first argument
                of its constructor.

           k
                The order of the switching function in Becke's weighting scheme.

           random_rotate
                Flag to control random rotation of spherical grids.

           mode
                Select one of the following options regarding atomic subgrids:

                * ``'discard'`` (the default) means that all information about
                  subgrids gets discarded.

                * ``'keep'`` means that a list of subgrids is kept, including
                  the integration weights of the local grids.

                * ``'only'`` means that only the subgrids are constructed and
                  that the computation of the molecular integration weights
                  (based on the Becke partitioning) is skipped.
        """
        natom, centers, numbers, pseudo_numbers = typecheck_geo(centers, numbers, pseudo_numbers)
        self._centers = centers
        self._numbers = numbers
        self._pseudo_numbers = pseudo_numbers

        # check if the mode argument is valid
        if mode not in ['discard', 'keep', 'only']:
            raise ValueError('The mode argument must be \'discard\', \'keep\' or \'only\'.')

        # transform agspec into a usable format
        if not isinstance(agspec, AtomicGridSpec):
            agspec = AtomicGridSpec(agspec)
        self._agspec = agspec

        # assign attributes
        self._k = k
        self._random_rotate = random_rotate
        self._mode = mode

        # allocate memory for the grid
        size = sum(agspec.get_size(self.numbers[i], self.pseudo_numbers[i]) for i in range(natom))
        points = np.zeros((size, 3), float)
        weights = np.zeros(size, float)
        self._becke_weights = np.ones(size, float)

        # construct the atomic grids
        if mode != 'discard':
            atgrids = []
        else:
            atgrids = None
        offset = 0

        if mode != 'only':
            # More recent covalent radii are used than in the original work of Becke.
            # No covalent radius is defined for elements heavier than Curium and a
            # default value of 3.0 Bohr is used for heavier elements.
            cov_radii = np.array([(periodic[n].cov_radius or 3.0) for n in self.numbers])

        # The actual work:
        logging.info('Preparing Becke-Lebedev molecular integration grid.')
        for i in range(natom):
            atsize = agspec.get_size(self.numbers[i], self.pseudo_numbers[i])
            atgrid = AtomicGrid(
                self.numbers[i], self.pseudo_numbers[i],
                self.centers[i], agspec, random_rotate,
                points[offset:offset + atsize])
            if mode != 'only':
                atbecke_weights = self._becke_weights[offset:offset + atsize]
                becke_helper_atom(points[offset:offset + atsize], atbecke_weights, cov_radii,
                                  self.centers, i, self._k)
                weights[offset:offset + atsize] = atgrid.weights * atbecke_weights
            if mode != 'discard':
                atgrids.append(atgrid)
            offset += atsize

        # finish
        IntGrid.__init__(self, points, weights, atgrids)

        # Some screen info
        self._log_init()

    def _get_centers(self):
        """The positions of the nuclei"""
        return self._centers.view()

    centers = property(_get_centers)

    def _get_numbers(self):
        """The element numbers"""
        return self._numbers.view()

    numbers = property(_get_numbers)

    def _get_pseudo_numbers(self):
        """The effective core charges"""
        return self._pseudo_numbers.view()

    pseudo_numbers = property(_get_pseudo_numbers)

    def _get_agspec(self):
        """The specifications of the atomic grids."""
        return self._agspec

    agspec = property(_get_agspec)

    def _get_k(self):
        """The order of the Becke switching function."""
        return self._k

    k = property(_get_k)

    def _get_random_rotate(self):
        """The random rotation flag."""
        return self._random_rotate

    random_rotate = property(_get_random_rotate)

    def _get_mode(self):
        """The MO of this molecular grid"""
        return self._mode

    mode = property(_get_mode)

    def _get_becke_weights(self):
        """The becke weights of the grid points"""
        return self._becke_weights

    becke_weights = property(_get_becke_weights)

    def _log_init(self):
        logging.info('Initialized: %s' % self)
        logging.info([
            ('Size', self.size),
            ('Switching function', 'k=%i' % self._k),
        ])
        # Cite reference
        self.biblio.append(('becke1988_multicenter', 'the multicenter integration scheme used for the molecular integration grid'))
        self.biblio.append(('cordero2008', 'the covalent radii used for the Becke-Lebedev molecular integration grid'))

    @doc_inherit(IntGrid)
    def integrate(self, *args, **kwargs):
        if self.mode == 'only':
            raise NotImplementedError(
                'When mode==\'only\', only the subgrids can be used for integration.')
        return IntGrid.integrate(self, *args, **kwargs)
