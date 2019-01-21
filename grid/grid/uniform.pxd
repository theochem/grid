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


cimport grid.cell

cdef extern from "grid/grid/uniform.h":
    cdef cppclass UniformGrid:
        double origin[3]
        double grid_rvecs[9]
        long shape[3]
        long pbc[3]

        UniformGrid(double* _origin, double* _grid_rvecs, long* _shape, long* _pbc)

        grid.cell.Cell* get_cell()
        grid.cell.Cell* get_grid_cell()

        void set_ranges_rcut(double* center, double rcut, long* ranges_begin, long* ranges_end)
        double dist_grid_point(double* center, long* i)
        void delta_grid_point(double* center, long* i)

    long index_wrap(long i, long high)
