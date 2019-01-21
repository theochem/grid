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
cimport cubic_spline

cdef extern from "grid/grid/evaluate.h":
    void eval_spline_grid(cubic_spline.CubicSpline* spline, double* center,
        double* output, double* points, grid.cell.Cell* cell,
        long npoint)

    void eval_decomposition_grid(cubic_spline.CubicSpline** splines,
        double* center, double* output, double* points, grid.cell.Cell* cell,
        long nspline, long npoint)
