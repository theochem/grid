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


cimport old_grids.cell
cimport cubic_spline

cdef extern from "old_grids/grid/evaluate.h":
    void eval_spline_grid(cubic_spline.CubicSpline* spline, double* center,
        double* output, double* points, old_grids.cell.Cell* cell,
        long npoint)

    void eval_decomposition_grid(cubic_spline.CubicSpline** splines,
        double* center, double* output, double* points, old_grids.cell.Cell* cell,
        long nspline, long npoint)
