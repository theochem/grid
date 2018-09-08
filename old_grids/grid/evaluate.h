// OLDGRIDS: Helpful Open-source Research TOol for N-fermion systems.
// Copyright (C) 2011-2017 The OLDGRIDS Development Team
//
// This file is part of OLDGRIDS.
//
// OLDGRIDS is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// OLDGRIDS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>
//
//--

// UPDATELIBDOCTITLE: Evaluation of splines on grids

#ifndef OLDGRIDS_GRID_EVALUATE_H
#define OLDGRIDS_GRID_EVALUATE_H


#include "old_grids/cell.h"
#include "old_grids/grid/cubic_spline.h"
#include "old_grids/grid/uniform.h"


void eval_spline_grid(CubicSpline* spline, double* center, double* output,
                      double* points, Cell* cell, long npoint);

void eval_decomposition_grid(CubicSpline** splines, double* center,
                             double* output, double* points, Cell* cell,
                             long nspline, long npoint);
#endif
