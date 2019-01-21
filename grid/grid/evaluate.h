// GRID is a numerical integration library for quantum chemistry.
//
// Copyright (C) 2011-2017 The GRID Development Team
//
// This file is part of GRID.
//
// GRID is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// GRID is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>
//
//--

// UPDATELIBDOCTITLE: Evaluation of splines on grids

#ifndef GRID_GRID_EVALUATE_H
#define GRID_GRID_EVALUATE_H


#include "grid/cell.h"
#include "grid/grid/cubic_spline.h"
#include "grid/grid/uniform.h"


void eval_spline_grid(CubicSpline* spline, double* center, double* output,
                      double* points, Cell* cell, long npoint);

void eval_decomposition_grid(CubicSpline** splines, double* center,
                             double* output, double* points, Cell* cell,
                             long nspline, long npoint);
#endif
