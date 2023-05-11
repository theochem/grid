
One-Dimensional Grids
======================

There are various choices of one dimensional grid for integrating functions over some finite or infinite
intervals.

.. csv-table:: One-Dimensional Quadrature Grids With Explicit Solutions
   :file: ./table_onedgrids.csv
   :widths: 31,70,70,70
   :delim: ;
   :header-rows: 1
   :align: center

Grid also includes popular quadrature grids that can integrate polynomials up to
degree :math:`2n - 1` on :math:`[-1, 1]`: :func:`Gauss-Laguerre<grid.onedgrid.GaussLaguerre>` (domain :math:`[0, \infty)`),
:func:`Gauss-Legendre<grid.onedgrid.GaussLegendre>`, and :func:`Gauss-Chebyshev<grid.onedgrid.GaussChebyshev>`.

FInally, the Trefethen grid class on :math:`[-1, 1]` is also included TODO: Include information about this.

- :func:`Trefethen (Clenshaw-Curtis)<grid.onedgrid.TrefethenCC>`
- :func:`Trefethen (Gauss-Chebyshev)<grid.onedgrid.TrefethenGC2>`
- :func:`Trefethen (General)<grid.onedgrid.TrefethenGeneral>`
- :func:`Trefethen Strip (Clenshaw-Curtis)<grid.onedgrid.TrefethenStripCC>`
- :func:`Trefethen Strip (Gauss-Chebyshev)<grid.onedgrid.TrefethenStripGC2>`
- :func:`Trefethen Strip (General)<grid.onedgrid.TrefethenStripGeneral>`



