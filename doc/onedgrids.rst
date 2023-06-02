
One-Dimensional Grids
======================

There are various choices of one dimensional grids for integrating functions over some finite or infinite
intervals.

.. csv-table:: One-Dimensional Quadrature Grids With Explicit Solutions
   :file: ./table_onedgrids.csv
   :widths: 31,70,70,70
   :delim: ;
   :header-rows: 1
   :align: center

Grid also includes popular quadrature grids that can integrate polynomials up to
degree :math:`2n - 1`:

- :func:`Gauss-Legendre<grid.onedgrid.GaussLegendre>` (domain :math:`[-1, 1]`),
- :func:`Gauss-Chebyshev<grid.onedgrid.GaussChebyshev>` (domain :math:`[-1, 1]`),
- :func:`Gauss-Laguerre<grid.onedgrid.GaussLaguerre>` (domain :math:`[0, \infty)`).

The Trefethen polynomial transformation of various grids on :math:`[-1, 1]` is also included:

- :func:`Trefethen (Clenshaw-Curtis)<grid.onedgrid.TrefethenCC>`,
- :func:`Trefethen (Gauss-Chebyshev)<grid.onedgrid.TrefethenGC2>`,
- :func:`Trefethen (General)<grid.onedgrid.TrefethenGeneral>`,
- :func:`Trefethen Strip (Clenshaw-Curtis)<grid.onedgrid.TrefethenStripCC>`,
- :func:`Trefethen Strip (Gauss-Chebyshev)<grid.onedgrid.TrefethenStripGC2>`,
- :func:`Trefethen Strip (General)<grid.onedgrid.TrefethenStripGeneral>`.
