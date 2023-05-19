Welcome to grid's documentation!
================================

Grid is a free and open-source Python library for numerical integration,
interpolation and differentiation of interest for the quantum chemistry community.
Grid is intended to support molecular density-functional (DFT) theory calculations,
but it also periodic boundary conditions.

Please use the following citation in any publication using grid library:

    **"Grid: A Python Library for Molecular Integration, Interpolation and More."**,
    X. D. Yang,  L. Pujal, A. Tehrani,  R. Hernandez‐Esparza,  E. Vohringer‐Martinez,
    T. Verstraelen, P. W. Ayers, F. Heidar‐Zadeh


The Grid source code is hosted on GitHub and is released under the GNU General Public License v3.0. We welcome
any contributions to the Grid library in accordance with our Code of Conduct; please see our Contributing
Guidelines. Please report any issues you encounter while using Grid library on
`GitHub Issues <https://github.com/theochem/grid/issues/new>`_. For further
information and inquiries please contact us at qcdevs@gmail.com.


Functionality
=============
* One-dimensional integration

  * quadratures

  * transformations

* Multivariate integration

  * Hypercubes:

    * Tensor product of one-dimensional quadratures

    * Rectilinear (a.k.a uniform) and skewed grids

  * Integration over sphere:

    * Lebedev-Laikov grids

    * Symmetric Spherical T-Designs

  * Molecular integration:

    * Atom in molecule weights (nuclear weight functions)

      * Hirshfeld weights

      * Becke weights

* Interpolation & differentiation

* General ODE solver and Poisson solver


Modules
=======

.. csv-table::
    :file: ./table_modules.csv
    :widths: 50, 70
    :delim: ;
    :header-rows: 1
    :align: center


.. toctree::
    :maxdepth: 2
    :caption: User Documentation

    ./installation.rst
    ./onedgrids.rst
    ./radial_transf.rst
    ./conventions.rst



.. toctree::
    :maxdepth: 1
    :caption: Example Tutorials

    ./notebooks/Molecular_Grid.ipynb
    ./notebooks/Molecular_Grid_Construction.ipynb
    ./notebooks/Cubic_Grids.ipynb
    ./notebooks/Interpolation_and_Poisson.ipynb
    ./notebooks/Multipole_Moments.ipynb


.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   pyapi/modules.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
