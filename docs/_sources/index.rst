Welcome to grid's documentation!
================================

Grid is a free and open-source Python library for numerical integration,
interpolation and differentiation of interest for the quantum chemistry community.
Grid is intended to support molecular density-functional (DFT) theory calculations,
but it also periodic boundary conditions.
<<<<<<< HEAD

Please use the following citation in any publication using grid library:

    **"Grid: A Python Library for Molecular Integration, Interpolation and More."**,
=======
  
Please use the following citation in any publication using grid library:

    **"Grid: A Python Library for Molecular Integration, Interpolation and More."**, 
>>>>>>> 32d8e9f8cd2fea5028d677bd401d7ca7dbc35a4d
    X. D. Yang,  L. Pujal, A. Tehrani,  R. Hernandez‐Esparza,  E. Vohringer‐Martinez,
    T. Verstraelen, P. W. Ayers, F. Heidar‐Zadeh


The Grid source code is hosted on GitHub and is released under the GNU General Public License v3.0. We welcome
any contributions to the Grid library in accordance with our Code of Conduct; please see our Contributing
Guidelines. Please report any issues you encounter while using Grid library on
`GitHub Issues <https://github.com/theochem/grid/issues/new>`_. For further
information and inquiries please contact us at qcdevs@gmail.com.

<<<<<<< HEAD



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
=======
Installation
============

Grid can be cloned via git,

.. code-block:: bash

   git clone https://github.com/theochem/grid.git

.. _usr_py_depend:

The following dependencies will be necessary for installation of Grid,

* Python >= 3.6: `https://www.python.org/ <http://www.python.org/>`_
* SciPy >= 1.4: `https://www.scipy.org/ <http://www.scipy.org/>`_
* NumPy >= 1.16: `https://www.numpy.org/ <http://www.numpy.org/>`_
* Pip >= 19.0: `https://pip.pypa.io/ <https://pip.pypa.io/>`_
* PyTest >= 2.6: `https://docs.pytest.org/ <https://docs.pytest.org/>`_
* Sympy >= : `https://www.sympy.org/en/index.html <https://www.sympy.org/en/index.html>`_
* Sphinx >= 2.3.0, if one wishes to build the documentation locally:
  `https://www.sphinx-doc.org/ <https://www.sphinx-doc.org/>`_


Then installation via pip can be done by going into the directory where Grid is downloaded to and running,

.. code-block:: bash

    pip install .

Successful installation can be checked by running the tests,

.. code-block:: bash

    pytest --pyargs grid

Details
========

.. csv-table:: Various Modules of Grid
>>>>>>> 32d8e9f8cd2fea5028d677bd401d7ca7dbc35a4d
    :file: ./table_modules.csv
    :widths: 50, 70
    :delim: ;
    :header-rows: 1
    :align: center


.. toctree::
    :maxdepth: 1
<<<<<<< HEAD
    :caption: User Documentation

    ./installation.rst
    ./onedgrids.rst
    ./radial_transf.rst



.. toctree::
    :maxdepth: 1
=======
>>>>>>> 32d8e9f8cd2fea5028d677bd401d7ca7dbc35a4d
    :caption: Example Tutorials

    ./notebooks/Molecular_Integration.ipynb
    ./notebooks/Interpolation_and_Poisson.ipynb
    ./notebooks/Cubic_Grids.ipynb
    ./notebooks/Multipole_Moments.ipynb


<<<<<<< HEAD
.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   pyapi/modules

=======
One-dimensional grids
---------------------

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


These one-dimensional grids can be transformed to other kinds of intervals using the
:func:`radial transform <grid.rtransform>` module.


.. csv-table:: Radial Transformation
   :file: ./table_rtransform.csv
   :widths: 31,70,70,70
   :delim: ;
   :header-rows: 1
   :align: center


.. toctree::
   :maxdepth: 4
   :caption: Contents:


Literature
^^^^^^^^^^

.. [#becke1988_multicenter] “A multicenter numerical integration scheme for polyatomic molecules”,
       Becke, A. D.; J. Chem. Phys. 1988 (v. 88 pp. 2547–2553); http://dx.doi.org/10.1063/1.454033

API Reference
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2
   :caption: Grid Modules

   pyapi/modules
  
 
>>>>>>> 32d8e9f8cd2fea5028d677bd401d7ca7dbc35a4d
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
