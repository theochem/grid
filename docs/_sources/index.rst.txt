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
    :file: ./table_modules.csv
    :widths: 50, 70
    :delim: ;
    :header-rows: 1
    :align: center


Integration In Three-Dimensions
-------------------------------

The main goal of Grid package is to integrate functions whose values locally concentrate
at various centers :math:`\{A_k\}` and tend to decay outwards from them.  As proposed by Becke [#becke1988_multicenter]_,
this is done as

.. math::
    \int_{\mathbb{R}^3} f(\vec{\textbf{r}}) d\vec{\textbf{r}} = \sum_{A} \int_{\mathbb{R}^3}
    w_A(\vec{\textbf{r}}) f(\vec{\textbf{r}}) d\vec{\textbf{r}},

where :math:`w_A` is known as the nuclear weight function (or atom in molecule weights) such that
it has value one close to  the center and decay's continuous over other centers with the condition that
:math:`\sum_{A} w_A(\vec{\textbf{r}}) = 1` for all :math:`\vec{\textbf{r}}`.  The module
:func:`Molecular Grids<grid.molgrid>` is responsible for computing integrals in this fashion with
either :func:`Becke<grid.becke>` or :func:`Hirshfeld<grid.hirshfeld>` atom in molecule weights.

The process to integrate each individual integral is done by first converting to spherical coordinates
:math:`(r, \theta, \phi)`:

.. math::
    \int_{\mathbb{R}^3} w_A(\vec{\textbf{r}}) f(\vec{\textbf{r}}) d\vec{\textbf{r}} =
    \int \int \int  w_A(r, \theta, \phi) f(r, \theta, \phi) r^2 \sin(\theta) dr d\theta d\phi


Then a one-dimensional grid is chosen over the radial grid (see :func:`radial <grid.onedgrid>` and
:func:`radial transform <grid.rtransform>` module) with weights :math:`w_i^{rad}` and an angular grid
with weights :math:`w_j^{ang}` is  chosen to integrate over the angles :math:`(\theta, \phi)`
including the sin factor. The combination of the two is handled by the :func:`atomic grid<grid.atomgrid>` module
with weights :math:`w_{ij} = w_i^{rad} w_j^{ang} r^2` to achieve the numerical
integral

.. math::
    \int \int \int  w_A(r, \theta, \phi) f(r, \theta, \phi) r^2 \sin(\theta) dr d\theta d\phi \approx
    \sum_{i=1}^{N_{rad}} \sum_{j=1}^{N_{ang}} w_{ij}(r, \theta, \phi)
    w_A(r, \theta, \phi) f(r, \theta, \phi).

For any general function :math:`f: \mathbb{R}^3\rightarrow \mathbb{R}`, Grid package offers various grids in
:func:`Cubic<grid.cubic>` class for constructing hyper-rectangular grids. If :math:`f` is periodic,
then the :func:`periodic<grid.periodicgrid>` module is useful.


Furthermore, the multipole moments can be computed, see :func:`moment function<grid.basegrid.Grid.moments>`.
For example, Cartesian type moments are
:math:`m_{n_x, n_y, n_z} = \int (x - X_c)^{n_x} (y - Y_c)^{n_y} (z - Z_c)^{n_z} f(r) dr`,
where :math:`\textbf{R}_c = (X_c, Y_c, Z_c)` is the center of the moment,
:math:`f(r)` is the density, and :math:`(n_x, n_y, n_z)` are the Cartesian orders.


Interpolation
-------------

It is known based on the properties of the real spherical harmonics :math:`Y_{lm}`
(see :func:`definition <grid.utils.generate_real_spherical_harmonics>` ) that any :math:`L_2` function
:math:`f : \mathbb{R}^3 \rightarrow \mathbb{R}` can be decomposed as

.. math::
    f(r, \theta, \phi) = \sum_{l=0}^\infty \sum_{m=-l}^l w_{lm}(r) Y_{lm}(\theta, \phi),

where the unknown :math:`w_{lm}` is a function of the radial component and is found as follows.
For a fixed :math:`r`, the radial component :math:`w_{lm}` is computed as the integral
:math:`w_{lm}(r) = \int \int f(r, \theta, \phi) Y_{lm}(\theta, \phi) \sin(\theta) d\theta d\phi`
over various values of :math:`r` and then interpolated :math:`\tilde{w}_{lm}` using a cubic spline.
This integral can be done using the :func:`angular grid<grid.angular>` module.
The interpolation of :math:`f` is simply then

.. math::
    f(r, \theta, \phi) \approx \sum_{l=0}^{L_{max}} \sum_{m=-l}^l \tilde{w}_{lm}(r) Y_{lm}(\theta, \phi),

where :math:`L_{max}` is the maximum chosen degree :math:`l` of the real spherical harmonics.
Derivatives up to second order can also be interpolated using this procedure.


Linear ODE Solver and Poisson
-----------------------------



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

Grid also includes popular quadrature grids that can integrate polynomials of
degree :math:`2n - 1` on :math:`[-1, 1]`: :func:`Gauss-Laguerre<grid.onedgrid.GaussLaguerre>` (domain :math:`[0, \infty)`),
:func:`Gauss-Legendre<grid.onedgrid.GaussLegendre>`, and :func:`Gauss-Chebyshev<grid.onedgrid.GaussChebyshev>`;

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



.. [#becke1988_multicenter] “A multicenter numerical integration scheme for polyatomic molecules”,
       Becke, A. D.; J. Chem. Phys. 1988 (v. 88 pp. 2547–2553); http://dx.doi.org/10.1063/1.454033

API Reference
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   pyapi/modules
  
 
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
