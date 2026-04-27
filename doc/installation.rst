.. _usr_installation:

Installation
############

.. _usr_py_depend:

Dependencies
============

The following dependencies are required for Grid:

* Python >= 3.10: http://www.python.org/
* NumPy >= 1.23.5: http://www.numpy.org/
* SciPy >= 1.15.0: http://www.scipy.org/
* SymPy >= 1.4.0: https://www.sympy.org/en/index.html
* PyTest >= 8.0.0: `https://docs.pytest.org/ <https://docs.pytest.org/>`_ (for running tests)
* Sphinx >= 2.3.0: `https://www.sphinx-doc.org/ <https://www.sphinx-doc.org/>`_ (for building docs)



Installation
============

Install the latest release from PyPI:

.. code-block:: bash

    pip install qc-grid

Install the latest development version directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/theochem/grid.git

Install from a local clone:

.. code-block:: bash

    git clone https://github.com/theochem/grid.git
    cd grid
    pip install .

Check the installation by running:

.. code-block:: bash

    pytest --pyargs grid

Building Documentation
======================

Build the documentation using Sphinx in the `_build` folder:

.. code-block:: bash

    cd ./doc
    ./gen_api.sh
    sphinx-build -b html . _build
