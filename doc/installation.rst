.. _usr_installation:

Installation
############

Downloading Code
================

The latest code can be obtained through theochem (https://github.com/theochem/grid) in Github,

.. code-block:: bash

   git clone https://github.com/theochem/grid.git

.. _usr_py_depend:

Dependencies
============

The following dependencies will be necessary for Procrustes to build properly,

* Python >= 3.0: http://www.python.org/
* NumPy >= 1.16.0: http://www.numpy.org/
* SciPy >= 1.4.0: http://www.scipy.org/
* Sphinx >= 2.3.0: https://www.sphinx-doc.org/
* SymPy >= 1.4.0: https://www.sympy.org/en/index.html
* PyTest >= 5.4.3: `https://docs.pytest.org/ <https://docs.pytest.org/>`_
* QA requirement: Tox >= 4.0.0: https://tox.wiki/en/latest/
* Sphinx >= 2.3.0, if one wishes to build the documentation locally:
  `https://www.sphinx-doc.org/ <https://www.sphinx-doc.org/>`_



Installation
============

Grid can be cloned via git,

.. code-block:: bash

   git clone https://github.com/theochem/grid.git


Then installation via pip can be done by going into the directory where Grid is downloaded to and running,

.. code-block:: bash

    cd grid
    pip install .

Successful installation can be checked by running the tests,

.. code-block:: bash

    pytest --pyargs grid









..
    The stable release of the package can be easily installed through the *pip* and
    *conda* package management systems, which install the dependencies automatically, if not
    available. To use *pip*, simply run the following command:

    .. code-block:: bash

        pip install qc-procrustes

    To use *conda*, one can either install the package through Anaconda Navigator or run the following
    command in a desired *conda* environment:

    .. code-block:: bash

        conda install -c theochem qc-procrustes


    Alternatively, the *Procrustes* source code can be download from GitHub (either the stable version
    or the development version) and then installed from source. For example, one can download the latest
    source code using *git* by:

    .. code-block:: bash

        # download source code
        git clone git@github.com:theochem/procrustes.git
        cd procrustes

    From the parent directory, the dependencies can either be installed using *pip* by:

    .. code-block:: bash

        # install dependencies using pip
        pip install -r requirements.txt


    or, through *conda* by:

    .. code-block:: bash

        # create and activate myenv environment
        # Procruste works with Python 3.6, 3.7, and 3.8
        conda create -n myenv python=3.6
        conda activate myenv
        # install dependencies using conda
        conda install --yes --file requirements.txt


    Finally, the *Procrustes* package can be installed (from source) by:

    .. code-block:: bash

        # install Procrustes from source
        pip install .

Building Documentation
======================

The following shows how to build the documentation using sphinx to the folder `_build`.

    .. code-block:: bash

        cd ./doc
        ./gen_api.sh
        sphinx-build -b html . _build
