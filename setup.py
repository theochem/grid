#!/usr/bin/env python
# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2019 The GRID Development Team
#
# This file is part of GRID.
#
# GRID is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GRID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
# pragma pylint: disable=invalid-name
"""Package build and install script."""
import os

from setuptools import find_packages, setup


def get_version_info():
    """Read __version__ and DEV_CLASSIFIER from version.py, using exec, not import."""
    fn_version = os.path.join("bfit", "_version.py")
    if os.path.isfile(fn_version):
        myglobals = {}
        with open(fn_version, "r") as f:
            exec(f.read(), myglobals)  # pylint: disable=exec-used
        return myglobals["__version__"], myglobals["DEV_CLASSIFIER"]
    return "0.0.0.post0", "Development Status :: 2 - Pre-Alpha"


def get_readme():
    """Load README.rst for display on PyPI."""
    with open('README.md') as fhandle:
        return fhandle.read()


VERSION, DEV_CLASSIFIER = get_version_info()


setup(
    name="qc-grid",
    version=VERSION,
    description="Grid Package",
    long_description=get_readme(),
    author="QC-Devs Community",
    author_email="qcdevs@gmail.com",
    url="https://github.com/theochem/grid",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "grid.data": ["*.*"],
        "grid.data.lebedev": ["*.npz"],
        "grid.data.spherical_design": ["*.npz"],
        "grid.data.prune_grid": ["*.npz"],
        "grid.data.proatoms": ["*.npz"],
    },
    zip_safe=False,
    install_requires=[
        "numpy>=1.16",
        "pytest>=2.6",
        "scipy>=1.4",
        "importlib_resources",
        "sympy",
        "numpydoc", "sphinx_copybutton", "sphinx-autoapi", "nbsphinx", "sphinx_rtd_theme",
        "sphinx_autodoc_typehints",
    ],
    classifiers=[
        'Development Status :: 0 - Released',
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Intended Audience :: Science/Research',
    ],
)
