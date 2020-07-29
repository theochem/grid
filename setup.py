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


from setuptools import find_packages, setup


def get_version():
    """Load the version from version.py, without importing it.

    This function assumes that the last line in the file contains a variable defining the
    version string with single quotes.

    """
    try:
        with open("grid/version.py", "r") as f:
            return f.read().split("=")[-1].replace("'", "").strip()
    except FileNotFoundError:
        return "0.0.0"


def get_readme():
    """Load README.rst for display on PyPI."""
    with open("README.md") as f:
        return f.read()


setup(
    name="grid",
    version=get_version(),
    description="Python Library for Numerical Molecular Integration.",
    long_description=get_readme(),
    author="HORTON-ChemTools Dev Team",
    author_email="horton.chemtools@gmail.com",
    url="https://github.com/theochem/grid",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "grid.data": ["*.*"],
        "grid.data.lebedev": ["*.npz"],
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
    ],
)
