#!/usr/bin/env python
# -*- coding: utf-8 -*-
# GRID is a numerical integration library for quantum chemistry.
#
# Copyright (C) 2011-2017 The GRID Development Team
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
#
# --
# pragma pylint: disable=invalid-name
"""Package build and install script."""

from setuptools import setup


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
    with open("README.rst") as f:
        return f.read()


setup(
    name="grid",
    version=get_version(),
    description="Legacy HORTON grid module",
    long_description=get_readme(),
    author="Toon Verstraelen",
    author_email="Toon.Verstraelen@UGent.be",
    url="https://github.com/theochem/grid",
    package_dir={"grid": "grid"},
    packages=["grid", "grid.test", "grid.grid",
              "grid.data", "grid.test.data", "grid.grid.data",
              "grid.grid.data.lebgrid"],
    package_data={"grid.data": ["*"],
                  "grid.test.data": ["*"],
                  "grid.grid.data": ["*"],
                  "grid.grid.data.lebgrid": ["*"]},
    include_package_data=True,
    zip_safe=False,
    setup_requires=["numpy>=1.0"],
    install_requires=["numpy>=1.0", "pytest>=2.6", "scipy", "matplotlib"],
)
