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


from glob import glob

import Cython.Build
import numpy as np
from setuptools import setup, Extension


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
    cmdclass={"build_ext": Cython.Build.build_ext},
    package_dir={"grid": "grid"},
    packages=["grid", "grid.test", "grid.grid",
              "grid.data", "grid.test.data", "grid.grid.data"],
    package_data={"grid.data": ["*"],
                  "grid.test.data": ["*"],
                  "grid.grid.data": ["*"]},
    include_package_data=True,
    ext_modules=[
        Extension("grid.grid.cext",
                  sources=glob("grid/grid/*.cpp") + [
                      "grid/grid/cext.pyx",
                      "grid/cell.cpp",
                      "grid/moments.cpp"],
                  depends=glob("grid/grid/*.pxd") + glob("grid/grid/*.h") + [
                      "grid/cell.pxd", "grid/cell.h",
                      "grid/moments.pxd", "grid/moments.h"],
                  include_dirs=[np.get_include(), "."],
                  extra_compile_args=["-std=c++11"],
                  language="c++", ),
        Extension("grid.cext",
                  sources=glob("grid/*.cpp") + ["grid/cext.pyx"],
                  depends=glob("grid/*.pxd") + glob("grid/*.h"),
                  include_dirs=[np.get_include(), "."],
                  extra_compile_args=["-std=c++11"],
                  language="c++"),
    ],
    zip_safe=False,
    setup_requires=["numpy>=1.0", "cython>=0.24.1"],
    install_requires=["numpy>=1.0", "nose>=0.11", "cython>=0.24.1", "scipy", "matplotlib"],
)
