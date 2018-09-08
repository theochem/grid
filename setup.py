#!/usr/bin/env python
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
        with open('old_grids/version.py', 'r') as f:
            return f.read().split('=')[-1].replace('\'', '').strip()
    except FileNotFoundError:
        return "0.0.0"


def get_readme():
    """Load README.rst for display on PyPI."""
    with open('README.rst') as f:
        return f.read()


setup(
    name='old_grids',
    version=get_version(),
    description='Legacy HORTON grid module',
    long_description=get_readme(),
    author='Toon Verstraelen',
    author_email='Toon.Verstraelen@UGent.be',
    url='https://github.com/theochem/old_grids',
    cmdclass={'build_ext': Cython.Build.build_ext},
    package_dir={'old_grids': 'old_grids'},
    packages=['old_grids', 'old_grids.test'],

    ext_modules=[
        Extension("old_grids.grid.cext",
        sources=glob('old_grids/grid/*.cpp') + [
                 "old_grids/grid/cext.pyx",
                 'old_grids/cell.cpp',
                 'old_grids/moments.cpp'],
        depends=glob('old_grids/grid/*.pxd') + glob('old_grids/grid/*.h') + [
            'old_grids/cell.pxd', 'old_grids/cell.h',
            'old_grids/moments.pxd', 'old_grids/moments.h'],
        include_dirs=[np.get_include(), '.'],
        extra_compile_args=['-std=c++11'],
        language="c++", ),
        Extension("old_grids.cext",
            sources=glob('old_grids/*.cpp') + ["old_grids/cext.pyx"],
            depends=glob('old_grids/*.pxd') + glob('old_grids/*.h'),
            include_dirs=[np.get_include(), '.'],
            extra_compile_args=['-std=c++11'],
            language="c++"),
    ],
    zip_safe=False,
    setup_requires=['numpy>=1.0', 'cython>=0.24.1'],
    install_requires=['numpy>=1.0', 'nose>=0.11', 'cython>=0.24.1', 'scipy', 'matplotlib', 'iodata',
                      'gbasis'],
)
