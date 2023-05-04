# GRID
[![Build Status](https://travis-ci.org/theochem/grid.svg?branch=master)](https://travis-ci.org/theochem/grid)
[![Build Status](https://dev.azure.com/yxt1991/Grid/_apis/build/status/theochem.grid?branchName=master)](https://dev.azure.com/yxt1991/Grid/_build/latest?definitionId=2&branchName=master)
[![codecov](https://codecov.io/gh/theochem/grid/branch/master/graph/badge.svg)](https://codecov.io/gh/theochem/grid)<br/>
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://docs.python.org/3/whatsnew/3.6.html)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![GitHub contributors](https://img.shields.io/github/contributors/theochem/grid.svg)](https://github.com/theochem/grid/graphs/contributors)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://black.readthedocs.io/en/stable/)

## About
Grid is a simple, free, and open-source Python library for numerical integration, interpolation and differentiation.
Primarly intended for the quantum chemistry community to assist in density-functional (DFT) theory calculations,
including support for periodic boundary conditions.

Please visit [**Grid Documentation**](https://grid.qcdevs.org/) for more information with 
examples about the software. 

To report any issues or ask questions, either [open an issue](
https://github.com/theochem/grid/issues/new) or email [qcdevs@gmail.com]().

## Functionality
* One-dimensional integration
  * Includes various:
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

## Citation
Please use the following citation in any publication using BFit library:

> **"Grid: A Python Library for Molecular Integration, Interpolation, and More"**,
> X.D. Yang, A. Tehrani, L. Pujal, R. Hernández-Esparza, 
> M. Chan, E. Vöhringer-Martinez, T. Verstraelen, F. Heidar-Zadeh, P. W. Ayers
> `REFERENCE <https://doi.org/TODO>`__.


## Dependencies
* Python >= 3.0: http://www.python.org/
* NumPy >= 1.16.0: http://www.numpy.org/
* SciPy >= 1.4.0: http://www.scipy.org/
* Sphinx >= 2.3.0: https://www.sphinx-doc.org/
* SymPy >= 1.4.0: https://www.sympy.org/en/index.html
* QA requirement: Tox >= 4.0.0: https://tox.wiki/en/latest/
* Testing requirement: pytest >= 7.0.0: https://docs.pytest.org/

## Installation
To install Grid to system:
```bash
pip install .
```
To run tests:
```bash
pytest --pyargs grid
```

## Local build and Testing
To install editable Grid locally:
```bash
pip install -e .
```
To run tests:
```bash
pytest tests
```

## Quality Assurance
To run QA locally:
```bash
tox
```

## Funding Acknowledgement
This software was developed using funding from a variety of international
sources including, but not limited to:
* Canarie
* Canada Research Chairs
* Digital Research Alliance of Canada: Compute Canada
* the European Union's Horizon 2020 Marie Sklodowska-Curie Actions (Individual Fellowship No 800130)
* the Foundation of Scientific Research--Flanders (FWO)
* McMaster University
* Queen's University
* National Fund for Scientific and Technological Development of Chile (FONDECYT)
* Natural Sciences and Engineering Research Council of Canada (NSERC)
* Research Board of Ghent University (BOF)
* SHARCNET

## License
GRID is distributed under [GPL License version 3](https://github.com/theochem/grid/blob/master/LICENSE) (GPL v3).

