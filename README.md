# GRID
[![Build Status](https://travis-ci.org/theochem/grid.svg?branch=master)](https://travis-ci.org/theochem/grid)
[![Build Status](https://dev.azure.com/yxt1991/Grid/_apis/build/status/theochem.grid?branchName=master)](https://dev.azure.com/yxt1991/Grid/_build/latest?definitionId=2&branchName=master)
[![codecov](https://codecov.io/gh/theochem/grid/branch/master/graph/badge.svg)](https://codecov.io/gh/theochem/grid)<br/>
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://docs.python.org/3/whatsnew/3.6.html)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/theochem/grid/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/theochem/grid.svg)](https://github.com/theochem/grid/graphs/contributors)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://black.readthedocs.io/en/stable/)

## About
GRID is a pythonic numerical integral package. It derived from legacy HORTON 2 numerical integration module.

## Platform
GRID is a pure python package supporting `Windows`, `Linux` and `MacOS`.

## Functionality
* 1d integral
* 1d transformation
* Spherical integral
* Becke-Lebedev grid & Molecular integral
* Interpolation & differentiation
* General ODE solver and Poisson solver

## License
GRID is distributed under [GPL License version 3](https://github.com/theochem/grid/blob/master/LICENSE) (GPL v3).

## Dependence
* Installation requirements: `numpy`, `scipy`, `importlib_resources`
* Testing requirement: `pytest`
* QA requirement: `tox`

## Installation
To install GRID to system:
```bash
pip install .
```
To run tests:
```bash
pytest --pyargs grid
```

## Local build and Testing
To install editable GRID locally:
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
* the Canada Research Chairs
* Compute Canada
* the European Union's Horizon 2020 Marie Sklodowska-Curie Actions (Individual Fellowship No 800130)
* the Foundation of Scientific Research--Flanders (FWO)
* McMaster University
* the National Fund for Scientific and Technological Development of Chile (FONDECYT)
* the Natural Sciences and Engineering Research Council of Canada (NSERC)
* the Research Board of Ghent University (BOF)
* Sharcnet
