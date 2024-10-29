# Grid

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![GitHub Actions CI Tox Status](https://github.com/theochem/grid/actions/workflows/pytest.yaml/badge.svg)](https://github.com/theochem/grid/actions/workflows/pytest.yaml)
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


## Citation
Please use the following citation in any publication:

> Tehrani, A., Yang, X.D., Martínez-González, M., Pujal, L., Hernández-Esparza, R., Chan, M., Vöhringer-Martinez, E., Verstraelen, T., Ayers, P.W. and Heidar-Zadeh, F., 2024. Grid: A Python library for molecular integration, interpolation, differentiation, and more. The Journal of Chemical Physics, 160(17).

with the following bibtex:

```
@article{tehrani2024grid,
  title={Grid: A Python library for molecular integration, interpolation, differentiation, and more},
  author={Tehrani, Alireza and Yang, Xiaotian Derrick and Mart{\'\i}nez-Gonz{\'a}lez, Marco and Pujal, Leila and Hern{\'a}ndez-Esparza, Raymundo and Chan, Matthew and V{\"o}hringer-Martinez, Esteban and Verstraelen, Toon and Ayers, Paul W and Heidar-Zadeh, Farnaz},
  journal={The Journal of Chemical Physics},
  volume={160},
  number={17},
  year={2024},
  publisher={AIP Publishing}
}
```

## Installation

Installation via pip can be done by the following command:
```bash
pip install git+https://github.com/theochem/grid.git@master
```

Local installation can be done as:
```bash
git clone https://github.com/theochem/grid.git
cd grid
pip install .
```

With later release in PyPi, Grid can be installed via
```bash
pip install qc-grid
```
