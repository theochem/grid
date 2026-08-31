# Welcome to Grid's Documentation!

Grid is a free and open-source Python library for numerical integration,
interpolation and differentiation of interest for the quantum chemistry community.
Grid is intended to support molecular density-functional (DFT) theory calculations,
and it also supports periodic boundary conditions.

## Citation

Please use the following citation in any publication using Grid:

> **Grid: A Python library for molecular integration, interpolation, differentiation, and more.**
> Alireza Tehrani, Xiaotian Derrick Yang, Marco Martínez-González, Leila Pujal,
> Raymundo Hernández-Esparza, Matthew Chan, Esteban Vöhringer-Martinez,
> Toon Verstraelen, Paul W. Ayers, Farnaz Heidar-Zadeh.
> *J. Chem. Phys. 160 (17), 172503 (2024).*
> https://doi.org/10.1063/5.0202240

```bibtex
@article{grid,
    author = {Tehrani, Alireza and Yang, Xiaotian Derrick and Martínez-González, Marco and Pujal, Leila and Hernández-Esparza, Raymundo and Chan, Matthew and Vöhringer-Martinez, Esteban and Verstraelen, Toon and Ayers, Paul W. and Heidar-Zadeh, Farnaz},
    title = {Grid: A Python library for molecular integration, interpolation, differentiation, and more},
    journal = {The Journal of Chemical Physics},
    volume = {160},
    number = {17},
    pages = {172503},
    year = {2024},
    month = {05},
    doi = {10.1063/5.0202240},
    url = {https://doi.org/10.1063/5.0202240},
}
```

## License and Contributions

The [Grid source code](https://github.com/theochem/grid) is hosted on GitHub and is released under the GNU General Public License v3.0. We welcome any contributions to Grid in accordance with our Code of Conduct and Contributing Guidelines. Please report issues on [GitHub Issues](https://github.com/theochem/grid/issues/new). For further information and inquiries please contact us at qcdevs@gmail.com.

## Key Features

### Integration
- **One-dimensional integration**: quadratures and transformations
- **Multivariate integration**:
  - Hypercubes with tensor product grids
  - Integration over spheres with angular grids
  - Molecular integration with various weight functions

### Transformations
- **Radial transformations** for converting one-dimensional grids between intervals
- Support for finite and infinite domains

### Analysis
- **Interpolation** on constructed grids
- **Differentiation** through grid-based methods
- **ODE solver** for linear ordinary differential equations
- **Poisson solver** for the Poisson equation

## Available Modules

| Module | Purpose |
| --- | --- |
| [`grid.onedgrid`](pyapi/grid.onedgrid) | One-dimensional quadrature grids and integration |
| [`grid.rtransform`](pyapi/grid.rtransform) | Radial transformations between intervals |
| [`grid.angular`](pyapi/grid.angular) | Angular grids for spherical integration |
| [`grid.atomgrid`](pyapi/grid.atomgrid) | Atomic grids for 3D integration |
| [`grid.molgrid`](pyapi/grid.molgrid) | Molecular grids with atom-in-molecule weights |
| [`grid.becke`](pyapi/grid.becke) | Becke weight functions |
| [`grid.hirshfeld`](pyapi/grid.hirshfeld) | Hirshfeld weight functions |
| [`grid.cubic`](pyapi/grid.cubic) | Hyper-rectangular grids in 2D or 3D |
| [`grid.periodicgrid`](pyapi/grid.periodicgrid) | Periodic grids for 1D, 2D, or 3D |
| [`grid.ode`](pyapi/grid.ode) | ODE solver |
| [`grid.poisson`](pyapi/grid.poisson) | Poisson equation solver |
| [`grid.utils`](pyapi/grid.utils) | Utility functions for coordinate transformations and spherical harmonics |
