# Grid Integration Scripts

Integration tools for combining quantum chemistry data with Grid library's CubicProTransform.

## Files

- `gbasis_cubicpro_integration.py` - GBasis quantum chemistry data integration
- `bfit_cubicpro_integration.py` - BFit promolecular parameter integration  
- `test_grid_analysis.py` - Functionality and performance testing

## Requirements

```bash
pip install numpy scipy pandas matplotlib
pip install git+https://github.com/theochem/iodata.git
pip install git+https://github.com/theochem/gbasis.git
```

## Usage

```bash
# Run individual integrators
python gbasis_cubicpro_integration.py
python bfit_cubicpro_integration.py

# Run test suite
python test_grid_analysis.py
```

## Test Architecture

### Functionality Testing
- **GBasis**: 5 molecules (He, H2, H2O, HCl, CH4) using 15³ grid
- **BFit**: 6 elements (H, He, Li, C, N, O) using 15³ grid

### Performance Testing  
- **Grid sizes**: 5³, 8³, 10³, 12³, 15³, 18³, 20³, 25³, 50³
- **Reference molecule**: H2O for performance scaling analysis

## Results Summary

| Method | Test Type | Average Result | Grid Size |
|--------|-----------|----------------|-----------|
| GBasis | 5 molecules | 0.92% error | 15³ |
| BFit | 6 elements | All passed | 15³ |
| Performance | 9 grid sizes | 0.01-33.75% error | 5³-50³ |

## Grid Performance (H2O)

| Grid | Points | Time | Error | Use Case |
|------|--------|------|-------|----------|
| 5³ | 125 | 0.13s | 33.75% | Quick preview |
| 15³ | 3,375 | 2.65s | 0.17% | **Recommended** |
| 25³ | 15,625 | 11.14s | 0.01% | High precision |
| 50³ | 125,000 | 90.34s | 0.06% | Extreme precision |

## API Reference

### GBasis Integration
```python
from gbasis_cubicpro_integration import GBasisCubicProIntegrator

integrator = GBasisCubicProIntegrator(grid_size=15)
result = integrator.integrate_fchk("data/h2o.fchk")
```

### BFit Integration
```python
from bfit_cubicpro_integration import BFitCubicProIntegrator

integrator = BFitCubicProIntegrator(grid_size=15)
result = integrator.integrate_promolecular_density('C')
```

## Data Requirements

- **FCHK files**: `data/*.fchk` (He, H2, H2O, HCl, CH4)
- **BFit data**: `data/kl_slsqp_results.npz`
