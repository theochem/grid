# BFit Integration and Five Systems Test

Scripts for integrating BFit promolecular coefficients into Grid library and testing electron density integration on five molecular systems.

## Files

### Core Scripts
- `integrate_bfit_coefficients.py` - Integrates BFit data into Grid library _PromolParams
- `five_systems_test.py` - Tests five molecular systems with performance analysis

### Results
- `../examples/BFit_Performance_Analysis.ipynb` - **Main results notebook** with comprehensive analysis

### Data
- `data/kl_slsqp_results.npz` - BFit coefficients and exponents
- `data/*.fchk` - FCHK files for test systems

## Requirements

```bash
pip install numpy scipy pandas matplotlib
pip install git+https://github.com/theochem/iodata.git
pip install git+https://github.com/theochem/gbasis.git
```

## Usage

### 1. Test BFit Integration
```bash
python scripts/integrate_bfit_coefficients.py
```

### 2. Run Five Systems Test
```bash
python scripts/five_systems_test.py
```

### 3. View Results
```bash
jupyter notebook examples/BFit_Performance_Analysis.ipynb
```

## Test Systems

| System | Type | Expected Electrons | 
|--------|------|-------------------|
| Helium | Atom | 2 |
| H2 | Homonuclear diatomic | 2 |
| HCl | Heteronuclear diatomic | 18 |
| CH4 | Small molecule | 10 |
| H2O | Additional system | 10 |

## Key Results

### CubicProTransform Method
- High precision integration (0.3% error across all systems)
- Grid transformation time: < 0.001 seconds
- Integration time: < 0.001 seconds per system
- Consistent performance across diverse molecular types

### GBasis Method  
- Error range: 0.9-9.2% (system dependent)
- Calculation time: 0.066-0.410 seconds
- Provides quantum mechanical electron density benchmark
- Grid size optimization: 16³-40³ points depending on system

## Implementation

### BFit Integration Process:
1. **Data Loading**: Loads coefficients/exponents from kl_slsqp_results.npz
2. **Coefficient Filtering**: Removes near-zero coefficients (threshold: 1e-12)
3. **Parameter Creation**: Creates _PromolParams objects for molecular systems
4. **Normalization**: Normalizes coefficients to match expected electron count
5. **Grid Generation**: Uses CubicProTransform with GaussChebyshev quadrature

### Testing Framework:
- **CubicProTransform**: Uses `transform.promol.promolecular()` for density evaluation
- **GBasis Integration**: Loads FCHK files with IOData, evaluates density with GBasis
- **Performance Metrics**: Measures integration accuracy, timing, and grid efficiency
- **Five Systems**: Validates across atoms, diatomics, and small molecules

### Technical Details:
- **Grid Integration**: `transform.integrate(promol_density)` method
- **Density Matrix**: SCF one-electron reduced density matrices from FCHK
- **Error Analysis**: Relative error vs expected electron count
- **Grid Optimization**: Adaptive grid sizing (16³ to 40³ points)
