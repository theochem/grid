#!/usr/bin/env python3
"""
GBasis-CubicProTransform integration
"""
import numpy as np
import time
import sys
from pathlib import Path
from dataclasses import dataclass

# Setup paths for Grid library
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Grid imports
try:
    from grid.onedgrid import GaussChebyshev
    from grid.protransform import CubicProTransform
    HAS_GRID = True
except ImportError as e:
    print(f"Grid library import failed: {e}")
    HAS_GRID = False

# GBasis imports
try:
    from iodata import load_one
    from gbasis.wrappers import from_iodata
    from gbasis.evals.density import evaluate_density
    HAS_GBASIS = True
except ImportError as e:
    print(f"GBasis library import failed: {e}")
    HAS_GBASIS = False

# Constants for adaptivity ratio calculation
CORE_DISTANCE_THRESHOLD = 2.0
OUTER_DISTANCE_THRESHOLD = 4.0

@dataclass
class IntegrationResult:
    """Results from GBasis-CubicProTransform integration"""
    calculated_electrons: float
    expected_electrons: int
    relative_error: float
    computation_time: float
    grid_points: int
    molecule_name: str
    adaptivity_ratio: float


class GBasisCubicProIntegrator:
    """Integrator combining quantum chemistry data with CubicProTransform"""
    
    def __init__(self, grid_size: int = 15):
        self.grid_size = grid_size
        self.available = HAS_GRID and HAS_GBASIS
        
        if not self.available:
            missing = []
            if not HAS_GRID:
                missing.append("Grid library")
            if not HAS_GBASIS:
                missing.append("GBasis library") 
            print(f"Warning: Missing {', '.join(missing)}")
    
    def _create_promolecular_params(self, coords, atnums):
        """Create default promolecular parameters for CubicProTransform
        
        This creates simple Gaussian parameters for each atom based on atomic number.
        Used when specific promolecular parameters are not available.
        
        Args:
            coords: Atomic coordinates array
            atnums: Atomic numbers array
            
        Returns:
            tuple: (coefficients, exponents) arrays
        """
        num_atoms = len(coords)
        max_functions = 4
        
        coeffs = np.zeros((num_atoms, max_functions))
        exps = np.zeros((num_atoms, max_functions))
        
        for i, Z in enumerate(atnums):
            base_coeff = float(Z)
            base_exp = 0.5 * Z**0.5
            
            # Create multiple functions with decreasing coefficients and exponents
            for j in range(max_functions):
                coeffs[i, j] = base_coeff * (0.7 ** j)
                exps[i, j] = base_exp * (2.0 ** (3-j))
        
        return coeffs, exps
    
    def _calculate_adaptivity_ratio(self, points):
        """Calculate adaptivity ratio for grid point distribution
        
        Measures how the grid points are distributed between core and outer regions.
        Higher ratios indicate more points concentrated near atomic centers.
        """
        distances = np.linalg.norm(points, axis=1)
        core_points = np.sum(distances < CORE_DISTANCE_THRESHOLD)
        outer_points = np.sum(distances > OUTER_DISTANCE_THRESHOLD)
        
        if outer_points > 0:
            return core_points / outer_points
        else:
            return float('inf')
    
    def integrate_fchk(self, fchk_file: str) -> IntegrationResult:
        """Perform integration on FCHK file"""
        if not self.available:
            raise RuntimeError("Required libraries not available")
        
        start_time = time.time()
        
        # Load data
        mol_data = load_one(fchk_file)
        ao_basis = from_iodata(mol_data)
        
        coords = mol_data.atcoords
        atnums = mol_data.atnums
        expected_electrons = int(np.sum(atnums))
        molecule_name = Path(fchk_file).stem
        
        if hasattr(mol_data, "one_rdms") and "scf" in mol_data.one_rdms:
            rdm = mol_data.one_rdms["scf"]
        else:
            raise ValueError("No SCF density matrix found")
        
        # Create grid
        coeffs, exps = self._create_promolecular_params(coords, atnums)
        oned_grid = GaussChebyshev(self.grid_size)
        transform = CubicProTransform([oned_grid, oned_grid, oned_grid], coeffs, exps, coords)
        
        # Integrate
        density = evaluate_density(rdm, ao_basis, transform.points)
        calculated_electrons = transform.integrate(density)
        
        # Calculate metrics
        adaptivity_ratio = self._calculate_adaptivity_ratio(transform.points)
        computation_time = time.time() - start_time
        absolute_error = abs(calculated_electrons - expected_electrons)
        relative_error = absolute_error / expected_electrons
        
        return IntegrationResult(
            calculated_electrons=calculated_electrons,
            expected_electrons=expected_electrons,
            relative_error=relative_error,
            computation_time=computation_time,
            grid_points=len(transform.points),
            molecule_name=molecule_name,
            adaptivity_ratio=adaptivity_ratio
        )


def run_integration_test():
    """Run integration test"""
    if not (HAS_GRID and HAS_GBASIS):
        print("Error: Required libraries not available")
        return
    
    integrator = GBasisCubicProIntegrator(grid_size=15)
    
    test_files = [
        ("data/He_t.fchk", "Helium"),
        ("data/H2.fchk", "Hydrogen"), 
        ("data/h2o.fchk", "Water"),
        ("data/HCl.fchk", "HCl"),
        ("data/CH4.fchk", "Methane")
    ]
    
    print("GBasis Integration Test")
    print("=" * 40)
    
    results = []
    for fchk_file, name in test_files:
        if not Path(fchk_file).exists():
            print(f"{name}: File not found")
            continue
            
        try:
            result = integrator.integrate_fchk(fchk_file)
            results.append(result)
            
            print(f"{name}: {result.calculated_electrons:.3f}/{result.expected_electrons} electrons, "
                  f"{result.relative_error*100:.2f}% error, {result.computation_time:.2f}s")
            
        except Exception as e:
            print(f"{name}: Failed - {e}")
    
    if results:
        avg_error = np.mean([r.relative_error for r in results])
        print(f"\nAverage error: {avg_error*100:.2f}%")


if __name__ == "__main__":
    run_integration_test() 