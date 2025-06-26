#!/usr/bin/env python3
"""
BFit data integration with CubicProTransform
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
    from grid.protransform import _PromolParams, CubicProTransform, _pad_coeffs_exps_with_zeros
    HAS_GRID = True
except ImportError as e:
    print(f"Grid library import failed: {e}")
    HAS_GRID = False

# Element to atomic number mapping (extended for more elements)
ELEMENT_TO_ATOMIC_NUMBER = {
    'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9, 'ne': 10,
    'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19, 'ca': 20,
    'sc': 21, 'ti': 22, 'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30,
    'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39, 'zr': 40,
    'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48, 'in': 49, 'sn': 50,
    'sb': 51, 'te': 52, 'i': 53, 'xe': 54
}

# Constants for filtering and calculations
COEFFICIENT_THRESHOLD = 1e-12  # Minimum coefficient value to include
CORE_DISTANCE_THRESHOLD = 1.0
OUTER_DISTANCE_THRESHOLD = 2.0


@dataclass
class BFitElementData:
    """Store BFit data for a single element"""
    symbol: str
    atomic_number: int
    coeffs: np.ndarray
    exps: np.ndarray
    num_s: int = 0
    num_p: int = 0


@dataclass
class IntegrationResult:
    """Results from CubicProTransform integration"""
    integral_value: float
    computation_time: float
    element: str
    adaptivity_ratio: float
    basis_functions: int


class BFitDataProcessor:
    """BFit data processor"""

    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.raw_data = None
        self.element_data = {}

    def load_bfit_data(self):
        """Load BFit data file"""
        try:
            self.raw_data = np.load(self.data_file)
            return True
        except Exception as e:
            print(f"Failed to load BFit data file: {e}")
            return False

    def parse_element_data(self):
        """Parse element data from BFit file"""
        if self.raw_data is None:
            return {}

        element_symbols = set()
        for key in self.raw_data.files:
            if "_" in key:
                symbol = key.split("_")[0]
                element_symbols.add(symbol)

        for symbol in element_symbols:
            try:
                coeffs_key = f"{symbol}_coeffs"
                exps_key = f"{symbol}_exps"

                if coeffs_key not in self.raw_data or exps_key not in self.raw_data:
                    continue

                all_coeffs = self.raw_data[coeffs_key]
                all_exps = self.raw_data[exps_key]

                num_s = self.raw_data.get(f"{symbol}_num_s", 0)
                num_p = self.raw_data.get(f"{symbol}_num_p", 0)

                # Convert to int if numpy scalar
                num_s = int(num_s) if isinstance(num_s, np.ndarray) else num_s
                num_p = int(num_p) if isinstance(num_p, np.ndarray) else num_p

                # Only use s-type functions
                if num_s > 0:
                    coeffs = all_coeffs[:num_s]
                    exps = all_exps[:num_s]
                else:
                    continue

                atomic_number = ELEMENT_TO_ATOMIC_NUMBER.get(symbol, 0)

                element_data = BFitElementData(
                    symbol=symbol,
                    atomic_number=atomic_number,
                    coeffs=coeffs,
                    exps=exps,
                    num_s=num_s,
                    num_p=num_p,
                )

                self.element_data[symbol] = element_data

            except Exception as e:
                print(f"Error parsing element {symbol}: {e}")

        return self.element_data


class BFitCubicProIntegrator:
    """BFit data integration using CubicProTransform"""
    
    def __init__(self, grid_size: int = 15):
        self.grid_size = grid_size
        self.bfit_processor = self._load_bfit_data()
        
    def _load_bfit_data(self):
        """Load BFit promolecular data"""
        data_file = Path(__file__).parent / "data" / "kl_slsqp_results.npz"
        
        if not data_file.exists():
            raise FileNotFoundError(f"BFit data file not found: {data_file}")
            
        processor = BFitDataProcessor(data_file)
        
        if not processor.load_bfit_data():
            raise RuntimeError("Failed to load BFit data")
            
        element_data = processor.parse_element_data()
        
        if not element_data:
            raise RuntimeError("No BFit element data successfully parsed")
            
        return processor
    
    def integrate_promolecular_density(self, element: str, coords=None):
        """Integrate promolecular density using CubicProTransform"""
        if coords is None:
            coords = np.array([[0.0, 0.0, 0.0]])
            
        start_time = time.time()
        
        # Get element data
        element_data = self.bfit_processor.element_data[element.lower()]
        
        # Filter valid coefficients
        valid_mask = np.abs(element_data.coeffs) > COEFFICIENT_THRESHOLD
        valid_coeffs = element_data.coeffs[valid_mask]
        valid_exps = element_data.exps[valid_mask]
        
        # Create promol params
        all_coeffs = [valid_coeffs]
        all_exps = [valid_exps]
        coeffs_padded, exps_padded = _pad_coeffs_exps_with_zeros(all_coeffs, all_exps)
        
        promol_params = _PromolParams(
            c_m=coeffs_padded, e_m=exps_padded, coords=coords.copy(), dim=3
        )
        
        # Create grid
        oned_grid = GaussChebyshev(self.grid_size)
        protrans_grid = CubicProTransform(
            [oned_grid, oned_grid, oned_grid], 
            promol_params.c_m, promol_params.e_m, promol_params.coords
        )
        
        # Integrate
        points = protrans_grid.points
        promol_density = protrans_grid.promol.promolecular(points)
        integral_value = protrans_grid.integrate(promol_density)
        
        # Calculate adaptivity
        adaptivity_ratio = self._calculate_adaptivity_ratio(points)
        
        computation_time = time.time() - start_time
        valid_basis = np.sum(valid_mask)
        
        return IntegrationResult(
            integral_value=integral_value,
            computation_time=computation_time,
            element=element,
            adaptivity_ratio=adaptivity_ratio,
            basis_functions=valid_basis
        )
    
    def _calculate_adaptivity_ratio(self, points):
        """Calculate adaptivity ratio for grid point distribution"""
        distances = np.linalg.norm(points, axis=1)
        core_points = np.sum(distances < CORE_DISTANCE_THRESHOLD)
        outer_points = np.sum(distances > OUTER_DISTANCE_THRESHOLD)
        
        if outer_points > 0:
            return core_points / outer_points
        else:
            return float('inf')


def run_integration_test():
    """Run BFit integration test"""
    try:
        integrator = BFitCubicProIntegrator(grid_size=15)
        
        # Test available elements
        test_elements = ['H', 'C', 'O', 'He']
        available_elements = [elem for elem in test_elements 
                            if elem.lower() in integrator.bfit_processor.element_data]
        
        if not available_elements:
            print("No test elements available")
            return
        
        print("BFit Integration Test")
        print("=" * 40)
        
        for element in available_elements:
            try:
                result = integrator.integrate_promolecular_density(element)
                
                print(f"{element}: {result.integral_value:.3f} integral, "
                      f"{result.adaptivity_ratio:.1f}:1 adaptivity, {result.computation_time:.2f}s")
                
            except Exception as e:
                print(f"{element}: Failed - {e}")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_integration_test() 