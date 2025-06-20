#!/usr/bin/env python3
"""
BFit Coefficients Integration Script

Integrates promolecular coefficients and exponents from theochem/BFit
kl_slsqp_results.npz file into Grid library's _PromolParams system.
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from grid.protransform import _PromolParams, CubicProTransform
    from grid.onedgrid import GaussChebyshev
except ImportError as e:
    print(f"Failed to import Grid library modules: {e}")
    sys.exit(1)
ELEMENT_TO_ATOMIC_NUMBER = {
    "h": 1,
    "he": 2,
    "li": 3,
    "be": 4,
    "b": 5,
    "c": 6,
    "n": 7,
    "o": 8,
    "f": 9,
    "ne": 10,
    "na": 11,
    "mg": 12,
    "al": 13,
    "si": 14,
    "p": 15,
    "s": 16,
    "cl": 17,
    "ar": 18,
    "k": 19,
    "ca": 20,
    "sc": 21,
    "ti": 22,
    "v": 23,
    "cr": 24,
    "mn": 25,
    "fe": 26,
    "co": 27,
    "ni": 28,
    "cu": 29,
    "zn": 30,
    "ga": 31,
    "ge": 32,
    "as": 33,
    "se": 34,
    "br": 35,
    "kr": 36,
    "rb": 37,
    "sr": 38,
    "y": 39,
    "zr": 40,
    "nb": 41,
    "mo": 42,
    "tc": 43,
    "ru": 44,
    "rh": 45,
    "pd": 46,
    "ag": 47,
    "cd": 48,
    "in": 49,
    "sn": 50,
    "sb": 51,
    "te": 52,
    "i": 53,
    "xe": 54,
}


@dataclass
class BFitElementData:
    """Store BFit data for a single element"""

    symbol: str
    atomic_number: int
    coeffs: np.ndarray
    exps: np.ndarray
    num_s: int = 0
    num_p: int = 0


class BFitDataProcessor:
    """Process BFit data and convert to Grid library format"""

    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.raw_data = None
        self.element_data = {}

    def load_bfit_data(self) -> bool:
        """Load BFit data file"""
        try:
            self.raw_data = np.load(self.data_file)
            return True
        except Exception as e:
            print(f"Failed to load data file: {e}")
            return False

    def parse_element_data(self) -> Dict[str, BFitElementData]:
        """Parse element data from BFit file"""
        if self.raw_data is None:
            print("Data not loaded, call load_bfit_data() first")
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
                    print(f"Missing coefficient or exponent data for element {symbol}")
                    continue

                coeffs = self.raw_data[coeffs_key]
                exps = self.raw_data[exps_key]

                num_s = self.raw_data.get(f"{symbol}_num_s", 0)
                num_p = self.raw_data.get(f"{symbol}_num_p", 0)

                atomic_number = ELEMENT_TO_ATOMIC_NUMBER.get(symbol, 0)

                element_data = BFitElementData(
                    symbol=symbol,
                    atomic_number=atomic_number,
                    coeffs=coeffs,
                    exps=exps,
                    num_s=int(num_s) if isinstance(num_s, np.ndarray) else num_s,
                    num_p=int(num_p) if isinstance(num_p, np.ndarray) else num_p,
                )

                self.element_data[symbol] = element_data
                print(
                    f"Parsed element {symbol.upper()} (Z={atomic_number}): "
                    f"{len(coeffs)} basis functions, s={element_data.num_s}, p={element_data.num_p}"
                )

            except Exception as e:
                print(f"Error parsing element {symbol}: {e}")

        return self.element_data

    def create_promol_params_for_molecule(
        self, atom_symbols: List[str], coordinates: np.ndarray
    ) -> "_PromolParams":
        """Create _PromolParams object for given molecule"""
        if len(atom_symbols) != len(coordinates):
            raise ValueError("Number of atom symbols must match number of coordinates")

        all_coeffs = []
        all_exps = []

        for symbol in atom_symbols:
            symbol_lower = symbol.lower()
            if symbol_lower not in self.element_data:
                raise ValueError(f"Parameter data not found for element {symbol}")

            element = self.element_data[symbol_lower]

            valid_mask = np.abs(element.coeffs) > 1e-12
            valid_coeffs = element.coeffs[valid_mask]
            valid_exps = element.exps[valid_mask]

            all_coeffs.append(valid_coeffs)
            all_exps.append(valid_exps)

        from grid.protransform import _pad_coeffs_exps_with_zeros

        coeffs_padded, exps_padded = _pad_coeffs_exps_with_zeros(all_coeffs, all_exps)

        promol_params = _PromolParams(
            c_m=coeffs_padded, e_m=exps_padded, coords=coordinates.copy(), dim=3
        )

        return promol_params

    def create_test_cubic_transform(
        self, atom_symbols: List[str], coordinates: np.ndarray, npoints: int = 50
    ) -> "CubicProTransform":
        """Create test CubicProTransform object"""
        promol_params = self.create_promol_params_for_molecule(atom_symbols, coordinates)

        oned_grid = GaussChebyshev(npoints)

        transform = CubicProTransform(
            [oned_grid, oned_grid, oned_grid],
            promol_params.c_m,
            promol_params.e_m,
            promol_params.coords,
        )

        return transform


def demonstrate_usage():
    """Demonstrate BFit data integration"""
    print("=" * 60)
    print("BFit Data Integration Demonstration")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    data_file = project_root / "scripts" / "data" / "kl_slsqp_results.npz"

    processor = BFitDataProcessor(data_file)

    if not processor.load_bfit_data():
        return

    element_data = processor.parse_element_data()

    if not element_data:
        print("No element data successfully parsed")
        return

    print("\n" + "=" * 60)
    print("Creating Test Molecules")
    print("=" * 60)

    print("\n--- Test Case 1: Hydrogen Atom ---")
    try:
        h_coords = np.array([[0.0, 0.0, 0.0]])
        h_transform = processor.create_test_cubic_transform(["h"], h_coords, npoints=30)

        test_function_values = np.ones(h_transform.size)
        integral_result = h_transform.integrate(test_function_values)
        print(f"Constant function integral: {integral_result:.6f}")

    except Exception as e:
        print(f"Hydrogen atom test failed: {e}")

    if "o" in element_data:
        print("\n--- Test Case 2: Water Molecule ---")
        try:
            water_coords = np.array(
                [[0.0, 0.0, 0.0], [1.4309, 0.0, 1.1071], [-1.4309, 0.0, 1.1071]]
            )
            water_symbols = ["o", "h", "h"]

            water_transform = processor.create_test_cubic_transform(
                water_symbols, water_coords, npoints=25
            )

            test_values = np.exp(-0.1 * np.linalg.norm(water_transform.points, axis=1) ** 2)
            integral_result = water_transform.integrate(test_values)
            print(f"Gaussian function integral: {integral_result:.6f}")

        except Exception as e:
            print(f"Water molecule test failed: {e}")

    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)

    print(f"Total elements processed: {len(element_data)}")
    print("Basis function counts by element:")
    for symbol, data in sorted(element_data.items()):
        valid_count = np.sum(np.abs(data.coeffs) > 1e-12)
        print(
            f"  {symbol.upper()} (Z={data.atomic_number}): {valid_count}/{len(data.coeffs)} valid basis functions"
        )


if __name__ == "__main__":
    demonstrate_usage()
