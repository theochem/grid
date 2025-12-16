#!/usr/bin/env python3
"""
BFit Data Processor

Processes promolecular coefficients and exponents from theochem/BFit
data files for integration with CubicProTransform in Grid library.

Note: This version handles s-type orbital data.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Element symbol to atomic number mapping
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
    """Store BFit s-type orbital data for a single element"""

    symbol: str
    atomic_number: int
    coeffs_s: np.ndarray  # S-type coefficients
    exps_s: np.ndarray  # S-type exponents
    num_s: int = 0
    # TODO: P-types spherical Gaussians should be supported later.


class BFitDataProcessor:
    """Process BFit data and convert to Grid library format"""

    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.raw_data: Optional[np.lib.npyio.NpzFile] = None
        self.element_data: Dict[str, BFitElementData] = {}

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

        # Get all elements with coefficients
        element_symbols = set()
        for key in self.raw_data.files:
            if "_coeffs_s" in key:
                symbol = key.replace("_coeffs_s", "")
                element_symbols.add(symbol)

        for symbol in element_symbols:
            try:
                coeffs_key = f"{symbol}_coeffs_s"
                exps_key = f"{symbol}_exps_s"

                if coeffs_key not in self.raw_data or exps_key not in self.raw_data:
                    print(f"Missing coefficient or exponent data for element {symbol}")
                    continue

                coeffs = self.raw_data[coeffs_key]
                exps = self.raw_data[exps_key]
                atomic_number = ELEMENT_TO_ATOMIC_NUMBER.get(symbol, 0)

                element_data = BFitElementData(
                    symbol=symbol,
                    atomic_number=atomic_number,
                    coeffs_s=coeffs,
                    exps_s=exps,
                    num_s=len(coeffs),
                )

                self.element_data[symbol] = element_data

            except Exception as e:
                print(f"Error parsing element {symbol}: {e}")

        return self.element_data

    def create_promol_params(self, symbols, coords, deps):
        """Create promolecular parameters from BFit data

        Args:
            symbols: List of element symbols
            coords: Array of atomic coordinates
            deps: Dictionary containing dependencies (_PromolParams, _pad_coeffs_exps_with_zeros)

        Returns:
            _PromolParams object
        """
        all_coeffs, all_exps, all_coords = [], [], []

        for i, symbol in enumerate(symbols):
            element_data = self.element_data[symbol.lower()]
            valid_mask = np.abs(element_data.coeffs_s) > 1e-12
            if np.any(valid_mask):
                all_coeffs.append(element_data.coeffs_s[valid_mask])
                all_exps.append(element_data.exps_s[valid_mask])
                all_coeffs[-1] *= np.sqrt(all_exps[-1] / np.pi) ** 3
                all_coords.append(coords[i])

        coeffs_padded, exps_padded = deps["_pad_coeffs_exps_with_zeros"](all_coeffs, all_exps)
        return deps["_PromolParams"](
            c_m=coeffs_padded, e_m=exps_padded, coords=np.array(all_coords), dim=3
        )
