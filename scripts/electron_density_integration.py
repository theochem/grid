#!/usr/bin/env python3
"""
Electron Density Integration Test

Compares two grid integration methods for molecular systems:
1. UniformGrid - Traditional equal-spacing cubic grid
2. CubicProTransform - Adaptive grid guided by promolecular density

Test systems: He, H2, HCl, CH4, H2O
"""

import os
import sys
import time
from pathlib import Path

import numpy as np


def setup_environment():
    """Setup environment and import all dependencies"""
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    sys.path.append(str(Path(__file__).parent))

    try:
        from iodata import load_one
        from gbasis.wrappers import from_iodata
        from gbasis.evals.density import evaluate_density
        from grid.onedgrid import GaussChebyshev
        from grid.protransform import CubicProTransform, _PromolParams, _pad_coeffs_exps_with_zeros
        from grid.cubic import UniformGrid
        from bfit_data_processor import BFitDataProcessor

        deps = {
            "load_one": load_one,
            "from_iodata": from_iodata,
            "evaluate_density": evaluate_density,
            "GaussChebyshev": GaussChebyshev,
            "CubicProTransform": CubicProTransform,
            "_PromolParams": _PromolParams,
            "UniformGrid": UniformGrid,
            "_pad_coeffs_exps_with_zeros": _pad_coeffs_exps_with_zeros,
            "BFitDataProcessor": BFitDataProcessor,
        }
        print("All dependencies loaded successfully")
        return deps

    except ImportError as e:
        print(f"Import error: {e}")
        return None


# Test systems: FCHK file paths for molecular systems
SYSTEMS_DATA = [
    "data/He_t.fchk",
    "data/H2.fchk",
    "data/HCl.fchk",
    "data/CH4.fchk",
    "data/h2o.fchk",
]

# Atomic number to element symbol mapping
ATOMIC_NUMBER_TO_SYMBOL = {
    1: "h",
    2: "he",
    3: "li",
    4: "be",
    5: "b",
    6: "c",
    7: "n",
    8: "o",
    9: "f",
    10: "ne",
    11: "na",
    12: "mg",
    13: "al",
    14: "si",
    15: "p",
    16: "s",
    17: "cl",
    18: "ar",
    19: "k",
    20: "ca",
}


class System:
    """Molecular system data container that loads information from FCHK files"""

    def __init__(self, fchk_file, deps):
        self.fchk_file = fchk_file
        self._load_from_fchk(deps)

    def _load_from_fchk(self, deps):
        """Load molecular information from FCHK file using iodata"""
        if not os.path.exists(self.fchk_file):
            raise FileNotFoundError(f"FCHK file not found: {self.fchk_file}")

        try:
            # Load molecular data
            mol_data = deps["load_one"](self.fchk_file)

            # Extract information
            self.coords = mol_data.atcoords  # Atomic coordinates
            atnums = mol_data.atnums  # Atomic numbers

            # Convert atomic numbers to element symbols
            self.symbols = [ATOMIC_NUMBER_TO_SYMBOL.get(num, f"Z{num}").lower() for num in atnums]

            # Calculate expected number of electrons
            self.expected_electrons = int(np.sum(atnums))

            # Extract system name from title or filename
            if hasattr(mol_data, "title") and mol_data.title:
                self.name = mol_data.title.strip()
            else:
                # Use filename without extension as fallback
                self.name = Path(self.fchk_file).stem

        except Exception as e:
            raise RuntimeError(f"Failed to load FCHK file {self.fchk_file}: {e}")

    def __str__(self):
        return f"{self.name}: {len(self.symbols)} atoms, {self.expected_electrons} electrons"


def load_molecular_data(fchk_file, deps):
    """Load molecular data from FCHK file"""
    if not os.path.exists(fchk_file):
        return None

    try:
        mol_data = deps["load_one"](fchk_file)
        ao_basis = deps["from_iodata"](mol_data)
        rdm = mol_data.one_rdms.get("scf") if hasattr(mol_data, "one_rdms") else None

        if rdm is None:
            return None

        return {"mol_data": mol_data, "ao_basis": ao_basis, "rdm": rdm}
    except Exception:
        return None


def run_uniform_grid_method(system, deps):
    """Execute UniformGrid integration method"""
    mol_data = load_molecular_data(system.fchk_file, deps)
    if not mol_data:
        return None, "Failed to load molecular data"

    try:
        # Measure grid construction time
        grid_start_time = time.time()
        grid = deps["UniformGrid"].from_molecule(
            mol_data["mol_data"].atnums,
            mol_data["mol_data"].atcoords,
            spacing=0.2,
            extension=5.0,
            rotate=True,
            weight="Trapezoid",
        )
        grid_construction_time = time.time() - grid_start_time

        # Measure electron density evaluation and integration time
        eval_start_time = time.time()
        rho = deps["evaluate_density"](mol_data["rdm"], mol_data["ao_basis"], grid.points)
        electrons = grid.integrate(rho)
        eval_integration_time = time.time() - eval_start_time

        total_time = grid_construction_time + eval_integration_time
        rel_error = abs(electrons - system.expected_electrons) / system.expected_electrons

        return {
            "electrons": electrons,
            "rel_error": rel_error,
            "time": total_time,
            "grid_time": grid_construction_time,
            "eval_time": eval_integration_time,
            "points": grid.size,
        }, "Success"

    except Exception as e:
        return None, f"UniformGrid error: {str(e)}"


def create_normalized_promol_params(system, bfit_processor, deps):
    """Create normalized promolecular parameters for a molecular system"""
    # Check element availability
    missing = [s for s in system.symbols if s.lower() not in bfit_processor.element_data]
    if missing:
        raise ValueError(f"Missing elements: {missing}")

    # Create initial promolecular parameters using BFitDataProcessor method
    promol_params = bfit_processor.create_promol_params(system.symbols, system.coords, deps)

    # Normalize to match expected electron count
    norm_factor = system.expected_electrons / promol_params.integrate_all()

    return deps["_PromolParams"](
        c_m=promol_params.c_m * norm_factor,
        e_m=promol_params.e_m,
        coords=promol_params.coords,
        dim=promol_params.dim,
    )


def run_transform_grid_method(system, bfit_processor, deps, grid_size=20):
    """Execute CubicProTransform integration method"""
    mol_data = load_molecular_data(system.fchk_file, deps)
    if not mol_data:
        return None, "Failed to load molecular data"

    try:
        # Measure grid construction time (promol params + transform grid)
        grid_start_time = time.time()

        # Create normalized promolecular parameters
        normalized_promol = create_normalized_promol_params(system, bfit_processor, deps)

        # Create transform grid
        oned_grid = deps["GaussChebyshev"](grid_size)
        transform = deps["CubicProTransform"](
            [oned_grid, oned_grid, oned_grid],
            normalized_promol.c_m,
            normalized_promol.e_m,
            normalized_promol.coords,
        )
        grid_construction_time = time.time() - grid_start_time

        # Measure electron density evaluation and integration time
        eval_start_time = time.time()
        rho = deps["evaluate_density"](mol_data["rdm"], mol_data["ao_basis"], transform.points)
        electrons = transform.integrate(rho)
        eval_integration_time = time.time() - eval_start_time

        total_time = grid_construction_time + eval_integration_time
        rel_error = abs(electrons - system.expected_electrons) / system.expected_electrons

        return {
            "electrons": electrons,
            "rel_error": rel_error,
            "time": total_time,
            "grid_time": grid_construction_time,
            "eval_time": eval_integration_time,
            "points": transform.size,
        }, "Success"

    except Exception as e:
        return None, f"CubicProTransform error: {str(e)}"


def initialize_bfit_processor(deps):
    """Initialize BFit processor for promolecular density parameters"""
    data_file = Path(__file__).parent / "data" / "result_kl_fpi_stype_only.npz"
    if not data_file.exists():
        return None

    processor = deps["BFitDataProcessor"](data_file)
    if processor.load_bfit_data():
        processor.parse_element_data()
        print(f"BFit data loaded: {data_file.name}")
        return processor
    return None


def print_results(systems, results):
    """Print comparison results in tabular format"""
    print(f"\n{'='*140}")
    print("ELECTRON DENSITY INTEGRATION COMPARISON")
    print(f"{'='*140}")
    print(f"{'System':<8}  {'UniformGrid':<65} {'CubicProTransform':<65}")
    print(
        f"{'':>6} {'Expected':<8} {'Electrons':>10} {'Error%':>8} {'Grid(s)':>8} {'Eval(s)':>8} {'Total(s)':>8} {'Points':>10} "
        f"{'Electrons':>10} {'Error%':>8} {'Grid(s)':>8} {'Eval(s)':>8} {'Total(s)':>8} {'Points':>10}"
    )
    print("-" * 140)

    for system in systems:
        result = results.get(system.name, {})
        uniform = result.get("uniform")
        transform = result.get("transform")

        # Format results
        if uniform:
            u_data = (
                f"{uniform['electrons']:10.3f}{uniform['rel_error']*100:8.4f}"
                f"{uniform['grid_time']:8.2f}{uniform['eval_time']:8.2f}{uniform['time']:8.2f}{uniform['points']:10d}"
            )
        else:
            u_data = f"{'N/A':>61}"

        if transform:
            t_data = (
                f"{transform['electrons']:10.3f}{transform['rel_error']*100:8.4f}"
                f"{transform['grid_time']:8.2f}{transform['eval_time']:8.2f}{transform['time']:8.2f}{transform['points']:10d}"
            )
        else:
            t_data = f"{'N/A':>61}"

        print(f"{system.name:<8} {system.expected_electrons:<8} {u_data} {t_data}")


def test_grid_sizes(system, bfit_processor, deps, grid_sizes=[8, 10, 12, 15, 18, 20, 25]):
    """Test CubicProTransform performance with different grid sizes"""
    results = {}

    # Test each grid size by calling the existing run_transform_grid_method
    for grid_size in grid_sizes:
        result, status = run_transform_grid_method(system, bfit_processor, deps, grid_size)
        if result is not None:  # Success case
            results[grid_size] = result
        # Skip failed grid sizes (when result is None)

    return results


def print_grid_size_results(system_name, results):
    """Print grid size test results in tabular format"""
    print(f"\n{system_name} - Transform Grid Size Performance:")
    print(
        f"{'Grid':<8} {'Points':<10} {'Electrons':<10} {'Error%':<10} {'Total(s)':<8} {'Grid(s)':<8} {'Eval(s)':<8}"
    )
    print("-" * 72)

    for grid_size in sorted(results.keys()):
        result = results[grid_size]
        print(
            f"{grid_size}^3{'':<5} {result['points']:<10} {result['electrons']:<10.3f} "
            f"{result['rel_error']*100:<10.4f} {result['time']:<8.3f} "
            f"{result['grid_time']:<8.3f} {result['eval_time']:<8.3f}"
        )


def main():
    """Main function - orchestrates the complete integration comparison workflow"""
    print("FIVE SYSTEMS ELECTRON DENSITY INTEGRATION TEST")
    print("=" * 50)

    # Environment setup
    deps = setup_environment()
    if not deps:
        print("Failed to setup environment")
        return

    # Initialize BFit processor for CubicProTransform
    bfit_processor = initialize_bfit_processor(deps)
    if not bfit_processor:
        print("Warning: BFit processor unavailable - CubicProTransform will be skipped")

    # Create test systems from FCHK files
    systems = []
    for fchk_file in SYSTEMS_DATA:
        try:
            system = System(fchk_file, deps)
            systems.append(system)
            print(f"Loaded: {system}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Skipped {fchk_file}: {e}")

    if not systems:
        print("No valid systems found. Exiting.")
        return

    results = {}

    print(f"\nTesting {len(systems)} molecular systems...")

    # Main integration comparison
    for i, system in enumerate(systems, 1):
        print(f"\n[{i}/5] {system.name}")
        system_results = {}

        # Test UniformGrid method
        uniform_result, status = run_uniform_grid_method(system, deps)
        if uniform_result:
            system_results["uniform"] = uniform_result
            print(
                f"  Uniform: {uniform_result['electrons']:.3f} electrons "
                f"({uniform_result['rel_error']*100:.4f}% error, {uniform_result['time']:.3f}s total)"
            )
            print(
                f"    Grid construction: {uniform_result['grid_time']:.3f}s, "
                f"Density evaluation: {uniform_result['eval_time']:.3f}s"
            )
        else:
            print(f"  Uniform: {status}")

        # Test CubicProTransform method
        if bfit_processor:
            transform_result, status = run_transform_grid_method(system, bfit_processor, deps)
            if transform_result:
                system_results["transform"] = transform_result
                print(
                    f"  Transform: {transform_result['electrons']:.3f} electrons "
                    f"({transform_result['rel_error']*100:.4f}% error, {transform_result['time']:.3f}s total)"
                )
                print(
                    f"    Grid construction: {transform_result['grid_time']:.3f}s, "
                    f"Density evaluation: {transform_result['eval_time']:.3f}s"
                )
            else:
                print(f"  Transform: {status}")

        results[system.name] = system_results

    # Print main comparison results
    print_results(systems, results)

    # Grid size performance analysis
    if bfit_processor:
        print(f"\n{'='*60}")
        print("TRANSFORM GRID SIZE PERFORMANCE ANALYSIS")
        print(f"{'='*60}")

        # Test grid size performance for all molecular systems
        for i, system in enumerate(systems, 1):
            print(f"\n[{i}/5] Testing {system.name} grid size performance...")
            grid_size_results = test_grid_sizes(system, bfit_processor, deps)

            if grid_size_results:
                print_grid_size_results(system.name, grid_size_results)


if __name__ == "__main__":
    main()
