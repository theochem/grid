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


# Test systems data: (name, symbols, coords, expected_electrons, fchk_file)
SYSTEMS_DATA = [
    ("Helium", ["he"], [[0, 0, 0]], 2, "data/He_t.fchk"),
    ("H2", ["h", "h"], [[0, 0, -0.71], [0, 0, 0.71]], 2, "data/H2.fchk"),
    ("HCl", ["h", "cl"], [[0, 0, -1.27], [0, 0, 1.27]], 18, "data/HCl.fchk"),
    (
        "CH4",
        ["c", "h", "h", "h", "h"],
        [
            [0, 0, 0],
            [1.085, 1.085, 1.085],
            [1.085, -1.085, -1.085],
            [-1.085, 1.085, -1.085],
            [-1.085, -1.085, 1.085],
        ],
        10,
        "data/CH4.fchk",
    ),
    (
        "H2O",
        ["o", "h", "h"],
        [[0, 0, 0], [1.431, 0, 1.107], [-1.431, 0, 1.107]],
        10,
        "data/h2o.fchk",
    ),
]


class System:
    """Molecular system data container"""

    def __init__(self, name, symbols, coords, expected_electrons, fchk_file):
        self.name = name
        self.symbols = symbols
        self.coords = np.array(coords)
        self.expected_electrons = expected_electrons
        self.fchk_file = fchk_file


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
        start_time = time.time()

        # Create uniform grid
        grid = deps["UniformGrid"].from_molecule(
            mol_data["mol_data"].atnums,
            mol_data["mol_data"].atcoords,
            spacing=0.2,
            extension=5.0,
            rotate=True,
            weight="Trapezoid",
        )

        # Evaluate and integrate electron density
        rho = deps["evaluate_density"](mol_data["rdm"], mol_data["ao_basis"], grid.points)
        electrons = grid.integrate(rho)
        total_time = time.time() - start_time

        rel_error = abs(electrons - system.expected_electrons) / system.expected_electrons

        return {
            "electrons": electrons,
            "rel_error": rel_error,
            "time": total_time,
            "points": grid.size,
        }, "Success"

    except Exception as e:
        return None, f"UniformGrid error: {str(e)}"


def create_promol_params(bfit_processor, symbols, coords, deps):
    """Create promolecular parameters from BFit data"""
    all_coeffs, all_exps, all_coords = [], [], []

    for i, symbol in enumerate(symbols):
        element_data = bfit_processor.element_data[symbol.lower()]
        valid_mask = np.abs(element_data.coeffs_s) > 1e-12
        if np.any(valid_mask):
            all_coeffs.append(element_data.coeffs_s[valid_mask])
            all_exps.append(element_data.exps_s[valid_mask])
            all_coords.append(coords[i])

    coeffs_padded, exps_padded = deps["_pad_coeffs_exps_with_zeros"](all_coeffs, all_exps)
    return deps["_PromolParams"](
        c_m=coeffs_padded, e_m=exps_padded, coords=np.array(all_coords), dim=3
    )


def create_normalized_promol_params(system, bfit_processor, deps):
    """Create normalized promolecular parameters for a molecular system"""
    # Check element availability
    missing = [s for s in system.symbols if s.lower() not in bfit_processor.element_data]
    if missing:
        raise ValueError(f"Missing elements: {missing}")

    # Create initial promolecular parameters
    promol_params = create_promol_params(bfit_processor, system.symbols, system.coords, deps)

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
        start_time = time.time()

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

        # Evaluate and integrate electron density
        rho = deps["evaluate_density"](mol_data["rdm"], mol_data["ao_basis"], transform.points)
        electrons = transform.integrate(rho)
        total_time = time.time() - start_time

        rel_error = abs(electrons - system.expected_electrons) / system.expected_electrons

        return {
            "electrons": electrons,
            "rel_error": rel_error,
            "time": total_time,
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
    print(f"\n{'='*100}")
    print("ELECTRON DENSITY INTEGRATION COMPARISON")
    print(f"{'='*100}")
    print(f"{'System':<8}  {'UniformGrid':<40} {'CubicProTransform':<40}")
    print(
        f"{'':>6} {'Expected':<8} {'Electrons':>10} {'Error%':>8} {'Time(s)':>8} {'Points':>10} "
        f"{'Electrons':>10} {'Error%':>8} {'Time(s)':>8} {'Points':>10}"
    )
    print("-" * 100)

    for system in systems:
        result = results.get(system.name, {})
        uniform = result.get("uniform")
        transform = result.get("transform")

        # Format results
        if uniform:
            u_data = (
                f"{uniform['electrons']:10.3f}{uniform['rel_error']*100:8.4f}"
                f"{uniform['time']:8.2f}{uniform['points']:10d}"
            )
        else:
            u_data = f"{'N/A':>36}"

        if transform:
            t_data = (
                f"{transform['electrons']:10.3f}{transform['rel_error']*100:8.4f}"
                f"{transform['time']:8.2f}{transform['points']:10d}"
            )
        else:
            t_data = f"{'N/A':>36}"

        print(f"{system.name:<8} {system.expected_electrons:<8} {u_data} {t_data}")


def test_grid_sizes(system, bfit_processor, deps, grid_sizes=[8, 10, 12, 15, 18, 20, 25]):
    """Test CubicProTransform performance with different grid sizes"""
    mol_data = load_molecular_data(system.fchk_file, deps)
    if not mol_data:
        return None

    try:
        # Create normalized promolecular parameters (shared across all grid sizes)
        normalized_promol = create_normalized_promol_params(system, bfit_processor, deps)
    except ValueError:
        return None

    results = {}

    # Test each grid size
    for grid_size in grid_sizes:
        try:
            start_time = time.time()

            # Create grid with specified size
            oned_grid = deps["GaussChebyshev"](grid_size)
            transform = deps["CubicProTransform"](
                [oned_grid, oned_grid, oned_grid],
                normalized_promol.c_m,
                normalized_promol.e_m,
                normalized_promol.coords,
            )

            # Evaluate and integrate electron density
            rho = deps["evaluate_density"](mol_data["rdm"], mol_data["ao_basis"], transform.points)
            electrons = transform.integrate(rho)

            total_time = time.time() - start_time
            rel_error = abs(electrons - system.expected_electrons) / system.expected_electrons

            results[grid_size] = {
                "electrons": electrons,
                "rel_error": rel_error,
                "time": total_time,
                "points": transform.size,
            }
        except Exception:
            continue  # Skip failed grid sizes

    return results


def print_grid_size_results(system_name, results):
    """Print grid size test results in tabular format"""
    print(f"\n{system_name} - Transform Grid Size Performance:")
    print(f"{'Grid':<8} {'Points':<10} {'Electrons':<10} {'Error%':<10} {'Time(s)':<8}")
    print("-" * 52)

    for grid_size in sorted(results.keys()):
        result = results[grid_size]
        print(
            f"{grid_size}³{'':<5} {result['points']:<10} {result['electrons']:<10.3f} "
            f"{result['rel_error']*100:<10.4f} {result['time']:<8.3f}"
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

    # Create test systems
    systems = [System(*data) for data in SYSTEMS_DATA]
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
                f"({uniform_result['rel_error']*100:.4f}% error, {uniform_result['time']:.2f}s)"
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
                    f"({transform_result['rel_error']*100:.4f}% error, {transform_result['time']:.2f}s)"
                )
            else:
                print(f"  Transform: {status}")

        results[system.name] = system_results

    # Print main comparison results
    print_results(systems, results)
    print("\nMain comparison completed!")

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
