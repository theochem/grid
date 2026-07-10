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
        from grid.protransform import CubicProTransform, _PromolParams, _pad_coeffs_exps_with_zeros, \
            PromolMolecule
        from grid.cubic import UniformGrid
        from grid.utils_promol import BFitDataProcessor
        from grid.utils_promol import ELEMENT_TO_ATOMIC_NUMBER

        # Create reverse mapping dynamically
        ATOMIC_NUMBER_TO_SYMBOL = {v: k for k, v in ELEMENT_TO_ATOMIC_NUMBER.items()}

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
            "PromolMolecule": PromolMolecule,
            "ATOMIC_NUMBER_TO_SYMBOL": ATOMIC_NUMBER_TO_SYMBOL,
        }
        print("All dependencies loaded successfully")
        return deps

    except ImportError as e:
        print(f"Import error: {e}")
        return None


def get_test_systems():
    """Find all available FCHK test files"""
    data_dir = Path(__file__).parent / "data"
    
    if not data_dir.exists() or not list(data_dir.glob("*.fchk")):
        examples_dir = Path(__file__).parent.parent.parent.parent / "examples"
        if examples_dir.exists():
            data_dir = examples_dir
        else:
            alt_examples = Path(__file__).parent.parent.parent / "examples"
            if alt_examples.exists():
                data_dir = alt_examples
    
    if data_dir.exists():
        fchk_files = list(data_dir.glob("*.fchk"))
        return [str(f) for f in fchk_files]
    
    return []


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

            # Convert atomic numbers to element symbols using dynamic mapping
            atomic_map = deps["ATOMIC_NUMBER_TO_SYMBOL"]
            self.symbols = [atomic_map.get(num, f"Z{num}").lower() for num in atnums]

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


def run_transform_grid_method(system, bfit_data_file, deps, grid_size=20):
    """Execute CubicProTransform integration method using PromolMolecule"""
    mol_data = load_molecular_data(system.fchk_file, deps)
    if not mol_data:
        return None, "Failed to load molecular data"

    try:
        # Measure grid construction time using PromolMolecule
        grid_start_time = time.time()

        # Create PromolMolecule instance
        promol_molecule = deps["PromolMolecule"](
            symbols=system.symbols,
            coords=system.coords,
            expected_electrons=system.expected_electrons,
            bfit_data_file=bfit_data_file
        )

        # Create transform grid
        transform = promol_molecule.create_transform_grid(grid_size=grid_size)
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


def initialize_bfit_data_file():
    """Get BFit data file path"""
    data_file = Path(__file__).parent / "data" / "result_kl_fpi_stype_only.npz"
    
    if not data_file.exists():
        examples_dir = Path(__file__).parent.parent.parent.parent / "examples"
        if not examples_dir.exists():
            examples_dir = Path(__file__).parent.parent.parent / "examples"
        
        data_file = examples_dir / "result_kl_fpi_stype_only.npz"
        
        if not data_file.exists() and examples_dir.exists():
            npz_files = list(examples_dir.glob("*.npz"))
            if npz_files:
                data_file = npz_files[0]
    
    if data_file.exists():
        return str(data_file)
    
    return None


def test_basic_functionality(deps):
    """Test basic functionality without requiring data files"""
    try:
        test_system = create_test_system()
        return "PromolMolecule" in deps
    except Exception:
        return False


def create_test_system():
    """Create a simple test system for basic functionality testing"""
    coords = np.array([[0.0, 0.0, 0.0]])
    symbols = ['h']
    expected_electrons = 1
    
    class MockSystem:
        def __init__(self):
            self.coords = coords
            self.symbols = symbols
            self.expected_electrons = expected_electrons
            self.name = "H_test"
            self.fchk_file = None
    
    return MockSystem()


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

    deps = setup_environment()
    if not deps:
        print("Failed to setup environment")
        return

    if not test_basic_functionality(deps):
        print("Basic functionality test failed. Stopping.")
        return

    bfit_data_file = initialize_bfit_data_file()
    if not bfit_data_file:
        print("Warning: BFit data file unavailable - CubicProTransform will be skipped")

    systems_data = get_test_systems()
    if not systems_data:
        print("No FCHK test systems found")
        return

    systems = []
    for fchk_file in systems_data:
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

    for i, system in enumerate(systems, 1):
        print(f"\n[{i}/{len(systems)}] {system.name}")
        system_results = {}

        uniform_result, status = run_uniform_grid_method(system, deps)
        if uniform_result:
            system_results["uniform"] = uniform_result
            print(
                f"  Uniform: {uniform_result['electrons']:.3f} electrons "
                f"({uniform_result['rel_error']*100:.4f}% error, {uniform_result['time']:.3f}s total)"
            )
        else:
            print(f"  Uniform: {status}")

        if bfit_data_file:
            transform_result, status = run_transform_grid_method(system, bfit_data_file, deps)
            if transform_result:
                system_results["transform"] = transform_result
                print(
                    f"  Transform: {transform_result['electrons']:.3f} electrons "
                    f"({transform_result['rel_error']*100:.4f}% error, {transform_result['time']:.3f}s total)"
                )
            else:
                print(f"  Transform: {status}")

        results[system.name] = system_results

    if results:
        print_results(systems, results)

        if bfit_data_file:
            print(f"\n{'='*60}")
            print("TRANSFORM GRID SIZE PERFORMANCE ANALYSIS")
            print(f"{'='*60}")

            for i, system in enumerate(systems, 1):
                print(f"\n[{i}/{len(systems)}] Testing {system.name} grid size performance...")
                grid_size_results = test_grid_sizes(system, bfit_data_file, deps)

                if grid_size_results:
                    print_grid_size_results(system.name, grid_size_results)


if __name__ == "__main__":
    main()
