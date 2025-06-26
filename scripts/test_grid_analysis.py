#!/usr/bin/env python3
"""
Grid Analysis Test Suite
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_functionality():
    """Test core functionality of GBasis and BFit integrators"""
    print("Functionality Test")
    print("=" * 30)
    
    # GBasis molecular systems
    try:
        from gbasis_cubicpro_integration import GBasisCubicProIntegrator
        
        molecules = [
            ("data/He_t.fchk", "He", 2),
            ("data/H2.fchk", "H2", 2),
            ("data/h2o.fchk", "H2O", 10),
            ("data/HCl.fchk", "HCl", 18),
            ("data/CH4.fchk", "CH4", 10)
        ]
        
        integrator = GBasisCubicProIntegrator(grid_size=15)
        if integrator.available:
            print("GBasis Results:")
            results = []
            
            for fchk_file, name, expected in molecules:
                if Path(fchk_file).exists():
                    result = integrator.integrate_fchk(fchk_file)
                    results.append(result)
                    error = result.relative_error * 100
                    print(f"  {name}: {result.calculated_electrons:.3f}/{expected} electrons, {error:.2f}% error")
            
            if results:
                avg_error = np.mean([r.relative_error for r in results]) * 100
                print(f"  Average error: {avg_error:.2f}%")
                assert avg_error < 5.0, "Average error should be reasonable"
    
    except ImportError:
        print("GBasis not available")
    
    # BFit elements
    try:
        from bfit_cubicpro_integration import BFitCubicProIntegrator
        
        integrator = BFitCubicProIntegrator(grid_size=15)
        elements = ['H', 'He', 'Li', 'C', 'N', 'O']
        available_elements = [e for e in elements if e.lower() in integrator.bfit_processor.element_data]
        
        print("BFit Results:")
        for element in available_elements:
            result = integrator.integrate_promolecular_density(element)
            print(f"  {element}: {result.integral_value:.3f} integral, {result.basis_functions} functions")
        
        assert len(available_elements) >= 3, "Should have at least 3 elements"
    
    except (ImportError, FileNotFoundError):
        print("BFit not available")


def test_performance():
    """Test grid size performance and boundary conditions"""
    print("Performance Test")
    print("=" * 30)
    
    try:
        from gbasis_cubicpro_integration import GBasisCubicProIntegrator
        
        if not Path("data/h2o.fchk").exists():
            pytest.skip("H2O file not found")
        
        # Standard grid sizes
        grid_sizes = [8, 10, 12, 15, 18, 20, 25]
        
        # Boundary conditions
        boundary_sizes = [5, 50]  # Extreme small and large
        
        all_sizes = boundary_sizes + grid_sizes
        results = []
        
        print("Grid Performance (H2O):")
        
        for size in all_sizes:
            try:
                integrator = GBasisCubicProIntegrator(grid_size=size)
                if not integrator.available:
                    continue
                
                start_time = time.perf_counter()
                result = integrator.integrate_fchk("data/h2o.fchk")
                total_time = time.perf_counter() - start_time
                
                results.append({
                    'size': size,
                    'points': result.grid_points,
                    'time': total_time,
                    'error': result.relative_error * 100,
                    'electrons': result.calculated_electrons
                })
                
                boundary_marker = " (boundary)" if size in boundary_sizes else ""
                print(f"  Grid {size}³{boundary_marker}: {result.grid_points:,} points, "
                      f"{total_time:.2f}s, {result.relative_error*100:.2f}% error")
                
            except Exception as e:
                print(f"  Grid {size}³: Failed - {str(e)[:50]}")
        
        if len(results) >= 3:
            # Performance analysis
            times = [r['time'] for r in results]
            errors = [r['error'] for r in results]
            
            print(f"Performance Summary:")
            print(f"  Time range: {min(times):.2f}s - {max(times):.2f}s")
            print(f"  Error range: {min(errors):.2f}% - {max(errors):.2f}%")
            
            # Basic performance assertions
            assert max(times) < 300, "No grid should take more than 5 minutes"
            assert min(errors) < 10, "Best grid should have reasonable error"
    
    except ImportError:
        print("GBasis not available for performance test")


if __name__ == "__main__":
    print("Grid Analysis Test Suite")
    print("=" * 50)
    
    try:
        test_functionality()
        print("✓ Functionality test passed")
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
    
    try:
        test_performance()
        print("✓ Performance test passed")
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
    
    print("=" * 50)
    print("Test completed") 