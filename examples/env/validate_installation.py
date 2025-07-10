#!/usr/bin/env python3
"""
Installation Validation Script for AReaL

This script validates that all critical dependencies are properly installed
and can be imported successfully. It's designed to be run in CI to validate
the installation procedure described in docs/tutorial/installation.md.
"""

import importlib
import sys
import traceback
import warnings
from importlib.metadata import version as get_version
from typing import Any, Dict, List, Optional

from packaging.version import Version


class InstallationValidator:
    def __init__(self):
        self.results = {}
        self.critical_failures = []
        self.warnings = []

    def test_import(self, module_name: str, required: bool = True, 
                   test_func: Optional[callable] = None) -> bool:
        """Test importing a module and optionally run additional tests."""
        try:
            module = importlib.import_module(module_name)
            
            # Run additional test if provided
            if test_func:
                test_func(module)
                
            self.results[module_name] = {"status": "SUCCESS", "error": None}
            print(f"✓ {module_name}")
            return True
            
        except ImportError as e:
            self.results[module_name] = {"status": "FAILED", "error": str(e)}
            if required:
                self.critical_failures.append(f"{module_name}: {str(e)}")
                print(f"✗ {module_name} (CRITICAL): {str(e)}")
            else:
                self.warnings.append(f"{module_name}: {str(e)}")
                print(f"⚠ {module_name} (OPTIONAL): {str(e)}")
            return False
            
        except Exception as e:
            self.results[module_name] = {"status": "ERROR", "error": str(e)}
            if required:
                self.critical_failures.append(f"{module_name}: {str(e)}")
                print(f"✗ {module_name} (CRITICAL ERROR): {str(e)}")
            else:
                self.warnings.append(f"{module_name}: {str(e)}")
                print(f"⚠ {module_name} (OPTIONAL ERROR): {str(e)}")
            return False

    def test_torch_cuda(self, torch_module):
        """Test PyTorch CUDA availability."""
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is not available in PyTorch")
        print(f"  - CUDA devices: {torch_module.cuda.device_count()}")
        print(f"  - CUDA version: {torch_module.version.cuda}")

    def test_flash_attn_functionality(self, flash_attn_module):
        """Test flash attention functionality."""
        # Try to import key functions
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        print("  - Flash attention functions imported successfully")

    def test_vllm_functionality(self, vllm_module):
        """Test vLLM basic functionality."""
        from vllm import LLM, SamplingParams
        print("  - vLLM core classes imported successfully")

    def test_sglang_functionality(self, sglang_module):
        """Test SGLang basic functionality."""
        # Basic import test is sufficient for CI
        import sgl_kernel
        from sglang import launch_server
        assert Version(get_version("sglang")) == Version("0.4.6.post4")
        print("  - SGLang imported successfully")
    
    def test_transformers(self, transformers_module):
        assert Version(get_version("transformers")) == Version("4.51.1")
        print("  - transformers imported successfully")

    def validate_critical_dependencies(self):
        """Validate critical dependencies that must be present."""
        print("\n=== Testing Critical Dependencies ===")
        
        # Core ML frameworks
        self.test_import("torch", required=True, test_func=self.test_torch_cuda)
        self.test_import("transformers", required=True, test_func=self.test_transformers)
        
        # Flash attention - critical for performance
        self.test_import("flash_attn", required=True, test_func=self.test_flash_attn_functionality)
        self.test_import("cugae", required=True)
        # Inference engines
        self.test_import("sglang", required=True, test_func=self.test_sglang_functionality)
        
        # Distributed computing
        self.test_import("ray", required=True)
        
        # Scientific computing
        self.test_import("numpy", required=True)
        self.test_import("scipy", required=True)
        
        # Configuration management
        self.test_import("hydra", required=True)
        self.test_import("omegaconf", required=True)
        
        # Data processing
        self.test_import("datasets", required=True)
        self.test_import("pandas", required=True)
        self.test_import("einops", required=True)
        
        # Monitoring and logging
        self.test_import("wandb", required=True)
        self.test_import("pynvml", required=True)
        
        # Networking
        self.test_import("aiohttp", required=True)
        self.test_import("fastapi", required=True)
        self.test_import("uvicorn", required=True)
        
        # Math libraries (for evaluation)
        self.test_import("sympy", required=True)
        self.test_import("latex2sympy2", required=True)

    def validate_optional_dependencies(self):
        """Validate optional dependencies."""
        print("\n=== Testing Optional Dependencies ===")
        
        # CUDA extensions (may not be available in all environments)
        self.test_import("vllm", required=False, test_func=self.test_vllm_functionality)
        self.test_import("grouped_gemm", required=False)
        self.test_import("flashattn_hopper", required=False)
        
        # Optional utilities
        self.test_import("tensorboardx", required=False)
        self.test_import("swanlab", required=False)
        self.test_import("matplotlib", required=False)
        self.test_import("seaborn", required=False)
        self.test_import("numba", required=False)
        self.test_import("nltk", required=False)

    def validate_cuda_extensions(self):
        """Validate CUDA-specific functionality."""
        print("\n=== Testing CUDA Extensions ===")
        
        try:
            import torch
            if torch.cuda.is_available():
                # Test basic CUDA tensor operations
                device = torch.device("cuda:0")
                x = torch.randn(10, device=device)
                y = torch.randn(10, device=device)
                z = x + y
                print("✓ Basic CUDA operations working")
                
                # Test flash attention if available
                try:
                    from flash_attn import flash_attn_func

                    # Create small test tensors
                    batch_size, seq_len, num_heads, head_dim = 1, 32, 4, 64
                    q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                                  device=device, dtype=torch.float16)
                    k = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                                  device=device, dtype=torch.float16)
                    v = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                                  device=device, dtype=torch.float16)
                    
                    # Test flash attention call
                    out = flash_attn_func(q, k, v)
                    print("✓ Flash attention CUDA operations working")
                    
                except Exception as e:
                    print(f"⚠ Flash attention CUDA test failed: {e}")
                    
            else:
                print("⚠ CUDA not available - skipping CUDA extension tests")
                
        except Exception as e:
            print(f"✗ CUDA extension validation failed: {e}")

    def run_validation(self):
        """Run complete validation suite."""
        print("AReaL Installation Validation")
        print("=" * 50)
        
        self.validate_critical_dependencies()
        self.validate_optional_dependencies()
        self.validate_cuda_extensions()
        
        # Print summary
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["status"] == "SUCCESS")
        failed_tests = total_tests - successful_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        
        if self.critical_failures:
            print(f"\n🚨 CRITICAL FAILURES ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                print(f"  - {failure}")
                
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # Determine overall result
        if self.critical_failures:
            print(f"\n❌ INSTALLATION VALIDATION FAILED")
            print("Please check the critical failures above and ensure all required")
            print("dependencies are properly installed according to the installation guide.")
            return False
        else:
            print(f"\n✅ INSTALLATION VALIDATION PASSED")
            if self.warnings:
                print("Note: Some optional dependencies failed but this won't affect")
                print("core functionality.")
            return True

def main():
    """Main entry point."""
    validator = InstallationValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()