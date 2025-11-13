"""
Base validation module for AReaL installation validation.

This module provides a base class with common validation logic that can be
extended for different validation scenarios (standard installation, Docker, etc.).
"""

import importlib
import re
import sys
from importlib.metadata import version as get_version
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for Python 3.10
    except ImportError:
        print("Error: tomllib/tomli not available. Install tomli: pip install tomli")
        sys.exit(1)

from packaging.specifiers import SpecifierSet
from packaging.version import Version


class BaseInstallationValidator:
    """Base class for validating installation dependencies."""

    # Map package names to their import names (when different)
    PACKAGE_IMPORT_MAP = {
        "hydra-core": "hydra",
        "megatron-core": "megatron",
        "PyYAML": "yaml",
        "python-dateutil": "dateutil",
        "python_dateutil": "dateutil",
        "pyzmq": "zmq",
        "nvidia-ml-py": "pynvml",
        "camel-ai": "camel",
        "python-debian": "debian",
        "python_debian": "debian",
        "openai-agents": "openai",
        "tensorboardx": "tensorboardX",
    }

    # Map packages to their CUDA sub-modules for deep validation
    # Subclasses can override or extend this
    CUDA_SUBMODULES = {
        "torch": ["torch.cuda"],
        "sglang": ["sgl_kernel", "sgl_kernel.flash_attn"],
        "vllm": ["vllm._C"],
        "flash-attn": ["flash_attn_2_cuda"],
        "megatron-core": [
            "megatron.core.parallel_state",
            "megatron.core.tensor_parallel",
        ],
    }

    # Packages to treat as critical (always fail if missing)
    # Subclasses can override this
    CRITICAL_PACKAGES = {
        "torch",
        "transformers",
        "flash-attn",
        "sglang",
        "megatron-core",
        "mbridge",
        "ray",
        "datasets",
        "hydra-core",
        "omegaconf",
        "wandb",
        "fastapi",
        "uvicorn",
    }

    def __init__(self, pyproject_path: Path | None = None):
        self.pyproject_path = pyproject_path
        self.dependencies = {}
        self.additional_packages = {}  # For packages not in pyproject.toml
        self.results = {}
        self.critical_failures = []
        self.warnings = []

    def parse_pyproject(self):
        """Parse pyproject.toml and extract dependencies."""
        if self.pyproject_path is None:
            print("No pyproject.toml path provided, skipping dependency parsing")
            return

        try:
            with open(self.pyproject_path, "rb") as f:
                data = tomllib.load(f)

            raw_deps = data.get("project", {}).get("dependencies", [])

            for dep in raw_deps:
                # Parse dependency string: "package==version" or "package>=version" etc.
                # Handle various formats: package, package==version, package>=version, package[extras]
                match = re.match(
                    r"^([a-zA-Z0-9_-]+)(\[.+\])?(([<>=!]+)([0-9.]+[a-z0-9.]*))?",
                    dep.strip(),
                )
                if match:
                    pkg_name = match.group(1)
                    extras = match.group(2) or ""
                    operator = match.group(4) or ""
                    version = match.group(5) or ""

                    # Store package info
                    self.dependencies[pkg_name] = {
                        "raw": dep,
                        "extras": extras,
                        "operator": operator,
                        "version": version,
                        "spec": f"{operator}{version}" if operator else "",
                    }

            print(f"Parsed {len(self.dependencies)} dependencies from pyproject.toml")

        except FileNotFoundError:
            print(f"Error: {self.pyproject_path} not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing pyproject.toml: {e}")
            sys.exit(1)

    def add_additional_package(
        self, pkg_name: str, version_spec: str = "", required: bool = True
    ):
        """Add a package that's not in pyproject.toml for validation."""
        operator = ""
        version = ""

        if version_spec:
            # Parse version spec like "==2.8.1" or ">=1.0.0"
            match = re.match(r"^([<>=!]+)?([0-9.]+[a-z0-9.]*)$", version_spec.strip())
            if match:
                operator = match.group(1) or "=="
                version = match.group(2)

        self.additional_packages[pkg_name] = {
            "raw": f"{pkg_name}{version_spec}",
            "extras": "",
            "operator": operator,
            "version": version,
            "spec": f"{operator}{version}" if operator else "",
            "required": required,
        }

    def normalize_package_name(self, pkg_name: str) -> str:
        """Normalize package name (handle dash/underscore differences)."""
        # PyPI normalizes package names: replace _ with - and lowercase
        return pkg_name.lower().replace("_", "-")

    def get_installed_version(self, pkg_name: str) -> str | None:
        """Get installed version of a package."""
        # Try both dash and underscore variants
        variants = [
            pkg_name,
            pkg_name.replace("-", "_"),
            pkg_name.replace("_", "-"),
        ]

        for variant in variants:
            try:
                return get_version(variant)
            except Exception:
                continue

        return None

    def check_version(self, pkg_name: str, spec_str: str) -> tuple[bool, str]:
        """
        Check if installed version matches the specification.

        Returns:
            (matches: bool, message: str)
        """
        installed = self.get_installed_version(pkg_name)

        if installed is None:
            return False, "Package not found in installation"

        if not spec_str:
            return True, f"Installed: {installed} (no version constraint)"

        try:
            spec = SpecifierSet(spec_str)
            installed_ver = Version(installed)

            if installed_ver in spec:
                return True, f"Installed: {installed} (matches {spec_str})"
            else:
                return (
                    False,
                    f"Version mismatch: Expected {spec_str}, found {installed}",
                )

        except Exception as e:
            return False, f"Version check error: {e}"

    def test_import(self, pkg_name: str, dep_info: dict, required: bool = True) -> bool:
        """Test importing a package and check its version."""
        # Get the import name (may differ from package name)
        import_name = self.PACKAGE_IMPORT_MAP.get(pkg_name, pkg_name.replace("-", "_"))

        try:
            # Import the package
            module = importlib.import_module(import_name)

            # Check version
            version_ok, version_msg = self.check_version(
                pkg_name, dep_info.get("spec", "")
            )

            if not version_ok:
                self.results[pkg_name] = {
                    "status": "VERSION_MISMATCH",
                    "error": version_msg,
                }
                if required:
                    self.critical_failures.append(f"{pkg_name}: {version_msg}")
                    print(f"✗ {pkg_name} (VERSION MISMATCH): {version_msg}")
                else:
                    self.warnings.append(f"{pkg_name}: {version_msg}")
                    print(f"⚠ {pkg_name} (VERSION MISMATCH): {version_msg}")
                return False

            # Test CUDA sub-modules if applicable
            if pkg_name in self.CUDA_SUBMODULES:
                self.test_cuda_submodules(pkg_name, module)

            self.results[pkg_name] = {"status": "SUCCESS", "error": None}
            print(f"✓ {pkg_name} - {version_msg}")
            return True

        except ImportError as e:
            self.results[pkg_name] = {"status": "IMPORT_FAILED", "error": str(e)}
            if required:
                self.critical_failures.append(f"{pkg_name}: Import failed - {str(e)}")
                print(f"✗ {pkg_name} (IMPORT FAILED): {str(e)}")
            else:
                self.warnings.append(f"{pkg_name}: Import failed - {str(e)}")
                print(f"⚠ {pkg_name} (IMPORT FAILED): {str(e)}")
            return False

        except Exception as e:
            self.results[pkg_name] = {"status": "ERROR", "error": str(e)}
            if required:
                self.critical_failures.append(f"{pkg_name}: {str(e)}")
                print(f"✗ {pkg_name} (ERROR): {str(e)}")
            else:
                self.warnings.append(f"{pkg_name}: {str(e)}")
                print(f"⚠ {pkg_name} (ERROR): {str(e)}")
            return False

    def test_cuda_submodules(self, pkg_name: str, module):
        """Test CUDA sub-modules for packages with CUDA dependencies."""
        submodules = self.CUDA_SUBMODULES.get(pkg_name, [])

        for submodule_name in submodules:
            try:
                # Special handling for torch.cuda
                if submodule_name == "torch.cuda":
                    if not module.cuda.is_available():
                        raise RuntimeError("CUDA is not available in PyTorch")
                    print(
                        f"  ├─ CUDA devices: {module.cuda.device_count()}, "
                        f"version: {module.version.cuda}"
                    )
                else:
                    # Import sub-module
                    importlib.import_module(submodule_name)
                    print(f"  ├─ {submodule_name} ✓")

            except Exception as e:
                print(f"  ├─ {submodule_name} ✗ ({str(e)})")
                self.warnings.append(
                    f"{pkg_name} CUDA extension ({submodule_name}): {str(e)}"
                )

    def validate_all_dependencies(self):
        """Validate all dependencies from pyproject.toml."""
        print("\n" + "=" * 70)
        print(self.get_validation_title())
        print("=" * 70)

        # Validate pyproject.toml dependencies
        if self.dependencies:
            print("\n=== Critical Dependencies ===")
            critical_count = 0
            for pkg_name, dep_info in sorted(self.dependencies.items()):
                normalized_name = self.normalize_package_name(pkg_name)
                is_critical = normalized_name in self.CRITICAL_PACKAGES

                if is_critical:
                    critical_count += 1
                    self.test_import(pkg_name, dep_info, required=True)

            print(
                f"\n=== Other Dependencies ({len(self.dependencies) - critical_count}) ==="
            )
            for pkg_name, dep_info in sorted(self.dependencies.items()):
                normalized_name = self.normalize_package_name(pkg_name)
                is_critical = normalized_name in self.CRITICAL_PACKAGES

                if not is_critical:
                    self.test_import(pkg_name, dep_info, required=False)

        # Validate additional packages
        if self.additional_packages:
            print(f"\n=== Additional Packages ({len(self.additional_packages)}) ===")
            for pkg_name, dep_info in sorted(self.additional_packages.items()):
                required = dep_info.get("required", True)
                self.test_import(pkg_name, dep_info, required=required)

    def test_cuda_functionality(self):
        """Run basic CUDA functionality tests. Can be overridden by subclasses."""
        print("\n=== CUDA Functionality Tests ===")

        try:
            import torch

            if not torch.cuda.is_available():
                print("⚠ CUDA not available - skipping CUDA tests")
                return

            # Test basic CUDA operations
            try:
                device = torch.device("cuda:0")
                x = torch.randn(10, device=device)
                y = torch.randn(10, device=device)
                _ = x + y
                print("✓ Basic CUDA tensor operations")
            except Exception as e:
                print(f"✗ Basic CUDA operations failed: {e}")

            # Test flash attention if available
            try:
                from flash_attn import flash_attn_func

                batch_size, seq_len, num_heads, head_dim = 1, 32, 4, 64
                q = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float16,
                )
                k = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float16,
                )
                v = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float16,
                )
                _ = flash_attn_func(q, k, v)
                print("✓ Flash attention CUDA operations")
            except Exception as e:
                print(f"⚠ Flash attention test failed: {e}")

        except ImportError:
            print("⚠ PyTorch not available - skipping CUDA tests")

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        total_tests = len(self.results)
        successful_tests = sum(
            1 for r in self.results.values() if r["status"] == "SUCCESS"
        )
        failed_tests = total_tests - successful_tests

        print(f"Total packages tested: {total_tests}")
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
            print("\n❌ INSTALLATION VALIDATION FAILED")
            print("Please fix the critical failures above and ensure all required")
            print("dependencies are properly installed.")
            return False
        else:
            print("\n✅ INSTALLATION VALIDATION PASSED")
            if self.warnings:
                print("Note: Some warnings were reported but core functionality")
                print("should not be affected.")
            return True

    def get_validation_title(self) -> str:
        """Get the title for validation output. Can be overridden by subclasses."""
        return "AReaL Installation Validation"

    def run(self):
        """Run the complete validation process."""
        self.parse_pyproject()
        self.validate_all_dependencies()
        self.test_cuda_functionality()
        success = self.print_summary()
        return success
