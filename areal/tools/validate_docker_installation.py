#!/usr/bin/env python3
"""
Docker Installation Validation Script for AReaL

This script validates dependencies in the Docker environment, which includes
additional packages (grouped_gemm, apex, transformer_engine, flash_attn_3)
and a different flash-attn version (2.8.1 instead of 2.8.3).
"""

import sys
from pathlib import Path

from validation_base import BaseInstallationValidator


class DockerInstallationValidator(BaseInstallationValidator):
    """Validates installation in Docker environment with additional packages."""

    # Extend CUDA sub-modules with Docker-specific packages
    CUDA_SUBMODULES = {
        **BaseInstallationValidator.CUDA_SUBMODULES,
        "grouped_gemm": ["grouped_gemm"],
        "apex": ["apex.optimizers", "apex.normalization"],
        "transformer_engine": ["transformer_engine.pytorch"],
        "flash_attn_3": ["flash_attn_3"],
    }

    # Add Docker-specific packages to critical list
    CRITICAL_PACKAGES = {
        *BaseInstallationValidator.CRITICAL_PACKAGES,
        "grouped_gemm",
        "apex",
        "transformer_engine",
        "flash_attn_3",
    }

    def __init__(self, pyproject_path: Path | None = None):
        super().__init__(pyproject_path)

    def parse_pyproject(self):
        """Parse pyproject.toml and override flash-attn version for Docker."""
        super().parse_pyproject()

        # Override flash-attn version to match Docker environment (2.8.1)
        if "flash-attn" in self.dependencies:
            self.dependencies["flash-attn"]["version"] = "2.8.1"
            self.dependencies["flash-attn"]["operator"] = "=="
            self.dependencies["flash-attn"]["spec"] = "==2.8.1"
            self.dependencies["flash-attn"]["raw"] = "flash-attn==2.8.1"
            print(
                "  Note: Overriding flash-attn version to 2.8.1 for Docker environment"
            )

        # Add Docker-specific packages not in pyproject.toml
        self.add_additional_package("grouped_gemm", required=True)
        self.add_additional_package("apex", required=True)
        self.add_additional_package("transformer_engine", required=True)
        self.add_additional_package("flash_attn_3", required=False)

    def test_cuda_functionality(self):
        """Run CUDA functionality tests including Docker-specific packages."""
        super().test_cuda_functionality()

        print("\n=== Docker-Specific CUDA Tests ===")

        # Test transformer engine FP8 if available
        try:
            import torch

            if not torch.cuda.is_available():
                print("⚠ CUDA not available - skipping transformer engine tests")
                return

            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe

            # Set dimensions for a small test
            in_features = 128
            out_features = 256
            hidden_size = 64

            # Initialize model and inputs
            model = te.Linear(in_features, out_features, bias=True)
            inp = torch.randn(hidden_size, in_features, device="cuda")

            # Create an FP8 recipe
            fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

            # Enable autocasting for the forward pass
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = model(inp)

            loss = out.sum()
            loss.backward()
            print("✓ Transformer Engine FP8 operations")

        except ImportError:
            print("⚠ Transformer Engine not available - skipping FP8 tests")
        except Exception as e:
            print(f"⚠ Transformer Engine FP8 test failed: {e}")

        # Test Apex fused optimizers if available
        try:
            import torch
            from apex.optimizers import FusedAdam

            # Create a simple model and optimizer
            model = torch.nn.Linear(10, 10).cuda()
            optimizer = FusedAdam(model.parameters(), lr=0.001)

            # Test a forward-backward pass
            x = torch.randn(5, 10, device="cuda")
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            print("✓ Apex FusedAdam optimizer")

        except ImportError:
            print("⚠ Apex not available - skipping Apex tests")
        except Exception as e:
            print(f"⚠ Apex optimizer test failed: {e}")

        # Test flash_attn_3 if available
        try:
            import flash_attn_3  # noqa: F401

            print("✓ Flash Attention 3 (Hopper) imported successfully")
        except ImportError:
            print("⚠ Flash Attention 3 not available (optional for Hopper GPUs)")

        # Test grouped_gemm if available
        try:
            import grouped_gemm  # noqa: F401

            print("✓ Grouped GEMM imported successfully")
        except ImportError:
            print("⚠ Grouped GEMM not available")
        except Exception as e:
            print(f"⚠ Grouped GEMM test failed: {e}")

    def get_validation_title(self) -> str:
        """Get the title for validation output."""
        return "AReaL Docker Installation Validation"


def main():
    """Main entry point."""
    # Find pyproject.toml
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: pyproject.toml not found at {pyproject_path}")
        sys.exit(1)

    validator = DockerInstallationValidator(pyproject_path)
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
