# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import contextlib
import io
import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import List, Set

import setuptools
import torch
import torch.utils.cpp_extension as torch_cpp_ext
from packaging.version import Version, parse
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Supported NVIDIA GPU architectures.
NVIDIA_SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}


def _is_cuda() -> bool:
    return os.getenv("REAL_CUDA", "0") == "1"


# Compiler flags.
CXX_FLAGS = ["-g", "-O3", "-std=c++17"]
NVCC_FLAGS = ["-O3", "-std=c++17"]

if _is_cuda() and CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. In GPU environment, CUDA must be available to build the package."
    )

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]


def glob(pattern: str):
    root = Path(__name__).parent
    return [str(p) for p in root.glob(pattern)]


def get_pybind11_include_path() -> str:
    pybind11_meta = subprocess.check_output(
        "python3 -m pip show pybind11", shell=True
    ).decode("ascii")
    for line in pybind11_meta.split("\n"):
        line = line.strip()
        if line.startswith("Location: "):
            return os.path.join(line.split(": ")[1], "pybind11", "include")


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from
    https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_torch_arch_list() -> Set[str]:
    # TORCH_CUDA_ARCH_LIST can have one or more architectures,
    # e.g."8.0" or "7.5,8.0,8.6+PTX".Here, the "8.6+PTX" option asks the
    # compiler to additionally include PTX code that can be runtime - compiled
    # and executed on the 8.6 or newer architectures.While the PTX code will
    # not give the best performance on the newer architectures, it provides
    # forward compatibility.
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    # List are separated by; or space.
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = NVIDIA_SUPPORTED_ARCHS.union(
        {s + "+PTX" for s in NVIDIA_SUPPORTED_ARCHS}
    )
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}."
        )
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}.",
            stacklevel=2,
        )
    return arch_list


# First, check the TORCH_CUDA_ARCH_LIST environment variable.
compute_capabilities = get_torch_arch_list()

if _is_cuda() and not compute_capabilities:
    # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
    # GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            raise RuntimeError(
                "GPUs with compute capability below 7.0 are not supported."
            )
        compute_capabilities.add(f"{major}.{minor}")

ext_modules = []

if _is_cuda():
    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
    if not compute_capabilities:
        # If no GPU is specified nor available, add all supported architectures
        # based on the NVCC CUDA version.
        compute_capabilities = NVIDIA_SUPPORTED_ARCHS.copy()
        if nvcc_cuda_version < Version("11.1"):
            compute_capabilities.remove("8.6")
        if nvcc_cuda_version < Version("11.8"):
            compute_capabilities.remove("8.9")
            compute_capabilities.remove("9.0")
    # Validate the NVCC CUDA version.
    if nvcc_cuda_version < Version("11.0"):
        raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
    if nvcc_cuda_version < Version("11.1") and any(
        cc.startswith("8.6") for cc in compute_capabilities
    ):
        raise RuntimeError(
            "CUDA 11.1 or higher is required for compute capability 8.6."
        )
    if nvcc_cuda_version < Version("11.8"):
        if any(cc.startswith("8.9") for cc in compute_capabilities):
            # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
            # However, GPUs with compute capability 8.9 can also run the code generated by
            # the previous versions of CUDA 11 and targeting compute capability 8.0.
            # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
            # instead of 8.9.
            warnings.warn(
                "CUDA 11.8 or higher is required for compute capability 8.9. "
                "Targeting compute capability 8.0 instead.",
                stacklevel=2,
            )
            compute_capabilities = set(
                cc for cc in compute_capabilities if not cc.startswith("8.9")
            )
            compute_capabilities.add("8.0+PTX")
        if any(cc.startswith("9.0") for cc in compute_capabilities):
            raise RuntimeError(
                "CUDA 11.8 or higher is required for compute capability 9.0."
            )

    NVCC_FLAGS_PUNICA = NVCC_FLAGS.copy()

    # Add target compute capabilities to NVCC flags.
    for capability in compute_capabilities:
        num = capability[0] + capability[2]
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]
        if int(capability[0]) >= 8:
            NVCC_FLAGS_PUNICA += [
                "-gencode",
                f"arch=compute_{num},code=sm_{num}",
            ]
            if capability.endswith("+PTX"):
                NVCC_FLAGS_PUNICA += [
                    "-gencode",
                    f"arch=compute_{num},code=compute_{num}",
                ]

    # Use NVCC threads to parallelize the build.
    if nvcc_cuda_version >= Version("11.2"):
        nvcc_threads = int(os.getenv("NVCC_THREADS", 8))
        num_threads = min(os.cpu_count(), nvcc_threads)
        NVCC_FLAGS += ["--threads", str(num_threads)]

    if nvcc_cuda_version >= Version("11.8"):
        NVCC_FLAGS += ["-DENABLE_FP8_E5M2"]

    # changes for punica kernels
    NVCC_FLAGS += torch_cpp_ext.COMMON_NVCC_FLAGS
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        with contextlib.suppress(ValueError):
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)

os.makedirs(os.path.join(ROOT_DIR, "realhf", "_C"), exist_ok=True)

no_ext = os.getenv("REAL_NO_EXT", "0") == "1"

if not no_ext and _is_cuda():
    gae_extension = CUDAExtension(
        name="realhf._C.cugae",
        sources=[
            "csrc/cugae/gae.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS
            + [
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ],
        },
        libraries=["cuda"],
    )
    ext_modules.append(gae_extension)

    interval_op_cuda = CUDAExtension(
        name="realhf._C.interval_op_cuda",
        sources=[
            "csrc/interval_op/interval_op.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
        libraries=["cuda"],
    )
    ext_modules.append(interval_op_cuda)

if not no_ext:
    interval_extension = setuptools.Extension(
        name="realhf._C.interval_op",
        sources=[
            "csrc/interval_op/interval_op.cpp",
        ],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-Wall",
            "-std=c++17",
        ],
        include_dirs=[
            get_pybind11_include_path(),
        ],
    )
    ext_modules.append(interval_extension)


setuptools.setup(
    name="realhf",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "": [
            "csrc/**/*.cu",
            "csrc/**/*.cuh",
            "csrc/**/*.hpp",
            "csrc/**/*.cpp",
        ],
    },
)
