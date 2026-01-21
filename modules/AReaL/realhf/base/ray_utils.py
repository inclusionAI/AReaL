# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import os
import subprocess


def check_ray_availability():
    return (
        int(
            subprocess.run(
                ["ray", "--help"],
                stdout=open(os.devnull, "wb"),
                stderr=open(os.devnull, "wb"),
            ).returncode
        )
        == 0
    )
