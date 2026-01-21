# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

from .logging import getLogger

logger = getLogger("importing")


def import_module(path: str, pattern: re.Pattern):
    dirname = Path(path)
    for x in os.listdir(dirname.absolute()):
        if not pattern.match(x):
            continue
        module_path = os.path.splitext(os.path.join(dirname, x))[0]
        assert "realhf" in module_path
        start_idx = path.rindex("realhf")
        module_path = module_path[start_idx:]
        module_path = "realhf." + module_path.replace(os.sep, ".").replace(
            "realhf.", ""
        )
        # logger.info(f"Automatically importing module {module_path}.")
        importlib.import_module(module_path)


def import_usercode(module_path: str, module_name: str):
    # Create a module spec
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    # Create a module object
    module = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules
    sys.modules[module_name] = module
    # Execute the module in its own namespace
    spec.loader.exec_module(module)
