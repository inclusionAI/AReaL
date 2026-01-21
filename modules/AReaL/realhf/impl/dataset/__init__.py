# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import os
import re

from realhf.base.importing import import_module

# Import all dataset implementations.
_p = re.compile(r"^(?!.*__init__).*\.py$")
_filepath = os.path.dirname(__file__)
import_module(_filepath, _p)
