# SPDX-License-Identifier: Apache-2.0

"""Synchronous diagnostic records used only by validation engines."""

import os
import random
from pathlib import Path

import numpy as np
import torch

from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json


def record(self, kind, value, *, step=None):
    root = Path(os.environ["AREAL_ELASTIC_DIAGNOSTICS_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{os.environ['AREAL_ELASTIC_LAUNCH_ID']}-{kind}"
    if step is not None:
        prefix += f"-{step:08d}"
    atomic_json(root / f"{prefix}-rank-{self.rank}.json", {"rank": self.rank, **value})


def rng_probe(self):
    # Intentionally sync four diagnostic draws, outside the production loop.
    return {
        "rank": self.rank,
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": torch.rand(()).item(),
        "cuda": torch.rand((), device=self.device).item(),
    }
