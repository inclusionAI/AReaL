__version__ = '0.4.2'

import os
from os.path import join as pjoin

from alfworld.utils import mkdirs


_default_alfworld_cache = os.path.expanduser("~/.cache/alfworld")
ALFWORLD_DATA = mkdirs(os.getenv("ALFWORLD_DATA", _default_alfworld_cache))
os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA  # Set the environment variable, in case it wasn't.

BUILTIN_DATA_PATH = pjoin(os.path.dirname(__file__), "data")
ALFRED_PDDL_PATH = pjoin(BUILTIN_DATA_PATH, 'alfred.pddl')
ALFRED_TWL2_PATH = pjoin(BUILTIN_DATA_PATH, 'alfred.twl2')
