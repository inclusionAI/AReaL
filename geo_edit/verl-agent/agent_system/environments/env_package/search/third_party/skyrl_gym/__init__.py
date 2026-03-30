"""Root `__init__` of the gym module setting the `__all__` of skyrl modules."""

from agent_system.environments.env_package.search.third_party.skyrl_gym.core import Env
from agent_system.environments.env_package.search.third_party.skyrl_gym import error

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.registration import (
    make,
    spec,
    register,
    registry,
    pprint_registry,
)
from agent_system.environments.env_package.search.third_party.skyrl_gym import tools, envs

# Define __all__ to control what's exposed
__all__ = [
    # core classes
    "Env",
    # registration
    "make",
    "spec",
    "register",
    "registry",
    "pprint_registry",
    # module folders
    "envs",
    "tools",
    "error",
]
__version__ = "0.0.0"
