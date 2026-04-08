from verl.workers.reward_manager.registry import register, REWARD_MANAGER_REGISTRY
from pathlib import Path

error_loaded_reward_manager = {}
def get_reward_manager_cls(name):
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    if name not in REWARD_MANAGER_REGISTRY:
        if name in error_loaded_reward_manager:
            print("Error loading reward manager:", name, "Please check your dependencies.")
            raise error_loaded_reward_manager[name]
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER_REGISTRY[name]

# search current directory for reward manager classes
current_dir = Path(__file__).parent
for file in current_dir.glob("*.py"):
    if file.name == "__init__.py":
        continue
    try:
        # import
        module = __import__(f"verl_tool.workers.reward_manager.{file.stem}", fromlist=[file.stem])
    except Exception as e:
        error_loaded_reward_manager[file.stem] = e
        pass

import verl.workers.reward_manager.registry
verl.workers.reward_manager.registry.get_reward_manager_cls = get_reward_manager_cls

# v0.7.1 also uses experimental reward_loop registry
try:
    import verl.experimental.reward_loop.reward_manager.registry as _exp_registry
    _exp_registry.get_reward_manager_cls = get_reward_manager_cls
    # Also merge verl-tool's registered managers into the experimental registry
    _exp_registry.REWARD_MANAGER.update(REWARD_MANAGER_REGISTRY)
except ImportError:
    pass