import copy
from typing import Any, Mapping


def clone_episode_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """
    Clone per-episode mutable inputs (especially chat `messages`) to avoid cross-episode
    mutation when rollouts run concurrently.

    This intentionally deep-copies only `messages` (if present) and shallow-copies the
    rest of the mapping to avoid heavy/unsafe deep copies of tensors or other objects.
    """

    cloned = dict(data)
    if "messages" in data:
        cloned["messages"] = copy.deepcopy(data["messages"])
    return cloned

