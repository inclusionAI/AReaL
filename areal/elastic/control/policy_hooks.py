# SPDX-License-Identifier: Apache-2.0

"""Optional CPU revision callbacks; the native trainer owns every GPU operation."""

from typing import Any, Protocol


def validate_unbound_adapter(adapter: Any) -> None:
    """Reject shared/closed helpers before a new trainer takes cleanup ownership."""
    if adapter is not None and (
        getattr(adapter, "owner", None) is not None or getattr(adapter, "closed", False)
    ):
        raise ValueError(
            "Policy adapter is already bound or closed; create a new instance"
        )


class PolicyHooks(Protocol):
    def poll(self) -> None: ...

    def recover(self) -> None: ...
