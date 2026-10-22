# SPDX-License-Identifier: Apache-2.0

"""Policy selection for native Elastic training and independent reference models."""

from ..contracts import SchedulingPolicy


def build_policy(name: str, *, idle_timeout: float) -> SchedulingPolicy:
    """Basic starts with the timeout baseline; Reference scheduling is pending."""
    if name == "basic":
        from .basic.policy import IdleTimeoutPolicy

        return IdleTimeoutPolicy(idle_timeout)
    if name == "reference":
        raise NotImplementedError(
            "actor_lifecycle.policy='reference' is not implemented: "
            "Reference currently provides offline models and a simulator only"
        )
    raise ValueError("actor_lifecycle.policy must be basic or reference")
