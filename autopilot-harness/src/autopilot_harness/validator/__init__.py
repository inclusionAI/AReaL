"""Static configuration validation, independent of collection and application."""

from .checker import configuration_id, load_policy, validate_configuration

__all__ = ["configuration_id", "load_policy", "validate_configuration"]
