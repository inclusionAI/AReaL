"""Unit tests for AEnvironment configuration."""

import pytest

from areal.infra.aenv.config import AenvConfig


def test_aenv_config_defaults_are_valid():
    """Test that default configuration values are valid."""
    config = AenvConfig()

    assert config.aenv_url == "http://localhost"
    assert config.env_name == "default"
    assert config.timeout == 30.0
    assert config.startup_timeout == 120.0
    assert config.tool_call_timeout == 30.0
    assert config.max_retries == 2
    assert config.retry_delay == 0.5
    assert config.auto_release is True
    assert config.turn_discount == 0.9
    assert config.tool_error_policy == "append_error"


def test_aenv_config_raises_for_invalid_timeout():
    """Test that timeout must be positive."""
    with pytest.raises(ValueError, match="timeout must be positive"):
        AenvConfig(timeout=0)


def test_aenv_config_raises_for_invalid_max_retries():
    """Test that max retries must be non-negative."""
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        AenvConfig(max_retries=-1)


def test_aenv_config_raises_for_invalid_turn_discount():
    """Test that turn discount must be in [0, 1]."""
    with pytest.raises(ValueError, match="turn_discount must be within"):
        AenvConfig(turn_discount=1.1)


def test_aenv_config_raises_for_empty_env_name():
    """Test that env name must be non-empty."""
    with pytest.raises(ValueError, match="env_name must be non-empty"):
        AenvConfig(env_name="")
