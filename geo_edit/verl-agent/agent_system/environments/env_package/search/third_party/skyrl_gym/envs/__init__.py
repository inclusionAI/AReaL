"""Registers the internal gym envs."""

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.registration import register

register(
    id="search",
    entry_point="skyrl_gym.envs.search.env:SearchEnv",
)
