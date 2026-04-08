# Auto-register verl_tool reward managers into verl's registries
# so that Ray actors (e.g. RewardLoopWorker) can find them.
try:
    import verl_tool.workers.reward_manager  # noqa: F401
except ImportError:
    pass
