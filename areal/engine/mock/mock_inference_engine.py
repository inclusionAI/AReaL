from concurrent.futures import Future

from areal.api.engine_api import InferenceEngine


class MockInferenceEngine(InferenceEngine):
    """
    No-op inference engine for single-controller mode.
    Exists only to satisfy FSDPEngine invariants.
    """

    def __init__(self):
        self._version = 0
        self.distributed_weight_update_initialized = False

    # ---- weight update APIs ----
    def init_weights_update_group(self, meta, rank_ids=None):
        fut = Future()
        self.distributed_weight_update_initialized = True
        fut.set_result(None)
        return fut

    def update_weights_from_distributed(self, meta, param_specs):
        fut = Future()
        fut.set_result(None)
        return fut

    def update_weights_from_disk(self, meta):
        fut = Future()
        fut.set_result(None)
        return fut

    # ---- rollout control (no-op) ----
    def pause_generation(self):
        pass

    def continue_generation(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    # ---- versioning ----
    def set_version(self, v):
        self._version = v

    def get_version(self):
        return self._version

    # ---- forbid accidental use ----
    async def agenerate(self, *a, **k):
        raise RuntimeError("MockInferenceEngine cannot generate")

    def submit(self, *a, **k):
        raise RuntimeError("MockInferenceEngine cannot rollout")
