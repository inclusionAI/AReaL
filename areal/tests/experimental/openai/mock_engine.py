"""Mock inference engine for testing without GPU.

This module provides a MockInferenceEngine that can be used with real RPC servers
and proxy workers to validate the proxy architecture without requiring GPU computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse

if TYPE_CHECKING:
    from areal.api.io_struct import LocalInfServerInfo


class MockInferenceEngine(InferenceEngine):
    """Mock inference engine for testing without GPU.

    Returns deterministic mock responses that exercise the full
    proxy architecture without actual model inference.

    This engine:
    - Accepts any initialization parameters (including addr for proxy compatibility)
    - Returns a fixed response "Hello! This is a mock response." for all requests
    - Tracks version numbers for weight update compatibility
    - Can be used with real RPC servers and proxy workers

    Example:
        >>> config = InferenceEngineConfig(tokenizer_path="path/to/tokenizer")
        >>> engine = MockInferenceEngine(config)
        >>> engine.initialize(addr="localhost:8000")
        >>> # Engine is now ready for use
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        self._initialized = False
        self._version = 0
        self._server_addr: str | None = None

    def initialize(
        self,
        addr: str | None = None,
        engine_id: str | None = None,
        train_data_parallel_size: int | None = None,
        **kwargs,
    ):
        """Initialize the mock engine.

        Parameters
        ----------
        addr : str, optional
            Server address (stored but not used for mock engine)
        engine_id : str, optional
            Engine identifier (ignored)
        train_data_parallel_size : int, optional
            Data parallel size (ignored)
        **kwargs
            Additional arguments (ignored)
        """
        self._server_addr = addr
        self._initialized = True

    @property
    def initialized(self) -> bool:
        """Check if the engine has been initialized."""
        return self._initialized

    def set_version(self, version: int):
        """Set the current weight version."""
        self._version = version

    def get_version(self) -> int:
        """Get the current weight version."""
        return self._version

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Generate a mock response.

        Returns a simple response with:
        - Input tokens copied from request
        - Mock output tokens ("Hello! This is a mock response.")
        - Zero logprobs for all output tokens
        - Current version for all output tokens

        Parameters
        ----------
        req : ModelRequest
            The model request containing input tokens and generation config

        Returns
        -------
        ModelResponse
            A mock response with deterministic output
        """
        tokenizer = req.tokenizer

        if tokenizer is None:
            raise ValueError("MockInferenceEngine requires a tokenizer in the request")

        # Generate a simple mock response
        mock_text = "Hello! This is a mock response."
        output_tokens = tokenizer.encode(mock_text, add_special_tokens=False)

        # Add EOS token if available
        if tokenizer.eos_token_id is not None:
            output_tokens.append(tokenizer.eos_token_id)

        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[self._version] * len(output_tokens),
            stop_reason="stop",
            tokenizer=tokenizer,
            latency=0.001,
            ttft=0.001,
        )

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch a mock server (no-op for mock engine).

        Returns a LocalInfServerInfo with localhost address since
        mock engine doesn't need a real server.

        Parameters
        ----------
        server_args : dict
            Server arguments (ignored)

        Returns
        -------
        LocalInfServerInfo
            Mock server info pointing to localhost
        """
        from areal.api.io_struct import LocalInfServerInfo

        # Return a dummy server info - mock engine doesn't need a real server
        return LocalInfServerInfo(host="127.0.0.1", port=0, process=None)

    def destroy(self):
        """Destroy the mock engine."""
        self._initialized = False
        self._server_addr = None

    def pause_generation(self):
        """Pause generation (no-op for mock engine)."""
        pass

    def continue_generation(self):
        """Continue generation (no-op for mock engine)."""
        pass

    def pause(self):
        """Pause the engine (no-op for mock engine)."""
        pass

    def resume(self):
        """Resume the engine (no-op for mock engine)."""
        pass

    def offload(self):
        """Offload the engine (no-op for mock engine)."""
        pass

    def onload(self, tags: list[str] | None = None):
        """Onload the engine (no-op for mock engine)."""
        pass

    def export_stats(self) -> dict[str, float]:
        """Export engine statistics.

        Returns
        -------
        dict
            Empty stats dictionary for mock engine
        """
        return {}
