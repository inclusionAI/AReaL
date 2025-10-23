# Any-Agent RL: Training Interface for Any OpenAI-Compatible Agents

## Overview

This feature enables **any agent framework that integrates with OpenAI APIs** to
leverage AReaL for reinforcement learning training without requiring modifications to
the agent's core code.

AReaL exposes an OpenAI-compatible proxy endpoint that wraps the underlying RL training
infrastructure. Users simply point their agent's `OPENAI_BASE_URL` to AReaL's service
URL, allowing seamless end-to-end RL model training through familiar OpenAI API
patterns.

## Key Benefits

1. **Minimal Integration Overhead**: Users only need to add lightweight trajectory
   markers and reward signals—no refactoring of existing agent code required
1. **Privacy-Preserving**: Datasets and agent orchestration logic remain on the user's
   infrastructure; only model requests and rewards are transmitted to the training
   service
1. **Fully Virtualized Infrastructure**: Distributed inference and training resources
   are abstracted away; users can run RL experiments from any machine with network
   access

## Usage Examples

### Phase 1: Self-Hosted Training on Local GPU Cluster

**Step 1: Launch the AReaL Service**

Users start the AReaL service on their GPU cluster:

```bash
python3 -m areal.launcher.slurm any_agent_grpo.py --config my-config.yaml
```

The service prints the endpoint URL to stdout:

```bash
[2025-10-23 12:34:56] INFO: AReaL any-agent server listening at http://192.168.1.100:8080
```

**Step 2: Run Custom Agent with AReaL Backend**

Users can launch their agent from any machine with network access to the service URL.

#### Example: Manual Trajectory Management

```python
import os
import asyncio
from aiohttp import ClientSession

class MyAgent:
    async def run(self, prompt):
        url = os.environ['OPENAI_BASE_URL']
        async with ClientSession(base_url=url) as session:
            ################ USER CODE START ################
            # Standard OpenAI chat completion request
            response = await session.post("/v1/chat/completions", json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}]
            })
            completion_data = await response.json()
            ################# USER CODE END #################

            # NEW HERE
            # Optional: Set intermediate reward for this completion
            await session.post("/rl/v1/set_reward", json={
                "completion_id": completion_data["id"],
                "reward": 0.5
            })

            ################ USER CODE START ################
            # ... additional agent steps ...
            ################# USER CODE END #################

            # NEW HERE
            # Finalize trajectory and set terminal reward
            await session.post("/rl/v1/finalize", json={
                "final_reward": 1.0
            })

from areal.core import AgentSession

async def run_training():
    for epoch in range(5):
        for prompt in dataset:
            # All requests within this context are grouped into a single trajectory
            async with AgentSession():
                await MyAgent().run(prompt)

if __name__ == "__main__":
    os.environ['OPENAI_BASE_URL'] = 'http://192.168.1.100:8080'
    asyncio.run(run_training())
```

#### Example: Simplified Batch Processing

AReaL provides helper utilities for streamlined batch trajectory collection:

```python
import os
import asyncio
from areal.core import AgentRunner

async def run_training():
    for epoch in range(5):
        # AgentRunner handles trajectory management automatically
        await AgentRunner().run(MyAgent().run, dataset)

if __name__ == "__main__":
    os.environ['OPENAI_BASE_URL'] = 'http://192.168.1.100:8080'
    asyncio.run(run_training())
```

### Phase 2: Managed Cloud Service with API Authentication

For a fully managed service, users authenticate via API keys. Training infrastructure is
provisioned on-demand:

```python
import os
import asyncio
from areal.core import AgentRunner

async def run_training():
    for epoch in range(5):
        # Training jobs are dynamically provisioned in the cloud
        await AgentRunner().run(MyAgent().run, dataset)

if __name__ == "__main__":
    os.environ['OPENAI_BASE_URL'] = 'https://api.areal.ai'
    os.environ['OPENAI_API_KEY'] = 'sk-proj-...'  # User's API key
    asyncio.run(run_training())
```

### Phase 3: Inference with Fine-Tuned Models

Upon training completion, the service returns model identifiers:

- **Experiment ID**: e.g., `"my-model"`
- **Trial ID**: e.g., `"2025-10-23-run-1"`

Users can deploy the fine-tuned model for inference:

```python
from areal.engine.sglang_remote import RemoteSGLangEngine

engine = RemoteSGLangEngine()
engine.initialize(
    experiment_name="my-model",
    trial_name="2025-10-23-run-1",
    user="username"
)
response = await engine.agenerate(prompt="What is reinforcement learning?")
print(response.text)
```

## Implementation Plan

### Step 1: Extend LiteLLM Proxy with RL-Specific Endpoints

**Objective**: Add trajectory tracking and reward management capabilities to the LiteLLM
proxy server.

The base proxy application is defined in `litellm/proxy/proxy_server.py` as the `app`
variable. We extend it by:

1. Overriding the standard chat completion endpoint to intercept and track all model
   requests
1. Adding custom RL endpoints for reward assignment and trajectory finalization
1. Maintaining session state for in-flight trajectories

**Implementation**:

```python
from litellm.proxy.proxy_server import app
from areal.experimental.openai.types import CompletionWithTokenLogpReward
from areal.core.types import ModelRequest
import time

# Attach AReaL inference engine and session state to the proxy application
app.engine = rollout_engine  # Reference to the InferenceEngine instance
app.session_cache = {}  # Mapping: session_id -> {completion_id -> CompletionWithTokenLogpReward}

@app.post("/v1/chat/completions")
async def chat_completions(request: dict, session_id: str):
    """
    OpenAI-compatible chat completion endpoint with trajectory tracking.

    Intercepts standard OpenAI API calls, forwards them to AReaL's inference engine,
    and caches results for RL reward assignment.
    """
    # Convert OpenAI request format to AReaL's internal ModelRequest
    model_request = ModelRequest(
        prompt=request["messages"],
        model=request.get("model", "default"),
        temperature=request.get("temperature", 1.0),
        max_tokens=request.get("max_tokens", 512)
    )

    # Execute inference through AReaL's engine
    response = await app.engine.agenerate(model_request)

    # Wrap response with trajectory tracking metadata
    completion = CompletionWithTokenLogpReward(
        id=generate_completion_id(),
        text=response.text,
        token_ids=response.token_ids,
        logprobs=response.logprobs,
        reward=None  # Reward assigned later via /rl/v1/set_reward
    )

    # Store completion in session cache for reward assignment
    if session_id not in app.session_cache:
        app.session_cache[session_id] = {}
    app.session_cache[session_id][completion.id] = completion

    # Return OpenAI-compatible response format
    return {
        "id": completion.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion.text},
            "finish_reason": "stop"
        }]
    }

@app.get("/rl/v1/start_session")
async def start_session():
    """
    Create a new trajectory session.

    Returns:
        session_id (str): Unique identifier for the trajectory session
    """
    session_id = generate_session_id()
    app.session_cache[session_id] = {}
    return {"session_id": session_id}

@app.post("/rl/v1/set_reward")
async def set_reward(session_id: str, completion_id: str, reward: float):
    """
    Assign a reward value to a specific completion within a trajectory.

    Args:
        session_id: Trajectory session identifier
        completion_id: Specific completion to assign reward to
        reward: Scalar reward value
    """
    if session_id not in app.session_cache:
        return {"error": "Session not found"}, 404
    if completion_id not in app.session_cache[session_id]:
        return {"error": "Completion not found"}, 404

    app.session_cache[session_id][completion_id].reward = reward
    return {"status": "success"}

@app.post("/rl/v1/finalize")
async def finalize(session_id: str, final_reward: float):
    """
    Mark a trajectory as complete and set its terminal reward.

    Args:
        session_id: Trajectory session identifier
        final_reward: Terminal reward for the entire trajectory
    """
    if session_id not in app.session_cache:
        return {"error": "Session not found"}, 404

    app.session_cache[session_id]["_final_reward"] = final_reward
    app.session_cache[session_id]["_completed"] = True
    return {"status": "success"}
```

### Step 2: Implement ProxyWorkflow for Trajectory Collection

**Objective**: Create a workflow adapter that consumes completed trajectories from user
agents instead of a static dataset.

In traditional AReaL workflows, `arun_episode()` iterates over a pre-defined dataset.
`ProxyWorkflow` inverts this model—it waits for external agents to complete trajectories
via the proxy API, then consumes them for training.

**Implementation**:

```python
from areal.api.workflow_api import RolloutWorkflow
import asyncio

class ProxyWorkflow(RolloutWorkflow):
    """
    Workflow adapter for training on trajectories submitted by external agents.

    Unlike standard workflows that iterate over datasets, ProxyWorkflow waits for
    user agents to complete trajectories via the proxy server API endpoints.
    """

    def __init__(self, app):
        """
        Args:
            app: Reference to the FastAPI/LiteLLM proxy server application
        """
        self.app = app
        self.lock = asyncio.Lock()  # Ensures thread-safe session reservation

    async def arun_episode(self, engine, data):
        """
        Wait for and consume a single completed trajectory from a user agent.

        Args:
            engine: AReaL inference engine (should match app.engine)
            data: Not used (set to None for proxy-based workflows)

        Returns:
            trajectory: Completed trajectory in AReaL's internal format
        """
        assert data is None, "ProxyWorkflow does not use static datasets"
        assert engine is self.app.engine, "Engine mismatch between workflow and proxy"

        # Step 1: Atomically reserve an available session
        async with self.lock:
            session_id = await self._reserve_session()

        # Step 2: Block until the user agent marks the session as complete
        await self._wait_for_completion(session_id)

        # Step 3: Convert session data to AReaL's trajectory format
        trajectory = self._export_trajectory(session_id)

        # Step 4: Clean up completed session from cache
        async with self.lock:
            del self.app.session_cache[session_id]

        return trajectory

    async def _reserve_session(self):
        """
        Atomically claim the first unreserved session from the cache.

        Returns:
            session_id (str): Reserved session identifier
        """
        while True:
            for session_id, session_data in self.app.session_cache.items():
                if not session_data.get("_reserved", False):
                    session_data["_reserved"] = True
                    return session_id
            # No available sessions; poll for new ones
            await asyncio.sleep(0.1)

    async def _wait_for_completion(self, session_id):
        """
        Block until the user agent marks the session as completed.

        Args:
            session_id: Session identifier to monitor
        """
        while not self.app.session_cache[session_id].get("_completed", False):
            await asyncio.sleep(0.1)

    def _export_trajectory(self, session_id):
        """
        Convert proxy cache format to AReaL's internal trajectory representation.

        Args:
            session_id: Session identifier to export

        Returns:
            trajectory: Dictionary containing completions and terminal reward
        """
        session_data = self.app.session_cache[session_id]

        # Extract completions (filter out metadata fields prefixed with '_')
        completions = [
            v for k, v in session_data.items()
            if not k.startswith("_")
        ]

        final_reward = session_data.get("_final_reward", 0.0)

        return {
            "completions": completions,
            "final_reward": final_reward
        }
```

### Step 3: Implement HTTP Router for Load Balancing

**Objective**: Provide a single entry point that distributes user requests across
multiple proxy server instances running on data-parallel worker nodes.

In distributed training setups, each data-parallel rank runs its own proxy server
instance. The router abstracts this complexity by exposing a unified endpoint and
load-balancing requests across all backend instances.

**Implementation**:

```python
from aiohttp import web
import aiohttp

class ProxyRouter:
    """
    HTTP load balancer for distributing requests across data-parallel proxy servers.

    Each data-parallel rank runs a separate proxy server instance. The router provides
    a single unified endpoint for users and balances load using round-robin scheduling.
    """

    def __init__(self, proxy_urls: list[str]):
        """
        Args:
            proxy_urls: List of backend proxy server URLs
                       (e.g., ["http://node1:8001", "http://node2:8001"])
        """
        self.proxy_urls = proxy_urls
        self.current_idx = 0

    async def route_request(self, request: web.Request) -> web.Response:
        """
        Forward incoming request to the next available proxy server using round-robin.

        Args:
            request: Incoming HTTP request from user

        Returns:
            Response from the selected backend proxy server
        """
        # Select next proxy server in round-robin order
        target_url = self.proxy_urls[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.proxy_urls)

        # Forward request to selected backend
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=f"{target_url}{request.path}",
                headers=request.headers,
                data=await request.read()
            ) as backend_response:
                # Relay backend response to client
                return web.Response(
                    body=await backend_response.read(),
                    status=backend_response.status,
                    headers=backend_response.headers
                )

    def start(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Start the HTTP router server.

        Args:
            host: Bind address (default: 0.0.0.0 for external access)
            port: Bind port (default: 8080)
        """
        app = web.Application()
        app.router.add_route('*', '/{path:.*}', self.route_request)
        web.run_app(app, host=host, port=port)
```

### Step 4: Integration and Testing

**Objective**: Validate end-to-end functionality from agent request ingestion through
model training.

**Test Plan**:

1. **Create Training Script**: Implement `any_agent_grpo.py` that instantiates
   `ProxyWorkflow` instead of standard dataset-based workflows

1. **Launch Distributed Infrastructure**:

   - On each data-parallel rank, initialize the inference engine and start the LiteLLM
     proxy server in a background thread
   - On rank 0, start the HTTP router and print the unified service URL for users

1. **Agent Connection**: Configure a test agent to point to the router URL and submit
   sample trajectories with rewards

1. **Trajectory Collection**: Verify that `ProxyWorkflow.arun_episode()` correctly
   blocks until trajectories are completed and exports them in the expected format

1. **Training Loop**: Confirm that AReaL's existing training pipeline consumes
   proxy-sourced trajectories identically to dataset-sourced trajectories (no
   modifications to training logic required)

1. **Model Checkpointing**: Verify that model checkpoints are saved with correct
   experiment/trial identifiers for subsequent inference

**Success Criteria**:

- Agent can submit trajectories without code modifications (only `OPENAI_BASE_URL`
  change)
- Training loop processes proxy trajectories and updates model weights
- Loss curves and metrics match expectations from equivalent dataset-based training
- Fine-tuned model can be loaded for inference using experiment/trial IDs
