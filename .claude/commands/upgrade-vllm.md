---
name: upgrade-vllm
description: Upgrade vLLM version in AReaL. Audits all vLLM API usage, cross-references upstream source, and updates call sites for compatibility.
argument-hint: <version>
---

## Usage

```
/upgrade-vllm $ARGUMENTS
```

**Arguments (`$VERSION`):** Target vLLM version tag or commit hash, e.g. `v0.18.0`,
`0.17.0`, or a commit SHA. If not given, get the required version from AReaL's
"pyproject.toml" and check whether the current code is fully compatible with the
specified version.

## Prerequisites — Source Code for Cross-Referencing

This command requires the upstream vLLM source repo to cross-reference API signatures.

### vLLM

```bash
VLLM_DIR="${REPO_ROOT}/vllm-src"
# Validate VERSION to prevent command injection
if [[ ! "$VERSION" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
  echo "Error: Invalid version format: $VERSION"
  exit 1
fi
if [ ! -d "$VLLM_DIR" ]; then
  git clone --depth 1 --branch "${VERSION}" https://github.com/vllm-project/vllm.git "$VLLM_DIR"
else
  cd "$VLLM_DIR" && git fetch origin && git checkout "${VERSION}" && cd -
fi
```

If cloning or checkout fails, report to the user immediately.

The relevant upstream source paths are:

- `vllm-src/vllm/entrypoints/openai/api_server.py`
- `vllm-src/vllm/entrypoints/openai/cli_args.py`
- `vllm-src/vllm/entrypoints/openai/completion/api_router.py`
- `vllm-src/vllm/entrypoints/openai/completion/protocol.py`
- `vllm-src/vllm/entrypoints/openai/engine/protocol.py`
- `vllm-src/vllm/entrypoints/openai/utils.py`
- `vllm-src/vllm/entrypoints/utils.py`
- `vllm-src/vllm/logger.py`
- `vllm-src/vllm/lora/request.py`
- `vllm-src/vllm/utils/argparse_utils.py`
- `vllm-src/vllm/v1/engine/__init__.py`
- `vllm-src/vllm/v1/engine/core.py`
- `vllm-src/vllm/v1/engine/output_processor.py`
- `vllm-src/vllm/v1/metrics/stats.py`
- `vllm-src/vllm/v1/request.py`
- `vllm-src/vllm/v1/worker/gpu_worker.py`
- `vllm-src/vllm/worker/worker.py`
- `vllm-src/vllm/lora/lora_model.py`
- `vllm-src/vllm/lora/peft_helper.py`
- `vllm-src/vllm/model_executor/model_loader/__init__.py`
- `vllm-src/vllm/envs.py`

______________________________________________________________________

## Affected Files

### Primary (engine layer — most likely to break)

| File                                             | Imports / Usage                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/engine/vllm_ext/areal_vllm_server.py`     | `entrypoints.openai.api_server` (`build_app`, `run_server`), `entrypoints.openai.cli_args`, `entrypoints.openai.completion.api_router` (`create_completion`), `entrypoints.openai.completion.protocol` (`CompletionRequest`), `entrypoints.openai.engine.protocol` (`ErrorResponse`, `OpenAIBaseModel`), `entrypoints.openai.utils` (`validate_json_request`), `entrypoints.utils`, `logger`, `lora.request`, `utils.argparse_utils`, `v1.engine`, `v1.engine.core`, `v1.metrics.stats`, `v1.request`, `v1.engine.output_processor` |
| `areal/engine/vllm_ext/vllm_worker_extension.py` | `logger`, `lora.lora_model`, `lora.peft_helper`, `lora.request`, `model_executor.model_loader`                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `areal/engine/vllm_remote.py`                    | `VLLMBackend` class (builds HTTP requests to vLLM endpoints), `RemotevLLMEngine` wrapper                                                                                                                                                                                                                                                                                                                                                                                                                                            |

### Secondary (infrastructure / platform layer)

| File                                  | Imports / Usage                                                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `areal/infra/platforms/cuda.py`       | `vllm.v1.worker.gpu_worker.Worker` (try), `vllm.worker.worker.Worker` (fallback) via try/except            |
| `areal/infra/platforms/unknown.py`    | Same as `cuda.py`                                                                                          |
| `areal/infra/platforms/platform.py`   | Abstract `get_vllm_worker_class()` method                                                                  |
| `areal/infra/launcher/vllm_server.py` | `vLLMServerWrapper`, `launch_server_cmd`, env vars (`VLLM_CACHE_ROOT`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`) |
| `areal/infra/launcher/ray.py`         | vLLM server launch in Ray cluster (imports `vLLMConfig`)                                                   |
| `areal/infra/launcher/local.py`       | vLLM server launch locally (imports `vLLMConfig`)                                                          |
| `areal/infra/launcher/slurm.py`       | vLLM server launch in Slurm (imports `vLLMConfig`)                                                         |
| `areal/infra/launcher/__init__.py`    | Re-exports `launch_vllm_server`, `vLLMServerWrapper`                                                       |
| `areal/infra/utils/launcher.py`       | `VLLM_CACHE_ROOT` path, vLLM allocation mode validation                                                    |

### Tertiary (config / API / workflow layer)

| File                                          | Usage                                                            |
| --------------------------------------------- | ---------------------------------------------------------------- |
| `areal/api/cli_args.py`                       | `vLLMConfig` dataclass — all vLLM CLI flags and server arguments |
| `areal/api/alloc_mode.py`                     | `"vllm"` as a backend literal type                               |
| `areal/api/io_struct.py`                      | `vision_msg_vllm` field on `ModelRequest`                        |
| `areal/trainer/rl_trainer.py`                 | `RemotevLLMEngine` initialization, `vLLMConfig.build_args()`     |
| `areal/workflow/vision_rlvr.py`               | Sets `vision_msg_vllm` on `ModelRequest`                         |
| `areal/tools/validate_docker_installation.py` | Checks `vllm` is importable, validates `vllm._C`                 |
| `areal/tools/validation_base.py`              | `vllm._C` as native extension verification                       |

### Test files

| File                              | Usage                                                      |
| --------------------------------- | ---------------------------------------------------------- |
| `tests/test_inference_engines.py` | `vLLMConfig`, `RemotevLLMEngine`, engine integration tests |
| `tests/test_model_utils.py`       | vLLM allocation mode regression tests                      |
| `tests/test_allocation_mode.py`   | vLLM allocation mode parsing tests                         |
| `tests/test_examples.py`          | vLLM integration test configurations                       |
| `tests/grpo/test_grpo.py`         | vLLM references in GRPO config tests                       |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

### 1. `vllm.entrypoints.openai.api_server`

**Source:** `vllm-src/vllm/entrypoints/openai/api_server.py`

#### `build_app(args, supported_tasks=None)`

Imported in `areal_vllm_server.py:9` as `_original_build_app`:

```python
from vllm.entrypoints.openai.api_server import build_app as _original_build_app
```

AReaL monkey-patches `build_app` to inject custom routes in
`areal_vllm_server.py:458-480`:

```python
import vllm.entrypoints.openai.api_server as _api_server_module

def _areal_build_app(args, supported_tasks=None):
    app = _original_build_app(args, supported_tasks=supported_tasks)
    # Remove vLLM's /v1/completions POST route so AReaL's takes precedence
    app.router.routes = [
        route for route in app.router.routes
        if not (hasattr(route, "path") and route.path == "/v1/completions"
                and hasattr(route, "methods") and "POST" in route.methods)
    ]
    app.include_router(router)
    return app

_api_server_module.build_app = _areal_build_app
```

**Check:** Does `build_app` still exist? Still accepts `(args, supported_tasks=None)`?
Does it return a FastAPI app? Does the returned app still have `router.routes` with
`path` and `methods` attributes on route objects?

#### `run_server(args)`

Imported in `areal_vllm_server.py:10`:

```python
from vllm.entrypoints.openai.api_server import run_server
```

Called in `areal_vllm_server.py:490`:

```python
uvloop.run(run_server(args))
```

**Check:** Signature. Is it still async? Does it still accept parsed args directly? Does
it internally call `build_app` (which AReaL monkey-patches)?

______________________________________________________________________

### 2. `vllm.entrypoints.openai.cli_args`

**Source:** `vllm-src/vllm/entrypoints/openai/cli_args.py`

#### `make_arg_parser(parser)`

Called in `areal_vllm_server.py:486`:

```python
parser = make_arg_parser(parser)
```

**Check:** Signature. Does it still accept a parser and return a parser? New required
args?

#### `validate_parsed_serve_args(args)`

Called in `areal_vllm_server.py:488`:

```python
validate_parsed_serve_args(args)
```

**Check:** Signature. Still exists?

______________________________________________________________________

### 3. `vllm.entrypoints.openai.completion.api_router`

**Source:** `vllm-src/vllm/entrypoints/openai/completion/api_router.py`

#### `create_completion` (aliased as `original_create_completion`)

Imported in `areal_vllm_server.py:12-14`:

```python
from vllm.entrypoints.openai.completion.api_router import (
    create_completion as original_create_completion,
)
```

Called in `areal_vllm_server.py:333`:

```python
response = await original_create_completion(request, raw_request)
```

**Check:** Signature `(request: CompletionRequest, raw_request: Request)` still valid?
Was this function moved or renamed? Is the module path still
`vllm.entrypoints.openai.completion.api_router`?

______________________________________________________________________

### 4. `vllm.entrypoints.openai.completion.protocol`

**Source:** `vllm-src/vllm/entrypoints/openai/completion/protocol.py`

#### `CompletionRequest`

Imported in `areal_vllm_server.py:15`:

```python
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
```

Used as type annotation in `areal_vllm_server.py:327`:

```python
async def create_completion(request: CompletionRequest, raw_request: Request):
```

**Check:** Still exists at this module path? Fields unchanged? Was this class moved?

______________________________________________________________________

### 5. `vllm.entrypoints.openai.engine.protocol`

**Source:** `vllm-src/vllm/entrypoints/openai/engine/protocol.py`

#### `ErrorResponse`

Imported in `areal_vllm_server.py:16`:

```python
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, OpenAIBaseModel
```

Used in route response model in `areal_vllm_server.py:320-322`:

```python
HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
```

**Check:** Still exists at this module path?

#### `OpenAIBaseModel`

AReaL request classes inherit from it in `areal_vllm_server.py:41-92`:

```python
class UpdateWeightsRequest(OpenAIBaseModel):
    model_path: str
    ...
```

**Check:** Still exists? Still a Pydantic model base class? Was it moved?

______________________________________________________________________

### 6. `vllm.entrypoints.openai.utils`

**Source:** `vllm-src/vllm/entrypoints/openai/utils.py`

#### `validate_json_request`

Imported in `areal_vllm_server.py:17`:

```python
from vllm.entrypoints.openai.utils import validate_json_request
```

Used as a FastAPI dependency in `areal_vllm_server.py:317`:

```python
@router.post("/v1/completions", dependencies=[Depends(validate_json_request)])
```

**Check:** Still exists at this module path? Still usable as a `Depends()` target?

______________________________________________________________________

### 7. `vllm.entrypoints.utils`

**Source:** `vllm-src/vllm/entrypoints/utils.py`

#### `cli_env_setup()`

Called in `areal_vllm_server.py:482`:

```python
cli_env_setup()
```

**Check:** Signature. Still exists?

#### `load_aware_call`

Used as decorator in `areal_vllm_server.py:326`:

```python
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
```

**Check:** Still a decorator? Signature?

#### `with_cancellation`

Used as decorator in `areal_vllm_server.py:325`:

```python
@with_cancellation
@load_aware_call
async def create_completion(...):
```

**Check:** Still a decorator? Order constraints?

______________________________________________________________________

### 8. `vllm.logger`

**Source:** `vllm-src/vllm/logger.py`

#### `init_logger(name)`

Called in `areal_vllm_server.py:33` and `vllm_worker_extension.py:15`:

```python
logger = init_logger("areal_vllm_server")
logger = init_logger("vllm_worker_extension")
```

**Check:** Signature unchanged?

______________________________________________________________________

### 9. `vllm.utils.argparse_utils`

**Source:** `vllm-src/vllm/utils/argparse_utils.py`

#### `FlexibleArgumentParser`

Used in `areal_vllm_server.py:483`:

```python
parser = FlexibleArgumentParser(
    description="vLLM OpenAI-Compatible RESTful API server."
)
```

**Check:** Still exists at this module path? Was it moved (e.g., to `vllm.utils`)?

______________________________________________________________________

### 10. `vllm.v1.engine` (V1 engine outputs)

**Source:** `vllm-src/vllm/v1/engine/__init__.py`

#### `EngineCoreOutput`

Constructed in `areal_vllm_server.py:355`:

```python
EngineCoreOutput(
    request_id=req.request_id,
    new_token_ids=[],
    finish_reason=FinishReason.ABORT,
    new_logprobs=None,
    new_prompt_logprobs_tensors=None,
    stop_reason=None,
)
```

**Check:** Constructor fields. New required fields? Field renames? Was
`new_prompt_logprobs_tensors` renamed? Was `stop_reason` removed?

#### `EngineCoreOutputs`

Constructed in `areal_vllm_server.py:371`:

```python
EngineCoreOutputs(outputs=outputs)
```

**Check:** Constructor signature. Still accepts `outputs` list?

#### `FinishReason`

Used in `areal_vllm_server.py:358`:

```python
finish_reason=FinishReason.ABORT
```

**Check:** `.ABORT` enum value still exists? Was `FinishReason` moved to a different
module?

______________________________________________________________________

### 11. `vllm.v1.engine.core`

**Source:** `vllm-src/vllm/v1/engine/core.py`

#### `EngineCore`

AReaL monkey-patches multiple methods onto this class in `areal_vllm_server.py:421-437`:

```python
setattr(EngineCore, "abort_all_reqs", abort_all_reqs)
setattr(EngineCore, "areal_injected_update_weight", areal_injected_update_weight)
setattr(EngineCore, "areal_injected_update_weight_lora", areal_injected_update_weight_lora)
setattr(EngineCore, "areal_injected_update_weight_xccl", areal_injected_update_weight_xccl)
setattr(EngineCore, "areal_injected_update_weight_lora_xccl", areal_injected_update_weight_lora_xccl)
```

The patched methods access these EngineCore internals:

- `self.scheduler` — scheduler object
- `self.scheduler.running` — set/list of running requests
- `self.scheduler.waiting` — set/list of waiting requests
- `self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)`
- `self.scheduler.reset_prefix_cache()` — returns bool
- `self.output_queue.put_nowait((client_index, engine_core_outputs))`
- `self.collective_rpc(method_name, args=(...))` — calls worker methods

Request objects in `scheduler.running` / `scheduler.waiting` have:

- `.request_id`
- `.client_index`

**Check:** Does `EngineCore` still have `scheduler`, `output_queue` attributes? Does
`scheduler` still have `running`, `waiting`, `finish_requests()`, `reset_prefix_cache()`
methods? Does `EngineCore` still have `collective_rpc()` and `call_utility_async()`? Has
the scheduler API changed (e.g., different method for aborting requests)? Is there now a
built-in `abort_all` method that replaces the monkey-patch?

______________________________________________________________________

### 12. `vllm.v1.engine.output_processor`

**Source:** `vllm-src/vllm/v1/engine/output_processor.py`

#### `RequestState`

Used via TYPE_CHECKING import in `areal_vllm_server.py:30-31`:

```python
if TYPE_CHECKING:
    from vllm.v1.engine.output_processor import RequestState
```

Used in the `finish_request` monkey-patch at `areal_vllm_server.py:411`:

```python
def finish_request(self, req_state: "RequestState"):
    if req_state.lora_name is None:
        return
    lora_stats = self.lora_name_to_stats[req_state.lora_name]
    if req_state.request_id in lora_stats.running_requests:
        lora_stats.running_requests.remove(req_state.request_id)
```

**Check:** Does `RequestState` still have `.lora_name` and `.request_id` attributes? Was
this class moved? Was it renamed?

______________________________________________________________________

### 13. `vllm.v1.metrics.stats`

**Source:** `vllm-src/vllm/v1/metrics/stats.py`

#### `LoRARequestStates`

Monkey-patched in `areal_vllm_server.py:444-448`:

```python
setattr(LoRARequestStates, "finish_request", finish_request)
```

This patch is guarded by a version check:

```python
if not pkg_version.is_version_greater_or_equal("vllm", "0.12.0"):
    setattr(LoRARequestStates, "finish_request", finish_request)
```

**Check:** Does `LoRARequestStates` still exist at this path? Does it still have a
`finish_request` method? Has the bug (that the monkey-patch fixes) been fixed upstream?
Does `LoRARequestStates` still have `.lora_name_to_stats` dict with `.running_requests`
set? If `vllm >= 0.12.0`, is the patch correctly skipped?

______________________________________________________________________

### 14. `vllm.v1.request`

**Source:** `vllm-src/vllm/v1/request.py`

#### `RequestStatus`

Used in `areal_vllm_server.py:368`:

```python
scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
```

**Check:** `.FINISHED_ABORTED` enum value still exists? Was `RequestStatus` moved?

______________________________________________________________________

### 15. `vllm.lora.request`

**Source:** `vllm-src/vllm/lora/request.py`

#### `LoRARequest`

Imported in both `areal_vllm_server.py:20` and `vllm_worker_extension.py:8`.

Constructed in `areal_vllm_server.py:154-161` (attribute-set pattern):

```python
lora_request = LoRARequest(
    lora_name=lora_name,
    lora_int_id=lora_int_id,
    lora_path=runtime_lora_path,
)
if base_model_name is not None:
    lora_request.base_model_name = base_model_name
```

Constructed in `vllm_worker_extension.py:59-64` (constructor arg pattern):

```python
LoRARequest(
    lora_name=lora_name,
    lora_int_id=lora_int_id,
    lora_path=lora_model_path,
    base_model_name=base_model_name,
)
```

**Check:** Constructor params. Is `base_model_name` still accepted as constructor arg?
Is it still settable as an attribute? Renamed fields?

______________________________________________________________________

### 16. `vllm.lora.lora_model`

**Source:** `vllm-src/vllm/lora/lora_model.py`

#### `LoRAModel.from_lora_tensors(...)`

Called in `vllm_worker_extension.py:235`:

```python
LoRAModel.from_lora_tensors(
    lora_model_id=self.areal_lora_int_id,
    tensors=normalized_weights,
    peft_helper=peft_helper,
    device=self.model_runner.device,
    dtype=self.model_runner.lora_manager.lora_config.lora_dtype,
    model_vocab_size=model_vocab_size,
    weights_mapper=getattr(self.model_runner.model, "hf_to_vllm_mapper", None),
)
```

**Check:** `from_lora_tensors` class method still exists? Signature unchanged? New
required params? `weights_mapper` param still accepted?

______________________________________________________________________

### 17. `vllm.lora.peft_helper`

**Source:** `vllm-src/vllm/lora/peft_helper.py`

#### `PEFTHelper.from_dict(config)`

Called in `vllm_worker_extension.py:226`:

```python
peft_config = {
    "r": self.areal_lora_rank,
    "lora_alpha": self.areal_lora_alpha,
    "target_modules": self.areal_lora_target_modules,
    "bias": self.areal_lora_bias,
}
peft_helper = PEFTHelper.from_dict(peft_config)
```

**Check:** `from_dict` class method still exists? Expected dict keys unchanged?

______________________________________________________________________

### 18. `vllm.model_executor.model_loader`

**Source:** `vllm-src/vllm/model_executor/model_loader/__init__.py`

#### `get_model_loader(load_config)`

Called in `vllm_worker_extension.py:32`:

```python
model_loader = get_model_loader(self.model_runner.vllm_config.load_config)
```

Then used as:

```python
model_loader.load_weights(
    self.model_runner.model, model_config=self.model_runner.model_config
)
```

**Check:** `get_model_loader` still exists? Return type still has
`load_weights(model, model_config=...)` method? Was it moved to a different module?

______________________________________________________________________

### 19. `vllm.envs`

**Source:** `vllm-src/vllm/envs.py`

#### `VLLM_USE_V1` (no longer directly checked)

Previously checked via `if envs.VLLM_USE_V1:` in platform files. AReaL now uses a
try/except import pattern instead in `cuda.py:31-53` and `unknown.py:31-53`:

```python
@classmethod
def get_vllm_worker_class(clas):
    try:
        from vllm.v1.worker.gpu_worker import Worker
        return Worker
    except ImportError:
        pass
    try:
        from vllm.worker.worker import Worker
        return Worker
    except ImportError as e:
        raise RuntimeError("vLLM is not installed or not properly configured.") from e
```

**Check:** Does `vllm.v1.worker.gpu_worker.Worker` still exist? Was V0
`vllm.worker.worker.Worker` removed entirely? Are there other env vars in `vllm.envs`
that AReaL might need?

______________________________________________________________________

### 20. `vllm.v1.worker.gpu_worker` / `vllm.worker.worker`

**Source:** `vllm-src/vllm/v1/worker/gpu_worker.py` and `vllm-src/vllm/worker/worker.py`

#### `Worker`

Imported in `cuda.py` and `unknown.py` for `get_vllm_worker_class()` using try/except
(see Section 19 above).

**Check:** `Worker` class still exists at these paths? Was V0 `vllm.worker.worker`
removed? Was V1 worker moved?

______________________________________________________________________

### 21. Private/internal APIs used by AReaL

These are internal vLLM APIs not part of the public interface. They are most likely to
break on upgrade.

#### Worker extension model runner internals

Accessed in `vllm_worker_extension.py`:

- `self.model_runner.model_config.model` — writable string field (line 31)
- `self.model_runner.vllm_config.load_config` — VllmConfig load config (line 32)
- `self.model_runner.model` — the loaded model object (line 35, 144)
- `self.model_runner.model.load_weights(weights=[(name, tensor)])` — weight loading
  (line 144)
- `self.model_runner.device` — torch device (line 136, 200, 239)
- `self.model_runner.lora_manager` — LoRA manager instance (line 58, 173, 177, 214)
- `self.model_runner.lora_manager.remove_adapter(lora_int_id)` (line 58, 214)
- `self.model_runner.lora_manager.list_adapters()` (line 177)
- `self.model_runner.lora_manager.lora_config.lora_dtype` (line 240)
- `self.model_runner.lora_manager.lora_config.lora_extra_vocab_size` (line 230)
- `self.model_runner.lora_manager.vocab_size` (line 233)
- `self.model_runner.lora_manager._adapter_manager` — private adapter manager (line 185)
- `self.model_runner.lora_manager._adapter_manager._registered_adapters[id]` — private
  dict (line 185)
- `self.model_runner.lora_manager._adapter_manager._add_adapter(model)` — private method
  (line 247)
- `self.model_runner.lora_manager._adapter_manager.activate_adapter(id)` — private
  method (line 248)
- `self.model_runner.model.hf_to_vllm_mapper` — optional attribute (line 243)
- `self.model_runner.add_lora(lora_request)` — public method for disk-based LoRA loading
  (line 66)
- `self.rank` — worker rank (line 279)

**Check:** All `model_runner` attributes still exist? Has `lora_manager` been
restructured? Are `_adapter_manager`, `_registered_adapters`, `_add_adapter`,
`activate_adapter` still available? Has `load_weights` signature changed on the model
object? Does `add_lora` still exist on `model_runner`?

#### EngineCore internals (monkey-patched)

Accessed by injected methods in `areal_vllm_server.py`:

- `self.scheduler` — scheduler instance
- `self.scheduler.running` — iterable of running requests
- `self.scheduler.waiting` — iterable of waiting requests
- `self.scheduler.finish_requests(request_ids, status)` — abort method
- `self.scheduler.reset_prefix_cache()` — returns bool
- `self.output_queue` — async queue
- `self.output_queue.put_nowait((client_index, outputs))` — enqueue outputs
- `self.collective_rpc(method, args=())` — RPC to workers
- `req.request_id` — on request objects in scheduler queues
- `req.client_index` — on request objects in scheduler queues

**Check:** All scheduler attributes/methods still exist? Was `output_queue` renamed or
restructured? Was `collective_rpc` moved or its signature changed?

#### Engine client APIs (called from route handlers)

Accessed via `raw_request.app.state.engine_client` in `areal_vllm_server.py`:

- `llm.engine_core.call_utility_async(method, *args)` — calls utility method on engine
  core (lines 173, 188, 202, 214, 298)
- `llm.collective_rpc(method, args=(...))` — calls method on all workers (lines 233,
  253, 273)

**Check:** Does `engine_client` still expose `engine_core` with `call_utility_async()`?
Does it still expose `collective_rpc()`?

#### openai_serving_models internals (runtime LoRA registration)

Accessed in `areal_vllm_server.py:116-166`:

- `app.state.openai_serving_models` — serving models state object
- `serving_models.lora_requests` — dict of LoRA requests (name -> LoRARequest)
- `request.lora_path` — path attribute on LoRARequest
- `request.lora_int_id` — int id attribute on LoRARequest

**Check:** Does `app.state.openai_serving_models` still exist? Does it still have a
`lora_requests` dict? Are the attribute names on LoRARequest objects unchanged?

______________________________________________________________________

### 22. Environment variables

Used in `vllm_server.py` and `vllm_remote.py`:

- `VLLM_CACHE_ROOT` — vLLM compile cache directory
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING` — set to `"True"` to enable runtime LoRA updates

**Check:** These env vars still recognized by vLLM? Any renamed? Any new required env
vars?

______________________________________________________________________

### 23. vLLM server CLI interface

AReaL builds vLLM CLI commands in `areal/api/cli_args.py` (`vLLMConfig` dataclass). The
CLI flags are converted to command-line arguments via `get_py_cmd()`:

```python
vLLMConfig.build_cmd_from_args(args)
# → python3 -m areal.engine.vllm_ext.areal_vllm_server --model ... --seed ...
```

Fields in `vLLMConfig` that map to vLLM CLI flags:

- `--model`, `--tokenizer`, `--seed`
- `--skip-tokenizer-init`, `--enforce-eager`
- `--dtype`, `--distributed-executor-backend`
- `--max-num-seqs`, `--block-size`, `--swap-space`
- `--cpu-offload-gb`, `--disable-sliding-window`
- `--max-model-len`
- `--no-enable-chunked-prefill`, `--no-enable-prefix-caching`
- `--gpu-memory-utilization`
- `--worker-extension-cls`
- `--enable-sleep-mode`, `--uvicorn-log-level`
- `--enable-lora`, `--max-lora-rank`, `--max-loras`, `--lora-modules`
- `--load-format`, `--trust-remote-code`
- `--tensor-parallel-size`, `--pipeline-parallel-size`
- `--host`, `--port`

**Check:** All CLI flags still accepted? Any renamed (e.g.,
`--distributed-executor-backend`)? Any removed? New required flags? Has
`--worker-extension-cls` been renamed or removed? Has `--no-enable-chunked-prefill`
behavior changed? Does `--enable-sleep-mode` still exist (enables `/sleep` and
`/wake_up` endpoints)?

______________________________________________________________________

### 24. vLLM HTTP endpoints

AReaL interacts with vLLM via HTTP. The following endpoints are used:

**Standard vLLM endpoints:**

- `GET /health` — health check (`vllm_remote.py:221`)
- `GET /v1/models` — server readiness check (`vllm_server.py:72`)
- `POST /v1/completions` — text generation (`vllm_remote.py:90`,
  `areal_vllm_server.py:316`)
- `POST /v1/chat/completions` — VLM chat generation (`vllm_remote.py:87`)
- `POST /v1/load_lora_adapter` — load LoRA adapter from disk (`vllm_remote.py:131`)
- `POST /sleep` — offload model to CPU (`vllm_remote.py:229`)
- `POST /wake_up` — reload model from CPU, with optional `?tags=` query
  (`vllm_remote.py:248`)

**Custom AReaL endpoints** (registered via `@router.post` in `areal_vllm_server.py`):

- `POST /areal_update_weights` — update full model weights from disk
- `POST /areal_update_weights_lora` — update LoRA weights from disk
- `POST /areal_update_weights_xccl` — update full model weights via NCCL/HCCL
- `POST /areal_update_weights_lora_xccl` — update LoRA weights via NCCL/HCCL
- `POST /areal_init_weights_update_group` — initialize distributed weight update group
- `POST /areal_set_update_weight_meta` — set weight metadata for XCCL update
- `POST /areal_set_update_weight_meta_lora` — set LoRA weight metadata for XCCL update
- `POST /areal_pause_generation` — pause generation and abort all requests
- `POST /areal_continue_generation` — resume generation

**Check:** Standard endpoints still at same paths? Response format unchanged? `/sleep`
and `/wake_up` still exist? `/v1/completions` response still has
`choices[0].logprobs.tokens` or `choices[0].logprobs.content` format? Has
`return_tokens_as_token_ids` param changed behavior (token format `"token:123"`)?

______________________________________________________________________

## Version-Guarded Code

AReaL has version-specific code paths:

```python
# areal_vllm_server.py:439-448
# Patch for LoRARequestStates management in vllm < v0.11.0
# This may be removed with vllm >= 0.12.x
from areal.utils import pkg_version

if not pkg_version.is_version_greater_or_equal("vllm", "0.12.0"):
    setattr(LoRARequestStates, "finish_request", finish_request)
```

**Check:** If upgrading to >= 0.12.0, verify the upstream
`LoRARequestStates.finish_request` fix is present. If upgrading past the fix version,
the guarded code becomes dead code — note for cleanup.

Also in `areal/api/cli_args.py` (comments):

```python
# IMPORTANT: vLLM V1 engine forces enable_chunked_prefill=True by default
# TODO(vllm-v0.11.0): vLLM v0.11.0 has inference quality issues when
# temperature=1.0
```

**Check:** Have these known issues been fixed in the target version?

______________________________________________________________________

## Upgrade Workflow

### Step 0: Prepare vLLM source

Clone or checkout the target version as described in Prerequisites above.

### Step 1: Audit `vllm` API signatures

For EACH entry in the API Usage Catalog above:

1. Open the upstream source file at the target version.
1. Compare the function/class signature against the current AReaL invocation.
1. Flag any of:
   - **Removed parameters** still passed by AReaL → must remove from call site
   - **Renamed parameters** → must rename in call site
   - **New required parameters** (no default) → must add to call site
   - **New optional parameters** with useful defaults → document but skip
   - **Changed return types** → must update consumers
   - **Removed functions/classes** → must find replacement
   - **Moved modules** → must update import paths
   - **Changed method signatures** on returned/internal objects → must update call sites
1. Record findings per-file.

### Step 2: Audit private/internal API compatibility

vLLM's internal APIs (Section 21) are the most fragile. For each:

1. Search the upstream source for the accessed attribute/method.
1. Verify it still exists at the expected path.
1. Check if vLLM has added official APIs that replace the private access patterns.
1. If an official API exists, prefer migrating to it over maintaining private API
   access.

### Step 3: Audit vLLM CLI flag compatibility

Compare `vLLMConfig` fields in `areal/api/cli_args.py` against the vLLM CLI:

```bash
cd vllm-src && python -m vllm.entrypoints.openai.api_server --help
```

Flag any removed/renamed CLI arguments.

### Step 4: Update `pyproject.toml`

Update the `vllm` version pin in `pyproject.toml`:

```toml
vllm = [
"vllm==X.Y.Z; sys_platform == 'linux' and platform_machine == 'x86_64'",
]
```

Run `uv lock` to verify dependency resolution.

### Step 5: Apply code changes

For each flagged incompatibility from Steps 1-3:

1. Update the call site in the affected file.
1. Preserve existing behavior — do NOT refactor beyond what's required.
1. If a function was removed, check the upstream migration guide or changelog.
1. If a module was moved, update the import path.

Priority order for applying changes:

1. `areal/engine/vllm_ext/areal_vllm_server.py` (highest risk — monkey-patches, V1
   engine internals, many imports)
1. `areal/engine/vllm_ext/vllm_worker_extension.py` (private API consumer — model
   runner, LoRA manager)
1. `areal/engine/vllm_remote.py` (HTTP interface — response parsing, endpoint paths)
1. `areal/infra/platforms/cuda.py` (V0/V1 worker import)
1. `areal/infra/platforms/unknown.py` (same as cuda.py)
1. `areal/infra/launcher/vllm_server.py` (env vars, server lifecycle)
1. `areal/api/cli_args.py` (`vLLMConfig` — CLI flag mapping)
1. `areal/infra/launcher/ray.py`
1. `areal/infra/launcher/local.py`
1. `areal/infra/launcher/slurm.py`
1. `areal/trainer/rl_trainer.py`
1. Test files

### Step 6: Update version-guarded code

1. Review the `pkg_version.is_version_greater_or_equal("vllm", "0.12.0")` guard in
   `areal_vllm_server.py`. If upgrading to >= 0.12.0, verify the upstream fix and
   potentially remove the dead code path.
1. Review TODO comments referencing specific vLLM versions.

### Step 7: Run pre-commit and tests

```bash
pre-commit run --all-files
uv run pytest tests/test_inference_engines.py -v
uv run pytest tests/test_model_utils.py -v
uv run pytest tests/test_allocation_mode.py -v
```

If a GPU with vLLM installed is available:

```bash
uv run pytest tests/test_examples.py -v -k vllm
```

### Step 8: Report changes

Output a summary in this format:

```
## Upgrade Summary: vLLM ${OLD_VERSION} → ${NEW_VERSION}

### Breaking Changes Found
- [file:line] description of change needed

### Module Moves / Renames
- [old_path] → [new_path] (affects: list of AReaL files)

### Private API Changes
- [internal_api] description of change (affects: list of AReaL files)

### CLI Flag Changes
- [flag] description (affects: vLLMConfig in cli_args.py)

### API Additions (new optional params, informational)
- [upstream_file] description

### Files Modified
- path/to/file.py: description of change

### Version-Guarded Code
- [file:line] status of version guard (still needed / can be removed)

### Tests
- ✅ pre-commit passed
- ✅ test_inference_engines passed
- ✅ test_model_utils passed
- ✅ test_allocation_mode passed
- ⬚ integration tests (requires GPU with vLLM installed)
```

If there are unresolvable breaking changes, STOP and ask the user before proceeding.
