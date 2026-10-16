# Sticky Model Routing (v2 SGLang)

Model routing is optional. With the default `rollout.model_router: disabled`, Data
Proxies keep sending generation requests directly to their original backends. The
existing Gateway and Data Router behavior is unchanged.

## Enable

For an existing **v2 actor + v2 SGLang rollout** configuration, set:

```yaml
rollout:
  _version: v2
  model_router: global  # disabled | node | global
```

| Value      | Candidate inference instances                             |
| ---------- | --------------------------------------------------------- |
| `disabled` | The Data Proxy's original backend, as before              |
| `node`     | Serving endpoints on the same head node as the Data Proxy |
| `global`   | All serving endpoints in this controller's model pool     |

For multi-node TP, an instance is identified by its serving head node. Node mode
requires the proxy guard to be colocated with that head. A node with only one serving
endpoint gets no balancing benefit from node mode.

## Data path

```text
Agent -> Gateway -> Data Proxy -> Model Router -> SGLang
                       |
                       +-> tokens, logp, reward, group export, RTensor
```

The existing Data Router still resolves the Gateway's target Data Proxy.
`rollout.routing_strategy` controls that placement. It is independent of `model_router`,
which only selects the model endpoint after the Data Proxy has constructed the native
generation payload.

The Controller starts and stops the Model Router using existing RPCGuard workers: one
router per serving head node in node mode, or one for the pool in global mode. It needs
no additional dependency or scheduler resource type. The authenticated
Gateway-to-Data-Proxy path and data ownership stay in place.

On a trajectory's first model call, the Model Router atomically selects the backend with
the fewest unfinished forwarded generation calls. Ties rotate. Later turns, including
abort/resubmit segments, retain that backend for prefix cache reuse. Each member of a
prompt group has its own session key, so the group can use multiple inference instances
while its data remains grouped at export. Session keys come from the authenticated Data
Proxy session, not caller headers.

This initial policy counts requests inside the router; it does not poll SGLang, KV-cache
utilization or GPU metrics. Tool execution time does not count as a model request.
Existing trajectories are never migrated to relieve a long tail.

## Training and recovery

The Controller closes model admission, pauses physical servers and drains all forwarded
calls before updating weights. It publishes the new version to Data Proxies and the
router before reopening admission. Each returned segment carries the version captured at
admission. A late, never-forwarded request waits for the new version without consuming
the abort/resubmit budget.

Training and evaluation controllers sharing inference servers also share one Model
Router pool. Independently initialized controllers must pass the training controller as
`model_router_owner` when reusing `server_infos`.

In colocated training, generation remains closed while memory is offloaded and reopens
after a complete `onload()` succeeds. Tagged partial onload after offload is rejected in
this mode. Unknown transport outcomes, drain timeouts, or partial control/weight-update
failures close the pool until job recovery; active calls are not retried or migrated
automatically.

## Scope and validation

- Supported: local v2 SGLang, native nonstreaming `/generate`. Images, sampling fields,
  routed-expert extensions and backend responses pass through as bytes.
- Unsupported configurations fail validation: v1, vLLM and external `api_url`. The
  common routing module can be reused by a future v1 adapter.
- Pools are static. `/configure_backend` is rejected while model routing is on. Use
  Controller lifecycle methods for pool-wide control; direct per-worker management calls
  do not establish the shared training barrier.
- Authenticated router `/status` reports versions, bindings and unfinished calls per
  backend. Bindings are released on session export/removal; abandoned session bindings
  remain until pool teardown to preserve affinity.

CPU tests cover actual HTTP forwarding, group/multimodal export, affinity, version
barriers and failures. GPU acceptance must additionally verify SGLang pause/abort
behavior for requests that arrive after the physical pause, numerical parity and
throughput on the intended multi-node deployment.
