# Local SWE rollout

This prepares CPU sandboxes and a bounded model-driven rollout through AReaL v2 on one
Linux machine. The standalone command collects version-0 trajectories on one inference
GPU; sandboxes use CPU. The [automatic runner](../README.md) reuses this producer with
training and weight publication between fresh episodes.

Each episode has a fresh SWE-bench task container. A companion container copies a static
persistent-shell server and the upstream grading helpers into a shared volume. The shell
server runs inside the task container, so commands execute in its repository and Python
environment. AEnvironment creates and deletes these pods; the client reaches their MCP
endpoints through the API proxy.

## Sources and prerequisites

The checked versions are:

| Component                                                         | Version / commit                                               |
| ----------------------------------------------------------------- | -------------------------------------------------------------- |
| AReaL                                                             | `v2.1.0`                                                       |
| [AEnvironment](https://github.com/inclusionAI/AEnvironment)       | `667f743428d358f4fb84857835d447cd1d9b31ba`                     |
| [AReaL-SWEAgent](https://github.com/areal-project/AReaL-SWEAgent) | `f144900d5e0a019bf5381b624c442b49999ffa4d`                     |
| kind                                                              | `v0.33.0`                                                      |
| kind node / kubectl                                               | `v1.34.11`                                                     |
| Go toolchain used for the static services                         | `go1.26.9`, Linux amd64                                        |
| Task image                                                        | `swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-7432:latest` |

Use Docker, Helm 3, kind, kubectl, Go, Python 3.12, and uv on Linux amd64. Docker must
be accessible to the project user. Set the following absolute paths for your machine;
keep generated files outside the source checkout:

```bash
export PROJECT_ROOT=/path/to/project-data
export AREAL_ROOT=/path/to/AReaL
export AENV_ROOT=/path/to/AEnvironment
export SWE_AGENT_ROOT=/path/to/AReaL-SWEAgent
export SETUP="$AREAL_ROOT/examples/autopilot/elastic/swe"
export BUILD="$PROJECT_ROOT/swe/build"
export KUBECONFIG="$PROJECT_ROOT/kind/kubeconfig"
mkdir -p "$BUILD/bin/swe_eval" "$PROJECT_ROOT/kind" "$PROJECT_ROOT/logs"
```

Check available disk space before downloading models or task images. Docker stores both
the host images and the copies imported into kind. This preflight reuses one cached SWE
image rather than downloading the entire benchmark's images. The kind config uses a 10
GiB absolute disk eviction floor for the shared development disk.

## Build and deploy

Build static Linux binaries from the pinned source checkouts:

```bash
(cd "$SWE_AGENT_ROOT/cc_golang_bash" && CGO_ENABLED=0 go build -o "$BUILD/bin/persistent-bash" . && go test ./...)
(cd "$AENV_ROOT/controller" && CGO_ENABLED=0 go build -o "$BUILD/bin/controller" ./cmd)
(cd "$AENV_ROOT/envhub" && CGO_ENABLED=0 go build -o "$BUILD/bin/envhub" .)
(cd "$AENV_ROOT/api-service" && CGO_ENABLED=0 go build -o "$BUILD/bin/api-service" .)
cp "$AENV_ROOT"/aenv/builtin-envs/swebench/src/eval/*.py "$BUILD/bin/swe_eval/"

docker build -f "$SETUP/Dockerfile.shell" -t autopilot-swe-shell:0.1.0 "$BUILD/bin"
for component in controller envhub api-service; do
  docker build -f "$SETUP/Dockerfile.service" --build-arg COMPONENT="$component" \
    -t "autopilot-aenv-$component:0.1.0" "$BUILD/bin"
done
docker tag swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-7432:latest \
  autopilot-swe-tasks:pytest-dev__pytest-7432

kind create cluster --name autopilot-swe --image kindest/node:v1.34.11 \
  --config "$SETUP/kind.yaml" --kubeconfig "$KUBECONFIG" --wait 120s
kind load docker-image --name autopilot-swe \
  autopilot-swe-shell:0.1.0 autopilot-aenv-controller:0.1.0 \
  autopilot-aenv-envhub:0.1.0 autopilot-aenv-api-service:0.1.0 \
  redis:alpine autopilot-swe-tasks:pytest-dev__pytest-7432

helm dependency update "$AENV_ROOT/deploy" --skip-refresh
helm lint "$AENV_ROOT/deploy" -f "$SETUP/aenv-values.yaml"
helm upgrade --install autopilot-aenv "$AENV_ROOT/deploy" \
  --kubeconfig "$KUBECONFIG" --namespace aenv \
  -f "$SETUP/aenv-values.yaml" --wait --timeout 180s
kubectl --kubeconfig "$KUBECONFIG" get pods -n aenv
```

Have `busybox:1.37.0`, `redis:alpine`, and `kindest/node:v1.34.11` available in Docker
before the commands above. If the host cannot reach a registry with valid TLS, download
on a machine that can, transfer using SSH, and use `docker load`. Keep TLS verification
enabled. With locally cached images, `DOCKER_BUILDKIT=0 docker build` also avoids
BuildKit attempting a remote base-image metadata lookup.

Do not add Helm's `--create-namespace`: this chart already creates its namespaces. The
values set `global.sandboxNamespace` explicitly to prevent duplicate OSS secrets in the
control namespace, and disable the duplicate `api-service-k8s` subchart. The registered
datasource must be an image **tag**, because this API constructs the task image as
`imagePrefix:datasource`.

Run these port forwards in separate terminals and keep them running:

```bash
kubectl --kubeconfig "$KUBECONFIG" -n aenv port-forward --address 127.0.0.1 \
  svc/api-service 8080:8080 8081:8081
kubectl --kubeconfig "$KUBECONFIG" -n aenv port-forward --address 127.0.0.1 \
  svc/envhub 8083:8083
```

Register once, then query the registration on subsequent runs:

```bash
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @"$SETUP/environment.json" http://127.0.0.1:8083/env/
curl --fail-with-body http://127.0.0.1:8083/env/autopilot-swe-bash/0.1.0
```

This configuration is an isolated development deployment, with ephemeral Redis state and
loopback access. It mounts neither the host Docker socket nor a Kubernetes service
account token into the task pod.

## Verify a real task without a model

Install the sandbox SDK in a dedicated environment. This preflight imports only
`aweagent.envs.swe` from the source checkout:

```bash
uv venv --python python3.12 "$PROJECT_ROOT/swe/client-venv"
uv --no-config pip install --python "$PROJECT_ROOT/swe/client-venv/bin/python" \
  -e "$AENV_ROOT/aenv" pydantic==2.11.10 loguru
export PYTHONPATH="$AREAL_ROOT:$SWE_AGENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AENV_SYSTEM_URL=http://127.0.0.1
export LOG_DIR="$PROJECT_ROOT/logs/swe"
```

The SDK requires Pydantic \<2.12 while OpenAI Agents requires >=2.12.3. The rollout
keeps them in separate Python processes connected by async JSON-lines RPC; neither
upstream checkout needs modification. `--no-config` prevents project uv overrides
replacing pins.

Obtain the `pytest-dev__pytest-7432` row, including the gold patch, from
[SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified).
Extract that instance's `details` object from the
[grading metadata](https://www.modelscope.cn/datasets/kangoal/swebench-scripts/resolve/master/swe-verify-details.json)
referenced by the pinned AEnvironment SWE image. Save the two objects as JSON. Then run:

```bash
python3 "$SETUP/prepare_fixture.py" --row /path/to/row.json \
  --details /path/to/details.json --output "$PROJECT_ROOT/swe/fixture.json"
"$PROJECT_ROOT/swe/client-venv/bin/python" -m examples.autopilot.elastic.swe.smoke_sandbox \
  --fixture "$PROJECT_ROOT/swe/fixture.json" --environment autopilot-swe-bash@0.1.0 \
  --report "$PROJECT_ROOT/swe/smoke-report.json" \
  --kubectl "$(command -v kubectl)" --kubeconfig "$KUBECONFIG"
```

The fixture builder checks the instance, repository, version, base commit, and reference
test lists. Its wrapper uses the upstream test-log parser and F2P/P2P grader: the raw
shell script may return zero after resetting files even when pytest failed. Missing
reference tests or an incomplete report fail validation.

The smoke test expects empty-patch reward **0** and gold-patch reward **1**, using
separate sandboxes. It checks persistent shell state, nonzero command exits, patch
applicability, completed grading, and pod deletion. Kubernetes queries use the explicit
kubeconfig and do not interpret an API/network error as successful deletion. When
running the client in Docker, use host networking and mount both the project directory
and these source checkouts at the paths used inside it.

Keep task data, generated fixtures, model files, reports, kubeconfig, and runtime logs
outside Git. The gold patch is used only to validate the environment; it is not an agent
rollout or a training result.

## Model rollout into the durable queue

Create a separate CPU environment for the pinned upstream agent:

```bash
uv venv --python python3.12 "$PROJECT_ROOT/swe/agent-venv"
uv --no-config pip install --python "$PROJECT_ROOT/swe/agent-venv/bin/python" \
  openai==2.24.0 openai-agents==0.10.5 pydantic==2.12.5 tenacity loguru pyyaml jinja2
```

Run from AReaL's CUDA runtime with both checkouts on `PYTHONPATH`. For Docker, use
`--init --network host`, expose the GPUs, and mount the paths used by both environments
at the same absolute paths inside the container. Set writable `XDG_CACHE_HOME`,
`FLASHINFER_WORKSPACE_BASE`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and
`TVM_FFI_CACHE_DIR`.

```bash
export PYTHONPATH="$AREAL_ROOT:$SWE_AGENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$SETUP/rollout_to_queue.py" --model /path/to/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --fixture "$PROJECT_ROOT/swe/fixture.json" --environment autopilot-swe-bash@0.1.0 \
  --agent-python "$PROJECT_ROOT/swe/agent-venv/bin/python" \
  --sandbox-python "$PROJECT_ROOT/swe/client-venv/bin/python" \
  --queue "$PROJECT_ROOT/swe/trajectory-queue" --output "$PROJECT_ROOT/swe/rollout-001" \
  --kubectl "$(command -v kubectl)" --kubeconfig "$KUBECONFIG" --device 0
```

The output directory must be new. The coordinator and v2 guards/router/gateway/data
proxy have CUDA disabled; only the owned SGLang process sees the selected GPU. The
upstream SWEAgent receives the issue, tools, and session credentials, never the gold
patch. `agent-config.yaml` bounds the run to four model turns; an unfinished run grades
its actual current diff. Success requires a model-selected repository tool and a
completed grading marker; reward 0 is acceptable and does not establish SWE quality.

Exported RTensors are localized before inference teardown. The outbox and SQLite queue
store original token IDs, masks, behavior logprobs, policy versions, and rewards as
values. Version 0 refers to the fixed initial model; no weight update is claimed. The
outbox can retry enqueue idempotently. Each attempt namespaces sessions with a fresh run
ID so inference restarts cannot collide. Upstream export is destructive: a crash between
export and outbox persistence can still lose that rollout. This producer is not yet an
atomic end-to-end handoff. Only enqueue establishes queue durability.

Reports include the agent trace, patch, grading output, queue status, independent pod
deletion, and inference-process/GPU cleanup. Do not place reports or credentials in Git.

## Validation

On 2026-10-17, the real `pytest-dev__pytest-7432` task passed this preflight on Linux
amd64: empty patch reward 0, gold patch reward 1, shell state and command failure
preserved, and both sandbox pods independently confirmed deleted. The fixture unit tests
also passed. This run used CPU sandboxes and did not load a policy model or execute a
training step.

Run the CPU tests from the AReaL repository:

```bash
python -m pytest -q tests/test_autopilot_swe_fixture.py tests/test_elastic_trajectory.py
```

The [elastic runner](../README.md) adds PPO consumption, checked weight publication,
data-triggered training and idle GPU release. Its single-producer gate must cover the
complete episode through durable enqueue.
