# Qwen3.8-27B Arena RL

This recipe packages the dense 27B code and configuration used by the 19-task Graybox
experiment. The branch starts from upstream `main` at `448de523`. Arena categorized
retries, RPC/checkpoint fixes and required-recovery checks are already upstream. The
additional core fix carries `de8d6796`: resolve rollout dump paths without importing
metrics SDKs in inference workers.

The new files provide the dense controller/actor entry points, the exact training
parameters and Slurm submission preflights. They do not modify the running trial.

| Setting       | Value                                                             |
| ------------- | ----------------------------------------------------------------- |
| Sampling      | temperature 1, top-p 1, top-k 100000                              |
| Batch / group | bs8 / gs12; at most 16 concurrent rollout groups                  |
| Training      | TP4 / PP4 / CP4 / EP1; CPU Adam offload                           |
| Inference     | 16 TP4 replicas; CUDA graph, memory saver and radix cache enabled |
| Attention     | dense provider `auto`; native GDN defaults; no PLE override       |
| LM head       | native projection; chunked logits disabled                        |
| Reward        | group normalization uses standard deviation                       |
| R3            | disabled                                                          |
| Recovery      | auto; model/optimizer/LR/RNG state retained; save every two hours |

Actor/controller use **areal-dev-sglang-0821.sif**, with the historical dependency
overlay providing **Megatron-Core 0.18.2**, **megatron-bridge 0.5.1** and **AWEX
0.8.2**. Rollout uses **deepep-base-85b53914.sif**, containing **SGLang
0.5.18.dev10+g85b539146**; the proxy overlay supplies **litellm 1.83.7** and **fastuuid
0.14.0**. Keep the two images and their dependency paths separate. The recipe references
these installed packages; it does not vendor runtime wheels or private environment
files.

## Launch

1. Copy `site.env.example` to an external private file and fill it using the successful
   site's paths, URLs and credentials. `QWEN_REPO` is the clean checkout of this branch.
   `QWEN_RUN_DIR` stores this attempt's config/preflight reports; `QWEN_OUTPUT_ROOT`
   stores training logs and checkpoints. Both directories must be outside the checkout.
   Fill the cluster mounts and dependency roots from the existing working environment;
   they are required inputs.

1. Use a short new trial such as `q27-bs8gs12-1013-1`. Keep the experiment name
   `chucai-qwen38-graybox19-rl`. The YAML retains the original Graybox stream, harness
   and reward reference; replacing the dataset requires reviewing all three together.

1. Commit the reviewed source, protect the profile with `chmod 600`, then run:

   ```bash
   QWEN_ENV_FILE="$PRIVATE_SITE_ENV" bash examples/swe/rl/qwen38_27b/submit.sh
   ```

The controller verifies the source revision, durable storage, Arena authentication,
rollout imports, structured config, W&B name length and dense model provider before
allocating training workers. Arena and Slurm cleanup remain scoped to this trial. Native
SGLang logs rotate under the trial's `native/` log directory. Existing NUMA settings in
`train.yaml` are preserved, including `SGLANG_NUMA_BIND_V2=0`.

## Recovery

Reuse the same profile, trial, run directory and committed source. For an explicit
recovery, set `QWEN_REQUIRE_RECOVER=1` and `QWEN_PREVIOUS_SUBMISSION` to the previous
attempt's immutable JSON receipt containing `controller_job_id`, `actor_job_id`,
`rollout_job_id` and `cleanup_job_id`. All four jobs must be terminal. The host checks
Slurm state; the actor container checks the matching fresh report and the published
checkpoint manifest. These checks do not replace actual distributed optimizer loading.

The independent experiment observer and logp/GDN diagnostic candidates are outside this
recipe. FA3 adapters, forced stable top-k, cache isolation and tensor replay patches are
not added by these files.
