# Model Worker

## Master-Model Worker Interaction

The master worker sends remote procedure calls (RPCs) to model workers to execute actual computations like `actor_train`. The figure below illustrates their interaction throughout an experiment:

![](master-model-interaction.png)

Model worker "compute" involves running a model interface with a specific backend (covered in detail later). For synchronous PPO algorithms, model workers sequentially execute:

+ `actor_gen`: `actor` model with SGLang backend + `PPOActorInterface.generate`
+ `rew_inf`: `reward` model (can be null for RLVR) + `MultiTaskRewardInterface.inference`  
+ `actor_train`: `actor` model with Megatron backend + `PPOActorInterface.train_step`

```{note}
For asynchronous PPO, only `actor_train` is executed.
```

## Communication Protocol

### Request-Reply Pattern

The master worker and model workers communicate through a `request_reply_stream` channel that handles requests and metadata responses. Actual data like `input_ids` transfers through other channels.

The master (client) can send these requests to model workers (servers):

+ **fetch**: Worker loads local dataset data and sends metadata (e.g., sequence length) to master for buffer storage
+ **spec**: Worker returns dataset specifications for master to calculate experiment steps
+ **model_config**: Worker provides transformer model configuration
+ **clear_data_cache**: Worker clears data transfer and GPU caches
+ **initialize**: Worker initializes parameters, gradient buffers, and optimizer states
+ **generate/inference/train_step**: Worker executes corresponding computation (note: "inference" refers to a single forward pass)

### Request Hooks

Computation requests ("generate"/"inference"/"train_step") support pre- and post-hooks for:

+ Data transfer (pre-hook)
+ Evaluation
+ Offloading  
+ Parameter reallocation
+ Checkpointing (post-hooks)

These hooks often require NCCL communication and synchronization between workers. Implementing them as dedicated hooks prevents deadlocks that could occur if these operations were interleaved with other NCCL communications.

### Request Types

+ **Blocking requests**: Long-running operations requiring NCCL synchronization. Workers can't execute immediately since concurrent blocking requests may need coordinated data transfers. The master sends a "flush" request to indicate that all concurrent requests have been sent.
+ **Non-blocking requests**: Shorter operations without NCCL requirements that can execute immediately.

## Data Management

### Distributed Dataset Storage

Datasets are distributed across model workers without overlap. For each model:

+ Processes with PP rank = -1 and TP rank = 0 serve as DP heads
+ Data is stored on DP heads of the model used in the first MFC (e.g., actor model DP heads for PPO)

During "fetch" requests:

1. DP head worker loads data into local buffer
2. Sends metadata to master
3. Master tracks metadata and later instructs workers which data to use for each MFC via computation request hooks

```{note}
For asynchronous RL, the "dataset" will be a `StreamDataset` instance that pulls data from the rollout worker. After data is loaded, the subsequent MFC calls follow the same procedure as described above.
```

### Data Transfer Process

For each MFC, the master:

1. Specifies which data to use
2. Provides data locations across workers
3. Workers redistribute data using:
    - `Redistributor`: Generates NCCL broadcast/gather/scatter communication plan  
    - `DataManager`: Executes the plan

After redistribution, workers with the same DP rank receive identical input data.

### MFC Output Handling

Only workers with PP rank=-1 and TP rank=0 produce output data. These workers:

1. Store data locally
2. Notify master of data locations
3. Master generates new redistribution plans for subsequent MFCs based on this layout information