# AReaL v1.0.0 Design Doc

---

Update 20250710

SFT example:

```bash
torchrun --nnodes 1 --nproc-per-node 8 examples/arealite/gsm8k_sft.py --config examples/arealite/configs/gsm8k_sft.yaml
```

---


We will provide both single-controller and SPMD user interfaces. The SPMD interface will be delivered with AReaLite, which is the paradigm most users are familiar with, just like using `torchrun` or `deepspeed`. However, this paradigm may lack some flexibility over global scheduling and control. To unlock the full potential with customized distributed execution, we will also provide a single-controller mode just like using Ray --- but our scheduler backend will not be restricted to Ray. Our code will be able to run with any scheduler in the cluster, such as native SLURM and K8S.

However, we want the user code to stay the same for both modes. The following is a simple usage example:

```python
def get_current_time():
    ...

def my_reward_fn(prompt, completion, prompt_ids, completion_ids, **kwargs):
    return len(completion_ids)

class MyRolloutWorkflow:
    def __init__(self, config: Any):
        self.config = config
        self.env = LocalToolingEnv()
        self.env.register_tool(get_current_time)

    async def arun_episode(self, engine: InferenceEngine, 
                           data: Dict[str, Any]) -> Dict[str, Tensor]:
        ...
        message = [
            {"role": "system", "message": ...},
            {"role": "user", "message": ...},
        ]
        for _ in range(self.config.num_turns):
            text = tokenizer.apply_chat_template(message, tools=self.env.list_tools())
            req = LLMRequest(text=text, ...)
            resp = await engine.agenerate(req)
            tool_name, tool_args = parse_tool(resp)
            cur_time = await self.env.aexecute(tool_name, tool_args)
            message += [{"role": "user", "message": f"The current time is {cur_time}"}]
        reward = my_reward_fn(None, None, None, req.input_ids, **data)
        ...
        return output

def main_grpo():
    dataset = load_dataset("openai/gsm8k", split="train")

    rollout_config, training_config = load_expr_config(sys.argv[1:])

    # Single-controller mode initialization
    scheduler = SlurmScheduler()
    rollout = RolloutController(
        SGLangEngine(rollout_config.engine),
        rollout_config.controller,
        scheduler,
    )
    actor = TrainController(
        MegatronGRPOActor(training_config.actor),
        config.training_controller_config,
        scheduler,
    )
    ref = TrainController(
        MegatronGRPOActor(training_config.ref),
        config.training_controller_config,
        scheduler,
    )
    # SPMD mode initialization
    # rollout = RemoteSGLangEngine(rollout_config.engine)
    # actor = MegatronGRPOActor(training_config.actor)
    # ref = MegatronGRPOActor(training_config.ref)

    rollout.initialize()
    actor.initialize()
    ref.initialize()

    # Synchronous RL
    dataloader = StatefulDataloader(dataset)
    for epoch in range(config.epoches):
        data_generator = iter(dataloader)
        for prompt in range(steps_per_epoch):
            prompt = next(data_generator)

            # Update inference engine weights
            future = rollout.update_weights(wcfg)
            actor.upload_weights(wcfg)
            future.result()

            # synchronous rollout
            rollout_batch = rollout.rollout(batch, workflow=MyRolloutWorkflow(rollout_config.workflow))
            # or asynchronous rollout with filtering and off-policyness control
            # rollout_batch = rollout.prepare_batch(batch, 
            #                                       workflow=MyRolloutWorkflow(rollout_config.workflow),
            #                                       should_accept=lambda x: x['rewards'].mean() > 0)

            # In the single-controller mode
            rollout_batch: DistributedBatch
            x: TensorDict = rollout_batch.load_data()
            # In the SPMD mode
            # rollout_batch: TensorDict

            batch['input_ids'] = rollout_batch['input_ids']
            batch['rewards'] = rollout_batch['rewards']

            # prepare train inputs
            batch['ref_logp'] = ref.compute_logp(batch)
            adv_batch = actor.compute_advantages_and_returns(batch)
            batch['advantages'] = adv_batch['advantages']

            # PPO update
            stats = actor.ppo_update(batch)
            print(stats)

if __name__ == "__main__":
    main_grpo()
```

The launch commands will be:

```bash
# Single-controller mode
python3 main_grpo.py --config my_config.yaml rollout.workflow.x=1
# SPMD mode
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m sglang.launch_server \
  --seed 1 --host x.x.x.x --port 7777 --dp_size 2 > server.out 2>&1 &
CUDA_VISIBLE_DEVICES=3,4 \
  torchrun --nnodes 1 --nproc-per-node 2 \
  main_grpo.py --config my_config.yaml \
  rollout.workflow.x=1 \
  rollout.engine.addresses="\[x.x.x.x, y.y.y.y\]"
```

## Core API

+ A specific algorithm must use these core components.
+ Concrete implementations must follow the API definition.

### TrainEngine

TrainEngine is a thin wrapper around existing training frameworks (FSDP, Megatron), providing a unified interface for RL algorithms for computation, parameter saving and loading, and providing a unified weight update interface for inference engines.

```python
#############################################
@dataclass
class WeightUpdateMeta:
    type: str
    path: str | None
    alloc_mode: AllocationMode | None
    comm_backend: str | None

@dataclass
class SaveLoadMeta:
    path: str
    weight_format: str
    with_optim: bool
    tokenizer: PreTrainedTokenizerFast | None
    base_model_path: str | None

class TrainEngine(abc.ABC):
    # control api
    def __init__(self, para_config)
        self.para_config = para_config
    
    def initialize(self, addr: str|None, ft_spec|None):
        # Initialize distributed environment, initialize and load model
        # addr is the corresponding service address when deploying sglang or fsdp/megatron remotely
        # The controller passes the master addr when calling
        pass

    def get_scheduling_config(self):
        # Get the resource configuration information required by the scheduler to schedule the engine,
        # such as the engine's image, cpu/gpu/memory size
        pass

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        pass

    async def upload_weights(self, meta: WeightUpdateMeta):
        pass

    def save(self, meta: SaveLoadMeta):
        pass

    def load(self, meta: SaveLoadMeta):
        pass

    # data api
    def step_lr_scheduler(self):
        """Step learning rate scheduler."""
        # Due to PPO minibatch updates, multiple train batches may need to be called
        # before calling step_lr_scheduler once, so this api needs to be separated
        raise NotImplementedError()
    
    def train_batch(
        self,
        input_: Dict,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict[str, float]:
        pass

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> torch.Tensor | None:
        pass

   @torch.no_grad()
    def forward(
        self,
        input_: Dict,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        pass
#############################################

# Implementation example
class FSDPEngine(TrainEngine):
    def __init__(self, config: EngineConfig):
        self.config = config
        self.weight_update_group_initialized = False

    def initialize(self, addr: str|None, ft_spec: FinetuneSpec):
        self.model_config = AutoConfig.from_pretrained(self.config.path)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(self.model_config)
        self.model = FSDP(model)
        self.optimizer = Adam(self.model.parameters())

    def destroy(self):
        del self.optimizer
        del self.model 
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    async def upload_weights(self, meta: WeightUpdateMeta):
        if meta.type == 'nccl':
            if not self.weight_update_group_initialized:
                await self.init_distributed_weight_update(meta)
            return await self.aupdate_weights_from_distributed()
        if meta.type == 'disk':
            self.save_to_hf(meta.path)
            return

    def save(self, meta):
        if meta.weight_format == 'hf':
            self.save_to_hf(meta.path, meta.tokenizer, meta.base_model_path)
        elif meta.weight_format == 'dcp':
            self.save_model_to_dcp(meta.path)

        if meta.with_optim:
            self.save_optimizer_state(meta.path)

    def load(self, meta):
        ...
    
    
    ############# Helper methods start #############

    def load_from_hf(self, path):
        sd = load_hf_state_dict(path)
        load_fsdp_state_dict(self.model, full_sd=sd)

    def save_to_hf(self, path,
        tokenizer: Optional[transformers.PreTrainedTokenizerFast] = None,
        base_model_path: Optional[str] = None,
                  ):
        if dist.rank() == 0:
            sd = {}
            for n, p in self.model.named_parameters():
                sd[n] = p.data.gather()
            torch.save(sd, path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        if base_model_path is not None:
            copy_hf_configs(base_model_path, path)
        dist.barrier()
        
    def save_for_recover(self, path: str):
        self.save_model_to_dcp(path)
        self.save_optimizer_state(path)

    def load_from_recover(self, path):
        self.load_model_from_dcp(path)
        self.load_optimizer_state(path)
        
    async def init_distributed_weight_update(self, meta: WeightUpdateMeta):
        # Initialize NCCL communication group for weight updates
        ...

    async def aupdate_weights_from_distributed(self):
        # Update inference weights through NCCL
        # Different engines (FSDP, Megatron) may have different weight aggregation,
        # splitting and communication methods
        # Keep this high-level interface instead of defining subdivided interfaces
        # to facilitate different engines implementing the most efficient weight communication path
        ...

    def load_model_from_dcp(self, path: str):
        # Load pytorch distributed checkpoint model from disk during recovery
        ...

    def save_model_to_dcp(self, path: str):
        # Save model in dcp format for recovery
        ...

    def save_optimizer_state(self, path: str):
        # Save optimizer state for recovery
        raise NotImplementedError()

    def load_optimizer_state(self, path: str):
        # Load optimizer state during recovery
        raise NotImplementedError()
```

### Algorithm-Specific TrainEngine API

Extended engines (such as Actor in PPO) provide convenient organization and calling of computational interfaces specific to algorithms. These computational interfaces maintain single-process computational logic, but can be called by controllers in top-level training scripts to complete distributed semantic computational orchestration.

```python
class Actor(Engine):
    
    @torch.no_grad()
    def compute_logps(self, input_: Dict[str, Tensor]) -> torch.Tensor:
        ... # unpad
        logps = self.forward(xxx)
        ... # pad back
        return logps

    def compute_advantages_and_returns(self, input_: Dict) -> Dict:
        pass

    def ppo_update(self, input_: Dict) -> List[Dict[str, float]]:
        ...
        all_stats = []
        for _ in range(self.ppo_n_minibatches):
            stats = self.train_batch(xxx, loss_fn=actor_loss_fn)
            all_stats.append(stats)
        return all_stats

class Critic(Engine):
    
    @torch.no_grad()
    def compute_values(self, input_: Dict) -> torch.Tensor:
        pass

    def ppo_update(self, input_: Dict) -> List[Dict[str, float]]:
        ...
        all_stats = []
        for _ in range(self.ppo_n_minibatches):
            stats = self.engine.train_batch(xxx, loss_fn=critic_loss_fn)
            all_stats.append(stats)
        return all_stats

class FSDPActor(FSDPEngine, Actor):
    pass

class MegatronActor(FSDPEngine, Actor):
    pass

class FSDPCritic(MegatronEngine, Critic):
    pass

class MegatronCritic(MegatronEngine, Critic):
    pass
```

### Inference Engine API

Define the InferenceEngine API in a local-like mode, rather than a client-server separated form, mainly for user-centered convenience in using the InferenceEngine as a tool.

InferenceEngine can internally start a SGLang subprocess (SGLangEngine), or call a remotely deployed service (RemoteSGLangEngine).

```python
class InferenceEngine(ABC):
    def __init__(self, config)
        self.config = config
        
    @abstractmethod
    def initialize(self, addr: str|None, ft_spec):
        """Start SGLang Engine, starting the engine will call model loading by default
        """
        self.tasks = []

    async def update_weights(self, meta) -> None:
        # Update model weights based on meta information
        pass

    async def agenerate(self, req: LLMRequest) -> LLMResponse:
        # Given a prompt, generate a response with LLM
        pass

    def submit(self, data: Dict[str, Any], workflow):
        """
        Asynchronously submit rollout request
        """
        task = asyncio.create_task(workflow.arun_episode(self, data))
        self.tasks.append(task)

    async def _wait_async(self, count: int, timeout: int) -> DistributedBatch:
        tik = time.time()
        n = 0
        results = []
        while time.time() - tik < timeout and n < count:
            done, _ = await asyncio.wait(self.tasks, return_when=FIRST_COMPLETED)
            for task in done:
                results.append(await task)
        if n < count:
            raise TimeoutError()
        return DistributedBatch(results)

    def wait(self, count: int, timeout: int) -> DistributedBatch:
        """
        Synchronous wait interface, until the request returns count records
        """
        return asyncio.run(self._wait_async(count, timeout))

    @abstractmethod
    def rollout(self, data: List[Dict[str, Any]], workflow) -> DistributedBatch:
        """
        Synchronously submit rollout request, until all inference requests are completed and returned
        """
        pass

###################################
# Implementation example
class SGLangEngine(InferenceEngine):

    def __init__(self, config: InfEngineConfig):
        self.config = config 
        self.weight_update_group_initialized = False

    async def update_weights(self, meta) -> None:
        if meta.type == 'nccl':
            if not self.weight_update_group_initialized:
                await self.init_distributed_weight_update(meta)
            return await self.aupdate_weights_from_distributed()
        if meta.type == 'disk':
            self.update_weights_from_disk(meta.path)
            return

    async def agenerate(self, req: LLMRequest) -> LLMResponse:
        # Given a prompt, generate a response with LLM
        return await self.llm.generate_async(xxx)
        
    # Weight update
    @abstractmethod
    def update_weights_from_disk(self, path) -> None:
        """Update model weights from disk"""
        ...

    @abstractmethod
    async def ainit_distributed_weights_update(self, meta_info: WeightUpdateMeta):
        # Initialize **all** needed weight synchronization communication groups and communication plans
        # (which communication type and which parameters to communicate at each step)
        # Depending on the engine's partitioning method, multiple communication groups may be initialized
        # for weight synchronization
        # Since both inference and training engines need to enter this function,
        # it needs to be defined as an async function
        ...

    @abstractmethod
    async def aupdate_weights_from_distributed(self) -> None:
        """Use the initialized weight communication plan and communication groups to update model weights with NCCL

        Since both inference and training engines need to enter this function,
        it needs to be defined as an async function
        """
        ...

    @abstractmethod
    def check_health(self) -> bool:
        """Check server health status
        
        Returns:
            bool: Whether the server is healthy
        """
        pass
```

### RolloutWorkflow

RolloutWorkflow is a class that provides the arun_episode method. This method has a fixed signature, used to collect one agent trajectory.

```python
class MyRolloutWorkflow:
    def __init__(self, config: Any):
        self.config = config
        self.tool_executor = ToolExecutor()
        self.tool_executor.register_tool(get_current_time)

    async def arun_episode(self, engine: InferenceEngine, 
                           data: Dict[str, Any]) -> Dict[str, Tensor]:
        ...
        req = LLMRequest(input_ids=data['input_ids'], ...)
        for _ in range(self.config.num_turns):
            resp = await engine.agenerate(req)
            res = await self.tool_executor.aexecute_tool(resp.completion)
            req.input_ids += res
        reward = my_reward_fn(None, None, None, req.input_ids, **data)
        ...
        return output
```

### RolloutController & TrainController

They have the same API as `InferenceEngine` and `TrainEngine`, respectively.

## Other Components

1. Algorithm workflows don't necessarily use these components; other replaceable components such as implementations in rllm or verl-agent can also be used
2. Mainly for internal implementation and division of labor, as a thin wrapper to facilitate adaptation of external packages

### Environment

1. Support multi-tool management and unified execution interface
2. Support local calling and mcp service calling
3. "Tools" belong to an instance rather than a class, register_tool is defined as a method rather than a static method, this is (1) to prevent tools from subclasses being registered to the base class, causing potential naming or calling conflicts; (2) to support multiple tasks for the same service (e.g., browser), with each task having a different toolset

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ToolingEnv:
     def __init__(self):
        self.is_initialized = False
        
        self.tool_registry: Dict[str, Callable] = {}
        self.tool_schemas: List[Dict[str, Any]] = []
         
    async def ainitialize(self):
        """
        Performs the initialization logic for the environment.
        For stateful environments, this is where resources are created and 
        prepared (e.g., launching a browser).
        """
        pass

    def list_tools(self) -> List[Dict[str, Any]]:
        pass

    async def aexecute(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        pass

    async def aclose(self):
        """
        Destroys the environment, releasing all held resources.
        This method is critical for stateful environments (e.g., a browser session).
        """
        pass

class MCPToolingEnv(ToolingEnv):
    def __init__(self, config: MCPToolingEnvConfig):
        self.config = config
        # init a mcp client
        self.mcp_client = mcp.session_client(config.url)
        self.tool_registry: Dict[str, Callable] = mcp.session_client.list_tools()
        self.tool_schemas: List[Dict[str, Any]] = []

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.tool_schemas

    def aexecute(self, tool_name: str, tool_args: Dict[str, Any]):
        pass

    
class LocalToolingEnv(ToolingEnv):

    @staticmethod
    def generate_schema(func: Callable) -> Dict[str, Any]:
        """
        Generates a JSON schema for a function using introspection.
        """
        # Use the function's docstring as the tool's description.
        description = inspect.getdoc(func) or "No description provided."
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        
        # Mapping from Python types to JSON Schema types
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        for name, param in sig.parameters.items():
            # Default to string type if type hint is missing or complex
            param_type = type_mapping.get(param.annotation, "string") 
            parameters["properties"][name] = {"type": param_type}
            
            # If a parameter has no default value, it is considered required.
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(name)
        
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters,
            }
        }

    def register_tool(self, func: Callable) -> Callable:
        """
        A decorator that registers a Python function as a tool in this environment.
        """
        if not callable(func):
            raise TypeError("The provided object must be a callable function.")
            
        tool_name = func.__name__
        if tool_name in self.tool_registry:
            raise ValueError(f"Tool with name '{tool_name}' is already registered.")
            
        # Add the function to the registry and its schema to the schema list.
        self.tool_registry[tool_name] = func
        self.tool_schemas.append(self.generate_schema(func))

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists all available tools provided by this environment and their descriptions.
        
        Returns:
            A list of dictionaries, where each dictionary describes a tool.
        """
        return self.tool_schemas

    async def aexecute(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Executes a specified tool.

        Args:
            tool_name (str): The name of the tool to execute.
            tool_args (Dict[str, Any]): The arguments required to call the tool.

        Returns:
            Any: The result of the tool's execution, typically a string or 
                 structured JSON.
        """
        if tool_name not in self._tool_registry:
            return f"Error: Tool '{tool_name}' is not registered."
            
        tool_func = self._tool_registry[tool_name]
        
        try:
            result = tool_func(**tool_args)
            return result
        except TypeError as e:
            # This exception is often triggered by missing or incorrect argument types.
            return f"Error executing '{tool_name}': Invalid arguments. Details: {e}"
        except Exception as e:
            return f"Error executing '{tool_name}': An unexpected error occurred. Details: {e}"
    
```

### Reward

1. Workflows for computing rewards using models and computing rewards based on rules should be separated
2. Rule-based reward computation is defined as a function with a predefined signature, which can be local or remote, and is generally wrapped in the rollout workflow; user-written reward functions can also not use this signature as long as they provide a workflow
3. Computing rewards using models requires users to initialize a controller/engine themselves and explicitly call it in the algorithm workflow

```python

############ Rule-based reward ############

# The signature is just a reference, can be defined arbitrarily
def reward_fn(prompt: str, completion: str, prompt_ids: List[int],
              completion_ids: List[int], **kwargs):
    # prompt: prompt string (the task this data needs to complete)
    # completion: trajectory string generated by the model based on the task
    # prompt_ids: token ids of the prompt
    # completion_ids: token ids of the trajectory generated by the model
    # kwargs: all other attributes of this data in the dataset,
    #         for example, solutions, input_outputs, etc.
    pass

############ Model-based reward ############

reward = TrainController(Critic())
rollout_controller = RolloutController(...)
for _ in range(epochs):
    for _ in range(steps_per_epoch):
        data = rollout_controller.rollout(prompt)
        data['reward'] = reward.compute_values(data)
        ...
```

### Dataset

Use huggingface datasets and pytorch torchdata. In Single-Controller mode, only one process per experiment is responsible for data loading.

```python
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

dataset = load_dataset(
    path,
    name=name,
    split=split,
    data_files=data_files,
)
dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

# Access the first data item
data: Dict = dataset[0]

# Access the "prompt" column
data: List = data['prompt']

# Data processing
def process_example(example, idx):
    # Add query_id column
    example["query_id"] = str(idx)
    example["prompt"] = example["question"]

    # used by the reward function
    example["method"] = reward_mode
    return example

dataset = dataset.map(
    lambda example, idx: process_example(example, idx),
    with_indices=True,
)

# Data loading and shuffle
from torchdata.stateful_dataloader import StatefulDataLoader
dataloader = StatefulDataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=lambda x: x,  # Can change the data batch packing method by changing this parameter
)
for data in dataloader:
    assert isinstance(data, list)
```