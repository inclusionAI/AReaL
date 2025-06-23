# AReaLite (TBD)

A simplified and easy-to-read version of AReaL with a minimal set of APIs.

## Expected Usage

Launch LLM servers and trainers at the same time according to the allocation mode:

```bash
python3 arealite.cli.main \
    experiment_name=my-exp trial_name=my-trial \
    mode=${torchrun|ray|slurm} \
    allocation_mode=sglang.d16p1m1+d32p2m1 \
    shutdown_server_on_exit=False \
    trainer.type=ppo trainer.ppo.async_training=True
```

Add new servers elastically:

```bash
python3 arealite.cli.launch_server \
    experiment_name=my-exp trial_name=my-trial
```

If the trainer dies, restart the experiment without re-launching the server:

```bash
python3 arealite.cli.main \
    experiment_name=my-exp trial_name=my-trial \
    min_required_servers=17
```

Run rollout or evaluation only:

```bash
python3 arealite.cli.main trainer.type=null \
    valid_dataset.path=huggingface/dataset
```

## How to use in its current form (Dev WIP)

1. Launch a wrapped SGLang server:

```bash
python3 arealite/cli/launch_server.py experiment_name=test_rollout trial_name=test_rollout
```

This command reads the configuration from `config/llm_server.yaml` and launches a local SGLang server. It registers the server address for later use by the trainer.

2. Run simple tests:

```bash
python3 arealite/tests/test_rollout.py
```

This command uses the same experiment and trial names as those used to launch LLM servers, so the test automatically finds the server address. You can then run the rollout loop with a pre-defined workflow to collect training data.

## Customization

All customizations follow similar procedures:

1. Inherit the base class (e.g., `Trainer`) and write your customized object under the `impl/` folder (e.g., `impl/trainer/ppo.py`).

2. Modify the factory class in the API file (e.g., `api/trainer_api.py`) to allow initialization of your customized class, e.g.:

```diff
@dataclass
class TrainerFactory:
    args: TrainingArgs

    def make_trainer(
        self,
        config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional["RolloutController"] = None,
        extra_args: Optional[Dict] = None,
    ) -> Trainer:
        if config.type == "ppo":
            from arealite.impl.trainer.ppo import SpmdPPOTrainer

            return SpmdPPOTrainer(
                self.args,
                config,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                rollout_controller=rollout_controller,
                extra_args=extra_args,
            )
+       if config.type == "sft":
+           from arealite.impl.trainer.sft import SpmdSFTTrainer
+           return SpmdSFTTrainer(
+               ...
+           )
        else:
            raise NotImplementedError(f"Unknown agent type: {config.type}")
```

3. Modify the CLI args so you can customize your agent through the command line:

```python
@dataclass
class SFTTrainerConfig:
    model: EngineConfig = field(
        default_factory=EngineConfig,
        metadata={"help": "Model configuration for SFT training"},
    )
    mb_spec: MicroBatchSpec = field(
        default_factory=MicroBatchSpec,
        metadata={"help": "Micro-batch specification for SFT training"},
    )
```

```diff
@dataclass
class TrainerConfig:
    type: str = field(
        default="ppo",
        metadata={"help": "Trainer type", "choices": ["ppo", "sft", "null"]},
    )
    ppo: Optional[PPOTrainerConfig] = field(
        default=None, metadata={"help": "PPO trainer configuration (if using PPO)"}
    )
+   sft: Optional[SFTTrainerConfig] = field(
+       default=None, metadata={"help": "SFT trainer configuration (if using SFT)"}
+   )

```

Then you can use your own trainer with a command like:

```bash
python3 train.py trainer.sft.model.path=Qwen/Qwen2-0.5B
```

## Unit Tests

Unit tests are placed under the `tests` folder, but currently they are essentially not *unit* tests but rather standalone scripts to ensure that individual components are runnable. 

They need further refactoring to use `pytest`.

## TODOs

- [ ] Finalize API design. (In-progress)
- [x] Implement standalone SGLang server (`impl/sglang_server.py`).
- [x] Implement SGLang client generation (`impl/sglang_client.py`).
- [x] Rollout pipeline (`tests/test_rollout.py`).
- [ ] FSDP2 engine with transformers models. (In-progress)
- [ ] SGLang update weights. (In-progress)
- [ ] Synchronous PPO training pipeline (`impl/trainer/ppo.py`). (In-progress)
- [x] SGLang rollout interruption.
- [x] Asynchronous RL system-wide utilities (e.g., `RolloutController`).
- [ ] CI and unittests. (FSDP trainer and PPO trainer in-progress.)
- [ ] Benchmark performance versus the original AReaL code.
- [ ] Various launching scripts: ray, torchrun, slurm.
- [ ] Design doc and user guide for transitting from v0.3.0.
- [ ] Examples of training GSM8K, TLDR, and a seach agent.
- [ ] Allow external persistent SGLang servers for debugging purposes.