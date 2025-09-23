#!/usr/bin/env python3
"""
Script to automatically generate CLI documentation from areal.api.cli_args dataclasses.
This creates markdown documentation compatible with jupyter-book.
"""

import inspect
import sys
import types
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Union, get_args, get_origin

# Add the project root to the path so we can import areal
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from areal.api.cli_args import (
    BaseExperimentConfig,
    ClusterSpecConfig,
    DatasetConfig,
    EvaluatorConfig,
    GenerationHyperparameters,
    GRPOConfig,
    InferenceEngineConfig,
    LauncherConfig,
    MicroBatchSpec,
    NormConfig,
    OptimizerConfig,
    PPOActorConfig,
    RecoverConfig,
    RWConfig,
    SaverConfig,
    SchedulerConfig,
    SFTConfig,
    SGLangConfig,
    StatsLoggerConfig,
    TrainEngineConfig,
)


def get_type_description(field_type) -> str:
    """Convert a type annotation to a readable string."""
    # Handle union types (Type | None)
    origin = get_origin(field_type)
    # Check if it's a Union type (includes both Union[Type, None] and Type | None syntax)
    if origin is Union or isinstance(field_type, types.UnionType):
        args = get_args(field_type)
        # Check if it's a union with None (optional type)
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return f"{get_type_description(non_none_type)} &#124; None"
        else:
            # Multiple non-None types in union
            return " &#124; ".join(get_type_description(arg) for arg in args)

    # Handle basic types
    if field_type == int:
        return "integer"
    elif field_type == float:
        return "float"
    elif field_type == str:
        return "string"
    elif field_type == bool:
        return "boolean"
    elif field_type == list or get_origin(field_type) == list:
        if get_args(field_type):
            inner_type = get_args(field_type)[0]
            return f"list of {get_type_description(inner_type)}"
        return "list"
    elif hasattr(field_type, "__name__"):
        return f"`{field_type.__name__}`"
    else:
        return str(field_type).replace("typing.", "")


def format_default_value(field_obj) -> str:
    """Format default values for display."""

    if field_obj.default is not inspect._empty:
        default_value = field_obj.default
        # Check for MISSING by string representation to avoid import issues
        if (
            str(default_value).startswith("'???'")
            or str(default_value) == "MISSING"
            or "MISSING" in str(type(default_value))
        ):
            return "**Required**"
        elif default_value is None:
            return "`None`"
        elif isinstance(default_value, str):
            return f'`"{default_value}"`'
        elif isinstance(default_value, list) and len(default_value) == 0:
            return "`[]`"
        elif isinstance(default_value, bool):
            return f"`{default_value}`"
        elif "MISSING" in str(default_value):
            return "**Required**"
        else:
            return f"`{default_value}`"
    elif field_obj.default_factory is not inspect._empty:
        try:
            factory_result = field_obj.default_factory()
            if isinstance(factory_result, list) and len(factory_result) == 0:
                return "`[]`"
            elif isinstance(factory_result, dict) and len(factory_result) == 0:
                return "`{}`"
            else:
                return f"*{type(factory_result).__name__}*"
        except:
            return f"*default {field_obj.default_factory.__name__}*"
    else:
        return "`None`"


def generate_config_section(
    config_class, title: str, description: str = "", anchor: str = ""
) -> str:
    """Generate documentation for a single configuration dataclass."""
    if not is_dataclass(config_class):
        return ""

    # Create anchor for table of contents linking
    if anchor:
        doc = f"(section-{anchor})=\n## {title}\n\n"
    else:
        doc = f"## {title}\n\n"

    if description:
        doc += f"{description}\n\n"

    # Only add docstring if it's not the auto-generated dataclass signature
    if config_class.__doc__ and not config_class.__doc__.startswith(
        config_class.__name__ + "("
    ):
        doc += f"{config_class.__doc__.strip()}\n\n"

    doc += "| Parameter | Type | Default | Description |\n"
    doc += "|-----------|------|---------|-------------|\n"

    for field in fields(config_class):
        field_name = field.name
        field_type = get_type_description(field.type)
        default_value = format_default_value(field)

        # Get help text from metadata
        help_text = field.metadata.get(
            "help",
            "No description available. Please check the description of this dataclass.",
        )

        # Get choices if available
        choices = field.metadata.get("choices")
        if choices:
            help_text += f" **Choices:** {', '.join([f'`{c}`' for c in choices])}"

        doc += f"| `{field_name}` | {field_type} | {default_value} | {help_text} |\n"

    doc += "\n"
    return doc


def generate_cli_documentation():
    """Generate the complete CLI documentation."""
    doc = """# Configurations

This page provides a comprehensive reference for all configuration parameters available in AReaL's command-line interface. These parameters are defined using dataclasses and can be specified in YAML configuration files or overridden via command line arguments.

## Usage

Configuration files are specified using the `--config` parameter:

```bash
python -m areal.launcher --config path/to/config.yaml
```

You can override specific parameters from the command line:

```bash
python -m areal.launcher --config path/to/config.yaml actor.lr=1e-4 seed=42
```

For detailed examples, see the experiment configurations in the `examples/` directory.

## Table of Contents

### Core Experiment Configurations
- [Base Experiment Configuration](section-base-experiment)
- [SFT Configuration](section-sft)
- [GRPO Configuration](section-grpo)
- [Reward Model Configuration](section-reward-model)

### Training Configurations
- [Training Engine Configuration](section-train-engine)
- [PPO Actor Configuration](section-ppo-actor)
- [Optimizer Configuration](section-optimizer)
- [Micro-batch Specification](section-microbatch)
- [Normalization Configuration](section-normalization)

### Inference Configurations
- [Inference Engine Configuration](section-inference-engine)
- [SGLang Configuration](section-sglang)
- [Generation Hyperparameters](section-generation)

### Dataset
- [Dataset Configuration](section-dataset)

### System and Cluster Configurations
- [Cluster Specification](section-cluster)
- [Launcher Configuration](section-launcher)

### Logging and Monitoring
- [Statistics Logger Configuration](section-stats-logger)
- [Checkpoint Saver Configuration](section-saver)
- [Evaluator Configuration](section-evaluator)
- [Recovery Configuration](section-recovery)
- [Scheduler Configuration](section-scheduler)

---

"""

    # Core experiment configurations
    doc += generate_config_section(
        BaseExperimentConfig,
        "Base Experiment Configuration",
        "Base configuration shared by all experiment types (SFT, GRPO, etc.)",
        "base-experiment",
    )

    doc += generate_config_section(
        SFTConfig,
        "SFT Configuration",
        "Configuration specific to supervised fine-tuning experiments",
        "sft",
    )

    doc += generate_config_section(
        GRPOConfig,
        "GRPO Configuration",
        "Configuration for GRPO reinforcement learning experiments",
        "grpo",
    )

    doc += generate_config_section(
        RWConfig,
        "Reward Model Configuration",
        "Configuration for training reward models",
        "reward-model",
    )

    # Training configurations
    doc += generate_config_section(
        TrainEngineConfig,
        "Training Engine Configuration",
        "Core configuration for model training, including optimization and backend settings",
        "train-engine",
    )

    doc += generate_config_section(
        PPOActorConfig,
        "PPO Actor Configuration",
        "Configuration for PPO actor models in RL training",
        "ppo-actor",
    )

    doc += generate_config_section(
        OptimizerConfig,
        "Optimizer Configuration",
        "Settings for model optimization during training",
        "optimizer",
    )

    doc += generate_config_section(
        MicroBatchSpec,
        "Micro-batch Specification",
        "Configuration for splitting data into micro-batches during training",
        "microbatch",
    )

    # Inference configurations
    doc += generate_config_section(
        InferenceEngineConfig,
        "Inference Engine Configuration",
        "Configuration for model inference and rollout generation",
        "inference-engine",
    )

    doc += generate_config_section(
        SGLangConfig,
        "SGLang Configuration",
        "Configuration for SGLang inference runtime",
        "sglang",
    )

    doc += generate_config_section(
        GenerationHyperparameters,
        "Generation Hyperparameters",
        "Parameters controlling text generation behavior during RL training",
        "generation",
    )

    # Data and normalization
    doc += generate_config_section(
        DatasetConfig,
        "Dataset Configuration",
        "Configuration for training and validation datasets",
        "dataset",
    )

    doc += generate_config_section(
        NormConfig,
        "Normalization Configuration",
        "Settings for data normalization (rewards, advantages, etc.)",
        "normalization",
    )

    # System and cluster configurations
    doc += generate_config_section(
        ClusterSpecConfig,
        "Cluster Specification",
        "Configuration for distributed training cluster setup",
        "cluster",
    )

    doc += generate_config_section(
        LauncherConfig,
        "Launcher Configuration",
        "Settings for launching training and inference processes",
        "launcher",
    )

    # Logging and monitoring
    doc += generate_config_section(
        StatsLoggerConfig,
        "Statistics Logger Configuration",
        "Configuration for experiment logging and monitoring",
        "stats-logger",
    )

    doc += generate_config_section(
        SaverConfig,
        "Checkpoint Saver Configuration",
        "Settings for saving model checkpoints",
        "saver",
    )

    doc += generate_config_section(
        EvaluatorConfig,
        "Evaluator Configuration",
        "Configuration for model evaluation during training",
        "evaluator",
    )

    doc += generate_config_section(
        RecoverConfig,
        "Recovery Configuration",
        "Settings for experiment recovery and fault tolerance",
        "recovery",
    )

    doc += generate_config_section(
        SchedulerConfig,
        "Scheduler Configuration",
        "Configuration for the AReaL scheduler service. Used for the single-controller mode. Experimental.",
        "scheduler",
    )

    return doc


def main():
    """Generate the CLI documentation and save it to a markdown file."""
    output_path = Path(__file__).parent / "cli_reference.md"

    try:
        documentation = generate_cli_documentation()

        with open(output_path, "w") as f:
            f.write(documentation)

        print(f"✅ CLI documentation generated successfully at: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
