"""E2E test entrypoint for speculative decoding with EAGLE.

This module provides a MinimalSpecDecodePPOTrainer that wraps the standard
PPOTrainer to collect and validate speculative decoding statistics (accept
rate, draft tokens) and MTP training loss during end-to-end test runs.

Usage:
    python -m tests.speculative_decoding.entrypoint --config <config_path>
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpecDecodeStats:
    """Accumulated speculative decoding statistics across training steps."""

    total_accept_tokens: int = 0
    total_draft_tokens: int = 0
    step_accept_rates: List[float] = field(default_factory=list)
    mtp_losses: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    @property
    def overall_accept_rate(self) -> float:
        """Compute overall accept rate across all steps."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accept_tokens / self.total_draft_tokens

    @property
    def mean_mtp_loss(self) -> float:
        """Compute mean MTP loss across all steps."""
        if not self.mtp_losses:
            return float("nan")
        return sum(self.mtp_losses) / len(self.mtp_losses)

    @property
    def mean_reward(self) -> float:
        """Compute mean reward across all steps."""
        if not self.rewards:
            return float("nan")
        return sum(self.rewards) / len(self.rewards)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of all collected statistics."""
        return {
            "total_accept_tokens": self.total_accept_tokens,
            "total_draft_tokens": self.total_draft_tokens,
            "overall_accept_rate": self.overall_accept_rate,
            "num_steps": len(self.step_accept_rates),
            "step_accept_rates": self.step_accept_rates,
            "mean_mtp_loss": self.mean_mtp_loss,
            "mean_reward": self.mean_reward,
            "mtp_losses": self.mtp_losses,
            "rewards": self.rewards,
        }


class MinimalSpecDecodePPOTrainer:
    """A minimal wrapper around PPOTrainer for speculative decoding E2E tests.

    This trainer intercepts training statistics to collect and validate
    speculative decoding metrics including:
    - Speculative accept rate (spec_accept_token_num / spec_draft_token_num)
    - MTP auxiliary loss (when enable_mtp_training is True)
    - Reward statistics

    It is designed for use in integration tests, not production training.
    """

    def __init__(self, config_path: str):
        """Initialize the trainer with a config file path.

        Args:
            config_path: Path to the experiment YAML configuration file.
        """
        self.config_path = config_path
        self.stats = SpecDecodeStats()
        self._trainer = None
        self._config = None

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file.

        Returns:
            Parsed configuration dictionary.
        """
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(self.config_path)
            self._config = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            import yaml

            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
        return self._config

    def _collect_step_stats(self, train_stat: Dict[str, Any]) -> None:
        """Extract speculative decoding stats from a training step result.

        Args:
            train_stat: Dictionary of statistics from one training step.
        """
        # Collect speculative decoding accept/draft token counts
        accept_tokens = train_stat.get("spec_accept_token_num", 0)
        draft_tokens = train_stat.get("spec_draft_token_num", 0)

        if draft_tokens > 0:
            self.stats.total_accept_tokens += accept_tokens
            self.stats.total_draft_tokens += draft_tokens
            step_rate = accept_tokens / draft_tokens
            self.stats.step_accept_rates.append(step_rate)
            logger.info(
                f"[SpecDecode] Step accept rate: {step_rate:.4f} "
                f"({accept_tokens}/{draft_tokens})"
            )

        # Collect MTP loss if present
        mtp_loss = train_stat.get("mtp_loss", None)
        if mtp_loss is not None:
            self.stats.mtp_losses.append(float(mtp_loss))
            logger.info(f"[MTPTrain] MTP loss: {mtp_loss:.6f}")

        # Collect rewards
        reward = train_stat.get("reward/mean", train_stat.get("reward", None))
        if reward is not None:
            self.stats.rewards.append(float(reward))

    def run(self, max_steps: Optional[int] = None) -> SpecDecodeStats:
        """Run the training loop and collect speculative decoding statistics.

        Args:
            max_steps: Maximum number of training steps. None runs the full
                config (total_train_epochs).

        Returns:
            SpecDecodeStats with all collected metrics.
        """
        config = self._load_config()
        experiment_name = config.get("experiment_name", "test-spec-decode")
        logger.info(
            f"Starting MinimalSpecDecodePPOTrainer for '{experiment_name}' "
            f"with config: {self.config_path}"
        )

        # Log speculative decoding configuration
        sglang_cfg = config.get("sglang", {})
        actor_cfg = config.get("actor", {})
        logger.info(
            f"Speculative config: algorithm={sglang_cfg.get('speculative_algorithm')}, "
            f"num_steps={sglang_cfg.get('speculative_num_steps')}, "
            f"num_draft_tokens={sglang_cfg.get('speculative_num_draft_tokens')}"
        )
        logger.info(
            f"MTP training: enabled={actor_cfg.get('enable_mtp_training', False)}, "
            f"num_layers={actor_cfg.get('mtp_num_layers', 0)}, "
            f"loss_scaling={actor_cfg.get('mtp_loss_scaling_factor', 0.0)}"
        )

        try:
            from areal.trainer.rl_trainer import PPOTrainer

            self._trainer = PPOTrainer(config)
            step = 0
            for train_stat in self._trainer.train():
                self._collect_step_stats(train_stat)
                step += 1
                if max_steps is not None and step >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}, stopping.")
                    break
        except ImportError as e:
            logger.warning(
                f"Could not import PPOTrainer: {e}. "
                f"Running in dry-run mode (config validation only)."
            )
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        # Print summary
        summary = self.stats.summary()
        logger.info(f"=== Speculative Decoding E2E Summary ===")
        logger.info(f"  Total steps: {summary['num_steps']}")
        logger.info(f"  Overall accept rate: {summary['overall_accept_rate']:.4f}")
        logger.info(f"  Mean MTP loss: {summary['mean_mtp_loss']:.4f}")
        logger.info(f"  Mean reward: {summary['mean_reward']:.4f}")

        return self.stats


def main():
    """CLI entrypoint for running speculative decoding E2E tests."""
    parser = argparse.ArgumentParser(
        description="Run speculative decoding E2E test with AReaL PPOTrainer"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (default: run full config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    trainer = MinimalSpecDecodePPOTrainer(config_path=args.config)
    stats = trainer.run(max_steps=args.max_steps)

    summary = stats.summary()
    print("\n=== Final Statistics ===")
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"  {key}: [{len(value)} entries]")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Exit with error if no steps completed and we expected some
    if summary["num_steps"] == 0 and args.max_steps != 0:
        logger.warning("No training steps were completed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
