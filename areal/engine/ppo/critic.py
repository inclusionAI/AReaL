import functools
from typing import Any, Dict, List

import torch

from areal.api.cli_args import MicroBatchSpec, PPOCriticConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp_engine import FSDPEngine
from areal.utils import stats_tracker
from areal.utils.data import split_padded_tensor_dict_into_mb_list
from areal.utils.functional import ppo_critic_loss_fn


class PPOCritic:
    def __init__(self, config: PPOCriticConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

    @torch.no_grad()
    def compute_values(self, data: Dict[str, Any]) -> torch.Tensor | None:
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat([x.squeeze(-1) for x in xs], dim=-1),
        )

    def ppo_update(self, data: Dict[str, Any]) -> List[Dict[str, float]]:
        all_stats = []
        ########## Logging code starts ##########
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        stats_tracker.scalar(**scalars)

        global_stats = stats_tracker.export(
            reduce_group=self.engine.data_parallel_group
        )
        ########## Logging code ends ##########

        for key in ["rewards", "tot_rewards", "kl_rewards", "versions"]:
            data.pop(key, None)

        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )
        for mb in mb_inputs.mbs:
            train_stat = self.engine.train_batch(
                mb,
                loss_fn=functools.partial(
                    ppo_loss_fn,
                    eps_clip=self.config.eps_clip,
                ),
                loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
            )
            stats_tracker.scalar(**train_stat)
            all_stats.append(
                stats_tracker.export(reduce_group=self.engine.data_parallel_group)
            )
        all_stats[0].update(global_stats)
        return all_stats


class FSDPPPOCritic(FSDPEngine):

    def __init__(self, config: PPOCriticConfig):
        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor | None:
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> List[Dict[str, float]]:
        return self.critic.ppo_update(*args, **kwargs)


def ppo_loss_fn(
    value: torch.Tensor,
    input_data: Dict,
    eps_clip: float,
):
    """Loss function for critic step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    value = value.squeeze(-1).float()
    old_value = input_data["values"].float()
    target_value = input_data["returns"].float()
    loss_mask = input_data["loss_mask"].bool()

    loss, stat = ppo_critic_loss_fn(
        value=value,
        old_value=old_value,
        target_value=target_value,
        value_eps_clip=eps_clip,
        loss_mask=loss_mask,
    )

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=torch.ones(value.shape[0], dtype=torch.bool, device=value.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
    )

    stats_tracker.stat(
        new_value=value.detach(),
        old_value=old_value,
        target_value=target_value,
        critic_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        denominator="n_valid_tokens",
    )

    clip_mask = stat["clip_mask"]
    clipped_new_value = torch.where(clip_mask, value.detach(), 0.0)
    clipped_old_value = torch.where(clip_mask, old_value, 0.0)
    stats_tracker.stat(
        clipped_new_value=clipped_new_value,
        clipped_old_value=clipped_old_value,
        denominator="clipped_tokens",
    )
    return loss
