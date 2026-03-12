"""Actor for self-distillation training.

Performs student/teacher forward passes and computes the self-distillation KL loss
in a single global-batch update.
"""

from __future__ import annotations

from typing import Any

import torch
from areal.trainer.ppo.stats import infer_token_denominator

from areal.api.cli_args import SelfDistillActorConfig
from areal.api.engine_api import TrainEngine
from areal.infra import TrainController
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import trace_perf
from areal.utils.data import concat_padded_tensors
logger = logging.getLogger("SelfDistillActor")


class SelfDistillActor:
    """Actor for self-distillation (SDPO).

    Parameters
    ----------
    config : SelfDistillActorConfig
        Actor configuration including self-distillation settings.
    engine : TrainEngine
        Training engine (FSDP, Megatron, or Archon).
    """

    def __init__(self, config: SelfDistillActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine
        self.distill_config = config.self_distillation

        logger.info("=" * 70)
        logger.info("SelfDistillActor Configuration")
        logger.info("=" * 70)
        logger.info("  distillation_topk: %s", self.distill_config.distillation_topk)
        logger.info("  is_clip: %s", self.distill_config.is_clip)
        logger.info(
            "  teacher_update_rate: %s", self.distill_config.teacher_update_rate
        )
        logger.info("=" * 70)

    @trace_perf("distill_actor.reorg_batch", category="misc")
    @torch.no_grad()
    def reorg_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        input_ids = data['input_ids'].cpu().numpy()
        loss_mask = data['loss_mask'].cpu().numpy()
        prompt_mask = data['prompt_mask'].cpu().numpy()
        bs = data['input_ids'].shape[0]

        all_new_data = []
        for i in range(bs):
            new_data = dict()
            prompt = input_ids[i][prompt_mask[i]].tolist()
            response = input_ids[i][loss_mask[i]].tolist()
            new_data['input_ids'] = prompt + response
            new_data['attention_mask'] = [1] * len(new_data['input_ids'])
            _loss_mask = [0] * len(prompt) + [1] * len(response)
            new_data['loss_mask'] = _loss_mask

            new_data['fb_input_ids'] = input_ids[i].tolist()
            new_data['fb_attention_mask'] = [1] * len(input_ids[i])
            new_data['fb_loss_mask'] = loss_mask[i].tolist()

            all_new_data.append({k: torch.tensor(v).unsqueeze(0) for k, v in new_data.items()})
        res = concat_padded_tensors(all_new_data)
        res['prompt_mask'] = torch.tensor(prompt_mask)
        res['logprobs'] = torch.roll(data['logprobs'], shifts=-1, dims=-1)
        res['loss_mask'] = torch.roll(data['loss_mask'], shifts=-1, dims=-1)
        res['fb_loss_mask'] = torch.roll(data['fb_loss_mask'], shifts=-1, dims=-1)
        return res

    @trace_perf("distill_actor.compute_teacher_logp", category="compute")
    @torch.no_grad()
    def compute_teacher_logp(self, data: dict[str, Any]) -> torch.Tensor:
        self.engine.eval()
        inp_data = dict(
            input_ids=data['fb_input_ids'],
            attention_mask=data['fb_attention_mask'],
        )
        logp = self.engine.forward(
            input_=inp_data,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )
        bs = logp.shape[0]
        max_len = data['prompt_mask'].shape[1]
        student_logp = []
        for i in range(bs):
            student_ologp = logp[i][data['fb_loss_mask'][i] == 1]
            # TODO: check logp shifts
            plen = data['prompt_mask'][i].sum().item()
            student_logp.append(torch.nn.functional.pad(
                student_ologp,
                (plen - 1, max_len - plen),
                value=0.0,
            ))
        return torch.stack(student_logp, dim=0)

    @trace_perf("distill_actor.self_distill_update", category="compute")
    @stats_tracker.scope_func_wrapper("distill_actor")
    def self_distill_update(self, data: dict[str, Any]) -> None:
        self.engine.train()
        self.engine.train_batch(
            input_=data,
            loss_fn=distill_loss_fn,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )


class DistillActorController(TrainController):
    """Controller for distributed self-distillation actor."""

    def reorg_batch(self, *args, **kwargs):
        return self._custom_function_call("reorg_batch", *args, **kwargs)

    def compute_teacher_logp(self, *args, **kwargs):
        return self._custom_function_call("compute_teacher_logp", *args, **kwargs)

    def self_distill_update(self, *args, **kwargs) -> None:
        self._custom_function_call("self_distill_update", *args, **kwargs)


def distill_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    is_clip: bool,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    student_log_probs = logprobs
    teacher_log_probs = input_data["teacher_log_probs"]
    loss_mask = input_data["loss_mask"]
    old_log_probs = input_data.get("old_log_probs")

    student_log_probs = torch.where(
        loss_mask.bool(), student_log_probs, 0.0
    )
    teacher_log_probs = torch.where(
        loss_mask.bool(), teacher_log_probs, 0.0
    )

    log_ratio = student_log_probs - teacher_log_probs
    per_token_loss = log_ratio.detach() * student_log_probs

    # Importance sampling clipping
    if is_clip:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")
        old_log_probs = torch.where(
            loss_mask.bool(), old_log_probs, 0.0
        )
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    loss = torch.where(loss_mask.bool(), per_token_loss, 0.0)

    # Log metrics
    stats_tracker.scalar(is_clip=is_clip)
    stats_tracker.denominator(
        n_valid_tokens=loss_mask,
        n_tokens=infer_token_denominator(input_data, loss_mask),
    )
    stats_tracker.stat(
        loss=loss,
        distill_approx_kl=log_ratio.detach(),
        student_logp=student_log_probs.detach(),
        teacher_logp=teacher_log_probs.detach(),
        denominator="n_valid_tokens",
    )
    if is_clip:
        stats_tracker.stat(
            clip_approx_kl=negative_approx_kl,
            imp_weight=ratio,
            denominator="n_valid_tokens",
        )
    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    return loss

