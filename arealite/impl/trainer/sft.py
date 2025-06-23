from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import time

import torch
import torch.distributed as dist
from datasets import Dataset

from arealite.api.cli_args import MicroBatchSpec, TrainerConfig, TrainingArgs
from arealite.api.trainer_api import Trainer
from arealite.api.engine_api import EngineFactory
from arealite.impl.rollout_controller import RolloutController
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, stats_tracker
from arealite.utils import (
    compute_varlen_position_indices,
    split_dict_tensor_with_cu_seqlens,
)
from realhf.impl.model.utils.functional import (
    build_shift_one_indices,
    gather_packed_shifted_log_probs,
)


logger = logging.getLogger("SFT Trainer")

def compute_packed_sft_loss(
    logits: torch.Tensor,
    input_: Dict[str, torch.Tensor],
) -> torch.Tensor:
    packed_input_ids: torch.Tensor = input_["packed_input_ids"].squeeze()
    cu_seqlens: torch.Tensor = input_["cu_seqlens"].squeeze()
    input_lens: torch.Tensor = cu_seqlens[1:] - cu_seqlens[:-1]
    cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
    prompt_mask = input_["prompt_mask"].squeeze()

    shift_one_indices = build_shift_one_indices(logits, cu_seqlens)
    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()
    prompt_mask = prompt_mask[shift_one_indices]
    logprobs = torch.where(prompt_mask, 0, logprobs)

    loss = -logprobs.sum() / prompt_mask.logical_not().count_nonzero()

    with torch.no_grad():
        seqlogp = torch.zeros(
            cu_seqlens.shape[0] - 1, device=logits.device, dtype=torch.float64
        )
        for i in range(cu_seqlens.shape[0] - 1):
            m = prompt_mask[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            logp = logprobs[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            assert cu_seqlens[i + 1] - i - 1 <= logprobs.shape[0], (
                cu_seqlens,
                logprobs.shape,
            )
            seqlogp[i] = torch.where(m, 0.0, logp.detach()).sum() / (
                m.numel() - m.count_nonzero()
            )

    ## Loggin stats
    stats_tracker.denominator(
        n_seqs=torch.ones(
            cu_seqlens.shape[0] - 1, dtype=torch.bool, device=logprobs.device
        ),
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=prompt_mask.logical_not(),
        prompt_tokens=prompt_mask,
    )
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    stats_tracker.stat(loss=-logprobs.detach(), denominator="n_valid_tokens")
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    # dist.all_reduce(
    #     vocab_min_logits, group=constants.tensor_parallel_group(), op=dist.ReduceOp.MIN
    # )
    # dist.all_reduce(
    #     vocab_max_logits, group=constants.tensor_parallel_group(), op=dist.ReduceOp.MAX
    # )
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    return loss

class SFTTrainer(Trainer):

    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional[RolloutController] = None,
        extra_args: Optional[Dict] = None,
    ):
        super().__init__(
            args,
            trainer_config,
            train_dataset,
            valid_dataset,
            rollout_controller,
            extra_args,
        )

        self.config = config = trainer_config.sft
        assert config is not None
        
        engine_factory = EngineFactory(args)
        self.model = engine_factory.make_engine(config.model)
        self.tokenizer = load_hf_tokenizer(config.model.path)
        
        self.mb_spec = config.mb_spec

    def _tokenize(self, strs: List[str]):
        # tokenize strings into unpadded tokens with lengths.
        return self.tokenizer(
            strs,
            padding=False,
            truncation=True,
            return_length=True,
            max_length=self.mb_spec.max_tokens_per_mb,
            return_attention_mask=False,
        )

    def _get_packed_input(self, data: Dict):
        prompts = data["prompt"]
        answers = data["answer"]
        inputs = [
            prompt + answer + self.tokenizer.eos_token for prompt, answer in zip(prompts, answers)
        ]
        tokenized_prompts = self._tokenize(prompts)
        tokenized_inputs = self._tokenize(inputs)

        # form a data batch
        prompt_lens = tokenized_prompts["length"]
        input_lens = tokenized_inputs["length"]

        print(input_lens)
        input_lens = torch.tensor(input_lens, dtype=torch.int)
        input_ids = [torch.tensor(seq, dtype=torch.long) for seq in tokenized_prompts["input_ids"]]

        prompt_mask = []
        for input_len, prompt_len in zip(input_lens, prompt_lens):
            assert input_len >= prompt_len, (input_len, prompt_len)
            pm = [1] * prompt_len + [0] * (input_len - prompt_len)
            prompt_mask.append(torch.tensor(pm, dtype=torch.bool))

        cu_seqlens = torch.nn.functional.pad(
            input_lens.cumsum(0, dtype=torch.int), (1, 0)
        )
        max_seqlen = int(torch.max(input_lens).item())
        packed_input_ids = torch.cat(input_ids, dim=0)
        prompt_mask = torch.cat(prompt_mask, dim=0)
        total_seqlen = int(cu_seqlens[-1].item())
        position_ids = compute_varlen_position_indices(total_seqlen, cu_seqlens)

        return dict(
            input_ids=packed_input_ids.unsqueeze(0).cuda(),
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0).cuda(),
            prompt_mask=prompt_mask.unsqueeze(0).cuda(),
            cu_seqlens=cu_seqlens.cuda(),
            max_seqlen=max_seqlen,
            use_cache=False,
        )

    def train(self, resume_from_checkpoint = None):
        self.model.init_distributed(None)
        self.create_train_dataloader()
        self.model.train()

        total_epochs = self.args.exp_ctrl.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)

        print(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
        model_version = 0
        start_time = time.monotonic()
        # dataloader: self.train_data_loader
        for epoch in range(total_epochs):
            for step in range(steps_per_epoch):
                t = time.monotonic()
                self.data_generator = iter(self.train_dataloader)
                data = next(self.data_generator)
                
                if dist.get_rank() == 0:
                    print(f"Epoch {epoch} Step {step} start ...")
                packed_input_data = self._get_packed_input(data)
                dist.barrier()
                if dist.get_rank() == 0:
                    print(f"Epoch {epoch} Step {step} data pre-process done.")
                model_version += 1
                stats = self.model.train_batch(
                    input_=packed_input_data,
                    loss_fn=compute_packed_sft_loss,
                    loss_weight_fn=lambda x: x["prompt_mask"]
                        .logical_not()
                        .count_nonzero(),
                    mb_spec=self.mb_spec,
                    version_steps=model_version,
                )
                stats_tracker.scalar(**stats)
                if dist.get_rank() == 0:
                    print(f"Epoch {epoch} Step {step} done. "
                        f"This step time elapsed {time.monotonic() - t:.2f}. "
                        f"Total time elapsed {time.monotonic() - start_time:.2f}")
                
                # if dist.get_rank() == 0:
                #     print(f"epoch {epoch} step {step}")
                #     print(f"input_ids={input_data["input_ids"].shape}")
                #     print(f"position_ids={input_data["position_ids"].shape}")
                #     print(f"cu_seqlens={input_data["cu_seqlens"]}")
                #     print(f"max_seqlen={input_data["max_seqlen"]}")                

    def save_checkpoint(self):
        pass
