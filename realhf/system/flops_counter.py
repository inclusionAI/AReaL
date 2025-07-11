import dataclasses
from typing import *

from realhf.api.core.data_api import SequenceSample
from realhf.api.core.dfg import MFCDef, ModelInterfaceType
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base.monitor import (
    caculuate_llama_forward_flops,
    calculate_llama_gen_flops,
    calculate_llama_train_flops,
)


@dataclasses.dataclass
class FlopsCounter:
    train_configs: List[ReaLModelConfig] = dataclasses.field(default_factory=list)
    train_bs: List[int] = dataclasses.field(default_factory=list)
    train_seqlens: List[List[int]] = dataclasses.field(default_factory=list)

    inf_configs: List[ReaLModelConfig] = dataclasses.field(default_factory=list)
    inf_bs: List[int] = dataclasses.field(default_factory=list)
    inf_seqlens: List[List[int]] = dataclasses.field(default_factory=list)

    gen_configs: List[ReaLModelConfig] = dataclasses.field(default_factory=list)
    gen_bs: List[int] = dataclasses.field(default_factory=list)
    prompt_lens: List[List[int]] = dataclasses.field(default_factory=list)
    gen_len: List[int] = dataclasses.field(default_factory=list)

    def clear(self):
        self.train_bs.clear()
        self.train_seqlens.clear()

        self.inf_bs.clear()
        self.inf_seqlens.clear()

        self.gen_bs.clear()
        self.prompt_lens.clear()
        self.gen_len.clear()

        self.train_configs.clear()
        self.inf_configs.clear()
        self.gen_configs.clear()

    def add_rpc(
        self, rpc: MFCDef, sample: SequenceSample, model_config: ReaLModelConfig
    ):
        # Record the data amount for each interface to compute FLOPs.
        # Since the user may arbitrarily specify input/output keys,
        # we can only try to find the most probable key name for computing FLOPs.
        # If such keys do not exist, we will use the key with the longest
        # sequence length in this model function call.
        acc_seqlens = {
            k: sum(sum(x) for x in slens) for k, slens in sample.seqlens.items()
        }
        seqlen_key = max(sample.seqlens, key=acc_seqlens.get)
        flops_seqlens = [sum(x) for x in sample.seqlens[seqlen_key]]
        if rpc.interface_type == ModelInterfaceType.GENERATE:
            self.gen_configs.append(model_config)
            self.gen_bs.append(sample.bs)
            self.gen_len.append(
                rpc.interface_impl.args["generation_config"]["min_new_tokens"]
            )
            self.prompt_lens.append(flops_seqlens)
        elif rpc.interface_type == ModelInterfaceType.TRAIN_STEP:
            self.train_configs.append(model_config)
            self.train_bs.append(sample.bs)
            self.train_seqlens.append(flops_seqlens)
        elif rpc.interface_type == ModelInterfaceType.INFERENCE:
            self.inf_configs.append(model_config)
            self.inf_bs.append(sample.bs)
            self.inf_seqlens.append(flops_seqlens)

    def get_flops(self) -> int:
        flops = 0
        for train_bs, train_seqlens, real_config in zip(
            self.train_bs,
            self.train_seqlens,
            self.train_configs,
        ):
            flops += calculate_llama_train_flops(
                checkpoint_activations_factor=4,
                batch_size=train_bs,
                seqlens=train_seqlens,
                num_layers=real_config.n_layers,
                hidden_size=real_config.hidden_dim,
                intermediate_size=real_config.intermediate_dim,
                vocab_size=real_config.vocab_size,
            )
        for inf_bs, inf_seqlens, real_config in zip(
            self.inf_bs,
            self.inf_seqlens,
            self.inf_configs,
        ):
            flops += caculuate_llama_forward_flops(
                batch_size=inf_bs,
                seqlens=inf_seqlens,
                num_layers=real_config.n_layers,
                hidden_size=real_config.hidden_dim,
                intermediate_size=real_config.intermediate_dim,
                vocab_size=real_config.vocab_size,
            )
        for gen_bs, prompt_lens, gen_len, real_config in zip(
            self.gen_bs,
            self.prompt_lens,
            self.gen_len,
            self.gen_configs,
        ):
            flops += calculate_llama_gen_flops(
                batch_size=gen_bs,
                prompt_lens=prompt_lens,
                gen_len=gen_len,
                num_layers=real_config.n_layers,
                hidden_size=real_config.hidden_dim,
                intermediate_size=real_config.intermediate_dim,
                vocab_size=real_config.vocab_size,
            )
        return flops
