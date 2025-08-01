# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import itertools
import json
import os
import pathlib
import shutil
import uuid
from typing import *

import pytest
import torch
import torch.distributed as dist

from realhf.api.core.config import ModelName, ModelShardID
from realhf.api.core.data_api import MicroBatchSpec, SequenceSample
from realhf.api.core.model_api import HF_MODEL_FAMILY_REGISTRY, ReaLModelConfig
from realhf.base import constants, logging, testing, topology
from realhf.base.datapack import flat2d
from realhf.base.testing import (
    LocalMultiProcessTest,
    clear_name_resolve,
    init_global_constants,
    make_random_packed_batches,
)

if TYPE_CHECKING:
    from realhf.impl.model.backend.mock_train import MockTrainEngine
    from realhf.impl.model.nn.real_llm_api import ReaLModel


def compute_critic_loss(
    logits: torch.Tensor,
    input_: SequenceSample,
) -> torch.Tensor:
    from realhf.impl.model.utils.functional import build_shift_one_indices

    input_lens = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]))
    cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
    shift_one_indices = build_shift_one_indices(logits, cu_seqlens)
    prompt_mask = input_.data["prompt_mask"][shift_one_indices]
    scores = logits.squeeze().float()[shift_one_indices]
    scores = torch.where(prompt_mask, 0, scores)

    loss = ((scores - torch.zeros_like(scores)) ** 2).sum() / (
        prompt_mask.numel() - prompt_mask.count_nonzero()
    )
    return loss, {"loss": loss.clone().detach()}


def create_model(
    tmp_dir: pathlib.Path,
    model_family_name: str,
    model_name,
    is_critic: int,
    instantiate=True,
) -> "ReaLModel":
    # NOTE: import here to avoid initializing CUDA context in the main process
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    with constants.model_scope(model_name):
        mconfig: ReaLModelConfig = getattr(
            ReaLModel, f"make_{model_family_name}_config"
        )()
        mconfig.is_critic = is_critic

        # initialize model
        model = ReaLModel(
            mconfig, dtype=torch.float32, device=constants.current_device()
        )
        model.eval()
        if instantiate:
            model.instantiate()
    if instantiate:
        _check_tied_embedding_weights(model_name, model)
    with constants.model_scope(model_name):
        if instantiate:
            init_save_path = tmp_dir / "init"
            # sync initialized parameters
            getattr(model, f"to_{model_family_name}")(None, init_save_path)
            dist.barrier(group=constants.parallelism_group())
            model = getattr(model, f"from_{model_family_name}")(
                init_save_path, init_critic_from_actor=False
            )
    if instantiate:
        _check_tied_embedding_weights(model_name, model)
    return model


def get_topo(model_name):
    with constants.model_scope(model_name):
        return constants.grid().topology()


def build_engine(module, model_name, trainable) -> "MockTrainEngine":
    from realhf.api.core import model_api
    from realhf.impl.model.backend.inference import PipelineInferenceBackend
    from realhf.impl.model.backend.mock_train import MockTrainBackend
    from realhf.impl.model.nn.real_llm_api import add_helper_functions

    with constants.model_scope(model_name):
        if constants.pipe_parallel_world_size() == 1:
            add_helper_functions(module)
        if trainable:
            backend = MockTrainBackend(optimizer_config=dict(lr=1e-4))
        else:
            backend = PipelineInferenceBackend()
        _model = backend.initialize(
            model_api.Model(
                None,
                module,
                None,
                module.device,
                module.dtype,
            ),
            model_api.FinetuneSpec(
                total_train_epochs=1,
                dataset_size=20,
                train_batch_size=1,
            ),
        )
    return _model.module


def setup_constants_and_param_realloc(
    from_model_name,
    to_model_name,
    from_pp_dp_tp,
    to_pp_dp_tp,
):
    from realhf.impl.model.comm.param_realloc import setup_param_realloc

    from_num_pp, from_num_dp, from_num_tp = from_pp_dp_tp
    to_num_pp, to_num_dp, to_num_tp = to_pp_dp_tp

    from_world_size = from_num_dp * from_num_tp * from_num_pp
    to_world_size = to_num_dp * to_num_tp * to_num_pp

    from_topo = topology.PipeDataTensorParallelTopology(
        num_dp=from_num_dp,
        num_tp=from_num_tp,
        num_pp=from_num_pp,
        sequence_parallel=False,
        gradient_checkpointing=False,
        max_prompt_len=None,
        gradient_accumulation_fusion=False,
    )
    to_topo = topology.PipeDataTensorParallelTopology(
        num_dp=to_num_dp,
        num_tp=to_num_tp,
        num_pp=to_num_pp,
        sequence_parallel=False,
        gradient_checkpointing=False,
        max_prompt_len=None,
        gradient_accumulation_fusion=False,
    )

    model_topos = {from_model_name: from_topo, to_model_name: to_topo}

    msid2mwid = {}
    for i in range(dist.get_world_size()):
        # We assume the `from_model` occupies the first serveral GPUs,
        # while the `to_model` occupies GPUs from the last one.
        # For example, when the world size of `from_model` is 6 and
        # the world size of `to_model` is 4, the GPU layout is:
        # GPU 0-3: from_model (shard 0-3)
        # GPU 4-5: from_model (shard 4-5) + to_model (shard 0-1)
        # GPU 6-7: to_model (shard 2-3)
        _model_names = []
        if i < from_world_size:
            _model_names.append(from_model_name)
        if i >= dist.get_world_size() - to_world_size:
            _model_names.append(to_model_name)
        for _model_name in _model_names:
            if _model_name == from_model_name:
                coord = model_topos[_model_name].get_coord(i)
            else:
                coord = model_topos[_model_name].get_coord(
                    i + to_world_size - dist.get_world_size()
                )
            k = ModelShardID(
                _model_name,
                dp_rank=coord.data,
                tp_rank=coord.tensor,
                pp_rank=coord.pipe,
                topo=model_topos[_model_name],
            )
            msid2mwid[k] = i

    init_global_constants(
        num_dp=from_num_dp,
        num_tp=from_num_tp,
        num_pp=from_num_pp,
        topo=from_topo,
        model_name=from_model_name,
        sequence_parallel=False,
        msid2mwid=msid2mwid,
    )

    init_global_constants(
        num_dp=to_num_dp,
        num_tp=to_num_tp,
        num_pp=to_num_pp,
        model_name=to_model_name,
        sequence_parallel=False,
        msid2mwid=msid2mwid,
    )

    pg_info = setup_param_realloc(
        model_topos=model_topos,
        msid2mwid=msid2mwid,
        param_realloc_pairs=[
            (from_model_name, to_model_name),
            (to_model_name, from_model_name),
        ],
    )
    return pg_info


def _check_tied_embedding_weights(model_name, model: "ReaLModel"):
    if not model.config.tied_embedding or model.config.is_critic:
        return
    with constants.model_scope(model_name):
        if not (constants.is_first_pipe_stage() or constants.is_last_pipe_stage()):
            return

        if constants.is_first_pipe_stage():
            w1 = w = model.layers[0].wte.weight
        if constants.is_last_pipe_stage():
            w2 = w = model.layers[-1].weight

        if constants.pipe_parallel_world_size() == 1:
            if model.config.tied_embedding and not model.config.is_critic:
                assert w1.data_ptr() == w2.data_ptr()
            else:
                assert w1.data_ptr() != w2.data_ptr()
        else:
            w_ = w.clone().detach()
            dist.all_reduce(
                w_,
                op=dist.ReduceOp.SUM,
                group=constants.grid().embedding_proc_group,
            )
            w_ /= dist.get_world_size(constants.grid().embedding_proc_group)
            if model.config.tied_embedding and not model.config.is_critic:
                assert torch.allclose(w_, w, atol=5e-4), (w_ - w).abs().max()
            else:
                assert not torch.allclose(w_, w), (w_ - w).abs().max()


@dataclasses.dataclass
class ParamRedistributer:
    from_model_name: ModelName
    to_model_name: ModelName
    from_model: Any
    to_model: Any
    pg_info: Any

    def _redist(self, m1, m2, n1, n2):
        from realhf.impl.model.comm.param_realloc import is_trainable

        if m1 is None and m2 is None:
            return
        with constants.model_scope(n1):
            t1 = constants.grid().topology()
        with constants.model_scope(n2):
            t2 = constants.grid().topology()
        m = m1 if m1 is not None else m2
        a, b, c = m.build_reparallelized_layers_async(
            from_model_name=n1,
            to_model_name=n2,
            from_topo=t1,
            to_topo=t2,
            to_model_config=m.config,
            pg_info=self.pg_info,
        )
        if m2 is not None and is_trainable(n1):
            m2.patch_reparallelization((a, b), eta=1.0)

        # if m1 is not None:
        #     assert m1.layers is None
        #     assert m1.contiguous_param is None
        # if m2 is not None:
        #     assert m2.layers is not None
        #     assert m2.contiguous_param is not None

    def forward(self):
        self._redist(
            self.from_model,
            self.to_model,
            self.from_model_name,
            self.to_model_name,
        )

    def backward(self):
        self._redist(
            self.to_model,
            self.from_model,
            self.to_model_name,
            self.from_model_name,
        )


def _load_all_pytorch_bin(path: pathlib.Path):
    if os.path.exists(path / "pytorch_model.bin.index.json"):
        with open(path / "pytorch_model.bin.index.json", "r") as f:
            hf_sd_mapping = json.load(f)["weight_map"]
        sd = {}
        for fn in hf_sd_mapping.values():
            sd.update(torch.load(path / fn, map_location="cpu"))
    else:
        sd = torch.load(path / "pytorch_model.bin", map_location="cpu")
    return sd


def _test_para_realloc(
    tmp_path: pathlib.Path,
    model_family_name: str,
    is_critic: bool,
    from_pp_dp_tp: Tuple,
    to_pp_dp_tp: Tuple,
    n_iterations: int,
    skip_saveload: bool,
):
    # os.environ["REAL_SAVE_MAX_SHARD_SIZE_BYTE"] = str(int(1e6))
    from realhf.impl.model.backend.mock_train import MockTrainEngine
    from realhf.impl.model.comm.param_realloc import set_trainable
    from realhf.impl.model.interface.sft_interface import compute_packed_sft_loss

    from_model_name = ModelName("param_realloc_test", 0)
    to_model_name = ModelName("param_realloc_test", 1)

    set_trainable(from_model_name, True)
    set_trainable(to_model_name, False)

    pg_info = setup_constants_and_param_realloc(
        from_model_name,
        to_model_name,
        from_pp_dp_tp,
        to_pp_dp_tp,
    )

    # Create model 1
    if dist.get_rank() < from_pp_dp_tp[0] * from_pp_dp_tp[1] * from_pp_dp_tp[2]:
        from_model = create_model(
            tmp_dir=tmp_path,
            model_family_name=model_family_name,
            model_name=from_model_name,
            is_critic=is_critic,
            instantiate=True,
        )
    else:
        from_model = None
    # Creat model 2
    if (
        dist.get_rank()
        >= dist.get_world_size() - to_pp_dp_tp[0] * to_pp_dp_tp[1] * to_pp_dp_tp[2]
    ):
        to_model = create_model(
            tmp_dir=tmp_path,
            model_family_name=model_family_name,
            model_name=to_model_name,
            is_critic=is_critic,
            instantiate=False,
        )
    else:
        to_model = None

    # Create redistributer.
    redist = ParamRedistributer(
        from_model_name,
        to_model_name,
        from_model,
        to_model,
        pg_info,
    )

    if from_model is not None:
        train_engine = build_engine(from_model, from_model_name, trainable=True)
        _check_tied_embedding_weights(from_model_name, from_model)
    if to_model is not None:
        inf_engine = build_engine(to_model, to_model_name, trainable=False)

    for i in range(n_iterations):
        # Create the same random data across all ranks.
        if from_model is not None:
            vocab_size = from_model.config.vocab_size
        elif to_model is not None:
            vocab_size = to_model.config.vocab_size
        else:
            # Give a random vocab size for sampling across the whole world.
            vocab_size = 100

        _v = torch.tensor(
            [vocab_size], dtype=torch.int32, device=constants.current_device()
        )
        dist.all_reduce(_v, op=dist.ReduceOp.MAX)
        vocab_size = _v.item()

        # Synchronize the data across all ranks.
        x = make_random_packed_batches(
            n_batches=1,
            batch_size=32,
            seq_len=32,
            vocab_size=vocab_size,
            dp_rank=0,
            dp_size=1,
        )[0].to_device(device=constants.current_device())

        # Synchronize the initial parameters at the start of this iteration.
        if not skip_saveload:
            if from_model is not None:
                with constants.model_scope(from_model_name):
                    getattr(from_model, f"to_{model_family_name}")(
                        None, tmp_path / f"save_from_{i}"
                    )
            dist.barrier()
            sd1 = _load_all_pytorch_bin(tmp_path / f"save_from_{i}")

        # Run redistribution.
        redist.forward()
        dist.barrier()

        # Synchronize the redistributed parameters. They should be identical to the initial parameters.
        # Also, they should be different from the parameters of the previous iteration
        # because we have updated the parameters.
        if not skip_saveload:
            if to_model is not None:
                with constants.model_scope(to_model_name):
                    getattr(to_model, f"to_{model_family_name}")(
                        None, tmp_path / f"save_to_{i}"
                    )
            dist.barrier()
            sd2 = _load_all_pytorch_bin(tmp_path / f"save_to_{i}")
            for k, v in sd1.items():
                assert torch.allclose(v, sd2[k], atol=2e-4), (
                    k,
                    (v - sd2[k]).abs().max(),
                    v.flatten()[:10],
                    sd2[k].flatten()[:10],
                )

        # Run a forward with the redistributed model.
        if to_model is not None:
            _check_tied_embedding_weights(to_model_name, to_model)
            with constants.model_scope(to_model_name):
                inf_engine.eval()
                logits1 = inf_engine.forward(input_=x, mb_spec=MicroBatchSpec())

        # Run redistribution backwards.
        redist.backward()
        dist.barrier()

        # Re-run redistribution to examine whether inference results are identical.
        redist.forward()
        dist.barrier()
        if to_model is not None:
            _check_tied_embedding_weights(to_model_name, to_model)
            with constants.model_scope(to_model_name):
                inf_engine.eval()
                logits2 = inf_engine.forward(input_=x, mb_spec=MicroBatchSpec())
            if logits1 is not None:
                assert torch.allclose(logits1, logits2, atol=2e-4)
        redist.backward()
        dist.barrier()

        # Synchronize the redistributed parameters. They should be identical to the initial parameters.
        if not skip_saveload:
            if from_model is not None:
                with constants.model_scope(from_model_name):
                    getattr(from_model, f"to_{model_family_name}")(
                        None, tmp_path / f"save_back_{i}"
                    )
            dist.barrier()
            sd3 = _load_all_pytorch_bin(tmp_path / f"save_back_{i}")
            for k, v in sd1.items():
                assert torch.allclose(v, sd3[k], atol=2e-4), (k, v, sd3[k])

        # Train the model.
        if from_model is not None:
            _check_tied_embedding_weights(from_model_name, from_model)
            train_engine.eval()

            p = from_model.contiguous_param.clone().detach()

            with constants.model_scope(from_model_name):
                train_engine: MockTrainEngine
                stats = train_engine.train_batch(
                    input_=x,
                    mb_spec=MicroBatchSpec(),
                    loss_fn=(
                        compute_packed_sft_loss
                        if not is_critic
                        else compute_critic_loss
                    ),
                    loss_weight_fn=lambda: 1,
                    token_normalize_scope="dp",
                    version_steps=i,
                )

            p_ = from_model.contiguous_param.clone().detach()
            # After training, the parameters should have changed.
            assert not torch.allclose(p, p_), (p - p_).abs().max()

        # Re-run redistribution to ensure that inference results changed.
        redist.forward()
        dist.barrier()
        if to_model is not None:
            _check_tied_embedding_weights(to_model_name, to_model)
            with constants.model_scope(to_model_name):
                inf_engine.eval()
                logits3 = inf_engine.forward(
                    input_=x,
                    mb_spec=MicroBatchSpec(),
                )
            if logits1 is not None:
                assert not torch.allclose(logits1, logits3)
        redist.backward()
        dist.barrier()
    print("success")


parallelism = [(4, 1, 1), (2, 2, 2), (1, 8, 1), (3, 2, 1), (2, 1, 2), (1, 2, 2)]


@pytest.mark.skip("NCCL-based parameter reallocation is not used currently.")
@pytest.mark.parametrize("model_family_name", ["gpt2", "llama"])
@pytest.mark.parametrize("is_critic", [False, True])
@pytest.mark.parametrize("from_pp_dp_tp", parallelism)
@pytest.mark.parametrize("to_pp_dp_tp", parallelism)
@pytest.mark.parametrize("skip_saveload", [False])
@pytest.mark.distributed
def test_param_realloc(
    tmp_path: pathlib.Path,
    model_family_name: str,
    is_critic: bool,
    from_pp_dp_tp: Tuple,
    to_pp_dp_tp: Tuple,
    skip_saveload: bool,
):
    if model_family_name == "gpt2" and (from_pp_dp_tp[-1] > 1 or to_pp_dp_tp[-1] > 1):
        # Since the vocabulary size of gpt2 is odd,
        # it does not support tensor model parallelism.
        return
    expr_name = uuid.uuid4()
    trial_name = uuid.uuid4()
    constants.set_force_cpu(True)
    test_impl = LocalMultiProcessTest(
        world_size=8,
        func=_test_para_realloc,
        expr_name=expr_name,
        trial_name=trial_name,
        timeout_secs=300,
        tmp_path=tmp_path,
        model_family_name=model_family_name,
        is_critic=is_critic,
        from_pp_dp_tp=from_pp_dp_tp,
        to_pp_dp_tp=to_pp_dp_tp,
        n_iterations=3,
        skip_saveload=skip_saveload,
    )
    test_impl.launch()
