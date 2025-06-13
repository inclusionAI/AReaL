import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal

import torch

from realhf.api.cli_args import ParallelismConfig
from refactoring.api.cli_args import EngineConfig, MicroBatchSpec, TrainingArgs


class SPMDBackendWrapper(abc.ABC):
    """A wrapper over the training/inference backends (e.g., FSDP, SGLang).
    We following the design of existing libraries, such as Megatron-LM and
    pytorch FSDP, which are mostly SPMD-based.
    """

    def __init__(self, args: TrainingArgs, config: EngineConfig):
        self.args = args
        self.config = config

    def init_parallelism_groups(self):
        pass

    def train_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[torch.Tensor, Dict], float],
        version_steps: int,
        token_normalize_scope: Literal["global", "dp"] = "global",
    ) -> Dict:
        """Update the model with a batch of data and a loss function.

        :param input_: The input data.
        :type input_: Dict
        :param mb_spec: The micro-batch specification, which defines how the input data
            should be split into micro-batches.
        :type mb_spec: MicroBatchSpec
        :param loss_fn: The loss function. It takes the output of the forward pass and the
            input data, returning the loss.
        :type loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor]
        :param loss_weight_fn: This function is used to calculate the number of valid tokens
            when normalizing loss across micro batches and DP ranks. Can be `lambda: 1`
            if just taking the average over batches.
        :type loss_weight_fn: Callable[[torch.Tensor, Dict], float]
        :param version_steps: The global step counter for this experiment,
            used by the backend to determine the learning rate schedule.
        :type version_steps: int
        :param global_normalize_scope: The scope of token-wise loss normalization. Choices:
            global: average across all micro batches across DP ranks.
            dp: average across micro batches in current DP rank.
            Default to "global".
        :type global_normalize_scope: Literal["global", "dp"]
        """

        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function.

        This method wraps :meth:`forward` with a customized ``post_hook`` and ``aggregate_fn``.
        :param input_: The input data.
        :type input_: Dict
        :param loss_fn: The loss function. It takes the output of the forward pass and the
            input data, returning the loss.
        :type loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor]
        :return: The aggregated scalar loss if on the last pipe stage.
        :rtype: torch.Tensor | None
        """

        def _loss_fn(out, inp_):

            # To prevent calling data reordering.

            return float(loss_fn(out, inp_))

        return self.forward(
            input_=input_,
            mb_spec=mb_spec,
            post_hook=_loss_fn,
            aggregate_fn=sum,
        )

    def forward(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run the forward pass or inference on the model. Note that it is
        gradient-free.

        To train the model, use :meth:`train_batch` instead.
        :param input_: The input data.
        :type input_: Dict
        :param post_hook: A function to apply to the output after the forward pass.
            It takes the output tensor and the input data, returning an arbitrary result.
            With a post_hook, we can process the output in mini-batches,
            reducing memory usage for operations such as gathering log-probabilities.
            If None, this function just returns the output tensor.
        :type post_hook: Callable[[torch.Tensor, Dict], Any] | None
        :param aggregate_fn: A function to aggregate the results of the post_hook.
        :type aggregate_fn: Callable[[List[Any]], Any]
        :return: The aggregated result of the post_hook from the last pipeline stage. Returns None otherwise.
        :rtype: Any | None
        """

        raise NotImplementedError()


class DataManager:
    def scatter(self, input_: Dict, mb_spec: MicroBatchSpec) -> Dict:
        """Scatter the input data across micro-batches."""
        # Implementation of scattering logic goes here.
        return input_

    def gather(self, output: Dict) -> Dict:
        return output

    def gather_stats(self, stats: Dict) -> Dict:
        return stats


import torch.distributed as dist
import torch.multiprocessing as mp


class EngineExecutor:
    def __init__(self, args: TrainingArgs, config: EngineConfig):
        self.config = config
        self.args = args

    def setup_distributed(self, args: TrainingArgs, para: ParallelismConfig):
        """Start worker processes"""
        procs = [
            mp.Process(
                target=self._worker_fn,
                args=(
                    i,
                    args,
                    para,
                ),
            )
            for i in range(para.world_size)
        ]
        for p in procs:
            p.start()

    def _worker_fn(self, rank, args, para: ParallelismConfig):
        """Worker process - handles FSDP training"""
        # Initialize distributed
        dist.init_process_group("nccl", rank=rank, world_size=para.world_size)
        torch.cuda.set_device(rank)

        factory = EngineFactory(args)
        engine = factory.make_engine(self.config)
        engine.init_parallelism_groups()

        # Worker training loop
        while True:
            try:
                # Receive global batch from controller
                batch = self._receive_batch(rank)
                if batch is None:  # Shutdown signal
                    break

                # Get worker's portion of the batch
                data, target = self._split_batch(batch, rank)

                # Standard FSDP training step
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()

                # Send loss back to controller
                self._send_result(loss.item(), rank)

            except Exception as e:
                print(f"Worker {rank} error: {e}")
                break

        dist.destroy_process_group()

    def _receive_batch(self, rank):
        """Receive batch data from controller"""
        # Simple implementation using file-based communication
        # In production, use proper IPC mechanisms
        import os
        import pickle
        import time

        while True:
            try:
                if os.path.exists(f"/tmp/batch_{rank}.pkl"):
                    with open(f"/tmp/batch_{rank}.pkl", "rb") as f:
                        batch = pickle.load(f)
                    os.remove(f"/tmp/batch_{rank}.pkl")
                    return batch
                elif os.path.exists("/tmp/shutdown.signal"):
                    return None
                time.sleep(0.01)
            except:
                time.sleep(0.01)

    def _send_result(self, loss, rank):
        """Send result back to controller"""
        import pickle

        with open(f"/tmp/result_{rank}.pkl", "wb") as f:
            pickle.dump(loss, f)

    def _split_batch(self, batch, rank):
        """Split global batch for this worker"""
        data, target = batch
        chunk_size = len(data) // self.world_size
        start = rank * chunk_size
        end = start + chunk_size if rank < self.world_size - 1 else len(data)
        return data[start:end].cuda(), target[start:end].cuda()

    def train_batch(self, batch):
        """Train on global batch - called from controller"""
        import os
        import pickle
        import time

        # Send batch to all workers
        for rank in range(self.world_size):
            with open(f"/tmp/batch_{rank}.pkl", "wb") as f:
                pickle.dump(batch, f)

        # Wait for results from all workers
        losses = []
        for rank in range(self.world_size):
            while not os.path.exists(f"/tmp/result_{rank}.pkl"):
                time.sleep(0.01)
            with open(f"/tmp/result_{rank}.pkl", "rb") as f:
                loss = pickle.load(f)
            losses.append(loss)
            os.remove(f"/tmp/result_{rank}.pkl")

        return sum(losses) / len(losses)

    def cleanup(self):
        """Shutdown workers"""
        import os

        with open("/tmp/shutdown.signal", "w") as f:
            f.write("shutdown")


class EngineExecutor:

    def __init__(
        self,
        base_engine: SPMDBackendWrapper,
        data_manager: DataManager,
    ):
        self.base_engine = base_engine
        self.data_manager = data_manager

    def train_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[torch.Tensor, Dict], float],
        version_steps: int,
        token_normalize_scope: Literal["global", "dp"] = "global",
    ) -> Dict:
        input_ = self.data_manager.scatter(input_, mb_spec)
        stats = self.base_engine.train_batch(
            input_,
            mb_spec,
            loss_fn,
            loss_weight_fn,
            version_steps,
            token_normalize_scope,
        )
        stats = self.data_manager.gather_stats(stats)
        return stats


@dataclass
class EngineFactory:
    args: TrainingArgs

    def make_engine(self, config: EngineConfig) -> SPMDBackendWrapper:
        """Create an engine based on the configuration."""
        if config.backend == "fsdp":
            from refactoring.impl.model.fsdp_engine import FSDPEngine

            base_engine = FSDPEngine(self.args, config)
        else:
            raise ValueError(f"Unsupported engine type: {config.backend}")
