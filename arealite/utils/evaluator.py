from typing import TYPE_CHECKING, Any, Callable

from arealite.api.cli_args import EvaluatorConfig
from arealite.api.io_struct import FinetuneSpec
from realhf.base import timeutil

if TYPE_CHECKING:
    from tensordict import TensorDict
    from torchdata.stateful_dataloader import StatefulDataLoader


class Evaluator:

    def __init__(self, config: EvaluatorConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_sepc = ft_spec
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )

    def evaluate(
        self,
        valid_dataloader: "StatefulDataLoader",
        evaluate_fn: Callable[["TensorDict"], Any],
        epoch: int,
        step: int,
        global_step: int,
    ):
        if not self.freq_ctl.check(
            epochs=int(step == self.ft_sepc.steps_per_epoch - 1), steps=1
        ):
            return
        for data in valid_dataloader:
            evaluate_fn(data)
