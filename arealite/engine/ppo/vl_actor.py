from typing import Dict, Optional, List
import torch

from tensordict import TensorDict
from arealite.engine.ppo.actor import PPOActor, PPOActorConfig
from arealite.api.engine_api import TrainEngine
from arealite.engine.vl_fsdp_engine import VL_FSDPEngine
from arealite.utils.functional import (
    gather_logprobs,
)
from arealite.utils.image import process_image


class VL_PPOActor(PPOActor):
    """VL_PPOActor is a PPO actor for Vision-Language tasks."""

    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        super().__init__(config, engine)
        
    @torch.no_grad()
    def compute_logp(
        self,
        data: TensorDict,
        temperature: Optional[float] = None,
    ) -> torch.Tensor | None:
       
        
        def calc_logprobs(logits, input_data):
            labels = torch.roll(input_data["input_ids"], shifts=-1, dims=-1)
            logprobs = gather_logprobs(logits, labels, temperature or 1.0)
            return logprobs

        self.engine.eval()
        return self.engine.forward(
            input_=data,
            post_hook=calc_logprobs,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

class VL_FSDPPPOActor(VL_FSDPEngine):
    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = VL_PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> None:
        self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> List[Dict[str, float]]:
        return self.actor.ppo_update(*args, **kwargs)


