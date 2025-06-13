import multiprocessing as mp
from typing import Dict, List

from realhf.api.core.data_api import gather_stat
from realhf.experiments.common.utils import AllocationMode
from refactoring.api.agent_api import AgentFactory
from refactoring.api.cli_args import PPOTrainerConfig, TrainingArgs
from refactoring.api.collector_api import TrajCollectorFactory
from refactoring.api.engine_api import EngineFactory
from refactoring.api.env_api import EnvFactory
from refactoring.api.io_struct import Trajectory
from refactoring.api.llm_client_api import LLMClientFactory
from refactoring.api.trainer_api import Trainer
from refactoring.utils import concat_padded_tensors




class PPOTrainer(Trainer):

    def __init__(self, args: TrainingArgs, config: PPOTrainerConfig):
        super().__init__(args, config)

        engine_factory = EngineFactory(args)
        self.actor = engine_factory.make_engine(config.actor)

        self.critic = None
        if config.critic is not None:
            self.critic = engine_factory.make_engine(config.critic)
        self.ref = None
        if config.ref is not None:
            self.ref = engine_factory.make_engine(config.ref)
        self.rew = None
        if config.rew is not None:
            self.rew = engine_factory.make_engine(config.rew)

        # Create a client to update weights
        client_factory = LLMClientFactory(args)
        self.llm_client = client_factory.make_client(args.collector.agent.inf_service)

    def train(self):

        for _ in range(50):
            # Different process may have different number of samples (even 0)
            trajectory: Trajectory = data_pipe.wait_for(global_bs=512)

            # redistribute data such that
            # rollout is duplicated across TP/PP/CP groups
            rollout = DataRedistributor.redistribute(trajectory)

            if self.ref is not None and self.config.ppo.ppo.kl_ctl != 0.0:
                ref_logp = self.ref.forward(rollout)
                rollout.update(**ref_logp)

            # Compute GAE here...
            ...

            stats = self.actor.train_batch(rollout, loss_fn=ppo_loss_fn)

            # TODO: resharding manager API
            self.actor.get_partial_hf_state_dict()
            self.llm_client.update_weights_from(self.actor)
