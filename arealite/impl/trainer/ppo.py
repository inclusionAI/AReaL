import multiprocessing as mp
from typing import Dict, List

from arealite.api.agent_api import AgentFactory
from arealite.api.cli_args import PPOTrainerConfig, TrainerConfig, TrainingArgs
from arealite.api.collector_api import TrajCollector, TrajCollectorFactory
from arealite.api.engine_api import EngineFactory
from arealite.api.env_api import EnvFactory
from arealite.api.io_struct import Trajectory
from arealite.api.llm_client_api import LLMClientFactory
from arealite.api.trainer_api import Trainer
from arealite.utils import concat_padded_tensors
from realhf.api.core.data_api import SequenceSample, gather_stat
from realhf.experiments.common.utils import AllocationMode
from realhf.impl.model.utils.padding import pad_input, unpad_input


class FsdpPPOTrainer(Trainer):

    def __init__(self, args: TrainingArgs, trainer_config: TrainerConfig):
        super().__init__(args, trainer_config)
        self.config = config = trainer_config.ppo

        # Create models
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

        # Rollout collector
        self.collector_factory = TrajCollectorFactory(args)

        # Create a client to update weights
        client_factory = LLMClientFactory(args)
        self.llm_client = client_factory.make_client(config.inf_service)

    def _setup_models(self):
        self.actor.init_distributed(self.config.ppo.actor_train)
        if self.critic is not None:
            self.critic.init_distributed(self.config.ppo.critic_train)
        if self.ref is not None:
            self.ref.ini

    def train(self):
        self._setup_models()

        for data in self.dataloader:
            bs: int

            # Run batched rollout
            # Make new collectors to avoid data race
            collectors = [
                self.collector_factory.make_collector(self.config.collector)
                for _ in range(bs)
            ]
            trajs = TrajCollector.run_episode_batch(collectors, env_options=data)

            data = concat_padded_tensors([traj.data for traj in trajs])
            stats = gather_stat([traj.stats for traj in trajs])

            # Convert trajectories to SequenceSample
            attn_mask = data["attention_mask"]
            packed_input_ids = unpad_input(data["packed_input_ids"], attn_mask)

            # Run reference model inference
            if self.ref is not None and self.config.kl_ctl != 0.0:
                ref_logits = self.ref.forward(data)
                ref_logp = gather_logp(ref_logits, input_ids)

            # Compute GAE here...
            ...

            stats = self.actor.train_batch(rollout, loss_fn=ppo_loss_fn)

            # TODO: resharding manager API
            state_dict = self.actor.get_hf_model_state_dict()
            self.llm_client.update_weights_from(self.actor)
