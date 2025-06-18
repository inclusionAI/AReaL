from typing import Optional

from datasets import Dataset

from arealite.api.cli_args import TrainerConfig, TrainingArgs
from arealite.api.engine_api import EngineFactory
from arealite.api.llm_client_api import LLMClientFactory
from arealite.api.trainer_api import Trainer
from arealite.impl.rollout_controller import RolloutController
from arealite.impl.traj_collector import TrajCollector
from arealite.utils import concat_padded_tensors
from realhf.api.core.data_api import gather_stat
from realhf.impl.model.utils.padding import unpad_input


class AgenticFsdpPPOTrainer(Trainer):
    # TODO: simplify to RLVR pipeline

    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional[RolloutController] = None,
    ):
        super().__init__(
            args, trainer_config, train_dataset, valid_dataset, rollout_controller
        )
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

        # Use provided rollout controller or create one if None
        if self.rollout_controller is None:
            self.rollout_controller = RolloutController(
                args,
                config.rollout_controller,
                config.collector,
            )

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
            # Run batched rollout
            # TODO: separate normal RLVR and agentic RL
            trajs = self.rollout_controller.run_episode_batch(env_options=data)

            data = concat_padded_tensors([traj.data for traj in trajs])
            stats = gather_stat([traj.stats for traj in trajs])

            # Convert trajectories to SequenceSample
            attn_mask = data["attention_mask"]
            unpad_input(data["packed_input_ids"], attn_mask)

            # Run reference model inference
            if self.ref is not None and self.config.kl_ctl != 0.0:
                ref_logits = self.ref.forward(data)
                gather_logp(ref_logits, input_ids)

            # Compute GAE here...
            ...

            stats = self.actor.train_batch(rollout, loss_fn=ppo_loss_fn)

            # TODO: resharding manager API
            # self.actor.get_hf_model_state_dict()
            self.actor.update_weights_to(self.llm_client)
            # self.llm_client.update_weights_from(self.actor)
