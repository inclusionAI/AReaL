import torch.distributed as dist

from arealite.impl.trainer.agentic.ppo import AgenticFsdpPPOTrainer


class FsdpAsyncPPOTrainer(AgenticFsdpPPOTrainer):
    def train(self):
        self._setup_models()
        self.rollout_controller.start_run_episode_loop(self.dataloader)

        world_size = dist.get_world_size()
        for _ in range(max_steps):
            # Wait until enough trajectories has been collected.
            trajs = self.rollout_controller.prepare_batch(
                batch_size=self.args.dataset.train_batch_size // world_size
            )

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

            stats = self.actor.async_train_batch(rollout, loss_fn=ppo_loss_fn)

            # TODO: resharding manager API
            self.actor.get_hf_model_state_dict()
            self.llm_client.update_weights_from(self.actor)
