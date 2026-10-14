"""CPU subprocess fixture: production PPOTrainer methods, fake GPU engines.

Reuses the pinned AReaL restore tests; pytest is a test-only dependency here.
This exercises the real trainer lifecycle/restore hook, not distributed kernels.
"""

import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from autopilot_harness.execution.runtime import (
    load_managed_configuration,
    run_training,
    source_digest,
)
from autopilot_harness.validator.constraints import parse_layout

AREAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AREAL))

from tests.test_reconfiguration import rollout_types  # noqa: E402
from tests.test_reconfiguration_restore import WORKFLOW, M, _fixture  # noqa: E402
from tests.test_reconfiguration_stop import _with_lifecycle  # noqa: E402


def attach_runtime_packets(trainer, configuration, events):
    """Logical rank packets around fake weights, not a tensor DCP simulation."""
    actor = trainer.actor
    layout = parse_layout(configuration["actor"]["backend"])
    trainer.config.actor.backend = configuration["actor"]["backend"]
    trainer.actor_alloc.backend = layout.backend
    dims = (layout.dp, layout.tp, layout.pp, layout.cp, layout.ep, layout.etp)
    identity = dict(
        backend=layout.backend,
        device_type="cuda",
        parallelism=dict(zip(M._PARALLEL_DIMS, dims)),
        schedule={"total_steps": 40},
    )
    if layout.backend == "megatron":
        identity["parallelism"].update(
            virtual_pipeline_parallel_size=1,
            use_sequence_parallel=layout.tp > 1,
        )
        save = actor.save

        def save_portable(meta):
            assert meta.weight_format == "dcp-reconfiguration"
            # The CPU engine persists fixture counters, not native DCP tensors.
            # Its old serializer accepts one label; check production routing
            # above before delegating the fake bytes to that serializer.
            save(replace(meta, weight_format="dcp"))

        actor.save = save_portable
    streams = [actor.rng] + [
        random.Random(10 + rank) for rank in range(1, layout.world_size)
    ]
    actor.rng_mapping = "preserve"
    actor.optimizer = SimpleNamespace(param_groups=[{"lr": 0.25, "fused": True}])
    actor.optimizer_config = SimpleNamespace(type="adam")
    actor.is_vision_model = False
    actor.rank, actor.world_size = 0, layout.world_size

    def worker(rank):
        mesh = {
            name: SimpleNamespace(get_local_rank=lambda v=v: v)
            for name, v in (
                ("dp", rank // (layout.cp * layout.tp)),
                ("sp", rank // layout.tp % layout.cp),
                ("tp", rank % layout.tp),
            )
        }
        return SimpleNamespace(
            rank=rank,
            world_size=layout.world_size,
            parallel_helper=SimpleNamespace(world_mesh=mesh, tp_enabled=layout.tp > 1),
        )

    actor.parallel_helper = worker(0).parallel_helper
    actor.lr_scheduler = SimpleNamespace(
        load_state_dict=lambda value: setattr(
            actor, "schedule_step", value["last_epoch"]
        )
    )

    def capture():
        return M._encode_runtime_state(
            {
                "version": 1,
                "ranks": [
                    M._encode_runtime_state(
                        dict(
                            rank=rank,
                            identity=identity,
                            scheduler={"last_epoch": actor.schedule_step},
                            lrs=[0.25],
                            rng=stream.getstate(),
                        )
                    )
                    for rank, stream in enumerate(streams)
                ],
            }
        )

    def load(state):
        M.load_runtime_scheduler(actor, state, identity=identity)
        events.append("actor.schedule")

    def restore_rng(state):
        for rank, stream in enumerate(streams):
            local = M.select_runtime_state(worker(rank), state, identity=identity)
            if local.get("layout_mapped"):
                # A CPU stand-in consumes the derived host seed; actual torch
                # DTensor/device generator installation requires GPU tests.
                stream.seed(
                    local["rng"].get("host_seed", local["rng"].get("root_seed"))
                )
                actor.rng_mapping = local["rng"]["policy"]
            else:
                stream.setstate(local["rng"])
        events.append("actor.rng")

    update = actor.ppo_update

    def update_all(batch):
        update(batch)
        for stream in streams[1:]:
            stream.random()

    actor.runtime_state_dict = capture
    actor.load_runtime_state_dict = load
    actor.restore_runtime_rng = restore_rng
    actor.ppo_update = update_all


def main():
    with pytest.MonkeyPatch.context() as monkeypatch:
        trainer, events = _fixture(monkeypatch, critic=True)
        trainer = _with_lifecycle(trainer, events)
        trainer.config.total_train_epochs = 4
        trainer.config.total_train_steps = 8
        configuration = {"fixture": "native-ppo-hooks"}
        apply_mode = os.environ.get("AUTOPILOT_APPLY_FIXTURE")
        observed = []
        if apply_mode:
            configuration = load_managed_configuration(
                json.loads(Path(apply_mode).read_text())
            )
            trainer.config.total_train_epochs = 20
            trainer.config.total_train_steps = 40
            trainer.config.actor.mb_spec = SimpleNamespace(
                **configuration["actor"]["mb_spec"]
            )
            trainer.config.actor.pad_to_maximum = configuration["actor"][
                "pad_to_maximum"
            ]
            attach_runtime_packets(trainer, configuration, events)
            # Exercise the native counter loader and explicit mapping while
            # training/communication stay CPU fakes.
            classes = rollout_types.__wrapped__(monkeypatch)
            limits = configuration["rollout"]
            manager = classes.StalenessManager(
                trainer.rollout,
                limits["max_concurrent_rollouts"] or limits["consumer_batch_size"],
                limits["consumer_batch_size"],
                limits["max_head_offpolicyness"],
            )
            trainer.rollout.staleness_manager = manager
            save_state, load_state = (
                trainer.rollout.state_dict,
                trainer.rollout.load_state_dict,
            )

            def save_rollout():
                return dict(save_state(), staleness=manager.state_dict())

            def load_rollout(state):
                manager.load_state_dict(state["staleness"])
                load_state(state)

            trainer.rollout.state_dict = save_rollout
            trainer.rollout.load_state_dict = load_rollout
        original_batch = trainer.actor.prepare_batch

        def batch(*args, **kwargs):
            time.sleep(0.04 if apply_mode else 0.08)
            if apply_mode:
                observed.append(trainer.config.actor.mb_spec.n_mbs)
            return original_batch(*args, **kwargs)

        trainer.actor.prepare_batch = batch

        def factory(restore):
            trainer._snapshot_restore = (
                M.TrainerSnapshotRestore(restore) if restore else None
            )
            return trainer

        run_training(
            factory,
            configuration=configuration,
            runtime_source_id=source_digest(AREAL / "areal"),
            train_kwargs={"workflow": WORKFLOW},
            snapshot_api=M,
            supports_config_apply=bool(apply_mode),
        )
        directory = Path(os.environ["AUTOPILOT_LAUNCH_FILE"]).parent
        (directory / "native-fixture.json").write_text(
            json.dumps(
                {
                    "events": events,
                    "actor_loads": trainer.actor.loads,
                    "critic_loads": trainer.critic.loads,
                    "schedule_step": trainer.actor.schedule_step,
                    "restored": trainer.snapshot_restore_result is not None,
                    "configuration": configuration,
                    "observed_n_mbs": observed,
                    "staleness": manager.state_dict() if apply_mode else None,
                    "rng_mapping": trainer.actor.rng_mapping if apply_mode else None,
                }
            )
        )


if __name__ == "__main__":
    main()
