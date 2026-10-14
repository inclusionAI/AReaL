# SPDX-License-Identifier: Apache-2.0
"""Native reconfiguration mapping contracts, imported without GPU packages."""

import random
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.test_reconfiguration import M


class NativeTransitionTests(unittest.TestCase):
    def setUp(self):
        self.transition = M.ConfigurationTransition(
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
            "sha256:" + "3" * 64,
            (16, 8, 2),
            (4, 8, 2),
        )
        self.state = dict(
            staleness=dict(
                schema_version=1,
                limits=[16, 8, 2],
                counters=dict(accepted=10, enqueued=7, rejected=2, running=0),
            ),
            inventory={"pending": list(range(7)), "ready": ["ready-result"]},
            task_ids={"next_id": 32},
            input_buffer={"remaining": [33, 34]},
            policy_version=3,
        )

    def test_lower_capacity_preserves_inventory_counters_and_source(self):
        original = deepcopy(self.state)
        mapped = self.transition.map_rollout_state(self.state, [4, 8, 2])
        expected = deepcopy(original)
        expected["staleness"]["limits"][0] = 4
        self.assertEqual(mapped, expected)
        mapped["inventory"]["pending"].clear()
        mapped["staleness"]["counters"]["accepted"] = 999
        self.assertEqual(self.state, original)

    def test_mismatched_source_target_or_non_drained_state_fails(self):
        for source_limits, target_limits, running in [
            ([15, 8, 2], [4, 8, 2], 0),
            ([16, 8, 2], [8, 8, 2], 0),
            ([16, 8, 2], [4, 8, 2], 1),
        ]:
            with self.subTest(
                limits=source_limits, target=target_limits, running=running
            ):
                self.state["staleness"]["limits"] = source_limits
                self.state["staleness"]["counters"]["running"] = running
                with self.assertRaises(M.SnapshotError):
                    self.transition.map_rollout_state(self.state, target_limits)

    def test_unknown_adapter_staleness_changes_and_invalid_limits_fail(self):
        for kwargs in [
            dict(adapter_id="unknown"),
            dict(target_limits=(4, 8, 3)),
            dict(target_limits=(4, 16, 2)),
            dict(target_limits=(0, 8, 2)),
            dict(target_limits=(True, 8, 2)),
            dict(plan_id="unbound"),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.transition, **kwargs)

    def test_restore_checks_transition_source_before_worker_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            context = M.SnapshotContext(
                "test",
                "runtime",
                self.transition.source_configuration_id,
                "state",
                dict(epoch=0, epoch_step=0, global_step=0, steps_per_epoch=2),
                1,
            )

            def save(path):
                (path / "weights").write_bytes(b"fixture")

            checkpoint = M.CheckpointParticipant(
                "fixture", save, lambda: None, lambda _: None
            )
            manager = M.ReconfigurationManager(
                checkpoints={"actor": checkpoint}, statefuls={}
            )
            snapshot = manager.save_snapshot(
                Path(directory) / "snapshot", context=context
            )
            request = M.SnapshotRestoreRequest(snapshot, context, self.transition)
            controller = M.TrainerSnapshotRestore(request)
            self.assertEqual(controller.next_step, 1)
            self.assertEqual(controller.transition, self.transition)
            with self.assertRaises(M.SnapshotError):
                M.TrainerSnapshotRestore(
                    replace(
                        request,
                        transition=replace(
                            self.transition,
                            source_configuration_id="sha256:" + "4" * 64,
                        ),
                    )
                )


class TrainingLayout:
    """Logical rank packets for mapping tests; no native tensor emulation."""

    def __init__(self, *, backend="fsdp", target=(2, 2, 1, 1, 1, 1)):
        self.backend = backend
        self.source = (4, 1, 1, 1, 1, 1)
        self.target = target
        self.transition = M.ConfigurationTransition(
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
            "sha256:" + "3" * 64,
            (16, 8, 2),
            (4, 8, 2),
            training_layout={
                "source": self.source,
                "target": self.target,
                "backend": backend,
                "rng_policy": M.MEGATRON_LAYOUT_RNG_POLICY
                if backend == "megatron"
                else M.FSDP_LAYOUT_RNG_POLICY,
            },
        )
        self.snapshot = M.SnapshotRef("/fixture", "saved-boundary", "4" * 64)

    def identity(self, dims):
        parallelism = dict(zip(M._PARALLEL_DIMS, dims))
        if self.backend == "megatron":
            parallelism.update(
                virtual_pipeline_parallel_size=1, use_sequence_parallel=dims[1] > 1
            )
        return dict(
            backend=self.backend,
            device_type="cuda",
            parallelism=parallelism,
            schedule={"total_steps": 100, "warmup_steps": 10},
        )

    def packet(self):
        return M._encode_runtime_state(
            {
                "version": 1,
                "ranks": [
                    M._encode_runtime_state(
                        dict(
                            rank=rank,
                            identity=self.identity(self.source),
                            scheduler={"last_epoch": 12, "_step_count": 13},
                            lrs=[0.25],
                            rng={"original_rank_stream": rank},
                        )
                    )
                    for rank in range(self.transition.training_layout.source_world_size)
                ],
            }
        )

    def mapped(self):
        return self.transition.map_training_state(self.packet(), self.snapshot)

    def engine(self, rank=0):
        # Coordinates are adapter inputs, not mocked FSDP/DTensor operations.
        dp, tp, _, cp, _, _ = self.target
        mesh = {
            name: SimpleNamespace(get_local_rank=lambda v=v: v)
            for name, v in (
                ("dp", rank // (cp * tp)),
                ("sp", rank // tp % cp),
                ("tp", rank % tp),
            )
        }
        return SimpleNamespace(
            rank=rank,
            world_size=dp * tp * cp,
            parallel_helper=SimpleNamespace(world_mesh=mesh, tp_enabled=tp > 1),
            optimizer=SimpleNamespace(param_groups=[{"lr": 99, "fused": True}]),
            optimizer_config=SimpleNamespace(type="adam"),
            is_vision_model=True,
            lr_scheduler=SimpleNamespace(load_state_dict=Mock()),
        )

    def select(self, rank=0, state=None):
        return M.select_runtime_state(
            self.engine(rank),
            state or self.mapped(),
            identity=self.identity(self.target),
        )


class TrainingLayoutTests(unittest.TestCase, TrainingLayout):
    setUp = TrainingLayout.__init__

    def test_target_streams_repeat_and_tp_peers_share_only_tensor_seed(self):
        state = self.mapped()
        before = random.getstate()
        locals_ = [self.select(rank, state) for rank in range(4)]
        self.assertEqual(before, random.getstate())
        self.assertEqual(
            [item["rng"]["host_seed"] for item in locals_],
            [self.select(rank, state)["rng"]["host_seed"] for rank in range(4)],
        )
        self.assertEqual(len({item["rng"]["host_seed"] for item in locals_}), 4)
        seeds = [item["rng"]["tensor_seed"] for item in locals_]
        self.assertEqual(seeds[0], seeds[1])
        self.assertEqual(seeds[2], seeds[3])
        self.assertNotEqual(seeds[0], seeds[2])
        for item in locals_:
            self.assertEqual(item["scheduler"], {"last_epoch": 12, "_step_count": 13})
            self.assertEqual(item["lrs"], [0.25])
            self.assertNotIn("original_rank_stream", item["rng"])
        provenance = M._decode_runtime_state(state)["seed_material"]
        self.assertEqual(provenance["snapshot_id"], self.snapshot.snapshot_id)
        self.assertEqual(provenance["plan_id"], self.transition.plan_id)
        changed = replace(self.transition, plan_id="sha256:" + "5" * 64)
        other = changed.map_training_state(self.packet(), self.snapshot)
        self.assertNotEqual(
            self.select(state=state)["rng"]["host_seed"],
            self.select(state=other)["rng"]["host_seed"],
        )

    def test_world_size_and_cp_changes_do_not_copy_source_rank_rng(self):
        self.target = (2, 2, 1, 2, 1, 1)
        self.transition = replace(
            self.transition,
            training_layout=M.TrainingLayoutTransition(self.source, self.target),
        )
        locals_ = [self.select(rank) for rank in range(8)]
        self.assertEqual([item["rank"] for item in locals_], list(range(8)))
        seeds = [item["rng"]["tensor_seed"] for item in locals_]
        self.assertEqual(len(set(seeds)), 4)
        self.assertTrue(all(seeds[i] == seeds[i + 1] for i in range(0, 8, 2)))

    def test_scheduler_load_and_fused_target_option_do_not_install_rng_early(self):
        engine = self.engine()
        with patch.object(M, "_install_fsdp_rng") as install:
            M.load_runtime_scheduler(
                engine, self.mapped(), identity=self.identity(self.target)
            )
            install.assert_not_called()
            self.assertEqual(
                engine.optimizer.param_groups, [{"lr": 0.25, "fused": False}]
            )
            engine.lr_scheduler.load_state_dict.assert_called_once_with(
                {"last_epoch": 12, "_step_count": 13}
            )
            M.restore_rng_state(self.select()["rng"])
            install.assert_called_once()

    def test_mismatched_target_source_schedule_or_progress_fails(self):
        for defect in ("target", "source", "scheduler", "lr", "schedule", "world"):
            with self.subTest(defect=defect):
                packet = M._decode_runtime_state(self.packet())
                rank = M._decode_runtime_state(packet["ranks"][1])
                engine, identity = self.engine(), self.identity(self.target)
                if defect == "source":
                    rank["identity"]["parallelism"]["data_parallel_size"] = 8
                elif defect == "scheduler":
                    rank["scheduler"]["last_epoch"] = 11
                elif defect == "lr":
                    rank["lrs"] = [0.5]
                elif defect == "target":
                    identity["parallelism"]["data_parallel_size"] = 8
                elif defect == "schedule":
                    identity["schedule"]["total_steps"] = 200
                else:
                    engine.world_size = 8
                packet["ranks"][1] = M._encode_runtime_state(rank)
                mapped = self.transition.map_training_state(
                    M._encode_runtime_state(packet), self.snapshot
                )
                with self.assertRaises(M.SnapshotError):
                    M.load_runtime_scheduler(engine, mapped, identity=identity)
                engine.lr_scheduler.load_state_dict.assert_not_called()

    def test_same_layout_restore_still_restores_exact_rank_rng(self):
        engine = self.engine()
        engine.world_size = 4
        result = M.select_runtime_state(
            engine, self.packet(), identity=self.identity(self.source)
        )
        self.assertEqual(result["rng"], {"original_rank_stream": 0})
        with self.assertRaises(M.SnapshotError):
            M.select_runtime_state(
                engine, self.packet(), identity=self.identity(self.target)
            )

    def test_same_layout_restore_supplies_process_local_dtensor_mesh(self):
        packet = M._decode_runtime_state(self.packet())
        rank = M._decode_runtime_state(packet["ranks"][0])
        rank["rng"]["dtensor"] = {
            "initialized": True,
            "distribute_region_enabled": True,
        }
        packet["ranks"][0] = M._encode_runtime_state(rank)
        state = M._encode_runtime_state(packet)
        engine = self.engine()
        local = M.select_runtime_state(
            engine, state, identity=self.identity(self.source)
        )
        self.assertIs(
            local["rng"]["_dtensor_mesh"], engine.parallel_helper.world_mesh["tp"]
        )
        persisted = M._decode_runtime_state(M._decode_runtime_state(state)["ranks"][0])
        self.assertNotIn("_dtensor_mesh", persisted["rng"])

    def test_layout_mapping_is_deferred_through_existing_runtime_owner(self):
        engine = SimpleNamespace(
            runtime_state_dict=Mock(return_value=self.packet()),
            load_runtime_state_dict=Mock(return_value=None),
            restore_runtime_rng=Mock(return_value=None),
        )
        owner = M._RestoreTrainingRuntime(
            M.EngineRuntimeState(engine), self.transition, self.snapshot
        )
        owner.load_state_dict(self.packet())
        state = engine.load_runtime_state_dict.call_args.args[0]
        self.assertEqual(M._decode_runtime_state(state)["version"], 2)
        engine.restore_runtime_rng.assert_not_called()
        owner.finalize_restore()
        engine.restore_runtime_rng.assert_called_once_with(state)

    def test_unimplemented_native_dimensions_cannot_be_enabled_by_mapping(self):
        for target in ((2, 2, 2, 1, 1, 1), (2, 2, 1, 1, 2, 1), self.source):
            with self.subTest(target=target), self.assertRaises(ValueError):
                M.TrainingLayoutTransition(self.source, target)


if __name__ == "__main__":
    unittest.main()
