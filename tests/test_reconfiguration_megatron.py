# SPDX-License-Identifier: Apache-2.0
"""Portable checkpoint interface/common-state tests without GPU dependencies.

Execute production Python methods with transport/engine stand-ins. These tests
do not simulate or claim to verify native tensor resharding or CUDA collectives.
"""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests import test_reconfiguration_transition as runtime_fixtures
from tests.test_reconfiguration import _source_function, _source_namespace

M = runtime_fixtures.M


CHECKPOINTER = "areal/engine/megatron_utils/checkpointer.py"


class LocalValue:
    def __init__(self, data):
        self.data = data


class TensorPlaceholder:
    pass


def source_namespace():
    scope = dict(
        copy=copy,
        LocalNonpersistentObject=LocalValue,
        os=os,
        torch=SimpleNamespace(Tensor=TensorPlaceholder, distributed=Mock()),
        metadata=SimpleNamespace(version=lambda _: "0.17.0"),
        mpu=Mock(),
        logger=Mock(),
        log_with_rank=Mock(),
    )
    return _source_namespace(
        CHECKPOINTER,
        {
            "_PORTABLE_METADATA",
            "_PORTABLE_COMMON",
            "_GROUP_KEYS",
            "_mcore_version",
            "_common_equal",
            "_common_to_cpu",
            "_optimizer_leaves",
            "_merge_optimizer_common",
            "_gather_optimizer_common",
            "_portable_optimizer_template",
            "MegatronCheckpointManager",
        },
        scope,
    )


def group(*, expert=False, wd=1.0, step=12):
    return dict(
        wd_mult=wd,
        lr_mult=1.0,
        is_decoupled_lr=False,
        is_expert_parallel=expert,
        lr=0.25,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        step=step,
    )


def leaf(groups, tensors=None):
    return dict(
        optimizer={"param_groups": groups},
        grad_scaler={"scale": 1024, "growth_tracker": 3},
        param_state={0: {"param": "native-master", "exp_avg": "native-moment"}}
        if tensors is None
        else tensors,
        param_state_sharding_type="fully_reshardable",
    )


def common(leaf_state):
    return {k: v for k, v in leaf_state.items() if not k.startswith("param_state")}


class CheckpointerTestCase(unittest.TestCase):
    def setUp(self):
        self.ns = source_namespace()
        modules = patch.dict(
            "sys.modules",
            {
                "megatron.core.dist_checkpointing.mapping": SimpleNamespace(
                    LocalNonpersistentObject=LocalValue
                ),
            },
        )
        modules.start()
        self.addCleanup(modules.stop)


class PortableCommonTests(CheckpointerTestCase):
    def test_ep_chain_merge_and_split_preserve_steps_scaler_and_tensor_template(self):
        merge = self.ns["_merge_optimizer_common"]
        template = self.ns["_portable_optimizer_template"]
        # Cover both distinct semantic groups and dense/expert copies of one group.
        for expert_wd in (0.0, 1.0):
            with self.subTest(expert_wd=expert_wd):
                source = {
                    0: leaf([group()]),
                    1: leaf([group(expert=True, wd=expert_wd)]),
                }
                source_before = copy.deepcopy(source)
                canonical = merge([[common(source[0]), common(source[1])]])
                # Target now owns both groups in one optimizer, in a different order.
                target = leaf([group(wd=expert_wd, step=0), group(step=0)])
                tensors = target["param_state"]
                template(target, canonical)
                self.assertIs(target["param_state"], tensors)
                self.assertEqual(
                    target["optimizer"].data["param_groups"],
                    [group(wd=expert_wd), group()],
                )
                self.assertEqual(target["grad_scaler"].data, source[0]["grad_scaler"])
                self.assertEqual(
                    target["param_state_sharding_type"].data, "fully_reshardable"
                )
                self.assertEqual(source, source_before)
                # Restore separate children while retaining target ownership.
                split = {
                    0: leaf([group(step=0)]),
                    1: leaf([group(expert=True, wd=expert_wd, step=0)]),
                }
                for child in split.values():
                    child["param_state"] = tensors
                template(split, canonical)
                for index, expected in enumerate(
                    (group(), group(expert=True, wd=expert_wd))
                ):
                    self.assertIs(split[index]["param_state"], tensors)
                    self.assertEqual(
                        split[index]["optimizer"].data["param_groups"], [expected]
                    )
                    self.assertEqual(
                        split[index]["grad_scaler"].data, source[0]["grad_scaler"]
                    )

    def test_missing_step_in_empty_child_inherits_global_progress(self):
        partial = group(expert=True)
        del partial["step"]
        canonical = self.ns["_merge_optimizer_common"](
            [
                [common(leaf([group()]))],
                [common(leaf([partial]))],
            ]
        )
        self.assertEqual(
            canonical["optimizer"]["param_groups"],
            [dict((k, v) for k, v in group().items() if k != "is_expert_parallel")],
        )

    def test_inconsistent_state_is_not_silently_collapsed(self):
        for field, value in (
            ("step", 99),
            ("lr", 0.5),
            ("betas", (0.8, 0.99)),
            ("eps", 1e-6),
        ):
            with self.subTest(field=field):
                changed = group(expert=True)
                changed[field] = value
                with self.assertRaises(ValueError):
                    self.ns["_merge_optimizer_common"](
                        [[common(leaf([group(), changed]))]]
                    )
        a, b = common(leaf([group()])), common(leaf([group()]))
        b["grad_scaler"]["growth_tracker"] = 99
        with self.assertRaises(ValueError):
            self.ns["_merge_optimizer_common"]([[a], [b]])

    def test_save_strips_source_chain_common_and_empty_param_state(self):
        source = {0: leaf([group()]), 1: leaf([group(expert=True)], tensors={})}
        source[0]["param_state"] = None
        self.ns["_portable_optimizer_template"](source)
        for child in source.values():
            self.assertEqual(set(child), {"param_state"})
            self.assertIsInstance(child["param_state"], LocalValue)

    def test_target_hyperparameters_or_semantic_groups_cannot_change(self):
        canonical = self.ns["_merge_optimizer_common"]([[common(leaf([group()]))]])
        for changed in (dict(group(), eps=1e-6), group(wd=0.0)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.ns["_portable_optimizer_template"](leaf([changed]), canonical)

    def test_collective_propagates_remote_capture_failure(self):
        dist = self.ns["torch"].distributed
        dist.get_world_size.return_value = 2

        def gather(output, payload):
            output[:] = [payload, {"error": "capture failed"}]

        dist.all_gather_object.side_effect = gather
        with self.assertRaisesRegex(ValueError, "capture"):
            self.ns["_gather_optimizer_common"](leaf([group()]))
        dist.all_gather_object.assert_called_once()


class CheckpointerInterfaceTests(CheckpointerTestCase):
    def setUp(self):
        super().setUp()
        cls = self.ns["MegatronCheckpointManager"]
        self.manager = cls.__new__(cls)
        self.manager.model = [
            SimpleNamespace(sharded_state_dict=Mock(return_value={"weight": "sharded"}))
        ]
        self.manager.optimizer = SimpleNamespace(
            sharded_state_dict=Mock(return_value=leaf([group()]))
        )
        self.manager.lr_scheduler = None
        self.manager.get_rng_state = Mock(return_value="native-rng")

    def test_portable_metadata_reaches_model_optimizer_and_loading_template(self):
        state = self.manager.generate_state_dict(
            with_rng=False, is_loading=True, optimizer_sharding_type="fully_reshardable"
        )
        call = self.manager.optimizer.sharded_state_dict.call_args
        self.assertTrue(call.kwargs["is_loading"])
        metadata = call.kwargs["metadata"]
        self.assertEqual(
            metadata,
            {
                "distrib_optim_sharding_type": "fully_reshardable",
                "chained_optim_avoid_prefix": True,
                "singleton_local_shards": False,
                "distrib_optim_fully_reshardable_mem_efficient": True,
            },
        )
        self.manager.model[0].sharded_state_dict.assert_called_once_with(
            metadata=metadata
        )
        self.assertNotIn("rng_state", state)
        self.manager.get_rng_state.assert_not_called()

    def test_ordinary_recovery_keeps_original_metadata_and_rng(self):
        for loading in (False, True):
            with self.subTest(loading=loading):
                self.manager.model[0].sharded_state_dict.reset_mock()
                state = self.manager.generate_state_dict(is_loading=loading)
                self.manager.model[0].sharded_state_dict.assert_called_once_with()
                self.assertEqual(
                    self.manager.optimizer.sharded_state_dict.call_args.kwargs,
                    {
                        "is_loading": loading,
                        "metadata": {"distrib_optim_sharding_type": "dp_reshardable"},
                    },
                )
                self.assertEqual(state["rng_state"], "native-rng")

    def test_unknown_dependency_version_cannot_claim_portable_format(self):
        self.ns["metadata"].version = lambda _: "0.18.0"
        with self.assertRaisesRegex(ValueError, "pinned"):
            self.manager.generate_state_dict(
                optimizer_sharding_type="fully_reshardable"
            )
        self.manager.model[0].sharded_state_dict.assert_not_called()

    def test_save_load_roundtrip_maps_common_state_into_target_chain(self):
        manager = self.manager
        manager.rank, manager.async_save, manager.use_dist_checkpointing = (
            0,
            False,
            True,
        )
        manager.use_checkpoint_opt_param_scheduler = False
        manager._reap_finished_async_saves = Mock()
        manager.wait_async_saves = Mock()
        self.ns["torch"].distributed.get_world_size.return_value = 1
        self.ns["torch"].distributed.all_gather_object.side_effect = (
            lambda out, value: out.__setitem__(0, value)
        )
        storage = {}

        def save(**kwargs):
            storage.update(kwargs)

        self.ns["save_dist_checkpointing"] = save
        with tempfile.TemporaryDirectory() as directory:
            manager.save_checkpoint(
                directory, with_rng=False, optimizer_sharding_type="fully_reshardable"
            )
            self.assertEqual(storage["content_metadata"], self.ns["_PORTABLE_METADATA"])
            source = storage["sharded_state_dict"]
            self.assertEqual(set(source["optimizer"]), {"param_state"})
            self.ns["dist_checkpointing"] = SimpleNamespace(
                load_common_state_dict=Mock(return_value=source),
                load_content_metadata=lambda **_: storage["content_metadata"],
            )
            manager.optimizer.sharded_state_dict.return_value = {
                0: leaf([group(step=0)]),
                1: leaf([group(expert=True, step=0)]),
            }
            manager.optimizer.load_state_dict = Mock()
            manager.model[0].load_state_dict = Mock()

            def unwrap(value):
                if isinstance(value, LocalValue):
                    return value.data
                if isinstance(value, dict):
                    return {key: unwrap(item) for key, item in value.items()}
                return value

            self.ns["load_dist_checkpointing"] = lambda **kwargs: unwrap(
                kwargs["sharded_state_dict"]
            )
            manager.load_checkpoint(
                directory, with_rng=False, optimizer_sharding_type="fully_reshardable"
            )
            restored = manager.optimizer.load_state_dict.call_args.args[0]
            self.assertEqual(restored[0]["optimizer"]["param_groups"], [group()])
            self.assertEqual(
                restored[1]["optimizer"]["param_groups"], [group(expert=True)]
            )
            self.assertEqual(restored[1]["param_state"][0]["exp_avg"], "native-moment")
            manager.wait_async_saves.assert_called_once()
            # A legacy file cannot be relabeled by choosing the new load option.
            storage["content_metadata"] = {"format": "dp_reshardable"}
            with self.assertRaisesRegex(ValueError, "declared portable"):
                manager.load_checkpoint(
                    directory,
                    with_rng=False,
                    optimizer_sharding_type="fully_reshardable",
                )

    def test_engine_selects_new_format_only_for_reconfiguration(self):
        for name in ("save", "load"):
            method = _source_function(
                "areal/engine/megatron_engine.py", "MegatronEngine", name
            )
            from contextlib import nullcontext

            engine = SimpleNamespace(
                _awex_adapter=None,
                _offload_aware_context=nullcontext,
                checkpointer=Mock(),
            )
            for fmt in ("dcp", "dcp-reconfiguration"):
                method(
                    engine,
                    SimpleNamespace(weight_format=fmt, path="fixture", with_optim=True),
                )
                call = getattr(engine.checkpointer, f"{name}_checkpoint").call_args
                self.assertEqual(call.args, ("fixture",))
                expected = {"with_optimizer": True}
                if fmt == "dcp-reconfiguration":
                    expected.update(
                        with_rng=False, optimizer_sharding_type="fully_reshardable"
                    )
                self.assertEqual(call.kwargs, expected)


class MegatronRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.helper = runtime_fixtures.TrainingLayout(
            backend="megatron", target=(1, 2, 2, 1, 2, 1)
        )

    def test_layout_change_restores_absolute_schedule_into_new_groups(self):
        groups = [{"lr": 99}, {"lr": 99}]
        scheduler = SimpleNamespace(num_steps=999)

        def load(value):
            self.assertEqual(scheduler.num_steps, 0)
            scheduler.num_steps += value["last_epoch"]
            groups[0]["lr"], groups[1]["lr"] = 0.25, 0.5

        scheduler.load_state_dict = load
        engine = SimpleNamespace(
            rank=0,
            world_size=4,
            optimizer=SimpleNamespace(param_groups=groups),
            lr_scheduler=scheduler,
        )
        M.load_runtime_scheduler(
            engine,
            self.helper.mapped(),
            identity=self.helper.identity(self.helper.target),
            reset_steps=True,
        )
        self.assertEqual(scheduler.num_steps, 12)
        self.assertEqual(groups, [{"lr": 0.25}, {"lr": 0.5}])

    def test_new_rng_is_deterministic_and_uses_native_parallel_seeding(self):
        engine = SimpleNamespace(rank=0, world_size=4)
        state = self.helper.mapped()
        identity = self.helper.identity(self.helper.target)
        local = M.select_runtime_state(engine, state, identity=identity)
        self.assertEqual(
            local, M.select_runtime_state(engine, state, identity=identity)
        )
        native_seed = Mock()
        np = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        generator = Mock()
        generator.manual_seed.return_value = generator
        torch = SimpleNamespace(
            Generator=Mock(return_value=generator), set_rng_state=Mock()
        )
        core = SimpleNamespace(
            mpu=SimpleNamespace(get_pipeline_model_parallel_rank=lambda: 1),
            tensor_parallel=SimpleNamespace(
                model_parallel_cuda_manual_seed=native_seed
            ),
        )
        platform = SimpleNamespace(current_platform=SimpleNamespace(device_type="cuda"))
        import random

        before = random.getstate()
        try:
            with patch.dict(
                "sys.modules",
                {
                    "numpy": np,
                    "torch": torch,
                    "megatron.core": core,
                    "areal.infra.platforms": platform,
                },
            ):
                M.install_megatron_rng(engine, local["rng"])
            seed = local["rng"]["root_seed"] + 100
            native_seed.assert_called_once_with(seed)
            generator.manual_seed.assert_called_once_with(seed)
            np.random.seed.assert_called_once_with(seed % 2**32)
        finally:
            random.setstate(before)


class SnapshotFormatTests(unittest.TestCase):
    def test_manifest_format_and_legacy_restore_routing(self):
        engine = Mock()
        engine.save.return_value = engine.load.return_value = None
        engine.wait_for_checkpoint.return_value = None

        def meta(path, weight_format, with_optim):
            return SimpleNamespace(
                path=path, weight_format=weight_format, with_optim=with_optim
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                "sys.modules",
                {"areal.api.io_struct": SimpleNamespace(SaveLoadMeta=meta)},
            ),
        ):
            checkpoint = M.CheckpointParticipant.for_engine(
                engine, format_id="areal-megatron-dcp"
            )
            self.assertEqual(checkpoint.format_id, M.MEGATRON_PORTABLE_FORMAT)
            checkpoint.save(Path(directory))
            checkpoint.load(Path(directory))
            self.assertEqual(
                engine.save.call_args.args[0].weight_format, "dcp-reconfiguration"
            )
            self.assertEqual(
                engine.load.call_args.args[0].weight_format, "dcp-reconfiguration"
            )
            context = M.SnapshotContext(
                "test",
                "runtime",
                "sha256:" + "1" * 64,
                "boundary",
                dict(epoch=0, epoch_step=0, global_step=0, steps_per_epoch=2),
                1,
            )

            def save(path):
                (path / "weights").write_bytes(b"legacy fixture")

            legacy = M.CheckpointParticipant(
                "areal-megatron-dcp", save, lambda: None, lambda _: None
            )
            snapshot = M.ReconfigurationManager(
                checkpoints={"actor": legacy}, statefuls={}
            ).save_snapshot(Path(directory) / "old", context=context)
            request = M.SnapshotRestoreRequest(snapshot, context)
            controller = M.TrainerSnapshotRestore(request)
            trainer = SimpleNamespace(
                train_dataloader=[0, 1],
                _reconfiguration_components=lambda: (
                    {"actor": checkpoint},
                    {"actor_runtime": M.EngineRuntimeState(engine)},
                ),
                rollout=Mock(),
                eval_rollout=None,
                config=SimpleNamespace(
                    gconfig=SimpleNamespace(
                        n_samples=1,
                        reward_normalization=None,
                        drop_incomplete_group=False,
                    )
                ),
                _is_v1_awex_colocate=lambda _: False,
            )
            with patch.object(M, "ReconfigurationManager") as factory:
                controller.restore(
                    trainer,
                    workflow="fixture",
                    workflow_kwargs=None,
                    should_accept_fn=None,
                    max_steps=2,
                )
                reader = factory.call_args.kwargs["checkpoints"]["actor"]
                reader.load(Path(directory))
            self.assertEqual(reader.format_id, "areal-megatron-dcp")
            self.assertEqual(engine.load.call_args.args[0].weight_format, "dcp")
            # The trainer's next capture remains portable after a legacy restore.
            self.assertEqual(checkpoint.format_id, M.MEGATRON_PORTABLE_FORMAT)
            transition = M.ConfigurationTransition(
                "sha256:" + "3" * 64,
                context.configuration_id,
                "sha256:" + "2" * 64,
                (16, 8, 2),
                (16, 8, 2),
                training_layout=dict(
                    source=(4, 1, 1, 1, 1, 1),
                    target=(2, 2, 1, 1, 1, 1),
                    backend="megatron",
                    rng_policy=M.MEGATRON_LAYOUT_RNG_POLICY,
                ),
            )
            with self.assertRaisesRegex(M.SnapshotError, "portable snapshot"):
                M.TrainerSnapshotRestore(
                    M.SnapshotRestoreRequest(snapshot, context, transition)
                )


if __name__ == "__main__":
    unittest.main()
