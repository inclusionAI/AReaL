# SPDX-License-Identifier: Apache-2.0

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from examples.autopilot.elastic.fixtures.groups import PPOGroup
from examples.autopilot.elastic.fixtures.weight_publication import WeightPublisher
from examples.autopilot.elastic.swe.trajectory import trajectory_payload

from areal.api.cli_args import PPOActorConfig
from areal.autopilot.elastic.runtime.state.checkpoint import (
    WEIGHT_TYING,
    artifact_manifest,
    atomic_json,
    tie_model_embeddings,
    verify_tied_export,
)
from areal.trainer.ppo.actor import PPOActor


@pytest.fixture
def payload():
    return trajectory_payload(
        {
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
            "loss_mask": [[0, 0, 1, 1]],
            "logprobs": [[0.0, 0.0, -0.2, -0.3]],
            "versions": [[-1, -1, 0, 0]],
            "rewards": [0.0],
            "original_rewards": [0.0],
        },
        task_id="issue",
        rollout_id="sample",
        model_revision="pinned",
        policy_version=0,
    )


def test_fixed_baseline_produces_negative_advantages_without_changing_task_reward(
    payload,
):
    """Nonzero PG signal comes from an explicit baseline, not fabricated success."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        reward_bias=-0.5,
        reward_norm=None,
        adv_norm=None,
        kl_ctl=0.0,
        discount=1.0,
        gae_lambda=1.0,
        mask_no_eos_with_zero=False,
        recompute_logprob=False,
        use_decoupled_loss=False,
    )
    actor = PPOActor(config, Mock())
    data = {key: torch.tensor(value) for key, value in payload["trajectory"].items()}
    actual = actor.compute_advantages([data])[0]
    selected = actual["advantages"][actual["loss_mask"].bool()]
    torch.testing.assert_close(
        selected, torch.full_like(selected, -0.5), rtol=0, atol=0
    )
    assert actual["rewards"].tolist() == [0.0]
    assert actual["original_rewards"].tolist() == [0.0]
    assert actual["loss_mask"].tolist() == [[0.0, 1.0, 1.0, 0.0]]


def test_stale_batch_is_rejected_before_any_optimizer_call(payload):
    """This prototype cannot silently treat stale behavior as current-policy data."""
    group = object.__new__(PPOGroup)
    group.step = 1
    group.compatibility = {"model_revision": "pinned"}
    group.actor = Mock()
    with pytest.raises(ValueError, match="current policy"):
        group.train_trajectory(payload)
    group.actor._gateway_post_result.assert_not_called()


@pytest.mark.parametrize("tied", [True, False])
def test_model_config_preserves_embedding_alias_before_optimizer_creation(
    tied, tmp_path
):
    """A loader-created duplicate must not become a second trainable output head."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            tie_word_embeddings=tied,
        )
    )
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    # Reproduce Transformers 5's duplicate-checkpoint materialization behavior.
    head.weight = torch.nn.Parameter(embedding.weight.detach().clone())
    model.all_tied_weights_keys.pop("lm_head.weight", None)
    tie_model_embeddings(model)
    assert (embedding.weight is head.weight) == tied
    if tied:
        assert (
            model.all_tied_weights_keys["lm_head.weight"] == "model.embed_tokens.weight"
        )
        optimizer = torch.optim.AdamW([embedding.weight], lr=0.1, weight_decay=0)
        before = embedding.weight.detach().clone()
        head(torch.ones(1, 8)).sum().backward()
        optimizer.step()
        assert not torch.equal(before, head.weight)
        torch.testing.assert_close(embedding.weight, head.weight, rtol=0, atol=0)
    # Match FSDPEngine's per-key BF16 cast, not just default shared-storage save.
    model.save_pretrained(
        tmp_path,
        state_dict={
            name: value.to(torch.bfloat16) for name, value in model.state_dict().items()
        },
    )
    verify_tied_export(tmp_path)


def test_tied_export_rejects_independently_updated_output_head(tmp_path):
    from safetensors.torch import save_file

    atomic_json(tmp_path / "config.json", {"tie_word_embeddings": True})
    save_file(
        {
            "model.embed_tokens.weight": torch.ones(2, 2),
            "lm_head.weight": torch.zeros(2, 2),
        },
        tmp_path / "model.safetensors",
    )
    with pytest.raises(AssertionError):
        verify_tied_export(tmp_path)


@pytest.fixture
def publication(tmp_path, monkeypatch):
    checkpoint = tmp_path / "generation"
    (checkpoint / "hf").mkdir(parents=True)
    (checkpoint / "hf/model.safetensors").write_bytes(b"immutable-weights")
    atomic_json(
        checkpoint / "manifest.json",
        {"global_step": 1, "compatibility": {"weight_tying": WEIGHT_TYING}},
    )
    atomic_json(
        checkpoint / "serving.json",
        {
            "version": 1,
            "files": artifact_manifest(checkpoint / "hf"),
            "weight_tying": WEIGHT_TYING,
        },
    )
    controller = Mock()
    controller._inf_addrs = ["http://backend"]
    controller._data_proxy_addrs = ["http://proxy"]
    controller.get_version.return_value = 0
    publisher = WeightPublisher(
        SimpleNamespace(controller=controller, owner="incarnation-a", admin_key="test"),
        state=tmp_path / "published.json",
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value.json.return_value = {"success": True}
    client.get.return_value.json.return_value = {"version": 1}
    monkeypatch.setattr(
        "examples.autopilot.elastic.fixtures.weight_publication.httpx.Client",
        lambda **kwargs: client,
    )
    verified = Mock()
    monkeypatch.setattr(publisher, "_verify_parameters", verified)
    return publisher, checkpoint, controller, client, verified


def test_backend_semantic_failure_keeps_admission_closed_and_version_unchanged(
    publication,
):
    """HTTP 200 alone cannot authorize a new policy version."""
    publisher, checkpoint, controller, client, verified = publication
    client.post.return_value.json.return_value = {"success": False}
    with pytest.raises(RuntimeError, match="rejected"):
        publisher.publish(checkpoint)
    assert not publisher.ready and not publisher.state.exists()
    controller.set_version.assert_not_called()
    controller.continue_generation.assert_not_called()
    verified.assert_not_called()
    with pytest.raises(RuntimeError, match="admission"):
        with publisher.episode():
            pytest.fail("A failed publication admitted an episode")


def test_publication_disk_failure_can_retry_same_checkpoint_without_training(
    publication, monkeypatch
):
    """Loading weights before losing the receipt must not require another update."""
    publisher, checkpoint, controller, client, verified = publication

    def no_space(*args):
        raise OSError("ENOSPC")

    monkeypatch.setattr(
        "examples.autopilot.elastic.fixtures.weight_publication.atomic_json", no_space
    )
    with pytest.raises(OSError, match="ENOSPC"):
        publisher.publish(checkpoint)
    assert not publisher.ready
    controller.continue_generation.assert_not_called()
    monkeypatch.setattr(
        "examples.autopilot.elastic.fixtures.weight_publication.atomic_json",
        atomic_json,
    )
    record = publisher.publish(checkpoint)
    assert record["version"] == 1 and publisher.ready
    assert (
        json.loads(publisher.state.read_text())["inference_incarnation"]
        == "incarnation-a"
    )
    assert client.post.call_count == verified.call_count == 2
    controller.resume.assert_called_once()


def test_receipt_does_not_open_admission_when_resume_fails(publication):
    publisher, checkpoint, controller, client, verified = publication
    controller.resume.side_effect = [RuntimeError("resume failed"), None]
    with pytest.raises(RuntimeError, match="resume failed"):
        publisher.publish(checkpoint)
    assert publisher.state.exists() and not publisher.ready
    with pytest.raises(RuntimeError, match="admission"):
        with publisher.episode():
            pytest.fail("An incomplete publication admitted a rollout")
    assert publisher.publish(checkpoint)["version"] == 1 and publisher.ready


def test_corrupt_checkpoint_export_is_rejected_before_reload(publication):
    """The checkpoint receipt binds publication to exact immutable artifacts."""
    publisher, checkpoint, controller, client, verified = publication
    (checkpoint / "hf/model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="artifact"):
        publisher.publish(checkpoint)
    client.post.assert_not_called()


def test_legacy_unverified_embedding_export_is_rejected_before_reload(publication):
    publisher, checkpoint, controller, client, verified = publication
    atomic_json(checkpoint / "manifest.json", {"global_step": 1})
    with pytest.raises(ValueError, match="weight tying"):
        publisher.publish(checkpoint)
    client.post.assert_not_called()
    assert not publisher.ready


def test_partial_proxy_version_failure_does_not_open_admission(publication):
    """All data proxies must confirm the version before episodes resume."""
    publisher, checkpoint, controller, client, verified = publication
    client.get.return_value.json.return_value = {"version": 0}
    with pytest.raises(RuntimeError, match="proxy version"):
        publisher.publish(checkpoint)
    assert not publisher.ready
    controller.resume.assert_not_called()


def test_new_inference_incarnation_reloads_even_with_an_existing_receipt(
    publication, monkeypatch
):
    """A receipt left by a dead backend does not prove new processes loaded it."""
    publisher, checkpoint, controller, client, verified = publication
    publisher.publish(checkpoint)
    publisher.service.owner = "incarnation-b"
    replacement = WeightPublisher(publisher.service, state=publisher.state)
    monkeypatch.setattr(replacement, "_verify_parameters", verified)
    replacement.initialize(checkpoint)
    assert replacement.ready
    assert client.post.call_count == verified.call_count == 2
    assert json.loads(replacement.state.read_text())["inference_incarnation"] == (
        "incarnation-b"
    )


def test_publication_waits_for_the_whole_episode_before_reloading(publication):
    """In-flight tool calls must not receive a new version label midway through."""
    publisher, checkpoint, controller, client, verified = publication
    publisher.initialize()
    attempted = threading.Event()

    def publish():
        attempted.set()
        return publisher.publish(checkpoint)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with publisher.episode() as version:
            assert version == 0
            future = pool.submit(publish)
            assert attempted.wait(timeout=5)
            assert not future.done()
            client.post.assert_not_called()
        assert future.result(timeout=5)["version"] == 1


def test_staged_publication_verifies_weights_without_opening_admission(publication):
    publisher, checkpoint, controller, client, verified = publication
    marker = Mock()
    publisher.service.mark_weights_verified = marker
    receipt = publisher.publish(checkpoint, admit=False)
    assert receipt["version"] == 1 and not publisher.ready
    verified.assert_called_once()
    marker.assert_called_once_with(1)
    controller.continue_generation.assert_not_called()
    controller.resume.assert_not_called()


def test_staged_initial_model_keeps_generation_paused(publication):
    publisher, checkpoint, controller, client, verified = publication
    publisher.initialize(admit=False)
    assert not publisher.ready
    controller.pause.assert_called_once()
    controller.pause_generation.assert_called_once()
    client.post.assert_not_called()
