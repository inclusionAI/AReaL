# SPDX-License-Identifier: Apache-2.0

import json
import stat
from pathlib import Path

import pytest

from areal.utils import fs
from areal.v2.cli.state import atomic_write_json as cli_atomic_write_json


def test_atomic_json_cli_import_preserves_existing_state_format(tmp_path):
    """The existing CLI import keeps its JSON and fallback serialization contract."""
    target = tmp_path / "nested" / "state.json"
    data = {"model_path": Path("model"), "label": "数学"}
    cli_atomic_write_json(target, data)
    assert target.read_text() == json.dumps(data, indent=2, default=str) + "\n"
    assert cli_atomic_write_json is fs.atomic_write_json
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.parametrize("value", [float("nan"), object()])
def test_atomic_json_strict_serialization_failure_preserves_previous_state(
    tmp_path, value
):
    """Invalid data cannot replace a complete state file or leave temporary files."""
    target = tmp_path / "state.json"
    target.write_text('{"status": "running"}\n')
    before = target.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        fs.atomic_write_json(target, {"value": value}, default=None, allow_nan=False)
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("error", [OSError("disk failure"), KeyboardInterrupt()])
def test_atomic_json_replace_failure_preserves_previous_state(
    tmp_path, monkeypatch, error
):
    """A failed or interrupted publication cleans up its temporary state file."""
    target = tmp_path / "state.json"
    target.write_text('{"status": "running"}\n')
    before = target.read_bytes()

    def fail_replace(*args):
        raise error

    monkeypatch.setattr(fs.os, "replace", fail_replace)
    with pytest.raises(type(error)):
        fs.atomic_write_json(target, {"status": "succeeded"})
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_json_success_replaces_complete_state(tmp_path):
    """Readers observe the complete new state after publication."""
    target = tmp_path / "state.json"
    fs.atomic_write_json(target, {"status": "running"})
    fs.atomic_write_json(target, {"status": "succeeded", "selected": 64})
    assert json.loads(target.read_text()) == {"status": "succeeded", "selected": 64}
    assert list(tmp_path.iterdir()) == [target]
