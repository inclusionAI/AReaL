import os
import sys

from sh import Command


def test_grpo_ray(tmp_path: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = (
        Command("python")
        .bake(m="areal.launcher.ray")
        .bake(os.path.join(base_dir, "entrypoint.py"))
    )

    cmd(
        f"cluster.fileroot={tmp_path}",
        config=os.path.join(base_dir, f"config.yaml"),
        _err=sys.stderr,
        _out=sys.stdout,
        _env=os.environ,
        _ok_code=1,  # AReaL exits with code 1 even when successful.
    )

    success_file = os.path.join(tmp_path, "success.txt")
    assert os.path.exists(success_file), "Training did not complete successfully"

    with open(success_file) as f:
        content = f.read()
        assert "Training completed successfully" in content
