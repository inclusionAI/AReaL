# SPDX-License-Identifier: Apache-2.0
"""Build one executable containing only the public runtime modules it needs.

This supports legacy deployments that materialize a single runner executable.
Dependencies can be preinstalled in the image, or explicitly bootstrapped in a
task-local venv for CPU integration checks. No credentials or oracle are bundled.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import tarfile
from pathlib import Path

FILES = [
    "examples/task_workflows/__init__.py",
    "examples/task_workflows/types.py",
    "examples/task_workflows/trace.py",
    "examples/task_workflows/benchmarks/__init__.py",
    "examples/task_workflows/benchmarks/dapo_math.py",
    "examples/task_workflows/environments/__init__.py",
    "examples/task_workflows/environments/process.py",
    "examples/task_workflows/environments/python.py",
    "examples/task_workflows/harnesses/__init__.py",
    "examples/task_workflows/harnesses/hermes_runtime.py",
    "examples/task_workflows/arena/__init__.py",
    "examples/task_workflows/arena/python_tool.py",
    "examples/task_workflows/arena/runner.py",
    "examples/task_workflows/arena/reward.py",
]


def build(
    output: Path,
    *,
    mode: str,
    bootstrap: bool = False,
    recipe: Path | None = None,
    vendor: Path | None = None,
) -> None:
    root = Path(__file__).resolve().parents[3]
    files = list(FILES)
    if mode in ("smoke", "bundle"):
        if recipe is None:
            raise ValueError("The smoke package needs a non-oracle recipe")
        files.append("examples/task_workflows/arena/smoke.py")
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w") as archive:
        for name in files:
            data = (root / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if recipe is not None:
            # Limit the smoke recipe to a prompt digest and authored tool code.
            value = json.loads(recipe.read_text())
            if set(value) != {"prompt_sha256", "code"}:
                raise ValueError("A smoke recipe must not contain oracle fields")
            payload = json.dumps(value).encode()
            info = tarfile.TarInfo("smoke-recipe.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        target = None
        if vendor is not None:
            target = json.loads((vendor / ".target.json").read_text())
            for source in sorted(vendor.rglob("*")):
                if source.is_symlink():
                    raise ValueError("Vendor tree must contain regular files")
                if source.is_file():
                    data = source.read_bytes()
                    info = tarfile.TarInfo(
                        "site-packages/" + source.relative_to(vendor).as_posix()
                    )
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
    payload = base64.b64encode(gzip.compress(memory.getvalue(), mtime=0)).decode()
    dependencies = (
        ["math-verify==0.8.0"] if mode == "reward" else ["hermes-agent==0.19.0"]
    )
    required = (
        {"math-verify": "0.8.0"} if mode == "reward" else {"hermes-agent": "0.19.0"}
    )
    code = f"""import base64,importlib.metadata,io,os,platform,shutil,subprocess,sys,tarfile,tempfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix="arena-hermes-"))
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode="r:gz") as archive:
    archive.extractall(root,filter="data")
python=sys.executable
target={target!r}
if target is not None:
    if list(sys.version_info[:2])!=target["python"] or platform.machine()!=target["machine"] or platform.system()!=target["system"]:
        raise RuntimeError("Vendor dependencies do not match this Python/platform")
    sys.path.insert(0,str(root/"site-packages"))
required={required!r}
missing=[]
for name,version in required.items():
    try:
        installed=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        installed=None
    if installed!=version:
        missing.append(name)
if missing and {bootstrap!r}:
    venv=root/"venv"
    if shutil.which("uv"):
        subprocess.run(["uv","venv","--python",python,str(venv)],check=True)
        python=str(venv/"bin/python")
        subprocess.run(["uv","pip","install","--python",python,*{dependencies!r}],check=True)
    else:
        subprocess.run([python,"-m","venv",str(venv)],check=True)
        python=str(venv/"bin/python")
        subprocess.run([python,"-m","pip","install",*{dependencies!r}],check=True)
elif missing:
    raise RuntimeError("Image needs exact dependencies: "+str(required))
os.environ["PYTHONPATH"]=str(root)+(os.pathsep+str(root/"site-packages") if target is not None else "")
os.environ["HERMES_HOME"]=str(root/"hermes-home")
os.environ["AUTOPILOT_SMOKE_RECIPE"]=str(root/"smoke-recipe.json")
module={mode!r}
if module=="bundle":
    module=os.environ.get("AUTOPILOT_ENTRYPOINT")
    if module not in ["runner","reward","smoke"]:
        raise RuntimeError("Bundle requires an explicit immutable entrypoint")
os.execv(python,[python,"-m","examples.task_workflows.arena."+module])
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "#!/bin/sh\nset -eu\nexec python3 - <<'ARENA_BOOTSTRAP_PY'\n"
        + code
        + "\nARENA_BOOTSTRAP_PY\n"
    )
    output.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["runner", "reward", "smoke", "bundle"], required=True
    )
    parser.add_argument("--bootstrap-deps", action="store_true")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument(
        "--vendor", type=Path, help="Offline site-packages with .target.json"
    )
    args = parser.parse_args()
    build(
        args.output,
        mode=args.mode,
        bootstrap=args.bootstrap_deps,
        recipe=args.recipe,
        vendor=args.vendor,
    )


if __name__ == "__main__":
    main()
