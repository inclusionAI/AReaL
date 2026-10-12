# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest

from areal.engine import sglang_mm_executor as patch

FIXTURE = Path(__file__).parent / "fixtures" / "sglang_mm_executor_119b5ffe.txt"


def make_executor():
    namespace = {}
    exec(
        compile(patch.patched_source(FIXTURE.read_bytes()), str(FIXTURE), "exec"),
        namespace,
    )
    return namespace["MultimodalProcessorExecutor"]({"image_size": 32}, max_workers=2)


def test_unknown_executor_is_not_modified(tmp_path, monkeypatch):
    target = tmp_path / "executor.py"
    target.write_bytes(b"# unsupported executor\n")
    before = target.read_bytes()
    monkeypatch.setattr("sys.argv", ["patch", "--target", str(target)])
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        patch.main()
    assert target.read_bytes() == before


def test_pinned_executor_patch_is_idempotent():
    patched = patch.patched_source(FIXTURE.read_bytes())
    assert patch.patched_source(patched) == patched


def test_valid_multimodal_call_preserves_arguments_and_result():
    executor = make_executor()

    def process(value, *, processor, multiplier):
        return value * multiplier, processor["image_size"]

    async def scenario():
        return await executor.run(process, 7, multiplier=3)

    try:
        assert asyncio.run(scenario()) == (21, 32)
    finally:
        executor.shutdown()


def test_exhausted_iterator_finishes_request_and_releases_lock():
    async def scenario():
        executor = make_executor()
        lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()
        loop_errors = []
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

        def process(*, processor):
            return next(iter(()))

        async def request():
            async with lock:
                return await executor.run(process)

        try:
            with pytest.raises(
                ValueError, match="exhausted an input iterator"
            ) as caught:
                await asyncio.wait_for(request(), timeout=2)
            assert isinstance(caught.value.__cause__, StopIteration)
            assert not lock.locked()
            assert not loop_errors
            # A rejected request must not poison the processor thread for later calls.
            assert (
                await executor.run(lambda *, processor: "next request")
                == "next request"
            )
        finally:
            loop.set_exception_handler(original_handler)
            executor.shutdown()

    asyncio.run(scenario())


def test_other_processor_errors_keep_their_type():
    executor = make_executor()

    def process(*, processor):
        raise RuntimeError("image decoding failed")

    async def scenario():
        with pytest.raises(RuntimeError, match="image decoding failed"):
            await asyncio.wait_for(executor.run(process), timeout=2)

    try:
        asyncio.run(scenario())
    finally:
        executor.shutdown()


def test_rl_recipe_applies_executor_patch_before_server_start():
    import yaml

    config = yaml.safe_load(
        (
            Path(__file__).parents[1] / "examples/swe/qwen38_flash_next/swe_mm_rl.yaml"
        ).read_text()
    )
    commands = config["rollout"]["scheduling_spec"][0]["additional_bash_cmds"]
    assert "python3 -m areal.engine.sglang_mm_executor" in commands
