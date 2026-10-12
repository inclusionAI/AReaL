# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from areal.engine import sglang_mm_executor as patch


@pytest.fixture
def executor_source(monkeypatch):
    # Keep only the call site; exercise it with real asyncio and a thread pool.
    source = (
        "class ProcessorRunner:\n"
        "    def run(self, function, args, processor, kwargs):\n" + patch.OLD
    ).encode()
    expected = source.replace(patch.OLD.encode(), patch.NEW.encode())
    monkeypatch.setattr(patch, "EXPECTED_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(patch, "PATCHED_SHA256", hashlib.sha256(expected).hexdigest())
    return source


@pytest.fixture
def run_processor(executor_source):
    namespace = {}
    exec(
        compile(patch.patched_source(executor_source), "executor.py", "exec"),
        namespace,
    )
    runner = namespace["ProcessorRunner"]()
    with ThreadPoolExecutor(max_workers=1) as executor:

        async def run(function, *args, **kwargs):
            return await asyncio.get_running_loop().run_in_executor(
                executor, runner.run, function, args, {"image_size": 32}, kwargs
            )

        yield run


def test_unknown_executor_is_not_modified(tmp_path, monkeypatch):
    target = tmp_path / "executor.py"
    target.write_bytes(b"# unsupported executor\n")
    before = target.read_bytes()
    monkeypatch.setattr("sys.argv", ["patch", "--target", str(target)])
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        patch.main()
    assert target.read_bytes() == before


def test_executor_patch_is_idempotent(executor_source):
    patched = patch.patched_source(executor_source)
    assert patch.patched_source(patched) == patched


def test_valid_multimodal_call_preserves_arguments_and_result(run_processor):
    def process(value, *, processor, multiplier):
        return value * multiplier, processor["image_size"]

    async def scenario():
        return await run_processor(process, 7, multiplier=3)

    assert asyncio.run(scenario()) == (21, 32)


def test_exhausted_iterator_finishes_request_and_releases_lock(run_processor):
    async def scenario():
        lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()
        loop_errors = []
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

        def process(*, processor):
            return next(iter(()))

        async def request():
            async with lock:
                return await run_processor(process)

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
                await run_processor(lambda *, processor: "next request")
                == "next request"
            )
        finally:
            loop.set_exception_handler(original_handler)

    asyncio.run(scenario())


def test_other_processor_errors_keep_their_type(run_processor):
    def process(*, processor):
        raise RuntimeError("image decoding failed")

    async def scenario():
        with pytest.raises(RuntimeError, match="image decoding failed"):
            await asyncio.wait_for(run_processor(process), timeout=2)

    asyncio.run(scenario())


def test_executor_cli_applies_patch_to_explicit_target(
    executor_source, tmp_path, monkeypatch
):
    target = tmp_path / "executor.py"
    target.write_bytes(executor_source)
    monkeypatch.setattr("sys.argv", ["patch", "--target", str(target)])
    patch.main()
    assert target.read_bytes() == patch.patched_source(executor_source)
    patch.main()
    assert target.read_bytes() == patch.patched_source(executor_source)


@pytest.mark.parametrize("count", [0, 2])
def test_executor_rejects_missing_or_ambiguous_call_site(count, monkeypatch):
    source = (patch.OLD * count).encode()
    monkeypatch.setattr(patch, "EXPECTED_SHA256", hashlib.sha256(source).hexdigest())
    with pytest.raises(RuntimeError, match="exactly one source replacement"):
        patch.patched_source(source)


def test_executor_rejects_unvalidated_patch_output(executor_source, monkeypatch):
    monkeypatch.setattr(patch, "PATCHED_SHA256", "incorrect")
    with pytest.raises(RuntimeError, match="does not match validated source"):
        patch.patched_source(executor_source)
