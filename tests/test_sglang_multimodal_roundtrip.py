# SPDX-License-Identifier: Apache-2.0
"""CPU preprocessing integration with an explicitly supplied local Qwen4 model.

Set AREAL_TEST_QWEN_MODEL_PATH in the validated SGLang rollout environment. No
model weights, GPU allocation, server startup, or network downloads are needed.
"""

import asyncio
import base64
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from areal.api import ModelRequest
from areal.engine.sglang_remote import SGLangBackend


@contextmanager
def preprocessing_server_args(model_path):
    from sglang.srt.runtime_context import get_context

    context = get_context()
    if hasattr(context, "override_server_args"):
        with context.override_server_args(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            mm_feature_transport="cpu",
            mm_process_config={},
            mm_processor_worker_num=1,
            mm_io_worker_num=1,
            mm_preprocess_cache_size_mb=0,
        ) as args:
            yield args
    else:
        from sglang.srt import server_args

        previous = server_args._global_server_args
        args = server_args.ServerArgs(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            mm_process_config={},
            disable_radix_cache=True,
        )
        server_args.set_global_server_args_for_tokenizer(args)
        try:
            yield args
        finally:
            server_args.set_global_server_args_for_tokenizer(previous)


@pytest.fixture(scope="module")
def qwen_roundtrip_runtime():
    model_path = os.environ.get("AREAL_TEST_QWEN_MODEL_PATH")
    if not model_path:
        pytest.skip("Set AREAL_TEST_QWEN_MODEL_PATH to a local Qwen model")

    model_type = json.loads((Path(model_path) / "config.json").read_text())[
        "model_type"
    ]
    if model_type in ("qwen3_5", "qwen3_5_moe"):
        from areal.engine import sglang_qwen35_vl

        # Install before importing either SGLang processor class so the base
        # class's actual device dispatch is covered too (inside a test runtime).
        sglang_qwen35_vl.main()

    qwen_vl = pytest.importorskip("sglang.srt.multimodal.processors.qwen_vl")
    from sglang.srt.environ import envs
    from transformers import AutoConfig, AutoProcessor

    # Isolate the patched class without changing the installed module or file.
    source = Path(qwen_vl.__file__).read_bytes()
    if model_type == "qwen4_exp":
        from areal.engine import sglang_qwen4_vl

        patched = sglang_qwen4_vl.patched_source(source)
    else:
        patched = sglang_qwen35_vl.patched_source(source)
    namespace = qwen_vl.__dict__.copy()
    exec(compile(patched, qwen_vl.__file__, "exec"), namespace)
    assert envs.SGLANG_MM_AVOID_RETOKENIZE.get()
    hf = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=True)
    # SGLang used to mutate size in place. A shared processor would mutate the
    # reference as well and conceal the mismatch we are trying to detect.
    hf_infer = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    with preprocessing_server_args(model_path) as args:
        processor = namespace["QwenVLImageProcessor"](
            config, args, hf_infer, transport_mode="default", skip_mm_pool=True
        )
        try:
            assert hf.image_processor.size == hf_infer.image_processor.size
            yield hf, processor
        finally:
            if hasattr(processor, "shutdown"):
                processor.shutdown()
            else:
                for name in ("io_executor", "cpu_executor", "load_mm_data_executor"):
                    getattr(processor, name).shutdown(wait=True)


def patterned_image(width: int, height: int) -> Image.Image:
    y, x = np.indices((height, width))
    pixels = np.stack(
        ((x * 17 + y * 13) % 256, (x * 3 + y * 29) % 256, (x + y * 5) % 256),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("shapes", "generated_tail"),
    [
        ([(320, 256)], False),
        ([(511, 257)], False),
        ([(257, 511)], False),
        ([(1001, 777)], False),
        ([(900, 1300)], False),
        ([(84, 56)], False),
        ([(4097, 4097)], False),
        ([(511, 257), (1001, 777)], False),
        ([(511, 257)], True),
    ],
)
def test_qwen_image_compaction_preserves_processor_tokens_grids_and_pixels(
    qwen_roundtrip_runtime, shapes, generated_tail, caplog
):
    hf, processor = qwen_roundtrip_runtime
    images = [patterned_image(*shape) for shape in shapes]
    text = (
        "<|im_start|>user\n"
        + "\n".join("<|vision_start|><|image_pad|><|vision_end|> image" for _ in images)
        + "\n<|im_end|>\n<|im_start|>assistant\n"
    )
    reference = hf(text=text, images=images, return_tensors="pt")
    original = reference["input_ids"][0].tolist()
    if generated_tail:
        # Separate encodings need not survive a decode/encode round trip.
        original += hf.tokenizer.encode("hello", add_special_tokens=False)
        original += hf.tokenizer.encode("world", add_special_tokens=False)

    image_data = []
    for image in images:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        image_data.append(
            "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        )
    req = ModelRequest(input_ids=original.copy(), image_data=image_data, processor=hf)
    wire = SGLangBackend().build_generation_request(req, False, 0)
    request_obj = SimpleNamespace(
        video_data=None, audio_data=None, rid="roundtrip", mm_sampling_kwargs={}
    )
    actual = asyncio.run(
        asyncio.wait_for(
            processor.process_mm_data_async(
                image_data=image_data,
                input_text=wire.payload["input_ids"],
                request_obj=request_obj,
            ),
            timeout=120,
        ),
    )

    assert actual.input_ids == original
    assert req.input_ids == original
    assert not any(
        "falling back to decode+retokenize" in r.message for r in caplog.records
    )
    grid = torch.cat([item.image_grid_thw for item in actual.mm_items])
    pixels = torch.cat([item.feature for item in actual.mm_items])
    torch.testing.assert_close(grid, reference["image_grid_thw"], rtol=0, atol=0)
    torch.testing.assert_close(pixels, reference["pixel_values"], rtol=0, atol=0)

    # Hash/pad IDs are expected only in image spans, after processor expansion.
    from sglang.srt.managers.mm_utils import (
        MultiModalityDataPaddingPatternMultimodalTokens,
    )
    from sglang.srt.managers.schedule_batch import MultimodalInputs

    for item in actual.mm_items:
        item.set_pad_value()
    padded_ids = MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
        actual.input_ids.copy(), MultimodalInputs(mm_items=actual.mm_items)
    )
    ids = torch.tensor(original, dtype=torch.long, device="cpu")
    padded = torch.tensor(padded_ids, dtype=torch.long, device="cpu")
    image_mask = ids == hf.image_token_id
    assert ids.shape == padded.shape
    torch.testing.assert_close(padded[~image_mask], ids[~image_mask], rtol=0, atol=0)
    assert actual.mrope_positions.shape == torch.Size((3, len(original)))
