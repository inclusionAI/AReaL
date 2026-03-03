# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.result

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class ScaffoldingOutput:
    text: str
    token_ids: list[int]
    data: Any = None


class ScaffoldingResult:
    def __init__(self):
        super().__init__()
        self.aqueue = asyncio.Queue()
        self.outputs = []
        # only support one output for now, so use an empty obj to init
        self.outputs.append(ScaffoldingOutput("", []))
        self._done = False
        self.task_collections = None

    def set_output(self, output: ScaffoldingOutput | Any):
        if isinstance(output, ScaffoldingOutput):
            self.set_output_streaming(output)
        # terminate
        self.set_output_streaming(None)

    def set_output_streaming(self, output: ScaffoldingOutput | Any):
        self.aqueue.put_nowait(output)

    def set_task_collections(self, task_collections: Mapping[str, Any]):
        self.task_collections = task_collections

    async def _aresult_step(self):
        obj = await self.aqueue.get()
        if obj is None:
            self._done = True
        else:  # obj is ScaffoldingOutput
            self.outputs[0] = obj

    def result(self, timeout: float | None = None) -> "ScaffoldingResult":
        if not self._done:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self.aresult(), loop).result()
        return self

    async def aresult(self) -> "ScaffoldingResult":
        while not self._done:
            await self._aresult_step()
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        await self._aresult_step()
        return self
