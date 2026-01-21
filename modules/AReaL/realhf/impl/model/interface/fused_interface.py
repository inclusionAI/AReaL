# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import torch

import realhf.api.core.model_api as model_api
import realhf.base.logging as logging
from realhf.api.core.config import ModelInterfaceAbstraction
from realhf.api.core.data_api import RL_TASKS, MicroBatchSpec, SequenceSample
from realhf.base.datapack import flat2d
from realhf.impl.model.nn.real_llm_api import ReaLModel

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FusedThreadingForwardInterface(model_api.ModelInterface):

    def __init__(self, interfaces: Dict[str, ModelInterfaceAbstraction]):
        self.interfaces = {
            key: model_api.make_interface(interface)
            for key, interface in interfaces.items()
        }

    def run_interface(
        self,
        interface_name: str,
        model,
        data,
        mb_spec,
    ) -> SequenceSample | None:
        tik = time.perf_counter()
        res = self.interfaces[interface_name].inference(model, data, mb_spec)
        t = time.perf_counter() - tik
        logger.info(f"Interface {interface_name} cost {t} s")
        return res

    def inference(
        self,
        model: model_api.Model,
        data: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample | None:
        with ThreadPoolExecutor(max_workers=len(self.interfaces)) as executor:
            tasks = []
            for interface_name in self.interfaces:
                task = executor.submit(
                    self.run_interface, interface_name, model, data, mb_spec
                )
                tasks.append(task)

            final_result = None
            for task in as_completed(tasks):
                res = task.result()
                if res is None:
                    continue
                if final_result is None:
                    final_result = res
                else:
                    final_result.update_(res)

        return final_result


model_api.register_interface("fused-threading", FusedThreadingForwardInterface)
