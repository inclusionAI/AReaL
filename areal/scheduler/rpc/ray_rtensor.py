from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ray
import torch

from areal.scheduler.rpc.rtensor import (
    BaseRTensor,
    BaseTensorShardInfo,
    _find_in_structure,
    _pad_cat_dim0,
)


@dataclass
class RayTensorShardInfo(BaseTensorShardInfo):
    ref: ray.ObjectRef


@dataclass
class RayRTensor(BaseRTensor):
    def to_local(self) -> torch.Tensor:
        if not self.data.is_meta:
            return self.data
        # Fetch all shards first
        tensors = self._fetch()
        self.data = _pad_cat_dim0(tensors)
        return self.data

    def _fetch(self) -> list[torch.Tensor]:
        return ray.get([s.ref for s in self.shards])

    def split(self) -> list[RayRTensor]:
        tensors = RayRTensor.split_tensor(self.data, self)
        return [RayRTensor(shards=[s], data=t) for s, t in zip(self.shards, tensors)]

    @classmethod
    def from_batched(cls, batch_tensor: torch.Tensor, layout: RayRTensor):
        if not batch_tensor.is_cpu and not batch_tensor.is_meta:
            raise ValueError("RTensor shards must be on CPU or meta device")

        tensors = cls.split_tensor(batch_tensor, layout)

        shards = []
        for tensor, shard_info in zip(tensors, layout.shards):
            ref = ray.put(tensor)
            info = RayTensorShardInfo(
                ref=ref,
                size=shard_info.size,
                seqlens=shard_info.seqlens.copy(),
            )
            shards.append(info)

            # Truncate at the maximum sequence length
            # to prevent over-padding
            if tensor.ndim > 1:
                tensor = tensor[:, : max(shard_info.seqlens)]

        return cls(shards=shards, data=batch_tensor.to("meta"))

    @staticmethod
    def extract_layout(obj: Any, layouts: Any):
        layout_rtensor = _find_in_structure(layouts, RayRTensor)
        result_tensor = _find_in_structure(obj, torch.Tensor)

        if layout_rtensor is None and result_tensor is not None:
            if not isinstance(obj, dict):
                raise RuntimeError(
                    "When input does not contain RayRTensor, "
                    "we expect to extract layouts from a dict batch "
                    f"returned by InferenceEngine. Get obj: {obj}, "
                    f"input layouts: {layouts}."
                )
            attn_mask = obj.get("attention_mask", None)
            if attn_mask is None:
                raise RuntimeError("`attention_mask` is not found")

            layout_rtensor = RayRTensor(
                shards=[
                    RayTensorShardInfo(
                        ref=None,  # placeholder to be filled later
                        size=attn_mask.shape[0],
                        seqlens=[int(am.sum()) for am in attn_mask],
                    )
                ],
                data=torch.empty_like(attn_mask, device="meta"),
            )
        return layout_rtensor

    @staticmethod
    def remotize(obj: Any, layout: RayRTensor) -> Any:
        if isinstance(obj, torch.Tensor):
            return RayRTensor.from_batched(obj.detach().cpu(), layout=layout)

        if isinstance(obj, dict):
            return {
                k: RayRTensor.remotize(obj=v, layout=layout) for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [RayRTensor.remotize(obj=item, layout=layout) for item in obj]

        if isinstance(obj, tuple):
            return tuple(RayRTensor.remotize(obj=item, layout=layout) for item in obj)

        return obj

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is torch.cat:
            return RayRTensor.cat(*args, **kwargs)

        raise NotImplementedError(
            f"RayRTensor does not implement torch function {func}"
        )
