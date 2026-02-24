"""
Vision Data Parallel utilities for AReaL.

Strategy: Distribute whole images across Ulysses SP ranks, not patches
within images. This avoids breaking cu_seqlens semantics while parallelizing
ViT computation.

When using Ulysses Sequence Parallelism (sp_size > 1), the attention layers
split across SP ranks via all-to-all, but the VisionTransformer (ViT) still
processes ALL images on every rank. This wastes memory proportional to
total_images.

Vision DP fixes this by distributing whole images across SP ranks:
- Before: Each of N SP ranks processes ALL images -> ViT memory = O(total_images)
- After: Each rank processes total_images/N images -> ViT memory = O(total_images/N)

Key design choices:
- Image-level distribution (not patch-level): avoids breaking ViT's internal
  cu_seqlens tracking
- Contiguous assignment: rank 0 gets images [0,1,...], rank 1 gets next chunk, etc.
  No reordering needed after all-gather.
- Gradient sync in backward: all_reduce(SUM) across SP ranks before slicing to
  recover the complete gradient for each image. Without this, gradients from
  vision tokens in other ranks' sequence shards would be lost.
- No additional gradient scaling needed: the all_reduce aggregates partial
  sequence gradients, making each rank's ViT backward equivalent to the non-DP
  baseline. FSDP's dp_sp reduce-scatter then handles DP averaging as usual.

Adapted from verl PR #5230 (https://github.com/verl-project/verl/pull/5230).
"""

import torch
import torch.distributed as dist
from torch.autograd import Function

from areal.models.fsdp.ulysses import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
)
from areal.utils import logging

logger = logging.getLogger("VisionDP")


def get_image_patch_counts(grid_thw: torch.Tensor) -> list[int]:
    """
    Compute number of patches per image from grid_thw.

    Args:
        grid_thw: [num_images, 3] tensor with (t, h, w) per image

    Returns:
        List of patch counts per image: [t*h*w for each image]
    """
    if grid_thw.numel() == 0:
        return []
    return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()


def get_image_embedding_counts(
    grid_thw: torch.Tensor, spatial_merge_size: int = 1
) -> list[int]:
    """
    Compute number of embeddings per image after spatial merging.

    VLMs like Qwen2-VL use a merger module that combines multiple patches into
    one embedding. The merger reduces spatial dimensions by spatial_merge_size
    in both h and w.

    Args:
        grid_thw: [num_images, 3] tensor with (t, h, w) per image
        spatial_merge_size: Merger's spatial reduction factor (default 1 = no merging)

    Returns:
        List of embedding counts per image: [t * (h/merge) * (w/merge) for each image]
    """
    if grid_thw.numel() == 0:
        return []

    if spatial_merge_size == 1:
        return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

    t = grid_thw[:, 0]
    h = grid_thw[:, 1] // spatial_merge_size
    w = grid_thw[:, 2] // spatial_merge_size
    return (t * h * w).tolist()


def assign_images_to_dp_ranks(
    patch_counts: list[int],
    dp_size: int,
) -> tuple[list[list[int]], list[int]]:
    """
    Assign whole images to DP ranks using load-balanced contiguous distribution.

    The algorithm uses greedy contiguous bin-packing:
    - Images are assigned in order (contiguous) to preserve ordering after gather
    - Split points are chosen to balance total patch load across ranks
    - Each rank gets at least one image when num_images >= dp_size

    Args:
        patch_counts: Number of patches per image
        dp_size: Number of DP ranks

    Returns:
        image_assignments: List of image indices per rank
        rank_patch_counts: Total patches per rank
    """
    num_images = len(patch_counts)
    if num_images == 0:
        return [[] for _ in range(dp_size)], [0] * dp_size

    image_assignments = [[] for _ in range(dp_size)]
    rank_loads = [0] * dp_size

    remaining_patches = sum(patch_counts)
    img_idx = 0
    for rank in range(dp_size):
        remaining_ranks = dp_size - rank
        remaining_images = num_images - img_idx

        if remaining_images <= 0:
            break

        # Dynamic target: distribute remaining patches evenly among remaining ranks
        target = remaining_patches / remaining_ranks

        # Must leave at least 1 image for each remaining rank
        max_images = remaining_images - (remaining_ranks - 1)

        # Greedily add images until we reach the target load or hit the max
        count = 0
        while img_idx < num_images and count < max_images:
            image_assignments[rank].append(img_idx)
            rank_loads[rank] += patch_counts[img_idx]
            img_idx += 1
            count += 1

            # Stop early once we've reached the target (always take at least 1)
            if rank_loads[rank] >= target:
                break

        remaining_patches -= rank_loads[rank]

    return image_assignments, rank_loads


def prepare_local_vision_inputs(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    image_assignments: list[list[int]],
    dp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Extract pixel values and grid_thw for this DP rank's assigned images.

    Args:
        pixel_values: [total_patches, patch_dim] all patches flattened
        grid_thw: [num_images, 3] all image grids
        image_assignments: image indices per rank from assign_images_to_dp_ranks
        dp_rank: current DP rank

    Returns:
        local_pixel_values: patches for this rank's images
        local_grid_thw: grid dimensions for this rank's images
        local_image_indices: which images this rank processes
    """
    local_indices = image_assignments[dp_rank]

    if len(local_indices) == 0:
        return (
            torch.empty(
                (0, pixel_values.shape[1]) if pixel_values.dim() > 1 else (0,),
                dtype=pixel_values.dtype,
                device=pixel_values.device,
            ),
            torch.empty((0, 3), dtype=grid_thw.dtype, device=grid_thw.device),
            [],
        )

    # local_indices are contiguous (e.g. [2, 3, 4]), so use tensor slicing
    first_img_idx = local_indices[0]
    last_img_idx = local_indices[-1]

    # Compute patch offsets using cumsum
    patch_counts = get_image_patch_counts(grid_thw)
    patch_counts_tensor = torch.tensor(
        patch_counts, device=grid_thw.device, dtype=torch.long
    )
    offsets = torch.cat(
        (
            torch.tensor([0], device=grid_thw.device, dtype=torch.long),
            torch.cumsum(patch_counts_tensor, dim=0),
        )
    )

    start_patch = offsets[first_img_idx].item()
    end_patch = offsets[last_img_idx + 1].item()

    local_pixel_values = pixel_values[start_patch:end_patch]
    local_grid_thw = grid_thw[first_img_idx : last_img_idx + 1]

    expected_patches = end_patch - start_patch
    assert local_pixel_values.shape[0] == expected_patches, (
        f"[Vision DP] Local patch count mismatch: "
        f"extracted={local_pixel_values.shape[0]}, expected={expected_patches}, "
        f"local_indices={local_indices}"
    )

    return local_pixel_values, local_grid_thw, local_indices


class GatherVisionEmbeddings(Function):
    """
    All-gather vision embeddings with gradient support.

    Since images are assigned contiguously (rank 0 gets [0,1], rank 1 gets [2,3], etc.),
    we can simply concat gathered results without reordering.

    Forward: all_gather + remove padding + concat
    Backward: all_reduce(SUM) to aggregate gradients from all sequence shards,
              then slice to extract this rank's image gradients
    """

    @staticmethod
    def forward(
        ctx,
        local_embeddings: torch.Tensor,
        dp_group,
        all_counts: list[int],
    ) -> torch.Tensor:
        dp_size = dist.get_world_size(dp_group)
        dp_rank = dist.get_rank(dp_group)
        ctx.dp_size = dp_size
        ctx.dp_group = dp_group
        ctx.all_counts = all_counts
        ctx.dp_rank = dp_rank

        if dp_size == 1:
            return local_embeddings

        max_count = max(all_counts) if all_counts else 0

        if max_count == 0:
            return local_embeddings

        hidden_size = local_embeddings.shape[1] if local_embeddings.dim() > 1 else 1
        ctx.hidden_size = hidden_size

        # 2. Pad to same length for all_gather
        if local_embeddings.shape[0] < max_count:
            pad_size = max_count - local_embeddings.shape[0]
            padding = torch.zeros(
                (pad_size, hidden_size),
                dtype=local_embeddings.dtype,
                device=local_embeddings.device,
            )
            local_padded = torch.cat([local_embeddings, padding], dim=0)
        else:
            local_padded = local_embeddings

        # 3. All-gather
        gathered = [torch.empty_like(local_padded) for _ in range(dp_size)]
        dist.all_gather(gathered, local_padded, group=dp_group)

        # 4. Remove padding and concat (no reordering needed - contiguous assignment)
        result_chunks = [gathered[r][: all_counts[r]] for r in range(dp_size)]
        result = torch.cat(result_chunks, dim=0)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        dp_size = ctx.dp_size

        if dp_size == 1:
            return grad_output, None, None

        all_counts = ctx.all_counts
        dp_rank = ctx.dp_rank
        dp_group = ctx.dp_group

        # Aggregate gradient contributions from all SP ranks.
        # Each rank only has non-zero grad for vision tokens in its own
        # sequence shard. Summing across ranks recovers the complete
        # gradient for every image before we slice by image assignment.
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=dp_group)

        # Extract gradients for this rank's images (contiguous slice)
        start = sum(all_counts[:dp_rank])
        end = start + all_counts[dp_rank]
        local_grad = grad_output[start:end]

        return local_grad, None, None


def gather_vision_embeddings(
    local_embeddings: torch.Tensor,
    dp_group,
    all_counts: list[int],
) -> torch.Tensor:
    """
    All-gather vision embeddings from all DP ranks.

    Args:
        local_embeddings: [local_patches, hidden_size] this rank's embeddings
        dp_group: Process group for the all-gather (Ulysses SP group)
        all_counts: Pre-computed embedding counts per rank (avoids an all_gather)

    Returns:
        all_embeddings: [total_patches, hidden_size] in original image order
    """
    if dp_group is None or dist.get_world_size(dp_group) == 1:
        return local_embeddings

    return GatherVisionEmbeddings.apply(local_embeddings, dp_group, all_counts)


def create_dp_vision_forward(original_forward):
    """
    Wrap VisionTransformer.forward for Vision DP (Data Parallel across SP ranks).

    This is a model-agnostic wrapper that works with any VisionTransformer
    that has a forward(self, hidden_states, grid_thw, **kwargs) -> Tensor signature.
    Tested with Qwen2-VL, Qwen2.5-VL, and Qwen3-VL VisionTransformers.

    Uses Ulysses SP globals for group/size/rank at runtime (set by _ensure_ready()
    before each forward pass).

    Args:
        original_forward: The original forward method to wrap

    Returns:
        Wrapped forward method with Vision DP support
    """

    def dp_vision_forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        sp_group = get_ulysses_sequence_parallel_group()
        sp_size = get_ulysses_sequence_parallel_world_size(sp_group)
        sp_rank = get_ulysses_sequence_parallel_rank(sp_group)

        if sp_size <= 1:
            return original_forward(self, hidden_states, grid_thw, **kwargs)

        # Move grid_thw to CPU once to avoid repeated GPU→CPU syncs in
        # metadata helpers (grid_thw is a tiny [num_images, 3] tensor).
        grid_thw_cpu = grid_thw.cpu()

        # Step 1: Get image assignment based on patch counts
        patch_counts = get_image_patch_counts(grid_thw_cpu)
        total_patches = sum(patch_counts)

        assert hidden_states.shape[0] == total_patches, (
            f"[Vision DP] Input patch count mismatch: "
            f"hidden_states.shape[0]={hidden_states.shape[0]}, "
            f"sum(grid_thw products)={total_patches}, "
            f"grid_thw.shape={grid_thw.shape}"
        )

        # Get spatial_merge_size from merger (VLMs like Qwen use merger to reduce embeddings)
        spatial_merge_size = 1
        if hasattr(self, "merger") and hasattr(self.merger, "spatial_merge_size"):
            spatial_merge_size = self.merger.spatial_merge_size
        elif hasattr(self, "spatial_merge_size"):
            spatial_merge_size = self.spatial_merge_size

        # Calculate embedding counts (after merger) for gather verification
        embedding_counts = get_image_embedding_counts(grid_thw_cpu, spatial_merge_size)
        total_embeddings = sum(embedding_counts)

        image_assignments, _ = assign_images_to_dp_ranks(patch_counts, sp_size)

        # Step 2: Extract local inputs
        local_pixels, local_grid_thw, local_indices = prepare_local_vision_inputs(
            hidden_states, grid_thw, image_assignments, sp_rank
        )

        # Step 3: Process local images
        if local_pixels.shape[0] > 0:
            local_embeddings = original_forward(
                self, local_pixels, local_grid_thw, **kwargs
            )
        else:
            # This rank has no images, create empty tensor with correct hidden size
            if hasattr(self, "merger") and hasattr(self.merger, "ln_q"):
                ln_q = self.merger.ln_q
                if hasattr(ln_q, "normalized_shape"):
                    hidden_size = ln_q.normalized_shape[0]
                elif hasattr(ln_q, "weight"):
                    hidden_size = ln_q.weight.shape[0]
                else:
                    raise RuntimeError(
                        f"Cannot determine hidden_size from ln_q. Type: {type(ln_q).__name__}"
                    )
            elif hasattr(self, "out_hidden_size"):
                hidden_size = self.out_hidden_size
            elif hasattr(self, "config") and hasattr(self.config, "hidden_size"):
                hidden_size = self.config.hidden_size
            else:
                raise RuntimeError(
                    f"Cannot determine hidden_size for VisionTransformer. "
                    f"Model type: {type(self).__name__}. "
                    f"Available attributes: {[a for a in dir(self) if not a.startswith('_')]}"
                )

            local_embeddings = torch.empty(
                (0, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Step 4: All-gather (contiguous assignment, no reordering needed)
        # Compute per-rank embedding counts locally (grid_thw is replicated on all ranks)
        all_counts = [
            sum(embedding_counts[i] for i in image_assignments[r])
            for r in range(sp_size)
        ]
        all_embeddings = gather_vision_embeddings(
            local_embeddings, sp_group, all_counts
        )

        assert all_embeddings.shape[0] == total_embeddings, (
            f"[Vision DP] Output embedding count mismatch: "
            f"all_embeddings.shape[0]={all_embeddings.shape[0]}, "
            f"expected={total_embeddings}"
        )

        return all_embeddings

    return dp_vision_forward


def _patch_vision_class(cls, class_name: str) -> None:
    """Patch a single VisionTransformer class with Vision DP, with idempotency guard."""
    if getattr(cls, "_vision_dp_patched", False):
        return
    original = cls.forward
    cls.forward = create_dp_vision_forward(original)
    cls._vision_dp_patched = True
    logger.info(f"[Vision DP] Patched {class_name}.forward")


def apply_vision_dp_patch():
    """
    Apply Vision DP monkey patch to supported VisionTransformer classes.

    Should be called BEFORE model loading (patches the class, not instance).
    The wrapped forward reads the Ulysses SP group/size/rank from globals at
    runtime, so the SP group doesn't need to be set at patch time.

    Safe to call multiple times — each class is only patched once.
    """
    # Patch Qwen2-VL VisionTransformer
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VisionTransformerPretrainedModel,
        )

        _patch_vision_class(
            Qwen2VisionTransformerPretrainedModel,
            "Qwen2VisionTransformerPretrainedModel",
        )
    except ImportError:
        pass

    # Patch Qwen2.5-VL VisionTransformer
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        _patch_vision_class(
            Qwen2_5_VisionTransformerPretrainedModel,
            "Qwen2_5_VisionTransformerPretrainedModel",
        )
    except ImportError:
        pass

    # Patch Qwen3-VL VisionModel
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionModel,
        )

        _patch_vision_class(Qwen3VLVisionModel, "Qwen3VLVisionModel")
    except ImportError:
        pass
