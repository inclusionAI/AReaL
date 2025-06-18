from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    device_map="cuda",
    attn_implementation="flash_attention_2",
    torch_dtype="float16",
)
# model = AutoModelForCausalLM.from_config(config, device='cuda').cuda()
model.eval()

import torch

from realhf.impl.model.utils.padding import unpad_input

bs = 8
with torch.no_grad():
    seqlens = torch.randint(3, 12, (bs,), dtype=torch.int, device="cuda")
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(0, 100, (bs, max_seqlen), dtype=torch.long, device="cuda")

    attn_mask = torch.zeros((bs, max_seqlen), dtype=torch.bool, device="cuda")
    attn_mask[
        torch.arange(0, max_seqlen, device="cuda").unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    print(attn_mask)

    packed_input_ids, indices, cu_seqlens, max_seqlen = unpad_input(
        input_ids, attn_mask
    )

    assert torch.allclose(
        cu_seqlens, torch.nn.functional.pad(seqlens.cumsum(0, dtype=torch.int), (1, 0))
    )
    position_ids = compute_varlen_position_indices(int(sum(seqlens)), cu_seqlens)

    logits2 = model(
        input_ids=input_ids, attention_mask=attn_mask, use_cache=False
    ).logits
    logits2, _, _, _ = unpad_input(logits2, attn_mask)

    logits1 = model(
        input_ids=packed_input_ids.unsqueeze(0),
        position_ids=position_ids.unsqueeze(0),
        attention_mask=None,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        use_cache=False,
    ).logits.squeeze()
    assert logits1.shape == logits2.shape
    assert torch.allclose(logits1, logits2), (torch.abs(logits1 - logits2)).max()
