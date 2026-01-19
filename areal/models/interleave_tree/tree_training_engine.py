from math import ceil

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

class TreeTrainingEngine:
    def __init__(self, model, dtype: torch.dtype, max_seq_len: int):
        self.model = model
        self.device = model.device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        kv_buffer_shape = (1, self.n_kv_heads, max_seq_len, self.head_dim)

        # Initialize KV cache buffers
        self.kv_cache = (
            [
                torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                for _ in range(self.n_layers)
            ],
            [
                torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                for _ in range(self.n_layers)
            ],
        )

        # Initialize gradient accumulators for KV cache
        self.grad_kv = (
            [
                torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                for _ in range(self.n_layers)
            ],
            [
                torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                for _ in range(self.n_layers)
            ],
        )

        # Token buffer and metadata
        self.tokens = torch.zeros((max_seq_len), device=self.device, dtype=torch.long)
        self.token_weight = torch.zeros((max_seq_len), device=self.device, dtype=dtype)
        self.extra_labels = [[] for _ in range(max_seq_len)]

        self.cur_len = 0

    def build_kv(self, start: int, end: int):
        """
        Build KV cache for tokens in [start, end).
        Uses the existing prefix cache [0, start) and computes KV for new tokens.
        """
        # Build prefix cache from existing KV
        prefix_cache = DynamicCache()
        for l in range(self.n_layers):
            prefix_cache.update(
                self.kv_cache[0][l][:, :, :start, :],
                self.kv_cache[1][l][:, :, :start, :],
                layer_idx=l,
            )

        # Forward pass to compute new KV
        out = self.model(
            self.tokens[start:end].unsqueeze(0),
            past_key_values=prefix_cache,
            use_cache=True,
        )
        new_cache = out.past_key_values

        # Write new KV into cache
        for l, layer in enumerate(new_cache.layers):
            self.kv_cache[0][l][:, :, start:end, :] = layer.keys[:, :, start:end, :]
            self.kv_cache[1][l][:, :, start:end, :] = layer.values[:, :, start:end, :]

    @torch.no_grad()
    def push(
        self,
        new_tokens: torch.LongTensor,
        weight: torch.Tensor,
        kv_len: int,
    ):
        B = new_tokens.numel()
        assert self.cur_len + B <= self.max_seq_len, (
            f"Exceeds max_seq_len: cur_len={self.cur_len}, new_tokens={B}, max={self.max_seq_len}"
        )

        # Add weights
        self.token_weight[: self.cur_len + B] += weight.to(self.device)

        # Write tokens
        self.tokens[self.cur_len : self.cur_len + B] = new_tokens

        # Build KV if needed
        if self.cur_len < kv_len:
            self.build_kv(self.cur_len, kv_len)

        self.cur_len += B

    def pop(self, start: int) -> float:
        assert 0 <= start < self.cur_len, f"Invalid start={start}, cur_len={self.cur_len}"

        end = self.cur_len
        B = end - start

        popped_tokens = self.tokens[start:end]

        # 1. Build prefix KV with requires_grad=True
        prefix_cache = DynamicCache()
        prefix_kv = []

        for l in range(self.n_layers):
            k = self.kv_cache[0][l][:, :, :start, :].detach().requires_grad_(True)
            v = self.kv_cache[1][l][:, :, :start, :].detach().requires_grad_(True)
            prefix_cache.update(k, v, layer_idx=l)
            prefix_kv.append((k, v))

        # 2. Forward pass on popped block (builds computation graph)
        out = self.model(
            popped_tokens.unsqueeze(0), past_key_values=prefix_cache, use_cache=True
        )
        logits = out.logits  # [1, B, vocab]
        block_cache = out.past_key_values

        # 3. Compute main prediction loss
        if B > 1:
            logits_main = logits[:, :-1, :]
            targets = self.tokens[start + 1 : end]
            weights = self.token_weight[start + 1 : end]
            ce_main_token_loss = F.cross_entropy(
                logits_main.reshape(-1, logits_main.size(-1)),
                targets,
                reduction="none",
            )
            ce_main_loss = (ce_main_token_loss * weights).sum()
        else:
            ce_main_loss = torch.tensor(0.0, device=self.device)

        # 4. Compute extra labels loss (for branch predictions)
        ce_extra_loss = torch.tensor(0.0, device=self.device)
        for i in range(start, end):
            if not self.extra_labels[i]:
                continue

            local_i = i - start
            logit = logits[:, local_i, :]

            for lbl, lw in self.extra_labels[i]:
                extra_loss = F.cross_entropy(
                    logit,
                    torch.tensor([lbl], device=self.device),
                    reduction="sum",
                )
                ce_extra_loss += lw * extra_loss

        ce_loss = ce_main_loss + ce_extra_loss

        # 5. Backward with gradient injection for popped KV
        roots = [ce_loss]
        grads = [torch.ones_like(ce_loss)]

        for l, layer in enumerate(block_cache.layers):
            k = layer.keys[:, :, start:end, :]
            v = layer.values[:, :, start:end, :]
            roots.extend([k, v])
            grads.extend(
                [
                    self.grad_kv[0][l][:, :, start:end, :],
                    self.grad_kv[1][l][:, :, start:end, :],
                ]
            )

        torch.autograd.backward(roots, grads)

        # 6. Accumulate prefix KV gradients
        for l, (k, v) in enumerate(prefix_kv):
            self.grad_kv[0][l][:, :, :start, :] += k.grad
            self.grad_kv[1][l][:, :, :start, :] += v.grad

        # 7. Inject extra label for the parent node (branch prediction)
        if start > 0:
            first = popped_tokens[0].item()
            w = self.token_weight[start].item()
            self.extra_labels[start - 1].append((first, w))

        # 8. Cleanup: truncate and clear buffers
        self.token_weight[start:end].zero_()
        for i in range(start, end):
            self.extra_labels[i].clear()
        for l in range(self.n_layers):
            self.grad_kv[0][l][:, :, start:end, :].zero_()
            self.grad_kv[1][l][:, :, start:end, :].zero_()
        self.cur_len = start

        return ce_loss.item()

    def pop_byblock(self, start: int, block_size: int) -> float:
        end = self.cur_len
        length = end - start
        n_blocks = ceil(length / block_size)
        block_size_actual = ceil(length / n_blocks)
        loss = 0.0

        # Pop blocks in reverse order
        for b in range(n_blocks):
            pop_start = max(end - (b + 1) * block_size_actual, start)
            loss += self.pop(pop_start)

        return loss

    def train(
        self,
        token_trie,
        block_size: int,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        inputs, weights, lcp_lens = token_trie.inputs, token_trie.weights, token_trie.lcp_lens

        # Process each sequence
        for i in range(len(inputs)):
            input_ids = inputs[i].to(self.device)
            weight = weights[i]
            seq_len = input_ids.size(0)

            # Pop diverged branch from previous sequence
            if i > 0:
                lcp = lcp_lens[i - 1]
                if lcp < self.cur_len:
                    total_loss += self.pop_byblock(lcp, block_size)

            # Push new tokens
            new_tokens = input_ids[self.cur_len :]

            # Determine KV length to build (optimize for next pop)
            lcp_next = lcp_lens[i] if i < len(inputs) - 1 else 0
            B = new_tokens.numel()
            next_pop_len = self.cur_len + B - lcp_next

            if next_pop_len > block_size:
                n_blocks = ceil(next_pop_len / block_size)
                block_size_actual = ceil(next_pop_len / n_blocks)
                kv_len = max(self.cur_len + B - block_size_actual, lcp_next)
            else:
                kv_len = lcp_next

            self.push(new_tokens, weight, kv_len)

        # Final pop for remaining tokens
        if self.cur_len > 0:
            total_loss += self.pop_byblock(0, block_size)

        return total_loss

