import torch
from typing import List, Union

def lcp_torch(a: torch.Tensor, b: torch.Tensor) -> int:
    L = min(a.numel(), b.numel())
    eq = a[:L] == b[:L]
    return L if eq.all() else int((~eq).to(torch.int32).argmax().item())

def leafization(input_ids: List[torch.LongTensor], weights: List[torch.Tensor]):
    """
        参数：
            input_ids: List of Token Tensor（按字典序排序）
            weights: List of Float Tensor，表示每个 Token Tensor 的 per token 的 loss 权重

        将完全重叠的前缀合并，同时将对应的权重相加，并计算 lcp_lens 列表。
    """
    
    lcp_lens = []
    for i in range(len(input_ids)-1):
        seq_L, seq_R = input_ids[i], input_ids[i+1]
        lcp = lcp_torch(seq_L, seq_R)
        L = min(seq_L.numel(), seq_R.numel())
        if lcp < L and seq_L[lcp] > seq_R[lcp]:
            raise ValueError("Input_ids not sorted in lexicographic order.")
        lcp_lens.append(lcp)

    input_ids_leafed = []
    weights_leafed = []
    lcp_lens_leafed = []
    fork = -1
    for i in range(len(input_ids)):
        if i == len(input_ids)-1 or lcp_lens[i] < min(input_ids[i].numel(), input_ids[i+1].numel()):
            input_ids_leafed.append(input_ids[i])
            if i < len(input_ids)-1:
                lcp_lens_leafed.append(lcp_lens[i])
            weight = weights[i]
            for k in range(fork+1, i):
                weight[:weights[k].numel()] += weights[k]
            weights_leafed.append(weight)
            fork = i

    return input_ids_leafed, weights_leafed, lcp_lens_leafed

class TokenTrie:
    def __init__(
        self,
        inputs: List[torch.LongTensor],
        weights: Union[List[float], List[torch.Tensor]],
        sorted: bool = False,
        dtype: torch.dtype = None,
    ):
        assert len(inputs) == len(weights)

        if isinstance(weights[0], float):    # per sequence weight -> per token weights
            if dtype is None:
                raise ValueError("dtype must be provided when weights is List[float]")

            weight_tensors = []
            for ids, w in zip(inputs, weights):
                L = len(ids)
                wt = torch.empty(L, dtype=dtype)
                wt[0] = 0.0
                wt[1:] = w / (L - 1)
                weight_tensors.append(wt)

            weights = weight_tensors

        # -------- sort by lexicographical order of input_ids --------
        if not sorted:
            pairs = list(zip(inputs, weights))
            pairs.sort(key=lambda x: x[0].tolist())
            inputs_sorted, weights_sorted = [p[0] for p in pairs], [p[1] for p in pairs]
        else:
            inputs_sorted, weights_sorted = inputs, weights

        # -------- leafization --------
        self.inputs, self.weights, self.lcp_lens = \
            leafization(inputs_sorted, weights_sorted)

        # -------- statistics --------
        self.n_tokens = sum(len(ids) for ids in inputs)
        self.n_leafed_tokens = sum(len(ids) for ids in self.inputs)
        self.n_tree_tokens = self.n_leafed_tokens - sum(self.lcp_lens)