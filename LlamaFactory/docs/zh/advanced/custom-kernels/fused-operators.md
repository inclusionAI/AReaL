# Fused Operators

LLaMA-Factory 提供了一系列针对特定硬件优化的融合算子。这些算子位于 `src/llamafactory/v1/plugins/model_plugins/kernels/ops` 目录下。

系统启动时，`scan_all_kernels` 函数会自动扫描该目录，注册所有可用的算子。您可以通过 `apply_default_kernels(model, include_kernels="auto")` 一键启用它们，或者使用 `apply_kernel` 单独启用。

以下是当前支持的融合算子详情：

## NpuFusedRMSNorm
RMSNorm（Root Mean Square Layer Normalization）是一种常用于大模型的归一化方法。在推理或训练中，RMSNorm 融合算子 将bias、residual等操作进行融合，可以减少显存访问次数，加速计算。

Ascend npu 通过 `torch_npu.npu_rms_norm` 接口提供 RMSNorm 融合算子调用接口，支持 float16, bfloat16, float 等数据格式。RMSNorm 算子常见于Qwen等LLM模型中，由于torch侧没有提供 RMSNorm 算子的接口，因此在模型中通常是以自定义类的形式出现，通过替换 RMSNorm 类的 `forward` 方法即可使能。

```python
def _npu_rms_forward(self, hidden_states):
    """NPU forward implementation for RMSNorm.

    Args:
        self: RMSNorm module instance with `weight` and `variance_epsilon`.
        hidden_states: Input hidden states tensor, same shape as the baseline.

    Returns:
        Normalized tensor consistent with the baseline RMSNorm behavior.
    """

    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
```

 在 LlamaFactory 中，通过 `NpuRMSNormKernel` 提供使能该融合算子的入口，只需要调用 `apply_kernel("npu_fused_rmsnorm", model=model)` 即可针对已适配的模型使能 npu RMSNorm 融合算子。

## NpuFusedSwiGlu
SwiGLU（Swish-Gated Linear Unit）是一种结合了Swish激活函数和门控线性单元（GLU）的混合激活函数，其主要功能是对输入张量进行门控线性变换，近年来被广泛应用于 LLM 模型中的 MLP 层。SwiGLU 融合算子将分割、激活、矩阵乘等多个操作融合为单一硬件指令，避免多次内核启动开销。

Ascend npu 通过 `torch_npu.npu_swiglu` 接口提供 SwiGLU 融合算子调用接口，支持 float16，bfloat16，float SwiGLU 算子常见于Qwen等LLM模型中，由于torch侧没有提供 SwiGLU 算子的接口，因此在模型中通常是以自定义类的形式出现，通过替换 SwiGLU 类的 `forward` 方法即可使能。替换过程可参考如下示例：

```python
# 原始 MLP forward 方法：
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

# 替换后的 forward 方法：
def _npu_swiglu_forward(self, hidden_state):
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )
```

 在 LLaMA-Factory 中，通过 `NpuSwiGluKernel` 提供使能该融合算子的入口，只需要调用 `apply_kernel("npu_fused_swiglu", model=model)` 即可针对已适配的模型使能 npu SwiGLU 融合算子。对于未适配的模型，如有需要，您可根据示例以及[开发者文档](../../dev-guide/plugins/model-plugins/kernels.md)自行适配。


## NpuFusedRoPE
RoPE（Rotary Positional Embedding，旋转式位置嵌入） 是一种位置编码技术，广泛应用于 Qwen 等 LLM 模型中，用于有效编码文本序列的位置信息。它结合了绝对位置编码的稳定性与相对位置编码的灵活性，同时具备优秀的长度泛化能力。传统 RoPE 算子通常在 LLM 等模型结构中通过自定义函数的形式实现。RoPE 融合算子将原计算流程合并为单个硬件优化算子，从而提升性能。

Ascend npu 通过 `torch_npu.npu_rotary_mul` 提供 RoPE 融合算子调用接口，支持 float16，bfloat16，float32 等数据格式。以 Qwen3 系列模型为例，通过替换其 `apply_rotary_pos_emb` 函数即可实现 RoPE融合算子使能：

```python
# 原始 apply_rotary_pos_emb：
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 替换 RoPE 融合算子后：
def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed
```

 在 LLaMA-Factory 中，通过 `NpuRoPEKernel` 提供使能该融合算子的入口，只需要调用 `apply_kernel("npu_fused_rope", model=model)` 即可针对已适配的模型使能 npu RoPE 融合算子。对于未适配的模型，如有需要，您可根据示例以及[开发者文档](../../dev-guide/plugins/model-plugins/kernels.md)自行适配。


## NpuFusedMoE
MoE（Mixture of Experts）模型通过稀疏激活扩展容量。在原生 Transformers 实现中，使用串行循环遍历专家，导致内核启动开销大、硬件利用率低。

**MoE 融合算子** 利用 **GMM（Grouped Matrix Multiplication，分组矩阵乘）** 技术，支持在单个硬件指令内并行处理多组不同形状（行数不一）的矩阵乘法，消减循环开销，同时无需额外的显存复制，显著提升训练性能。

Ascend npu 通过 `torch_npu.npu_grouped_matmul` 等接口提供底层支持，通过替换模型中的 MoE Block forward 方法，即可利用 NPU 的分组矩阵乘能力。

核心逻辑替换如下（简化示意）：

```python
def _npu_moe_forward(self, hidden_states, routing_weights, router_indices):
    # 1. 排序：将乱序的 Token 按指派的专家归类，并生成索引映射
    permuted_states, row_map = torch_npu.npu_moe_token_permute(hidden_states, router_indices)

    # 2. 统计：计算每个专家需要处理的 Token 数量
    tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)

    # 3. 计算 (GMM)：一次性并行计算所有专家的权重，自动适配不同专家的输入长度
    inter_states = torch_npu.npu_grouped_matmul(permuted_states, self.gate_up_proj_weights, split_sizes=tokens_per_expert, ...)
    inter_states = torch_npu.npu_swiglu(inter_states)
    output = torch_npu.npu_grouped_matmul(inter_states, self.down_proj_weights, split_sizes=tokens_per_expert, ...)

    # 4. 还原：将结果恢复成原始 Token 顺序并应用路由权重
    return torch_npu.npu_moe_token_unpermute(output, row_map, routing_weights)
```

在 LLaMA-Factory 中，通过 `NpuFusedMoEKernel` 提供使能该融合算子的入口。只需要调用 `apply_kernel("npu_fused_moe", model=model)` 即可针对已适配的模型使能 NPU MoE 融合算子。对于未适配的模型，您也可以参考上述示例代码以及[开发者文档](../../dev-guide/plugins/model-plugins/kernels.md)自行适配。
