# AReaL vs HybridEngine 对比验证方案

## 背景

BailingMoeV2.5 在 AReaL 中的 Megatron 适配已完成（Lightning Attention + MLA + MoE），mini 模型 SFT 验证通过（初始 loss=0.567，远低于目标 ≤2）。现需要与 HybridEngine 基线对比，验证 AReaL 实现的正确性。

对比在两台机器上分别运行：
- **AReaL**: 当前机器（有修改后的代码和模型）
- **HybridEngine**: 另一台有 antllm + 内部 Megatron-LM fork 的 pod

两端生成的结果保存到文件，然后离线对比。

## 对比结果

### 第一轮（随机 token 输入，seq_len=128）

HybridEngine 初始使用 `chunk_lightning_attn`（fla 默认线性 slopes），loss 差异极大：

| 指标 | AReaL | HybridEngine | 差异 |
|------|-------|-------------|------|
| Loss | 13.85 | 23.53 | 69.9% |
| PPL | 1,032,954 | 16,522,050,687 | 极大 |

**根因**: `chunk_lightning_attn` 内部使用线性 slopes `-(8/H)*(1-idx/N)*[0,1,...,H-1]`，
而模型实际训练使用 ALiBi geometric slopes。修复 HybridEngine 端改用 ALiBi slopes 后对齐。

### 第二轮（随机 token 输入，HybridEngine 修复后）

| 指标 | AReaL | HybridEngine | 差异 |
|------|-------|-------------|------|
| Loss | 13.8479 | 13.7540 | **0.68%** |
| PPL | 1,032,954 | 940,334 | 8.97% |

Loss 差异 0.68%（< 1% 阈值），PPL 差异是 loss 差异的 exp 放大效应。
随机 token 导致 loss 过高（~13.8），exp 放大后 PPL 差异看起来大。

### 第三轮（真实文本输入，seq_len=512）

使用实际中英文混合文本，loss 降到正常范围：

**AReaL 端结果**:
- Loss: 1.512, PPL: 4.54

等待 HybridEngine 端重新运行对比。

## 关键发现：g_gamma 公式差异

| 实现 | g_gamma 公式 | 正确性 |
|------|-------------|--------|
| AReaL | ALiBi geometric slopes × `(1-(idx-1)/(N-1)+1e-5)` | ✅ 匹配 HF + Megatron-LM 参考 |
| fla `chunk_lightning_attn` | 线性 `-(8/H)*(1-idx/N)*[0,1,...,H-1]` | ❌ fla 自己的默认值，非 BailingMoe 实际行为 |
| HF 参考 (`modeling_bailing_moe_v2_5.py:754`) | `build_slope_tensor()` → ALiBi slopes | ✅ 官方实现 |
| Megatron-LM 参考 (`attention.py:1744`) | `_build_slope_tensor()` → ALiBi slopes | ✅ 官方实现 |

**结论**: AReaL 的实现正确。HybridEngine 需要使用 `chunk_simple_gla` + ALiBi g_gamma，
不能直接调用 `chunk_lightning_attn`。

## 对比项

1. **End-to-end loss/ppl**: 相同输入的交叉熵 loss 和 perplexity
2. **单层输出**: 每层 TransformerLayer 输出的 hidden_states 逐层对比
3. **MoE 层**: 固定 router 权重（使 routing 确定性），对比 MoE 输出

## 模型和配置

- **模型**: mini v2.5 (`/storage/openpsi/models/moe-mini-v25-e256-ep8tp1pp1-fp8-structure-fitted-adamw-new-3T`)
  - 20 layers, hidden=2048, 16 heads, 256 experts, layer_group_size=5
  - 15 Lightning + 5 MLA layers, 1 Dense + 19 MoE layers
- **并行策略**: EP=8, TP=1, PP=1（单机 8 卡，最简配置避免并行引入差异）
- **精度**: bf16
- **输入**: 真实中英文混合文本，tokenize 后保存到文件供两端共用

## 脚本

```
comparison/
├── bailing-moe-v25-comparison-plan.md   # 本文档
├── test_input.pt                         # 共用输入数据
├── areal_outputs.pt                      # AReaL 端结果
├── hybrid_outputs.pt                     # HybridEngine 端结果（待生成）
└── scripts/
    ├── generate_test_data.py             # 生成共用输入（支持真实文本/随机 token）
    ├── compare_areal.py                  # AReaL 端 forward + 提取
    ├── compare_hybrid_engine.py          # HybridEngine 端 forward + 提取
    └── compare_results.py               # 离线对比两端结果
```

## 运行步骤

### Step 1: 生成输入数据

```bash
# 真实文本（推荐，loss 在 2-5 范围）
python scripts/generate_test_data.py \
    --model_path /storage/openpsi/models/moe-mini-v25-e256-ep8tp1pp1-fp8-structure-fitted-adamw-new-3T \
    --seq_len 512 --batch_size 1 --output test_input.pt

# 随机 token（快速冒烟测试，loss ~13）
python scripts/generate_test_data.py \
    --model_path /storage/openpsi/models/moe-mini-v25-e256-ep8tp1pp1-fp8-structure-fitted-adamw-new-3T \
    --seq_len 128 --batch_size 1 --random --output test_input.pt
```

### Step 2: AReaL 端（当前机器）

```bash
PYTHONPATH=/storage/openpsi/codes/chucai.dzq/gh/AReaL:$PYTHONPATH \
AREAL_SPMD_MODE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=8 scripts/compare_areal.py \
    --model_path /storage/openpsi/models/moe-mini-v25-e256-ep8tp1pp1-fp8-structure-fitted-adamw-new-3T \
    --input test_input.pt \
    --output areal_outputs.pt
```

### Step 3: HybridEngine 端（其他机器 pod）

**重要**: HybridEngine 端必须使用 ALiBi slopes 的 `chunk_simple_gla`，不能直接调用 `chunk_lightning_attn`。

```bash
PYTHONPATH=/home/admin/Asystem-HybridEngine:/home/admin/antllm:/home/admin/Megatron-LM:/home/admin/flash-linear-attention:$PYTHONPATH \
USE_MAX_V2=1 HYBRID_IGNORE_LOAD_CHECK=1 \
torchrun --nproc_per_node=8 scripts/compare_hybrid_engine.py \
    --model_path /storage/openpsi/models/moe-mini-v25-e256-ep8tp1pp1-fp8-structure-fitted-adamw-new-3T \
    --input test_input.pt \
    --output hybrid_outputs.pt
```

### Step 4: 对比（任意机器，仅 CPU）

```bash
python scripts/compare_results.py \
    --areal areal_outputs.pt \
    --hybrid hybrid_outputs.pt
```

## 验证标准

| 指标 | 期望 |
|------|------|
| Layer-0 输出（embedding 后第一层，Dense MLP） | max abs diff < 1e-4 |
| Lightning Attention 层输出 | max abs diff < 1e-3 |
| MLA 层输出 | max abs diff < 1e-3 |
| MoE 层输出（router 确定性时） | max abs diff < 1e-3 |
| End-to-end loss 差异 | < 1% |
| End-to-end ppl 差异 | < 5%（正常 loss 范围内） |

## MoE Router 固定方案

对比 MoE 层时，两端使用完全相同的 HF 权重（包括 router gate），且禁用 input_jitter (`moe_input_jitter_eps=0`)。由于权重相同且输入相同，router 决策应完全一致。

## 脚本详解

### generate_test_data.py

- 纯 CPU 脚本，不需要 GPU
- **默认模式**: 使用中英文混合真实文本，tokenize 后截断/重复到 seq_len
- **随机模式** (`--random`): 生成随机 token ids（仅用于快速冒烟测试）
- 输出 `test_input.pt` 包含 `input_ids`, `labels`, `loss_mask`, `position_ids`, `attention_mask`
- 两端均加载此文件确保输入完全一致

### compare_areal.py

- 通过 mbridge `AutoBridge.from_pretrained()` 加载模型（自动匹配 bailing_moe_linear/bailing_hybrid bridge）
- 初始化 Megatron parallel state + RNG tracker (`model_parallel_cuda_manual_seed`)
- 在每个 TransformerLayer 和 self_attention 上注册 `forward_hook` 提取中间结果
- 两次 forward：一次带 labels 得 loss，一次不带得 logits
- 按 PP rank 保存结果文件（PP=1 时只有一个文件）

### compare_hybrid_engine.py

- 使用 `MegatronBackend` + `antllm.bailing_moe_model_provider` 构建模型
- 配置内嵌 MINI_V25_CONFIG dict，基于 HybridEngine 的 `mini_v2.5.yaml`
- 需要 DCP 格式 checkpoint（`--dcp_path`）
- Hook 注册方式与 AReaL 端一致

### compare_results.py

- 纯 CPU 脚本
- 逐层对比：max abs diff, mean abs diff, relative L2, cosine similarity
- 输出汇总表格
- 自动判定 OK/WARN/FAIL

## HybridEngine 环境要求

HybridEngine pod 需要以下包：
- `antllm`（从 code.alipay.com/ai-dls/antllm.git）
- `Megatron-LM`（从 code.alipay.com/Arc/Megatron-LM.git）
- `flash-linear-attention`（fla，Lightning Attention kernel）
- `asystem_runtime`（HybridEngine 本身）

PYTHONPATH 示例：
```bash
export PYTHONPATH=/home/admin/Asystem-HybridEngine:/home/admin/antllm:/home/admin/Megatron-LM:/home/admin/flash-linear-attention:$PYTHONPATH
```
