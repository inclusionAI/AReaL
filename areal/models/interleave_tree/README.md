### 测试梯度计算

- 运行 `exp_bf16.sh` 在 bf16 下测试，运行 `exp_fp32.sh` 在 fp32 下测试。

    - 需要制定模型路径

    - 梯度比对文件输出到 `grad/` 下，为 `.txt`，测试了 $128$ 和 $2048$（训练中真正使用）两种块大小。

- 测试脚本 `run.py` 的参数

| 参数名 | 类型 | 默认值 | 作用说明 |
|------|------|--------|---------|
| `--model-folder` | `str` | 无（必填） | 模型所在的根目录路径 |
| `--model` | `str` | 无（必填） | 模型名称（与 `model-folder` 拼接得到完整路径） |
| `--data` | `str` | 无（必填） | 训练数据文件路径（`.pt` 格式） |
| `--dtype` | `str` | `bf16` | 模型与计算使用的数据类型，可选 `bf16` / `fp16` / `fp32` |
| `--attn-imp` | `str` | `flash_attention_2` | 注意力实现方式：`flash_attention_2` / `sdpa` / `eager` |
| `--train-imp` | `str` | `dense` | 训练实现方式：`dense` 为标准训练，`tree` 为 Tree Training |
| `--block-size` | `int` | `2048` | Tree Training 中每次反向传播处理的最大 token block 大小 |
| `--throw-prefix` | `int` | `None` | 从每个序列开头丢弃的前缀 token 数，用于节约显存 |
| `--grad-file` | `str` | `None` | 若指定路径，则将训练后模型的完整梯度保存到该文件 |

