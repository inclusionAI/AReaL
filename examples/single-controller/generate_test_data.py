"""Generate fixed test data for AReaL vs HybridEngine comparison.

Usage:
    # Use real text (recommended, produces realistic loss in 2-5 range):
    python generate_test_data.py \
        --model_path /storage/openpsi/models/moe-mini-v25-e256-ep8tp1pp1-fp8-structure-fitted-adamw-new-3T \
        --seq_len 512 --batch_size 1 \
        --output /tmp/comparison/test_input.pt

    # Use random tokens (high loss ~13, only for quick smoke test):
    python generate_test_data.py \
        --model_path /storage/openpsi/models/moe-mini-v25-e256-ep8tp1pp1-fp8-structure-fitted-adamw-new-3T \
        --seq_len 128 --batch_size 1 --random \
        --output /tmp/comparison/test_input.pt
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

# Representative text for testing (mix of Chinese and English, common patterns)
_TEST_TEXT = """\
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。\
近年来，深度学习技术的发展推动了AI在自然语言处理、计算机视觉等领域的突破性进展。\
大型语言模型（Large Language Models，LLMs）通过在海量文本数据上进行预训练，学习了丰富的语言知识和推理能力。\
这些模型通常采用Transformer架构，利用自注意力机制来捕捉文本中的长程依赖关系。\
强化学习（Reinforcement Learning，RL）是机器学习的一个重要分支，研究智能体如何在环境中采取行动以最大化累积奖励。\
在大语言模型的对齐训练中，RLHF（Reinforcement Learning from Human Feedback）方法被广泛采用。\
该方法首先训练一个奖励模型来评估生成文本的质量，然后使用PPO或GRPO等算法来优化语言模型的策略。\
分布式训练是大规模模型训练的关键技术，包括数据并行、张量并行、流水线并行和专家并行等多种策略。\
混合专家模型（Mixture of Experts，MoE）通过稀疏激活机制，在保持模型容量的同时降低计算成本。\
MoE模型中的路由器负责将输入token分配给不同的专家，常见的路由策略包括Top-K选择和负载均衡优化。\
Lightning Attention是一种线性注意力机制，通过将注意力计算分解为递归形式来实现线性时间复杂度。\
与标准的softmax注意力不同，线性注意力使用衰减因子来模拟位置编码的效果，适合处理超长序列。\
Multi-Latent Attention（MLA）通过低秩投影来压缩KV缓存，有效减少推理时的显存占用。\
在实际部署中，模型的推理速度和显存效率是关键考量因素，需要在模型质量和计算资源之间取得平衡。\
The development of artificial intelligence has been marked by several key milestones. \
From the early days of symbolic AI to the current era of deep learning, the field has undergone tremendous transformation. \
Modern language models are trained on diverse datasets encompassing multiple languages and domains, \
enabling them to perform a wide range of tasks including text generation, translation, summarization, and reasoning. \
The scaling laws observed in large language models suggest that model performance improves predictably with increased \
compute, data, and model size, though the exact relationships depend on the specific architecture and training methodology. \
Efficient training of these models requires sophisticated distributed computing frameworks that can handle the \
massive computational requirements while maintaining numerical stability and training efficiency.\
"""


def main():
    parser = argparse.ArgumentParser(description="Generate test data for comparison")
    parser.add_argument("--model_path", type=str, required=True, help="HF model path for tokenizer")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--output", type=str, default="/tmp/comparison/test_input.pt", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--random", action="store_true", help="Use random token IDs instead of real text")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab_size: {vocab_size}")

    if args.random:
        print("Mode: random tokens")
        input_ids = torch.randint(1, vocab_size, (args.batch_size, args.seq_len), dtype=torch.long)
    else:
        print("Mode: real text")
        # Tokenize the test text, repeat/truncate to desired length
        tokens = tokenizer.encode(_TEST_TEXT, add_special_tokens=False)
        print(f"  Tokenized text length: {len(tokens)} tokens")
        # Repeat if needed to fill seq_len
        while len(tokens) < args.seq_len:
            tokens = tokens + tokens
        tokens = tokens[: args.seq_len]
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, S]
        if args.batch_size > 1:
            input_ids = input_ids.expand(args.batch_size, -1).contiguous()

    # Labels = same as input_ids (standard causal LM, model internally shifts)
    labels = input_ids.clone()

    # Loss mask: all 1s (compute loss on all tokens)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

    # Position ids
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(args.batch_size, -1)

    # Causal attention mask [batch, 1, seq, seq]
    causal_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool))
    attention_mask = causal_mask.expand(args.batch_size, -1, -1, -1)

    data = {
        "input_ids": input_ids,         # [B, S]
        "labels": labels,               # [B, S]
        "loss_mask": loss_mask,          # [B, S]
        "position_ids": position_ids,    # [B, S]
        "attention_mask": attention_mask, # [B, 1, S, S]
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "vocab_size": vocab_size,
        "seed": args.seed,
        "mode": "random" if args.random else "text",
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved test data to {args.output}")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  labels shape: {labels.shape}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  mode: {'random' if args.random else 'text'}")
    print(f"  seed: {args.seed}")


if __name__ == "__main__":
    main()
