import argparse
import time
import torch
from transformers import AutoModelForCausalLM

from token_trie import TokenTrie
from tree_training_engine import TreeTrainingEngine
from dense import train


def strip_padding(ids: torch.Tensor, padding_token: int = 0) -> torch.Tensor:
    assert ids.dim() == 1, "ids must be a 1D tensor"
    mask = ids != padding_token
    last = mask.nonzero(as_tuple=False)[-1].item()
    return ids[: last + 1]

def parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def save_gradients(model, path: str):
    """
    Save all parameter gradients to a file.
    Stored as a dict: {param_name: grad_tensor (cpu)}
    """
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu()
        else:
            grads[name] = None
    torch.save(grads, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-imp", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--train-imp", type=str, default="dense",
                        choices=["dense", "tree"])

    parser.add_argument("--block-size", type=int, default=2048)

    parser.add_argument("--throw-prefix", type=int, default=None,
                    help="Number of prefix tokens to throw away from each sequence in the dataset")
    parser.add_argument("--grad-file", type=str, default=None,
                    help="Path to save full model gradients (torch.save)")

    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)

    # -------- load data --------
    data = torch.load(args.data, map_location="cpu")
    input_ids = data["input_data"]["input_ids"]
    input_ids = [strip_padding(ids.squeeze(0)) for ids in input_ids]
    if args.throw_prefix is not None:
        input_ids = [ids[args.throw_prefix:] for ids in input_ids if ids.numel() > args.throw_prefix]

    weights = [1.0] * len(input_ids)
    n_tokens = sum(len(ids) for ids in input_ids)

    # -------- load model --------
    model_path = f"{args.model_folder}/{args.model}"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=args.attn_imp,
        device_map="cuda",
    )

    torch.cuda.synchronize()
    start_time = time.time()

    # ================= dense =================
    if args.train_imp == "dense":
        loss = train(model, input_ids, weights)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        throughput = n_tokens / elapsed

        print(f"[Dense Training]")
        print(f"Loss       : {loss:.6f}")
        print(f"Time (s)   : {elapsed:.3f}")
        print(f"Throughput : {throughput:.2f} tokens/s")

    # ================= tree =================
    else:
        trie = TokenTrie(input_ids, weights, dtype=dtype)

        max_seq_len = max(len(ids) for ids in input_ids)

        engine = TreeTrainingEngine(
            model=model,
            max_seq_len=max_seq_len,
            dtype=dtype,
        )

        loss = engine.train(
            token_trie=trie,
            block_size=args.block_size,
        )

        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        throughput = n_tokens / elapsed
        throughput_leafed = trie.n_leafed_tokens / elapsed
        throughput_tree = trie.n_tree_tokens / elapsed

        print(f"[Tree Training]")
        print(f"Loss       : {loss:.6f}")
        print(f"Time (s)   : {elapsed:.3f}")
        print(f"Throughput : {throughput:.2f} tokens/s, {throughput_leafed:.2f} leafed-tokens/s, {throughput_tree:.2f} tree-tokens/s")

        print(
            f"n_tokens        = {trie.n_tokens}\n"
            f"n_leafed_tokens = {trie.n_leafed_tokens}\n"
            f"n_tree_tokens   = {trie.n_tree_tokens}"
        )
        print(
            f"Overlap Ratio   = "
            f"{trie.n_tokens / trie.n_tree_tokens:.4f}x, "
            f"{trie.n_leafed_tokens / trie.n_tree_tokens:.4f}x (Leafed)"
        )

    # -------- save gradients --------
    if args.grad_file is not None:
        save_gradients(model, args.grad_file)


if __name__ == "__main__":
    main()