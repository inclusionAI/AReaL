import fire
import json
import torch
import numpy as np
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def plot_entropy_bar(entropy, labels, title="Token Entropy", save_path="entropy_plot.png"):
    """
    Plot the token entropy with color highlighting based on masks and background shading.
    
    Args:
        entropy (list): List of entropy values corresponding to each token.
        labels (List[str]): List of labels for the tokens, e.g., "prompt", "action" or "obs".
        title (str): Plot title.
        save_path (str): Path where the plot will be saved.
    """
    # Color map for distinguishing between the parts
    color_map = {"prompt": "green", "action": "red", "obs": "blue"}
    
    plt.figure(figsize=(15 + len(entropy) * 0.01, 4))
    clipped_entropy = np.clip(entropy, 0, 10)
    token_indices = np.arange(len(entropy))

    # Initialize to hold color and label settings
    token_colors = [color_map.get(label, "gray") for label in labels]
    alpha_values = [0.6 if label == "prompt" else 0.9 for label in labels]  # Lighter for prompts, darker for actions and obs
    
    # Plot background color for each section
    last_idx = 0
    last_label = labels[0]
    for i in range(len(labels)):
        if labels[i] != last_label:
            plt.axvspan(last_idx, i - 1, color=color_map[last_label], alpha=0.1, label=f"{last_label.capitalize()} Background")
            last_idx = i
            last_label = labels[i]
    plt.axvspan(last_idx, len(labels) - 1, color=color_map[last_label], alpha=0.1, label=f"{last_label.capitalize()} Background")
    
    # Bar plot with clear separation for each token part
    for i in range(len(entropy)):
        plt.bar(i, clipped_entropy[i], color=token_colors[i], alpha=alpha_values[i])

    plt.title(title)
    plt.xlabel("Token Index")
    plt.ylabel("Entropy")
    plt.tight_layout()

    # Adding a legend to make distinction clear
    plt.legend(handles=[plt.Line2D([0], [0], color=color_map["prompt"], lw=4),
                        plt.Line2D([0], [0], color=color_map["action"], lw=4),
                        plt.Line2D([0], [0], color=color_map["obs"], lw=4)],
               labels=["Prompt", "Action", "Obs"], title="Token Type")
    
    # Grid lines for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig(save_path, dpi=300)
    return save_path

def main(
    file_path:str,
    model_name:str = "Qwen/Qwen2.5-Math-1.5B",
    batch_size=4,
    vis_dir: str = "entropy_vis",
):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    pad_token_id = tokenizer.pad_token_id

    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    data = datasets.Dataset.from_list(data)
    data = data.filter(lambda x: x['num_turn'] > 0, num_proc=8, desc="Filtering dataset with num_turn > 0")
    print(data)

    full_inputs = [x['prompt'] + x['response'] for x in data]
    full_inputs_with_mask = [x['prompt'] + x['response_with_loss_mask'] for x in data]

    # Tokenize the inputs
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_paths = []
    entropy_avgs = [] # list of sum entropy values, [0] for prompt, [1] for action 1, [2] for obs 1, [3] for action 2, [4] for obs 2, ...
    for i in tqdm(range(0, len(full_inputs), batch_size), desc="Processing batches", total=len(full_inputs) // batch_size):
        prompts = data['prompt'][i:i + batch_size]
        batch = full_inputs[i:i + batch_size]
        batch_with_mask = full_inputs_with_mask[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding="longest").to(model.device)
        inputs_with_mask = tokenizer(batch_with_mask, return_tensors='pt', padding="longest").to(model.device)
        attention_mask = inputs['attention_mask']

        # Get the model outputs
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits # [batch_size, seq_len, vocab_size]
        probs = torch.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        log_probs = torch.log(probs + 1e-9)  # [batch_size, seq_len, vocab_size]
        batch_entropy = -(probs * log_probs * attention_mask.unsqueeze(-1)).sum(dim=-1) # [batch_size, seq_len]
        entrypy_list = []
        for j in tqdm(range(len(batch_entropy)), desc=f"Processing batch {i//batch_size}", leave=False, total=len(batch_entropy)):
            effective_entry = batch_entropy[j][attention_mask[j] == 1].cpu().numpy()
            labels = ["prompt"] * len(tokenizer.encode(prompts[j], add_special_tokens=False))
            labels += ["action"] * (len(effective_entry) - len(labels))
            masks = inputs_with_mask['input_ids'][j][attention_mask[j] == 1]
            masks = (masks != pad_token_id).cpu().numpy()
            for k in range(len(labels)):
                if masks[k] == 0:
                    labels[k] = "obs"
            save_path = vis_dir / f"entropy_plot_sample_{i* batch_size + j}.png"
            # plot_entropy_bar(effective_entry.cpu().numpy(), labels, title=f"Token Entropy for Batch {i//batch_size}, Sample {j}", save_path=save_path)
            # print(f"Saved plot to {save_path}")
            # Calculate average entropy for each type
            last_idx = 0
            last_label = labels[0]
            avg_entropy = []
            for k in range(len(labels)):
                if labels[k] != last_label:
                    avg_entropy.append(effective_entry[last_idx:k].mean().item())
                    last_idx = k
                    last_label = labels[k]
            for k in range(len(avg_entropy)):
                if len(entropy_avgs) <= k:
                    entropy_avgs.append([])
                entropy_avgs[k].append(avg_entropy[k])
            
            entrypy_list.append(effective_entry)
            vis_paths.append(save_path)

    entropy_avgs = [np.mean(entropy) for entropy in entropy_avgs]
    for i, avg in enumerate(entropy_avgs):
        if i == 0:
            print(f"Average prompt entropy: {avg:.4f}")
        elif i % 2 == 1:
            print(f"Average action {i//2 + 1} entropy: {avg:.4f}")
        else:
            print(f"Average obs {i//2} entropy: {avg:.4f}")

        
if __name__ == "__main__":
    fire.Fire(main)

"""
This script calculates the entropy of model outputs for a given set of prompts and responses.
It uses a specified language model to compute the entropy of the responses based on the prompts.
The script reads a JSON file containing prompts and responses, tokenizes them, and computes the entropy
for each response using the model's logits.
```
# Usage:
python scripts/visualize_entropy.py --file_path path/to/data.json --model_name Qwen/Qwen2.5-Math-1.5B --batch_size 1
python scripts/visualize_entropy.py --file_path /home/dongfu/WorkSpace/verl-tool/verl_step_records/torl-fsdp-agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6debug/torl-step-1.json --model_name Qwen/Qwen2.5-Math-1.5B --batch_size 2
```
"""