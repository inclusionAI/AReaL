import trl
import torch
from typing import List, Any, Dict, Union, Optional
from torch.utils.data import SequentialSampler

def add_and_init_special_tokens(model, tokenizer, new_special_tokens: Optional[List[str]] = None):
    """
    Adds new special tokens to the tokenizer and initializes their embeddings.
    """
    if new_special_tokens is None:
        new_special_tokens = [
            "<Think>", "</Think>", "<Parallel>", "</Parallel>", "<Goal>", "</Goal>", 
            "<Outline>", "</Outline>", "<Path>", "</Path>", "<Conclusion>", "</Conclusion>"
        ]
    
    tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    model.resize_token_embeddings(new_num_tokens=len(tokenizer), pad_to_multiple_of=64)

    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied = embed.weight.data_ptr() == lm_head.weight.data_ptr()

    for tok in new_special_tokens:
        base_word = tok.strip("<>")
        base_ids = tokenizer(base_word, add_special_tokens=False).input_ids
        
        if all(i != tokenizer.unk_token_id for i in base_ids):
            avg_embed = embed(torch.tensor(base_ids, device=model.device)).mean(dim=0)
            special_id = tokenizer.convert_tokens_to_ids(tok)
            embed.weight.data[special_id] = avg_embed
            
            if not tied and lm_head.weight.shape == embed.weight.shape:
                avg_lm_logits = lm_head.weight.data[base_ids].mean(dim=0)
                lm_head.weight.data[special_id] = avg_lm_logits.clone()
        else:
            valid_ids = [i for i in base_ids if i != tokenizer.unk_token_id]
            print(f"Warning: Failed to init {tok}, some base tokens are unknown. Using available tokens: {[tokenizer.convert_ids_to_tokens(i) for i in valid_ids]}")
            if valid_ids:
                avg_embed = embed(torch.tensor(valid_ids, device=model.device)).mean(dim=0)
                special_id = tokenizer.convert_tokens_to_ids(tok)
                embed.weight.data[special_id] = avg_embed
                if not tied and lm_head.weight.shape == embed.weight.shape:
                    avg_lm_logits = lm_head.weight.data[valid_ids].mean(dim=0)
                    lm_head.weight.data[special_id] = avg_lm_logits.clone()


TAG_TOKEN_IDS = {
    'parallel_start': '<Parallel>',
    'parallel_end': '</Parallel>',
    'path_start': '<Path>',
    'path_end': '</Path>',
    'goal_start': '<Goal>',
    'goal_end': '</Goal>',
    'conclusion_start': '<Conclusion>',
    'conclusion_end': '</Conclusion>',
}


def generate_multiverse_attention_mask(input_ids, tokenizer, device='cpu'):
    seq_len = len(input_ids)
    # Start with a lower triangular matrix (causal mask)
    bool_attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)) # Keep bool intermediate mask

    # Assuming single-token tags for simplicity based on original code
    # If tags can be multi-token, this conversion needs adjustment
    parallel_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['parallel_start'])
    parallel_end_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['parallel_end'])
    path_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['path_start'])
    path_end_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['path_end'])
    #print(path_start_id, path_end_id, parallel_start_id, parallel_end_id)

    structure_stack = []
    i = 0
    while i < seq_len:
        current_token_id = input_ids[i]

        # Check <Parallel> start
        if current_token_id == parallel_start_id:
            structure_stack.append({'type': 'parallel', 'start_marker_index': i, 'path_spans': []})
            i += 1
            continue

        # Check <Path> start
        elif current_token_id == path_start_id:
            structure_stack.append({'type': 'path', 'start_marker_index': i})
            i += 1
            continue

        # Check </Path> end
        elif current_token_id == path_end_id:
            path_end_marker_index = i + 1

            if not structure_stack or structure_stack[-1]['type'] != 'path':
                raise ValueError(f"</Path> found at index {i} without a matching <Path> block on stack.")

            closed_path_block = structure_stack.pop()

            # Find the nearest enclosing parallel block to add this path span
            enclosing_parallel_block = None
            for block in reversed(structure_stack):
                if block['type'] == 'parallel':
                    enclosing_parallel_block = block
                    break

            if enclosing_parallel_block is None:
                raise ValueError(f"Path block ending at {i} is not enclosed within any <Parallel> block.")

            # Add the span including markers
            path_start_marker_index = closed_path_block['start_marker_index']
            if path_start_marker_index < path_end_marker_index:
                 enclosing_parallel_block['path_spans'].append((path_start_marker_index, path_end_marker_index))

            i = path_end_marker_index
            continue

        # Check </Parallel> end
        elif current_token_id == parallel_end_id:
            parallel_end_marker_index = i + 1

            if not structure_stack or structure_stack[-1]['type'] != 'parallel':
                 raise ValueError(f"</Parallel> found at index {i} without a matching <Parallel> block on stack.")

            closed_parallel_block = structure_stack.pop()
            #print(closed_parallel_block)
            path_spans_in_this_block = closed_parallel_block['path_spans']

            num_paths = len(path_spans_in_this_block)
            if num_paths > 1:
                all_i_indices_to_mask = []
                all_j_indices_to_mask = []
                for path_idx_a in range(num_paths):
                    start_a, end_a = path_spans_in_this_block[path_idx_a]
                    # Ensure valid span before creating range
                    if start_a >= end_a: continue
                    indices_a = torch.arange(start_a, end_a, device=device)

                    for path_idx_b in range(path_idx_a + 1, num_paths):
                        start_b, end_b = path_spans_in_this_block[path_idx_b]
                        # Ensure valid span before creating range
                        if start_b >= end_b: continue
                        indices_b = torch.arange(start_b, end_b, device=device)

                        # Use broadcasting to get all (i, j) pairs efficiently
                        grid_i, grid_j = torch.meshgrid(indices_a, indices_b, indexing='ij')

                        all_i_indices_to_mask.append(grid_i.flatten())
                        all_j_indices_to_mask.append(grid_j.flatten())

                if all_i_indices_to_mask: # Check if there's anything to mask
                    final_i = torch.cat(all_i_indices_to_mask)
                    final_j = torch.cat(all_j_indices_to_mask)

                    # Apply mask using advanced indexing (ensure indices are valid)
                    # For bool mask, False means masked
                    bool_attention_mask[final_i, final_j] = False
                    bool_attention_mask[final_j, final_i] = False # Symmetric masking
            elif num_paths <= 1:
                # No masking needed if 0 or 1 path within the parallel block
                pass

            i = parallel_end_marker_index
            continue

        # Move to next token if no tag matched
        i += 1
    # --- End of parsing loop ---

    # Final check for unclosed blocks
    if structure_stack:
        print(structure_stack)
        print(input_ids)
        unclosed_types = [block['type'] for block in structure_stack]
        raise ValueError(f"Input sequence ended with unclosed blocks: {unclosed_types}")

    # Convert the final boolean mask to float format (0.0 for True, -inf for False)
    float_attention_mask = torch.full_like(bool_attention_mask, -torch.inf, dtype=torch.float)
    float_attention_mask = float_attention_mask.masked_fill(bool_attention_mask, 0.0)

    return float_attention_mask


def generate_multiverse_position_ids(input_ids: List[int], tokenizer) -> List[int]:
    """Generates position IDs accounting for Parallel, Goal, Path, Conclusion structure."""
    # Get special token IDs
    tag_ids = {
        tag: tokenizer.convert_tokens_to_ids(token)
        for tag, token in TAG_TOKEN_IDS.items()
    }

    position_ids = torch.arange(len(input_ids), device='cpu', dtype=torch.long)
    parallel_stack = []

    i = 0
    while i < len(input_ids):
        token_id = input_ids[i]
        current_block_state = parallel_stack[-1] if parallel_stack else None
        # --- Tag Matching Logic --- 
        if token_id == tag_ids['parallel_start']:
            #print(f"Parallel start at {i}")
            parallel_stack.append({
                'goal_end_pos_id': -1,
                'max_path_len': 0,
                'is_in_goal': False,
                'is_in_path': False,
                'is_in_conclusion': False,
            })
        
        elif token_id == tag_ids['goal_start'] and current_block_state:
            #print(f"Goal start at {i}")
            current_block_state['is_in_goal'] = True

        elif token_id == tag_ids['goal_end'] and current_block_state and current_block_state['is_in_goal']:
            #print(f"Goal end at {i}")
            current_block_state['goal_end_pos_id'] = position_ids[i]
            current_block_state['is_in_goal'] = False

        elif token_id == tag_ids['path_start'] and current_block_state and current_block_state['goal_end_pos_id'] != -1:
            current_block_state['is_in_path'] = True
            position_ids[i:] -= position_ids[i] - (current_block_state['goal_end_pos_id'] + 1)

        elif token_id == tag_ids['path_end'] and current_block_state and current_block_state['is_in_path']:
            # Update max path length for this parallel block
            current_block_state['max_path_len'] = max(
                current_block_state['max_path_len'],
                position_ids[i] - current_block_state['goal_end_pos_id']
            )
            # Reset path state
            current_block_state['is_in_path'] = False
        
        elif token_id == tag_ids['conclusion_start'] and current_block_state and current_block_state['goal_end_pos_id'] != -1:
            current_block_state['is_in_conclusion'] = True
            # Conclusion starts after the conceptual space of the longest path
            position_ids[i:] -= position_ids[i] - (current_block_state['goal_end_pos_id'] + current_block_state['max_path_len'] + 1)

        elif token_id == tag_ids['conclusion_end'] and current_block_state and current_block_state['is_in_conclusion']:
            current_block_state['is_in_conclusion'] = False

        elif token_id == tag_ids['parallel_end'] and parallel_stack:
            parallel_stack.pop()
        
        i += 1

    # Final check for unclosed blocks (optional, for robustness)
    if parallel_stack:
        print("Warning: Input sequence ended with unclosed <Parallel> blocks.")
        # Depending on requirements, either raise error or handle gracefully
        # raise ValueError("Input sequence ended with unclosed <Parallel> blocks.")

    # Sanity check length
    if len(position_ids) != len(input_ids):
         raise ValueError("Position ID generation length mismatch!")

    return position_ids


class MultiverseDataCollatorForCompletionOnlyLM(trl.DataCollatorForCompletionOnlyLM):
    def __init__(self, *args, max_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # First, generate full attention masks and position ids for complete sequences
        attention_masks = []
        position_ids = []
        
        for example in examples:
            # Get the complete input_ids (before any truncation)
            if isinstance(example, dict):
                input_ids = example['input_ids']
            else:
                input_ids = example
            
            # Generate full attention mask and position ids based on complete sequence
            attention_mask = generate_multiverse_attention_mask(input_ids, self.tokenizer)
            position_id = generate_multiverse_position_ids(input_ids, self.tokenizer)
            
            attention_masks.append(attention_mask)
            position_ids.append(position_id)
        
        # Apply the standard collation with truncated examples
        batch = super().torch_call(examples)
        
        # Get the final sequence length after truncation
        final_seq_len = batch['input_ids'].shape[1]
        
        # Create custom attention masks and position ids with the same truncation
        batch['attention_mask'] = torch.zeros(len(examples), 1, final_seq_len, final_seq_len, dtype=torch.float, device='cpu')
        batch['position_ids'] = torch.zeros(len(examples), final_seq_len, dtype=torch.long, device='cpu')

        for i in range(len(examples)):
            # Apply the same truncation to attention mask and position ids
            batch['attention_mask'][i, 0] = attention_masks[i][:final_seq_len, :final_seq_len]
            batch['position_ids'][i] = position_ids[i][:final_seq_len]
            batch['input_ids'][i] = batch['input_ids'][i][:final_seq_len]
            batch['labels'][i] = batch['labels'][i][:final_seq_len]
        
        return batch


class SequentialSFTTrainer(trl.SFTTrainer):
    """
    Custom SFTTrainer that uses sequential sampling instead of random sampling
    """
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """Override sampler method to use sequential sampling instead of random sampling"""
        if self.train_dataset is None or not hasattr(self.train_dataset, '__len__'):
            return None
        
        # If group_by_length is set, still use length-grouped sampler
        if self.args.group_by_length:
            return super()._get_train_sampler()
        else:
            # Use sequential sampler
            return SequentialSampler(self.train_dataset)