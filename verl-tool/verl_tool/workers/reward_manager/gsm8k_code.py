"""
GSM8K-Code Reward Manager for MT-GRPO

This reward manager is designed for code execution tasks like GSM8K,
providing both turn-level and outcome-level rewards:

- turn_reward: Based on code execution success rate (trajectory-level scalar)
- outcome_reward: Based on final answer correctness (trajectory-level scalar)

The turn_reward is passed through reward_extra_info for MT-GRPO to use.
"""

import re
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
from verl import DataProto
from verl.workers.reward_manager.registry import register


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract answer from <answer> tags or \\boxed{}."""
    # Try <answer> tags first
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try \\boxed{}
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove whitespace and convert to lowercase
    answer = answer.strip().lower()
    # Remove common formatting
    answer = answer.replace(",", "").replace("$", "").replace("%", "")
    # Try to extract just the number
    num_match = re.search(r'[-+]?\d*\.?\d+', answer)
    if num_match:
        return num_match.group()
    return answer


def check_answer_correctness(predicted: str, ground_truth: Any) -> bool:
    """Check if predicted answer matches ground truth."""
    if predicted is None:
        return False
    
    pred_norm = normalize_answer(predicted)
    
    # Handle different ground_truth formats
    if isinstance(ground_truth, dict):
        gt_values = ground_truth.get('target', ground_truth.get('answer', ''))
    elif isinstance(ground_truth, list):
        gt_values = ground_truth
    else:
        gt_values = [ground_truth]
    
    if not isinstance(gt_values, list):
        gt_values = [gt_values]
    
    for gt in gt_values:
        gt_norm = normalize_answer(str(gt))
        if pred_norm == gt_norm:
            return True
        # Try numerical comparison
        try:
            if abs(float(pred_norm) - float(gt_norm)) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
    
    return False


def compute_turn_reward(tool_interact_info: List[Dict[str, Any]]) -> float:
    """
    Compute turn-level reward based on code execution success.
    
    This mimics the code_execution_reward_func from Multi-Turn-RL-Agent:
    - Only counts steps where code was actually executed (valid_action=True)
    - Checks if execution was successful (no error in output)
    - Returns: 0.3 * (success_ratio) + 0.05 * (num_successes)
    
    Args:
        tool_interact_info: List of tool interaction info dicts from agent_loop
        
    Returns:
        Turn reward value (trajectory-level scalar, typically 0.0 to ~0.5)
    """
    if not tool_interact_info:
        return 0.0
    
    total_code_steps = 0  # Steps where code was actually executed
    successful_executions = 0
    
    for info in tool_interact_info:
        if info is None:
            continue
        
        # Check if action was valid (code was executed)
        valid_action = info.get('valid_action', False)
        if not valid_action:
            continue
        
        total_code_steps += 1
        
        # Check if execution was successful (no error)
        obs = info.get('obs', '')
        if isinstance(obs, str):
            # Check for error indicators
            is_error = (
                'Error:' in obs or
                'Traceback' in obs or
                'timed out' in obs.lower() or
                'exception' in obs.lower()
            )
            if not is_error and obs.strip():
                successful_executions += 1
    
    # Return proportional reward (matching original implementation)
    if total_code_steps == 0:
        return 0.0
    
    # Formula from Multi-Turn-RL-Agent
    return 0.3 * (successful_executions / total_code_steps) + 0.05 * successful_executions


@register("gsm8k_code")
class GSM8KCodeRewardManager:
    """
    Reward Manager for GSM8K Code task with MT-GRPO support.
    
    This reward manager:
    1. Computes outcome_reward based on answer correctness (1.0 if correct)
    2. Computes turn_reward based on code execution success rate
    3. Returns turn_reward through reward_extra_info for MT-GRPO
    
    The turn_reward will be used by MT-GRPO to assign turn-level advantages
    to tokens that participated in tool calling.
    """
    
    name = "gsm8k_code"
    
    def __init__(
        self,
        tokenizer=None,
        num_examine: int = 1,
        correct_reward: float = 1.0,
        format_reward: float = 0.1,
        **kwargs
    ) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.correct_reward = correct_reward
        self.format_reward = format_reward
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Compute rewards for GSM8K Code task.
        
        Returns:
            If return_dict=True:
                {
                    "reward_tensor": tensor of shape (batch_size, seq_len),
                    "reward_extra_info": {
                        "turn_reward": list of turn rewards for MT-GRPO,
                        "accuracy": list of accuracy scores,
                        ...
                    }
                }
            Otherwise:
                reward_tensor
        """
        # Check for pre-computed rm_scores
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]
        
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # For logging
        already_printed = 0
        
        for i in range(batch_size):
            data_item = data[i]
            
            # Get prompt and response
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            response_ids = data_item.batch['responses']
            attention_mask = data_item.batch['attention_mask']
            valid_response_length = attention_mask[prompt_length:].sum().item()
            valid_response_ids = response_ids[:int(valid_response_length)]
            
            # Decode response
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Get ground truth
            ground_truth = data_item.non_tensor_batch.get('reward_model', {}).get('ground_truth', None)
            
            # Get tool interaction info for turn reward
            tool_interact_info = data_item.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            
            # Compute turn_reward (trajectory-level scalar)
            turn_reward = compute_turn_reward(tool_interact_info) if tool_interact_info else 0.0
            reward_extra_info['turn_reward'].append(turn_reward)
            
            # Compute outcome_reward (answer correctness)
            extracted_answer = extract_final_answer(response_str)
            is_correct = check_answer_correctness(extracted_answer, ground_truth)
            
            if is_correct:
                outcome_reward = self.correct_reward
                accuracy = 1.0
            elif extracted_answer is not None:
                outcome_reward = self.format_reward  # Partial reward for having an answer
                accuracy = 0.0
            else:
                outcome_reward = 0.0
                accuracy = 0.0
            
            # Store metrics
            reward_extra_info['accuracy'].append(accuracy)
            reward_extra_info['has_answer'].append(1.0 if extracted_answer else 0.0)
            reward_extra_info['outcome_reward'].append(outcome_reward)
            
            # Count tool usage
            num_turns = len(tool_interact_info) if tool_interact_info else 0
            num_valid = sum(1 for t in (tool_interact_info or []) if t and t.get('valid_action', False))
            reward_extra_info['num_turns'].append(num_turns)
            reward_extra_info['num_valid_actions'].append(num_valid)
            
            # Place outcome reward at the last valid position
            valid_len = int(valid_response_length)
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = outcome_reward
            
            # Log samples for debugging
            if already_printed < self.num_examine:
                already_printed += 1
                print("=" * 50)
                print(f"[Sample {i}]")
                print(f"[Response] {response_str[:500]}...")
                print(f"[Ground Truth] {ground_truth}")
                print(f"[Extracted Answer] {extracted_answer}")
                print(f"[Is Correct] {is_correct}")
                print(f"[Turn Reward] {turn_reward}")
                print(f"[Outcome Reward] {outcome_reward}")
                print(f"[Num Turns] {num_turns}, [Num Valid] {num_valid}")
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info)
            }
        return reward_tensor
