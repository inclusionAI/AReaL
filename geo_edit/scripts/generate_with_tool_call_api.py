import json
import os
from PIL import Image
import argparse
from geo_edit.agents.api_agent import APIBasedAgent, AgentConfig
from geo_edit.agents.vllm_agent import VLLMBasedAgent
from geo_edit.environment.action import TOOL_FUNCTIONS
from geo_edit.environment.task.google_vision_qa_task import GoogleVisionQATask
from geo_edit.environment.task.openai_vision_qa_task import OpenAIVisionQATask
from geo_edit.environment.task.vllm_vision_qa_task import VLLMVisionQATask
from geo_edit.config import (
    build_agent_configs,
    build_openai_agent_configs,
    build_vllm_agent_configs,
)
from geo_edit.constants import MAX_TOOL_CALLS, get_system_prompt
from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec
from datasets import load_dataset
import logging
from tqdm import tqdm
import shutil
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    # argparse 
    parser = argparse.ArgumentParser(description="Generate content with tool calls using API models.")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the selected provider.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_SPECS.keys()), help="Dataset adapter name.")
    parser.add_argument("--model_name_or_path", type=str, default="gemini-3-pro-preview", help="Model name or path.")
    parser.add_argument("--model_type", type=str, default="Google", choices=["Google", "OpenAI", "vLLM"], help="Model provider.")
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for vLLM OpenAI-compatible server.")
    parser.add_argument("--port", type=int, default=None, help="Port for vLLM OpenAI-compatible server.")
    parser.add_argument("--max_concurrent_requests", type=int, default=32, help="Maximum number of concurrent requests.")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="Sampling rate for the dataset.")
    parser.add_argument("--use_tools", action=argparse.BooleanOptionalAction, default=True, help="Enable tool calling for the agent.")
    args = parser.parse_args()
    seed=42
    api_key= args.api_key
    if args.model_type in {"Google", "OpenAI"} and not api_key:
        raise ValueError("API key must be provided for Google/OpenAI models.")
    dataset_path= args.dataset_path
    dataset= load_dataset("parquet", data_files=dataset_path)["train"]

    dataset_spec = get_dataset_spec(args.dataset_name)
    max_output_tokens= None

    if args.sample_rate<1.0:
        sample_size= int(len(dataset)* args.sample_rate)
        dataset= dataset.shuffle(seed=seed).select(range(sample_size))
        logger.info(f"Sampled {sample_size} examples from the dataset.")
    
    output_path= args.output_dir
    if not args.use_tools and dataset_spec.notool_prompt_template is None:
        logger.warning("Dataset %s has no no-tool template; using tool template.", dataset_spec.name)
    system_prompt = get_system_prompt(args.model_type) if args.use_tools else ""
    tool_mode = "AUTO" if args.use_tools else "NONE"
    if args.model_type == "Google":
        agent_configs = build_agent_configs(
            max_output_tokens=max_output_tokens,
            thinking_level="low",
            include_thoughts=True,
            temperature=1.0,
            system_prompt=system_prompt,
            candidate_count=1,
            tool_mode=tool_mode,
            disable_automatic_function_calling=True,
        )
    elif args.model_type == "OpenAI":
        agent_configs = build_openai_agent_configs(
            max_output_tokens=max_output_tokens,
            temperature=1.0,
            system_prompt=system_prompt,
            tool_mode=tool_mode,
            reasoning_level="medium",
        )
    else:
        agent_configs = build_vllm_agent_configs(
            max_output_tokens=max_output_tokens,
            temperature=1.0,
            system_prompt=system_prompt,
            tool_mode=tool_mode,
        )

    config = AgentConfig(
        model_type=args.model_type,
        model_name=args.model_name_or_path,
        api_key=api_key,
        api_base=args.api_base,
        port=args.port,
        generate_config=agent_configs.generate_config,
        n_retry=3,
    )
    agent_cls = VLLMBasedAgent if args.model_type == "vLLM" else APIBasedAgent
    api_agent=agent_cls(config)
    
    meta_info_list= []
    
    for item in tqdm(dataset):
        api_agent.reset()
        id= str(item[dataset_spec.id_key])
        if os.path.exists(os.path.join(output_path, id)) and os.path.exists(os.path.join(output_path, id, "meta_info.jsonl")):
            # Add meta info loading
            with open(os.path.join(output_path, id, "meta_info.jsonl"), "r", encoding="utf-8") as f:
                meta_info= json.loads(f.readline().strip())
                meta_info_list.append(meta_info)
            logging.info(f"Example id: {id} already processed, skipping.")
            continue
        logging.info(f"Processing example id: {id}")
        
        task_save_dir= os.path.join(output_path, id)
        os.makedirs(task_save_dir, exist_ok=True)
        answer= item[dataset_spec.answer_key]
        image_url = None
        text_only = dataset_spec.image_key is None
        if dataset_spec.image_key:
            image = item.get(dataset_spec.image_key)
            assert isinstance(image, Image.Image), "Image data must be a PIL Image."
            image_url = os.path.join(task_save_dir, "input_image.png")
            image.save(image_url)

        text_prompt = dataset_spec.build_prompt(item, args.use_tools)
        
        if args.model_type == "Google":
            task_cls = GoogleVisionQATask
        elif args.model_type == "OpenAI":
            task_cls = OpenAIVisionQATask
        else:
            task_cls = VLLMVisionQATask
        task_kwargs = {}
        if text_only:
            task_kwargs["text_only"] = True
        if args.model_type == "vLLM":
            task_kwargs["system_prompt"] = system_prompt
        task= task_cls(
            task_id=id,
            task_prompt=text_prompt,
            task_answer=answer,
            task_image_path=image_url,
            tool_functions=TOOL_FUNCTIONS if args.use_tools else {},
            save_dir=task_save_dir,
            **task_kwargs,
        )
        max_tool_calls = MAX_TOOL_CALLS
        for i in range(max_tool_calls):
            try:
                action, extra_info = api_agent.act(task.contents)
                function_call_part_list = task.parse_action(step=i+1, action=action, extra_info=extra_info)
            except Exception as e:
                task.state = False
                logging.error(f"Error during agent action for example id: {id} at step {i+1}: {e}", exc_info=True)
                break
        
            if not function_call_part_list:
                logging.info("Final response generated without further tool calls.")
                break
            
            task.update_observation_from_action(function_call_part_list)   
        else:
            logging.info("Max tool calls reached; forcing final answer without tool calls.")
            FORCE_ANSWER_PROMPT = "Max tool calls reached. Please provide the final answer without further tool calls."
            task.append_prompt(FORCE_ANSWER_PROMPT)
            original_generate_config = api_agent.config.generate_config
            api_agent.config.generate_config = agent_configs.force_generate_config
            action, extra_info = api_agent.act(task.contents)
            api_agent.config.generate_config = original_generate_config
            
            _ = task.parse_action(step=max_tool_calls + 1, action=action, extra_info=extra_info)

        if task.state:
            meta_info = task.save_trajectory()
            meta_info_list.append(meta_info)
        else:
            continue
    
    total_tool_calls = 0
    total_tokens = 0
    tool_usage_counts = {}
    reach_max_tool_call_count = 0
    direct_answer_count = 0
    for info in meta_info_list:
        total_tool_calls += info["function_call_total_count"]
        total_tokens += info["tokens_used_total"]
        if info["total_steps"] >= MAX_TOOL_CALLS :
            reach_max_tool_call_count += 1
        if info["function_call_total_count"] == 0:
            direct_answer_count += 1
        for tool_name, count in info["function_call_each_count"].items():
            tool_usage_counts[tool_name] = tool_usage_counts.get(tool_name, 0) + count

    global_meta_info = {
        "total_examples": len(meta_info_list),
        "total_tool_calls": total_tool_calls,
        "total_tokens": total_tokens,
        "tool_usage_counts": tool_usage_counts,
        "reach_max_tool_call_count": reach_max_tool_call_count,
        "direct_answer_count": direct_answer_count,
    }
    global_meta_info_jsonl_path = os.path.join(output_path, "global_meta_info.jsonl")
    with open(global_meta_info_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(global_meta_info) + "\n")
            
if __name__ == "__main__":
    main()
