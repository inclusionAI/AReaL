import json
import os
from PIL import Image
import argparse
from ..agents.api_agent import APIBasedAgent, AgentConfig
from ..environment.action import TOOL_FUNCTIONS
from ..environment.task.google_vision_qa_task import GoogleVisionQATask
from ..environment.task.openai_vision_qa_task import OpenAIVisionQATask
from ..config import (
    MATHVISION_INPUT_TEMPLATE,
    NOTOOL_INPUT_TEMPLATE,
    build_agent_configs,
    build_openai_agent_configs,
)
from ..constants import SYSTEM_PROMPT, MAX_TOOL_CALLS
from datasets import load_dataset
import logging
from tqdm import tqdm
import shutil
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    # argparse 
    parser = argparse.ArgumentParser(description="Generate content with tool calls using API models.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the selected provider.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--model_name_or_path", type=str, default="gemini-3-pro-preview", help="Model name or path.")
    parser.add_argument("--model_type", type=str, default="Google", choices=["Google", "OpenAI"], help="Model provider.")
    parser.add_argument("--max_concurrent_requests", type=int, default=32, help="Maximum number of concurrent requests.")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="Sampling rate for the dataset.")
    args = parser.parse_args()
    seed=42
    api_key= args.api_key
    dataset_path= args.dataset_path
    dataset= load_dataset("parquet", data_files=dataset_path)["train"]

    # filter out examples without image_preview
    dataset= dataset.filter(lambda x: x["image_preview"] is not None)
    print(f"Dataset size after filtering: {len(dataset)}")
    max_output_tokens= None

    if args.sample_rate<1.0:
        sample_size= int(len(dataset)* args.sample_rate)
        dataset= dataset.shuffle(seed=seed).select(range(sample_size))
        logger.info(f"Sampled {sample_size} examples from the dataset.")
    
    output_path= args.output_dir
    if args.model_type == "Google":
        agent_configs = build_agent_configs(
            max_output_tokens=max_output_tokens,
            thinking_level="low",
            include_thoughts=True,
            temperature=1.0,
            system_prompt=SYSTEM_PROMPT,
            candidate_count=1,
            tool_mode="AUTO",
            disable_automatic_function_calling=True,
        )
    else:
        agent_configs = build_openai_agent_configs(
            max_output_tokens=max_output_tokens,
            temperature=1.0,
            system_prompt=SYSTEM_PROMPT,
            tool_mode="AUTO",
            reasoning_level="medium",
        )

    config = AgentConfig(
        model_type=args.model_type,
        model_name=args.model_name_or_path,
        api_key=api_key,
        generate_config=agent_configs.generate_config,
        n_retry=3,
    )
    api_agent=APIBasedAgent(config)
    
    meta_info_list= []
    
    INPUT_TEMPLATE= MATHVISION_INPUT_TEMPLATE
    # INPUT_TEMPLATE= NOTOOL_INPUT_TEMPLATE
    for item in tqdm(dataset):
        api_agent.reset()
        id= item["id"]
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
        question= item["question"]
        options= item.get("options", "")
        answer= item["answer"]
        if isinstance(item["image_preview"], Image.Image):
            image= item["image_preview"]
            image_url= os.path.join(task_save_dir, "input_image.png")
            image.save(image_url)
        else:
            image_url= item["image_preview"]
        
        text_prompt= INPUT_TEMPLATE.format(question=question, options=options)
        
        task_cls = (
            GoogleVisionQATask if args.model_type == "Google" else OpenAIVisionQATask
        )
        task= task_cls(
            task_id=id,
            task_prompt=text_prompt,
            task_answer=answer,
            task_image_path=image_url,
            tool_functions=TOOL_FUNCTIONS,
            save_dir=task_save_dir,
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
            if args.model_type == "Google":
                task.contents.append(FORCE_ANSWER_PROMPT)
            else:
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
