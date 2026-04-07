# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dataset preprocessing for nl2sql task
# Convert the skysql training dataset to the format accepted by verl-tool

import argparse
import logging
import os
import tempfile

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
# Original skysql file path: "/map-vepfs/yi/verltool_paper/sql_experiment/get_skysql/SkyRL-SQL-653-data-newfmt/train.parquet"
DEFAULT_DATABASE_PATH = "./data/synsql/data/SynSQL-2.5M/databases"
# Note: Database files are retrieved as: <database_path>/<db_id>/<db_id>.sqlite

# The system prompt and format of the skysql dataset:
# Original system prompt includes:
# - Task Overview: You are a data science expert
# - Database Engine: SQLite
# - Instructions for SQL query generation
# - Format requirements with <think>, <sql>, <observation>, <solution> blocks
# Example format shows interaction pattern for SQL querying tasks

# Note: skysql dataset has already integrated the system prompt into the prompt column, 
# so we made no modifications during dataset format conversion.
# below ones are only used for visualization purpose.
SKYSQL_SYSTEM_PROMPT = """
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question within limited turns. You should breakdown the problem, draft your reasoning process, and generate the solution.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

External Knowledge:
{external_knowledge}

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query. It should include detailed considerations such as analisying questions, 
summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, thinking of how to call SQL tools, and revisiting previous steps.


Format:
- Conduct thinking inside <think>...</think> blocks every time you get new observation or information. 
- You can use SQL tool written within a single <sql>your sql</sql> block to explore or verify. SQL tool output will be shown as dataframe inside <observation>...</observation>. Based on this observation, you can think again and refine.
- The returned dataframe will be truncated in 50 rows if observation is too long. 
- If you find no further exploration is needed or reaches max turns, you MUST directly provide the final SQL query solution inside <solution>...</solution>. 

------------------------ START OF EXAMPLE ------------------------
Question: how many pigs are in the farm? 
    Database Schema:
    Table: animals
    - id (INTEGER, PRIMARY KEY)
    - species (TEXT)
    - age (INTEGER)
    - name (TEXT)

<think>I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.</think>
<sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</sql>
<observation>
+----------+
| COUNT(*) |
+----------+
|   12     |
+----------+
</observation>
<think>The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.</think>
<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>
------------------------ END OF EXAMPLE ------------------------
"""

SKYSQL_USER_CONTENT = "{db_details}:<db_details> {external_knowledge}: <external_knowledge> ; {question}: <question>"


def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SkySQL format.
    
    Original data transformation logic:
    - verl-tool uses "prompt" to query the model
    - uses "db_path" to access corresponding database 
    - uses "gt_sql" to evaluate the model's output
    
    Original converted_data structure included:
    - data_source: from curr_data["data_source"]
    - prompt: from curr_data["prompt"] 
    - ability: "skysql"
    - reward_model: with ground_truth from curr_data["reward_spec"]["ground_truth"] and style: "rule"
    - extra_info: with db_id, gt_sql, index, question, split, db_path

    Args:
        row: DataFrame row containing the original SkySQL data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    # Extract data from original row structure
    data_source = row.get("data_source", "skysql")
    prompt = row.get("prompt", [])
    db_id = row.get("db_id", "")
    
    
    
    # Extract ground truth from reward_spec
    reward_spec = row.get("reward_spec", {})
    ground_truth = reward_spec.get("ground_truth", "")
    
    # Extract extra_info from original structure
    original_extra_info = row.get("extra_info", {})
    original_index = original_extra_info.get("index", row_index)
    original_split_name = original_extra_info.get("split", current_split_name)
    
    # Build database path following original pattern: {DATABASE_PATH}/{db_id}/{db_id}.sqlite
    db_path = f"{args.database_path}/{db_id}/{db_id}.sqlite"
    
    if original_split_name == current_split_name:
        # Build reward_model structure
        reward_model_data = {
            "ground_truth": ground_truth,
            "style": "rule"
        }
        
        # Build complete extra_info structure following verl-tool requirements
        extra_info = {
            "db_id": db_id,
            "gt_sql": ground_truth,  # Ground truth SQL for evaluation
            "index": original_index,
            "question": prompt,  # Original question/prompt
            "split": current_split_name,
            "db_path": db_path  # Path to SQLite database file
        }
        
        return pd.Series(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "skysql",  # Fixed ability type for SQL tasks
                "reward_model": reward_model_data,
                "extra_info": extra_info,
                "metadata": row.get("metadata"),  # Preserve any existing metadata
            }
        )
    else:
        return None


def main():
    """
    Main processing function following the pattern from search_r1.py
    Downloads, processes, and saves SkySQL dataset files
    """
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    # Process files using temporary directory (following search_r1.py pattern)
    with tempfile.TemporaryDirectory() as tmp_download_dir:
        for split in ["train"]:
            parquet_filename = f"{split}.parquet"
            logger.info(f"Processing {split} split...")

            try:
                # Download or load Parquet file
                if args.hf_repo_id:
                    # Download from HuggingFace if repo_id provided
                    logger.info(f"Downloading {parquet_filename} from {args.hf_repo_id}")
                    local_parquet_filepath = hf_hub_download(
                        repo_id=args.hf_repo_id,
                        filename=parquet_filename,
                        repo_type="dataset",
                        local_dir=tmp_download_dir,
                        local_dir_use_symlinks=False,
                    )
                else:
                    # Use local file path (original: skysql_file_path)
                    local_parquet_filepath = args.local_parquet_path
                    if not os.path.exists(local_parquet_filepath):
                        logger.warning(f"Local file not found: {local_parquet_filepath}")
                        continue

                # Load and process Parquet file
                df_raw = pd.read_parquet(local_parquet_filepath)
                logger.info(f"Loaded {len(df_raw)} rows from {parquet_filename}")

                def apply_process_row(row, split_name=split):
                    return process_single_row(row, current_split_name=split_name, row_index=row.name)

                df_processed = df_raw.apply(apply_process_row, axis=1)

                # Save processed DataFrame
                output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
                df_processed.to_parquet(output_file_path, index=False)
                logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
                processed_files.append(output_file_path)

            except EntryNotFoundError:
                logger.warning(f"{parquet_filename} not found in repository {args.hf_repo_id}")
            except Exception as e:
                logger.error(f"Error processing {split} split: {e}")

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")

    # Copy to HDFS if specified (following search_r1.py pattern)
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SkySQL dataset and convert to verl-tool format.")
    
    # Original file path as default for local processing
    parser.add_argument(
        "--local_parquet_path", 
        default="/map-vepfs/yi/verltool_paper/sql_experiment/get_skysql/SkyRL-SQL-653-data-newfmt/train.parquet",
        help="Local path to SkySQL parquet file (original skysql_file_path)."
    )
    
    # Optional HuggingFace repo for downloading
    parser.add_argument(
        "--hf_repo_id", 
        default=None, 
        help="Optional HuggingFace dataset repository ID for downloading."
    )
    
    parser.add_argument(
        "--local_dir",
        default="./data/skysql_processed",
        help="Local directory to save the processed Parquet files.",
    )
    
    # Database path configuration
    parser.add_argument(
        "--database_path",
        default=DEFAULT_DATABASE_PATH,
        help="Path to database files directory (original DATABASE_PATH).",
    )
    
    parser.add_argument(
        "--hdfs_dir", 
        default=None, 
        help="Optional HDFS directory to copy the Parquet files to."
    )

    args = parser.parse_args()

    main()

# Example usage:    
# python examples/data_preprocess/skysql_train.py --local_parquet_path "/map-vepfs/yi/verltool_paper/sql_experiment/get_skysql/SkyRL-SQL-653-data-newfmt/train.parquet"