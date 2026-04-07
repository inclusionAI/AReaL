# Dataset preprocessing for Spider test datasets (dev and test splits)
# Convert the Spider dev/test datasets to the format accepted by verl-tool

import argparse
import json
import logging
import os
import re
import tempfile
from tqdm import tqdm
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# These should remain relative to the parent directory of verl-tool as they will be hard-coded to the datasets
DEFAULT_DATABASE_PATH = "./data/synsql/data/spider/database"
DEFAULT_DATABASE_PATH_TEST = "./data/synsql/data/spider/test_database"
DEFAULT_DATABASE_PATH_DK = "./data/synsql/data/Spider-DK/database"
DEFAULT_DATABASE_PATH_BIRD = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/bird/dev_20240627/dev_databases"

# These should be absolute paths pointing to specific local files when preprocessing the datasets
DEFAULT_SPIDER_DEV_PROMPT = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/dev_spider.json"
DEFAULT_SPIDER_DEV_SCHEMA = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/spider/dev_gold.sql"

DEFAULT_SPIDER_TEST_PROMPT = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/test_spider.json"
DEFAULT_SPIDER_TEST_SCHEMA = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/spider/test_gold.sql"

DEFAULT_SPIDER_DK_PROMPT = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/dev_spider_dk.json"
DEFAULT_SPIDER_DK_SCHEMA = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/Spider-DK/spider_dk_gold.sql"

DEFAULT_SPIDER_SYN_PROMPT = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/dev_spider_syn.json"
DEFAULT_SPIDER_SYN_SCHEMA = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/Spider-Syn/spider_syn_gold.sql"

DEFAULT_SPIDER_REALISTIC_PROMPT = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/dev_spider_realistic.json"
DEFAULT_SPIDER_REALISTIC_SCHEMA = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/spider-realistic/spider_realistic_gold.sql"

DEFAULT_BIRD_DEV_PROMPT = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/dev_bird.json"
DEFAULT_BIRD_DEV_SCHEMA = "/map-vepfs/yi/verltool_paper/sql_experiment/verl-tool/data/synsql/data/bird/dev_20240627/dev.sql"

# System prompt for SQL generation tasks
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

SKYSQL_USER_CONTENT = "{db_details}:<db_details> ;\n {external_knowledge}: <external_knowledge> ;\n {question}: <question>"


def extract_schema_and_question(text):
    """
    Extract Database Schema and Question from the given text format.
    
    Args:
        text (str): The input text containing the schema and question
    
    Returns:
        tuple: (database_schema, question) or (None, None) if not found
    """
    
    # Regex pattern to extract Database Schema
    # Looks for "Database Schema:" followed by any content until "Question:"
    schema_pattern = r'Database Schema:\s*\n(.*?)(?=\n\s*This schema)'
    
    # Regex pattern to extract Question
    # Looks for "Question:" followed by any content until "Instructions:"
    question_pattern = r'Question:\s*\n(.*?)(?=\n\s*Instructions:)'
    
    # Extract schema
    schema_match = re.search(schema_pattern, text, re.DOTALL)
    database_schema = schema_match.group(1).strip() if schema_match else None
    
    # Extract question
    question_match = re.search(question_pattern, text, re.DOTALL)
    question = question_match.group(1).strip() if question_match else None
    
    return database_schema, question


def process_prompt(raw_prompt):
    """
    Process raw prompt to extract schema and question, then format for user content.
    
    Args:
        raw_prompt (str): Raw input prompt
        
    Returns:
        str: Formatted user content
    """
    database_schema, question = extract_schema_and_question(raw_prompt)
    
    user_prompt = SKYSQL_USER_CONTENT.replace("<db_details>", database_schema).replace("<external_knowledge>", "").replace("<question>", question)
    return user_prompt


def process_spider_dataset(dataset_name, prompt_file, schema_file, database_path):
    """
    Process a Spider dataset (dev or test) and convert to verl-tool format.
    
    Args:
        dataset_name (str): Name of the dataset (spider_dev or spider_test)
        prompt_file (str): Path to JSON file with prompts
        schema_file (str): Path to SQL file with schema info
        database_path (str): Path to database directory
        
    Returns:
        list: List of processed pandas Series objects
    """
    logger.info(f"Processing {dataset_name} dataset...")
    
    # Read input files
    with open(prompt_file, 'r') as f:
        prompt_data = json.load(f)
    with open(schema_file, 'r') as f:
        schema_lines = f.read().strip().split('\n')
    
    # Validate data consistency
    if len(prompt_data) != len(schema_lines):
        logger.warning(f"Mismatch in data length: {len(prompt_data)} prompts vs {len(schema_lines)} schema lines")
        min_length = min(len(prompt_data), len(schema_lines))
        prompt_data = prompt_data[:min_length]
        schema_lines = schema_lines[:min_length]
    
    processed_data = []
    
    for i in tqdm(range(len(prompt_data)), desc=f"Processing {dataset_name}", unit="row"):
    # for i in tqdm(range(10), desc=f"Processing {dataset_name}", unit="row"):  # just for testing
    
        try:
            raw_prompt = prompt_data[i]["input_seq"]
            gt_sql = prompt_data[i]["output_seq"]
            processed_user_prompt = process_prompt(raw_prompt)
            
            # Build conversation prompt
            prompt = [
                {
                    "role": "system",
                    "content": SKYSQL_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": processed_user_prompt
                }
            ]
            
            # Extract database ID from schema line
            db_id = schema_lines[i].split("\t")[-1].strip()
            
            # Build database path following original pattern: {DATABASE_PATH}/{db_id}/{db_id}.sqlite
            db_path = f"{database_path}/{db_id}/{db_id}.sqlite"
            
            # check if the db actually exists
            if not os.path.exists(db_path):
                if dataset_name == "spider_test" or dataset_name == "spider_syn":
                    # check if the db exists in the test_database path
                    db_path = f"{DEFAULT_DATABASE_PATH_TEST}/{db_id}/{db_id}.sqlite"
                    if not os.path.exists(db_path):
                        raise ValueError(f"Database file does not exist even in test_database path: {db_path}")
                else:
                    raise ValueError(f"Database file does not exist in original path: {db_path}")

            # Create processed row
            curr_row_pd = pd.Series(
                {
                    "data_source": dataset_name,
                    "prompt": prompt,
                    "ability": "nl2sql",
                    "extra_info": {
                        "db_id": db_id,
                        "gt_sql": gt_sql,
                        "index": i,
                        "question": processed_user_prompt,
                        "split": "test",
                        "db_path": db_path,
                    }
                }
            )
            
            processed_data.append(curr_row_pd)
            
        except Exception as e:
            logger.error(f"Error processing row {i} in {dataset_name}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(processed_data)} rows for {dataset_name}")
    return processed_data



def main():
    """
    Main processing function that processes Spider dev and test datasets
    and saves them in verl-tool format.
    """
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []
    all_processed_data = []
    global_index = 0

    try:
        # Process Spider dev dataset
        if args.process_dev:
            logger.info("Processing Spider dev dataset...")
            spider_dev_data = process_spider_dataset(
                dataset_name="spider_dev", 
                prompt_file=args.spider_dev_prompt,
                schema_file=args.spider_dev_schema,
                database_path=args.database_path
            )
            
            if spider_dev_data:
                # Update indices
                for row in spider_dev_data:
                    row['extra_info']['index'] = global_index
                    global_index += 1
                all_processed_data.extend(spider_dev_data)

                if not args.merge_all:
                    # Convert to DataFrame and save separately
                    df_dev = pd.DataFrame(spider_dev_data)
                    dev_output_path = os.path.join(local_save_dir, "spider_dev.parquet")
                    df_dev.to_parquet(dev_output_path, index=False)
                    logger.info(f"Saved {len(df_dev)} Spider dev rows to {dev_output_path}")
                    processed_files.append(dev_output_path)

        # Process Spider test dataset
        if args.process_test:
            logger.info("Processing Spider test dataset...")
            spider_test_data = process_spider_dataset(
                dataset_name="spider_test",
                prompt_file=args.spider_test_prompt,
                schema_file=args.spider_test_schema,
                database_path=args.database_path
            )
            
            if spider_test_data:
                # Update indices
                for row in spider_test_data:
                    row['extra_info']['index'] = global_index
                    global_index += 1
                all_processed_data.extend(spider_test_data)

                if not args.merge_all:
                    # Convert to DataFrame and save separately
                    df_test = pd.DataFrame(spider_test_data)
                    test_output_path = os.path.join(local_save_dir, "spider_test.parquet")
                    df_test.to_parquet(test_output_path, index=False)
                    logger.info(f"Saved {len(df_test)} Spider test rows to {test_output_path}")
                    processed_files.append(test_output_path)

        # Process Spider DK dataset
        if args.process_dk:
            logger.info("Processing Spider DK dataset...")
            spider_dk_data = process_spider_dataset(
                dataset_name="spider_dk",
                prompt_file=args.spider_dk_prompt,
                schema_file=args.spider_dk_schema,
                database_path=args.spider_dk_database_path
            )
            
            if spider_dk_data:
                # Update indices
                for row in spider_dk_data:
                    row['extra_info']['index'] = global_index
                    global_index += 1
                all_processed_data.extend(spider_dk_data)

                if not args.merge_all:
                    # Convert to DataFrame and save separately
                    df_dk = pd.DataFrame(spider_dk_data)
                    dk_output_path = os.path.join(local_save_dir, "spider_dk.parquet")
                    df_dk.to_parquet(dk_output_path, index=False)
                    logger.info(f"Saved {len(df_dk)} Spider DK rows to {dk_output_path}")
                    processed_files.append(dk_output_path)
        
        # Process Spider Syn dataset
        if args.process_syn:
            logger.info("Processing Spider Syn dataset...")
            spider_syn_data = process_spider_dataset(
                dataset_name="spider_syn",
                prompt_file=args.spider_syn_prompt,
                schema_file=args.spider_syn_schema,
                database_path=args.database_path
            )
            
            if spider_syn_data:
                # Update indices
                for row in spider_syn_data:
                    row['extra_info']['index'] = global_index
                    global_index += 1
                all_processed_data.extend(spider_syn_data)

                if not args.merge_all:
                    # Convert to DataFrame and save separately
                    df_syn = pd.DataFrame(spider_syn_data)
                    syn_output_path = os.path.join(local_save_dir, "spider_syn.parquet")
                    df_syn.to_parquet(syn_output_path, index=False)
                    logger.info(f"Saved {len(df_syn)} Spider Syn rows to {syn_output_path}")
                    processed_files.append(syn_output_path)
                    
        # Process Spider Realistic dataset
        if args.process_realistic:
            logger.info("Processing Spider Realistic dataset...")
            spider_realistic_data = process_spider_dataset(
                dataset_name="spider_realistic",
                prompt_file=args.spider_realistic_prompt,
                schema_file=args.spider_realistic_schema,   
                database_path=args.database_path
            )
            
            if spider_realistic_data:
                # Update indices
                for row in spider_realistic_data:
                    row['extra_info']['index'] = global_index
                    global_index += 1
                all_processed_data.extend(spider_realistic_data)

                if not args.merge_all:
                    # Convert to DataFrame and save separately
                    df_realistic = pd.DataFrame(spider_realistic_data)
                    realistic_output_path = os.path.join(local_save_dir, "spider_realistic.parquet")
                    df_realistic.to_parquet(realistic_output_path, index=False)
                    logger.info(f"Saved {len(df_realistic)} Spider Realistic rows to {realistic_output_path}")
                    processed_files.append(realistic_output_path)

        # Process Bird dev dataset
        if args.process_bird:
            logger.info("Processing Bird dev dataset...")
            bird_dev_data = process_spider_dataset(
                dataset_name="bird_dev",
                prompt_file=args.bird_dev_prompt,
                schema_file=args.bird_dev_schema,
                database_path=args.bird_dev_database_path
            )
            
            if bird_dev_data:
                # Update indices
                for row in bird_dev_data:
                    row['extra_info']['index'] = global_index
                    global_index += 1
                all_processed_data.extend(bird_dev_data)

                if not args.merge_all:
                    # Convert to DataFrame and save separately
                    df_bird = pd.DataFrame(bird_dev_data)
                    bird_output_path = os.path.join(local_save_dir, "bird_dev.parquet")
                    df_bird.to_parquet(bird_output_path, index=False)
                    logger.info(f"Saved {len(df_bird)} Bird dev rows to {bird_output_path}")
                    processed_files.append(bird_output_path)

                
                
        # Save all data into one file if specified
        if args.merge_all and all_processed_data:
            df_all = pd.DataFrame(all_processed_data)
            all_output_path = os.path.join(local_save_dir, "test.parquet")
            df_all.to_parquet(all_output_path, index=False)
            logger.info(f"Saved {len(df_all)} total rows to {all_output_path}")
            processed_files = [all_output_path]

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Spider dev/test datasets and convert to verl-tool format.")
    
    # Dataset selection
    parser.add_argument(
        "--process_dev", 
        action="store_true",
        default=False,
        help="Process Spider dev dataset."
    )
    
    parser.add_argument(
        "--process_test", 
        action="store_true",
        default=False,
        help="Process Spider test dataset."
    )
    
    parser.add_argument(
        "--process_dk", 
        action="store_true",
        default=False,
        help="Process Spider DK dataset."
    )
    
    parser.add_argument(
        "--process_syn", 
        action="store_true",
        default=False,
        help="Process Spider Syn dataset."
    )
    
    parser.add_argument(
        "--process_realistic", 
        action="store_true",
        default=False,
        help="Process Spider Realistic dataset."
    )
    
    parser.add_argument(
        "--process_bird", 
        action="store_true",
        default=False,
        help="Process Bird dataset."
    )
    
    
    # param: do we merge all 4 datasets into one?
    parser.add_argument(
        "--merge_all",
        action="store_true",
        default=True,
        help="Merge all 6 datasets into one."
    )
    
    # Output configuration
    parser.add_argument(
        "--local_dir",
        default="./data/spider_processed",
        help="Local directory to save the processed Parquet files.",
    )
    
    parser.add_argument(
        "--hdfs_dir", 
        default=None, 
        help="Optional HDFS directory to copy the Parquet files to."
    )

    # prompt json files for each dataset
    parser.add_argument(
        "--spider_dev_prompt",
        default=DEFAULT_SPIDER_DEV_PROMPT,
        help="Path to Spider dev JSON file with prompts."
    )
    parser.add_argument(
        "--spider_test_prompt",
        default=DEFAULT_SPIDER_TEST_PROMPT,
        help="Path to Spider test JSON file with prompts."
    )
    parser.add_argument(
        "--spider_dk_prompt",
        default=DEFAULT_SPIDER_DK_PROMPT,
        help="Path to Spider DK JSON file with prompts."
    )
    parser.add_argument(
        "--spider_syn_prompt",
        default=DEFAULT_SPIDER_SYN_PROMPT,
        help="Path to Spider Syn JSON file with prompts."
    )
    parser.add_argument(
        "--spider_realistic_prompt",
        default=DEFAULT_SPIDER_REALISTIC_PROMPT,
        help="Path to Spider Realistic JSON file with prompts."
    )
    parser.add_argument(
        "--bird_dev_prompt",
        default=DEFAULT_BIRD_DEV_PROMPT,
        help="Path to Bird dev JSON file with prompts."
    )
    
    
    # sql files for each dataset
    parser.add_argument(
        "--spider_dev_schema",
        default=DEFAULT_SPIDER_DEV_SCHEMA,
        help="Path to Spider dev SQL file with schema info."
    )
    parser.add_argument(
        "--spider_test_schema",
        default=DEFAULT_SPIDER_TEST_SCHEMA,
        help="Path to Spider test SQL file with schema info."
    )
    parser.add_argument(
        "--spider_dk_schema",
        default=DEFAULT_SPIDER_DK_SCHEMA,
        help="Path to Spider DK SQL file with schema info."
    )
    parser.add_argument(
        "--spider_syn_schema",
        default=DEFAULT_SPIDER_SYN_SCHEMA,
        help="Path to Spider Syn SQL file with schema info."
    )
    parser.add_argument(
        "--spider_realistic_schema",
        default=DEFAULT_SPIDER_REALISTIC_SCHEMA,
        help="Path to Spider Realistic SQL file with schema info."
    )
    parser.add_argument(
        "--bird_dev_schema",
        default=DEFAULT_BIRD_DEV_SCHEMA,
        help="Path to Bird dev SQL file with schema info."
    )
    
    # database paths for each dataset
    parser.add_argument(
        "--database_path",
        default=DEFAULT_DATABASE_PATH,
        help="Path to Spider dev database files directory."
    )
    parser.add_argument(
        "--spider_dk_database_path",
        default=DEFAULT_DATABASE_PATH_DK,
        help="Path to Spider DK database files directory."
    )
    parser.add_argument(
        "--bird_dev_database_path",
        default=DEFAULT_DATABASE_PATH_BIRD,
        help="Path to Bird dev database files directory."
    )
    
    
    args = parser.parse_args()

    main()

# Example usage for processing all datasets
# python ./prepare_test_dataset.py --process_dev --process_test --process_dk --process_syn --process_realistic --process_bird --local_dir "./processed_spider_datasets"

# in reality nobody is using the realistic dataset so skip it
# python ./prepare_test_dataset.py --process_dev --process_test --process_dk --process_syn --process_bird --local_dir "./processed_spider_datasets"

# python ./prepare_test_dataset.py --process_syn --local_dir "./processed_spider_datasets"

