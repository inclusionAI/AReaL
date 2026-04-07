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

import logging
import fire
import datasets
import json
import pandas as pd
from pathlib import Path
from functools import partial

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

test_databases_source_sub_dir_map = {
    "spider_dev": "spider/test_database",
    "spider_test": "spider/test_database",
    "spider_dk": "Spider-DK/database",
    "spider_syn": "spider/test_database",
    "spider_realistic": "spider/test_database",
}
train_databases_source_sub_dir_map = {
    "synsql": "synsql",
    "spider": "spider/database",
}

def main(
    dataset_path: str = "VerlTool/SkyRL-SQL-Reproduction",
    train_databases_dir: str = "./data/skysql/databases",
    test_databases_dir: str = "./data/skysql/databases",
    save_dir: str = "./data/skysql",
):
    
    dataset = datasets.load_dataset(dataset_path)
    train_databases_dir = Path(train_databases_dir)
    test_databases_dir = Path(test_databases_dir)
    assert train_databases_dir.exists(), f"Train databases directory {train_databases_dir} does not exist."
    assert test_databases_dir.exists(), f"Test databases directory {test_databases_dir} does not exist."
    
    def process_item(item, split):
        """
        Process a single item from the dataset to match verl-tool format.
        
        Args:
            item: A single item from the dataset.
        
        Returns:
            dict: Processed item in the required format.
        """
        db_id = item['extra_info'].get("db_id", "")
        data_source = item.get("data_source", "skysql")
        if split == 'train':
            # db_path = train_databases_dir / db_id / f"{db_id}.sqlite"
            db_path = train_databases_dir / train_databases_source_sub_dir_map[data_source] / db_id / f"{db_id}.sqlite"
            assert db_path.exists(), f"Database file {db_path} does not exist for db_id {db_id} in split {split}, data_source {data_source}."
        else:
            db_path = test_databases_dir / test_databases_source_sub_dir_map[data_source] / db_id / f"{db_id}.sqlite"
        assert db_path.exists(), f"Database file {db_path} does not exist for db_id {db_id} in split {split}, data_source {data_source}."
        
        # item['extra_info']['db_path'] = str(db_path.relative_to("."))  # Make db_path relative to current directory
        item['extra_info']['db_path'] = str(db_path.resolve())  # Use absolute path
        question = item['prompt'][-1]['content']  # Extract question from the last prompt entry
        item['extra_info']['question'] = question 
        
        # if you want new system prompt, then add here, by default the prompt is the same as the original one
        item['prompt'] = [
            {"role": "system", "content": SKYSQL_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        return item
        
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset = train_dataset.map(
        partial(process_item, split='train'),
        num_proc=4,
    )
    test_dataset = test_dataset.map(
        partial(process_item, split='test'),
        num_proc=4
    )
    print("Processed train dataset:", train_dataset)
    print("Processed test dataset:", test_dataset)
    # Save the processed datasets
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    train_dataset.to_parquet(save_path / "train.parquet")
    test_dataset.to_parquet(save_path / "test.parquet")
    all_dataset = datasets.DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    all_dataset.push_to_hub("VerlTool/SkyRL-SQL-Reproduction")
    print(f"Processed datasets saved to {save_path}")
    with open(save_path / "train.json", "w") as f:
        json.dump([x for x in train_dataset], f, indent=4, ensure_ascii=False)
    
    all_test_data_sources = test_dataset.unique("data_source")
    for data_source in all_test_data_sources:
        data_source_test_dataset = test_dataset.filter(lambda x: x["data_source"] == data_source)
        data_source_save_path = save_path / f"test_{data_source}.parquet"
        data_source_test_dataset.to_parquet(data_source_save_path)
        print(f"Processed {data_source} test dataset saved to {data_source_save_path}")
    
if __name__ == "__main__":
    fire.Fire(main)