import pandas as pd
import json
import sys

def parquet_to_jsonl(parquet_path, jsonl_path):
    """Convert parquet file to JSONL format with Problem and CoT keys"""
    df = pd.read_parquet(parquet_path)
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Extract problem and generated_solution directly from row
            problem = row.get("problem", "")
            answer = row.get("solution", "")
            # Create the output format with Problem and CoT keys
            output_record = {
                "problem": problem,
                "answer": answer,
            }
            
            f.write(json.dumps(output_record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parquet_to_jsonl.py input.parquet output.jsonl")
        sys.exit(1)
    
    try:
        parquet_to_jsonl(sys.argv[1], sys.argv[2])
        print(f"Successfully converted {sys.argv[1]} to {sys.argv[2]}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)