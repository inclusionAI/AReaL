import fire
import datasets
from pathlib import Path
def main(
    dataset_path="VerlTool/deepsearch",
    output_directory="data/deepsearch",
):
    """Main entry point for the dataset loading script
    
    Run with:
        python examples/data_preprocess/deepsearch.py --dataset_path=VerlTool/deepsearch
    """
    # Load the dataset
    dataset = datasets.load_dataset(dataset_path)
    output_directory = Path(output_directory)
    dataset['train'].to_parquet(output_directory / "hard_search_1k.parquet")
    dataset['test_hle'].to_parquet(output_directory / "hle_test.parquet")
    dataset['test_gaia'].to_parquet(output_directory / "gaia_test.parquet")
    dataset['test_webwalker'].to_parquet(output_directory / "webwalker_test.parquet")
    dataset['test_xbench'].to_parquet(output_directory / "xbench_test.parquet")

if __name__ == "__main__":
    fire.Fire(main)