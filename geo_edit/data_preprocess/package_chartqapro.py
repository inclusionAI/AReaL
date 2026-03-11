"""Package ChartQAPro dataset to parquet format.

Downloads the dataset from HuggingFace, flattens Question/Answer lists, and saves as parquet.

ChartQAPro contains:
- Question: List[str] (multiple questions per chart, e.g., conversational)
- Answer: List[str] (corresponding answers)
- Question Type: str (Factoid, Multi Choice, Conversational, Hypothetical, Fact Checking)
- image: binary (chart image)
- Year: List[str] (year relevance for each question, "NO" if not applicable)
- Paragraph: str

Usage:
    python geo_edit/data_preprocess/package_chartqapro.py --out_dir /path/to/output
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset


def package_chartqapro(out_dir: Path) -> Path:
    """Package ChartQAPro dataset into parquet format.

    Each chart may have multiple questions (e.g., conversational). This function
    flattens them so each row contains a single question-answer pair with its
    corresponding image.
    """
    out_parquet = out_dir / "chartqapro_test_dataset.parquet"

    print("Loading ChartQAPro dataset from HuggingFace...")
    ds = load_dataset("ahmed-masry/ChartQAPro")["test"]

    print(f"Original dataset size: {len(ds)} charts")

    # Flatten the dataset: expand Question/Answer lists into individual rows
    flattened_rows = []
    global_id = 0

    for chart_idx, item in enumerate(ds):
        questions = item.get("Question", [])
        answers = item.get("Answer", [])
        question_type = item.get("Question Type", "")
        image = item.get("image")
        years = item.get("Year", [])
        paragraph = item.get("Paragraph", "")

        # Each question-answer pair becomes a separate row
        for q_idx, (question, answer) in enumerate(zip(questions, answers)):
            year = years[q_idx] if q_idx < len(years) else ""
            flattened_rows.append(
                {
                    "id": global_id,
                    "chart_id": chart_idx,
                    "question_idx": q_idx,
                    "question": question,
                    "answer": answer,
                    "question_type": question_type,
                    "year": year,
                    "paragraph": paragraph,
                    "image": image,
                }
            )
            global_id += 1

    # Create new dataset from flattened rows
    flattened_ds = Dataset.from_list(flattened_rows)

    # Save to parquet
    flattened_ds.to_parquet(str(out_parquet))
    print(f"Saved parquet: {out_parquet}")
    print(f"Total question-answer pairs: {len(flattened_ds)}")

    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Package ChartQAPro dataset to parquet.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to current directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_chartqapro(out_dir)


if __name__ == "__main__":
    main()
