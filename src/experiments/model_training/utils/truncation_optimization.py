#!/usr/bin/env python3
"""
Token distribution analysis for truncation optimization in model training.
This script analyzes token distributions in a dataset to help decide optimal truncation limits.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.model_training.dataset_prep.prepare_tau_dataset import (
    load_as_hf_dataset,
)
from experiments.model_training.sft_trl.train_sft_trl import get_tokenizer
from tau2.utils.utils import DATA_DIR


def analyze_tokens(example):
    """Count tokens and assistant token positions in an example."""
    return {
        "num_tokens": len(example["input_ids"]),
        "num_system_tokens": len(example["system_input_ids"]),
        "num_assistant_tokens": sum(example["assistant_masks"]),
        "assistant_token_positions": [
            i for i, mask in enumerate(example["assistant_masks"]) if mask
        ],
        "first_assistant_token_index": next(
            (i for i, mask in enumerate(example["assistant_masks"]) if mask), None
        ),
        "last_assistant_token_index": len(example["assistant_masks"])
        - next(
            (i for i, mask in enumerate(example["assistant_masks"][::-1]) if mask), None
        ),
    }


def count_messages(example):
    """Count messages in an example."""
    return {
        "num_messages": len(example["messages"]),
    }


def remove_system_messages_from_example(example):
    """Remove system messages from an example's messages."""
    filtered_messages = [msg for msg in example["messages"] if msg["role"] != "system"]
    return {"messages": filtered_messages}


def extract_system_messages(example):
    """Extract system messages from an example's messages."""
    return {
        "system_messages": [
            msg for msg in example["messages"] if msg["role"] == "system"
        ]
    }


# TODO: Generalize this so that it can be used with any conversational SFT HF Dataset in OpenaAI format.
def analyze_token_distribution(
    dataset_path,
    model_name,
    chat_template_path=None,
    num_samples=None,
    remove_system_messages=False,
):
    """
    Analyze token distribution in a dataset for truncation optimization.

    Args:
        dataset_path (str): Path to the dataset file
        model_name (str): Name of the model for tokenization
        chat_template_path (str, optional): Path to chat template file
        num_samples (int, optional): Number of examples to sample from dataset
        remove_system_messages (bool, optional): Whether to remove system messages from the dataset

    Returns:
        pandas.DataFrame: DataFrame with token statistics
    """
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_as_hf_dataset(dataset_path)
    print(f"Dataset loaded with {len(dataset)} examples")

    # Sample dataset if specified
    if num_samples and num_samples < len(dataset):
        print(f"Sampling {num_samples} examples from dataset")
        dataset = dataset.select(range(num_samples))

    dataset = dataset.map(extract_system_messages)

    if remove_system_messages:
        print("Removing system messages from examples...")
        dataset = dataset.map(remove_system_messages_from_example)

    # Count messages
    print("Counting messages...")
    dataset = dataset.map(count_messages)

    print("Applying tokenizer...")
    tokenizer = get_tokenizer(model_name, chat_template_path=chat_template_path)
    print("Tokenization complete")

    # Tokenize with assistant mask for assistant only loss
    print("Tokenizing dataset with assistant masks...")
    dataset = dataset.map(
        lambda x: tokenizer.apply_chat_template(
            conversation=x["messages"],
            tools=x["tools"],
            return_dict=True,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
        )
    )

    # Tokenize system messages only
    print("Tokenizing system messages only...")
    dataset = dataset.map(
        lambda x: {
            **x,
            **{
                f"system_{k}": v
                for k, v in tokenizer.apply_chat_template(
                    conversation=x["system_messages"],
                    tools=x["tools"],
                    return_dict=True,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_assistant_tokens_mask=True,
                ).items()
            },
        }
    )

    # Count tokens
    print("Counting tokens...")
    dataset = dataset.map(analyze_tokens)

    # Convert to pandas DataFrame
    print("Creating pandas DataFrame...")
    df = pd.DataFrame(
        [
            {
                "num_tokens": example["num_tokens"],
                "num_system_tokens": example["num_system_tokens"],
                "num_convo_tokens": example["num_tokens"]
                - example["num_system_tokens"],
                "num_assistant_tokens": example["num_assistant_tokens"],
                "first_assistant_token_index": example["first_assistant_token_index"],
                "last_assistant_token_index": example["last_assistant_token_index"],
                "assistant_token_positions": example["assistant_token_positions"],
                "num_messages": example["num_messages"],
            }
            for example in dataset
        ]
    )

    return df


def create_assistant_token_position_distribution(df) -> pd.DataFrame:
    """
    Create a distribution of assistant token positions
    """

    # Count the number of assistant tokens at each position
    assistant_token_positions = df["assistant_token_positions"].explode()
    assistant_token_position_counts = Counter(assistant_token_positions)

    return pd.DataFrame(
        assistant_token_position_counts.items(), columns=["position", "count"]
    )


def create_visualizations(df, output_path: str | Path):
    """
    Create visualization plots for token distribution analysis.

    Args:
        df (pandas.DataFrame): DataFrame with token statistics
        output_path (str): Path to save the visualization
    """
    print("Creating visualization plots...")

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Token Distribution Analysis for Truncation Optimization",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Create a GridSpec layout:
    # Top row: 2 larger plots (Total tokens and Assistant token loss)
    # Bottom row: 3 smaller plots (System, Assistant, Conversation tokens)
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.4)

    # Top row - spans 3 columns each
    ax1 = fig.add_subplot(gs[0, :3])  # Total tokens (left half)
    ax2 = fig.add_subplot(gs[0, 3:])  # Assistant token loss (right half)

    # Bottom row - spans 2 columns each
    ax3 = fig.add_subplot(gs[1, :2])  # System tokens
    ax4 = fig.add_subplot(gs[1, 2:4])  # Assistant tokens
    ax5 = fig.add_subplot(gs[1, 4:])  # Conversation tokens

    # Plot 1: Distribution of Total Tokens (top left)
    sns.histplot(data=df, x="num_tokens", bins=20, ax=ax1, kde=True, color="purple")
    ax1.set_title("Distribution of Total Tokens", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Tokens")
    ax1.set_ylabel("Count")
    mean_total = df["num_tokens"].mean()
    std_total = df["num_tokens"].std()
    ax1.axvline(
        mean_total,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_total:.1f}",
    )
    ax1.axvline(
        mean_total + std_total,
        color="orange",
        linestyle=":",
        label=f"+1σ: {mean_total + std_total:.1f}",
    )
    ax1.axvline(
        mean_total - std_total,
        color="orange",
        linestyle=":",
        label=f"-1σ: {mean_total - std_total:.1f}",
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Plot 2: Assistant Token Loss by Truncation Position (top right)
    assistant_pos_df = create_assistant_token_position_distribution(df)

    if not assistant_pos_df.empty:
        # Sort by position to ensure correct cumulative sum
        assistant_pos_df = assistant_pos_df.sort_values("position")

        # Calculate cumulative sum from the end
        assistant_pos_df["cumulative_tokens_lost"] = assistant_pos_df.loc[
            ::-1, "count"
        ].cumsum()[::-1]

        # Calculate total assistant tokens for percentage calculation
        total_assistant_tokens = assistant_pos_df["count"].sum()
        assistant_pos_df["percentage_tokens_lost"] = (
            assistant_pos_df["cumulative_tokens_lost"] / total_assistant_tokens
        ) * 100

        # Create dual-axis plot
        ax2_twin = ax2.twinx()

        # Plot number of tokens lost on primary y-axis
        ax2.plot(
            assistant_pos_df["position"],
            assistant_pos_df["cumulative_tokens_lost"],
            linewidth=2.5,
            color="darkred",
            label="Tokens Lost",
        )
        ax2.set_ylabel(
            "Number of Assistant Tokens Lost", color="darkred", fontweight="bold"
        )
        ax2.tick_params(axis="y", labelcolor="darkred")
        ax2.grid(True, alpha=0.2)

        # Plot percentage on secondary y-axis
        ax2_twin.plot(
            assistant_pos_df["position"],
            assistant_pos_df["percentage_tokens_lost"],
            linewidth=2.5,
            color="darkblue",
            linestyle="--",
            label="Percentage Lost",
        )
        ax2_twin.set_ylabel(
            "Percentage of Assistant Tokens Lost (%)",
            color="darkblue",
            fontweight="bold",
        )
        ax2_twin.tick_params(axis="y", labelcolor="darkblue")

        ax2.set_title(
            "Assistant Token Loss by Truncation Position",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("Truncation Position (Token Index)")

        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    else:
        ax2.text(
            0.5,
            0.5,
            "No assistant token position data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title(
            "Assistant Token Loss by Truncation Position",
            fontsize=12,
            fontweight="bold",
        )

    # Plot 3: Distribution of System Tokens (bottom left)
    if "num_system_tokens" in df.columns:
        sns.histplot(
            data=df, x="num_system_tokens", bins=15, ax=ax3, kde=True, color="blue"
        )
        ax3.set_title("Distribution of System Tokens", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Number of System Tokens", fontsize=10)
        ax3.set_ylabel("Count", fontsize=10)
        mean_system = df["num_system_tokens"].mean()
        std_system = df["num_system_tokens"].std()
        ax3.axvline(
            mean_system,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"μ: {mean_system:.0f}",
        )
        ax3.axvline(
            mean_system + std_system,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"±σ: {std_system:.0f}",
        )
        ax3.axvline(mean_system - std_system, color="orange", linestyle=":", alpha=0.7)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.2)
    else:
        ax3.text(
            0.5,
            0.5,
            "System token data not available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("Distribution of System Tokens", fontsize=11, fontweight="bold")

    # Plot 4: Distribution of Assistant Tokens (bottom middle)
    sns.histplot(
        data=df, x="num_assistant_tokens", bins=15, ax=ax4, kde=True, color="orange"
    )
    ax4.set_title("Distribution of Assistant Tokens", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Number of Assistant Tokens", fontsize=10)
    ax4.set_ylabel("Count", fontsize=10)
    mean_assistant = df["num_assistant_tokens"].mean()
    std_assistant = df["num_assistant_tokens"].std()
    ax4.axvline(
        mean_assistant,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"μ: {mean_assistant:.0f}",
    )
    ax4.axvline(
        mean_assistant + std_assistant,
        color="darkred",
        linestyle=":",
        alpha=0.7,
        label=f"±σ: {std_assistant:.0f}",
    )
    ax4.axvline(
        mean_assistant - std_assistant, color="darkred", linestyle=":", alpha=0.7
    )
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.2)

    # Plot 5: Distribution of Conversation Tokens (bottom right)
    if "num_convo_tokens" in df.columns:
        sns.histplot(
            data=df, x="num_convo_tokens", bins=15, ax=ax5, kde=True, color="green"
        )
        ax5.set_title(
            "Distribution of Conversation Tokens", fontsize=11, fontweight="bold"
        )
        ax5.set_xlabel("Number of Conversation Tokens", fontsize=10)
        ax5.set_ylabel("Count", fontsize=10)
        mean_convo = df["num_convo_tokens"].mean()
        std_convo = df["num_convo_tokens"].std()
        ax5.axvline(
            mean_convo,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"μ: {mean_convo:.0f}",
        )
        ax5.axvline(
            mean_convo + std_convo,
            color="darkgreen",
            linestyle=":",
            alpha=0.7,
            label=f"±σ: {std_convo:.0f}",
        )
        ax5.axvline(mean_convo - std_convo, color="darkgreen", linestyle=":", alpha=0.7)
        ax5.legend(loc="upper right", fontsize=8)
        ax5.grid(True, alpha=0.2)
    else:
        ax5.text(
            0.5,
            0.5,
            "Conversation token data not available",
            ha="center",
            va="center",
            transform=ax5.transAxes,
        )
        ax5.set_title(
            "Distribution of Conversation Tokens", fontsize=11, fontweight="bold"
        )

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")
    plt.show()


def print_analysis_summary(df):
    """
    Print comprehensive analysis summary for truncation optimization.

    Args:
        df (pandas.DataFrame): DataFrame with token statistics
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS FOR TRUNCATION OPTIMIZATION")
    print("=" * 60)
    print(f"Dataset size: {len(df)} examples")
    print(
        f"Total tokens - Mean: {df['num_tokens'].mean():.1f}, Std: {df['num_tokens'].std():.1f}, Max: {df['num_tokens'].max()}"
    )
    print(
        f"Assistant tokens - Mean: {df['num_assistant_tokens'].mean():.1f}, Std: {df['num_assistant_tokens'].std():.1f}, Max: {df['num_assistant_tokens'].max()}"
    )
    print(
        f"First assistant index - Mean: {df['first_assistant_token_index'].mean():.1f}, Std: {df['first_assistant_token_index'].std():.1f}"
    )
    print(
        f"Last assistant index - Mean: {df['last_assistant_token_index'].mean():.1f}, Std: {df['last_assistant_token_index'].std():.1f}"
    )
    print(
        f"Assistant token range: {df['first_assistant_token_index'].min():.0f} to {df['last_assistant_token_index'].max():.0f}"
    )


def main():
    """Main function to run the token distribution analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze token distribution in a dataset for truncation optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python truncation_optimization.py --dataset data/train.jsonl --model Qwen/Qwen2.5-0.5B-instruct
  python truncation_optimization.py --dataset data/train.jsonl --model Qwen/Qwen2.5-0.5B-instruct --chat-template template.jinja --num-samples 1000
        """,
    )

    parser.add_argument(
        "--dataset", required=True, help="Path to the dataset file (required)"
    )

    parser.add_argument(
        "--model", required=True, help="Model name for tokenization (required)"
    )

    parser.add_argument("--chat-template", help="Path to chat template file (optional)")

    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of examples to sample from dataset (optional, uses all if not specified)",
    )

    parser.add_argument(
        "--output",
        default=DATA_DIR / "analyses" / "token_distribution_analysis.png",
        help=f"Output path for visualization (default: {DATA_DIR}/analyses/token_distribution_analysis.png)",
    )

    parser.add_argument(
        "--remove-system-messages",
        action="store_true",
        help="Remove system messages from the dataset (default: False)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    if args.chat_template and not Path(args.chat_template).exists():
        print(f"Error: Chat template file not found: {args.chat_template}")
        sys.exit(1)

    try:
        # Run analysis
        df = analyze_token_distribution(
            dataset_path=args.dataset,
            model_name=args.model,
            chat_template_path=args.chat_template,
            num_samples=args.num_samples,
            remove_system_messages=args.remove_system_messages,
        )

        # Print summary
        print_analysis_summary(df)

        # Create visualizations
        create_visualizations(df=df, output_path=args.output)

        print(f"\nAnalysis complete! Visualization saved to: {args.output}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
