#! /usr/bin/env python3
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import make_interp_spline

from experiments.eval.run_eval import get_simulation_results
from tau2.domains.telecom.tasks.utils import (
    get_intent_from_task_id,
    get_num_issues_from_task_id,
    get_persona_from_task_id,
)
from tau2.metrics.agent_metrics import compute_metrics, prepare_dfs
from tau2.metrics.break_down_metrics import (
    result_reward_actions_analysis,
    result_reward_analysis,
)
from tau2.utils.display import ConsoleDisplay

#### VISUALIZATION FUNCTIONS ####
MODES_MAP = {
    "base": "default",
    "solo": "no-user",
    "gt": "oracle-plan",
    "solo-gt": "oracle-plan-no-user",
    "default": "default",
    "no-user": "no-user",
    "op": "oracle-plan",
    "no-user-op": "oracle-plan-no-user",
}


MODES = ["default", "no-user", "oracle-plan"]
DOMAINS = ["retail", "airline", "telecom"]

TELECOM_INTENTS_ORDER = {"service_issue": 1, "mobile_data_issue": 2, "mms_issue": 3}
TELECOM_INTENTS_COLORS = {
    "service_issue": "#4E79A7",  # Rich blue
    "mobile_data_issue": "#F28E2B",  # Warm orange
    "mms_issue": "#76B7B2",  # Soft teal
}
TELECOM_PERSONAS_ORDER = {"None": 1, "Easy": 2, "Hard": 3}
PERSONA_COLORS = {"None": "#4C72B0", "Easy": "#DD8452", "Hard": "#55A868"}

DOMAIN_COLORS = {
    "retail": "#4C72B0",  # Nice blue
    "airline": "#DD8452",  # Warm orange
    "telecom": "#55A868",  # Fresh green
    "telecom-workflow": "#C44E52",  # Rich red
}
MODE_STYLES = {"default": "-", "no-user": "--", "oracle-plan": ":"}
MODE_COLORS = {"default": "#4C72B0", "no-user": "#DD8452", "oracle-plan": "#C44E52"}
MODE_MARKERS = {"default": "o", "no-user": "s", "oracle-plan": "D"}

LLMS = [
    "gpt-4.1-2025-04-14",
    "o4-mini-2025-04-16",
    "claude-3-7-sonnet-20250219",
    "gpt-4.1-mini-2025-04-14",
    "qwen2.5:7b",
]

LLMS_FOR_TELECOM_ANALYSIS = [
    "gpt-4.1-2025-04-14",
    "o4-mini-2025-04-16",
    "qwen2.5:7b",
]

LLM_NAME_MAP = {
    "gpt-4.1-2025-04-14": "gpt4.1",
    "o4-mini-2025-04-16": "o4-mini",
    "claude-3-7-sonnet-20250219": "claude-3.7-sonnet",
    "gpt-4.1-mini-2025-04-14": "gpt4.1-mini",
    "qwen2.5:7b": "qwen2.5-7b",
}

LLM_COLORS = {
    "gpt-4.1-2025-04-14": "#4C72B0",  # Nice blue
    "o4-mini-2025-04-16": "#DD8452",  # Warm orange
    "claude-3-7-sonnet-20250219": "#55A868",  # Fresh green
    "gpt-4.1-mini-2025-04-14": "#C44E52",  # Rich red
    "qwen2.5:7b": "#8C564B",  # Rich brown
}


def get_pass_hat_k_values(df: pd.DataFrame) -> Tuple[List[int], List[float]]:
    """
    Get the pass^k values for the df_metrics dataframe.
    """
    pass_cols = [col for col in df.columns if col.startswith("pass_hat_")]
    k_values = [int(col.split("_")[-1]) for col in pass_cols]
    pass_values = [df[col].iloc[0] for col in pass_cols]
    return k_values, pass_values


def plot_pass_k_metrics_per_llm_per_domain_bar_chart(
    fig_dir: Path, df_metrics: pd.DataFrame
):
    """
    Creates bar charts showing Pass^k metrics for each LLM across different domains.

    This function generates a separate bar chart for each LLM, showing how Pass^k varies
    across different domains (retail, airline, telecom). Each chart includes:
    - Bars for each domain showing Pass^k values
    - Smooth lines connecting the bars for better visualization
    - Value labels on top of each bar
    - A legend identifying each domain

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_metrics (pd.DataFrame): DataFrame containing the metrics data
    """

    # Filter data for base mode and user simulator
    df_metrics = df_metrics[(df_metrics["mode"] == "default")]
    if len(df_metrics) == 0:
        logger.warning(f"No data found for default mode. Skipping...")
        return

    # Get unique k values
    k_values = sorted(
        [
            int(col.split("^")[1])
            for col in df_metrics.columns
            if col.startswith("pass^")
        ]
    )

    tested_domains = df_metrics["domain"].unique()
    domains_to_plot = [d for d in DOMAINS if d in tested_domains]

    # Create a separate figure for each LLM
    for llm in df_metrics["llm"].unique():
        # Filter data for this LLM
        llm_data = df_metrics[df_metrics["llm"] == llm]
        if len(llm_data) == 0:
            logger.warning(f"No data found for {llm}. Skipping...")
            continue

        # Create figure
        plt.figure(figsize=(7, 3))

        # Add title with LLM name
        plt.title(LLM_NAME_MAP[llm], fontsize=16)

        # Set up bar positions
        x = np.arange(len(k_values))
        width = 0.8 / len(domains_to_plot)  # Width of bars

        # Plot bars for each domain in the specified order
        for i, domain in enumerate(domains_to_plot):
            domain_data = llm_data[llm_data["domain"] == domain]

            # Get pass^k values for this domain
            pass_values = []
            for k in k_values:
                col = f"pass^{k}"
                if col in domain_data.columns:
                    pass_values.append(domain_data[col].mean())
                else:
                    pass_values.append(0)

            # Calculate x positions for bars and markers
            x_pos = x + i * width - 0.4 + width / 2

            # Plot bars
            plt.bar(
                x_pos,
                pass_values,
                width,
                label=domain.capitalize(),
                color=DOMAIN_COLORS[domain],
                alpha=0.7,
                zorder=2,
            )
            if len(k_values) > 1:
                # Create smooth line that passes through the markers
                x_smooth = np.linspace(x_pos[0], x_pos[-1], 300)
                spl = make_interp_spline(x_pos, pass_values, k=3)
                y_smooth = spl(x_smooth)

                # Add smoothed line over the bars
                plt.plot(
                    x_smooth,
                    y_smooth,
                    color=DOMAIN_COLORS[domain],
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=1,
                )

                # Add markers at actual data points
                plt.scatter(
                    x_pos,
                    pass_values,
                    color=DOMAIN_COLORS[domain],
                    s=30,
                    alpha=0.7,
                    zorder=3,
                    marker="o",
                )

            # Add value labels on top of bars
            for j, v in enumerate(pass_values):
                plt.text(
                    x_pos[j],
                    v + 0.02,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

        plt.xlabel("k", fontsize=12)
        plt.ylabel("Pass^k", fontsize=12)
        if "o4" in llm:  # FIX ME: Hack to only get the legend once.
            plt.legend(fontsize=10, framealpha=0.9)
        plt.xticks(x, k_values, fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(bottom=0, top=0.9)

        # Remove top and right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()
        file_name = f"pass_k_metrics_default_all_domains_{llm}.pdf"
        plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close()


def plot_avg_pass_k_metrics_per_llm_per_mode(
    fig_dir: Path, df_metrics: pd.DataFrame, telecom_version: str
):
    """
    Creates bar charts showing average Pass^k metrics for each LLM across different modes in the telecom domain.

    This function generates a bar chart comparing the average Pass^k performance of different LLMs
    across various modes (base, base-solo, gt, gt-solo) in the telecom domain. The chart includes:
    - Bars for each mode showing average Pass^k values
    - Error bars showing standard deviation
    - Value labels on top of each bar
    - A legend identifying each LLM

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_metrics (pd.DataFrame): DataFrame containing the metrics data
    """
    llms = df_metrics["llm"].unique()
    llms = [model for model in LLMS_FOR_TELECOM_ANALYSIS if model in llms]
    n_llms = len(llms)
    if n_llms == 0:
        logger.warning(f"No LLMs found for telecom domain. Skipping...")
        return

    _, axes = plt.subplots(n_llms, 1, figsize=(10, 4 * n_llms))
    if n_llms == 1:
        axes = [axes]

    # Create bar plot for average Pass^k across modes and LLMs
    plt.figure(figsize=(6, 3))  # Reduced width from 10 to 8

    # Filter for telecom domain and base user simulator
    df_metrics = df_metrics[(df_metrics["domain"] == telecom_version)]
    if len(df_metrics) == 0:
        logger.warning(f"No data found for telecom domain. Skipping...")
        return

    # Calculate average pass^k for each mode and LLM
    x = np.arange(len(MODES))  # the label locations
    width = 0.35  # Reduced from 0.8/len(llms) to 0.35 for tighter spacing
    for i, llm in enumerate(llms):
        # Filter data per llm
        llm_data = df_metrics[df_metrics["llm"] == llm]
        averages = []
        stds = []
        for mode in MODES:
            mode_data = llm_data[llm_data["mode"] == mode]
            if len(mode_data) > 0:
                # Get all pass^k values for this mode and LLM
                k_values, pass_values = get_pass_hat_k_values(mode_data)
                # Calculate average and std
                avg = np.mean(pass_values)
                std = np.std(pass_values)
                averages.append(avg)
                stds.append(std)
            else:
                averages.append(0)
                stds.append(0)
        bars = plt.bar(
            x + i * width - 0.2,  # Reduced offset from 0.4 to 0.2
            averages,
            width,
            label=LLM_NAME_MAP[llm],
            yerr=stds,
            capsize=5,
            alpha=0.7,
            color=LLM_COLORS[llm],  # Add color from LLM_COLORS
            ecolor="black",  # This will be overridden by the errorbar call
            error_kw=dict(
                linestyle=":", alpha=0.3, capsize=5, capthick=1, elinewidth=1
            ),
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            # Offset text position to the right by 20% of bar width
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,  # Increased from 10 to 12
            )

    plt.xlabel("Mode", fontsize=12)  # Increased font size
    plt.ylabel("Average Pass^k", fontsize=12)  # Increased font size
    plt.xticks(x, MODES, fontsize=12)  # Increased font size
    plt.yticks(fontsize=12)  # Increased font size
    if "workflow" not in telecom_version:
        plt.legend(loc="upper left", fontsize=10, framealpha=0.9)  # Increased font size
    plt.ylim(bottom=0, top=1.0)
    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    file_name = f"pass_k_metrics_{telecom_version}_avg_bar.pdf"
    plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
    plt.close()


def plot_pass_one_metrics_per_llm_per_mode(
    fig_dir: Path, df_metrics: pd.DataFrame, telecom_version: str
):
    """
    Creates bar charts showing Pass^1 metrics for each LLM across different modes in the telecom domain.

    This function generates a bar chart comparing the Pass^1 performance of different LLMs
    across various modes (base, base-solo, gt, gt-solo) in the telecom domain. The chart includes:
    - Bars for each mode showing average Pass^k values
    - Error bars showing standard deviation
    - Value labels on top of each bar
    - A legend identifying each LLM

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_metrics (pd.DataFrame): DataFrame containing the metrics data
    """
    llms = df_metrics["llm"].unique()
    llms = [model for model in LLMS_FOR_TELECOM_ANALYSIS if model in llms]
    n_llms = len(llms)
    if n_llms == 0:
        logger.warning(f"No LLMs found for telecom domain. Skipping...")
        return

    _, axes = plt.subplots(n_llms, 1, figsize=(10, 4 * n_llms))
    if n_llms == 1:
        axes = [axes]

    # Create bar plot for average Pass^k across modes and LLMs
    plt.figure(figsize=(6, 3))  # Reduced width from 10 to 8

    # Filter for telecom domain and base user simulator
    df_metrics = df_metrics[(df_metrics["domain"] == telecom_version)]
    if len(df_metrics) == 0:
        logger.warning(f"No data found for telecom domain. Skipping...")
        return

    # Calculate average pass^k for each mode and LLM
    x = np.arange(len(MODES))  # the label locations
    width = 0.35  # Reduced from 0.8/len(llms) to 0.35 for tighter spacing
    for i, llm in enumerate(llms):
        # Filter data per llm
        llm_data = df_metrics[df_metrics["llm"] == llm]
        pass_one_values = []
        for mode in MODES:
            mode_data = llm_data[llm_data["mode"] == mode]
            if len(mode_data) > 0:
                print(mode_data.columns)
                # Get all pass^k values for this mode and LLM
                _, pass_values = get_pass_hat_k_values(mode_data)
                # Calculate average and std
                pass_one_values.append(pass_values[0])
            else:
                pass_one_values.append(0)
        bars = plt.bar(
            x + i * width - 0.2,  # Reduced offset from 0.4 to 0.2
            pass_one_values,
            width,
            label=LLM_NAME_MAP[llm],
            capsize=5,
            alpha=0.7,
            color=LLM_COLORS[llm],  # Add color from LLM_COLORS
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            # Offset text position to the right by 20% of bar width
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,  # Increased from 10 to 12
            )

    plt.xlabel("Mode", fontsize=12)  # Increased font size
    plt.ylabel("Pass^1", fontsize=12)  # Increased font size
    plt.xticks(x, MODES, fontsize=12)  # Increased font size
    plt.yticks(fontsize=12)  # Increased font size
    if "workflow" not in telecom_version:
        plt.legend(loc="upper left", fontsize=10, framealpha=0.9)  # Increased font size
    plt.ylim(bottom=0, top=1.0)
    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    file_name = f"pass_one_metrics_{telecom_version}_bar.pdf"
    plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
    plt.close()


def plot_pass_k_metrics_per_llm_per_mode(
    fig_dir: Path, df_metrics: pd.DataFrame, telecom_version: str
):
    """
    Creates bar charts showing Pass^k metrics for each LLM across different modes in the telecom domain,
    with one subplot per LLM and colors representing different k values.

    This function generates a multi-panel plot where each panel shows the Pass^k performance
    for a specific LLM. Each panel includes:
    - Bars for each mode (base, base-solo, gt, gt-solo)
    - Different colors for different k values
    - Value labels on top of each bar
    - A legend identifying each k value

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_metrics (pd.DataFrame): DataFrame containing the metrics data
    """
    # Filter for telecom domain and base user simulator
    df_metrics = df_metrics[(df_metrics["domain"] == telecom_version)]
    if len(df_metrics) == 0:
        logger.warning(f"No data found for {telecom_version} domain. Skipping...")
        return

    # Get unique k values
    k_values = sorted(
        [
            int(col.split("^")[1])
            for col in df_metrics.columns
            if col.startswith("pass^")
        ]
    )

    # Create color map for k values
    colors = plt.cm.tab10(np.linspace(0, 1, len(k_values)))

    # Create a figure with subplots for each LLM
    llms = df_metrics["llm"].unique()
    fig, axes = plt.subplots(len(llms), 1, figsize=(8, 4 * len(llms)))
    if len(llms) == 1:
        axes = [axes]

    # Set up bar positions
    x = np.arange(len(MODES))  # Base positions for modes
    width = 0.8 / len(k_values)  # Width of each bar

    # Plot for each LLM
    for llm_idx, llm in enumerate(llms):
        ax = axes[llm_idx]
        ax.set_title(f"LLM: {LLM_NAME_MAP[llm]}", fontsize=12)

        # Plot each k value
        for k_idx, k in enumerate(k_values):
            # Filter data per llm
            llm_data = df_metrics[df_metrics["llm"] == llm]
            values = []
            for mode in MODES:
                mode_data = llm_data[llm_data["mode"] == mode]
                if len(mode_data) > 0:
                    col = f"pass^{k}"
                    if col in mode_data.columns:
                        values.append(mode_data[col].mean())
                    else:
                        values.append(0)
                else:
                    values.append(0)

            # Calculate x positions for this k value
            x_pos = x + k_idx * width - 0.4 + width / 2

            # Plot bars
            bars = ax.bar(
                x_pos,
                values,
                width,
                label=f"k={k}",
                color=colors[k_idx],
                alpha=0.7,
            )

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # Only show x-axis label on bottom subplot
        if llm_idx == len(llms) - 1:
            ax.set_xlabel("Mode", fontsize=12)
        else:
            ax.set_xlabel("")

        ax.set_ylabel("Pass^k", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(MODES, fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        # Only show legend on top subplot
        if llm_idx == 0:
            ax.legend(fontsize=10, framealpha=0.9)
        ax.set_ylim(bottom=0)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    file_name = f"pass_k_metrics_{telecom_version}_per_k.pdf"
    plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
    plt.close()


def results_per_intent_telecom(
    fig_dir: Path, df_pass_hat_k: pd.DataFrame, telecom_version: str
):
    """
    Creates bar charts showing Pass^k metrics for each intent in the telecom domain.

    This function generates a separate bar chart for each LLM, showing how Pass^k varies
    across different intents in the telecom domain. Each chart includes:
    - Bars for each intent showing Pass^k values
    - Value labels on top of each bar
    - A legend identifying each intent
    - Different colors for different intents

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_pass_hat_k (pd.DataFrame): DataFrame containing Pass^k metrics data
    """
    # Filter for telecom domain and user simulator
    df_pass_hat_k = df_pass_hat_k[(df_pass_hat_k["domain"] == telecom_version)]

    df_pass_hat_k["intent"] = df_pass_hat_k["task_id"].apply(get_intent_from_task_id)
    llms = df_pass_hat_k["llm"].unique()
    modes = df_pass_hat_k["mode"].unique()
    rows = []
    for llm in llms:
        for mode in modes:
            for intent in df_pass_hat_k["intent"].unique():
                df_intent = df_pass_hat_k[
                    (df_pass_hat_k["llm"] == llm)
                    & (df_pass_hat_k["mode"] == mode)
                    & (df_pass_hat_k["intent"] == intent)
                ]
                row = {"llm": llm, "mode": mode, "intent": intent}
                for column in df_intent.columns:
                    if match := re.match(r"pass\^(\d+)", column):
                        k = int(match.group(1))
                        row[k] = (df_intent[column].mean(), df_intent[column].std())
                rows.append(row)
    df_phk_per_intent = pd.DataFrame(rows)

    # Create plots for each LLM showing pass^k vs k for each intent in base mode
    for llm in llms:
        # Filter for base mode and this LLM
        df_llm = df_phk_per_intent[
            (df_phk_per_intent["llm"] == llm) & (df_phk_per_intent["mode"] == "default")
        ]

        if len(df_llm) == 0:
            logger.warning(f"No data found for LLM {llm} in default mode. Skipping...")
            continue

        # Create figure
        plt.figure(figsize=(9, 4))

        # Add title with LLM name
        plt.title(LLM_NAME_MAP[llm], fontsize=16)

        # Get unique k values
        k_values = sorted([col for col in df_llm.columns if isinstance(col, int)])

        # Create color map for intents
        intents = [
            (intent, TELECOM_INTENTS_ORDER.get(intent, 0))
            for intent in df_llm["intent"].unique()
        ]
        intents = [i for i, _ in sorted(intents, key=lambda x: x[1])]
        colors = [TELECOM_INTENTS_COLORS[intent] for intent in intents]

        # Set up bar positions
        x = np.arange(len(k_values))
        width = 0.8 / len(intents)  # Width of bars

        # Plot each intent
        for i, (intent, color) in enumerate(zip(intents, colors)):
            df_intent = df_llm[df_llm["intent"] == intent]
            if len(df_intent) == 0:
                continue

            # Extract mean values
            means = []
            for k in k_values:
                if k in df_intent.columns:
                    mean, _ = df_intent[k].iloc[0]
                    means.append(mean)
                else:
                    means.append(0)

            # Calculate x positions for bars
            x_pos = x + i * width - 0.4 + width / 2

            # Plot bars
            bars = plt.bar(
                x_pos,
                means,
                width,
                color=color,
                label=intent,
                alpha=0.7,
            )

            # Add value labels on top of bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=16,  # Increased from 14
                )

        plt.xlabel("k", fontsize=16)  # Increased from 12
        plt.ylabel("Pass^k", fontsize=16)  # Increased from 12
        if (
            "gpt" in llm and "mini" not in llm
        ):  # FIX ME: Hack to only get the legend once.
            plt.legend(
                loc="upper right", fontsize=14, framealpha=0.9
            )  # Increased from 10
        plt.xticks(x, k_values, fontsize=14)  # Increased from 12
        plt.yticks(fontsize=14)  # Increased from 12
        current_top = plt.gca().get_ylim()[1]
        plt.ylim(bottom=0, top=current_top * 1.1)  # Add 10% padding to top

        # Remove top and right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()
        file_name = f"pass_k_vs_k_per_intent_{telecom_version}_{llm}.pdf"
        plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close()


def results_per_persona_telecom(
    fig_dir: Path, df_pass_hat_k: pd.DataFrame, telecom_version: str
):
    """
    Creates bar charts showing Pass^k metrics for each persona in the telecom domain.

    This function generates a separate bar chart for each LLM, showing how Pass^k varies
    across different personas in the telecom domain. Each chart includes:
    - Bars for each persona showing Pass^k values
    - Value labels on top of each bar
    - A legend identifying each persona
    - Different colors for different personas

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_pass_hat_k (pd.DataFrame): DataFrame containing Pass^k metrics data
    """
    # Filter for telecom domain and user simulator
    df_pass_hat_k = df_pass_hat_k[(df_pass_hat_k["domain"] == telecom_version)]

    df_pass_hat_k["persona"] = df_pass_hat_k["task_id"].apply(get_persona_from_task_id)
    llms = df_pass_hat_k["llm"].unique()
    modes = df_pass_hat_k["mode"].unique()
    rows = []
    for llm in llms:
        for mode in modes:
            for persona in df_pass_hat_k["persona"].unique():
                df_persona = df_pass_hat_k[
                    (df_pass_hat_k["llm"] == llm)
                    & (df_pass_hat_k["mode"] == mode)
                    & (df_pass_hat_k["persona"] == persona)
                ]
                row = {"llm": llm, "mode": mode, "persona": persona}
                for column in df_persona.columns:
                    if match := re.match(r"pass\^(\d+)", column):
                        k = int(match.group(1))
                        row[k] = (df_persona[column].mean(), df_persona[column].std())
                rows.append(row)
    df_phk_per_persona = pd.DataFrame(rows)

    # Create plots for each LLM showing pass^k vs k for each persona in base mode
    for llm in llms:
        # Filter for base mode and this LLM
        df_llm = df_phk_per_persona[
            (df_phk_per_persona["llm"] == llm)
            & (df_phk_per_persona["mode"] == "default")
        ]

        if len(df_llm) == 0:
            logger.warning(f"No data found for LLM {llm} in default mode. Skipping...")
            continue

        # Create figure
        plt.figure(figsize=(8, 4))
        plt.title(LLM_NAME_MAP[llm], fontsize=16)

        # Get unique k values
        k_values = sorted([col for col in df_llm.columns if isinstance(col, int)])

        # Get personas in order
        personas = [
            (persona, TELECOM_PERSONAS_ORDER.get(persona, 0))
            for persona in df_llm["persona"].unique()
        ]
        personas = [i for i, _ in sorted(personas, key=lambda x: x[1])]

        # Set up bar positions
        x = np.arange(len(k_values))
        width = 0.8 / len(personas)  # Width of bars

        # Plot each persona
        for i, persona in enumerate(personas):
            df_persona = df_llm[df_llm["persona"] == persona]
            if len(df_persona) == 0:
                continue

            # Extract mean values
            means = []
            for k in k_values:
                if k in df_persona.columns:
                    mean, _ = df_persona[k].iloc[0]
                    means.append(mean)
                else:
                    means.append(0)

            # Calculate x positions for bars
            x_pos = x + i * width - 0.4 + width / 2

            # Get color from PERSONA_COLORS, default to a gray if not found
            color = PERSONA_COLORS.get(persona, "#808080")

            # Plot bars
            bars = plt.bar(
                x_pos,
                means,
                width,
                color=color,
                label=persona,
                alpha=0.7,
            )

            # Add value labels on top of bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                )

        plt.xlabel("k", fontsize=12)
        plt.ylabel("Pass^k", fontsize=12)
        if (
            "gpt" in llm and "mini" not in llm
        ):  # FIX ME: Hack to only get the legend once.
            plt.legend(loc="upper right", fontsize=12, framealpha=0.9)
        plt.xticks(x, k_values, fontsize=12)
        plt.yticks(fontsize=12)
        current_top = plt.gca().get_ylim()[1]
        plt.ylim(bottom=0, top=current_top * 1.1)  # Add 10% padding to top

        # Remove top and right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()
        file_name = f"pass_k_vs_k_per_persona_{telecom_version}_{llm}.pdf"
        plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close()


def comparing_personas_to_solo_mode(
    fig_dir: Path,
    df_pass_hat_k: pd.DataFrame,
    telecom_version: str = "telecom",
):
    """
    Creates bar charts comparing base and base-solo modes for each persona in the telecom domain.

    This function generates a bar chart where:
    - Each persona has two bars (base and base-solo modes)
    - The y-axis shows pass^1 values
    - Different colors distinguish between base and base-solo modes
    - Value labels are shown on top of each bar

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_pass_hat_k (pd.DataFrame): DataFrame containing Pass^k metrics data
    """
    # Filter for telecom domain
    df_pass_hat_k = df_pass_hat_k[(df_pass_hat_k["domain"] == telecom_version)]

    # Extract persona from task_id
    df_pass_hat_k["persona"] = df_pass_hat_k["task_id"].apply(get_persona_from_task_id)

    modes = ["default", "no-user"]

    # Filter for base and base-solo modes
    df_pass_hat_k = df_pass_hat_k[df_pass_hat_k["mode"].isin(modes)]

    # Create a separate figure for each LLM
    for llm in df_pass_hat_k["llm"].unique():
        # Filter data for this LLM
        df_llm = df_pass_hat_k[df_pass_hat_k["llm"] == llm]

        if len(df_llm) == 0:
            logger.warning(f"No data found for LLM {llm}. Skipping...")
            continue

        # Create figure
        plt.figure(figsize=(8, 4))

        # Get unique personas and sort them according to TELECOM_PERSONAS_ORDER
        personas = [
            (persona, TELECOM_PERSONAS_ORDER.get(persona, 0))
            for persona in df_llm["persona"].unique()
        ]
        personas = [i for i, _ in sorted(personas, key=lambda x: x[1])]

        # Set up bar positions
        x = np.arange(len(personas))
        width = 0.35  # Width of bars

        # Plot bars for each mode
        for i, mode in enumerate(modes):
            mode_data = df_llm[df_llm["mode"] == mode]

            # Calculate mean pass^1 for each persona
            means = []
            for persona in personas:
                persona_data = mode_data[mode_data["persona"] == persona]
                if len(persona_data) > 0:
                    mean = persona_data["pass^1"].mean()
                else:
                    mean = 0
                means.append(mean)

            # Calculate x positions for bars
            x_pos = x + i * width - width / 2

            # Plot bars
            bars = plt.bar(
                x_pos,
                means,
                width,
                label=mode,
                color="blue" if mode == "base" else "red",
                alpha=0.7,
            )

            # Add value labels on top of bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

        plt.xlabel("Persona", fontsize=12)
        plt.ylabel("Pass^1", fontsize=12)
        plt.xticks(x, personas, fontsize=12)
        plt.yticks(fontsize=12)
        current_top = plt.gca().get_ylim()[1]
        plt.ylim(bottom=0, top=current_top * 1.1)  # Add 10% padding to top
        plt.legend(fontsize=10, framealpha=0.9)

        # Remove top and right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()
        file_name = f"persona_comparison_default_vs_no-user_{telecom_version}_{llm}.pdf"
        plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close()


def impact_of_solo_mode_on_performance(
    fig_dir: Path, df_metrics: pd.DataFrame, telecom_version: str
):
    """
    Analyzes and visualizes the impact of communication on task performance in the telecom domain.

    This function compares performance between modes with and without communication:
    - base vs base-solo: Shows impact of communication in standard mode
    - gt vs gt-solo: Shows impact of communication in ground truth mode

    The function:
    1. Calculates performance differences between communication-enabled and solo modes
    2. Saves the results to a CSV file for further analysis
    3. Prints a summary of the impact for each LLM

    Args:
        fig_dir (Path): Directory where the analysis results will be saved
        df_metrics (pd.DataFrame): DataFrame containing the metrics data
    """
    llms = df_metrics["llm"].unique()

    # Filter for telecom domain and base user simulator
    df_metrics = df_metrics[(df_metrics["domain"] == telecom_version)]
    if len(df_metrics) == 0:
        logger.warning(f"No data found for {telecom_version} domain. Skipping...")
        return
    rows = []
    for i, llm in enumerate(llms):
        # Filter data per llm
        llm_data = df_metrics[df_metrics["llm"] == llm]
        row = {"llm": llm}
        for mode in MODES:
            mode_data = llm_data[llm_data["mode"] == mode]
            if len(mode_data) > 0:
                # Get all pass^k values for this mode and LLM
                k_values, pass_values = get_pass_hat_k_values(mode_data)
                # Calculate average and std
                avg = np.mean(pass_values)
                std = np.std(pass_values)
                row[mode] = avg
                row[f"{mode}_std"] = std
            else:
                row[mode] = 0
                row[f"{mode}_std"] = 0
        rows.append(row)
    df_interactions = pd.DataFrame(rows)
    df_interactions["no-user - default"] = (
        df_interactions["no-user"] - df_interactions["default"]
    )
    print(df_interactions[["llm", "no-user - default"]])
    fig_dir.mkdir(parents=True, exist_ok=True)
    df_interactions.to_csv(fig_dir / f"interactions_{telecom_version}.csv", index=False)


def plot_pass_k_vs_num_actions(
    fig_dir: Path,
    df_pass_hat_k: pd.DataFrame,
    use_modes=None,
    telecom_version: str = "telecom",
):
    """
    Creates line plots showing how Pass^k varies with the number of actions for each LLM and mode.

    This function generates two types of plots:
    1. A multi-panel plot showing Pass^k vs number of actions for each mode, with different
       lines representing different k values
    2. A multi-panel plot comparing different LLMs' performance for Pass^k vs number of actions

    Each plot includes:
    - Lines with markers showing the relationship between Pass^k and number of actions
    - A background plot showing the proportion of tasks for each number of actions (in %)
    - Grid lines for better readability
    - Appropriate legends and labels
    - Special handling for 'no-solution' cases (-1):
        * Different marker for -1 values
        * No line connecting -1 to other points
        * Vertical dashed line separating -1 from other values

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_pass_hat_k (pd.DataFrame): DataFrame containing Pass^k metrics data
        use_modes (list, optional): List of modes to include in the plot. Defaults to None
        telecom_version (str, optional): Version of the telecom domain. Defaults to "telecom"
    """
    if use_modes is None:
        use_modes = MODES[:]
    # Filter for telecom domain and base user simulator
    df_pass_hat_k_filtered = df_pass_hat_k[(df_pass_hat_k["domain"] == telecom_version)]
    if len(df_pass_hat_k_filtered) == 0:
        logger.warning(f"No data found for {telecom_version} domain. Skipping...")
        return

    # Get unique k values
    k_values = sorted(
        [
            int(col.split("^")[1])
            for col in df_pass_hat_k.columns
            if col.startswith("pass^")
        ]
    )

    # Create color gradient from blue to red
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(k_values)))
    # Create a separate figure for each LLM
    for llm in df_pass_hat_k_filtered["llm"].unique():
        # Calculate task proportions for background plot (once per LLM)
        task_proportions = (
            df_pass_hat_k_filtered[
                (df_pass_hat_k_filtered["llm"] == llm)
                & (df_pass_hat_k_filtered["mode"] == "default")
            ]["num_actions"].value_counts(normalize=True)
            * 100
        )  # Convert to percentage

        # Get all possible num_actions values including -1 (transfer)
        all_num_actions = sorted(task_proportions.index)
        x_pos = np.arange(len(all_num_actions))

        # Create a figure with one plot per mode
        modes = [m for m in MODES if m in use_modes]  # Ensure order matches MODES
        _, axes = plt.subplots(len(modes), 1, figsize=(8, 4 * len(modes)))
        if len(modes) == 1:
            axes = [axes]

        # Plot for each mode
        for mode_idx, mode in enumerate(modes):
            ax = axes[mode_idx]
            ax.set_title(mode, fontsize=12)

            # Create a second y-axis for task proportions
            ax2 = ax.twinx()
            ax2.set_ylabel("Proportion of Tasks (%)", fontsize=12, color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

            # Plot task proportions as a background bar chart
            ax2.bar(
                x_pos,
                task_proportions.values,
                alpha=0.2,
                color="gray",
                label="Task Proportion",
            )
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(
                [str(int(x)) if x != -1 else "transfer" for x in all_num_actions],
                fontsize=12,
            )
            ax2.set_ylim(0, max(task_proportions.values) * 1.1)

            # Filter data for this mode and LLM
            mode_data = df_pass_hat_k_filtered[
                (df_pass_hat_k_filtered["mode"] == mode)
                & (df_pass_hat_k_filtered["llm"] == llm)
            ]

            # Plot each k value
            for k_idx, k in enumerate(k_values):
                col = f"pass^{k}"
                # Group by num_actions and calculate mean
                grouped = mode_data.groupby("num_actions")[col].mean()

                # Create a series with all possible num_actions values
                full_series = pd.Series(index=all_num_actions, dtype=float)
                full_series.update(grouped)
                full_series = full_series.fillna(0)  # Fill missing values with 0

                # Split data into -1 and non--1 values
                no_solution_idx = all_num_actions.index(-1)
                regular_indices = [
                    i for i in range(len(all_num_actions)) if i != no_solution_idx
                ]

                # Plot regular values with line and circle markers
                ax.plot(
                    [x_pos[i] for i in regular_indices],
                    [full_series.values[i] for i in regular_indices],
                    marker="o",
                    linewidth=2,
                    color=colors[k_idx],
                    label=f"k={k}",
                )

                # Plot -1 value with different marker (square) and no connecting line
                ax.plot(
                    x_pos[no_solution_idx],
                    full_series.values[no_solution_idx],
                    marker="s",
                    markersize=8,
                    color=colors[k_idx],
                    linestyle="",
                )

            # Add vertical dashed line separating -1 from other values
            no_solution_idx = all_num_actions.index(-1)
            ax.axvline(
                x=x_pos[no_solution_idx] + 0.5,
                color="gray",
                linestyle="--",
                alpha=0.5,
            )

            # Only show x-axis label on bottom figure
            if mode_idx == len(modes) - 1:
                ax.set_xlabel("Number of Actions", fontsize=12)
            else:
                ax.set_xlabel("")

            ax.set_ylabel("Pass^k", fontsize=12)
            ax.grid(True, alpha=0.3)
            # Only show legend on top figure
            if mode_idx == 0:
                ax.legend(fontsize=12)
            current_top = ax.get_ylim()[1]
            ax.set_ylim(bottom=0, top=current_top * 1.1)  # Add 10% padding to top
            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Set x-axis ticks and labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [str(int(x)) if x != -1 else "transfer" for x in all_num_actions],
                fontsize=12,
            )
            ax.tick_params(axis="y", labelsize=12)

        plt.tight_layout()
        file_name = f"pass_k_vs_num_actions_{telecom_version}_{llm}.pdf"
        plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close()


def plot_pass_k_vs_num_actions_all_llms(
    fig_dir: Path,
    df_pass_hat_k: pd.DataFrame,
    use_modes=None,
    telecom_version: str = "telecom",
):
    """
    Creates line plots showing how Pass^k varies with the number of actions for each LLM and mode,
    with one curve per LLM and mode combination on a single figure.

    This function generates a plot comparing different LLMs' performance for Pass^k vs number of actions.
    The plot includes:
    - Lines with markers showing the relationship between Pass^k and number of actions
    - Different line styles for different modes (using MODE_STYLES)
    - Different colors for different LLMs (using LLM_COLORS)
    - Different markers for different modes (using MODE_MARKERS)
    - A background plot showing the proportion of tasks for each number of actions (in %)
    - Grid lines for better readability
    - Appropriate legends and labels
    - Special handling for 'no-solution' cases (-1):
        * Uses the same marker and color as the corresponding mode and LLM
        * No line connecting -1 to other points
        * Vertical dashed line separating -1 from other values

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_pass_hat_k (pd.DataFrame): DataFrame containing Pass^k metrics data
        use_modes (list, optional): List of modes to include in the plot. Defaults to None
        telecom_version (str, optional): Version of the telecom domain. Defaults to "telecom"
    """
    if use_modes is None:
        use_modes = MODES[:]
    # Filter for telecom domain and base user simulator
    df_pass_hat_k_filtered = df_pass_hat_k[(df_pass_hat_k["domain"] == telecom_version)]
    if len(df_pass_hat_k_filtered) == 0:
        logger.warning(f"No data found for {telecom_version} domain. Skipping...")
        return

    # Get unique k values
    k_values = sorted(
        [
            int(col.split("^")[1])
            for col in df_pass_hat_k.columns
            if col.startswith("pass^")
        ]
    )
    # Use first k value for this plot
    k = k_values[0]

    modes = df_pass_hat_k_filtered["mode"].unique()
    count_actions_mode = "default" if "default" in modes else modes[0]
    if count_actions_mode not in modes:
        logger.warning(f"No data found for {count_actions_mode}. Using {modes[0]} instead to count number of actions in each task.")
        count_actions_mode = modes[0]

    # Calculate task proportions for background plot
    task_proportions = (
        df_pass_hat_k_filtered[(df_pass_hat_k_filtered["mode"] == count_actions_mode)][
            "num_actions"
        ].value_counts(normalize=True)
        * 100
    )  # Convert to percentage

    # Get all possible num_actions values including -1 (transfer)
    all_num_actions = sorted(task_proportions.index)
    x_pos = np.arange(len(all_num_actions))

    # Create a single figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a second y-axis for task proportions
    ax2 = ax.twinx()
    ax2.set_ylabel("Proportion of Tasks (%)", fontsize=16, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=16)

    # Plot task proportions as a background bar chart
    ax2.bar(
        x_pos,
        task_proportions.values,
        alpha=0.2,
        color="gray",
        label="Task Proportion",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [str(int(x)) if x != -1 else "transfer" for x in all_num_actions],
        fontsize=16,
    )
    ax2.set_ylim(0, max(task_proportions.values) * 1.1)

    # Store handles and labels for legend
    handles = []
    labels = []
    legend_data = []

    # Plot for each mode and LLM combination
    for mode in use_modes:
        mode_data = df_pass_hat_k_filtered[df_pass_hat_k_filtered["mode"] == mode]

        for llm in mode_data["llm"].unique():
            if llm not in LLMS_FOR_TELECOM_ANALYSIS:
                continue
            llm_data = mode_data[mode_data["llm"] == llm]
            col = f"pass^{k}"

            # Group by num_actions and calculate mean
            grouped = llm_data.groupby("num_actions")[col].mean()

            # Create a series with all possible num_actions values
            full_series = pd.Series(index=all_num_actions, dtype=float)
            full_series.update(grouped)
            full_series = full_series.fillna(0)  # Fill missing values with 0

            # Split data into -1 and non--1 values
            no_solution_idx = all_num_actions.index(-1)
            regular_indices = [
                i for i in range(len(all_num_actions)) if i != no_solution_idx
            ]

            # Create label combining LLM and mode
            label = f"{LLM_NAME_MAP[llm]} ({mode})"

            # Plot regular values with line and mode-specific markers
            line = ax.plot(
                [x_pos[i] for i in regular_indices],
                [full_series.values[i] for i in regular_indices],
                marker=MODE_MARKERS[mode],
                linewidth=2,
                color=LLM_COLORS[llm],
                linestyle=MODE_STYLES[mode],
                label=label,
            )[0]

            # Plot -1 value with the same marker and color as the mode and LLM
            ax.plot(
                x_pos[no_solution_idx],
                full_series.values[no_solution_idx],
                marker=MODE_MARKERS[mode],
                markersize=8,
                linestyle="",
                color=LLM_COLORS[llm],
            )

            # Store handle and label for legend
            legend_data.append((line, label, llm, mode))

    # Sort legend data by LLM and mode
    legend_data.sort(key=lambda x: (x[2], x[3]))  # Sort by LLM first, then mode
    handles = [item[0] for item in legend_data]
    labels = [item[1] for item in legend_data]

    # Add vertical dashed line separating -1 from other values
    no_solution_idx = all_num_actions.index(-1)
    ax.axvline(
        x=x_pos[no_solution_idx] + 0.5,
        color="gray",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel("Number of Actions", fontsize=16)
    ax.set_ylabel(f"Pass^{k}", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels, fontsize=16, loc="upper right", framealpha=0.9)
    current_top = ax.get_ylim()[1]
    ax.set_ylim(bottom=0, top=current_top * 1.1)  # Add 10% padding to top

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set x-axis ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(int(x)) if x != -1 else "transfer" for x in all_num_actions],
        fontsize=16,
    )
    ax.tick_params(axis="y", labelsize=16)

    plt.tight_layout()
    file_name = f"pass_k_vs_num_actions_{telecom_version}_all_modes_llm_comparison.pdf"
    plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
    plt.close()


def plot_pass_k_vs_num_issues_all_llms(
    fig_dir: Path,
    df_pass_hat_k: pd.DataFrame,
    use_modes=None,
    telecom_version: str = "telecom",
):
    """
    Creates line plots showing how Pass^k varies with the number of issues for each LLM and mode,
    with one curve per LLM and mode combination on a single figure.

    This function generates a plot comparing different LLMs' performance for Pass^k vs number of issues.
    The plot includes:
    - Lines with markers showing the relationship between Pass^k and number of issues
    - Different line styles for different modes (using MODE_STYLES)
    - Different colors for different LLMs (using LLM_COLORS)
    - Different markers for different modes (using MODE_MARKERS)
    - A background plot showing the proportion of tasks for each number of issues (in %)
    - Grid lines for better readability
    - Appropriate legends and labels
    - Special handling for 'no-solution' cases (-1):
        * Uses the same marker and color as the corresponding mode and LLM
        * No line connecting -1 to other points
        * Vertical dashed line separating -1 from other values

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_pass_hat_k (pd.DataFrame): DataFrame containing Pass^k metrics data
        use_modes (list, optional): List of modes to include in the plot. Defaults to None
        telecom_version (str, optional): Version of the telecom domain. Defaults to "telecom"
    """
    if use_modes is None:
        use_modes = MODES[:]
    # Filter for telecom domain and base user simulator
    df_pass_hat_k_filtered = df_pass_hat_k[(df_pass_hat_k["domain"] == telecom_version)]
    if len(df_pass_hat_k_filtered) == 0:
        logger.warning(f"No data found for {telecom_version} domain. Skipping...")
        return

    # Get unique k values
    k_values = sorted(
        [
            int(col.split("^")[1])
            for col in df_pass_hat_k.columns
            if col.startswith("pass^")
        ]
    )
    # Use first k value for this plot
    k = k_values[0]

    # Calculate num_issues from task_id, setting to -1 where num_actions is -1
    df_pass_hat_k_filtered["num_issues"] = df_pass_hat_k_filtered.apply(
        lambda row: -1
        if row["num_actions"] == -1
        else get_num_issues_from_task_id(row["task_id"]),
        axis=1,
    )

    modes = df_pass_hat_k_filtered["mode"].unique()
    count_issues_mode = "default" if "default" in modes else modes[0]
    if count_issues_mode not in modes:
        logger.warning(f"No data found for {count_issues_mode}. Using {modes[0]} instead to count number of issues in each task.")
        count_issues_mode = modes[0]

    # Calculate task proportions for background plot
    task_proportions = (
        df_pass_hat_k_filtered[(df_pass_hat_k_filtered["mode"] == count_issues_mode)][
            "num_issues"
        ].value_counts(normalize=True)
        * 100
    )  # Convert to percentage

    # Get all possible num_issues values including -1 (transfer)
    all_num_issues = sorted(task_proportions.index)
    x_pos = np.arange(len(all_num_issues))

    # Create a single figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a second y-axis for task proportions
    ax2 = ax.twinx()
    ax2.set_ylabel("Proportion of Tasks (%)", fontsize=16, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=16)

    # Plot task proportions as a background bar chart
    ax2.bar(
        x_pos,
        task_proportions.values,
        alpha=0.2,
        color="gray",
        label="Task Proportion",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [str(int(x)) if x != -1 else "transfer" for x in all_num_issues],
        fontsize=16,
    )
    ax2.set_ylim(0, max(task_proportions.values) * 1.1)

    # Store handles and labels for legend
    # handles = []
    # labels = []
    legend_data = []

    # Plot for each mode and LLM combination
    for mode in use_modes:
        mode_data = df_pass_hat_k_filtered[df_pass_hat_k_filtered["mode"] == mode]

        for llm in mode_data["llm"].unique():
            if llm not in LLMS_FOR_TELECOM_ANALYSIS:
                continue
            llm_data = mode_data[mode_data["llm"] == llm]
            col = f"pass^{k}"

            # Group by num_issues and calculate mean
            grouped = llm_data.groupby("num_issues")[col].mean()

            # Create a series with all possible num_issues values
            full_series = pd.Series(index=all_num_issues, dtype=float)
            full_series.update(grouped)
            full_series = full_series.fillna(0)  # Fill missing values with 0

            # Split data into -1 and non--1 values
            no_solution_idx = all_num_issues.index(-1)
            regular_indices = [
                i for i in range(len(all_num_issues)) if i != no_solution_idx
            ]

            # Create label combining LLM and mode
            label = f"{LLM_NAME_MAP[llm]} ({mode})"

            # Plot regular values with line and mode-specific markers
            line = ax.plot(
                [x_pos[i] for i in regular_indices],
                [full_series.values[i] for i in regular_indices],
                marker=MODE_MARKERS[mode],
                linewidth=2,
                color=LLM_COLORS[llm],
                linestyle=MODE_STYLES[mode],
                label=label,
            )[0]

            # Plot -1 value with the same marker and color as the mode and LLM
            ax.plot(
                x_pos[no_solution_idx],
                full_series.values[no_solution_idx],
                marker=MODE_MARKERS[mode],
                markersize=8,
                linestyle="",
                color=LLM_COLORS[llm],
            )

            # Store handle and label for legend
            legend_data.append((line, label, llm, mode))

    # Sort legend data by LLM and mode
    legend_data.sort(key=lambda x: (x[2], x[3]))  # Sort by LLM first, then mode
    # handles = [item[0] for item in legend_data]
    # labels = [item[1] for item in legend_data]

    # Add vertical dashed line separating -1 from other values
    no_solution_idx = all_num_issues.index(-1)
    ax.axvline(
        x=x_pos[no_solution_idx] + 0.5,
        color="gray",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel("Number of Sub-Tasks", fontsize=16)
    ax.set_ylabel(f"Pass^{k}", fontsize=16)
    ax.grid(True, alpha=0.3)
    # ax.legend(handles, labels, fontsize=10, loc="upper right", framealpha=0.9)
    current_top = ax.get_ylim()[1]
    ax.set_ylim(bottom=0, top=current_top * 1.1)  # Add 10% padding to top

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set x-axis ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(int(x)) if x != -1 else "transfer" for x in all_num_issues],
        fontsize=14,
    )
    ax.tick_params(axis="y", labelsize=16)

    plt.tight_layout()
    file_name = f"pass_k_vs_num_issues_{telecom_version}_all_modes_llm_comparison.pdf"
    plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
    plt.close()


def plot_pass_k_vs_num_issues(
    fig_dir: Path,
    df_pass_hat_k: pd.DataFrame,
    use_modes=None,
    telecom_version: str = "telecom",
):
    """
    Creates line plots showing how Pass^k varies with the number of issues for each LLM and mode.

    This function generates two types of plots:
    1. A multi-panel plot showing Pass^k vs number of issues for each mode, with different
       lines representing different k values
    2. A multi-panel plot comparing different LLMs' performance for Pass^k vs number of issues

    Each plot includes:
    - Lines with markers showing the relationship between Pass^k and number of issues
    - A background plot showing the proportion of tasks for each number of issues (in %)
    - Grid lines for better readability
    - Appropriate legends and labels
    - Special handling for 'no-solution' cases (-1):
        * Different marker for -1 values
        * No line connecting -1 to other points
        * Vertical dashed line separating -1 from other values

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_pass_hat_k (pd.DataFrame): DataFrame containing Pass^k metrics data
        use_modes (list, optional): List of modes to include in the plot. Defaults to None
        telecom_version (str, optional): Version of the telecom domain. Defaults to "telecom"
    """
    if use_modes is None:
        use_modes = MODES[:]
    # Filter for telecom domain and base user simulator
    df_pass_hat_k_filtered = df_pass_hat_k[(df_pass_hat_k["domain"] == telecom_version)]
    if len(df_pass_hat_k_filtered) == 0:
        logger.warning(f"No data found for {telecom_version} domain. Skipping...")
        return

    # Get unique k values
    k_values = sorted(
        [
            int(col.split("^")[1])
            for col in df_pass_hat_k.columns
            if col.startswith("pass^")
        ]
    )
    # Calculate num_issues from task_id, setting to -1 where num_actions is -1
    df_pass_hat_k_filtered["num_issues"] = df_pass_hat_k_filtered.apply(
        lambda row: -1
        if row["num_actions"] == -1
        else get_num_issues_from_task_id(row["task_id"]),
        axis=1,
    )

    # Create color gradient from blue to red
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(k_values)))
    # Create a separate figure for each LLM
    for llm in df_pass_hat_k_filtered["llm"].unique():
        # Calculate task proportions for background plot (once per LLM)
        task_proportions = (
            df_pass_hat_k_filtered[
                (df_pass_hat_k_filtered["llm"] == llm)
                & (df_pass_hat_k_filtered["mode"] == "default")
            ]["num_issues"].value_counts(normalize=True)
            * 100
        )  # Convert to percentage

        # Get all possible num_issues values including -1 (transfer)
        all_num_issues = sorted(task_proportions.index)
        x_pos = np.arange(len(all_num_issues))

        # Create a figure with one plot per mode
        modes = [m for m in MODES if m in use_modes]  # Ensure order matches MODES
        _, axes = plt.subplots(len(modes), 1, figsize=(8, 4 * len(modes)))
        if len(modes) == 1:
            axes = [axes]

        # Plot for each mode
        for mode_idx, mode in enumerate(modes):
            ax = axes[mode_idx]
            ax.set_title(mode, fontsize=12)

            # Create a second y-axis for task proportions
            ax2 = ax.twinx()
            ax2.set_ylabel("Proportion of Tasks (%)", fontsize=12, color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

            # Plot task proportions as a background bar chart
            ax2.bar(
                x_pos,
                task_proportions.values,
                alpha=0.2,
                color="gray",
                label="Task Proportion",
            )
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(
                [str(int(x)) if x != -1 else "transfer" for x in all_num_issues],
                fontsize=12,
            )
            ax2.set_ylim(0, max(task_proportions.values) * 1.1)

            # Filter data for this mode and LLM
            mode_data = df_pass_hat_k_filtered[
                (df_pass_hat_k_filtered["mode"] == mode)
                & (df_pass_hat_k_filtered["llm"] == llm)
            ]

            # Plot each k value
            for k_idx, k in enumerate(k_values):
                col = f"pass^{k}"
                # Group by num_issues and calculate mean
                grouped = mode_data.groupby("num_issues")[col].mean()

                # Create a series with all possible num_issues values
                full_series = pd.Series(index=all_num_issues, dtype=float)
                full_series.update(grouped)
                full_series = full_series.fillna(0)  # Fill missing values with 0

                # Split data into -1 and non--1 values
                no_solution_idx = all_num_issues.index(-1)
                regular_indices = [
                    i for i in range(len(all_num_issues)) if i != no_solution_idx
                ]

                # Plot regular values with line and circle markers
                ax.plot(
                    [x_pos[i] for i in regular_indices],
                    [full_series.values[i] for i in regular_indices],
                    marker="o",
                    linewidth=2,
                    color=colors[k_idx],
                    label=f"k={k}",
                )

                # Plot -1 value with different marker (square) and no connecting line
                ax.plot(
                    x_pos[no_solution_idx],
                    full_series.values[no_solution_idx],
                    marker="s",
                    markersize=8,
                    color=colors[k_idx],
                    linestyle="",
                )

            # Add vertical dashed line separating -1 from other values
            no_solution_idx = all_num_issues.index(-1)
            ax.axvline(
                x=x_pos[no_solution_idx] + 0.5,
                color="gray",
                linestyle="--",
                alpha=0.5,
            )

            # Only show x-axis label on bottom figure
            if mode_idx == len(modes) - 1:
                ax.set_xlabel("Number of Issues", fontsize=12)
            else:
                ax.set_xlabel("")

            ax.set_ylabel("Pass^k", fontsize=12)
            ax.grid(True, alpha=0.3)
            # Only show legend on top figure
            if mode_idx == 0:
                ax.legend(fontsize=12)
            current_top = ax.get_ylim()[1]
            ax.set_ylim(bottom=0, top=current_top * 1.1)  # Add 10% padding to top
            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Set x-axis ticks and labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [str(int(x)) if x != -1 else "transfer" for x in all_num_issues],
                fontsize=12,
            )
            ax.tick_params(axis="y", labelsize=12)

        plt.tight_layout()
        file_name = f"pass_k_vs_num_issues_{telecom_version}_all_modes_{llm}.pdf"
        plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close()

        # Create a new figure for pass^k vs num_issues with one curve per LLM
        modes = [m for m in MODES if m in use_modes]  # Ensure order matches MODES
        fig, axes = plt.subplots(
            len(modes), 1, figsize=(8, 4 * len(modes))
        )  # Reduced width from 10 to 8
        if len(modes) == 1:
            axes = [axes]

        # Get unique k values
        k_values = sorted(
            [
                int(col.split("^")[1])
                for col in df_pass_hat_k.columns
                if col.startswith("pass^")
            ]
        )
        # Use first k value for this plot
        k = k_values[0]

        # Plot for each mode
        for mode_idx, mode in enumerate(modes):
            ax = axes[mode_idx]
            ax.set_title(mode, fontsize=12)  # Removed "Mode: " prefix

            # Create a second y-axis for task proportions
            ax2 = ax.twinx()
            ax2.set_ylabel("Proportion of Tasks (%)", fontsize=12, color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

            # Plot task proportions as a background bar chart (using the same proportions as above)
            ax2.bar(
                x_pos,
                task_proportions.values,
                alpha=0.2,
                color="gray",
                label="Task Proportion",
            )
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(
                [str(int(x)) if x != -1 else "transfer" for x in all_num_issues],
                fontsize=12,
            )
            ax2.set_ylim(0, max(task_proportions.values) * 1.1)

            # Filter data for this mode
            mode_data = df_pass_hat_k_filtered[df_pass_hat_k_filtered["mode"] == mode]

            # Plot each LLM
            for llm in mode_data["llm"].unique():
                llm_data = mode_data[mode_data["llm"] == llm]
                col = f"pass^{k}"

                # Group by num_actions and calculate mean
                grouped = llm_data.groupby("num_issues")[col].mean()

                # Create a series with all possible num_actions values
                full_series = pd.Series(index=all_num_issues, dtype=float)
                full_series.update(grouped)
                full_series = full_series.fillna(0)  # Fill missing values with 0

                # Split data into -1 and non--1 values
                no_solution_idx = all_num_issues.index(-1)
                regular_indices = [
                    i for i in range(len(all_num_issues)) if i != no_solution_idx
                ]

                # Plot regular values with line and circle markers
                ax.plot(
                    [x_pos[i] for i in regular_indices],
                    [full_series.values[i] for i in regular_indices],
                    marker="o",
                    linewidth=2,
                    label=llm,
                )

                # Plot -1 value with different marker (square) and no connecting line
                ax.plot(
                    x_pos[no_solution_idx],
                    full_series.values[no_solution_idx],
                    marker="s",
                    markersize=8,
                    linestyle="",
                )

            # Add vertical dashed line separating -1 from other values
            no_solution_idx = all_num_issues.index(-1)
            ax.axvline(
                x=x_pos[no_solution_idx] + 0.5,
                color="gray",
                linestyle="--",
                alpha=0.5,
            )

            # Only show x-axis label on bottom figure
            if mode_idx == len(modes) - 1:
                ax.set_xlabel("Number of Issues", fontsize=12)
            else:
                ax.set_xlabel("")

            ax.set_ylabel("Pass^k", fontsize=12)
            ax.grid(True, alpha=0.3)
            # Only show legend on top figure
            if mode_idx == 0:
                ax.legend(fontsize=12)
            current_top = ax.get_ylim()[1]
            ax.set_ylim(bottom=0, top=current_top * 1.1)  # Add 10% padding to top
            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Set x-axis ticks and labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [str(int(x)) if x != -1 else "transfer" for x in all_num_issues],
                fontsize=12,
            )
            ax.tick_params(axis="y", labelsize=12)

        plt.tight_layout()
        file_name = (
            f"pass_k_vs_num_issues_{telecom_version}_all_modes_llm_comparison.pdf"
        )
        plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close()


def get_reward_actions_analysis(
    fig_dir: Path,
    df_reward_actions_analysis: pd.DataFrame,
    telecom_version: str = "telecom",
):
    """
    Creates a table showing the success rate of each action in the telecom domain.

    This function generates a table where:
    - Each row represents an action name
    - Each column represents a specific LLM-mode combination
    - Values show the success rate (proportion of successful executions) for each action
    - Includes a column showing the frequency of each action across all trials

    Args:
        fig_dir (Path): Directory where the analysis results will be saved
        df_reward_actions_analysis (pd.DataFrame): DataFrame containing action analysis data with columns:
            - requestor: Who made the action (assistant/user)
            - action_name: Name of the action
            - action: Full function format
            - action_match: Boolean success indicator
            - task_id: Task identifier
            - trial: Trial number
    """
    # Filter for telecom domain only
    df_telecom = df_reward_actions_analysis[
        df_reward_actions_analysis["domain"] == telecom_version
    ]
    if len(df_telecom) == 0:
        logger.warning("No data found for telecom domain. Skipping...")
        return

    # Calculate total frequency of each action across all trials
    action_freq = df_telecom.groupby("action_name").size()
    total_actions = action_freq.sum()
    action_freq = action_freq / total_actions  # Convert to proportion

    # Group by action_name, llm, and mode to calculate success rates
    df_grouped = df_telecom.groupby(["action_name", "llm", "mode"])["action_match"].agg(
        ["count", "sum"]
    )
    df_grouped["success_rate"] = df_grouped["sum"] / df_grouped["count"]

    # Pivot the table to get success rates for each action across different LLM-mode combinations
    df_table = df_grouped["success_rate"].unstack(["llm", "mode"]).reset_index()

    # Add frequency column
    df_table["frequency"] = df_table["action_name"].map(action_freq)

    # Sort by action_name for better readability
    df_table = df_table.sort_values("action_name")

    # Save to CSV
    file_name = f"action_success_rates_{telecom_version}.csv"
    df_table.to_csv(fig_dir / file_name, index=False)


def plot_reward_analysis(fig_dir: Path, df_reward_analysis: pd.DataFrame):
    """
    Creates multiple plots analyzing failure reasons and write action performance.

    This function generates several types of plots:
    1. Bar charts showing failure proportions for database checks and communication
       for each LLM in airline and retail domains
    2. A summary bar chart showing total failure proportions across all LLMs
    3. Histograms showing the distribution of correct write actions in failed tasks
       for different task length categories in the telecom domain

    Each plot includes:
    - Appropriate labels and legends
    - Value annotations
    - Clear visual distinction between different categories
    - Grid lines where appropriate

    Args:
        fig_dir (Path): Directory where the figures will be saved
        df_reward_analysis (pd.DataFrame): DataFrame containing reward analysis data
    """
    # Filter for the specified user simulator, base mode, and selected domains
    df_reward_analysis = df_reward_analysis[(df_reward_analysis["mode"] == "default")]

    # Create a figure with subplots for each LLM
    llms = df_reward_analysis["llm"].unique()
    n_llms = len(llms)
    _, axes = plt.subplots(n_llms, 1, figsize=(8, 5 * n_llms))
    if n_llms == 1:
        axes = [axes]

    for i, llm in enumerate(llms):
        ax = axes[i]
        llm_data = df_reward_analysis[df_reward_analysis["llm"] == llm]

        # Calculate proportions for each domain and factor
        failure_data = []
        for domain in ["airline", "retail"]:
            domain_data = llm_data[llm_data["domain"] == domain]
            if len(domain_data) > 0:
                # Calculate database proportions
                db_false = domain_data[~domain_data["database"]].shape[0]
                db_total = domain_data[
                    domain_data["database"].isin([True, False])
                ].shape[0]
                db_prop = db_false / db_total if db_total > 0 else 0

                # Calculate communication proportions
                comm_false = domain_data[~domain_data["communication"]].shape[0]
                comm_total = domain_data[
                    domain_data["communication"].isin([True, False])
                ].shape[0]
                comm_prop = comm_false / comm_total if comm_total > 0 else 0

                failure_data.append(
                    {"domain": domain, "database": db_prop, "communication": comm_prop}
                )

        if failure_data:
            df_failures = pd.DataFrame(failure_data)

            # Create bar chart
            x = np.arange(2)  # 2 domains
            width = 0.35  # Width of bars

            # Plot bars side by side
            ax.bar(
                x - width / 2,
                df_failures["database"],
                width,
                label="DB Check",
                color="red",
                alpha=0.7,
            )
            ax.bar(
                x + width / 2,
                df_failures["communication"],
                width,
                label="Communicate Info",
                color="blue",
                alpha=0.7,
            )

            # Add value labels on top of bars
            for j, (db_prop, comm_prop) in enumerate(
                zip(df_failures["database"], df_failures["communication"])
            ):
                ax.text(
                    j - width / 2,
                    db_prop + 0.02,
                    f"{db_prop:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )
                ax.text(
                    j + width / 2,
                    comm_prop + 0.02,
                    f"{comm_prop:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

            ax.set_title(f"LLM: {llm}", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(["Airline", "Retail"], fontsize=12)
            ax.set_ylabel("Proportion of Failures", fontsize=12)
            # Only show legend on the top figure
            if i == 0:
                ax.legend(fontsize=12)
            ax.set_ylim(bottom=0)  # Keep bottom at 0, let top adjust automatically

            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    file_name = f"failure_analysis_airline_retail.pdf"
    plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
    plt.close()

    # Create a summary plot showing total proportions across all LLMs
    plt.figure(figsize=(8, 6))

    # Calculate total proportions for each domain and factor
    failure_data = []
    for domain in ["airline", "retail"]:
        domain_data = df_reward_analysis[df_reward_analysis["domain"] == domain]
        if len(domain_data) > 0:
            # Calculate database proportions
            db_false = domain_data[~domain_data["database"]].shape[0]
            db_total = domain_data[domain_data["database"].isin([True, False])].shape[0]
            db_prop = db_false / db_total if db_total > 0 else 0

            # Calculate communication proportions
            comm_false = domain_data[~domain_data["communication"]].shape[0]
            comm_total = domain_data[
                domain_data["communication"].isin([True, False])
            ].shape[0]
            comm_prop = comm_false / comm_total if comm_total > 0 else 0

            failure_data.append(
                {"domain": domain, "database": db_prop, "communication": comm_prop}
            )

    if failure_data:
        df_failures = pd.DataFrame(failure_data)

        # Create bar chart
        x = np.arange(2)  # 2 domains
        width = 0.35  # Width of bars

        # Plot bars side by side
        plt.bar(
            x - width / 2,
            df_failures["database"],
            width,
            label="DB Check",
            color="red",
            alpha=0.7,
        )
        plt.bar(
            x + width / 2,
            df_failures["communication"],
            width,
            label="Communicate Info",
            color="blue",
            alpha=0.7,
        )

        # Add value labels on top of bars
        for j, (db_prop, comm_prop) in enumerate(
            zip(df_failures["database"], df_failures["communication"])
        ):
            plt.text(
                j - width / 2,
                db_prop + 0.02,
                f"{db_prop:.1%}",
                ha="center",
                va="bottom",
                fontsize=12,
            )
            plt.text(
                j + width / 2,
                comm_prop + 0.02,
                f"{comm_prop:.1%}",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        plt.xticks(x, ["Airline", "Retail"], fontsize=12)
        plt.ylabel("Proportion of Failures", fontsize=12)
        plt.legend(fontsize=10, framealpha=0.9)
        plt.ylim(bottom=0)  # Keep bottom at 0, let top adjust automatically

        # Remove top and right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()
        file_name = f"total_failure_analysis_airline_retail.pdf"
        plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close()

    # Create a new figure for proportion of correct write actions in failed tasks
    df_failed = df_reward_analysis[
        (df_reward_analysis["mode"] == "default")
        & (df_reward_analysis["domain"] == "telecom")  # Only telecom domain
    ]

    # Define task length categories
    task_categories = {
        "short": (1, 3),  # 1-3 actions
        "medium": (4, 6),  # 4-6 actions
        "long": (7, float("inf")),  # 7+ actions
    }

    # Create histogram bins from 0 to 1 in steps of 0.1
    bins = np.arange(0, 1.1, 0.1)
    bin_width = 0.1  # Width of each bin
    bar_width = bin_width / (
        len(df_failed["llm"].unique()) + 1
    )  # Make bars thinner than bin width
    colors = ["blue", "red"]  # Different colors for each LLM

    # Create a figure for each task length category
    for category, (min_actions, max_actions) in task_categories.items():
        plt.figure(figsize=(8, 4))

        # Filter data for this category
        if max_actions == float("inf"):
            category_data = df_failed[df_failed["num_write_action"] >= min_actions]
        else:
            category_data = df_failed[
                (df_failed["num_write_action"] >= min_actions)
                & (df_failed["num_write_action"] <= max_actions)
            ]

        # Calculate proportions for each LLM
        for llm_idx, llm in enumerate(category_data["llm"].unique()):
            llm_data = category_data[category_data["llm"] == llm]

            if len(llm_data) > 0:
                # Calculate proportion of correct write actions
                llm_data["prop_correct"] = (
                    llm_data["num_correct_write_action"] / llm_data["num_write_action"]
                )

                # Calculate histogram
                hist, bin_edges = np.histogram(
                    llm_data["prop_correct"], bins=bins, density=False
                )

                # Calculate bar positions
                x = bin_edges[:-1] + (llm_idx + 1) * bar_width

                # Plot bars
                plt.bar(
                    x,
                    hist,
                    width=bar_width,
                    alpha=0.7,
                    color=colors[llm_idx],
                    label=llm,
                )

        # Add title and labels
        if max_actions == float("inf"):
            title = f"Distribution of Correct Actions in Telecom Domain\nTasks with {min_actions}+ Actions"
        else:
            title = f"Distribution of Correct Actions in Telecom Domain\nTasks with {min_actions}-{max_actions} Actions"

        plt.title(title, fontsize=12)
        plt.xlabel("Proportion of Correct Actions", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.xticks(bins + bin_width / 2, [f"{b:.1f}" for b in bins], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, framealpha=0.9)

        # Remove top and right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()
        file_name = f"failed_tasks_write_actions_{category}.pdf"
        plt.savefig(fig_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close()


def get_cost_info(fig_dir: Path, df: pd.DataFrame) -> None:
    """
    Get the cost info for the given dataframe.
    """
    df = df[(df["mode"] == "default")]
    with open(fig_dir / "cost_info.txt", "w") as f:
        f.write("Mean cost per LLM:\n")
        f.write(df.groupby(["llm"])[["agent_cost", "user_cost"]].mean().to_string())
        f.write("\n\nSum cost per LLM:\n")
        f.write(df.groupby(["llm"])[["agent_cost", "user_cost"]].sum().to_string())
    # df.reset_index(inplace=True)
    # return df


def analyze_results(exp_dir: Path):
    """
    Analyze the results of the given experiment and create plots for pass^k metrics.

    Args:
        exp_dir (Path): Path to the experiment directory containing simulation results
    """
    results = get_simulation_results(exp_dir)
    rows = []
    dfs = []
    dfs_pass_hat_k = []
    dfs_reward_analysis = []
    dfs_reward_actions_analysis = []
    for params, simulation_results in results:
        ConsoleDisplay.console.print(
            f"Analyzing results for {params['llm']} on {params['domain']} in {params['mode']} mode ..."
        )
        params["mode"] = MODES_MAP[params["mode"]]
        row = deepcopy(params)
        metrics = compute_metrics(simulation_results)
        df, df_pass_hat_k = prepare_dfs(simulation_results)
        df["llm"] = params["llm"]
        df["domain"] = params["domain"]
        df["mode"] = params["mode"]
        dfs.append(df)
        df_pass_hat_k.reset_index(inplace=True)
        df_pass_hat_k["llm"] = params["llm"]
        df_pass_hat_k["domain"] = params["domain"]
        df_pass_hat_k["mode"] = params["mode"]
        dfs_pass_hat_k.append(df_pass_hat_k)
        row.update(metrics.as_dict())
        rows.append(row)
        df_reward_analysis = result_reward_analysis(simulation_results)
        df_reward_analysis["llm"] = params["llm"]
        df_reward_analysis["domain"] = params["domain"]
        df_reward_analysis["mode"] = params["mode"]
        dfs_reward_analysis.append(df_reward_analysis)
        df_reward_actions_analysis = result_reward_actions_analysis(simulation_results)
        df_reward_actions_analysis["llm"] = params["llm"]
        df_reward_actions_analysis["domain"] = params["domain"]
        df_reward_actions_analysis["mode"] = params["mode"]
        dfs_reward_actions_analysis.append(df_reward_actions_analysis)
    df = pd.concat(dfs)
    df_metrics = pd.DataFrame(rows)
    df_pass_hat_k = pd.concat(dfs_pass_hat_k)
    fig_dir = exp_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df_reward_analysis = pd.concat(dfs_reward_analysis)
    df_reward_actions_analysis = pd.concat(dfs_reward_actions_analysis)
    get_cost_info(fig_dir, df)

    plot_pass_k_metrics_per_llm_per_domain_bar_chart(fig_dir, df_pass_hat_k)

    ## TELECOM  ONLY
    for telecom_version in ["telecom", "telecom-workflow"]:
        # Create new plot for impact of communication on performance
        plot_avg_pass_k_metrics_per_llm_per_mode(
            fig_dir, df_metrics, telecom_version=telecom_version
        )
        plot_pass_one_metrics_per_llm_per_mode(
            fig_dir, df_metrics, telecom_version=telecom_version
        )
        # impact_of_solo_mode_on_performance(fig_dir, df_metrics)

        # plot_pass_k_metrics_per_llm_per_mode(
        #     fig_dir, df_pass_hat_k, telecom_version=telecom_version
        # )

        # # Create new plot for pass^k vs num_actions
        # plot_pass_k_vs_num_actions(
        #     fig_dir,
        #     df_pass_hat_k,
        #     use_modes=["default", "no-user"],
        #     telecom_version=telecom_version,
        # )

        # Call the new function to create the all LLMs comparison plot
        plot_pass_k_vs_num_actions_all_llms(
            fig_dir,
            df_pass_hat_k,
            use_modes=["default", "no-user"],
            telecom_version=telecom_version,
        )
        plot_pass_k_vs_num_issues_all_llms(
            fig_dir,
            df_pass_hat_k,
            use_modes=["default", "no-user"],
            telecom_version=telecom_version,
        )

        # Create new plot for pass^k vs num_issues
        # plot_pass_k_vs_num_issues(
        #     fig_dir,
        #     df_pass_hat_k,
        #     use_modes=["default", "no-user"],
        #     telecom_version=telecom_version,
        # )

        # Create new plot for results per intent in telecom domain
        results_per_intent_telecom(
            fig_dir, df_pass_hat_k, telecom_version=telecom_version
        )

        # Create new plot for results per persona in telecom domain
        results_per_persona_telecom(
            fig_dir, df_pass_hat_k, telecom_version=telecom_version
        )

        # # Create new plot for comparing personas to solo mode
        # comparing_personas_to_solo_mode(
        #     fig_dir, df_pass_hat_k, telecom_version=telecom_version
        # )

        # # Create new plot for comparing personas to solo mode
        # comparing_personas_to_solo_mode(fig_dir, df_pass_hat_k, gt=True)

        # # Create new plot for reward analysis
        # plot_reward_analysis(fig_dir, df_reward_analysis)

        # Create new plot for failed tasks write actions
        get_reward_actions_analysis(
            fig_dir, df_reward_actions_analysis, telecom_version=telecom_version
        )
