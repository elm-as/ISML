"""Specialized publication figures for Cohort 2 (Global) and Cohort 3 (Teen)."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def plot_teen_depression_clinical_curves(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    optimal_threshold: float,
    save_path: Path,
) -> None:
    """Render dual-panel ROC and Precision-Recall curves for adolescent depression detection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, oof_probs)
    idx = int(np.argmin(np.abs(roc_thresholds - optimal_threshold)))
    ax1.plot(fpr, tpr, color="#2563EB", linewidth=2.5, label="LightGBM Classifier")
    ax1.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=1.2, label="Random Guess")
    ax1.scatter(
        [fpr[idx]],
        [tpr[idx]],
        color="#DC2626",
        s=90,
        zorder=5,
        label=f"Youden J Cutoff ({optimal_threshold:.2f})",
    )
    ax1.set_title("Cohort 3: Adolescent Depression ROC Curve", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, labelpad=6)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10, labelpad=6)
    ax1.legend(loc="lower right", frameon=True)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, oof_probs)
    ax2.plot(recall, precision, color="#059669", linewidth=2.5, label="PR Curve (Severe 2.6% Imbalance)")
    baseline = np.mean(y_true)
    ax2.axhline(baseline, color="#94A3B8", linestyle="--", linewidth=1.2, label=f"Prevalence Baseline ({baseline:.1%})")
    ax2.set_title("Cohort 3: Adolescent Precision-Recall Trade-off", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Recall (Clinical Coverage)", fontsize=10, labelpad=6)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=10, labelpad=6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_teen_protective_factors(
    teen_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Render physical activity and sleep comparison across adolescent depression status."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    label_map = {0: "Not Depressed (n=1,169)", 1: "Clinical Depression (n=31)"}
    plot_df = teen_df.copy()
    plot_df["status"] = plot_df["depression_label"].map(label_map)

    palette = {"Not Depressed (n=1,169)": "#0284C7", "Clinical Depression (n=31)": "#EF4444"}

    # Sleep Hours Boxplot
    sns.boxplot(
        data=plot_df,
        x="status",
        y="sleep_hours",
        palette=palette,
        hue="status",
        legend=False,
        ax=ax1,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    ax1.axhline(8.5, color="#10B981", linestyle="--", linewidth=1.5, label="Recommended Teen Sleep (8.5h)")
    ax1.set_title("Adolescent Sleep Duration by Depression Status", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Clinical Status", fontsize=10, labelpad=6)
    ax1.set_ylabel("Average Nightly Sleep (Hours)", fontsize=10, labelpad=6)
    ax1.legend(loc="upper right", frameon=True)

    # Physical Activity Boxplot
    sns.barplot(
        data=plot_df,
        x="status",
        y="physical_activity",
        palette=palette,
        hue="status",
        legend=False,
        ax=ax2,
        capsize=0.1,
    )
    ax2.set_title("Protective Physical Activity Engagement Level", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Clinical Status", fontsize=10, labelpad=6)
    ax2.set_ylabel("Physical Activity Level Score", fontsize=10, labelpad=6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_global_addiction_mental_health(
    global_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Render cross-national regression plot of addiction vs mental health score across 113 nations."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    palette = {"Yes": "#EF4444", "No": "#10B981", "Unsure": "#F59E0B"}

    sns.scatterplot(
        data=global_df,
        x="Addicted_Score",
        y="Mental_Health_Score",
        hue="Affects_Academic_Performance",
        palette=palette,
        alpha=0.60,
        s=45,
        ax=ax,
    )

    sns.regplot(
        data=global_df,
        x="Addicted_Score",
        y="Mental_Health_Score",
        scatter=False,
        color="#0F172A",
        line_kws={"linewidth": 2.2, "linestyle": "-"},
        ax=ax,
    )

    ax.set_title(
        "Cohort 2: Global Cross-National Addiction vs. Mental Health Score (113 Countries)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Self-Reported Addiction Score (1 to 10)", fontsize=10, labelpad=6)
    ax.set_ylabel("Mental Health Score (1 to 10)", fontsize=10, labelpad=6)
    ax.legend(title="Affects Academics", frameon=True, loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
