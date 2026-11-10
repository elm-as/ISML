"""Publication-grade visualization generator for ISML."""

from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def plot_behavioral_clustering_quadrant(
    df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Render scatter quadrant of daily screen time versus sleep duration."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    palette = {
        "Beneficial": "#10B981",
        "Neutral": "#F59E0B",
        "Negative": "#EF4444",
    }

    sns.scatterplot(
        data=df,
        x="Daily_Usage_Hours",
        y="Sleep_Duration_Hours",
        hue="Overall_Impact",
        palette=palette,
        alpha=0.65,
        s=40,
        ax=ax,
    )

    # Reference clinical thresholds
    ax.axvline(6.0, color="#64748B", linestyle="--", linewidth=1.2, label="Excessive Screen (>6h)")
    ax.axhline(7.0, color="#3B82F6", linestyle="--", linewidth=1.2, label="Clinical Sleep Baseline (7h)")

    ax.set_title("ISML: Daily Screen Time vs. Sleep Duration across Overall Impact Regimes", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Daily Social Media Usage (Hours/Day)", fontsize=11, labelpad=8)
    ax.set_ylabel("Nightly Sleep Duration (Hours)", fontsize=11, labelpad=8)
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix_impact(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    class_names: Tuple[str, ...],
    save_path: Path,
) -> None:
    """Render normalized 3x3 confusion matrix for overall impact."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    y_pred = np.argmax(oof_probs, axis=1)
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )

    ax.set_title("ISML: Normalized Out-of-Fold Confusion Matrix (5-Fold Ensemble)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Impact Category", fontsize=11, labelpad=8)
    ax.set_ylabel("Empirical Ground Truth Category", fontsize=11, labelpad=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_screen_time_sleep_debt(
    df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Render sleep debt versus screen load across digital platforms."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    order = df.groupby("Primary_Platform")["Daily_Usage_Hours"].median().sort_values(ascending=False).index

    sns.boxplot(
        data=df,
        x="Primary_Platform",
        y="Daily_Usage_Hours",
        order=order,
        hue="Primary_Platform",
        palette="crest",
        legend=False,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )

    ax.set_title("Daily Screen Time Distribution by Primary Social Platform", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Primary Platform", fontsize=11, labelpad=8)
    ax.set_ylabel("Daily Usage Hours", fontsize=11, labelpad=8)
    plt.xticks(rotation=25)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_gpa_stress_dose_response(
    df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Render dose-response relationship between perceived stress and academic GPA."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    palette = {
        "Beneficial": "#10B981",
        "Neutral": "#F59E0B",
        "Negative": "#EF4444",
    }

    sns.scatterplot(
        data=df,
        x="Perceived_Stress_Score",
        y="Academic_Performance_GPA",
        hue="Overall_Impact",
        palette=palette,
        alpha=0.5,
        s=35,
        ax=ax,
    )

    sns.regplot(
        data=df,
        x="Perceived_Stress_Score",
        y="Academic_Performance_GPA",
        scatter=False,
        color="#1E293B",
        line_kws={"linewidth": 2, "linestyle": "-."},
        ax=ax,
    )

    ax.set_title("Dose-Response: Perceived Stress vs. Cumulative Academic GPA", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Perceived Stress Score (PSS 0-40)", fontsize=11, labelpad=8)
    ax.set_ylabel("Cumulative GPA (1.70 - 4.00)", fontsize=11, labelpad=8)
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_feature_importance_shap(
    importance_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Render composite feature importance and attribution bar plot."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    top_10 = importance_df.head(10).sort_values("composite_importance", ascending=True)

    y_pos = np.arange(len(top_10))
    ax.barh(y_pos, top_10["composite_importance"], color="#0284C7", edgecolor="#0369A1", height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10["feature"], fontsize=10)
    ax.set_title("ISML: Top 10 Psychometric and Digital Attribution Drivers", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Composite Importance Score (%)", fontsize=11, labelpad=8)

    for i, v in enumerate(top_10["composite_importance"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
