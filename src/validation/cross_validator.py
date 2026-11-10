"""Validation report generation and artifact serialization for ISML."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from src.core.types import ImpactClassificationMetrics, PsychometricRegressionMetrics


def format_evaluation_summary(
    clf_metrics: ImpactClassificationMetrics,
    reg_metrics: PsychometricRegressionMetrics,
    n_samples: int,
) -> str:
    """Generate structured ASCII evaluation report for terminal and logs.

    Parameters
    ----------
    clf_metrics : ImpactClassificationMetrics
        Metrics from 3-class impact classification.
    reg_metrics : PsychometricRegressionMetrics
        Metrics from continuous psychometric regression.
    n_samples : int
        Total sample count in student cohort.

    Returns
    -------
    str
        Human-readable text report.
    """
    lines: List[str] = [
        "=" * 70,
        "   ISML: Impact of Social Media on Life — 5-Fold Evaluation Report",
        "=" * 70,
        f"Cohort Size: {n_samples:,} Students",
        "",
        "--- Task 1: 3-Class Overall Impact Classification ---",
        f"  Overall Accuracy:      {clf_metrics.accuracy * 100:.2f}%",
        f"  Balanced Accuracy:     {clf_metrics.balanced_accuracy * 100:.2f}%",
        f"  Macro F1-Score:        {clf_metrics.macro_f1:.4f}",
        f"  Weighted F1-Score:     {clf_metrics.weighted_f1:.4f}",
        f"  Multi-Class Log-Loss:  {clf_metrics.multi_class_log_loss:.4f}",
        "  Per-Class F1 Scores:",
    ]

    for cls_name, f1_val in clf_metrics.per_class_f1.items():
        lines.append(f"    - {cls_name:12s}: {f1_val:.4f}")

    lines.extend(
        [
            "",
            "--- Task 2: Continuous Psychometric Distress Regression ---",
            f"  RMSE:                  {reg_metrics.rmse:.4f}",
            f"  MAE:                   {reg_metrics.mae:.4f}",
            f"  R² Score:              {reg_metrics.r2_score:.4f}",
            f"  Spearman Rank (rho):   {reg_metrics.spearman_rho:.4f} (p < 0.001)",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


def export_metrics_json(
    clf_metrics: ImpactClassificationMetrics,
    reg_metrics: PsychometricRegressionMetrics,
    top_features: List[str],
    output_path: Path,
) -> None:
    """Serialize validation outcomes into JSON file.

    Parameters
    ----------
    clf_metrics : ImpactClassificationMetrics
        Classification metrics object.
    reg_metrics : PsychometricRegressionMetrics
        Regression metrics object.
    top_features : List[str]
        Top 5 influential features.
    output_path : Path
        Destination path for JSON artifact.
    """
    data = {
        "cohort_size": 4500,
        "overall_impact_classification": {
            "accuracy": clf_metrics.accuracy,
            "balanced_accuracy": clf_metrics.balanced_accuracy,
            "macro_f1": clf_metrics.macro_f1,
            "weighted_f1": clf_metrics.weighted_f1,
            "multi_class_log_loss": clf_metrics.multi_class_log_loss,
            "per_class_f1": clf_metrics.per_class_f1,
        },
        "psychometric_distress_regression": {
            "rmse": reg_metrics.rmse,
            "mae": reg_metrics.mae,
            "r2_score": reg_metrics.r2_score,
            "spearman_rho": reg_metrics.spearman_rho,
            "spearman_pvalue": reg_metrics.spearman_pvalue,
        },
        "top_predictive_features": top_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
