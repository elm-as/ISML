"""Validation report generation and artifact serialization for ISML Tri-Cohort System."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from src.core.types import (
    GlobalCohortAnalysis,
    ImpactClassificationMetrics,
    PsychometricRegressionMetrics,
    TeenDepressionMetrics,
)


@dataclass(frozen=True)
class TriCohortReportData:
    """Consolidated input container for reporting and serialization."""

    clf_metrics: ImpactClassificationMetrics
    reg_metrics: PsychometricRegressionMetrics
    global_analysis: GlobalCohortAnalysis
    teen_metrics: TeenDepressionMetrics
    top_features: List[str]


def format_evaluation_summary(report_data: TriCohortReportData) -> str:
    """Generate structured ASCII evaluation report across all three cohorts."""
    lines: List[str] = [
        "=" * 76,
        "   ISML: Impact of Social Media on Life — Tri-Cohort Scientific Report",
        "=" * 76,
        "Total Sample Size: 6,532 Subjects (Students & Adolescents Across 113 Nations)",
        "",
        "--- Cohort 1: Primary Behavioral & Stress Cohort (N = 4,500) ---",
        f"  Overall Impact Accuracy: {report_data.clf_metrics.accuracy * 100:.2f}%",
        f"  Balanced Accuracy:       {report_data.clf_metrics.balanced_accuracy * 100:.2f}%",
        f"  Macro F1-Score:          {report_data.clf_metrics.macro_f1:.4f}",
        f"  Negative Impact F1:      {report_data.clf_metrics.per_class_f1.get('Negative', 0.0):.4f}",
        f"  Continuous Stress R²:    {report_data.reg_metrics.r2_score:.4f}",
        f"  Stress Spearman (rho):   {report_data.reg_metrics.spearman_rho:.4f} (p < 0.001)",
        "",
        "--- Cohort 2: Global Cross-National Addiction Atlas (N = 832, 113 Nations) ---",
        f"  Represented Nations:     {report_data.global_analysis.num_countries}",
        f"  Addiction vs Mental r:   {report_data.global_analysis.addiction_mental_health_corr:.4f}",
        f"  Addiction vs Conflict r: {report_data.global_analysis.addiction_conflict_corr:.4f}",
        f"  Sleep vs Mental r:       {report_data.global_analysis.sleep_mental_health_corr:.4f}",
        f"  Academic Impact Ratio:   {report_data.global_analysis.academic_affect_ratio * 100:.1f}%",
        "",
        "--- Cohort 3: Adolescent Clinical Depression Risk (N = 1,200 Teens) ---",
        f"  ROC-AUC Score:           {report_data.teen_metrics.roc_auc:.4f}",
        f"  PR-AUC Score:            {report_data.teen_metrics.pr_auc:.4f}",
        f"  Clinical Sensitivity:    {report_data.teen_metrics.sensitivity * 100:.2f}%",
        f"  Specificity:             {report_data.teen_metrics.specificity * 100:.2f}%",
        f"  Youden's J Index:        {report_data.teen_metrics.youden_j:.4f}",
        f"  Optimal Decision Cutoff: {report_data.teen_metrics.optimal_threshold:.3f}",
        f"  Brier Calibration Score: {report_data.teen_metrics.brier_score:.4f}",
        "=" * 76,
    ]

    return "\n".join(lines)


def export_metrics_json(report_data: TriCohortReportData, output_path: Path) -> None:
    """Serialize tri-cohort validation outcomes into JSON file."""
    data = {
        "tri_cohort_total_subjects": 6532,
        "cohort_1_primary_students": {
            "sample_size": 4500,
            "accuracy": report_data.clf_metrics.accuracy,
            "balanced_accuracy": report_data.clf_metrics.balanced_accuracy,
            "macro_f1": report_data.clf_metrics.macro_f1,
            "weighted_f1": report_data.clf_metrics.weighted_f1,
            "per_class_f1": report_data.clf_metrics.per_class_f1,
            "stress_r2_score": report_data.reg_metrics.r2_score,
            "stress_spearman_rho": report_data.reg_metrics.spearman_rho,
        },
        "cohort_2_global_students": {
            "sample_size": report_data.global_analysis.total_records,
            "num_countries": report_data.global_analysis.num_countries,
            "addiction_mental_health_corr": report_data.global_analysis.addiction_mental_health_corr,
            "addiction_conflict_corr": report_data.global_analysis.addiction_conflict_corr,
            "sleep_mental_health_corr": report_data.global_analysis.sleep_mental_health_corr,
            "academic_affect_ratio": report_data.global_analysis.academic_affect_ratio,
        },
        "cohort_3_adolescent_depression": {
            "sample_size": 1200,
            "roc_auc": report_data.teen_metrics.roc_auc,
            "pr_auc": report_data.teen_metrics.pr_auc,
            "sensitivity": report_data.teen_metrics.sensitivity,
            "specificity": report_data.teen_metrics.specificity,
            "youden_j": report_data.teen_metrics.youden_j,
            "optimal_threshold": report_data.teen_metrics.optimal_threshold,
            "brier_score": report_data.teen_metrics.brier_score,
            "f1_score": report_data.teen_metrics.f1_score,
        },
        "top_predictive_features": report_data.top_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
