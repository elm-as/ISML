"""Immutable domain types and evaluation metrics for ISML."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StudentSurveyRecord:
    """Immutable single student behavioral and psychometric record."""

    student_id: str
    age: int
    gender: str
    academic_level: str
    primary_platform: str
    daily_usage_hours: float
    weekend_extra_hours: float
    device_type: str
    sleep_duration_hours: float
    sleep_quality_score: int
    late_night_usage: bool
    social_comparison_frequency: str
    perceived_stress_score: float
    mental_health_index: int
    academic_performance_gpa: float
    overall_impact: str


@dataclass(frozen=True)
class BehavioralFeatureMatrix:
    """Engineered matrix structure for multi-task modeling."""

    features: pd.DataFrame
    target_impact: np.ndarray
    target_stress: np.ndarray
    target_gpa: np.ndarray
    feature_names: Tuple[str, ...]
    student_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ImpactClassificationMetrics:
    """Evaluation metrics for 3-class overall impact classification."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    multi_class_log_loss: float
    per_class_f1: Dict[str, float]


@dataclass(frozen=True)
class PsychometricRegressionMetrics:
    """Evaluation metrics for continuous stress and GPA forecasting."""

    rmse: float
    mae: float
    r2_score: float
    spearman_rho: float
    spearman_pvalue: float


@dataclass(frozen=True)
class PipelineRunResult:
    """Consolidated outputs and artifacts of the complete ISML pipeline."""

    classification_metrics: ImpactClassificationMetrics
    stress_metrics: PsychometricRegressionMetrics
    feature_importance_df: pd.DataFrame
    artifact_paths: Tuple[str, ...]
