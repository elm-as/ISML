"""Immutable domain types and evaluation metrics for ISML Tri-Cohort System."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StudentSurveyRecord:
    """Immutable single student behavioral and psychometric record (Cohort 1)."""

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
class GlobalStudentSurveyRecord:
    """Immutable single global student survey record across 113 countries (Cohort 2)."""

    student_id: int
    age: int
    gender: str
    academic_level: str
    country: str
    avg_daily_usage_hours: float
    most_used_platform: str
    affects_academic_performance: str
    sleep_hours_per_night: float
    mental_health_score: int
    relationship_status: str
    conflicts_over_social_media: int
    addicted_score: int


@dataclass(frozen=True)
class TeenSurveyRecord:
    """Immutable adolescent mental health & depression survey record (Cohort 3)."""

    age: int
    gender: str
    daily_social_media_hours: float
    platform_usage: str
    sleep_hours: float
    screen_time_before_sleep: float
    academic_performance: float
    physical_activity: float
    social_interaction_level: str
    stress_level: int
    anxiety_level: int
    addiction_level: int
    depression_label: int


@dataclass(frozen=True)
class BehavioralFeatureMatrix:
    """Engineered matrix structure for multi-task modeling (Cohort 1)."""

    features: pd.DataFrame
    target_impact: np.ndarray
    target_stress: np.ndarray
    target_gpa: np.ndarray
    feature_names: Tuple[str, ...]
    student_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ImpactClassificationMetrics:
    """Evaluation metrics for 3-class overall impact classification (Cohort 1)."""

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
class TeenDepressionMetrics:
    """Evaluation metrics for adolescent binary depression risk classification (Cohort 3)."""

    roc_auc: float
    pr_auc: float
    brier_score: float
    sensitivity: float
    specificity: float
    youden_j: float
    optimal_threshold: float
    f1_score: float


@dataclass(frozen=True)
class GlobalCohortAnalysis:
    """Statistical and epidemiological cross-national findings (Cohort 2)."""

    total_records: int
    num_countries: int
    addiction_mental_health_corr: float
    addiction_conflict_corr: float
    sleep_mental_health_corr: float
    academic_affect_ratio: float
    top_platforms: Dict[str, int]


@dataclass(frozen=True)
class PipelineRunResult:
    """Consolidated outputs and artifacts of the complete tri-cohort ISML pipeline."""

    cohort1_metrics: ImpactClassificationMetrics
    cohort1_stress: PsychometricRegressionMetrics
    cohort2_analysis: GlobalCohortAnalysis
    cohort3_metrics: TeenDepressionMetrics
    feature_importance_df: pd.DataFrame
    artifact_paths: Tuple[str, ...]
