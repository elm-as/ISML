"""Integration tests for ISML Tri-Cohort data pipeline and models."""

import numpy as np
import pandas as pd
import pytest

from src.core.config import ModelConfig
from src.feature_engineering.global_features import compute_global_features
from src.feature_engineering.teen_features import compute_teen_features
from src.modeling.global_analyzer import analyze_global_cohort
from src.modeling.teen_depression_classifier import (
    optimize_youden_threshold,
    prepare_teen_matrix,
    train_teen_depression_classifier,
)


def test_teen_feature_engineering_derives_correct_indices() -> None:
    """Verify adolescent feature calculations."""
    data = {
        "age": [16, 17],
        "gender": ["male", "female"],
        "daily_social_media_hours": [4.0, 6.0],
        "platform_usage": ["TikTok", "Instagram"],
        "sleep_hours": [6.5, 5.0],
        "screen_time_before_sleep": [2.0, 3.0],
        "academic_performance": [3.2, 2.8],
        "physical_activity": [1.0, 0.0],
        "social_interaction_level": ["medium", "low"],
        "stress_level": [6, 8],
        "anxiety_level": [5, 9],
        "addiction_level": [7, 9],
        "depression_label": [0, 1],
    }
    df = pd.DataFrame(data)
    eng = compute_teen_features(df)

    assert "sleep_deficit_hours" in eng.columns
    assert "bedtime_screen_share" in eng.columns
    assert "physical_to_screen_ratio" in eng.columns
    assert "psychological_distress_index" in eng.columns

    # 8.5 - 6.5 = 2.0
    assert np.isclose(eng["sleep_deficit_hours"].iloc[0], 2.0)
    # 8.5 - 5.0 = 3.5
    assert np.isclose(eng["sleep_deficit_hours"].iloc[1], 3.5)


def test_global_feature_engineering_and_analysis() -> None:
    """Verify cross-national feature calculations and correlations."""
    data = {
        "Student_ID": [1, 2, 3, 4],
        "Age": [19, 21, 20, 22],
        "Gender": ["Male", "Female", "Female", "Male"],
        "Academic_Level": ["Undergraduate", "Postgraduate", "Undergraduate", "High School"],
        "Country": ["United States", "France", "United States", "Japan"],
        "Avg_Daily_Usage_Hours": [3.0, 5.0, 4.0, 2.0],
        "Most_Used_Platform": ["Instagram", "TikTok", "Instagram", "Twitter"],
        "Affects_Academic_Performance": ["No", "Yes", "Yes", "No"],
        "Sleep_Hours_Per_Night": [8.0, 6.0, 6.5, 7.5],
        "Mental_Health_Score": [8, 4, 5, 7],
        "Relationship_Status": ["Single", "In a Relationship", "Single", "Single"],
        "Conflicts_Over_Social_Media": [1, 4, 3, 1],
        "Addicted_Score": [3, 9, 8, 2],
    }
    df = pd.DataFrame(data)
    eng = compute_global_features(df)

    assert "conflict_per_usage_hour" in eng.columns
    assert "addiction_to_sleep_ratio" in eng.columns

    analysis, summary = analyze_global_cohort(eng)
    assert analysis.total_records == 4
    assert analysis.num_countries == 3
    assert analysis.addiction_mental_health_corr < 0.0  # inverse relationship


def test_youden_threshold_optimization() -> None:
    """Verify Youden J statistic threshold search."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    oof_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    opt_th, youden_j, sens, spec, f1 = optimize_youden_threshold(y_true, oof_probs)

    assert 0.0 < opt_th < 1.0
    assert youden_j > 0.8
    assert sens == 1.0
    assert spec == 1.0
