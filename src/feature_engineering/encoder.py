"""Categorical and feature matrix encoding for ISML models."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.core.types import BehavioralFeatureMatrix

IMPACT_TO_LABEL: Dict[str, int] = {
    "Beneficial": 0,
    "Neutral": 1,
    "Negative": 2,
}

LABEL_TO_IMPACT: Dict[int, str] = {v: k for k, v in IMPACT_TO_LABEL.items()}

SOCIAL_COMPARISON_MAP: Dict[str, int] = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Frequently": 3,
    "Always": 4,
}

ACADEMIC_LEVEL_MAP: Dict[str, int] = {
    "High School": 0,
    "Undergraduate": 1,
    "Postgraduate": 2,
}

GENDER_MAP: Dict[str, int] = {
    "Female": 0,
    "Male": 1,
    "Non-Binary": 2,
    "Prefer not to say": 3,
}

DEVICE_MAP: Dict[str, int] = {
    "Smartphone": 0,
    "Laptop/PC": 1,
    "Tablet": 2,
}

PLATFORM_MAP: Dict[str, int] = {
    "Instagram": 0,
    "TikTok": 1,
    "YouTube": 2,
    "Snapchat": 3,
    "Reddit": 4,
    "X (Twitter)": 5,
    "LinkedIn": 6,
}


def build_feature_matrix(df: pd.DataFrame) -> BehavioralFeatureMatrix:
    """Transform augmented survey dataframe into structured modeling matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Augmented survey dataset containing behavioral features.

    Returns
    -------
    BehavioralFeatureMatrix
        Container with numerical feature array and target vectors.
    """
    encoded = df.copy()

    # Categorical mapping
    encoded["social_comparison_code"] = (
        encoded["Social_Comparison_Frequency"].map(SOCIAL_COMPARISON_MAP).fillna(2)
    )
    encoded["academic_level_code"] = (
        encoded["Academic_Level"].map(ACADEMIC_LEVEL_MAP).fillna(1)
    )
    encoded["gender_code"] = encoded["Gender"].map(GENDER_MAP).fillna(0)
    encoded["device_code"] = encoded["Device_Type"].map(DEVICE_MAP).fillna(0)
    encoded["platform_code"] = (
        encoded["Primary_Platform"].map(PLATFORM_MAP).fillna(0)
    )
    encoded["late_night_int"] = encoded["Late_Night_Usage"].astype(int)

    feature_cols = [
        "Age",
        "Daily_Usage_Hours",
        "Weekend_Extra_Hours",
        "Sleep_Duration_Hours",
        "Sleep_Quality_Score",
        "late_night_int",
        "Perceived_Stress_Score",
        "Mental_Health_Index",
        "Academic_Performance_GPA",
        "screen_to_sleep_ratio",
        "weekly_screen_load_hours",
        "sleep_debt_hours",
        "nocturnal_friction_index",
        "stress_wellbeing_disparity",
        "academic_vulnerability_flag",
        "social_comparison_code",
        "academic_level_code",
        "gender_code",
        "device_code",
        "platform_code",
    ]

    features = encoded[feature_cols].astype(float)

    # Encode targets
    y_impact = encoded["Overall_Impact"].map(IMPACT_TO_LABEL).values
    y_stress = encoded["Perceived_Stress_Score"].values.astype(float)
    y_gpa = encoded["Academic_Performance_GPA"].values.astype(float)

    student_ids = tuple(encoded["Student_ID"].tolist())
    feature_names = tuple(feature_cols)

    return BehavioralFeatureMatrix(
        features=features,
        target_impact=y_impact,
        target_stress=y_stress,
        target_gpa=y_gpa,
        feature_names=feature_names,
        student_ids=student_ids,
    )

