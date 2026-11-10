"""Global cross-national behavioral feature engineering (Cohort 2)."""

import numpy as np
import pandas as pd


def compute_global_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cross-national friction, sleep balance, and distress features.

    Parameters
    ----------
    df : pd.DataFrame
        Validated raw global student survey telemetry.

    Returns
    -------
    pd.DataFrame
        Engineered global survey dataset.
    """
    engineered = df.copy()

    usage = engineered["Avg_Daily_Usage_Hours"].astype(float) + 0.1
    sleep = engineered["Sleep_Hours_Per_Night"].astype(float) + 0.1
    conflicts = engineered["Conflicts_Over_Social_Media"].astype(float)
    addiction = engineered["Addicted_Score"].astype(float)
    mental_health = engineered["Mental_Health_Score"].astype(float)

    # Interpersonal friction per usage hour
    engineered["conflict_per_usage_hour"] = conflicts / usage

    # Addiction to sleep preservation ratio
    engineered["addiction_to_sleep_ratio"] = addiction / sleep

    # Subjective well-being deficit
    engineered["mental_health_deficit"] = 10.0 - mental_health

    # Binary academic disruption
    engineered["academic_disruption_flag"] = (
        engineered["Affects_Academic_Performance"].astype(str).str.strip().eq("Yes").astype(int)
    )

    # Cross-national standardized usage z-score by platform
    platform_mean = engineered.groupby("Most_Used_Platform")["Avg_Daily_Usage_Hours"].transform("mean")
    platform_std = engineered.groupby("Most_Used_Platform")["Avg_Daily_Usage_Hours"].transform("std").fillna(1.0)
    engineered["platform_normalized_usage_z"] = (usage - platform_mean) / platform_std

    return engineered
