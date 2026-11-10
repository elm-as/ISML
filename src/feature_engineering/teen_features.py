"""Adolescent behavioral and lifestyle feature engineering (Cohort 3)."""

import numpy as np
import pandas as pd


def compute_teen_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive chronobiological, physical, and psychological indices for teens.

    Parameters
    ----------
    df : pd.DataFrame
        Validated raw adolescent survey telemetry.

    Returns
    -------
    pd.DataFrame
        Engineered adolescent dataset.
    """
    engineered = df.copy()

    # Adolescent sleep debt (baseline recommended: 8.5 hours for teens)
    engineered["sleep_deficit_hours"] = np.maximum(
        0.0, 8.5 - engineered["sleep_hours"].astype(float)
    )

    # Bedtime screen burden ratio
    screen_total = engineered["daily_social_media_hours"].astype(float) + 1e-4
    bedtime_screen = engineered["screen_time_before_sleep"].astype(float)
    engineered["bedtime_screen_share"] = bedtime_screen / screen_total

    # Protective ratio: physical activity vs sedentary screen time
    activity = engineered["physical_activity"].astype(float)
    engineered["physical_to_screen_ratio"] = activity / (screen_total + 1.0)

    # Psychometric distress composite
    stress = engineered["stress_level"].astype(float)
    anxiety = engineered["anxiety_level"].astype(float)
    engineered["psychological_distress_index"] = (stress + anxiety) / 2.0

    # Nocturnal circadian disruption index
    engineered["nocturnal_circadian_burden"] = bedtime_screen * engineered["sleep_deficit_hours"]

    # Digital saturation score
    addiction = engineered["addiction_level"].astype(float)
    engineered["digital_saturation_index"] = (screen_total * addiction) / 10.0

    return engineered

