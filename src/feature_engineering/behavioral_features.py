"""Behavioral, chronobiological, and psychometric feature engineering."""

import numpy as np
import pandas as pd


def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive quantitative chronobiological and behavioral indicators.

    Calculates:
    - screen_to_sleep_ratio: Ratio of daily social media screen time to sleep
    - weekly_screen_load_hours: Aggregate weekday + weekend screen load
    - sleep_debt_hours: Deficit relative to clinical 8.0h baseline
    - nocturnal_friction_index: Compound nocturnal usage and poor sleep quality
    - stress_wellbeing_disparity: Ratio of perceived stress to mental health index
    - academic_vulnerability_flag: Binary flag for GPA below 3.0 threshold

    Parameters
    ----------
    df : pd.DataFrame
        Input survey dataset with raw metrics.

    Returns
    -------
    pd.DataFrame
        Augmented dataframe with derived behavioral columns.
    """
    out = df.copy()

    # 1. Screen-to-Sleep Ratio
    out["screen_to_sleep_ratio"] = out["Daily_Usage_Hours"] / (
        out["Sleep_Duration_Hours"] + 1e-4
    )

    # 2. Cumulative Weekly Screen Load (5 weekdays + 2 weekend days)
    out["weekly_screen_load_hours"] = (
        5.0 * out["Daily_Usage_Hours"]
        + 2.0 * (out["Daily_Usage_Hours"] + out["Weekend_Extra_Hours"])
    )

    # 3. Sleep Debt relative to 8.0 hours
    out["sleep_debt_hours"] = np.maximum(0.0, 8.0 - out["Sleep_Duration_Hours"])

    # 4. Nocturnal Friction Index (Late night usage * sleep quality deficit)
    late_int = out["Late_Night_Usage"].astype(int)
    quality_deficit = 6 - out["Sleep_Quality_Score"]
    out["nocturnal_friction_index"] = late_int * quality_deficit

    # 5. Stress-to-Wellbeing Disparity
    out["stress_wellbeing_disparity"] = out["Perceived_Stress_Score"] / (
        out["Mental_Health_Index"] + 1.0
    )

    # 6. Academic Vulnerability Flag
    out["academic_vulnerability_flag"] = (
        out["Academic_Performance_GPA"] < 3.0
    ).astype(int)

    # 7. Target Binary Indicators
    out["is_negative_impact"] = (out["Overall_Impact"] == "Negative").astype(int)
    out["is_beneficial_impact"] = (out["Overall_Impact"] == "Beneficial").astype(int)

    return out

