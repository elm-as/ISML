"""Unit tests for behavioral and chronobiological feature engineering."""

import pandas as pd
from src.feature_engineering.behavioral_features import compute_behavioral_features


def test_behavioral_features_are_pure_and_deterministic() -> None:
    """Verify that feature derivations are pure and create expected indicators."""
    data = {
        "Daily_Usage_Hours": [4.0, 10.0],
        "Weekend_Extra_Hours": [1.0, 2.0],
        "Sleep_Duration_Hours": [8.0, 5.0],
        "Sleep_Quality_Score": [4, 1],
        "Late_Night_Usage": [False, True],
        "Perceived_Stress_Score": [10.0, 30.0],
        "Mental_Health_Index": [90, 45],
        "Academic_Performance_GPA": [3.8, 2.4],
        "Overall_Impact": ["Beneficial", "Negative"],
    }
    df = pd.DataFrame(data)
    augmented = compute_behavioral_features(df)

    # 1. Screen to sleep ratio
    assert augmented.loc[0, "screen_to_sleep_ratio"] == pytest.approx(4.0 / 8.0, rel=1e-3)
    assert augmented.loc[1, "screen_to_sleep_ratio"] == pytest.approx(10.0 / 5.0, rel=1e-3)

    # 2. Sleep debt
    assert augmented.loc[0, "sleep_debt_hours"] == 0.0
    assert augmented.loc[1, "sleep_debt_hours"] == 3.0

    # 3. Nocturnal friction
    assert augmented.loc[0, "nocturnal_friction_index"] == 0  # Late_Night_Usage is False
    assert augmented.loc[1, "nocturnal_friction_index"] == 5  # 1 * (6 - 1)

    # 4. Academic vulnerability
    assert augmented.loc[0, "academic_vulnerability_flag"] == 0
    assert augmented.loc[1, "academic_vulnerability_flag"] == 1

    # 5. Impact flags
    assert augmented.loc[0, "is_beneficial_impact"] == 1
    assert augmented.loc[1, "is_negative_impact"] == 1


import pytest

