"""Unit tests for feature matrix builder and encoding."""

import pandas as pd
from src.feature_engineering.behavioral_features import compute_behavioral_features
from src.feature_engineering.encoder import build_feature_matrix


def test_encoder_constructs_valid_feature_matrix() -> None:
    """Verify that build_feature_matrix maps all categoricals and outputs non-null matrix."""
    data = {
        "Student_ID": ["STU_001", "STU_002"],
        "Age": [19, 21],
        "Gender": ["Female", "Male"],
        "Academic_Level": ["Undergraduate", "Postgraduate"],
        "Primary_Platform": ["Instagram", "TikTok"],
        "Daily_Usage_Hours": [3.5, 9.0],
        "Weekend_Extra_Hours": [1.0, 2.5],
        "Device_Type": ["Smartphone", "Laptop/PC"],
        "Sleep_Duration_Hours": [7.5, 5.0],
        "Sleep_Quality_Score": [4, 2],
        "Late_Night_Usage": [False, True],
        "Social_Comparison_Frequency": ["Rarely", "Frequently"],
        "Perceived_Stress_Score": [12.0, 28.0],
        "Mental_Health_Index": [85, 52],
        "Academic_Performance_GPA": [3.6, 2.8],
        "Overall_Impact": ["Beneficial", "Neutral"],
    }
    df = pd.DataFrame(data)
    augmented = compute_behavioral_features(df)
    matrix = build_feature_matrix(augmented)

    assert matrix.features.shape == (2, 20)
    assert not matrix.features.isnull().any().any()
    assert matrix.target_impact[0] == 0  # Beneficial
    assert matrix.target_impact[1] == 1  # Neutral
    assert len(matrix.student_ids) == 2

