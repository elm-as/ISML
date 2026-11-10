"""Unit tests for single-student CLI triage utility."""

import pandas as pd
from predict import format_triage_report, triage_student
from src.feature_engineering.behavioral_features import compute_behavioral_features


def test_triage_student_extracts_correct_fields() -> None:
    """Verify that student triage accurately pulls metrics and formats report."""
    data = {
        "Student_ID": ["STU_9999"],
        "Age": [20],
        "Gender": ["Male"],
        "Academic_Level": ["Undergraduate"],
        "Primary_Platform": ["YouTube"],
        "Daily_Usage_Hours": [5.5],
        "Weekend_Extra_Hours": [2.0],
        "Device_Type": ["Smartphone"],
        "Sleep_Duration_Hours": [7.0],
        "Sleep_Quality_Score": [3],
        "Late_Night_Usage": [True],
        "Social_Comparison_Frequency": ["Sometimes"],
        "Perceived_Stress_Score": [16.0],
        "Mental_Health_Index": [78],
        "Academic_Performance_GPA": [3.35],
        "Overall_Impact": ["Beneficial"],
    }
    df = pd.DataFrame(data)
    augmented = compute_behavioral_features(df)

    res = triage_student("STU_9999", augmented)
    assert res is not None
    assert res["student_id"] == "STU_9999"
    assert res["daily_hours"] == 5.5
    assert res["overall_impact"] == "Beneficial"

    report = format_triage_report(res)
    assert "STU_9999" in report
    assert "Screen-to-Sleep Ratio" in report

