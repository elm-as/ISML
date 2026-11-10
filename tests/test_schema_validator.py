"""Unit tests for ISML schema validator across cohorts."""

import pandas as pd
import pytest
from src.data_pipeline.schema_validator import (
    validate_cohort2_schema,
    validate_cohort3_schema,
    validate_feature_matrix_shapes,
    validate_survey_schema,
)


def test_schema_validator_rejects_empty_dataframe() -> None:
    """Empty dataframe must trigger immediate validation rejection."""
    df = pd.DataFrame()
    is_valid, errors = validate_survey_schema(df)
    assert not is_valid
    assert any("empty" in err.lower() for err in errors)


def test_schema_validator_rejects_missing_columns() -> None:
    """Missing required columns must be flagged."""
    df = pd.DataFrame({"Student_ID": ["STU_001"], "Age": [18]})
    is_valid, errors = validate_survey_schema(df)
    assert not is_valid
    assert any("mandatory" in err.lower() for err in errors)


def test_schema_validator_rejects_invalid_impact() -> None:
    """Unrecognized impact categories must fail validation."""
    data = {
        "Student_ID": ["STU_001"],
        "Age": [19],
        "Gender": ["Female"],
        "Academic_Level": ["Undergraduate"],
        "Primary_Platform": ["Instagram"],
        "Daily_Usage_Hours": [4.0],
        "Weekend_Extra_Hours": [1.0],
        "Device_Type": ["Smartphone"],
        "Sleep_Duration_Hours": [7.0],
        "Sleep_Quality_Score": [3],
        "Late_Night_Usage": [True],
        "Social_Comparison_Frequency": ["Sometimes"],
        "Perceived_Stress_Score": [15.0],
        "Mental_Health_Index": [80],
        "Academic_Performance_GPA": [3.4],
        "Overall_Impact": ["Catastrophic_Invalid"],
    }
    df = pd.DataFrame(data)
    is_valid, errors = validate_survey_schema(df)
    assert not is_valid
    assert any("invalid overall_impact" in err.lower() for err in errors)


def test_cohort2_schema_validator() -> None:
    """Validate Cohort 2 schema requirements."""
    df = pd.DataFrame({"Student_ID": [1], "Age": [20]})
    is_valid, errors = validate_cohort2_schema(df)
    assert not is_valid
    assert any("missing columns" in err.lower() for err in errors)


def test_cohort3_schema_validator() -> None:
    """Validate Cohort 3 schema requirements."""
    df = pd.DataFrame({"age": [16], "depression_label": [0]})
    is_valid, errors = validate_cohort3_schema(df)
    assert not is_valid
    assert any("missing columns" in err.lower() for err in errors)


def test_feature_matrix_validator_detects_shape_mismatches() -> None:
    """Ensure length mismatches raise ValueError."""
    x = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    y1 = pd.Series([0, 1, 0])
    y2 = pd.Series([10.0, 20.0])  # Length 2 instead of 3
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_feature_matrix_shapes(x, y1, y2)
