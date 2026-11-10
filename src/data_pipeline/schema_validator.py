"""Schema and domain constraint validators for ISML tri-cohort datasets."""

from typing import List, Tuple
import pandas as pd


COHORT1_COLUMNS: Tuple[str, ...] = (
    "Student_ID",
    "Age",
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Device_Type",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Late_Night_Usage",
    "Social_Comparison_Frequency",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
    "Academic_Performance_GPA",
    "Overall_Impact",
)

COHORT2_COLUMNS: Tuple[str, ...] = (
    "Student_ID",
    "Age",
    "Gender",
    "Academic_Level",
    "Country",
    "Avg_Daily_Usage_Hours",
    "Most_Used_Platform",
    "Affects_Academic_Performance",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
    "Relationship_Status",
    "Conflicts_Over_Social_Media",
    "Addicted_Score",
)

COHORT3_COLUMNS: Tuple[str, ...] = (
    "age",
    "gender",
    "daily_social_media_hours",
    "platform_usage",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "social_interaction_level",
    "stress_level",
    "anxiety_level",
    "addiction_level",
    "depression_label",
)

VALID_IMPACTS: Tuple[str, ...] = ("Beneficial", "Neutral", "Negative")


def validate_survey_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verify Cohort 1 survey dataframe conforms to physical and domain constraints."""
    errors: List[str] = []

    if df.empty:
        errors.append("Cohort 1 survey dataset is completely empty")
        return False, errors

    missing_cols = [c for c in COHORT1_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing mandatory columns: {missing_cols}")
        return False, errors

    if df["Student_ID"].isnull().any():
        errors.append("Student_ID contains null values")

    if (df["Age"] < 13).any() or (df["Age"] > 35).any():
        errors.append("Age values outside valid range [13, 35]")

    if (df["Daily_Usage_Hours"] < 0).any():
        errors.append("Negative screen hours detected")

    if (df["Sleep_Duration_Hours"] < 0).any():
        errors.append("Negative sleep duration detected")

    if not df["Sleep_Quality_Score"].isin([1, 2, 3, 4, 5]).all():
        errors.append("Sleep_Quality_Score must be integers in [1, 5]")

    invalid_impacts = set(df["Overall_Impact"].unique()) - set(VALID_IMPACTS)
    if invalid_impacts:
        errors.append(f"Invalid Overall_Impact categories: {invalid_impacts}")

    return len(errors) == 0, errors


def validate_cohort2_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verify Cohort 2 global survey dataframe conforms to schema constraints."""
    errors: List[str] = []

    if df.empty:
        errors.append("Cohort 2 dataset is empty")
        return False, errors

    missing_cols = [c for c in COHORT2_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Cohort 2 missing columns: {missing_cols}")
        return False, errors

    if (df["Avg_Daily_Usage_Hours"] < 0).any():
        errors.append("Negative average daily usage hours detected in Cohort 2")

    if (df["Sleep_Hours_Per_Night"] < 0).any():
        errors.append("Negative sleep hours detected in Cohort 2")

    if (df["Addicted_Score"] < 1).any() or (df["Addicted_Score"] > 10).any():
        errors.append("Addicted_Score outside valid range [1, 10]")

    return len(errors) == 0, errors


def validate_cohort3_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verify Cohort 3 adolescent depression survey conforms to schema constraints."""
    errors: List[str] = []

    if df.empty:
        errors.append("Cohort 3 dataset is empty")
        return False, errors

    missing_cols = [c for c in COHORT3_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Cohort 3 missing columns: {missing_cols}")
        return False, errors

    if (df["age"] < 12).any() or (df["age"] > 22).any():
        errors.append("Adolescent age outside valid range [12, 22]")

    if not df["depression_label"].isin([0, 1]).all():
        errors.append("depression_label must be strictly binary [0, 1]")

    return len(errors) == 0, errors


def validate_feature_matrix_shapes(
    features: pd.DataFrame,
    target_impact: pd.Series,
    target_distress: pd.Series,
) -> None:
    """Ensure feature matrix and target vectors match in row dimension."""
    if features.empty:
        raise ValueError("Feature matrix cannot be empty")

    n_samples = len(features)
    if len(target_impact) != n_samples or len(target_distress) != n_samples:
        raise ValueError(
            f"Dimension mismatch: features={n_samples}, "
            f"target_impact={len(target_impact)}, target_distress={len(target_distress)}"
        )
