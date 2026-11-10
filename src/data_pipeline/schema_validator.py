"""Schema and domain constraint validator for ISML survey records."""

from typing import List, Tuple
import pandas as pd


EXPECTED_COLUMNS: Tuple[str, ...] = (
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

VALID_IMPACTS: Tuple[str, ...] = ("Beneficial", "Neutral", "Negative")


def validate_survey_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verify that survey dataframe conforms to physical and domain constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or preprocessed student survey dataset.

    Returns
    -------
    Tuple[bool, List[str]]
        Boolean validation outcome and list of violations.
    """
    errors: List[str] = []

    if df.empty:
        errors.append("Survey dataset is completely empty")
        return False, errors

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing mandatory columns: {missing_cols}")
        return False, errors

    if df["Student_ID"].isnull().any():
        errors.append("Student_ID contains null values")

    if (df["Age"] < 15).any() or (df["Age"] > 30).any():
        errors.append("Age values outside valid range [15, 30]")

    if (df["Daily_Usage_Hours"] < 0).any():
        errors.append("Negative screen hours detected")

    if (df["Sleep_Duration_Hours"] < 0).any():
        errors.append("Negative sleep duration detected")

    if not df["Sleep_Quality_Score"].isin([1, 2, 3, 4, 5]).all():
        errors.append("Sleep_Quality_Score must be integers in [1, 5]")

    invalid_impacts = set(df["Overall_Impact"].unique()) - set(VALID_IMPACTS)
    if invalid_impacts:
        errors.append(f"Invalid Overall_Impact categories: {invalid_impacts}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_feature_matrix_shapes(
    features: pd.DataFrame,
    target_impact: pd.Series,
    target_distress: pd.Series,
) -> None:
    """Ensure feature matrix and target vectors match in row dimension.

    Parameters
    ----------
    features : pd.DataFrame
        Design feature matrix.
    target_impact : pd.Series
        Encoded categorical impact target.
    target_distress : pd.Series
        Continuous distress / GPA target.

    Raises
    ------
    ValueError
        If dimensions do not align or feature matrix is empty.
    """
    if features.empty:
        raise ValueError("Feature matrix cannot be empty")

    n_samples = len(features)
    if len(target_impact) != n_samples or len(target_distress) != n_samples:
        raise ValueError(
            f"Dimension mismatch: features={n_samples}, "
            f"target_impact={len(target_impact)}, target_distress={len(target_distress)}"
        )
