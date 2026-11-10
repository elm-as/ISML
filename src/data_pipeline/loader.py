"""Data ingestion and preprocessing for ISML survey telemetry."""

from pathlib import Path
from typing import Tuple
import pandas as pd

from src.data_pipeline.schema_validator import validate_survey_schema


def load_raw_survey(file_path: Path) -> pd.DataFrame:
    """Load survey CSV, impute minor missing values, and validate schema.

    Parameters
    ----------
    file_path : Path
        Filesystem path to raw survey CSV.

    Returns
    -------
    pd.DataFrame
        Sanitized and validated survey dataframe.

    Raises
    ------
    FileNotFoundError
        If raw CSV does not exist.
    ValueError
        If dataset fails schema validation.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Survey CSV file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Impute missing values deterministically with medians
    if "Perceived_Stress_Score" in df.columns and df["Perceived_Stress_Score"].isnull().any():
        median_stress = float(df["Perceived_Stress_Score"].median())
        df["Perceived_Stress_Score"] = df["Perceived_Stress_Score"].fillna(median_stress)

    if "Academic_Performance_GPA" in df.columns and df["Academic_Performance_GPA"].isnull().any():
        median_gpa = float(df["Academic_Performance_GPA"].median())
        df["Academic_Performance_GPA"] = df["Academic_Performance_GPA"].fillna(median_gpa)

    is_valid, errors = validate_survey_schema(df)
    if not is_valid:
        raise ValueError(f"Survey schema validation failed: {errors}")

    return df


def save_processed_survey(df: pd.DataFrame, output_path: Path) -> None:
    """Persist engineered survey dataset into Parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and feature-augmented survey dataframe.
    output_path : Path
        Destination Parquet file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
