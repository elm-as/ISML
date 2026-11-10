"""Data ingestion and preprocessing for ISML tri-cohort survey telemetry."""

from pathlib import Path
from typing import Tuple
import pandas as pd

from src.data_pipeline.schema_validator import (
    validate_cohort2_schema,
    validate_cohort3_schema,
    validate_survey_schema,
)


def load_raw_survey(file_path: Path) -> pd.DataFrame:
    """Load Cohort 1 survey CSV, impute minor missing values, and validate schema."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cohort 1 survey CSV not found: {file_path}")

    df = pd.read_csv(file_path)

    if "Perceived_Stress_Score" in df.columns and df["Perceived_Stress_Score"].isnull().any():
        median_stress = float(df["Perceived_Stress_Score"].median())
        df["Perceived_Stress_Score"] = df["Perceived_Stress_Score"].fillna(median_stress)

    if "Academic_Performance_GPA" in df.columns and df["Academic_Performance_GPA"].isnull().any():
        median_gpa = float(df["Academic_Performance_GPA"].median())
        df["Academic_Performance_GPA"] = df["Academic_Performance_GPA"].fillna(median_gpa)

    is_valid, errors = validate_survey_schema(df)
    if not is_valid:
        raise ValueError(f"Cohort 1 schema validation failed: {errors}")

    return df


def load_cohort2_global(file_path: Path) -> pd.DataFrame:
    """Load Cohort 2 global multi-country survey and validate schema."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cohort 2 global survey CSV not found: {file_path}")

    df = pd.read_csv(file_path)

    # Impute missing values if any
    if df.isnull().any().any():
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        cat_cols = df.select_dtypes(include=["object"]).columns
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0])

    is_valid, errors = validate_cohort2_schema(df)
    if not is_valid:
        raise ValueError(f"Cohort 2 schema validation failed: {errors}")

    return df


def load_cohort3_teen(file_path: Path) -> pd.DataFrame:
    """Load Cohort 3 adolescent depression survey and validate schema."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cohort 3 adolescent survey CSV not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.isnull().any().any():
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        cat_cols = df.select_dtypes(include=["object"]).columns
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0])

    is_valid, errors = validate_cohort3_schema(df)
    if not is_valid:
        raise ValueError(f"Cohort 3 schema validation failed: {errors}")

    return df


def save_processed_survey(df: pd.DataFrame, output_path: Path) -> None:
    """Persist engineered survey dataset into Parquet format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
