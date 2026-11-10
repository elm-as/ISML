"""Global configuration and hyperparameters for ISML Tri-Cohort System."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DataConfig:
    """Paths and technical specifications for survey telemetry across cohorts."""

    # Cohort 1: Primary Behavioral & Stress Survey (4,500 students)
    raw_data_path: Path = Path("data/raw/social_media_impact_on_life.csv")
    processed_data_path: Path = Path("data/processed/social_media_engineered.parquet")

    # Cohort 2: Global Cross-National Survey (832 students, 113 countries)
    global_raw_path: Path = Path("data/raw/global_student_social_media_832.csv")
    global_processed_path: Path = Path("data/processed/global_student_engineered.parquet")

    # Cohort 3: Adolescent Clinical Depression Survey (1,200 teens)
    teen_raw_path: Path = Path("data/raw/Teen_Mental_Health_Dataset.csv")
    teen_processed_path: Path = Path("data/processed/teen_mental_health_engineered.parquet")

    artifacts_dir: Path = Path("artifacts")
    images_dir: Path = Path("images")
    notebooks_dir: Path = Path("notebooks")

    expected_row_count: int = 4500
    expected_column_count: int = 16

    raw_numerical_columns: Tuple[str, ...] = (
        "Age",
        "Daily_Usage_Hours",
        "Weekend_Extra_Hours",
        "Sleep_Duration_Hours",
        "Sleep_Quality_Score",
        "Perceived_Stress_Score",
        "Mental_Health_Index",
        "Academic_Performance_GPA",
    )

    categorical_columns: Tuple[str, ...] = (
        "Gender",
        "Academic_Level",
        "Primary_Platform",
        "Device_Type",
        "Late_Night_Usage",
        "Social_Comparison_Frequency",
    )

    impact_classes: Tuple[str, ...] = (
        "Beneficial",
        "Neutral",
        "Negative",
    )


@dataclass(frozen=True)
class ModelConfig:
    """Cross-validation and model hyperparameter settings."""

    random_seed: int = 42
    n_splits: int = 5

    # LightGBM Classifier Hyperparameters (Cohort 1)
    lgb_n_estimators: int = 350
    lgb_learning_rate: float = 0.04
    lgb_max_depth: int = 5
    lgb_num_leaves: int = 24
    lgb_subsample: float = 0.85
    lgb_colsample_bytree: float = 0.85

    # CatBoost Classifier Hyperparameters (Cohort 1)
    cb_iterations: int = 400
    cb_learning_rate: float = 0.04
    cb_depth: int = 5
    cb_l2_leaf_reg: float = 4.0

    # Continuous Distress Regressor Settings (Cohort 1)
    reg_n_estimators: int = 250
    reg_learning_rate: float = 0.05
    reg_max_depth: int = 4

    # Teen Depression Binary Classifier Hyperparameters (Cohort 3)
    teen_scale_pos_weight: float = 37.0  # 97.4% neg / 2.6% pos class imbalance
    teen_n_estimators: int = 300
    teen_learning_rate: float = 0.03
    teen_max_depth: int = 4
    teen_num_leaves: int = 15


@dataclass(frozen=True)
class ProjectConfig:
    """Root configuration aggregator for ISML."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
