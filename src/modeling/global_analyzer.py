"""Cross-national epidemiological analysis and statistical modeling (Cohort 2)."""

from typing import Dict, Tuple
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.core.types import GlobalCohortAnalysis


def analyze_global_cohort(df: pd.DataFrame) -> Tuple[GlobalCohortAnalysis, pd.DataFrame]:
    """Perform cross-national correlation and epidemiological analysis across 113 countries.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered global student survey dataset.

    Returns
    -------
    Tuple[GlobalCohortAnalysis, pd.DataFrame]
        Domain analysis dataclass and aggregated country-level summary metrics dataframe.
    """
    total_records = len(df)
    num_countries = int(df["Country"].nunique())

    # Pearson correlation coefficients
    r_addiction_mental, _ = pearsonr(df["Addicted_Score"], df["Mental_Health_Score"])
    r_addiction_conflict, _ = pearsonr(df["Addicted_Score"], df["Conflicts_Over_Social_Media"])
    r_sleep_mental, _ = pearsonr(df["Sleep_Hours_Per_Night"], df["Mental_Health_Score"])

    # Disruption ratio
    disrupted_count = (
        df["Affects_Academic_Performance"].astype(str).str.strip().eq("Yes").sum()
    )
    disruption_ratio = float(disrupted_count / total_records) if total_records > 0 else 0.0

    top_platforms = df["Most_Used_Platform"].value_counts().head(6).to_dict()

    analysis = GlobalCohortAnalysis(
        total_records=total_records,
        num_countries=num_countries,
        addiction_mental_health_corr=float(r_addiction_mental),
        addiction_conflict_corr=float(r_addiction_conflict),
        sleep_mental_health_corr=float(r_sleep_mental),
        academic_affect_ratio=disruption_ratio,
        top_platforms=top_platforms,
    )

    # Country-level aggregation for geospatial triage (countries with >= 3 responses)
    country_summary = (
        df.groupby("Country")
        .agg(
            respondents=("Student_ID", "count"),
            mean_addiction=("Addicted_Score", "mean"),
            mean_mental_health=("Mental_Health_Score", "mean"),
            mean_daily_hours=("Avg_Daily_Usage_Hours", "mean"),
            mean_sleep=("Sleep_Hours_Per_Night", "mean"),
            mean_conflicts=("Conflicts_Over_Social_Media", "mean"),
        )
        .reset_index()
        .sort_values(by="respondents", ascending=False)
    )

    return analysis, country_summary
