"""Single-student behavioral triage CLI for ISML."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from src.core.config import ProjectConfig
from src.data_pipeline.loader import load_raw_survey
from src.feature_engineering.behavioral_features import compute_behavioral_features
from src.feature_engineering.encoder import build_feature_matrix


def triage_student(
    student_id: str,
    df: pd.DataFrame,
) -> Optional[Dict[str, object]]:
    """Retrieve student telemetry and calculate risk scores.

    Parameters
    ----------
    student_id : str
        Target student identifier (e.g. STU_2024000).
    df : pd.DataFrame
        Augmented survey dataset.

    Returns
    -------
    Optional[Dict[str, object]]
        Extracted student telemetry dictionary or None if not found.
    """
    matches = df[df["Student_ID"] == student_id]
    if matches.empty:
        return None

    row = matches.iloc[0]

    return {
        "student_id": row["Student_ID"],
        "age": int(row["Age"]),
        "gender": str(row["Gender"]),
        "academic_level": str(row["Academic_Level"]),
        "platform": str(row["Primary_Platform"]),
        "daily_hours": float(row["Daily_Usage_Hours"]),
        "weekend_extra": float(row["Weekend_Extra_Hours"]),
        "sleep_hours": float(row["Sleep_Duration_Hours"]),
        "sleep_quality": int(row["Sleep_Quality_Score"]),
        "late_night": bool(row["Late_Night_Usage"]),
        "social_comparison": str(row["Social_Comparison_Frequency"]),
        "stress_score": float(row["Perceived_Stress_Score"]),
        "mental_health": int(row["Mental_Health_Index"]),
        "gpa": float(row["Academic_Performance_GPA"]),
        "overall_impact": str(row["Overall_Impact"]),
        "screen_to_sleep_ratio": float(row["screen_to_sleep_ratio"]),
        "weekly_screen_load": float(row["weekly_screen_load_hours"]),
        "sleep_debt": float(row["sleep_debt_hours"]),
    }


def format_triage_report(info: Dict[str, object]) -> str:
    """Format single student telemetry into structured console report."""
    lines = [
        "=" * 70,
        f"         ISML — Student Behavioral Telemetry Report: {info['student_id']}",
        "=" * 70,
        f"Demographics: {info['age']}yo | {info['gender']} | {info['academic_level']}",
        f"Primary Digital Platform: {info['platform']}",
        "",
        "Digital Consumption & Chronobiology:",
        f"  - Daily Screen Time:       {info['daily_hours']:.1f} hrs/day",
        f"  - Weekend Extra Time:      {info['weekend_extra']:.1f} hrs/day",
        f"  - Weekly Screen Load:      {info['weekly_screen_load']:.1f} hrs/week",
        f"  - Nightly Sleep Duration:  {info['sleep_hours']:.1f} hrs (Debt: {info['sleep_debt']:.1f} hrs)",
        f"  - Sleep Quality Score:     {info['sleep_quality']} / 5",
        f"  - Screen-to-Sleep Ratio:   {info['screen_to_sleep_ratio']:.2f}",
        f"  - Frequent Late-Night:     {info['late_night']}",
        "",
        "Psychometric & Academic Profiles:",
        f"  - Social Comparison:       {info['social_comparison']}",
        f"  - Perceived Stress (PSS):  {info['stress_score']:.1f} / 40.0",
        f"  - Mental Health Index:     {info['mental_health']} / 100",
        f"  - Cumulative GPA:          {info['gpa']:.2f} / 4.00",
        f"  - Clinical Impact Regime:  {info['overall_impact']}",
        "=" * 70,
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="ISML Single Student Triage")
    parser.add_argument("--student-id", type=str, default="STU_2024000", help="Student ID to query")
    args = parser.parse_args()

    cfg = ProjectConfig()
    df = load_raw_survey(cfg.data.raw_data_path)
    aug = compute_behavioral_features(df)

    res = triage_student(args.student_id, aug)
    if not res:
        print(f"Error: Student ID '{args.student_id}' not found.")
        sys.exit(1)

    print(format_triage_report(res))


if __name__ == "__main__":
    main()

