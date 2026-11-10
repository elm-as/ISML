"""Main orchestrator for ISML Tri-Cohort end-to-end data science pipeline."""

from pathlib import Path
import pandas as pd

from src.core.config import ProjectConfig
from src.core.types import PipelineRunResult
from src.data_pipeline.loader import (
    load_cohort2_global,
    load_cohort3_teen,
    load_raw_survey,
    save_processed_survey,
)
from src.feature_engineering.behavioral_features import compute_behavioral_features
from src.feature_engineering.encoder import build_feature_matrix
from src.feature_engineering.global_features import compute_global_features
from src.feature_engineering.teen_features import compute_teen_features
from src.interpretability.feature_importance import build_importance_dataframe
from src.modeling.distress_regressor import train_distress_regressor
from src.modeling.global_analyzer import analyze_global_cohort
from src.modeling.impact_classifier import train_impact_classifier
from src.modeling.teen_depression_classifier import (
    prepare_teen_matrix,
    train_teen_depression_classifier,
)
from src.validation.cross_validator import (
    TriCohortReportData,
    export_metrics_json,
    format_evaluation_summary,
)
from src.visualization.cohort_figures import (
    plot_global_addiction_mental_health,
    plot_teen_depression_clinical_curves,
    plot_teen_protective_factors,
)
from src.visualization.dashboard_builder import (
    DashboardPayload,
    generate_interactive_dashboard,
)
from src.visualization.figures import (
    plot_behavioral_clustering_quadrant,
    plot_confusion_matrix_impact,
    plot_feature_importance_shap,
    plot_gpa_stress_dose_response,
    plot_screen_time_sleep_debt,
)


def run_pipeline(config: ProjectConfig) -> PipelineRunResult:
    """Execute end-to-end ISML tri-cohort pipeline from ingestion to artifact delivery."""
    # -------------------------------------------------------------
    # Cohort 1: Primary Behavioral & Stress Cohort (N = 4,500)
    # -------------------------------------------------------------
    print("1. Ingesting Cohort 1 survey telemetry (4,500 students)...")
    c1_raw = load_raw_survey(config.data.raw_data_path)
    c1_eng = compute_behavioral_features(c1_raw)
    save_processed_survey(c1_eng, config.data.processed_data_path)
    matrix = build_feature_matrix(c1_eng)

    print("2. Training Cohort 1 5-Fold Stratified Overall Impact Classifier...")
    clf_metrics, oof_probs, clf_importances = train_impact_classifier(
        matrix.features, matrix.target_impact, config.model
    )
    print(f"   Cohort 1 Accuracy: {clf_metrics.accuracy * 100:.2f}% | Macro F1: {clf_metrics.macro_f1:.4f}")

    print("3. Training Cohort 1 5-Fold Leak-Free Psychometric Distress Regressor...")
    reg_metrics, _, shap_vals, reg_importances, reg_feat_names = train_distress_regressor(
        matrix.features, matrix.target_stress, config.model
    )
    print(f"   Cohort 1 Stress R²: {reg_metrics.r2_score:.4f} | Spearman Rho: {reg_metrics.spearman_rho:.4f}")

    importance_df = build_importance_dataframe(
        matrix.feature_names,
        clf_importances,
        reg_feat_names,
        reg_importances,
        shap_vals,
    )
    imp_csv_path = config.data.artifacts_dir / "feature_importance_benchmark.csv"
    imp_csv_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(imp_csv_path, index=False)

    # -------------------------------------------------------------
    # Cohort 2: Global Cross-National Cohort (N = 832, 113 Nations)
    # -------------------------------------------------------------
    print("4. Ingesting Cohort 2 global multi-country telemetry (832 students, 113 countries)...")
    c2_raw = load_cohort2_global(config.data.global_raw_path)
    c2_eng = compute_global_features(c2_raw)
    save_processed_survey(c2_eng, config.data.global_processed_path)
    c2_analysis, country_summary = analyze_global_cohort(c2_eng)
    print(f"   Cohort 2 Nations: {c2_analysis.num_countries} | Addiction vs Mental Health r: {c2_analysis.addiction_mental_health_corr:.4f}")

    # -------------------------------------------------------------
    # Cohort 3: Adolescent Clinical Depression Cohort (N = 1,200)
    # -------------------------------------------------------------
    print("5. Ingesting Cohort 3 adolescent depression survey (1,200 teenagers)...")
    c3_raw = load_cohort3_teen(config.data.teen_raw_path)
    c3_eng = compute_teen_features(c3_raw)
    save_processed_survey(c3_eng, config.data.teen_processed_path)
    c3_features, c3_target, _ = prepare_teen_matrix(c3_eng)

    print("6. Training Cohort 3 5-Fold Cost-Sensitive Depression Classifier...")
    teen_metrics, teen_oof, _ = train_teen_depression_classifier(
        c3_features, c3_target, config.model
    )
    print(f"   Cohort 3 ROC-AUC: {teen_metrics.roc_auc:.4f} | Sensitivity: {teen_metrics.sensitivity * 100:.2f}% | Youden J: {teen_metrics.youden_j:.4f}")

    # -------------------------------------------------------------
    # Visualizations & Publication Artifacts (300 DPI)
    # -------------------------------------------------------------
    print("7. Generating publication figures (300 DPI) across all cohorts...")
    img_dir = config.data.images_dir
    img_dir.mkdir(parents=True, exist_ok=True)

    # Cohort 1 Figures
    plot_behavioral_clustering_quadrant(c1_eng, img_dir / "behavioral_clustering_quadrant.png")
    plot_confusion_matrix_impact(matrix.target_impact, oof_probs, config.data.impact_classes, img_dir / "confusion_matrix_impact.png")
    plot_screen_time_sleep_debt(c1_eng, img_dir / "screen_time_sleep_debt_tradeoff.png")
    plot_gpa_stress_dose_response(c1_eng, img_dir / "gpa_stress_dose_response.png")
    plot_feature_importance_shap(importance_df, img_dir / "feature_importance_shap.png")

    # Cohort 2 & 3 Figures
    plot_global_addiction_mental_health(c2_eng, img_dir / "global_cross_national_addiction.png")
    plot_teen_depression_clinical_curves(c3_target, teen_oof, teen_metrics.optimal_threshold, img_dir / "teen_depression_clinical_curves.png")
    plot_teen_protective_factors(c3_eng, img_dir / "teen_protective_factors.png")

    # -------------------------------------------------------------
    # Dashboard & Metrics Serialization
    # -------------------------------------------------------------
    print("8. Compiling standalone interactive HTML5 dashboard...")
    dash_path = config.data.artifacts_dir / "dashboard.html"
    dashboard_payload = DashboardPayload(
        cohort1_df=c1_eng,
        cohort2_df=c2_eng,
        cohort3_df=c3_eng,
        clf_metrics=clf_metrics.__dict__,
        reg_metrics=reg_metrics.__dict__,
        teen_metrics=teen_metrics.__dict__,
        global_analysis=c2_analysis.__dict__,
        importance_df=importance_df,
    )
    generate_interactive_dashboard(dashboard_payload, dash_path)

    print("9. Exporting metrics summary JSON and printing scientific report...")
    top_5_features = importance_df["feature"].head(5).tolist()
    report_data = TriCohortReportData(
        clf_metrics=clf_metrics,
        reg_metrics=reg_metrics,
        global_analysis=c2_analysis,
        teen_metrics=teen_metrics,
        top_features=top_5_features,
    )
    metrics_json_path = config.data.artifacts_dir / "metrics_summary.json"
    export_metrics_json(report_data, metrics_json_path)

    summary_text = format_evaluation_summary(report_data)
    print("\n" + summary_text)

    artifact_paths = (
        str(imp_csv_path),
        str(dash_path),
        str(metrics_json_path),
        str(img_dir / "behavioral_clustering_quadrant.png"),
        str(img_dir / "confusion_matrix_impact.png"),
        str(img_dir / "screen_time_sleep_debt_tradeoff.png"),
        str(img_dir / "gpa_stress_dose_response.png"),
        str(img_dir / "feature_importance_shap.png"),
        str(img_dir / "global_cross_national_addiction.png"),
        str(img_dir / "teen_depression_clinical_curves.png"),
        str(img_dir / "teen_protective_factors.png"),
    )

    return PipelineRunResult(
        cohort1_metrics=clf_metrics,
        cohort1_stress=reg_metrics,
        cohort2_analysis=c2_analysis,
        cohort3_metrics=teen_metrics,
        feature_importance_df=importance_df,
        artifact_paths=artifact_paths,
    )


def main() -> None:
    """CLI orchestrator entrypoint."""
    cfg = ProjectConfig()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
