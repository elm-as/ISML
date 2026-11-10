"""Main orchestrator for ISML end-to-end data science pipeline."""

from pathlib import Path
import pandas as pd

from src.core.config import ProjectConfig
from src.core.types import PipelineRunResult
from src.data_pipeline.loader import load_raw_survey, save_processed_survey
from src.feature_engineering.behavioral_features import compute_behavioral_features
from src.feature_engineering.encoder import build_feature_matrix
from src.interpretability.feature_importance import build_importance_dataframe
from src.modeling.distress_regressor import train_distress_regressor
from src.modeling.impact_classifier import train_impact_classifier
from src.validation.cross_validator import export_metrics_json, format_evaluation_summary
from src.visualization.dashboard_builder import generate_interactive_dashboard
from src.visualization.figures import (
    plot_behavioral_clustering_quadrant,
    plot_confusion_matrix_impact,
    plot_feature_importance_shap,
    plot_gpa_stress_dose_response,
    plot_screen_time_sleep_debt,
)


def run_pipeline(config: ProjectConfig) -> PipelineRunResult:
    """Execute end-to-end ISML pipeline from ingestion to artifact delivery."""
    print("1. Ingesting raw survey telemetry...")
    raw_df = load_raw_survey(config.data.raw_data_path)
    print(f"   Successfully loaded {len(raw_df):,} student records.")

    print("2. Deriving chronobiological and behavioral features...")
    engineered_df = compute_behavioral_features(raw_df)
    save_processed_survey(engineered_df, config.data.processed_data_path)
    print("   Engineered dataset exported to Parquet.")

    print("3. Encoding feature matrix...")
    matrix = build_feature_matrix(engineered_df)
    print(f"   Constructed feature matrix with shape {matrix.features.shape}.")

    print("4. Training 5-Fold Stratified Overall Impact Classifier (Task 1)...")
    clf_metrics, oof_probs, clf_importances = train_impact_classifier(
        matrix.features, matrix.target_impact, config.model
    )
    print(f"   Task 1 Accuracy: {clf_metrics.accuracy * 100:.2f}% | Macro F1: {clf_metrics.macro_f1:.4f}")

    print("5. Training 5-Fold Leak-Free Psychometric Distress Regressor (Task 2)...")
    reg_metrics, oof_preds, shap_vals, reg_importances, reg_feat_names = train_distress_regressor(
        matrix.features, matrix.target_stress, config.model
    )
    print(f"   Task 2 Stress R²: {reg_metrics.r2_score:.4f} | Spearman Rho: {reg_metrics.spearman_rho:.4f}")

    print("6. Synthesizing feature importances and SHAP values...")
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

    print("7. Generating publication figures (300 DPI)...")
    img_dir = config.data.images_dir
    plot_behavioral_clustering_quadrant(
        engineered_df, img_dir / "behavioral_clustering_quadrant.png"
    )
    plot_confusion_matrix_impact(
        matrix.target_impact,
        oof_probs,
        config.data.impact_classes,
        img_dir / "confusion_matrix_impact.png",
    )
    plot_screen_time_sleep_debt(
        engineered_df, img_dir / "screen_time_sleep_debt_tradeoff.png"
    )
    plot_gpa_stress_dose_response(
        engineered_df, img_dir / "gpa_stress_dose_response.png"
    )
    plot_feature_importance_shap(
        importance_df, img_dir / "feature_importance_shap.png"
    )

    print("8. Compiling standalone interactive HTML5 dashboard...")
    dash_path = config.data.artifacts_dir / "dashboard.html"
    generate_interactive_dashboard(
        engineered_df,
        clf_metrics.__dict__,
        reg_metrics.__dict__,
        importance_df,
        dash_path,
    )

    print("9. Exporting metrics summary JSON...")
    top_5_features = importance_df["feature"].head(5).tolist()
    metrics_json_path = config.data.artifacts_dir / "metrics_summary.json"
    export_metrics_json(clf_metrics, reg_metrics, top_5_features, metrics_json_path)

    summary_text = format_evaluation_summary(clf_metrics, reg_metrics, len(raw_df))
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
    )

    return PipelineRunResult(
        classification_metrics=clf_metrics,
        stress_metrics=reg_metrics,
        feature_importance_df=importance_df,
        artifact_paths=artifact_paths,
    )


def main() -> None:
    """CLI orchestrator entrypoint."""
    cfg = ProjectConfig()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
