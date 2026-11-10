"""Unit tests for impact classifier and distress regressor."""

import numpy as np
import pandas as pd
from src.core.config import ModelConfig
from src.modeling.distress_regressor import train_distress_regressor
from src.modeling.impact_classifier import train_impact_classifier


def test_models_produce_bounded_valid_metrics() -> None:
    """Verify that models run on synthetic mini-cohort and produce valid bounded metrics."""
    np.random.seed(42)
    n = 60

    features = pd.DataFrame(
        {
            "Age": np.random.randint(18, 25, size=n).astype(float),
            "Daily_Usage_Hours": np.random.uniform(2, 12, size=n),
            "Sleep_Duration_Hours": np.random.uniform(4, 9, size=n),
            "Perceived_Stress_Score": np.random.uniform(10, 35, size=n),
            "Mental_Health_Index": np.random.uniform(40, 90, size=n),
        }
    )

    y_impact = np.array([0] * 40 + [1] * 12 + [2] * 8)
    y_stress = features["Perceived_Stress_Score"].values

    cfg = ModelConfig(
        n_splits=2,
        lgb_n_estimators=10,
        cb_iterations=10,
        reg_n_estimators=10,
    )

    # 1. Test Impact Classifier
    clf_metrics, oof_probs, clf_imp = train_impact_classifier(features, y_impact, cfg)
    assert 0.0 <= clf_metrics.accuracy <= 1.0
    assert 0.0 <= clf_metrics.macro_f1 <= 1.0
    assert oof_probs.shape == (n, 3)

    # 2. Test Distress Regressor
    reg_metrics, oof_preds, shap_vals, reg_imp, reg_names = train_distress_regressor(features, y_stress, cfg)
    assert reg_metrics.rmse >= 0.0
    assert len(oof_preds) == n
    assert len(reg_names) > 0
