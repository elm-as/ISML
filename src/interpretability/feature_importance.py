"""Feature importance rankings and SHAP attribution for ISML."""

from typing import List, Tuple
import numpy as np
import pandas as pd


def build_importance_dataframe(
    clf_feature_names: Tuple[str, ...],
    clf_importances: np.ndarray,
    reg_feature_names: Tuple[str, ...],
    reg_importances: np.ndarray,
    shap_matrix: np.ndarray,
) -> pd.DataFrame:
    """Consolidate tree-based gain and SHAP values into a structured ranking.

    Handles potential dimension differences due to leak-prevention drops.

    Parameters
    ----------
    clf_feature_names : Tuple[str, ...]
        Names of the classifier features.
    clf_importances : np.ndarray
        Feature importances from the impact classifier.
    reg_feature_names : Tuple[str, ...]
        Names of the regressor features.
    reg_importances : np.ndarray
        Feature importances from the distress regressor.
    shap_matrix : np.ndarray
        Aggregated SHAP attribution values across regressor samples.

    Returns
    -------
    pd.DataFrame
        Ranked feature importance dataframe sorted by composite attribution.
    """
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

    # Normalize to percentages
    clf_norm = 100.0 * (clf_importances / (np.sum(clf_importances) + 1e-8))
    reg_norm = 100.0 * (reg_importances / (np.sum(reg_importances) + 1e-8))
    shap_norm = 100.0 * (mean_abs_shap / (np.sum(mean_abs_shap) + 1e-8))

    clf_map = dict(zip(clf_feature_names, clf_norm))
    reg_map = dict(zip(reg_feature_names, reg_norm))
    shap_map = dict(zip(reg_feature_names, shap_norm))

    records = []
    for feat in clf_feature_names:
        c_val = clf_map.get(feat, 0.0)
        r_val = reg_map.get(feat, 0.0)
        s_val = shap_map.get(feat, 0.0)
        comp = (c_val + r_val + s_val) / 3.0
        records.append(
            {
                "feature": feat,
                "classifier_gain_pct": c_val,
                "regressor_gain_pct": r_val,
                "mean_abs_shap_pct": s_val,
                "composite_importance": comp,
            }
        )

    df = pd.DataFrame(records).sort_values("composite_importance", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)
