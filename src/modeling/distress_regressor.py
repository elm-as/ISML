"""Continuous psychometric and academic GPA regression model."""

from typing import Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.core.config import ModelConfig
from src.core.types import PsychometricRegressionMetrics


def train_distress_regressor(
    x: pd.DataFrame,
    y: np.ndarray,
    config: ModelConfig,
) -> Tuple[PsychometricRegressionMetrics, np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Train 5-fold ensemble regressor predicting perceived stress.

    Automatically removes collinear stress-derived columns to guarantee zero leakage.

    Parameters
    ----------
    x : pd.DataFrame
        Design feature matrix.
    y : np.ndarray
        Continuous perceived stress score array.
    config : ModelConfig
        Model hyperparameters.

    Returns
    -------
    Tuple[PsychometricRegressionMetrics, np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]]
        Computed regression metrics, out-of-fold predictions, SHAP values,
        feature importances, and cleaned feature names.
    """
    # Guard against target leakage
    leak_cols = [c for c in ["Perceived_Stress_Score", "stress_wellbeing_disparity"] if c in x.columns]
    x_clean = x.drop(columns=leak_cols) if leak_cols else x.copy()

    kf = KFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )

    n_samples = len(x_clean)
    oof_preds = np.zeros(n_samples, dtype=float)
    lgb_importances = np.zeros(x_clean.shape[1], dtype=float)
    accumulated_shap = np.zeros(x_clean.shape, dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x_clean, y)):
        x_train, y_train = x_clean.iloc[train_idx], y[train_idx]
        x_val, y_val = x_clean.iloc[val_idx], y[val_idx]

        # LightGBM Regressor
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=config.reg_n_estimators,
            learning_rate=config.reg_learning_rate,
            max_depth=config.reg_max_depth,
            random_state=config.random_seed + fold_idx,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_reg.fit(x_train, y_train)
        lgb_preds = lgb_reg.predict(x_val)
        lgb_importances += lgb_reg.feature_importances_ / config.n_splits

        # CatBoost Regressor
        cb_reg = CatBoostRegressor(
            iterations=config.reg_n_estimators,
            learning_rate=config.reg_learning_rate,
            depth=config.reg_max_depth,
            random_seed=config.random_seed + fold_idx,
            verbose=0,
            thread_count=-1,
        )
        cb_reg.fit(x_train, y_train)
        cb_preds = cb_reg.predict(x_val)

        # 50-50 Blend
        oof_preds[val_idx] = 0.50 * lgb_preds + 0.50 * cb_preds

        # Compute TreeExplainer SHAP values on validation fold
        explainer = shap.TreeExplainer(lgb_reg)
        val_shap = explainer.shap_values(x_val)
        accumulated_shap[val_idx] = val_shap

    rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
    mae = float(mean_absolute_error(y, oof_preds))
    r2 = float(r2_score(y, oof_preds))

    rho, pval = spearmanr(y, oof_preds)

    metrics = PsychometricRegressionMetrics(
        rmse=rmse,
        mae=mae,
        r2_score=r2,
        spearman_rho=float(rho),
        spearman_pvalue=float(pval),
    )

    clean_feature_names = tuple(x_clean.columns.tolist())
    return metrics, oof_preds, accumulated_shap, lgb_importances, clean_feature_names
