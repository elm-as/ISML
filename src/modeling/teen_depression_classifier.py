"""Adolescent clinical depression classification pipeline (Cohort 3)."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.core.config import ModelConfig
from src.core.types import TeenDepressionMetrics


def prepare_teen_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Transform engineered teen survey into design matrix and binary target.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered adolescent dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, List[str]]
        Design feature matrix X, binary target vector y, and list of feature names.
    """
    target = df["depression_label"].values.astype(int)

    drop_cols = ["depression_label", "gender", "platform_usage", "social_interaction_level"]
    numerical_features = df.drop(columns=[c for c in drop_cols if c in df.columns])

    categorical_cols = [c for c in ["gender", "platform_usage", "social_interaction_level"] if c in df.columns]
    encoded_cats = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=float)

    feature_matrix = pd.concat([numerical_features, encoded_cats], axis=1)
    feature_names = feature_matrix.columns.tolist()

    return feature_matrix, target, feature_names


def optimize_youden_threshold(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Identify probability cutoff maximizing Youden's J statistic (Sensitivity + Specificity - 1).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary outcomes.
    oof_probs : np.ndarray
        Out-of-fold predicted probabilities.

    Returns
    -------
    Tuple[float, float, float, float, float]
        Optimal threshold, Youden's J, sensitivity, specificity, and F1 score.
    """
    threshold_candidates = np.linspace(0.10, 0.90, 81)
    best_j = -1.0
    best_threshold = 0.50

    for threshold in threshold_candidates:
        binary_preds = (oof_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, binary_preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j_stat = sens + spec - 1.0

        if j_stat > best_j:
            best_j = j_stat
            best_threshold = float(threshold)

    optimal_preds = (oof_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, optimal_preds).ravel()
    final_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    final_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    final_f1 = float(f1_score(y_true, optimal_preds, zero_division=0))

    return best_threshold, float(best_j), final_sens, final_spec, final_f1


def train_teen_depression_classifier(
    features: pd.DataFrame,
    target: np.ndarray,
    config: ModelConfig,
) -> Tuple[TeenDepressionMetrics, np.ndarray, np.ndarray]:
    """Train 5-fold Stratified LightGBM with cost-sensitive weighting for teen depression.

    Parameters
    ----------
    features : pd.DataFrame
        Design feature matrix.
    target : np.ndarray
        Binary depression status vector.
    config : ModelConfig
        Model hyperparameter and cross-validation configuration.

    Returns
    -------
    Tuple[TeenDepressionMetrics, np.ndarray, np.ndarray]
        Computed validation metrics, out-of-fold probability vector, and feature importances.
    """
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    oof_probabilities = np.zeros(len(target), dtype=float)
    fold_importances = np.zeros(features.shape[1], dtype=float)

    for train_idx, val_idx in skf.split(features, target):
        x_train, y_train = features.iloc[train_idx], target[train_idx]
        x_val, y_val = features.iloc[val_idx], target[val_idx]

        model = LGBMClassifier(
            scale_pos_weight=config.teen_scale_pos_weight,
            n_estimators=config.teen_n_estimators,
            learning_rate=config.teen_learning_rate,
            max_depth=config.teen_max_depth,
            num_leaves=config.teen_num_leaves,
            random_state=config.random_seed,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(x_train, y_train)

        oof_probabilities[val_idx] = model.predict_proba(x_val)[:, 1]
        fold_importances += model.feature_importances_ / config.n_splits

    roc_auc = float(roc_auc_score(target, oof_probabilities))
    pr_auc = float(average_precision_score(target, oof_probabilities))
    brier = float(brier_score_loss(target, oof_probabilities))

    opt_th, youden, sens, spec, f1 = optimize_youden_threshold(target, oof_probabilities)

    metrics = TeenDepressionMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        sensitivity=sens,
        specificity=spec,
        youden_j=youden,
        optimal_threshold=opt_th,
        f1_score=f1,
    )

    return metrics, oof_probabilities, fold_importances
