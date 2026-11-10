"""Stratified ensemble multi-class classifier for student overall impact."""

from typing import Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold

from src.core.config import ModelConfig
from src.core.types import ImpactClassificationMetrics

CLASS_NAMES: Tuple[str, ...] = ("Beneficial", "Neutral", "Negative")


def train_impact_classifier(
    x: pd.DataFrame,
    y: np.ndarray,
    config: ModelConfig,
) -> Tuple[ImpactClassificationMetrics, np.ndarray, np.ndarray]:
    """Train 5-fold stratified soft-voting ensemble of LightGBM and CatBoost.

    Parameters
    ----------
    x : pd.DataFrame
        Design feature matrix.
    y : np.ndarray
        Encoded multi-class target labels (0, 1, 2).
    config : ModelConfig
        Model hyperparameters and cross-validation parameters.

    Returns
    -------
    Tuple[ImpactClassificationMetrics, np.ndarray, np.ndarray]
        Computed classification metrics, out-of-fold probabilities,
        and averaged feature importances.
    """
    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )

    n_samples = len(x)
    n_classes = len(CLASS_NAMES)
    oof_probs = np.zeros((n_samples, n_classes), dtype=float)
    lgb_importances = np.zeros(x.shape[1], dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        x_train, y_train = x.iloc[train_idx], y[train_idx]
        x_val, y_val = x.iloc[val_idx], y[val_idx]

        # LightGBM with balanced class weights
        lgb_model = lgb.LGBMClassifier(
            n_estimators=config.lgb_n_estimators,
            learning_rate=config.lgb_learning_rate,
            max_depth=config.lgb_max_depth,
            num_leaves=config.lgb_num_leaves,
            subsample=config.lgb_subsample,
            colsample_bytree=config.lgb_colsample_bytree,
            class_weight="balanced",
            random_state=config.random_seed + fold_idx,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_model.fit(x_train, y_train)
        lgb_probs = lgb_model.predict_proba(x_val)
        lgb_importances += (
            lgb_model.feature_importances_ / config.n_splits
        )

        # CatBoost with balanced class weights
        cb_model = CatBoostClassifier(
            iterations=config.cb_iterations,
            learning_rate=config.cb_learning_rate,
            depth=config.cb_depth,
            l2_leaf_reg=config.cb_l2_leaf_reg,
            auto_class_weights="Balanced",
            random_seed=config.random_seed + fold_idx,
            verbose=0,
            thread_count=-1,
        )
        cb_model.fit(x_train, y_train)
        cb_probs = cb_model.predict_proba(x_val)

        # Soft-voting blend: 50% LightGBM + 50% CatBoost
        oof_probs[val_idx] = 0.50 * lgb_probs + 0.50 * cb_probs

    y_pred = np.argmax(oof_probs, axis=1)

    acc = float(accuracy_score(y, y_pred))
    balanced_acc = float(balanced_accuracy_score(y, y_pred))
    macro_f1 = float(f1_score(y, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y, y_pred, average="weighted"))
    loss = float(log_loss(y, oof_probs))

    per_class_raw = f1_score(y, y_pred, average=None)
    per_class_dict: Dict[str, float] = {
        CLASS_NAMES[i]: float(per_class_raw[i]) for i in range(n_classes)
    }

    metrics = ImpactClassificationMetrics(
        accuracy=acc,
        balanced_accuracy=balanced_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        multi_class_log_loss=loss,
        per_class_f1=per_class_dict,
    )

    return metrics, oof_probs, lgb_importances
