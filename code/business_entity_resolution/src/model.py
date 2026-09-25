"""
ML Challenge 2026 — Business Entity Resolution
Asymmetric GBDT Matching Model & 5-Fold Cross Validation
"""

import os
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore', category=UserWarning)


class AsymmetricEntityMatcher:
    """
    LightGBM-based pairwise entity matching classifier with asymmetric precision weighting.
    """
    def __init__(
        self,
        neg_weight: float = 4.0,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        n_estimators: int = 300,
        random_state: int = 42,
        device: Optional[str] = None
    ):
        self.neg_weight = neg_weight
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.random_state = random_state

        if device:
            self.device = device
        else:
            self.device = "cuda" if os.environ.get("USE_GPU", "0") == "1" else "cpu"
        self.scale_pos_weight = 1.0 / float(neg_weight)
        self.model = None
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Trains LightGBM classifier with early stopping on validation logloss."""
        self.feature_names = feature_names
        device_param = "gpu" if self.device == "cuda" else "cpu"

        self.model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            device=device_param,
            n_jobs=-1,
            verbose=-1
        )

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

    def get_feature_importances(self) -> np.ndarray:
        """Returns feature importance scores from trained LightGBM model."""
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return np.zeros(len(self.feature_names or []), dtype=np.float32)


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns match probabilities P(Match = 1) for candidate pairs."""
        if self.model is None or len(X) == 0:
            return np.zeros(len(X), dtype=np.float32)
        return self.model.predict_proba(X)[:, 1]

    def save(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'feature_names': self.feature_names}, f)
        print(f"  [Model] Saved trained matcher checkpoint to: {model_path}")

    def load(self, model_path: str):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']


def train_5fold_cv(
    candidate_pairs: List[Tuple[str, str]],
    labels: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    s1_ids: List[str],
    n_splits: int = 5
) -> Tuple[AsymmetricEntityMatcher, np.ndarray]:
    """
    Runs 5-fold cross-validation partitioned strictly by Source 1 entity IDs (GroupKFold)
    to prevent cross-fold data leakage. Returns best model and OOF predictions.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = np.array(s1_ids)

    oof_preds = np.zeros(len(labels), dtype=np.float32)
    models = []

    print(f"  [Training] Starting {n_splits}-fold GroupKFold cross validation over {len(labels):,} candidate pairs...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, labels, groups=groups), start=1):
        X_tr, y_tr = X[train_idx], labels[train_idx]
        X_va, y_va = X[val_idx], labels[val_idx]

        matcher = AsymmetricEntityMatcher()
        matcher.fit(X_tr, y_tr, feature_names=feature_names, X_val=X_va, y_val=y_va)
        val_pred = matcher.predict_proba(X_va)
        oof_preds[val_idx] = val_pred
        models.append(matcher)

        pos_rate = float(y_va.mean())
        print(f"    Fold {fold}: Val pairs={len(val_idx):,}, Pos rate={pos_rate:.3f}, Mean Pred={val_pred.mean():.3f}")

    # Retrain final full model on entire dataset
    print(f"  [Training] Fitting full model across all {len(labels):,} pairs...")
    full_matcher = AsymmetricEntityMatcher()
    full_matcher.fit(X, labels, feature_names=feature_names)

    return full_matcher, oof_preds
