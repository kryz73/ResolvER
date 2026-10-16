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
from logger import make_progress_bar

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
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None

        try:
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
            self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        except Exception:
            # Fall back to CPU if LightGBM GPU build is not present in environment
            self.model = lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                n_estimators=self.n_estimators,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                device="cpu",
                n_jobs=-1,
                verbose=-1
            )
            self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)

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


class SelectiveCrossEncoderReranker:
    """
    Multilingual Cross-Encoder (BAAI/bge-reranker-v2-m3) that scores candidate pairs
    selectively on the ambiguous tail (e.g. 0.46 <= prob <= 0.56 or top-K).
    Accelerated with CUDA and FP16 via torch.amp.autocast when GPU is present.
    Loads on cuda:1 by default to avoid VRAM conflict with dense blocker on cuda:0.
    Uses FP32 weights with runtime autocast to prevent LayerNormKernelImpl crash on XLM-RoBERTa.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", top_k: int = 1, batch_size: int = 32):
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self.model = None
        try:
            import torch
            if torch.cuda.is_available() and os.environ.get("USE_GPU", "1") == "1":
                # Prefer cuda:1 to avoid VRAM conflict with dense blocker on cuda:0
                n_gpus = torch.cuda.device_count()
                self.device = f"cuda:{n_gpus - 1}" if n_gpus > 1 else "cuda:0"
            else:
                self.device = "cpu"
        except Exception:
            self.device = "cpu"
        self._init_model()

    def _init_model(self):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                # Initialize in standard FP32 - NEVER call model.half() directly
                # because XLM-RoBERTa LayerNorm crashes with "LayerNormKernelImpl not implemented for Half"
                self.model = CrossEncoder(self.model_name, max_length=256, device=self.device)
                print(f"  [Reranker] Successfully loaded {self.model_name} on {self.device}")
            except Exception as e:
                import traceback
                print(f"  [Reranker WARNING] CrossEncoder init FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                self.model = None

    def rerank_pairs(self, text_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Computes cross-encoder probability scores for a list of (text1, text2) pairs."""
        self._init_model()
        if self.model is None or not text_pairs:
            return np.zeros(len(text_pairs), dtype=np.float32)

        # Bug 4 Fix: CrossEncoder.predict() expects list of [str, str] lists, not tuples
        formatted_pairs = [
            [str(a).strip() or "unknown", str(b).strip() or "unknown"]
            for a, b in text_pairs
        ]

        try:
            import torch
            if "cuda" in self.device and torch.cuda.is_available():
                # Safe FP16 inference via autocast (hardware rule 3) - preserves FP32 LayerNorm
                with torch.amp.autocast("cuda"):
                    scores = self.model.predict(
                        formatted_pairs,
                        batch_size=self.batch_size,
                        show_progress_bar=False
                    )
            else:
                scores = self.model.predict(
                    formatted_pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False
                )
            # Normalize logits via sigmoid
            probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -15.0, 15.0)))
            return np.ascontiguousarray(probs, dtype=np.float32)
        except RuntimeError as e:
            # Handle CUDA OOM with fallback
            if "out of memory" in str(e).lower():
                import gc, torch
                torch.cuda.empty_cache()
                gc.collect()
                fallback_bs = max(4, self.batch_size // 4)
                print(f"  [Reranker] CUDA OOM encountered — retrying with batch_size={fallback_bs}...")
                try:
                    with torch.amp.autocast("cuda"):
                        scores = self.model.predict(
                            formatted_pairs,
                            batch_size=fallback_bs,
                            show_progress_bar=False
                        )
                    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -15.0, 15.0)))
                    return np.ascontiguousarray(probs, dtype=np.float32)
                except Exception as e2:
                    import traceback
                    print(f"  [Reranker ERROR] Fallback also failed: {type(e2).__name__}: {e2}\n{traceback.format_exc()}")
                    return np.zeros(len(text_pairs), dtype=np.float32)
            else:
                import traceback
                print(f"  [Reranker ERROR] Pair prediction failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                return np.zeros(len(text_pairs), dtype=np.float32)
        except Exception as e:
            import traceback
            print(f"  [Reranker ERROR] Pair prediction failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return np.zeros(len(text_pairs), dtype=np.float32)


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
    fold_iter = enumerate(gkf.split(X, labels, groups=groups), start=1)
    fold_iter = make_progress_bar(fold_iter, total=n_splits, desc="[Training] 5-Fold GroupKFold CV", mininterval=2.0, leave=False)

    for fold, (train_idx, val_idx) in fold_iter:
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
