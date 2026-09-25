"""
ML Challenge 2026 — Business Entity Resolution
Experiment Tracking, Structured Logging & Error Analysis Engine
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd
import numpy as np


class ExperimentTracker:
    """
    Manages structured file and console logging, metric tracking,
    feature importance export, threshold tuning curves, and error analysis dumps.
    """
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = experiment_name or f"run_{timestamp}"
        self.exp_dir = self.log_dir / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.exp_dir / "pipeline.log"
        self._setup_logger()

        self.metadata: Dict[str, Any] = {
            "experiment_name": self.exp_name,
            "start_time": datetime.now().isoformat(),
            "config": {},
            "blocking_metrics": {},
            "training_metrics": {},
            "threshold_metrics": {},
            "timing_seconds": {},
        }
        self.timers: Dict[str, float] = {}

    def _setup_logger(self):
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        console_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(console_fmt)
        self.logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        file_fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(file_fmt)
        self.logger.addHandler(fh)

    def log(self, message: str, level: str = "info"):
        if level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.info(message)

    def start_timer(self, stage: str):
        self.timers[stage] = time.time()
        self.log(f"Stage [{stage}] started.")

    def stop_timer(self, stage: str) -> float:
        if stage in self.timers:
            elapsed = time.time() - self.timers[stage]
            self.metadata["timing_seconds"][stage] = round(elapsed, 2)
            self.log(f"Stage [{stage}] completed in {elapsed:.2f}s ({elapsed/60:.2f}m).")
            return elapsed
        return 0.0

    def record_config(self, config_dict: Dict[str, Any]):
        self.metadata["config"].update(config_dict)

    def record_blocking_metrics(
        self,
        total_true_matches: int,
        captured_matches: int,
        pair_completeness: float,
        total_candidates: int,
        candidates_per_entity: float,
        reduction_ratio: float,
        by_country: Optional[Dict[str, Any]] = None
    ):
        self.metadata["blocking_metrics"] = {
            "total_ground_truth_matches": total_true_matches,
            "captured_matches": captured_matches,
            "pair_completeness_pct": round(pair_completeness, 3),
            "total_candidate_pairs": total_candidates,
            "avg_candidates_per_entity": round(candidates_per_entity, 2),
            "reduction_ratio_pct": round(reduction_ratio, 6),
            "by_country": by_country or {}
        }
        self.log(f"Overall Blocking Metrics: PC={pair_completeness:.2f}%, RR={reduction_ratio:.4f}%, Avg Candidates={candidates_per_entity:.1f}")

    def save_feature_importance(self, feature_names: List[str], importances: np.ndarray, top_n: int = 30):
        """Saves feature importance to CSV and logs top features."""
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
        csv_path = self.exp_dir / "feature_importance.csv"
        df.to_csv(csv_path, index=False)
        self.log(f"Saved feature importance table to {csv_path}")

        top_str = "\n".join([f"    {r.feature:25s}: {r.importance:.4f}" for _, r in df.head(top_n).iterrows()])
        self.log(f"Top {top_n} Feature Importances:\n{top_str}")

    def save_tuning_curve(self, tuning_records: List[Dict[str, Any]]):
        """Saves threshold tuning grid search results to CSV."""
        df = pd.DataFrame(tuning_records)
        csv_path = self.exp_dir / "threshold_tuning_curve.csv"
        df.to_csv(csv_path, index=False)
        self.log(f"Saved threshold tuning curve to {csv_path}")

    def dump_error_analysis(
        self,
        candidate_pairs: List[Tuple[str, str]],
        labels: np.ndarray,
        pred_probs: np.ndarray,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        tau: float = 0.82,
        top_k_errors: int = 50
    ):
        """
        Dumps diagnostic tables of False Positives and False Negatives
        with their raw names, addresses, similarity scores, and probabilities.
        """
        s1_lookup = s1_df.set_index("entity_id").to_dict("index")
        target_lookup = target_df.set_index("entity_id").to_dict("index")

        fp_records = []
        fn_records = []

        for (s1_id, t_id), y_true, p_pred in zip(candidate_pairs, labels, pred_probs):
            y_pred = 1 if p_pred >= tau else 0
            if s1_id not in s1_lookup or t_id not in target_lookup:
                continue

            s1 = s1_lookup[s1_id]
            tg = target_lookup[t_id]

            rec = {
                "s1_id": s1_id,
                "target_id": t_id,
                "country": s1.get("country", ""),
                "prob": round(float(p_pred), 4),
                "s1_name": s1.get("business_name", ""),
                "target_name": tg.get("business_name", ""),
                "s1_address": s1.get("business_address", ""),
                "target_address": tg.get("business_address", "")
            }

            if y_true == 0 and y_pred == 1:
                fp_records.append(rec)
            elif y_true == 1 and y_pred == 0:
                fn_records.append(rec)

        # Sort FPs by highest probability (most confident false merges)
        fp_df = pd.DataFrame(fp_records).sort_values(by="prob", ascending=False).head(top_k_errors)
        fn_df = pd.DataFrame(fn_records).sort_values(by="prob", ascending=True).head(top_k_errors)

        fp_path = self.exp_dir / "error_analysis_false_positives.csv"
        fn_path = self.exp_dir / "error_analysis_false_negatives.csv"

        fp_df.to_csv(fp_path, index=False)
        fn_df.to_csv(fn_path, index=False)
        self.log(f"Error Analysis: Saved top {len(fp_df)} FPs to {fp_path} and top {len(fn_df)} FNs to {fn_path}")

    def finalize(self, macro_f05: float, best_tau: float, best_tau_single: float):
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["threshold_metrics"] = {
            "validation_macro_f05": round(macro_f05, 4),
            "optimal_tau": round(best_tau, 4),
            "optimal_singleton_tau": round(best_tau_single, 4)
        }
        json_path = self.exp_dir / "run_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        self.log(f"Experiment summary written to {json_path}")
        self.log(f"All diagnostic logs and artifacts available in: {self.exp_dir}")
