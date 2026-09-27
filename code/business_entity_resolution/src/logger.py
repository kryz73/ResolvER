"""
ML Challenge 2026 — Business Entity Resolution
Experiment Tracking, Structured Logging, Timing Diagnostics & Error Analysis Engine
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


def make_progress_bar(
    iterable,
    desc: str = "",
    total: Optional[int] = None,
    leave: bool = False,
    mininterval: float = 2.0
):
    """
    Clean, flicker-free progress bar configured for Kaggle notebooks and non-interactive logs.
    Prevents newline spam by throttling update interval to >= 2.0s and suppressing trailing bars.
    """
    try:
        from tqdm.auto import tqdm
        return tqdm(
            iterable,
            desc=f"  {desc}",
            total=total,
            leave=leave,
            mininterval=mininterval,
            ncols=85,
            file=sys.stdout
        )
    except Exception:
        return iterable


class ExperimentTracker:
    """
    Manages structured file and console logging, fine-grained hierarchical process timing,
    feature importance export, threshold tuning curves, and error analysis dumps.
    Keeps terminal output clean, high-level, and concise while routing detailed debug telemetry to disk.
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
            "timing_breakdown": []
        }
        self.timers: Dict[str, float] = {}
        self.pipeline_start_time = time.time()

    def _setup_logger(self):
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # Console handler: INFO level only (clean, concise)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        console_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(console_fmt)
        self.logger.addHandler(ch)

        # File handler: DEBUG level (captures all telemetry and diagnostics)
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(file_fmt)
        self.logger.addHandler(fh)

    def log(self, message: str, level: str = "info", terminal: bool = True):
        """
        Logs message to file, and optionally to terminal.
        If terminal=False or level='debug', message is routed exclusively to pipeline.log.
        """
        if not terminal or level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.info(message)

    def start_timer(self, stage: str, is_substage: bool = False):
        self.timers[stage] = time.time()
        if not is_substage:
            self.log(f"Stage [{stage}] started.", terminal=True)
        else:
            self.log(f"  Sub-process [{stage}] started.", terminal=False)

    def stop_timer(self, stage: str, is_substage: bool = False, category: str = "phase") -> float:
        if stage in self.timers:
            elapsed = time.time() - self.timers.pop(stage)
            self.record_time(stage, elapsed, category=category)
            if not is_substage:
                self.log(f"Stage [{stage}] completed in {elapsed:.2f}s ({elapsed/60:.2f}m).", terminal=True)
            else:
                self.log(f"  Sub-process [{stage}] completed in {elapsed:.2f}s.", terminal=False)
            return elapsed
        return 0.0

    def record_time(self, name: str, elapsed: float, category: str = "phase"):
        """Records elapsed time for a phase or sub-process."""
        rounded = round(elapsed, 2)
        self.metadata["timing_seconds"][name] = round(self.metadata["timing_seconds"].get(name, 0.0) + rounded, 2)
        # Keep an ordered trace
        self.metadata["timing_breakdown"].append({
            "name": name,
            "seconds": rounded,
            "category": category
        })

    def add_time(self, name: str, elapsed: float, category: str = "sub"):
        """Accumulates runtime for recurrent batch sub-processes (e.g. GPU dense query, CPU lexical)."""
        rounded = round(elapsed, 2)
        self.metadata["timing_seconds"][name] = round(self.metadata["timing_seconds"].get(name, 0.0) + rounded, 2)

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
        self.log(
            f"Blocking Recall (PC): {pair_completeness:.2f}% | Reduction: {reduction_ratio:.4f}% | Avg Cands: {candidates_per_entity:.1f}",
            terminal=True
        )

    def save_feature_importance(self, feature_names: List[str], importances: np.ndarray, top_n: int = 10):
        """Saves feature importance table to CSV, writes full list to log file, and prints clean top 10 on console."""
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
        csv_path = self.exp_dir / "feature_importance.csv"
        df.to_csv(csv_path, index=False)
        self.log(f"Saved feature importance table ({len(df)} features) to {csv_path}", terminal=False)

        # Full feature dump strictly to log file
        full_str = "\n".join([f"    {r.feature:28s}: {r.importance:.4f}" for _, r in df.iterrows()])
        self.log(f"Complete Feature Importances:\n{full_str}", terminal=False)

        # Clean top 10 on terminal
        top_df = df.head(top_n)
        lines = [f"\n--- TOP {top_n} FEATURE IMPORTANCES ---"]
        for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
            lines.append(f"  {rank:2d}. {row.feature:26s} : {row.importance:8.1f}")
        self.log("\n".join(lines), terminal=True)

    def save_tuning_curve(self, tuning_records: List[Dict[str, Any]]):
        """Saves threshold tuning grid search results to CSV."""
        df = pd.DataFrame(tuning_records)
        csv_path = self.exp_dir / "threshold_tuning_curve.csv"
        df.to_csv(csv_path, index=False)
        self.log(f"Saved threshold tuning curve ({len(df)} points) to {csv_path}", terminal=False)

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
        """Dumps diagnostic tables of False Positives and False Negatives to disk without terminal flooding."""
        needed_target_ids = {t_id for _, t_id in candidate_pairs}
        target_subset = target_df[target_df["entity_id"].isin(needed_target_ids)]

        s1_lookup = s1_df.set_index("entity_id").to_dict("index")
        target_lookup = target_subset.set_index("entity_id").to_dict("index")

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
                "s1_name": s1.get("business_name") or s1.get("norm_name") or s1.get("clean_name", ""),
                "target_name": tg.get("business_name") or tg.get("norm_name") or tg.get("clean_name", ""),
                "s1_address": s1.get("business_address") or s1.get("norm_address", ""),
                "target_address": tg.get("business_address") or tg.get("norm_address", "")
            }

            if y_true == 0 and y_pred == 1:
                fp_records.append(rec)
            elif y_true == 1 and y_pred == 0:
                fn_records.append(rec)

        fp_df = pd.DataFrame(fp_records).sort_values(by="prob", ascending=False).head(top_k_errors)
        fn_df = pd.DataFrame(fn_records).sort_values(by="prob", ascending=True).head(top_k_errors)

        fp_path = self.exp_dir / "error_analysis_false_positives.csv"
        fn_path = self.exp_dir / "error_analysis_false_negatives.csv"

        fp_df.to_csv(fp_path, index=False)
        fn_df.to_csv(fn_path, index=False)
        self.log(f"Error Analysis Diagnostics: Saved top {len(fp_df)} FPs to {fp_path.name} and top {len(fn_df)} FNs to {fn_path.name}.", terminal=True)

    def generate_timing_table(self) -> Tuple[str, pd.DataFrame]:
        """
        Constructs a structured ASCII timing table breaking down every pipeline phase
        and recurring sub-process, including async GPU/CPU concurrency metrics.
        """
        total_time = max(0.1, time.time() - self.pipeline_start_time)
        t_dict = self.metadata["timing_seconds"]

        # Define canonical phases and sub-processes
        hierarchy = [
            ("Phase 1: Ingestion & Preprocessing", "data_loading_and_preprocessing", "phase"),
            ("  ├─ S1 Reference Loading & Preprocessing", "p1_s1_loading", "sub"),
            ("  └─ S2/S3 Target Streaming & Sampling", "p1_target_streaming", "sub"),
            ("Phase 2: Hybrid Candidate Blocking (Train)", "blocking", "phase"),
            ("  ├─ Lexical Target Index Fit", "p2_lex_fit", "sub"),
            ("  ├─ Dense Target Index Fit (FAISS)", "p2_dense_fit", "sub"),
            ("  └─ Pipelined Retrieval (GPU Dense || CPU Lex)", "p2_retrieval", "sub"),
            ("Phase 3: Multi-Signal Feature Extraction", "feature_extraction", "phase"),
            ("Phase 4: Asymmetric GBDT 5-Fold CV Training", "model_training_cv", "phase"),
            ("Phase 5: Decision Threshold Calibration & Grid Search", "threshold_tuning", "phase"),
            ("Phase 6: Sharded Test Inference & Packaging", "test_inference", "phase"),
            ("  ├─ S1 Global Preprocessing", "p6_s1_preprocess", "sub"),
            ("  ├─ Single-Pass S2/S3 Target Streaming", "p6_target_streaming", "sub"),
            ("  ├─ Target Preprocessing", "p6_target_preprocess", "sub"),
            ("  ├─ Target Indexing (Lexical + Dense FAISS)", "p6_blocker_indexing", "sub"),
            ("  ├─ Async Pipelined Retrieval (GPU Dense || CPU Lex)", "p6_async_retrieval", "sub"),
            ("  │   * GPU Dense Retrieval (Async)", "GPU Dense Retrieval (Async)", "detail"),
            ("  │   * CPU Lexical Multi-Core Retrieval (Async)", "CPU Lexical Multi-Core Retrieval (Async)", "detail"),
            ("  │   * Concurrency Pipelined Savings", "Async Pipelined Concurrency Savings", "savings"),
            ("  ├─ Chunked Feature Extraction & GBDT Scoring", "p6_feature_scoring", "sub"),
            ("  ├─ Selective Cross-Encoder Reranking", "p6_reranker", "sub"),
            ("  ├─ Candidate & Matching TSV Writing", "p6_tsv_writing", "sub"),
            ("  └─ Official Competition Validation Script", "p6_validator", "sub"),
        ]

        rows = []
        for display_name, key, row_type in hierarchy:
            sec = t_dict.get(key, 0.0)
            if sec > 0.0 or row_type == "phase":
                pct = (sec / total_time) * 100.0 if row_type != "savings" else 0.0
                rows.append({
                    "Stage / Process": display_name,
                    "Seconds": sec,
                    "Minutes": sec / 60.0,
                    "Share_Pct": pct,
                    "Type": row_type
                })

        # Add any unlisted custom timers
        accounted_keys = {k for _, k, _ in hierarchy}
        for k, sec in t_dict.items():
            if k not in accounted_keys and sec > 0.5:
                rows.append({
                    "Stage / Process": f"Other: {k}",
                    "Seconds": sec,
                    "Minutes": sec / 60.0,
                    "Share_Pct": (sec / total_time) * 100.0,
                    "Type": "other"
                })

        df = pd.DataFrame(rows)

        # Format ASCII table
        w_name = 62
        lines = [
            "\n" + "=" * 98,
            f"{'PIPELINE EXECUTION TIMING SUMMARY TABLE':^98}",
            "=" * 98,
            f"{'Phase / Sub-Process':<{w_name}} {'Duration (s)':>12} {'Duration (m)':>12} {'Share (%)':>10}",
            "-" * 98
        ]

        for _, r in df.iterrows():
            name = r["Stage / Process"]
            sec = r["Seconds"]
            minute = r["Minutes"]
            r_type = r["Type"]

            if r_type == "savings":
                sec_str = f"+{sec:6.2f}s"
                min_str = f"+{minute:6.2f}m"
                pct_str = "Saved"
            elif r_type == "detail":
                sec_str = f"[{sec:6.2f}s]"
                min_str = f"[{minute:6.2f}m]"
                pct_str = f"[{r['Share_Pct']:5.1f}%]"
            else:
                sec_str = f"{sec:7.2f}s"
                min_str = f"{minute:7.2f}m"
                pct_str = f"{r['Share_Pct']:6.1f}%"

            lines.append(f"{name:<{w_name}} {sec_str:>12} {min_str:>12} {pct_str:>10}")

        lines.append("-" * 98)
        tot_min = total_time / 60.0
        lines.append(f"{'TOTAL END-TO-END EXECUTION RUNTIME':<{w_name}} {total_time:>11.2f}s {tot_min:>11.2f}m {'100.0%':>10}")
        lines.append("=" * 98 + "\n")

        table_str = "\n".join(lines)
        return table_str, df

    def print_timing_summary(self):
        """Prints timing summary table to terminal, logs to disk, and saves timing_summary.csv."""
        table_str, df = self.generate_timing_table()
        csv_path = self.exp_dir / "timing_summary.csv"
        df.to_csv(csv_path, index=False)

        self.log(table_str, terminal=True)
        self.log(f"Saved complete timing breakdown to {csv_path}", terminal=False)

    def finalize(self, macro_f05: float, best_tau: float, best_tau_single: float):
        self.metadata["end_time"] = datetime.now().isoformat()
        total_runtime_s = round(time.time() - self.pipeline_start_time, 2)
        self.metadata["total_runtime_seconds"] = total_runtime_s
        self.metadata["threshold_metrics"] = {
            "validation_macro_f05": round(macro_f05, 4),
            "optimal_tau": round(best_tau, 4),
            "optimal_singleton_tau": round(best_tau_single, 4)
        }

        # Print and persist timing summary
        self.print_timing_summary()

        json_path = self.exp_dir / "run_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        self.log(f"Experiment summary written to {json_path}", terminal=True)
        self.log(f"All diagnostic logs and artifacts available in: {self.exp_dir}", terminal=True)
