"""
ML Challenge 2026 — Business Entity Resolution
Production End-to-End Orchestrator Pipeline (Approach 4)
Supports Cloud GPU execution, structured logging, feature importance export,
error analysis diagnostic dumps, and official submission verification.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd

from config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH,
    OUTPUT_MATCHING_PATH, OUTPUT_CANDIDATE_PATH,
    VALIDATE_SCRIPT, TEST_DIR, ARTIFACT_DIR,
    DEFAULT_MATCH_THRESHOLD, SINGLETON_THRESHOLD,
    MAX_MATCHES_PER_ENTITY, MAX_CANDIDATES_PER_ENTITY,
    SBERT_MODEL_NAME
)
from preprocessing import preprocess_dataframe
from blocking import run_hybrid_blocking, write_candidate_pairs_tsv
from features import build_feature_matrix
from model import AsymmetricEntityMatcher, train_5fold_cv
from postprocessing import (
    evaluate_macro_f05, tune_decision_thresholds,
    apply_decision_rules, write_matching_results_tsv
)
from logger import ExperimentTracker


def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Loads train_ground_truth.tsv into {s1_id: set(matched_ids)} via ultra-fast zip comprehension."""
    gt_df = pd.read_csv(gt_path, sep="\t").fillna("")
    return {
        s1: set(m.split(',')) if m else set()
        for s1, m in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids'])
    }


def run_pipeline(
    train_sample_size: Optional[int] = None,
    test_mode: bool = False,
    use_dense: bool = False,
    dense_model_name: str = SBERT_MODEL_NAME,
    device: str = "cpu",
    log_dir: str = "logs",
    experiment_name: Optional[str] = None
):
    tracker = ExperimentTracker(log_dir=log_dir, experiment_name=experiment_name)
    tracker.log("=" * 80)
    tracker.log("2026 AMAZON BUSINESS ENTITY RESOLUTION — HYBRID PIPELINE (APPROACH 4)")
    tracker.log("=" * 80)

    # Set GPU environment flag
    if device == "cuda":
        os.environ["USE_GPU"] = "1"
    else:
        os.environ["USE_GPU"] = "0"

    tracker.record_config({
        "train_sample_size": train_sample_size,
        "test_mode": test_mode,
        "use_dense_blocking": use_dense,
        "dense_model_name": dense_model_name if use_dense else None,
        "device": device,
        "max_matches_per_entity": MAX_MATCHES_PER_ENTITY,
        "max_candidates_per_entity": MAX_CANDIDATES_PER_ENTITY,
    })

    model_checkpoint = str(ARTIFACT_DIR / "matcher_model.pkl")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: DATA LOADING & PREPROCESSING
    # ──────────────────────────────────────────────────────────────────────────
    tracker.start_timer("data_loading_and_preprocessing")
    sample_desc = f"{train_sample_size:,}" if train_sample_size else "FULL DATASET (Cloud GPU)"
    tracker.log(f"Phase 1: Loading Training Reference & Targets (Sample: {sample_desc})...")

    s1_full = pd.read_csv(TRAIN_S1_PATH, sep="\t")
    if train_sample_size and train_sample_size < len(s1_full):
        s1_train = s1_full.sample(n=train_sample_size, random_state=42).reset_index(drop=True)
    else:
        s1_train = s1_full

    s1_train_ids = set(s1_train['entity_id'])
    gt_all = load_ground_truth(str(TRAIN_GT_PATH))
    gt_train = {sid: gt_all[sid] for sid in s1_train_ids if sid in gt_all}

    # Collect target IDs matching the S1 sample
    target_ids_in_sample = set()
    for s_set in gt_train.values():
        target_ids_in_sample.update(s_set)

    # Efficient target loading when sampling vs full cloud run
    if train_sample_size and train_sample_size < len(s1_full):
        tracker.log("Streaming target records (all true positive targets + distractor background)...")
        s2_chunks = []
        for chunk in pd.read_csv(TRAIN_S2_PATH, sep="\t", chunksize=250000):
            hits = chunk[chunk['entity_id'].isin(target_ids_in_sample)]
            distractors = chunk[~chunk['entity_id'].isin(target_ids_in_sample)].sample(frac=0.015, random_state=42)
            s2_chunks.append(pd.concat([hits, distractors]))
            if sum(len(c) for c in s2_chunks) > 100000:
                break
        s2_sample = pd.concat(s2_chunks, ignore_index=True)

        s3_chunks = []
        for chunk in pd.read_csv(TRAIN_S3_PATH, sep="\t", chunksize=250000):
            hits = chunk[chunk['entity_id'].isin(target_ids_in_sample)]
            distractors = chunk[~chunk['entity_id'].isin(target_ids_in_sample)].sample(frac=0.015, random_state=42)
            s3_chunks.append(pd.concat([hits, distractors]))
            if sum(len(c) for c in s3_chunks) > 100000:
                break
        s3_sample = pd.concat(s3_chunks, ignore_index=True)
        s23_train = pd.concat([s2_sample, s3_sample], ignore_index=True).drop_duplicates(subset=['entity_id'])
    else:
        tracker.log("Loading complete target tables (Sources 2 and 3)...")
        s2_full = pd.read_csv(TRAIN_S2_PATH, sep="\t")
        s3_full = pd.read_csv(TRAIN_S3_PATH, sep="\t")
        s23_train = pd.concat([s2_full, s3_full], ignore_index=True).drop_duplicates(subset=['entity_id'])

    tracker.log(f"Loaded: {len(s1_train):,} Reference S1 | {len(s23_train):,} Target Records (S2+S3)")

    tracker.log("Executing high-speed vectorized text cleaning, transliteration, and postal normalization...")
    s1_train_proc = preprocess_dataframe(s1_train, is_reference=True)
    s23_train_proc = preprocess_dataframe(s23_train, is_reference=False)
    tracker.stop_timer("data_loading_and_preprocessing")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: COUNTRY-PARTITIONED DUAL BLOCKING
    # ──────────────────────────────────────────────────────────────────────────
    tracker.start_timer("blocking")
    tracker.log("Phase 2: Generating Candidate Pairs via Country-Partitioned Inverted Indexing...")
    all_candidates: Dict[str, Set[str]] = {}
    dense_model = dense_model_name if use_dense else None

    for country in sorted(s1_train_proc['country'].unique()):
        s1_c = s1_train_proc[s1_train_proc['country'] == country]
        s23_c = s23_train_proc[s23_train_proc['country'] == country]
        tracker.log(f"  Blocking country [{country}]: {len(s1_c):,} S1 against {len(s23_c):,} targets...")
        c_dict = run_hybrid_blocking(
            s1_c, s23_c,
            max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY,
            dense_model_name=dense_model
        )
        all_candidates.update(c_dict)

    # Evaluate Blocking Performance (Pair Completeness & Reduction Ratio)
    total_true_matches = sum(len(gt_train.get(sid, set())) for sid in s1_train_ids)
    captured_matches = 0
    total_candidate_pairs = 0

    for s1_id in s1_train_ids:
        c_set = all_candidates.get(s1_id, set())
        true_set = gt_train.get(s1_id, set())
        captured_matches += len(c_set & true_set)
        total_candidate_pairs += len(c_set)

    pair_completeness = (captured_matches / total_true_matches * 100.0) if total_true_matches > 0 else 0.0
    total_cartesian = float(len(s1_train)) * float(len(s23_train))
    reduction_ratio = (1.0 - (float(total_candidate_pairs) / total_cartesian)) * 100.0 if total_cartesian > 0 else 100.0
    candidates_per_entity = total_candidate_pairs / float(len(s1_train)) if len(s1_train) > 0 else 0.0

    tracker.record_blocking_metrics(
        total_true_matches=total_true_matches,
        captured_matches=captured_matches,
        pair_completeness=pair_completeness,
        total_candidates=total_candidate_pairs,
        candidates_per_entity=candidates_per_entity,
        reduction_ratio=reduction_ratio
    )
    tracker.stop_timer("blocking")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3: MULTI-SIGNAL FEATURE EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────
    tracker.start_timer("feature_extraction")
    tracker.log("Phase 3: Building Pairwise Feature Matrix across 28+ signals...")
    pair_list = []
    labels = []
    pair_s1_ids = []

    for s1_id, c_set in all_candidates.items():
        true_set = gt_train.get(s1_id, set())
        for tid in c_set:
            pair_list.append((s1_id, tid))
            labels.append(1 if tid in true_set else 0)
            pair_s1_ids.append(s1_id)

    labels = np.array(labels, dtype=np.int32)
    X, feat_names = build_feature_matrix(pair_list, s1_train_proc, s23_train_proc)
    tracker.log(f"Feature matrix built: {X.shape[0]:,} candidate rows x {X.shape[1]} features.")
    tracker.stop_timer("feature_extraction")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4: ASYMMETRIC GBDT TRAINING & 5-FOLD CV
    # ──────────────────────────────────────────────────────────────────────────
    tracker.start_timer("model_training_cv")
    tracker.log("Phase 4: Training Asymmetric LightGBM Matcher with 5-Fold GroupKFold...")
    matcher, oof_probs = train_5fold_cv(pair_list, labels, X, feat_names, pair_s1_ids, n_splits=5)
    matcher.save(model_checkpoint)

    # Save feature importances to CSV and report top features
    importances = matcher.get_feature_importances()
    tracker.save_feature_importance(feat_names, importances)
    tracker.stop_timer("model_training_cv")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 5: THRESHOLD CALIBRATION & SINGLETON BOUNCER
    # ──────────────────────────────────────────────────────────────────────────
    tracker.start_timer("threshold_tuning")
    tracker.log("Phase 5: Running Grid Line Search for Optimal (tau, tau_singleton) on Macro F0.5...")
    best_tau, best_tau_single, val_macro_f05, tuning_records = tune_decision_thresholds(
        pair_list, oof_probs, gt_train, list(s1_train_ids)
    )
    tracker.save_tuning_curve(tuning_records)
    tracker.log(f">>> OOF VALIDATION MACRO F0.5 = {val_macro_f05:.4f} <<<")
    tracker.stop_timer("threshold_tuning")

    # Error Analysis Diagnostic Dump
    tracker.log("Generating Error Analysis Diagnostics (False Positives and False Negatives)...")
    tracker.dump_error_analysis(
        pair_list, labels, oof_probs, s1_train, s23_train, tau=best_tau, top_k_errors=50
    )
    tracker.finalize(val_macro_f05, best_tau, best_tau_single)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 6: TEST SET INFERENCE & VALIDATION
    # ──────────────────────────────────────────────────────────────────────────
    if test_mode:
        tracker.start_timer("test_inference")
        tracker.log("\n" + "=" * 80)
        tracker.log("PHASE 6: FULL TEST SET INFERENCE & SUBMISSION GENERATION")
        tracker.log("=" * 80)

        tracker.log("Loading Test Files (including France zero-shot partition)...")
        test_s1 = pd.read_csv(TEST_S1_PATH, sep="\t")
        test_s2 = pd.read_csv(TEST_S2_PATH, sep="\t")
        test_s3 = pd.read_csv(TEST_S3_PATH, sep="\t")
        test_s23 = pd.concat([test_s2, test_s3], ignore_index=True).drop_duplicates(subset=['entity_id'])
        tracker.log(f"Test Reference S1: {len(test_s1):,} | Test Targets: {len(test_s23):,}")

        tracker.log("Preprocessing Test Records (standardizing French legal forms, street types, CEDEX)...")
        test_s1_proc = preprocess_dataframe(test_s1, is_reference=True)
        test_s23_proc = preprocess_dataframe(test_s23, is_reference=False)

        tracker.log("Executing Test Candidate Blocking across US, India, and France...")
        test_candidates: Dict[str, Set[str]] = {}
        for country in ["US", "India", "France"]:
            s1_c = test_s1_proc[test_s1_proc['country'] == country]
            s23_c = test_s23_proc[test_s23_proc['country'] == country]
            tracker.log(f"  Blocking Test Country [{country}]: {len(s1_c):,} S1 against {len(s23_c):,} targets...")
            c_dict = run_hybrid_blocking(
                s1_c, s23_c,
                max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY,
                dense_model_name=dense_model
            )
            test_candidates.update(c_dict)

        # Write candidate_pairs.tsv
        test_s1_ids = test_s1['entity_id'].tolist()
        write_candidate_pairs_tsv(test_candidates, test_s1_ids, str(OUTPUT_CANDIDATE_PATH))

        tracker.log("Extracting Feature Matrix for Test Candidate Pairs...")
        test_pairs = [(sid, tid) for sid, cset in test_candidates.items() for tid in cset]
        X_test, _ = build_feature_matrix(test_pairs, test_s1_proc, test_s23_proc)
        tracker.log(f"Test candidate matrix: {X_test.shape[0]:,} rows x {X_test.shape[1]} features.")

        tracker.log("Scoring Test Candidates via Asymmetric Matcher...")
        test_probs = matcher.predict_proba(X_test)

        tracker.log(f"Applying Thresholds (tau={best_tau:.3f}, singleton={best_tau_single:.3f}, cap={MAX_MATCHES_PER_ENTITY})...")
        final_test_matches = apply_decision_rules(
            test_pairs, test_probs, test_s1_ids,
            match_threshold=best_tau,
            singleton_threshold=best_tau_single,
            max_matches=MAX_MATCHES_PER_ENTITY
        )

        # Write matching_results.tsv
        write_matching_results_tsv(final_test_matches, test_s1_ids, str(OUTPUT_MATCHING_PATH))
        tracker.stop_timer("test_inference")

        # Official Validator
        tracker.log("\n" + "=" * 80)
        tracker.log("RUNNING OFFICIAL COMPETITION VALIDATION SCRIPT")
        tracker.log("=" * 80)
        if Path(VALIDATE_SCRIPT).exists():
            cmd = f"{sys.executable} {VALIDATE_SCRIPT} --matching {OUTPUT_MATCHING_PATH} --candidate {OUTPUT_CANDIDATE_PATH} --test-dir {TEST_DIR}"
            tracker.log(f"Command: {cmd}")
            ret = os.system(cmd)
            if ret == 0:
                tracker.log(">> SUCCESS: Submission passes all format checks and constraints! <<")
            else:
                tracker.log(f">> WARNING: Validator exited with code {ret}. Review output above. <<", level="warning")
        else:
            tracker.log(f">> Note: Validator script not found at {VALIDATE_SCRIPT}. Skipping automated check. <<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 Amazon Entity Resolution — Approach 4 Pipeline")
    parser.add_argument("--sample-size", type=int, default=None, help="Train sample size (default: None for full dataset on Cloud GPU)")
    parser.add_argument("--test", action="store_true", help="Run full test set inference and generate submission TSVs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Compute device for models (cpu or cuda)")
    parser.add_argument("--use-dense", action="store_true", help="Enable dense multilingual vector blocking (recommended on Cloud GPU)")
    parser.add_argument("--dense-model", type=str, default=SBERT_MODEL_NAME, help="HuggingFace model for dense embeddings")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save execution logs and experiment metrics")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name identifier")

    args = parser.parse_args()

    run_pipeline(
        train_sample_size=args.sample_size,
        test_mode=args.test,
        use_dense=args.use_dense,
        dense_model_name=args.dense_model,
        device=args.device,
        log_dir=args.log_dir,
        experiment_name=args.exp_name
    )
