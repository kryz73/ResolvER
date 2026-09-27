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
from collections import defaultdict
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
    SBERT_MODEL_NAME, RERANKER_MODEL_NAME,
    RERANK_TOP_K, RERANK_AMBIGUOUS_MIN, RERANK_AMBIGUOUS_MAX
)
from preprocessing import preprocess_dataframe
from blocking import (
    run_hybrid_blocking, write_candidate_pairs_tsv,
    fit_hybrid_blocker, query_hybrid_blocker
)
from features import build_feature_matrix
from model import AsymmetricEntityMatcher, train_5fold_cv, SelectiveCrossEncoderReranker
from postprocessing import (
    evaluate_macro_f05, tune_decision_thresholds,
    apply_decision_rules, write_matching_results_tsv
)
from logger import ExperimentTracker, make_progress_bar


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
    test_sample_size: Optional[int] = None,
    use_dense: bool = False,
    dense_model_name: str = SBERT_MODEL_NAME,
    use_reranker: bool = False,
    reranker_model_name: Optional[str] = None,
    test_only: bool = False,
    model_checkpoint_path: Optional[str] = None,
    optimal_tau: Optional[float] = None,
    optimal_singleton_tau: Optional[float] = None,
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

    dense_model = dense_model_name if use_dense else None

    tracker.record_config({
        "train_sample_size": train_sample_size,
        "test_mode": test_mode or test_only,
        "test_only": test_only,
        "use_dense_blocking": use_dense,
        "dense_model_name": dense_model_name if use_dense else None,
        "device": device,
        "max_matches_per_entity": MAX_MATCHES_PER_ENTITY,
        "max_candidates_per_entity": MAX_CANDIDATES_PER_ENTITY,
    })

    model_checkpoint = str(ARTIFACT_DIR / "matcher_model.pkl")
    val_macro_f05 = 0.0

    if test_only:
        test_mode = True
        tracker.start_timer("model_loading")
        chk_path = Path(model_checkpoint_path) if model_checkpoint_path else (ARTIFACT_DIR / "matcher_model.pkl")
        if not chk_path.exists():
            for cand in [
                Path("output/artifacts/matcher_model.pkl"),
                Path("/content/output/artifacts/matcher_model.pkl"),
                Path("/kaggle/working/output/artifacts/matcher_model.pkl"),
                Path("../output/artifacts/matcher_model.pkl")
            ]:
                if cand.exists():
                    chk_path = cand
                    break

        if not chk_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {chk_path}. Please train a model first or specify --model-checkpoint."
            )

        tracker.log("=" * 80)
        tracker.log(f"MODE: INFERENCE ONLY (Using Pretrained Checkpoint: {chk_path})")
        tracker.log("=" * 80)

        matcher = AsymmetricEntityMatcher()
        matcher.load(str(chk_path))
        tracker.log(f"Loaded trained matcher with {len(matcher.feature_names)} features.")

        best_tau = optimal_tau if optimal_tau is not None else 0.390
        best_tau_single = optimal_singleton_tau if optimal_singleton_tau is not None else 0.250

        # Try to auto-load calibrated thresholds from run_summary.json
        summary_candidates = [
            tracker.log_dir / "kaggle_run" / "run_summary.json",
            tracker.log_dir / "run_summary.json",
            Path("logs/kaggle_run/run_summary.json"),
            Path("/content/logs/kaggle_run/run_summary.json"),
            Path("/kaggle/working/logs/kaggle_run/run_summary.json")
        ]
        if optimal_tau is None:
            for sc in summary_candidates:
                if sc.exists():
                    try:
                        import json
                        with open(sc) as sf:
                            s_data = json.load(sf)
                        best_tau = float(s_data.get("threshold_metrics", {}).get("optimal_tau", best_tau))
                        best_tau_single = float(s_data.get("threshold_metrics", {}).get("optimal_singleton_tau", best_tau_single))
                        val_macro_f05 = float(s_data.get("threshold_metrics", {}).get("validation_macro_f05", 0.0))
                        tracker.log(f"Auto-loaded calibrated operating thresholds from {sc}: tau = {best_tau:.3f}, tau_singleton = {best_tau_single:.3f}")
                        break
                    except Exception:
                        pass

        tracker.log(f"Operating Thresholds: Match tau = {best_tau:.3f}, Singleton tau_singleton = {best_tau_single:.3f}")
        tracker.stop_timer("model_loading")
    else:
        # ──────────────────────────────────────────────────────────────────────────
        # PHASE 1: DATA LOADING & PREPROCESSING
        # ──────────────────────────────────────────────────────────────────────────
        tracker.start_timer("data_loading_and_preprocessing")
        sample_desc = f"{train_sample_size:,}" if train_sample_size else "FULL DATASET (Cloud GPU)"
        tracker.log(f"Phase 1: Loading Training Reference & Targets (Sample: {sample_desc})...")

        t_s1_start = time.time()
        s1_full = pd.read_csv(TRAIN_S1_PATH, sep="\t")
        if train_sample_size is not None:
            if isinstance(train_sample_size, float) and 0.0 < train_sample_size < 1.0:
                n_sample = int(len(s1_full) * train_sample_size)
            else:
                n_sample = int(train_sample_size)
            if n_sample < len(s1_full):
                s1_train = s1_full.sample(n=n_sample, random_state=42).reset_index(drop=True)
            else:
                s1_train = s1_full
        else:
            s1_train = s1_full
    
        s1_train_ids = set(s1_train['entity_id'])
        gt_all = load_ground_truth(str(TRAIN_GT_PATH))
        gt_train = {sid: gt_all[sid] for sid in s1_train_ids if sid in gt_all}
        tracker.add_time("p1_s1_loading", time.time() - t_s1_start, category="sub")
    
        # Collect target IDs matching the S1 sample
        target_ids_in_sample = set()
        for s_set in gt_train.values():
            target_ids_in_sample.update(s_set)
    
        # Efficient target loading when sampling vs full cloud run
        t_tgt_start = time.time()
        if len(s1_train) < len(s1_full):
            tracker.log(f"Streaming target records (100% of true positive targets + scaled distractor background for {len(s1_train):,} entities)...", terminal=False)
            max_distractors_per_source = max(75000, int(len(s1_train) * 0.75))
            distractor_frac = min(0.04, max(0.015, (len(s1_train) / len(s1_full)) * 0.25))
            s2_hits = []
            s2_distractors = []
            for chunk in make_progress_bar(pd.read_csv(TRAIN_S2_PATH, sep="\t", chunksize=500000), desc="Streaming Source 2 Chunks", mininterval=2.0, leave=False):
                h = chunk[chunk['entity_id'].isin(target_ids_in_sample)]
                if len(h) > 0:
                    s2_hits.append(h)
                if sum(len(d) for d in s2_distractors) < max_distractors_per_source:
                    d = chunk[~chunk['entity_id'].isin(target_ids_in_sample)].sample(frac=distractor_frac, random_state=42)
                    s2_distractors.append(d)
            s2_sample = pd.concat(s2_hits + s2_distractors, ignore_index=True)
    
            s3_hits = []
            s3_distractors = []
            for chunk in make_progress_bar(pd.read_csv(TRAIN_S3_PATH, sep="\t", chunksize=500000), desc="Streaming Source 3 Chunks", mininterval=2.0, leave=False):
                h = chunk[chunk['entity_id'].isin(target_ids_in_sample)]
                if len(h) > 0:
                    s3_hits.append(h)
                if sum(len(d) for d in s3_distractors) < max_distractors_per_source:
                    d = chunk[~chunk['entity_id'].isin(target_ids_in_sample)].sample(frac=distractor_frac, random_state=42)
                    s3_distractors.append(d)
            s3_sample = pd.concat(s3_hits + s3_distractors, ignore_index=True)
            s23_train = pd.concat([s2_sample, s3_sample], ignore_index=True).drop_duplicates(subset=['entity_id'])
        else:
            tracker.log("Loading complete target tables (Sources 2 and 3) via streaming...", terminal=False)
            s2_chunks = []
            for chunk in make_progress_bar(pd.read_csv(TRAIN_S2_PATH, sep="\t", chunksize=500_000),
                                           desc="Streaming Source 2 targets", mininterval=2.0, leave=False):
                s2_chunks.append(chunk)
            s2_full = pd.concat(s2_chunks, ignore_index=True)
            del s2_chunks
    
            s3_chunks = []
            for chunk in make_progress_bar(pd.read_csv(TRAIN_S3_PATH, sep="\t", chunksize=500_000),
                                           desc="Streaming Source 3 targets", mininterval=2.0, leave=False):
                s3_chunks.append(chunk)
            s3_full = pd.concat(s3_chunks, ignore_index=True)
            del s3_chunks
    
            s23_train = pd.concat([s2_full, s3_full], ignore_index=True).drop_duplicates(subset=['entity_id'])
            del s2_full, s3_full
        tracker.add_time("p1_target_streaming", time.time() - t_tgt_start, category="sub")
    
        tracker.log(f"Loaded: {len(s1_train):,} Reference S1 | {len(s23_train):,} Target Records (S2+S3)")
    
        # Determine whether to drop raw columns (save memory on large datasets)
        n_targets_total = len(s23_train)
        drop_raw = n_targets_total > 100_000
        tracker.log("Executing high-speed vectorized text cleaning, transliteration, and postal normalization...", terminal=False)
        s1_train_proc = preprocess_dataframe(s1_train, is_reference=True, drop_raw_columns=drop_raw)
        s23_train_proc = preprocess_dataframe(s23_train, is_reference=False, drop_raw_columns=drop_raw)
        del s23_train
        import gc; gc.collect()
        tracker.stop_timer("data_loading_and_preprocessing")
    
        # ──────────────────────────────────────────────────────────────────────────
        # PHASE 2: COUNTRY-PARTITIONED DUAL BLOCKING
        # ──────────────────────────────────────────────────────────────────────────
        tracker.start_timer("blocking")
        tracker.log("Phase 2: Generating Candidate Pairs via Country-Partitioned Inverted Indexing & RRF Fusion...")
        all_candidates: Dict[str, Set[str]] = {}
        train_provenance: Dict[Tuple[str, str], Dict[str, float]] = {}
        dense_model = dense_model_name if use_dense else None
    
        for country in sorted(s1_train_proc['country'].unique()):
            s1_c = s1_train_proc[s1_train_proc['country'] == country]
            s23_c = s23_train_proc[s23_train_proc['country'] == country]
            tracker.log(f"  Blocking country [{country}]: {len(s1_c):,} S1 against {len(s23_c):,} targets...")

            # Fit blockers once per country
            lex_blocker, dense_blocker = fit_hybrid_blocker(
                s23_c,
                max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY,
                dense_model_name=dense_model
            )

            # Bug 6 Fix: Batch S1 queries in 50k chunks to prevent memory explosion with dense encoding
            s1_batch_size = 50_000
            n_s1 = len(s1_c)
            for b_start in range(0, n_s1, s1_batch_size):
                b_end = min(b_start + s1_batch_size, n_s1)
                batch_s1 = s1_c.iloc[b_start:b_end]
                c_dict, prov_dict = query_hybrid_blocker(
                    batch_s1, lex_blocker, dense_blocker=dense_blocker,
                    max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY,
                    return_provenance=True,
                    tracker=tracker
                )
                all_candidates.update(c_dict)
                train_provenance.update(prov_dict)

            del lex_blocker, dense_blocker
            import gc; gc.collect()
    
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
        total_cartesian = float(len(s1_train)) * float(n_targets_total)
        reduction_ratio = (1.0 - (float(total_candidate_pairs) / total_cartesian)) * 100.0 if total_cartesian > 0 else 100.0
        candidates_per_entity = total_candidate_pairs / float(len(s1_train)) if len(s1_train) > 0 else 0.0
    
        # Country-specific blocking breakdown (US vs India)
        country_blocking_metrics = {}
        for c_val in sorted(s1_train_proc['country'].unique()):
            c_s1 = s1_train_proc[s1_train_proc['country'] == c_val]
            c_s1_ids = set(c_s1['entity_id'])
            c_true = sum(len(gt_train.get(sid, set())) for sid in c_s1_ids)
            c_captured = sum(len(all_candidates.get(sid, set()) & gt_train.get(sid, set())) for sid in c_s1_ids)
            c_candidates = sum(len(all_candidates.get(sid, set())) for sid in c_s1_ids)
            c_recall = (c_captured / c_true * 100.0) if c_true > 0 else 0.0
            c_avg_cand = c_candidates / len(c_s1_ids) if len(c_s1_ids) > 0 else 0.0
    
            country_blocking_metrics[c_val] = {
                "entities": len(c_s1_ids),
                "true_matches": c_true,
                "captured_matches": c_captured,
                "pair_completeness_pct": round(c_recall, 3),
                "avg_candidates_per_entity": round(c_avg_cand, 2)
            }
    
        tracker.log("\n" + "=" * 75)
        tracker.log("BLOCKING RECALL BREAKDOWN BY COUNTRY (US vs INDIA):")
        for c_name, c_data in country_blocking_metrics.items():
            tracker.log(f"  * {c_name:6s}: Blocking Recall (PC) = {c_data['pair_completeness_pct']:6.2f}% | Captured {c_data['captured_matches']:,} / {c_data['true_matches']:,} matches | Avg Cands: {c_data['avg_candidates_per_entity']:.1f}")
        tracker.log(f"  * OVERALL: Blocking Recall (PC) = {pair_completeness:6.2f}% | Captured {captured_matches:,} / {total_true_matches:,} matches | Avg Cands: {candidates_per_entity:.1f}")
        tracker.log("=" * 75 + "\n")
    
        tracker.record_blocking_metrics(
            total_true_matches=total_true_matches,
            captured_matches=captured_matches,
            pair_completeness=pair_completeness,
            total_candidates=total_candidate_pairs,
            candidates_per_entity=candidates_per_entity,
            reduction_ratio=reduction_ratio,
            by_country=country_blocking_metrics
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
        X, feat_names = build_feature_matrix(
            pair_list, s1_train_proc, s23_train_proc,
            provenance_map=train_provenance
        )
        del train_provenance
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
    
        # Country-specific F0.5 evaluation (US vs India)
        val_preds = apply_decision_rules(
            pair_list, oof_probs, list(s1_train_ids),
            match_threshold=best_tau,
            singleton_threshold=best_tau_single,
            max_matches=MAX_MATCHES_PER_ENTITY
        )
        tracker.log("\n" + "=" * 75)
        tracker.log("VALIDATION MACRO F0.5 BREAKDOWN BY COUNTRY (US vs INDIA):")
        for c_val in sorted(s1_train_proc['country'].unique()):
            c_ids = s1_train_proc[s1_train_proc['country'] == c_val]['entity_id'].tolist()
            c_f05 = evaluate_macro_f05(val_preds, gt_train, c_ids)
            tracker.log(f"  * {c_val:6s}: Macro F0.5 = {c_f05:.4f} ({len(c_ids):,} entities)")
        tracker.log(f"  * OVERALL: Macro F0.5 = {val_macro_f05:.4f} ({len(s1_train_ids):,} entities)")
        tracker.log("=" * 75 + "\n")
    
        tracker.stop_timer("threshold_tuning")
    
        # Error Analysis Diagnostic Dump
        tracker.log("Generating Error Analysis Diagnostics (False Positives and False Negatives)...")
        tracker.dump_error_analysis(
            pair_list, labels, oof_probs, s1_train_proc, s23_train_proc, tau=best_tau, top_k_errors=50
        )
        tracker.finalize(val_macro_f05, best_tau, best_tau_single)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 6: TEST SET INFERENCE & VALIDATION
    # ──────────────────────────────────────────────────────────────────────────
    if test_mode:
        import gc
        tracker.log("Freeing training memory before launching test inference...")
        try:
            del X, labels, oof_probs, pair_list, val_preds
            del s1_train, s1_train_proc, s23_train_proc, all_candidates
            del gt_all, gt_train
        except Exception:
            pass
        gc.collect()

        tracker.start_timer("test_inference")
        tracker.log("\n" + "=" * 80)
        tracker.log("PHASE 6: MEMORY-EFFICIENT SHARDED TEST INFERENCE")
        tracker.log("=" * 80)

        import gc
        from features import extract_features_and_score_chunked

        reranker = None
        if use_reranker:
            r_name = reranker_model_name or RERANKER_MODEL_NAME
            tracker.log(f"Initializing Multilingual Cross-Encoder Reranker ({r_name})...")
            reranker = SelectiveCrossEncoderReranker(model_name=r_name)

        test_s1_full = pd.read_csv(TEST_S1_PATH, sep="\t")
        test_s1_ids = test_s1_full['entity_id'].tolist()
        total_test_s1 = len(test_s1_ids)
        tracker.log(f"Total Test S1 entities: {total_test_s1:,}")

        if test_sample_size and test_sample_size < total_test_s1:
            tracker.log(f"Sampling {test_sample_size:,} Test S1 entities for dry-run verification...")
            test_s1_to_process = test_s1_full.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        else:
            test_s1_to_process = test_s1_full

        # Fix 1.5: Preprocess Test S1 globally and compute name frequencies once across full test set
        tracker.log("Preprocessing Test S1 entities globally...")
        t_p6_s1 = time.time()
        test_s1_proc = preprocess_dataframe(test_s1_to_process, is_reference=True, drop_raw_columns=True)
        global_test_s1_freqs = test_s1_proc['norm_name'].value_counts().to_dict()
        tracker.add_time("p6_s1_preprocess", time.time() - t_p6_s1, category="sub")
        tracker.log(f"Global Test S1 frequencies computed: {len(global_test_s1_freqs):,} unique names across {len(test_s1_proc):,} entities.")

        # Fix 2.3: Single-pass streaming of S2 and S3 targets partitioned by country
        tracker.log("Streaming S2 and S3 targets in a single pass (partitioned by country)...")
        t_p6_tgt_stream = time.time()
        s23_by_country = {c: [] for c in ["US", "India", "France"]}
        for s_path, s_name in [(TEST_S2_PATH, "S2"), (TEST_S3_PATH, "S3")]:
            tracker.log(f"  Streaming {s_name} from {s_path}...")
            for chunk in make_progress_bar(pd.read_csv(s_path, sep="\t", chunksize=500_000),
                                           desc=f"  Streaming {s_name}", mininterval=2.0, leave=False):
                for c in ["US", "India", "France"]:
                    sub = chunk[chunk['country'] == c]
                    if len(sub) > 0:
                        s23_by_country[c].append(sub)
        gc.collect()
        tracker.add_time("p6_target_streaming", time.time() - t_p6_tgt_stream, category="sub")

        # Initialize output files with headers (will append per-shard results)
        os.makedirs(os.path.dirname(str(OUTPUT_MATCHING_PATH)), exist_ok=True)
        os.makedirs(os.path.dirname(str(OUTPUT_CANDIDATE_PATH)), exist_ok=True)

        matching_header_written = False
        candidate_header_written = False

        # Track all S1 IDs that got written (to fill missing singletons at the end)
        written_s1_ids = set()

        for country in ["US", "India", "France"]:
            tracker.log(f"\n{'─' * 70}")
            tracker.log(f"SHARD [{country}]: Loading, Blocking, Scoring...")
            tracker.log(f"{'─' * 70}")

            # ── 6a. Load only this country's S1 ──
            shard_s1_proc = test_s1_proc[test_s1_proc['country'] == country].reset_index(drop=True)
            if len(shard_s1_proc) == 0:
                tracker.log(f"  [{country}] No S1 entities found, skipping.")
                continue
            tracker.log(f"  [{country}] S1 entities: {len(shard_s1_proc):,}")

            # ── 6b. Assemble and deduplicate this country's S2+S3 targets ──
            raw_country_chunks = s23_by_country.pop(country, [])
            if raw_country_chunks:
                shard_s23 = pd.concat(raw_country_chunks, ignore_index=True).drop_duplicates(subset=['entity_id'])
            else:
                shard_s23 = pd.DataFrame()
            del raw_country_chunks
            gc.collect()
            tracker.log(f"  [{country}] Loaded {len(shard_s23):,} target records (S2+S3)")

            # ── 6c. Preprocess targets (drop raw columns to save memory) ──
            tracker.log(f"  [{country}] Preprocessing targets...")
            t_p6_tgt_proc = time.time()
            shard_s23_proc = preprocess_dataframe(shard_s23, is_reference=False, drop_raw_columns=True)
            del shard_s23
            gc.collect()
            tracker.add_time("p6_target_preprocess", time.time() - t_p6_tgt_proc, category="sub")

            # ── 6d. Fit Blocker on Shard Targets ──
            tracker.log(f"  [{country}] Indexing {len(shard_s23_proc):,} target records for blocking...")
            t_p6_idx = time.time()
            lex_blocker, dense_blocker = fit_hybrid_blocker(
                shard_s23_proc,
                max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY,
                dense_model_name=dense_model
            )
            tracker.add_time("p6_blocker_indexing", time.time() - t_p6_idx, category="sub")

            # Index target DataFrame by entity_id for fast on-demand batch lookup (eliminates 8.5GB dictionary overhead)
            shard_s23_indexed = shard_s23_proc.set_index('entity_id')
            del shard_s23_proc
            gc.collect()

            # ── 6e. Process S1 Entities in Memory-Bounded Batches ──
            s1_batch_size = 50_000
            n_shard_s1 = len(shard_s1_proc)
            n_batches = int(np.ceil(n_shard_s1 / s1_batch_size))
            tracker.log(f"  [{country}] Processing {n_shard_s1:,} S1 entities in {n_batches} batch(es) of {s1_batch_size:,}...")

            for b_idx, b_start in enumerate(range(0, n_shard_s1, s1_batch_size), start=1):
                b_end = min(b_start + s1_batch_size, n_shard_s1)
                batch_s1_df = shard_s1_proc.iloc[b_start:b_end]
                batch_s1_ids = batch_s1_df['entity_id'].tolist()

                # Query blocker for this batch with multi-channel RRF and provenance
                batch_candidates, batch_prov = query_hybrid_blocker(
                    batch_s1_df, lex_blocker, dense_blocker=dense_blocker,
                    max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY,
                    return_provenance=True,
                    tracker=tracker
                )

                # Write candidate_pairs.tsv (append mode)
                t_p6_tsv = time.time()
                write_mode = "w" if not candidate_header_written else "a"
                with open(str(OUTPUT_CANDIDATE_PATH), write_mode, encoding="utf-8") as f:
                    if not candidate_header_written:
                        f.write("source1_entity_id\tcandidate_entity_ids\n")
                        candidate_header_written = True
                    for s1_id in batch_s1_ids:
                        c_ids = batch_candidates.get(s1_id, set())
                        if c_ids:
                            f.write(f"{s1_id}\t{','.join(sorted(c_ids))}\n")
                        else:
                            f.write(f"{s1_id}\t\n")
                        written_s1_ids.add(s1_id)
                tracker.add_time("p6_tsv_writing", time.time() - t_p6_tsv, category="sub")

                # Build lookup dict ONLY for targets retrieved in this batch (< 50MB instead of 8.5GB)
                needed_tids = list({tid for cset in batch_candidates.values() for tid in cset if tid in shard_s23_indexed.index})
                batch_target_lookup = shard_s23_indexed.loc[needed_tids].to_dict('index')

                # Extract features + score in 250K chunks with candidate provenance
                t_p6_feat = time.time()
                batch_pairs = [(sid, tid) for sid, cset in batch_candidates.items() for tid in cset]
                batch_s1_lookup = batch_s1_df.set_index('entity_id').to_dict('index')

                scored_triples = extract_features_and_score_chunked(
                    batch_pairs, batch_s1_lookup, batch_target_lookup, global_test_s1_freqs,
                    matcher, chunk_size=250_000,
                    provenance_map=batch_prov
                )
                tracker.add_time("p6_feature_scoring", time.time() - t_p6_feat, category="sub")

                # Optional Selective Cross-Encoder Reranking on the ambiguous probability tail
                t_p6_rerank = time.time()
                if reranker is not None and reranker.model is not None:
                    # Group ambiguous pairs by s1 entity and take at most RERANK_TOP_K
                    s1_ambiguous = defaultdict(list)
                    for i, (sid, tid, p) in enumerate(scored_triples):
                        if RERANK_AMBIGUOUS_MIN <= p <= RERANK_AMBIGUOUS_MAX:
                            s1_ambiguous[sid].append((i, p))

                    ambiguous_indices = []
                    for sid, cand_list in s1_ambiguous.items():
                        cand_list.sort(key=lambda x: x[1], reverse=True)
                        for item in cand_list[:RERANK_TOP_K]:
                            ambiguous_indices.append(item[0])
                    if ambiguous_indices:
                        pairs_to_rerank = [
                            (
                                f"{batch_s1_lookup.get(scored_triples[i][0], {}).get('norm_name', '')} {batch_s1_lookup.get(scored_triples[i][0], {}).get('norm_address', '')}",
                                f"{batch_target_lookup.get(scored_triples[i][1], {}).get('norm_name', '')} {batch_target_lookup.get(scored_triples[i][1], {}).get('norm_address', '')}"
                            )
                            for i in ambiguous_indices
                        ]
                        rerank_scores = reranker.rerank_pairs(pairs_to_rerank)
                        updated_triples = list(scored_triples)
                        for idx, r_score in zip(ambiguous_indices, rerank_scores):
                            sid, tid, orig_p = updated_triples[idx]
                            blended_p = 0.5 * orig_p + 0.5 * float(r_score)
                            updated_triples[idx] = (sid, tid, blended_p)
                        scored_triples = updated_triples
                tracker.add_time("p6_reranker", time.time() - t_p6_rerank, category="sub")

                del batch_target_lookup, batch_s1_lookup, batch_pairs, batch_candidates, batch_prov
                gc.collect()

                # Group scores by S1 entity and apply thresholds
                entity_preds = defaultdict(list)
                for s1_id, t_id, prob in scored_triples:
                    entity_preds[s1_id].append((t_id, prob))
                del scored_triples
                gc.collect()

                t_p6_tsv2 = time.time()
                write_mode = "w" if not matching_header_written else "a"
                with open(str(OUTPUT_MATCHING_PATH), write_mode, encoding="utf-8") as f:
                    if not matching_header_written:
                        f.write("source1_entity_id\tmatched_entity_ids\n")
                        matching_header_written = True

                    for s1_id in batch_s1_ids:
                        c_list = entity_preds.get(s1_id, [])
                        if not c_list:
                            f.write(f"{s1_id}\t\n")
                            continue
                        max_p = max(p for _, p in c_list)
                        if max_p < best_tau_single:
                            f.write(f"{s1_id}\t\n")
                        else:
                            matches = [tid for tid, p in sorted(c_list, key=lambda x: x[1], reverse=True)
                                       if p >= best_tau][:MAX_MATCHES_PER_ENTITY]
                            f.write(f"{s1_id}\t{','.join(matches)}\n")
                tracker.add_time("p6_tsv_writing", time.time() - t_p6_tsv2, category="sub")

                del entity_preds
                gc.collect()

            del lex_blocker, dense_blocker, shard_s1_proc, shard_s23_indexed
            gc.collect()
            tracker.log(f"  [{country}] All batches complete. Memory released.")

        # Fill in any S1 IDs that weren't in any shard (should be none, but defensive)
        missing_s1 = set(test_s1_ids) - written_s1_ids
        if missing_s1:
            tracker.log(f"  [Cleanup] Writing {len(missing_s1):,} missing S1 IDs as singletons...")
            with open(str(OUTPUT_MATCHING_PATH), "a", encoding="utf-8") as f:
                for s1_id in sorted(missing_s1):
                    f.write(f"{s1_id}\t\n")
            with open(str(OUTPUT_CANDIDATE_PATH), "a", encoding="utf-8") as f:
                for s1_id in sorted(missing_s1):
                    f.write(f"{s1_id}\t\n")

        del test_s1_full, test_s1_proc, global_test_s1_freqs
        gc.collect()

        tracker.log(f"\nTest inference complete: {total_test_s1:,} S1 entities processed across 3 country shards.")
        tracker.stop_timer("test_inference")

        # Official Validator
        t_p6_val = time.time()
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
        tracker.add_time("p6_validator", time.time() - t_p6_val, category="sub")

        # Finalize and print full timing summary table
        final_f05 = val_macro_f05 if 'val_macro_f05' in locals() else 0.0
        tracker.finalize(final_f05, best_tau, best_tau_single)


def run_loco_evaluation(
    sample_size: Optional[int] = None,
    device: str = "cpu",
    use_dense: bool = False,
    dense_model_name: str = SBERT_MODEL_NAME,
    log_dir: str = "logs",
    experiment_name: Optional[str] = "loco_france_proxy"
):
    """
    Leave-One-Country-Out (LOCO) France Proxy Validation Protocol:
    1. Trains Asymmetric LightGBM strictly on US data.
    2. Validates domain transfer on India data (unseen country).
    3. Evaluates Macro F0.5 with stratified singleton/multi-match breakdown.
    This serves as an empirical verification of domain transfer for zero-shot France.
    """
    tracker = ExperimentTracker(log_dir=log_dir, experiment_name=experiment_name)
    tracker.log("=" * 80)
    tracker.log("LEAVE-ONE-COUNTRY-OUT (LOCO) DOMAIN-TRANSFER VALIDATION (US -> INDIA)")
    tracker.log("=" * 80)
    tracker.log("Empirical validation of language/domain invariance for zero-shot France test set.")

    # 1. Load data
    tracker.start_timer("loco_data_loading")
    s1_full = pd.read_csv(TRAIN_S1_PATH, sep="\t")
    gt_all = load_ground_truth(str(TRAIN_GT_PATH))

    # Split by country
    s1_us = s1_full[s1_full['country'] == 'US'].reset_index(drop=True)
    s1_in = s1_full[s1_full['country'] == 'India'].reset_index(drop=True)

    if sample_size:
        half_sample = max(100, sample_size // 2)
        s1_us = s1_us.sample(n=min(len(s1_us), half_sample), random_state=42).reset_index(drop=True)
        s1_in = s1_in.sample(n=min(len(s1_in), half_sample), random_state=42).reset_index(drop=True)

    tracker.log(f"Training Domain [US]: {len(s1_us):,} entities | Validation Domain [India]: {len(s1_in):,} entities")

    gt_us = {sid: gt_all[sid] for sid in s1_us['entity_id'] if sid in gt_all}
    gt_in = {sid: gt_all[sid] for sid in s1_in['entity_id'] if sid in gt_all}

    # Collect target IDs
    needed_targets = set()
    for s in list(gt_us.values()) + list(gt_in.values()):
        needed_targets.update(s)

    # Load targets (Sources 2 and 3)
    s2_chunks = []
    for chunk in make_progress_bar(pd.read_csv(TRAIN_S2_PATH, sep="\t", chunksize=250000), desc="  [Data] Streaming Source 2 Chunks", mininterval=2.0, leave=False):
        hits = chunk[chunk['entity_id'].isin(needed_targets)]
        distractors = chunk[~chunk['entity_id'].isin(needed_targets)].sample(frac=0.015, random_state=42)
        s2_chunks.append(pd.concat([hits, distractors]))
        if sum(len(c) for c in s2_chunks) > 120000:
            break
    s2_sample = pd.concat(s2_chunks, ignore_index=True)

    s3_chunks = []
    for chunk in make_progress_bar(pd.read_csv(TRAIN_S3_PATH, sep="\t", chunksize=250000), desc="  [Data] Streaming Source 3 Chunks", mininterval=2.0, leave=False):
        hits = chunk[chunk['entity_id'].isin(needed_targets)]
        distractors = chunk[~chunk['entity_id'].isin(needed_targets)].sample(frac=0.015, random_state=42)
        s3_chunks.append(pd.concat([hits, distractors]))
        if sum(len(c) for c in s3_chunks) > 120000:
            break
    s3_sample = pd.concat(s3_chunks, ignore_index=True)
    s23_df = pd.concat([s2_sample, s3_sample], ignore_index=True).drop_duplicates(subset=['entity_id'])

    # Preprocess
    s1_us_proc = preprocess_dataframe(s1_us, is_reference=True)
    s1_in_proc = preprocess_dataframe(s1_in, is_reference=True)
    s23_proc = preprocess_dataframe(s23_df, is_reference=False)
    tracker.stop_timer("loco_data_loading")

    # 2. Blocking on US (Train) and India (Validation)
    tracker.start_timer("loco_blocking")
    s23_us = s23_proc[s23_proc['country'] == 'US']
    s23_in = s23_proc[s23_proc['country'] == 'India']

    dense_model = dense_model_name if use_dense else None
    if device == "cuda":
        os.environ["USE_GPU"] = "1"

    tracker.log("Blocking US training entities...")
    us_cands = run_hybrid_blocking(s1_us_proc, s23_us, max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY, dense_model_name=dense_model)
    tracker.log("Blocking India validation entities...")
    in_cands = run_hybrid_blocking(s1_in_proc, s23_in, max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY, dense_model_name=dense_model)
    tracker.stop_timer("loco_blocking")

    # 3. Features & Train on US
    tracker.start_timer("loco_training")
    us_pairs, us_labels = [], []
    for s1_id, c_set in us_cands.items():
        true_set = gt_us.get(s1_id, set())
        for tid in c_set:
            us_pairs.append((s1_id, tid))
            us_labels.append(1 if tid in true_set else 0)

    X_us, feat_names = build_feature_matrix(us_pairs, s1_us_proc, s23_us)
    y_us = np.array(us_labels, dtype=np.int32)
    tracker.log(f"US Training Pairs: {len(X_us):,} with {len(feat_names)} domain-invariant features.")

    matcher = AsymmetricEntityMatcher(device=device)
    matcher.fit(X_us, y_us, feature_names=feat_names)
    tracker.stop_timer("loco_training")

    # 4. Predict on India & Tune Thresholds
    tracker.start_timer("loco_evaluation")
    in_pairs, in_labels = [], []
    for s1_id, c_set in in_cands.items():
        true_set = gt_in.get(s1_id, set())
        for tid in c_set:
            in_pairs.append((s1_id, tid))
            in_labels.append(1 if tid in true_set else 0)

    X_in, _ = build_feature_matrix(in_pairs, s1_in_proc, s23_in)
    in_probs = matcher.predict_proba(X_in)
    tracker.log(f"India Validation Pairs: {len(X_in):,}")

    in_s1_ids = s1_in['entity_id'].tolist()
    best_tau, best_tau_single, val_macro_f05, _ = tune_decision_thresholds(
        in_pairs, in_probs, gt_in, in_s1_ids
    )

    tracker.log("\n" + "=" * 80)
    tracker.log(f">> LOCO ZERO-SHOT TRANSFER (US -> INDIA) MACRO F0.5 = {val_macro_f05:.4f} <<")
    tracker.log(f">> Optimal Operating Thresholds: tau={best_tau:.3f}, tau_singleton={best_tau_single:.3f} <<")
    tracker.log("=" * 80)
    tracker.stop_timer("loco_evaluation")
    tracker.finalize(val_macro_f05, best_tau, best_tau_single)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 Amazon Entity Resolution — Approach 4 Pipeline")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "loco_check"],
                        help="Execution mode: 'standard' (full/CV pipeline) or 'loco_check' (Leave-One-Country-Out France proxy)")
    parser.add_argument("--sample-size", type=int, default=None, help="Train sample size (default: None for full dataset on Cloud GPU)")
    parser.add_argument("--test", action="store_true", help="Run full test set inference and generate submission TSVs")
    parser.add_argument("--test-sample-size", type=int, default=None, help="Sample size for test set inference verification")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Compute device for models (cpu or cuda)")
    parser.add_argument("--use-dense", action="store_true", help="Enable dense multilingual vector blocking (recommended on Cloud GPU)")
    parser.add_argument("--dense-model", type=str, default=SBERT_MODEL_NAME, help="HuggingFace model for dense embeddings")
    parser.add_argument("--use-reranker", action="store_true", help="Enable multilingual cross-encoder reranking on ambiguous predictions")
    parser.add_argument("--reranker-model", type=str, default=RERANKER_MODEL_NAME, help="HuggingFace model for cross-encoder reranking")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save execution logs and experiment metrics")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name identifier")

    args = parser.parse_args()

    if args.mode == "loco_check":
        run_loco_evaluation(
            sample_size=args.sample_size,
            device=args.device,
            log_dir=args.log_dir,
            experiment_name=args.exp_name or "loco_france_proxy"
        )
    else:
        run_pipeline(
            train_sample_size=args.sample_size,
            test_mode=args.test,
            test_sample_size=args.test_sample_size,
            use_dense=args.use_dense,
            dense_model_name=args.dense_model,
            use_reranker=args.use_reranker,
            reranker_model_name=args.reranker_model,
            device=args.device,
            log_dir=args.log_dir,
            experiment_name=args.exp_name
        )
