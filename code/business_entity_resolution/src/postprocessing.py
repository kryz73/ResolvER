"""
ML Challenge 2026 — Business Entity Resolution
Metric-Aligned Post-Processing, Singleton Guardrail & TSV Exporter
"""

import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_f05_score(pred_set: Set[str], true_set: Set[str]) -> float:
    """
    Computes per-entity F0.5 score:
      F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singleton rule:
      If true_set is empty: 1.0 if pred_set is empty, else 0.0
    """
    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0

    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0

    precision = float(tp) / float(len(pred_set))
    recall = float(tp) / float(len(true_set))

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> float:
    """Computes Macro F0.5 average across all required Source 1 entities."""
    total_score = 0.0
    for s1_id in all_s1_ids:
        preds = predictions.get(s1_id, set())
        trues = ground_truth.get(s1_id, set())
        total_score += compute_f05_score(preds, trues)

    return total_score / float(len(all_s1_ids)) if all_s1_ids else 0.0


from config import (
    TAU_SWEEP_MIN, TAU_SWEEP_MAX, TAU_SWEEP_STEP,
    SINGLETON_SWEEP_MIN, SINGLETON_SWEEP_MAX, SINGLETON_SWEEP_STEP
)


def evaluate_stratified_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> Dict[str, float]:
    """
    Computes overall Macro F0.5 and stratified breakdown across:
    1. Singletons (0 true matches)
    2. Single-match entities (1 true match)
    3. Multi-match entities (2-11 true matches)
    """
    singleton_scores = []
    single_match_scores = []
    multi_match_scores = []
    all_scores = []

    for s1_id in all_s1_ids:
        preds = predictions.get(s1_id, set())
        trues = ground_truth.get(s1_id, set())
        score = compute_f05_score(preds, trues)
        all_scores.append(score)

        if not trues:
            singleton_scores.append(score)
        elif len(trues) == 1:
            single_match_scores.append(score)
        else:
            multi_match_scores.append(score)

    return {
        "overall_f05": float(np.mean(all_scores)) if all_scores else 0.0,
        "singleton_f05": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "single_match_f05": float(np.mean(single_match_scores)) if single_match_scores else 0.0,
        "multi_match_f05": float(np.mean(multi_match_scores)) if multi_match_scores else 0.0,
        "n_singletons": len(singleton_scores),
        "n_single_match": len(single_match_scores),
        "n_multi_match": len(multi_match_scores),
        "n_total": len(all_scores)
    }


def tune_decision_thresholds(
    candidate_pairs: List[Tuple[str, str]],
    pred_probs: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> Tuple[float, float, float, List[Dict]]:
    """
    Grid-searches the optimal decision threshold (tau) and singleton threshold (tau_single)
    to directly maximize the Macro F0.5 evaluation metric on OOF validation predictions.
    Returns: (best_tau, best_tau_single, best_macro_f05, tuning_records)
    """
    # Group predictions by S1 entity
    entity_candidates = defaultdict(list)
    for (s1_id, t_id), prob in zip(candidate_pairs, pred_probs):
        entity_candidates[s1_id].append((t_id, float(prob)))

    best_score = -1.0
    best_tau = 0.82
    best_tau_single = 0.75
    best_preds = {}
    tuning_records = []

    tau_range = np.arange(TAU_SWEEP_MIN, TAU_SWEEP_MAX + 1e-5, TAU_SWEEP_STEP)
    singleton_range = np.arange(SINGLETON_SWEEP_MIN, SINGLETON_SWEEP_MAX + 1e-5, SINGLETON_SWEEP_STEP)

    print(f"  [Threshold Tuning] Grid searching tau in [{TAU_SWEEP_MIN:.2f}, {TAU_SWEEP_MAX:.2f}] & tau_single in [{SINGLETON_SWEEP_MIN:.2f}, {SINGLETON_SWEEP_MAX:.2f}]...")
    for tau in tqdm(tau_range, desc="  [Threshold Tuning] Grid Search", mininterval=0.5):
        for tau_single in singleton_range:
            preds = {}
            for s1_id in all_s1_ids:
                c_list = entity_candidates.get(s1_id, [])
                if not c_list:
                    preds[s1_id] = set()
                    continue

                max_p = max(p for _, p in c_list)
                # Singleton Bouncer: if best probability < tau_single, predict empty
                if max_p < tau_single:
                    preds[s1_id] = set()
                else:
                    # Filter candidates above tau, sort by prob, cap at 11
                    filtered = [tid for tid, p in sorted(c_list, key=lambda x: x[1], reverse=True) if p >= tau][:11]
                    preds[s1_id] = set(filtered)

            score = evaluate_macro_f05(preds, ground_truth, all_s1_ids)
            tuning_records.append({
                "tau": round(float(tau), 4),
                "tau_singleton": round(float(tau_single), 4),
                "macro_f05": round(float(score), 4)
            })
            if score > best_score:
                best_score = score
                best_tau = float(tau)
                best_tau_single = float(tau_single)
                best_preds = preds

    strat = evaluate_stratified_f05(best_preds, ground_truth, all_s1_ids)
    print(f"  [Threshold Tuning] Best Macro F0.5 = {best_score:.4f} (tau={best_tau:.3f}, tau_single={best_tau_single:.3f})")
    print(f"    Stratified Breakdown:")
    print(f"      - Overall Macro F0.5:     {strat['overall_f05']:.4f}")
    print(f"      - Singletons ({strat['n_singletons']:,}):       {strat['singleton_f05']:.4f}")
    print(f"      - 1-Match Entities ({strat['n_single_match']:,}): {strat['single_match_f05']:.4f}")
    print(f"      - Multi-Match ({strat['n_multi_match']:,}):      {strat['multi_match_f05']:.4f}")

    return best_tau, best_tau_single, best_score, tuning_records



def apply_decision_rules(
    candidate_pairs: List[Tuple[str, str]],
    pred_probs: np.ndarray,
    all_s1_ids: List[str],
    match_threshold: float = 0.82,
    singleton_threshold: float = 0.75,
    max_matches: int = 11
) -> Dict[str, Set[str]]:
    """
    Converts candidate match probabilities into final discrete entity match predictions
    using metric-aligned thresholding and singleton protection.
    """
    entity_candidates = defaultdict(list)
    for (s1_id, t_id), prob in zip(candidate_pairs, pred_probs):
        entity_candidates[s1_id].append((t_id, float(prob)))

    final_matches = {}
    for s1_id in all_s1_ids:
        c_list = entity_candidates.get(s1_id, [])
        if not c_list:
            final_matches[s1_id] = set()
            continue

        max_p = max(p for _, p in c_list)
        # Singleton Guardrail
        if max_p < singleton_threshold:
            final_matches[s1_id] = set()
        else:
            matches = [tid for tid, p in sorted(c_list, key=lambda x: x[1], reverse=True) if p >= match_threshold][:max_matches]
            final_matches[s1_id] = set(matches)

    return final_matches


def write_matching_results_tsv(
    matches_dict: Dict[str, Set[str]],
    required_s1_ids: List[str],
    output_path: str
):
    """
    Writes final entity matches to matching_results.tsv in the official competition format.
    Format: source1_entity_id \t matched_entity_ids (comma-separated, empty for singletons)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    iterator = required_s1_ids
    if len(required_s1_ids) >= 10000:
        iterator = tqdm(required_s1_ids, desc="  [Output] Writing matching_results.tsv", mininterval=1.0)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in iterator:
            m_ids = matches_dict.get(s1_id, set())
            if m_ids:
                f.write(f"{s1_id}\t{','.join(sorted(m_ids))}\n")
            else:
                f.write(f"{s1_id}\t\n")
    print(f"  [Output] Successfully wrote final matches to: {output_path}")
