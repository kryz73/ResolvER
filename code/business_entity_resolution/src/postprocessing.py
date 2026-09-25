"""
ML Challenge 2026 — Business Entity Resolution
Metric-Aligned Post-Processing, Singleton Guardrail & TSV Exporter
"""

import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


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


def tune_decision_thresholds(
    candidate_pairs: List[Tuple[str, str]],
    pred_probs: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> Tuple[float, float, float]:
    """
    Grid-searches the optimal decision threshold (tau) and singleton threshold (tau_single)
    to directly maximize the Macro F0.5 evaluation metric on OOF validation predictions.
    Returns: (best_tau, best_tau_single, best_macro_f05)
    """
    # Group predictions by S1 entity
    entity_candidates = defaultdict(list)
    for (s1_id, t_id), prob in zip(candidate_pairs, pred_probs):
        entity_candidates[s1_id].append((t_id, float(prob)))

    best_score = -1.0
    best_tau = 0.82
    best_tau_single = 0.75
    tuning_records = []

    tau_range = np.linspace(0.35, 0.85, 26)
    singleton_range = np.linspace(0.30, 0.80, 21)

    print("  [Threshold Tuning] Grid searching (tau, tau_single) on validation Macro F0.5...")
    for tau in tau_range:
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

    print(f"  [Threshold Tuning] Best Macro F0.5 = {best_score:.4f} (tau={best_tau:.3f}, tau_single={best_tau_single:.3f})")
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
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in required_s1_ids:
            m_ids = matches_dict.get(s1_id, set())
            if m_ids:
                f.write(f"{s1_id}\t{','.join(sorted(m_ids))}\n")
            else:
                f.write(f"{s1_id}\t\n")
    print(f"  [Output] Successfully wrote final matches to: {output_path}")
