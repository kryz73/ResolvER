"""
ML Challenge 2026 — Business Entity Resolution
Metric-Aligned Post-Processing, Singleton Guardrail & TSV Exporter
"""

import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from logger import make_progress_bar


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
    # Group and pre-sort predictions by S1 entity once (massive speedup)
    entity_candidates = defaultdict(list)
    for (s1_id, t_id), prob in zip(candidate_pairs, pred_probs):
        entity_candidates[s1_id].append((t_id, float(prob)))

    sorted_candidates = {}
    max_probs = {}
    for s1_id in all_s1_ids:
        c_list = entity_candidates.get(s1_id, [])
        if c_list:
            s_list = sorted(c_list, key=lambda x: x[1], reverse=True)[:15]
            sorted_candidates[s1_id] = s_list
            max_probs[s1_id] = s_list[0][1]
        else:
            sorted_candidates[s1_id] = []
            max_probs[s1_id] = 0.0

    gt_sets = {sid: ground_truth.get(sid, set()) for sid in all_s1_ids}
    gt_lens = {sid: len(gt_sets[sid]) for sid in all_s1_ids}
    n_entities = float(len(all_s1_ids)) if all_s1_ids else 1.0

    best_score = -1.0
    best_tau = 0.55
    best_tau_single = 0.45
    best_preds = {}
    tuning_records = []

    tau_range = np.arange(TAU_SWEEP_MIN, TAU_SWEEP_MAX + 1e-5, TAU_SWEEP_STEP)
    singleton_range = np.arange(SINGLETON_SWEEP_MIN, SINGLETON_SWEEP_MAX + 1e-5, SINGLETON_SWEEP_STEP)

    print(f"  [Threshold Tuning] Grid searching tau in [{TAU_SWEEP_MIN:.2f}, {TAU_SWEEP_MAX:.2f}] & tau_single in [{SINGLETON_SWEEP_MIN:.2f}, {SINGLETON_SWEEP_MAX:.2f}]...")
    for tau in make_progress_bar(tau_range, desc="[Threshold Tuning] Grid Search", mininterval=2.0, leave=False):
        for tau_single in singleton_range:
            total_score = 0.0
            for sid in all_s1_ids:
                m_p = max_probs[sid]
                n_true = gt_lens[sid]
                if m_p < tau_single:
                    if n_true == 0:
                        total_score += 1.0
                    continue

                c_list = sorted_candidates[sid]
                pred_list = [tid for tid, p in c_list if p >= tau]
                n_pred = len(pred_list)

                if n_pred == 0 and n_true == 0:
                    total_score += 1.0
                    continue
                if n_pred == 0 or n_true == 0:
                    continue

                tp = sum(1 for tid in pred_list if tid in gt_sets[sid])
                if tp == 0:
                    continue

                prec = float(tp) / float(n_pred)
                rec = float(tp) / float(n_true)
                denom = 0.25 * prec + rec
                if denom > 0:
                    total_score += (1.25 * prec * rec) / denom

            score = total_score / n_entities
            tuning_records.append({
                "tau": round(float(tau), 4),
                "tau_singleton": round(float(tau_single), 4),
                "macro_f05": round(float(score), 4)
            })
            if score > best_score:
                best_score = score
                best_tau = float(tau)
                best_tau_single = float(tau_single)

    # Reconstruct best predictions for stratified report
    for sid in all_s1_ids:
        if max_probs[sid] < best_tau_single:
            best_preds[sid] = set()
        else:
            best_preds[sid] = {tid for tid, p in sorted_candidates[sid] if p >= best_tau}

    strat = evaluate_stratified_f05(best_preds, ground_truth, all_s1_ids)
    print(f"  [Threshold Tuning] Best Macro F0.5 = {best_score:.4f} (tau={best_tau:.3f}, tau_single={best_tau_single:.3f})")
    print(f"    Stratified Breakdown:")
    print(f"      - Overall Macro F0.5:     {strat['overall_f05']:.4f}")
    print(f"      - Singletons ({strat['n_singletons']:,}):       {strat['singleton_f05']:.4f}")
    print(f"      - 1-Match Entities ({strat['n_single_match']:,}): {strat['single_match_f05']:.4f}")
    print(f"      - Multi-Match ({strat['n_multi_match']:,}):      {strat['multi_match_f05']:.4f}")

    return best_tau, best_tau_single, best_score, tuning_records



def select_expected_f05(
    candidates_with_probs: List[Tuple[str, float]],
    tau_singleton: float = 0.45,
    max_k: int = 15
) -> List[str]:
    """
    Greedy Dynamic-K Set Selection maximizing expected Macro F0.5 utility:
      E[F0.5(k)] = (5 * sum_{j=1}^k p_j) / (4 * k + sum_{j=1}^K p_j)
    """
    if not candidates_with_probs:
        return []

    sorted_c = sorted(candidates_with_probs, key=lambda x: x[1], reverse=True)[:max_k]
    if not sorted_c or sorted_c[0][1] < tau_singleton:
        return []

    probs = [p for _, p in sorted_c]
    total_prob_mass = sum(probs)
    if total_prob_mass <= 0:
        return []

    best_k = 0
    best_expected_utility = 0.0
    cum_prob = 0.0

    for k in range(1, len(sorted_c) + 1):
        cum_prob += probs[k - 1]
        denom = 4.0 * float(k) + total_prob_mass
        util = (5.0 * cum_prob) / denom if denom > 0 else 0.0
        if util > best_expected_utility:
            best_expected_utility = util
            best_k = k

    if best_k == 0:
        return []
    return [tid for tid, _ in sorted_c[:best_k]]


def apply_decision_rules(
    candidate_pairs: List[Tuple[str, str]],
    pred_probs: np.ndarray,
    all_s1_ids: List[str],
    match_threshold: float = 0.82,
    singleton_threshold: float = 0.75,
    max_matches: int = 15,
    use_expected_f05: bool = False
) -> Dict[str, Set[str]]:
    """
    Converts candidate match probabilities into final discrete entity match predictions
    using metric-aligned thresholding or expected F0.5 dynamic-k selection.
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

        if use_expected_f05:
            matches = select_expected_f05(c_list, tau_singleton=singleton_threshold, max_k=max_matches)
            final_matches[s1_id] = set(matches)
        else:
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
        iterator = make_progress_bar(required_s1_ids, desc="Writing matching_results.tsv", mininterval=2.0, leave=False)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in iterator:
            m_ids = matches_dict.get(s1_id, set())
            if m_ids:
                f.write(f"{s1_id}\t{','.join(sorted(m_ids))}\n")
            else:
                f.write(f"{s1_id}\t\n")
    print(f"  [Output] Successfully wrote final matches to: {output_path}")
