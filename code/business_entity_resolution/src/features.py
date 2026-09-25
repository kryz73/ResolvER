"""
ML Challenge 2026 — Business Entity Resolution
Multi-Signal Pairwise Feature Extraction Engine
"""

import math
import re
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extracts character n-grams from a string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Computes Jaccard index between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection) / float(union) if union > 0 else 0.0


def extract_pairwise_features(
    s1_row: pd.Series,
    target_row: pd.Series,
    s1_name_freq: int = 1,
    sbert_sim: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes a 25+ signal feature dictionary for a candidate pair (S1, Target).
    """
    feats = {}

    n1 = str(s1_row.get('norm_name', '')).strip()
    n2 = str(target_row.get('norm_name', '')).strip()
    c1 = str(s1_row.get('clean_name', '')).strip()
    c2 = str(target_row.get('clean_name', '')).strip()
    a1 = str(s1_row.get('norm_address', '')).strip()
    a2 = str(target_row.get('norm_address', '')).strip()

    country = s1_row.get('country', 'US')
    target_id = str(target_row.get('entity_id', ''))

    # ── 1. Name String Distances ──────────────────────────────────────────
    feats['name_ratio'] = fuzz.ratio(n1, n2) / 100.0
    feats['name_token_sort'] = fuzz.token_sort_ratio(n1, n2) / 100.0
    feats['name_token_set'] = fuzz.token_set_ratio(n1, n2) / 100.0
    feats['name_partial'] = fuzz.partial_ratio(n1, n2) / 100.0
    feats['clean_name_ratio'] = fuzz.ratio(c1, c2) / 100.0

    # Token-level metrics
    w1 = set(n1.split())
    w2 = set(n2.split())
    feats['name_token_jaccard'] = jaccard_similarity(w1, w2)
    feats['name_3gram_jaccard'] = jaccard_similarity(char_ngrams(n1, 3), char_ngrams(n2, 3))

    # Prefix match & Brand Discriminators
    first_w1 = n1.split()[0] if n1 else ""
    first_w2 = n2.split()[0] if n2 else ""
    feats['name_first_word_match'] = 1.0 if first_w1 and first_w1 == first_w2 else 0.0
    feats['first_word_ratio'] = fuzz.ratio(first_w1, first_w2) / 100.0 if (first_w1 and first_w2) else 0.0
    feats['first_letter_match'] = 1.0 if (first_w1 and first_w2 and first_w1[0] == first_w2[0]) else 0.0

    # Length discrepancies
    l1, l2 = len(n1), len(n2)
    feats['name_len_diff'] = float(abs(l1 - l2))
    feats['name_len_ratio'] = float(min(l1, l2)) / float(max(l1, l2, 1))

    # ── 2. Address String Distances ───────────────────────────────────────
    addr_missing_1 = bool(s1_row.get('address_missing', False)) or not a1 or a1 == 'nan'
    addr_missing_2 = bool(target_row.get('address_missing', False)) or not a2 or a2 == 'nan'

    feats['is_addr_missing_1'] = 1.0 if addr_missing_1 else 0.0
    feats['is_addr_missing_2'] = 1.0 if addr_missing_2 else 0.0
    feats['is_addr_missing_either'] = 1.0 if (addr_missing_1 or addr_missing_2) else 0.0

    if not addr_missing_1 and not addr_missing_2:
        feats['addr_ratio'] = fuzz.ratio(a1, a2) / 100.0
        feats['addr_token_sort'] = fuzz.token_sort_ratio(a1, a2) / 100.0
        feats['addr_token_set'] = fuzz.token_set_ratio(a1, a2) / 100.0
        aw1 = set(a1.split())
        aw2 = set(a2.split())
        feats['addr_token_jaccard'] = jaccard_similarity(aw1, aw2)
    else:
        feats['addr_ratio'] = 0.0
        feats['addr_token_sort'] = 0.0
        feats['addr_token_set'] = 0.0
        feats['addr_token_jaccard'] = 0.0

    # ── 3. Structural & Number Matches ────────────────────────────────────
    nums1 = set(re.findall(r'\b\d+\b', a1))
    nums2 = set(re.findall(r'\b\d+\b', a2))
    if nums1 and nums2:
        feats['number_overlap_jaccard'] = jaccard_similarity(nums1, nums2)
        feats['has_number_exact_match'] = 1.0 if len(nums1 & nums2) > 0 else 0.0
        feats['has_number_conflict'] = 1.0 if len(nums1 & nums2) == 0 else 0.0
    else:
        feats['number_overlap_jaccard'] = 0.5
        feats['has_number_exact_match'] = 0.5
        feats['has_number_conflict'] = 0.0

    p1 = str(s1_row.get('postal_code', '')).strip()
    p2 = str(target_row.get('postal_code', '')).strip()
    if p1 and p2:
        feats['postal_exact_match'] = 1.0 if p1 == p2 else 0.0
        feats['postal_conflict'] = 1.0 if p1 != p2 else 0.0
    else:
        feats['postal_exact_match'] = 0.5
        feats['postal_conflict'] = 0.0

    # ── 4. Franchise & Name Frequency Interaction ─────────────────────────
    feats['name_log_freq'] = math.log1p(s1_name_freq)
    # If a name is ubiquitous (e.g. Subway), require high address score
    feats['freq_x_addr_sort'] = feats['addr_token_sort'] / (1.0 + feats['name_log_freq'])
    feats['name_x_addr_ratio'] = feats['name_ratio'] * feats['addr_ratio']

    # ── 5. Source Indicators ──────────────────────────────────────────────
    feats['is_source_2'] = 1.0 if target_id.startswith('S2-') else 0.0
    feats['is_source_3'] = 1.0 if target_id.startswith('S3-') else 0.0

    # (Note: Raw country one-hot indicators are omitted to enforce complete
    # domain and language invariance for zero-shot generalization to France)

    # ── 6. Semantic Embedding Distance ────────────────────────────────────
    if sbert_sim is not None:
        feats['sbert_cosine'] = float(sbert_sim)

    return feats


def build_feature_matrix(
    candidate_pairs: List[Tuple[str, str]],
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    s1_name_freqs: Optional[Dict[str, int]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Constructs a 2D feature matrix (N_pairs x N_features) from a list of candidate ID pairs
    using high-speed in-memory dictionary lookups.
    Returns: (X, feature_names)
    """
    s1_lookup = s1_df.set_index('entity_id').to_dict('index')
    target_lookup = target_df.set_index('entity_id').to_dict('index')

    if s1_name_freqs is None:
        s1_name_freqs = s1_df['norm_name'].value_counts().to_dict()

    feature_rows = []
    feature_names = None

    for s1_id, t_id in candidate_pairs:
        if s1_id not in s1_lookup or t_id not in target_lookup:
            continue
        s1_row = s1_lookup[s1_id]
        t_row = target_lookup[t_id]
        freq = s1_name_freqs.get(s1_row.get('norm_name', ''), 1)

        feats = extract_pairwise_features(s1_row, t_row, s1_name_freq=freq)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        feature_rows.append([feats[k] for k in feature_names])

    X = np.array(feature_rows, dtype=np.float32) if feature_rows else np.empty((0, len(feature_names or [])))
    return X, feature_names or []

