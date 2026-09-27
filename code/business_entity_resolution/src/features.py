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
    provenance: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Computes a 30+ signal feature dictionary for a candidate pair (S1, Target),
    incorporating string metrics, token overlap, structural numbers, core brand,
    acronym match, and candidate retrieval provenance (RRF score, channel ranks, dense similarity).
    """
    feats = {}

    n1 = str(s1_row.get('norm_name', '')).strip()
    n2 = str(target_row.get('norm_name', '')).strip()
    c1 = str(s1_row.get('clean_name', '')).strip()
    c2 = str(target_row.get('clean_name', '')).strip()
    b1 = str(s1_row.get('core_brand', '')).strip()
    b2 = str(target_row.get('core_brand', '')).strip()
    a1 = str(s1_row.get('norm_address', '')).strip()
    a2 = str(target_row.get('norm_address', '')).strip()

    target_id = str(target_row.get('entity_id', ''))

    # ── 1. Name String Distances ──────────────────────────────────────────
    feats['name_ratio'] = fuzz.ratio(n1, n2) / 100.0
    feats['name_token_sort'] = fuzz.token_sort_ratio(n1, n2) / 100.0
    feats['name_token_set'] = fuzz.token_set_ratio(n1, n2) / 100.0
    feats['name_partial'] = fuzz.partial_ratio(n1, n2) / 100.0
    feats['clean_name_ratio'] = fuzz.ratio(c1, c2) / 100.0

    # Core Brand identity match (legal suffixes stripped)
    if b1 and b2:
        feats['core_brand_ratio'] = fuzz.ratio(b1, b2) / 100.0
        feats['core_brand_exact'] = 1.0 if b1 == b2 else 0.0
    else:
        feats['core_brand_ratio'] = feats['name_ratio']
        feats['core_brand_exact'] = 0.0

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

    # Brand Acronym / Abbreviation Match (e.g. IBM vs International Business Machines)
    words1 = [w for w in n1.split() if len(w) > 0]
    words2 = [w for w in n2.split() if len(w) > 0]
    acronym1 = "".join([w[0] for w in words1 if w[0].isalnum()])
    acronym2 = "".join([w[0] for w in words2 if w[0].isalnum()])
    feats['acronym_match'] = 1.0 if (acronym1 and acronym2 and (acronym1 == n2 or acronym2 == n1 or acronym1 == acronym2)) else 0.0

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

    # ── 3. Structural & Number Matches (Clean tri-state indicators) ──────
    nums1 = set(re.findall(r'\b\d+\b', a1))
    nums2 = set(re.findall(r'\b\d+\b', a2))
    nums_missing = bool(not nums1 or not nums2)
    feats['number_missing'] = 1.0 if nums_missing else 0.0
    if not nums_missing:
        feats['number_overlap_jaccard'] = jaccard_similarity(nums1, nums2)
        feats['has_number_exact_match'] = 1.0 if len(nums1 & nums2) > 0 else 0.0
        feats['has_number_conflict'] = 1.0 if len(nums1 & nums2) == 0 else 0.0
    else:
        feats['number_overlap_jaccard'] = 0.0
        feats['has_number_exact_match'] = 0.0
        feats['has_number_conflict'] = 0.0

    p1 = str(s1_row.get('postal_code', '')).strip()
    p2 = str(target_row.get('postal_code', '')).strip()
    postal_missing = bool(not p1 or not p2)
    feats['postal_missing'] = 1.0 if postal_missing else 0.0
    if not postal_missing:
        feats['postal_exact_match'] = 1.0 if p1 == p2 else 0.0
        feats['postal_conflict'] = 1.0 if p1 != p2 else 0.0
    else:
        feats['postal_exact_match'] = 0.0
        feats['postal_conflict'] = 0.0

    # ── 4. Brand Initial & Locality Contradictions (Explicit Vetoes) ─────────
    b1_initial = b1[0] if b1 else ""
    b2_initial = b2[0] if b2 else ""
    feats['brand_initial_conflict'] = 1.0 if (b1_initial and b2_initial and b1_initial != b2_initial) else 0.0

    loc_words_1 = re.findall(r'\b[a-z]{3,}\b', a1)
    loc_words_2 = re.findall(r'\b[a-z]{3,}\b', a2)
    loc1 = loc_words_1[-1] if loc_words_1 else ""
    loc2 = loc_words_2[-1] if loc_words_2 else ""
    loc_missing = bool(not loc1 or not loc2)
    feats['city_conflict'] = 1.0 if (not loc_missing and loc1 != loc2) else 0.0

    # Composite Veto: high name match but geographic contradiction
    geo_conflict = bool(feats['city_conflict'] == 1.0 or feats['has_number_conflict'] == 1.0 or feats['postal_conflict'] == 1.0)
    feats['high_name_x_geo_conflict'] = 1.0 if (feats['name_ratio'] >= 0.85 and geo_conflict) else 0.0

    # ── 5. Franchise & Information-Theoretic Surprisal ─────────────────────
    feats['name_log_freq'] = math.log1p(s1_name_freq)
    feats['name_surprisal'] = math.log2(2_000_000.0 / (float(s1_name_freq) + 1.0))
    feats['freq_x_addr_sort'] = feats['addr_token_sort'] / (1.0 + feats['name_log_freq'])
    feats['name_x_addr_ratio'] = feats['name_ratio'] * feats['addr_ratio']

    # ── 6. Source Indicators ──────────────────────────────────────────────
    feats['is_source_2'] = 1.0 if target_id.startswith('S2-') else 0.0
    feats['is_source_3'] = 1.0 if target_id.startswith('S3-') else 0.0

    # ── 6. Candidate Retrieval Provenance & Semantic Features ────────────
    if provenance:
        feats['rrf_score'] = float(provenance.get('rrf_score', 0.0))
        feats['channel_count'] = float(provenance.get('channel_count', 1.0))
        feats['rank_word'] = float(provenance.get('rank_word', 999.0))
        feats['rank_locality'] = float(provenance.get('rank_locality', 999.0))
        feats['rank_postal'] = float(provenance.get('rank_postal', 999.0))
        feats['rank_char3'] = float(provenance.get('rank_char3', 999.0))
        feats['rank_dense'] = float(provenance.get('rank_dense', 999.0))
        feats['dense_cosine_sim'] = float(provenance.get('dense_cosine_sim', 0.0))
        feats['has_dense_sim'] = float(provenance.get('has_dense_sim', 0.0))
    else:
        feats['rrf_score'] = 0.0
        feats['channel_count'] = 1.0
        feats['rank_word'] = 999.0
        feats['rank_locality'] = 999.0
        feats['rank_postal'] = 999.0
        feats['rank_char3'] = 999.0
        feats['rank_dense'] = 999.0
        feats['dense_cosine_sim'] = 0.0
        feats['has_dense_sim'] = 0.0

    return feats


from logger import make_progress_bar


def build_feature_matrix(
    candidate_pairs: List[Tuple[str, str]],
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    s1_name_freqs: Optional[Dict[str, int]] = None,
    provenance_map: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Constructs a 2D feature matrix (N_pairs x N_features) from candidate ID pairs
    using high-speed in-memory dictionary lookups and candidate provenance.
    Returns: (X, feature_names)
    """
    s1_lookup = s1_df.set_index('entity_id').to_dict('index')
    target_lookup = target_df.set_index('entity_id').to_dict('index')

    if s1_name_freqs is None:
        s1_name_freqs = s1_df['norm_name'].value_counts().to_dict()

    feature_rows = []
    feature_names = None

    pair_iterator = candidate_pairs
    if len(candidate_pairs) >= 1000:
        pair_iterator = make_progress_bar(candidate_pairs, desc=f"[Features] Extracting {len(candidate_pairs):,} Pairs", mininterval=2.0, leave=False)

    for s1_id, t_id in pair_iterator:
        if s1_id not in s1_lookup or t_id not in target_lookup:
            continue
        s1_row = s1_lookup[s1_id]
        t_row = target_lookup[t_id]
        freq = s1_name_freqs.get(s1_row.get('norm_name', ''), 1)

        prov = provenance_map.get((s1_id, t_id)) if provenance_map else None
        feats = extract_pairwise_features(s1_row, t_row, s1_name_freq=freq, provenance=prov)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        feature_rows.append([feats[k] for k in feature_names])

    X = np.array(feature_rows, dtype=np.float32) if feature_rows else np.empty((0, len(feature_names or [])))
    return X, feature_names or []


def extract_features_and_score_chunked(
    candidate_pairs: List[Tuple[str, str]],
    s1_lookup: Dict[str, Dict],
    target_lookup: Dict[str, Dict],
    s1_name_freqs: Dict[str, int],
    matcher,
    chunk_size: int = 250_000,
    provenance_map: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None
) -> List[Tuple[str, str, float]]:
    """
    Memory-efficient feature extraction + scoring pipeline.
    Processes pairs in fixed-size chunks, scoring each chunk immediately
    and discarding the feature matrix. Never materializes the full N×features array.

    Returns: list of (s1_id, target_id, prob) for all valid pairs.
    """
    results = []
    total = len(candidate_pairs)
    feature_names = None

    for start in make_progress_bar(range(0, total, chunk_size),
                                   desc=f"[Features+Score] Chunked ({total:,} pairs, {chunk_size//1000}K/chunk)",
                                   mininterval=2.0, leave=False):
        end = min(start + chunk_size, total)
        chunk_pairs = candidate_pairs[start:end]

        chunk_rows = []
        valid_pairs = []
        for s1_id, t_id in chunk_pairs:
            if s1_id not in s1_lookup or t_id not in target_lookup:
                continue
            s1_row = s1_lookup[s1_id]
            t_row = target_lookup[t_id]
            freq = s1_name_freqs.get(s1_row.get('norm_name', ''), 1)

            prov = provenance_map.get((s1_id, t_id)) if provenance_map else None
            feats = extract_pairwise_features(s1_row, t_row, s1_name_freq=freq, provenance=prov)
            if feature_names is None:
                feature_names = sorted(feats.keys())
            chunk_rows.append([feats[k] for k in feature_names])
            valid_pairs.append((s1_id, t_id))

        if not chunk_rows:
            continue

        X_chunk = np.array(chunk_rows, dtype=np.float32)
        probs = matcher.predict_proba(X_chunk)

        for (s1_id, t_id), prob in zip(valid_pairs, probs):
            results.append((s1_id, t_id, float(prob)))

        # Explicitly free chunk arrays
        del X_chunk, chunk_rows, valid_pairs

    return results

