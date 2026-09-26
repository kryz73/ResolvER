"""
ML Challenge 2026 — Business Entity Resolution
Dual Sparse-Dense Candidate Generation & Blocking
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from tqdm import tqdm

from config import MAX_CANDIDATES_PER_ENTITY

try:
    import faiss
except ImportError:
    faiss = None

# Guarded 2-letter tokens exclusion set (common prepositions, units, indicators)
STOP_2LETTER = {
    'co', 'in', 'to', 'at', 'no', 'rd', 'st', 'dr', 'ln', 'av', 'of', 'on', 'by',
    'is', 'it', 'as', 'or', 'an', 'am', 'pm', 'us', 'uk', 'll', 'al', 'de', 'la',
    'le', 'du'
}


def extract_brand_tokens(name: str) -> List[str]:
    """
    Extracts significant brand tokens:
    - Words with len >= 3
    - 2-letter tokens (e.g. 3M, GE, HP, Om, AI, BP) excluding stop tokens in STOP_2LETTER.
    """
    raw_tokens = re.findall(r'\b[a-z0-9]{2,}\b', name)
    return [t for t in raw_tokens if len(t) >= 3 or t not in STOP_2LETTER]


def extract_locality_token(norm_addr: str) -> str:
    """Extracts trailing locality / city word token (len >= 3, alphabetic)."""
    if not norm_addr:
        return ""
    words = re.findall(r'\b[a-z]{3,}\b', norm_addr)
    return words[-1] if words else ""


class LexicalBlocker:
    """
    High-recall sparse blocker using inverted token indexes, character 3-grams,
    guarded 2-letter brands, and composite locality/postal keys.
    """
    def __init__(self, min_token_len: int = 3, max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY):
        self.min_token_len = min_token_len
        self.max_candidates = max_candidates_per_entity
        self.token_index = defaultdict(list)
        self.postal_index = defaultdict(list)
        self.locality_index = defaultdict(list)

    def fit(self, target_df: pd.DataFrame):
        """Builds multi-tier inverted indexes on target records (Sources 2 and 3)."""
        self.token_index.clear()
        self.postal_index.clear()
        self.locality_index.clear()

        eids = target_df['entity_id'].tolist()
        norm_names = target_df['norm_name'].tolist()
        norm_addrs = target_df['norm_address'].tolist()
        postals = target_df['postal_code'].tolist()

        iterator = zip(eids, norm_names, norm_addrs, postals)
        if len(eids) >= 1000:
            iterator = tqdm(iterator, total=len(eids), desc=f"  [Blocking] Indexing Targets ({len(eids):,})", mininterval=0.5)

        for eid, norm_name, norm_addr, postal in iterator:

            # 1. Significant brand tokens (including guarded 2-letter brands)
            words = extract_brand_tokens(norm_name)
            for w in set(words):
                self.token_index[w].append(eid)

            # Prefix token of the brand name (first 3 chars)
            prefix3 = norm_name[:3] if norm_name else ""
            if prefix3:
                self.token_index[f"^_{prefix3}"].append(eid)

            # 2. Character 3-grams for words >= 4 characters (captures typos/misspellings)
            for w in words:
                if len(w) >= 4:
                    for i in range(len(w) - 2):
                        self.token_index[w[i:i+3]].append(eid)

            # 3. Exact postal code index
            if postal:
                self.postal_index[postal].append(eid)

            # 4. Composite locality keys (robust against >93% missing postal codes)
            city_tok = extract_locality_token(norm_addr)
            if prefix3:
                if postal:
                    self.locality_index[f"loc_{prefix3}_{postal}"].append(eid)
                elif city_tok:
                    self.locality_index[f"loc_{prefix3}_{city_tok}"].append(eid)

            # 5. Street name tokens from address (if available)
            addr_words = re.findall(r'\b[a-z]{4,}\b', norm_addr)
            for aw in set(addr_words[:3]):
                self.token_index[f"addr_{aw}"].append(eid)

    def retrieve_channels(self, query_norm_name: str, query_norm_addr: str, query_postal: str) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieves candidate target IDs partitioned by retrieval channel:
        - 'word': whole brand words and 3-char prefix matches
        - 'locality': composite locality key matches
        - 'postal': exact postal code matches
        - 'char3': character 3-gram matches
        - 'street': street address token matches
        Returns: {channel_name: [(target_id, channel_score), ...]}
        """
        channel_hits = {
            'word': defaultdict(int),
            'locality': [],
            'postal': [],
            'char3': defaultdict(int),
            'street': defaultdict(int),
        }
        words = extract_brand_tokens(query_norm_name)
        prefix3 = query_norm_name[:3] if query_norm_name else ""

        # 1. Whole word matches & Prefix matches
        for w in set(words):
            target_ids = self.token_index.get(w, [])
            if len(target_ids) < 15000:
                for tid in target_ids:
                    channel_hits['word'][tid] += 5

        if prefix3:
            prefix_targets = self.token_index.get(f"^_{prefix3}", [])
            if len(prefix_targets) < 2500:
                for tid in prefix_targets:
                    channel_hits['word'][tid] += 3

        # 2. Composite Locality Match (High Selectivity Key)
        city_tok = extract_locality_token(query_norm_addr)
        loc_key = f"loc_{prefix3}_{query_postal}" if query_postal else (f"loc_{prefix3}_{city_tok}" if city_tok else None)
        if loc_key:
            loc_targets = self.locality_index.get(loc_key, [])
            if len(loc_targets) < 500:
                channel_hits['locality'] = [(tid, 6.0) for tid in loc_targets]

        # 3. Character 3-gram matches (for words >= 4 chars)
        tri_seen = set()
        for w in words:
            if len(w) >= 4:
                for i in range(len(w) - 2):
                    tri = w[i:i+3]
                    if tri not in tri_seen:
                        tri_seen.add(tri)
                        target_ids = self.token_index.get(tri, [])
                        if len(target_ids) < 2000:
                            for tid in target_ids:
                                channel_hits['char3'][tid] += 1

        # 4. Exact postal matches
        if query_postal:
            postal_targets = self.postal_index.get(query_postal, [])
            if len(postal_targets) < 500:
                channel_hits['postal'] = [(tid, 4.0) for tid in postal_targets]

        # 5. Street name matches from address
        if query_norm_addr:
            addr_words = re.findall(r'\b[a-z]{4,}\b', query_norm_addr)
            for aw in set(addr_words[:3]):
                target_ids = self.token_index.get(f"addr_{aw}", [])
                if len(target_ids) < 1500:
                    for tid in target_ids:
                        channel_hits['street'][tid] += 2

        # Convert dict accumulators to sorted ranking lists
        results = {}
        if channel_hits['word']:
            results['word'] = sorted(channel_hits['word'].items(), key=lambda x: x[1], reverse=True)[:50]
        if channel_hits['locality']:
            results['locality'] = channel_hits['locality'][:50]
        if channel_hits['postal']:
            results['postal'] = channel_hits['postal'][:50]
        if channel_hits['char3']:
            results['char3'] = sorted(channel_hits['char3'].items(), key=lambda x: x[1], reverse=True)[:50]
        if channel_hits['street']:
            results['street'] = sorted(channel_hits['street'].items(), key=lambda x: x[1], reverse=True)[:50]

        return results

    def retrieve_candidates(self, query_norm_name: str, query_norm_addr: str, query_postal: str) -> List[Tuple[str, float]]:
        """Convenience method returning ranked candidates combining sparse channels."""
        channels = self.retrieve_channels(query_norm_name, query_norm_addr, query_postal)
        ranked_ids, _ = compute_rrf_fusion(channels, max_candidates=self.max_candidates)
        return [(tid, 1.0) for tid in ranked_ids]


def compute_rrf_fusion(
    channel_rankings: Dict[str, List[Tuple[str, float]]],
    k: int = 20,
    weights: Optional[Dict[str, float]] = None,
    max_candidates: int = MAX_CANDIDATES_PER_ENTITY,
    gamma: float = 0.10,
    min_candidates: int = 35
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Combines multi-channel ranked candidate lists using Reciprocal Rank Fusion (RRF):
      RRF(d) = sum_{c} w_c / (k + rank_c(d))

    Applies Adaptive Dynamic-K Cutoff (Relative Score Pruning):
      Keep candidate d iff RRF(d) >= gamma * RRF(top1), bounded by [min_candidates, max_candidates].
    """
    default_weights = {
        'dense': 1.5,
        'word': 1.2,
        'locality': 1.0,
        'postal': 1.0,
        'char3': 0.8,
        'street': 0.7,
    }
    w_map = weights or default_weights

    rrf_scores = defaultdict(float)
    channel_membership = defaultdict(set)
    channel_ranks = defaultdict(dict)
    dense_sims = {}

    for ch_name, candidates in channel_rankings.items():
        w = w_map.get(ch_name, 1.0)
        for rank, (tid, score) in enumerate(candidates, start=1):
            rrf_scores[tid] += w / (float(k) + float(rank))
            channel_membership[tid].add(ch_name)
            channel_ranks[tid][ch_name] = float(rank)
            if ch_name == 'dense':
                dense_sims[tid] = float(score)

    if not rrf_scores:
        return [], {}

    # Adaptive Dynamic-K Cutoff: Relative Score Pruning
    sorted_all = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_all:
        top_score = sorted_all[0][1]
        threshold_score = gamma * top_score
        pruned = [item for item in sorted_all if item[1] >= threshold_score]
        if len(pruned) < min_candidates:
            pruned = sorted_all[:min_candidates]
        elif len(pruned) > max_candidates:
            pruned = pruned[:max_candidates]
        sorted_items = pruned
    else:
        sorted_items = []
    ordered_ids = [tid for tid, _ in sorted_items]

    provenance = {}
    for tid in ordered_ids:
        r_dict = channel_ranks.get(tid, {})
        provenance[tid] = {
            'rrf_score': float(rrf_scores[tid]),
            'channel_count': float(len(channel_membership.get(tid, set()))),
            'rank_word': float(r_dict.get('word', 999.0)),
            'rank_locality': float(r_dict.get('locality', 999.0)),
            'rank_postal': float(r_dict.get('postal', 999.0)),
            'rank_char3': float(r_dict.get('char3', 999.0)),
            'rank_dense': float(r_dict.get('dense', 999.0)),
            'dense_cosine_sim': float(dense_sims.get(tid, 0.0)),
            'has_dense_sim': 1.0 if tid in dense_sims else 0.0,
        }

    return ordered_ids, provenance



class DenseBlocker:
    """
    Semantic dense blocker using multilingual embeddings (BGE-M3 or SBERT) and FAISS index.
    Accelerated with CUDA and FP16 when GPU is present (e.g. Kaggle T4 GPU).
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", max_candidates: int = 30):
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.model = None
        self.index = None
        self.target_ids = []
        try:
            import torch
            self.device = "cuda" if (torch.cuda.is_available() and os.environ.get("USE_GPU", "1") == "1") else "cpu"
        except Exception:
            self.device = "cpu"

    def _init_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            except Exception:
                # Fallback to MiniLM if BGE-M3 is unavailable
                fallback_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self.model = SentenceTransformer(fallback_name, device=self.device)
            if self.device == "cuda":
                try:
                    self.model = self.model.half()
                except Exception:
                    pass

    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        self._init_model()
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Multi-GPU parallelization if multiple GPUs exist (e.g. Kaggle dual T4)
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 1 and len(texts) >= 5000:
                target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                print(f"  [Multi-GPU Dense Blocking] Parallelizing encoding across {len(target_devices)} GPUs: {target_devices}...")
                pool = self.model.start_multi_process_pool(target_devices=target_devices)
                emb = self.model.encode_multi_process(
                    texts,
                    pool,
                    batch_size=batch_size,
                    normalize_embeddings=True
                )
                self.model.stop_multi_process_pool(pool)
                return np.ascontiguousarray(emb, dtype=np.float32)
        except Exception as e:
            print(f"  [Multi-GPU Note] Multi-process pool fallback to single device ({e})")

        show_prog = len(texts) >= 50000
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_prog,
            normalize_embeddings=True,
            device=self.device
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def fit(self, target_df: pd.DataFrame, batch_size: int = 256):
        """Encodes target records and builds FAISS index."""
        if faiss is None:
            return

        texts = (target_df['norm_name'] + " " + target_df['norm_address']).tolist()
        self.target_ids = target_df['entity_id'].tolist()

        embeddings = self.encode(texts, batch_size=batch_size)
        dim = embeddings.shape[1]

        # Use IndexFlatIP for normalized cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve_candidates(self, query_texts: List[str], batch_size: int = 256) -> Dict[int, List[Tuple[str, float]]]:
        """Performs batch vector query and returns candidates with their raw cosine similarity scores."""
        if self.index is None or faiss is None:
            return {}

        q_embs = self.encode(query_texts, batch_size=batch_size)
        sims, indices = self.index.search(q_embs, self.max_candidates)

        results = {}
        for q_idx in range(len(query_texts)):
            q_res = []
            for rank in range(self.max_candidates):
                t_idx = indices[q_idx, rank]
                if 0 <= t_idx < len(self.target_ids):
                    score = float(sims[q_idx, rank])
                    q_res.append((self.target_ids[t_idx], score))
            results[q_idx] = q_res

        return results


def fit_hybrid_blocker(
    s23_df: pd.DataFrame,
    max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY,
    dense_model_name: Optional[str] = None
) -> Tuple[LexicalBlocker, Optional[DenseBlocker]]:
    """
    Fits sparse and optional dense blockers on target records (Sources 2 and 3).
    Returns: (lex_blocker, dense_blocker)
    """
    print(f"  [Blocking] Building sparse lexical index over {len(s23_df):,} target records...")
    lex_blocker = LexicalBlocker(max_candidates_per_entity=max_candidates_per_entity)
    lex_blocker.fit(s23_df)

    dense_blocker = None
    if dense_model_name and faiss is not None:
        try:
            print(f"  [Blocking] Building dense vector index with {dense_model_name}...")
            dense_blocker = DenseBlocker(model_name=dense_model_name, max_candidates=25)
            dense_blocker.fit(s23_df)
        except Exception as e:
            print(f"  [Blocking Warning] Dense blocker init skipped due to: {e}")
            dense_blocker = None

    return lex_blocker, dense_blocker


def query_hybrid_blocker(
    s1_df: pd.DataFrame,
    lex_blocker: LexicalBlocker,
    dense_blocker: Optional[DenseBlocker] = None,
    max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY,
    return_provenance: bool = False
):
    """
    Queries fitted blockers for a batch of S1 reference entities using Reciprocal Rank Fusion (RRF).
    Eliminates arbitrary set hash truncation by ranking across all sparse and dense channels.

    Returns:
      - If return_provenance is False: mapping {source1_entity_id: set(candidate_target_ids)}
      - If return_provenance is True: (candidates_dict, provenance_dict)
        where provenance_dict is mapping {(s1_id, tid): provenance_info}
    """
    s1_ids = s1_df['entity_id'].tolist()
    norm_names = s1_df['norm_name'].tolist()
    norm_addrs = s1_df['norm_address'].tolist()
    postals = s1_df['postal_code'].tolist()

    # Pre-retrieve dense candidates in batch if dense blocker is present
    dense_results = {}
    if dense_blocker is not None and dense_blocker.index is not None:
        try:
            q_texts = (s1_df['norm_name'] + " " + s1_df['norm_address']).tolist()
            dense_results = dense_blocker.retrieve_candidates(q_texts)
        except Exception as e:
            print(f"  [Blocking Warning] Dense query skipped due to: {e}")
            dense_results = {}

    candidates_per_s1: Dict[str, Set[str]] = {}
    provenance_per_pair: Dict[Tuple[str, str], Dict[str, float]] = {}

    iterator = enumerate(zip(s1_ids, norm_names, norm_addrs, postals))
    if len(s1_ids) >= 100:
        iterator = tqdm(iterator, total=len(s1_ids), desc=f"  [Blocking] RRF Query S1 ({len(s1_ids):,})", mininterval=0.5)

    for q_idx, (s1_id, norm_name, norm_addr, postal) in iterator:
        channels = lex_blocker.retrieve_channels(norm_name, norm_addr, postal)

        if q_idx in dense_results and dense_results[q_idx]:
            channels['dense'] = dense_results[q_idx]

        ordered_cands, prov_map = compute_rrf_fusion(channels, max_candidates=max_candidates_per_entity)
        candidates_per_s1[s1_id] = set(ordered_cands)

        if return_provenance:
            for tid, p_info in prov_map.items():
                provenance_per_pair[(s1_id, tid)] = p_info

    if return_provenance:
        return candidates_per_s1, provenance_per_pair
    return candidates_per_s1


def run_hybrid_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY,
    dense_model_name: Optional[str] = None,
    return_provenance: bool = False
):
    """
    Executes hybrid blocking (Sparse Multi-channel + Dense FAISS + RRF Fusion) for a single country partition.
    Convenience wrapper around fit_hybrid_blocker and query_hybrid_blocker.
    """
    lex_blocker, dense_blocker = fit_hybrid_blocker(
        s23_df,
        max_candidates_per_entity=max_candidates_per_entity,
        dense_model_name=dense_model_name
    )
    return query_hybrid_blocker(
        s1_df,
        lex_blocker,
        dense_blocker=dense_blocker,
        max_candidates_per_entity=max_candidates_per_entity,
        return_provenance=return_provenance
    )


def write_candidate_pairs_tsv(
    candidates_dict: Dict[str, Set[str]],
    required_s1_ids: List[str],
    output_path: str
):
    """
    Writes candidate pairs to candidate_pairs.tsv in the official competition format.
    Format: source1_entity_id \t candidate_entity_ids (comma-separated)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    iterator = required_s1_ids
    if len(required_s1_ids) >= 10000:
        iterator = tqdm(required_s1_ids, desc="  [Output] Writing candidate_pairs.tsv", mininterval=1.0)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in iterator:
            c_ids = candidates_dict.get(s1_id, set())
            if c_ids:
                f.write(f"{s1_id}\t{','.join(sorted(c_ids))}\n")
            else:
                f.write(f"{s1_id}\t\n")
    print(f"  [Output] Successfully wrote candidate pairs to: {output_path}")
