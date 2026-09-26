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

    def retrieve_candidates(self, query_norm_name: str, query_norm_addr: str, query_postal: str) -> List[Tuple[str, float]]:
        """Retrieves candidate target IDs for a single S1 entity using multi-signal keys."""
        candidate_scores = defaultdict(int)
        words = extract_brand_tokens(query_norm_name)
        prefix3 = query_norm_name[:3] if query_norm_name else ""

        # 1. Whole word matches (High Weight, capped at DF < 15,000)
        for w in set(words):
            target_ids = self.token_index.get(w, [])
            if len(target_ids) < 15000:
                for tid in target_ids:
                    candidate_scores[tid] += 5

        # Brand prefix match (First 3 characters)
        if prefix3:
            prefix_targets = self.token_index.get(f"^_{prefix3}", [])
            if len(prefix_targets) < 2500:
                for tid in prefix_targets:
                    candidate_scores[tid] += 3

        # 2. Composite Locality Match (High Selectivity Key)
        city_tok = extract_locality_token(query_norm_addr)
        loc_key = f"loc_{prefix3}_{query_postal}" if query_postal else (f"loc_{prefix3}_{city_tok}" if city_tok else None)
        if loc_key:
            loc_targets = self.locality_index.get(loc_key, [])
            if len(loc_targets) < 500:
                for tid in loc_targets:
                    candidate_scores[tid] += 6

        # 3. Character 3-gram matches (Typos / Substring overlap, for words >= 4 chars)
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
                                candidate_scores[tid] += 1

        # 4. Exact postal matches
        if query_postal:
            postal_targets = self.postal_index.get(query_postal, [])
            if len(postal_targets) < 500:
                for tid in postal_targets:
                    candidate_scores[tid] += 4

        # 5. Street name matches from address
        if query_norm_addr:
            addr_words = re.findall(r'\b[a-z]{4,}\b', query_norm_addr)
            for aw in set(addr_words[:3]):
                target_ids = self.token_index.get(f"addr_{aw}", [])
                if len(target_ids) < 1500:
                    for tid in target_ids:
                        candidate_scores[tid] += 2

        if not candidate_scores:
            return []

        # Sort by candidate score and keep top-K
        top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:self.max_candidates]
        return top_candidates



class DenseBlocker:
    """
    Semantic dense blocker using multilingual SBERT embeddings and FAISS index.
    """
    def __init__(self, model_name: str, similarity_cutoff: float = 0.60, max_candidates: int = 20):
        self.model_name = model_name
        self.similarity_cutoff = similarity_cutoff
        self.max_candidates = max_candidates
        self.model = None
        self.index = None
        self.target_ids = []

    def _init_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        self._init_model()
        emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
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
        """Performs batch vector query and returns candidates exceeding similarity cutoff."""
        if self.index is None or faiss is None:
            return {}

        q_embs = self.encode(query_texts, batch_size=batch_size)
        sims, indices = self.index.search(q_embs, self.max_candidates)

        results = {}
        for q_idx in range(len(query_texts)):
            q_res = []
            for rank in range(self.max_candidates):
                score = float(sims[q_idx, rank])
                if score < self.similarity_cutoff:
                    break
                t_idx = indices[q_idx, rank]
                if t_idx < len(self.target_ids):
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
            dense_blocker = DenseBlocker(model_name=dense_model_name, similarity_cutoff=0.62, max_candidates=20)
            dense_blocker.fit(s23_df)
        except Exception as e:
            print(f"  [Blocking Warning] Dense blocker init skipped due to: {e}")
            dense_blocker = None

    return lex_blocker, dense_blocker


def query_hybrid_blocker(
    s1_df: pd.DataFrame,
    lex_blocker: LexicalBlocker,
    dense_blocker: Optional[DenseBlocker] = None,
    max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY
) -> Dict[str, Set[str]]:
    """
    Queries fitted blockers for a batch of S1 reference entities.
    Returns: mapping {source1_entity_id: set(candidate_target_ids)}
    """
    candidates_per_s1: Dict[str, Set[str]] = defaultdict(set)

    s1_ids = s1_df['entity_id'].tolist()
    norm_names = s1_df['norm_name'].tolist()
    norm_addrs = s1_df['norm_address'].tolist()
    postals = s1_df['postal_code'].tolist()

    iterator = zip(s1_ids, norm_names, norm_addrs, postals)
    if len(s1_ids) >= 100:
        iterator = tqdm(iterator, total=len(s1_ids), desc=f"  [Blocking] Querying S1 ({len(s1_ids):,})", mininterval=0.5)

    for s1_id, norm_name, norm_addr, postal in iterator:
        hits = lex_blocker.retrieve_candidates(norm_name, norm_addr, postal)
        for tid, _ in hits:
            candidates_per_s1[s1_id].add(tid)

    if dense_blocker is not None and dense_blocker.index is not None:
        try:
            q_texts = (s1_df['norm_name'] + " " + s1_df['norm_address']).tolist()
            dense_hits = dense_blocker.retrieve_candidates(q_texts)
            for q_idx, hits in dense_hits.items():
                s1_id = s1_ids[q_idx]
                for tid, _ in hits:
                    candidates_per_s1[s1_id].add(tid)
        except Exception as e:
            print(f"  [Blocking Warning] Dense query skipped due to: {e}")

    # Enforce candidate ceiling per S1 entity
    final_candidates = {}
    for s1_id in s1_ids:
        c_set = candidates_per_s1.get(s1_id, set())
        if len(c_set) > max_candidates_per_entity:
            final_candidates[s1_id] = set(list(c_set)[:max_candidates_per_entity])
        else:
            final_candidates[s1_id] = c_set

    return final_candidates


def run_hybrid_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY,
    dense_model_name: Optional[str] = None
) -> Dict[str, Set[str]]:
    """
    Executes hybrid blocking (Sparse BM25 + Dense FAISS) for a single country partition.
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
        max_candidates_per_entity=max_candidates_per_entity
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
