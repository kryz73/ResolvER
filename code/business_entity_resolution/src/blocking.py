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

try:
    import faiss
except ImportError:
    faiss = None


class LexicalBlocker:
    """
    High-recall sparse blocker using inverted token indexes, character 3-grams,
    and exact postal/locality keys.
    """
    def __init__(self, min_token_len: int = 3, max_candidates_per_entity: int = 25):
        self.min_token_len = min_token_len
        self.max_candidates = max_candidates_per_entity
        self.token_index = defaultdict(list)
        self.postal_index = defaultdict(list)
        self.target_records = {}

    def fit(self, target_df: pd.DataFrame):
        """Builds multi-key inverted indexes on target records (Sources 2 and 3)."""
        self.token_index.clear()
        self.postal_index.clear()
        self.target_records.clear()

        eids = target_df['entity_id'].tolist()
        norm_names = target_df['norm_name'].tolist()
        norm_addrs = target_df['norm_address'].tolist()
        postals = target_df['postal_code'].tolist()

        for eid, norm_name, norm_addr, postal in zip(eids, norm_names, norm_addrs, postals):
            self.target_records[eid] = norm_name

            # 1. Whole word tokens from name
            words = re.findall(r'\b[a-z0-9]{3,}\b', norm_name)
            for w in set(words):
                self.token_index[w].append(eid)

            # 2. Character 3-grams of significant name words (captures typos/misspellings)
            for w in words:
                if len(w) >= 4:
                    for i in range(len(w) - 2):
                        self.token_index[w[i:i+3]].append(eid)

            # 3. Exact postal code index
            if postal:
                self.postal_index[postal].append(eid)

            # 4. Street name tokens from address (if available)
            addr_words = re.findall(r'\b[a-z]{4,}\b', norm_addr)
            for aw in set(addr_words[:3]):
                self.token_index[f"addr_{aw}"].append(eid)

    def retrieve_candidates(self, query_norm_name: str, query_norm_addr: str, query_postal: str) -> List[Tuple[str, float]]:
        """Retrieves candidate target IDs for a single S1 entity using multi-signal keys."""
        candidate_scores = defaultdict(int)
        words = re.findall(r'\b[a-z0-9]{3,}\b', query_norm_name)

        # 1. Whole word matches (High Weight)
        for w in set(words):
            target_ids = self.token_index.get(w, [])
            if len(target_ids) < 3000:
                for tid in target_ids:
                    candidate_scores[tid] += 5

        # 2. Character 3-gram matches (Typos / Substring overlap)
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

        # 3. Exact postal matches (Medium Weight)
        if query_postal:
            postal_targets = self.postal_index.get(query_postal, [])
            if len(postal_targets) < 500:
                for tid in postal_targets:
                    candidate_scores[tid] += 4

        # 4. Street name matches from address
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


def run_hybrid_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_candidates_per_entity: int = 35,
    dense_model_name: Optional[str] = None
) -> Dict[str, Set[str]]:
    """
    Executes hybrid blocking (Sparse BM25 + Dense FAISS) for a single country partition.
    Returns: mapping {source1_entity_id: set(candidate_target_ids)}
    """
    print(f"  [Blocking] Building sparse lexical index over {len(s23_df):,} target records...")
    lex_blocker = LexicalBlocker(max_candidates_per_entity=25)
    lex_blocker.fit(s23_df)

    candidates_per_s1: Dict[str, Set[str]] = defaultdict(set)

    # 1. Sparse Pass
    print(f"  [Blocking] Querying sparse index for {len(s1_df):,} reference entities...")
    s1_ids = s1_df['entity_id'].tolist()
    norm_names = s1_df['norm_name'].tolist()
    norm_addrs = s1_df['norm_address'].tolist()
    postals = s1_df['postal_code'].tolist()

    for s1_id, norm_name, norm_addr, postal in zip(s1_ids, norm_names, norm_addrs, postals):
        hits = lex_blocker.retrieve_candidates(norm_name, norm_addr, postal)
        for tid, _ in hits:
            candidates_per_s1[s1_id].add(tid)



    # 2. Dense Pass (if model specified and available)
    if dense_model_name and faiss is not None:
        try:
            print(f"  [Blocking] Building dense vector index with {dense_model_name}...")
            dense_blocker = DenseBlocker(model_name=dense_model_name, similarity_cutoff=0.62, max_candidates=20)
            dense_blocker.fit(s23_df)

            print(f"  [Blocking] Querying dense vector index...")
            q_texts = (s1_df['norm_name'] + " " + s1_df['norm_address']).tolist()
            dense_hits = dense_blocker.retrieve_candidates(q_texts)

            s1_ids = s1_df['entity_id'].tolist()
            for q_idx, hits in dense_hits.items():
                s1_id = s1_ids[q_idx]
                for tid, _ in hits:
                    candidates_per_s1[s1_id].add(tid)
        except Exception as e:
            print(f"  [Blocking Warning] Dense blocking skipped due to: {e}")

    # Enforce candidate ceiling per S1 entity
    final_candidates = {}
    for s1_id in s1_df['entity_id']:
        c_set = candidates_per_s1.get(s1_id, set())
        if len(c_set) > max_candidates_per_entity:
            final_candidates[s1_id] = set(list(c_set)[:max_candidates_per_entity])
        else:
            final_candidates[s1_id] = c_set

    return final_candidates


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
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in required_s1_ids:
            c_ids = candidates_dict.get(s1_id, set())
            if c_ids:
                f.write(f"{s1_id}\t{','.join(sorted(c_ids))}\n")
            else:
                f.write(f"{s1_id}\t\n")
    print(f"  [Output] Successfully wrote candidate pairs to: {output_path}")
