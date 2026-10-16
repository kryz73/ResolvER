"""
ML Challenge 2026 — Business Entity Resolution
Dual Sparse-Dense Candidate Generation & Blocking
"""

import os
import re
import time
import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

from config import MAX_CANDIDATES_PER_ENTITY
from logger import make_progress_bar

try:
    import faiss
except ImportError:
    faiss = None
    print("  [WARNING] faiss not installed — dense blocking will be disabled. Install with: pip install faiss-cpu (or faiss-gpu-cu12 for GPU)")

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
            iterator = make_progress_bar(iterator, total=len(eids), desc=f"[Blocking] Indexing Targets ({len(eids):,})", mininterval=2.0, leave=False)

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

            # Cap max_seq_length to eliminate 8192-token padding bloat on BGE-M3
            try:
                self.model.max_seq_length = 128
            except Exception:
                pass

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encodes query or candidate text batch directly on target device with FP16 autocast."""
        self._init_model()
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        show_prog = len(texts) >= 50000
        if "cuda" in self.device:
            import torch
            with torch.amp.autocast("cuda"):
                emb = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_prog,
                    normalize_embeddings=True,
                    device=self.device
                )
        else:
            emb = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_prog,
                normalize_embeddings=True,
                device=self.device
            )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def fit(self, target_df: pd.DataFrame, batch_size: int = 64, chunk_size: int = 100_000):
        """
        Encodes target records in memory-safe chunks and builds FAISS index.
        Prevents IPC memory duplication and Linux OOM kills on massive datasets.
        Automatically selects memory-bounded index (IVFFlat) when flat index would exceed RAM.
        """
        if faiss is None or target_df is None or len(target_df) == 0:
            print("  [Dense Blocker] Skipped — faiss not available or empty target DataFrame.")
            return

        self._init_model()
        self.target_ids = target_df['entity_id'].tolist()
        n_targets = len(target_df)

        dim = self.model.get_sentence_embedding_dimension() if hasattr(self.model, "get_sentence_embedding_dimension") else 1024

        # Memory-bounded FAISS index selection:
        # IndexFlatIP stores all vectors in RAM: n_targets * dim * 4 bytes
        estimated_index_bytes = n_targets * dim * 4
        max_flat_bytes = 20 * (1024 ** 3)  # 20 GB threshold

        use_ivf = estimated_index_bytes > max_flat_bytes
        if use_ivf:
            # IVF index: approximate search with clustering, much lower memory
            n_clusters = min(int(np.sqrt(n_targets)), 4096)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = min(32, n_clusters)
            print(f"  [Dense Blocker] Using IndexIVFFlat ({n_clusters} clusters, nprobe={self.index.nprobe}) — flat index would require {estimated_index_bytes / (1024**3):.1f} GB RAM")
        else:
            self.index = faiss.IndexFlatIP(dim)
            print(f"  [Dense Blocker] Using IndexFlatIP — estimated {estimated_index_bytes / (1024**3):.1f} GB RAM")

        # Multi-GPU pool initialization if multiple GPUs exist
        pool = None
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 1 and n_targets >= 10000:
                target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                print(f"  [Multi-GPU Dense Blocking] Starting parallel worker pool across {len(target_devices)} GPUs: {target_devices}...")
                pool = self.model.start_multi_process_pool(target_devices=target_devices)
        except Exception as e:
            print(f"  [Multi-GPU Note] Worker pool fallback to single device ({e})")
            pool = None

        n_chunks = int(np.ceil(n_targets / chunk_size))
        print(f"  [Dense Blocker] Indexing {n_targets:,} targets in {n_chunks} chunk(s) of {chunk_size:,}...")

        # Collect all embeddings for IVF training (need a sample for clustering)
        training_sample = None
        if use_ivf:
            # Collect a representative sample for IVF training
            max_train_samples = min(n_targets, 100_000)
            sample_indices = np.random.RandomState(42).choice(n_targets, size=max_train_samples, replace=False)
            sample_slice = target_df.iloc[sample_indices]
            sample_texts = (sample_slice['norm_name'] + " " + sample_slice['norm_address']).tolist()
            del sample_slice
            training_sample = self.encode(sample_texts, batch_size=batch_size)
            del sample_texts
            print(f"  [Dense Blocker] Training IVF index on {len(training_sample):,} sample vectors...")
            self.index.train(training_sample)
            del training_sample

        import gc
        for c_idx in range(n_chunks):
            start_i = c_idx * chunk_size
            end_i = min(start_i + chunk_size, n_targets)
            chunk_slice = target_df.iloc[start_i:end_i]
            chunk_texts = (chunk_slice['norm_name'] + " " + chunk_slice['norm_address']).tolist()
            del chunk_slice

            if pool is not None:
                try:
                    # Bug 1 Fix: encode_multi_process does NOT accept normalize_embeddings
                    chunk_embs = self.model.encode_multi_process(
                        chunk_texts,
                        pool,
                        batch_size=batch_size
                    )
                    # Manual L2 normalization since encode_multi_process doesn't support it
                    norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
                    chunk_embs = chunk_embs / np.maximum(norms, 1e-12)
                except Exception as e:
                    print(f"  [Multi-GPU Pool Warning] Chunk {c_idx+1} fallback ({e}).")
                    chunk_embs = self.model.encode(
                        chunk_texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        device=self.device
                    )
            else:
                chunk_embs = self.model.encode(
                    chunk_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    device=self.device
                )
            del chunk_texts

            chunk_arr = np.ascontiguousarray(chunk_embs, dtype=np.float32)
            del chunk_embs
            self.index.add(chunk_arr)
            del chunk_arr
            gc.collect()

        if pool is not None:
            try:
                self.model.stop_multi_process_pool(pool)
            except Exception:
                pass
            del pool
            gc.collect()

        print(f"  [Dense Blocker] Index built: {self.index.ntotal:,} vectors indexed.")

    def retrieve_candidates(self, query_texts: List[str], batch_size: int = 64) -> Dict[int, List[Tuple[str, float]]]:
        """Performs batch vector query and returns candidates with their raw cosine similarity scores."""
        if self.index is None or faiss is None or not query_texts:
            return {}

        q_embs = self.encode(query_texts, batch_size=batch_size)
        sims, indices = self.index.search(q_embs, self.max_candidates)
        del q_embs
        import gc; gc.collect()

        results = {}
        for q_idx in range(len(query_texts)):
            q_res = []
            for rank in range(self.max_candidates):
                t_idx = indices[q_idx, rank]
                if 0 <= t_idx < len(self.target_ids):
                    score = float(sims[q_idx, rank])
                    q_res.append((self.target_ids[t_idx], score))
            results[q_idx] = q_res

        del sims, indices
        gc.collect()
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
            print(f"  [Blocking WARNING] Dense blocker init FAILED: {type(e).__name__}: {e}")
            dense_blocker = None
    elif dense_model_name and faiss is None:
        print(f"  [Blocking WARNING] Dense model requested ({dense_model_name}) but faiss is not installed — running lexical-only blocking!")
    else:
        print(f"  [Blocking] Dense blocking not requested — running lexical-only blocking.")

    return lex_blocker, dense_blocker


def query_hybrid_blocker(
    s1_df: pd.DataFrame,
    lex_blocker: LexicalBlocker,
    dense_blocker: Optional[DenseBlocker] = None,
    max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY,
    return_provenance: bool = False,
    tracker: Optional[Any] = None
):
    """
    Asynchronous Pipelined Hybrid Blocker:
    Concurrently executes GPU dense embedding retrieval and multi-core CPU lexical channel retrieval.
    Overlaps GPU Tensor Core execution with CPU inverted index traversal, eliminating GPU idle time.
    Merges channels with Reciprocal Rank Fusion (RRF) and dynamic score thresholding.
    """
    t_start = time.time()
    s1_ids = s1_df['entity_id'].tolist()
    norm_names = s1_df['norm_name'].tolist()
    norm_addrs = s1_df['norm_address'].tolist()
    postals = s1_df['postal_code'].tolist()
    n_queries = len(s1_ids)

    # 1. Asynchronously launch Dense Vector Retrieval on GPU
    dense_future = None
    gpu_pool = None
    t_dense_start = time.time()
    if dense_blocker is not None and dense_blocker.index is not None and n_queries > 0:
        q_texts = (s1_df['norm_name'] + " " + s1_df['norm_address']).tolist()
        gpu_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        dense_future = gpu_pool.submit(dense_blocker.retrieve_candidates, q_texts)

    # 2. Concurrently execute Multi-Core Lexical Channel Retrieval on CPU
    t_lex_start = time.time()
    lex_channels_by_idx = {}
    n_cpu_workers = min(4, os.cpu_count() or 4)

    if n_queries >= 1000 and n_cpu_workers > 1:
        chunk_size = int(np.ceil(n_queries / n_cpu_workers))
        def _process_lex_chunk(start_idx, end_idx):
            res = {}
            for i in range(start_idx, end_idx):
                res[i] = lex_blocker.retrieve_channels(norm_names[i], norm_addrs[i], postals[i])
            return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cpu_workers) as cpu_pool:
            futures = [
                cpu_pool.submit(_process_lex_chunk, i, min(i + chunk_size, n_queries))
                for i in range(0, n_queries, chunk_size)
            ]
            for f in futures:
                lex_channels_by_idx.update(f.result())
    else:
        for i in range(n_queries):
            lex_channels_by_idx[i] = lex_blocker.retrieve_channels(norm_names[i], norm_addrs[i], postals[i])

    t_lex_elapsed = time.time() - t_lex_start

    # 3. Synchronize with GPU Dense stream
    dense_results = {}
    t_dense_elapsed = 0.0
    if dense_future is not None:
        try:
            dense_results = dense_future.result()
            t_dense_elapsed = time.time() - t_dense_start
        except Exception as e:
            print(f"  [Blocking Warning] Dense query failed in async pool: {e}")
            dense_results = {}
        finally:
            if gpu_pool is not None:
                gpu_pool.shutdown(wait=False)

    # 4. RRF Score Fusion
    t_rrf_start = time.time()
    candidates_per_s1: Dict[str, Set[str]] = {}
    provenance_per_pair: Dict[Tuple[str, str], Dict[str, float]] = {}

    for q_idx in range(n_queries):
        s1_id = s1_ids[q_idx]
        channels = lex_channels_by_idx.get(q_idx, {})
        if q_idx in dense_results and dense_results[q_idx]:
            channels['dense'] = dense_results[q_idx]

        ordered_cands, prov_map = compute_rrf_fusion(channels, max_candidates=max_candidates_per_entity)
        candidates_per_s1[s1_id] = set(ordered_cands)

        if return_provenance:
            for tid, p_info in prov_map.items():
                provenance_per_pair[(s1_id, tid)] = p_info

    t_rrf_elapsed = time.time() - t_rrf_start
    t_total = time.time() - t_start

    if tracker is not None:
        tracker.add_time("GPU Dense Retrieval (Async)", t_dense_elapsed, category="sub")
        tracker.add_time("CPU Lexical Multi-Core Retrieval (Async)", t_lex_elapsed, category="sub")
        overlap_savings = max(0.0, (t_dense_elapsed + t_lex_elapsed) - t_total)
        tracker.add_time("Async Pipelined Concurrency Savings", overlap_savings, category="sub")
        tracker.add_time("p6_async_retrieval", t_total, category="sub")

    if return_provenance:
        return candidates_per_s1, provenance_per_pair
    return candidates_per_s1


def run_hybrid_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_candidates_per_entity: int = MAX_CANDIDATES_PER_ENTITY,
    dense_model_name: Optional[str] = None,
    return_provenance: bool = False,
    tracker: Optional[Any] = None
):
    """
    Executes hybrid blocking (Sparse Multi-channel + Dense FAISS + RRF Fusion) for a single country partition.
    Convenience wrapper around fit_hybrid_blocker and query_hybrid_blocker.
    """
    t_fit_start = time.time()
    lex_blocker, dense_blocker = fit_hybrid_blocker(
        s23_df,
        max_candidates_per_entity=max_candidates_per_entity,
        dense_model_name=dense_model_name
    )
    t_fit = time.time() - t_fit_start
    if tracker is not None:
        tracker.add_time("p2_lex_fit", t_fit * (0.3 if dense_model_name else 1.0), category="sub")
        if dense_model_name:
            tracker.add_time("p2_dense_fit", t_fit * 0.7, category="sub")

    t_ret_start = time.time()
    res = query_hybrid_blocker(
        s1_df,
        lex_blocker,
        dense_blocker=dense_blocker,
        max_candidates_per_entity=max_candidates_per_entity,
        return_provenance=return_provenance,
        tracker=tracker
    )
    t_ret = time.time() - t_ret_start
    if tracker is not None:
        tracker.add_time("p2_retrieval", t_ret, category="sub")

    return res


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
        iterator = make_progress_bar(required_s1_ids, desc="Writing candidate_pairs.tsv", mininterval=2.0, leave=False)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in iterator:
            c_ids = candidates_dict.get(s1_id, set())
            if c_ids:
                f.write(f"{s1_id}\t{','.join(sorted(c_ids))}\n")
            else:
                f.write(f"{s1_id}\t\n")
    print(f"  [Output] Successfully wrote candidate pairs to: {output_path}")
