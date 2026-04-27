"""
Query Engine — Orchestrates the full retrieval + re-ranking + answer pipeline.

Supports:
- LSH-based retrieval (MinHash + SimHash hybrid)
- TF-IDF baseline (exact)
- Recommendation re-ranking (extension)
- Side-by-side comparison mode
"""
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from config import CHUNKS_FILE, INDEX_DIR, TOP_K
from minhash_lsh import (
    text_to_shingles,
    compute_minhash_signature,
    MinHashLSHIndex,
    build_minhash_lsh_index,
    jaccard_from_signatures,
)
from simhash import (
    compute_simhash,
    SimHashIndex,
    build_simhash_index,
    hamming_similarity,
)
from tfidf_baseline import TFIDFBaseline, build_tfidf_baseline
from recommender import rerank_chunks


class QueryEngine:
    """
    Central query engine that manages all indices and orchestrates retrieval.
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self.chunks_by_id: dict[int, dict] = {}
        self.minhash_lsh_index: MinHashLSHIndex | None = None
        self.simhash_index: SimHashIndex | None = None
        self.tfidf_baseline: TFIDFBaseline | None = None
        self._tfidf_weights: dict[int, dict[str, float]] | None = None

    def load_chunks(self, chunks_file: Path = CHUNKS_FILE):
        """Load chunks from JSON file."""
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        print(f"[Engine] Loaded {len(self.chunks)} chunks.")

    def build_all_indices(self):
        """Build all three indices from scratch."""
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")

        # 1. TF-IDF (build first so we can use weights for SimHash)
        self.tfidf_baseline = build_tfidf_baseline(self.chunks)
        self._tfidf_weights = self.tfidf_baseline.get_tfidf_weights(self.chunks)

        # 2. MinHash + LSH
        self.minhash_lsh_index = build_minhash_lsh_index(self.chunks)

        # 3. SimHash (using TF-IDF weights for better fingerprints)
        self.simhash_index = build_simhash_index(self.chunks, self._tfidf_weights)

        # Save indices
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.minhash_lsh_index.save(INDEX_DIR)
        self.simhash_index.save(INDEX_DIR)
        self.tfidf_baseline.save(INDEX_DIR)
        print("[Engine] All indices built and saved.")

    def load_indices(self):
        """Load pre-built indices from disk."""
        self.minhash_lsh_index = MinHashLSHIndex.load(INDEX_DIR)
        self.simhash_index = SimHashIndex.load(INDEX_DIR)
        self.tfidf_baseline = TFIDFBaseline.load(INDEX_DIR)
        print("[Engine] All indices loaded from disk.")

    def _retrieve_lsh(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Retrieve using hybrid LSH (MinHash + SimHash combined via re-ranking).
        Returns timing + results.
        """
        tracemalloc.start()
        start_time = time.perf_counter()

        # MinHash + LSH retrieval
        query_shingles = text_to_shingles(query)
        query_minhash = compute_minhash_signature(query_shingles)
        minhash_results = self.minhash_lsh_index.query(query_minhash, query_shingles, top_k=top_k * 3)

        # SimHash retrieval
        query_simhash = compute_simhash(query)
        simhash_results = self.simhash_index.query(query_simhash, top_k=top_k * 3)

        # Merge candidates
        candidate_ids = set()
        jaccard_scores = {}
        hamming_scores = {}

        for cid, jac in minhash_results:
            candidate_ids.add(cid)
            jaccard_scores[cid] = jac

        for cid, ham in simhash_results:
            candidate_ids.add(cid)
            hamming_scores[cid] = ham

        # Get cosine scores for candidates (for re-ranking)
        cosine_scores = self.tfidf_baseline.query_all_scored(query)

        # Re-rank using recommendation system
        reranked = rerank_chunks(
            candidate_ids=candidate_ids,
            chunks_by_id=self.chunks_by_id,
            jaccard_scores=jaccard_scores,
            cosine_scores=cosine_scores,
            hamming_scores=hamming_scores,
            top_k=top_k,
        )

        elapsed = time.perf_counter() - start_time
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results = []
        for cid, score, breakdown in reranked:
            chunk = self.chunks_by_id[cid]
            results.append({
                **chunk,
                "score": score,
                "score_breakdown": breakdown,
            })

        return {
            "method": "LSH (MinHash + SimHash + Re-ranking)",
            "results": results,
            "time_ms": round(elapsed * 1000, 2),
            "peak_memory_kb": round(peak_mem / 1024, 2),
            "num_candidates": len(candidate_ids),
        }

    def _retrieve_tfidf(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Retrieve using TF-IDF baseline (exact).
        Returns timing + results.
        """
        tracemalloc.start()
        start_time = time.perf_counter()

        tfidf_results = self.tfidf_baseline.query(query, top_k=top_k)

        elapsed = time.perf_counter() - start_time
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results = []
        for cid, score in tfidf_results:
            chunk = self.chunks_by_id[cid]
            results.append({
                **chunk,
                "score": score,
                "score_breakdown": {"cosine": round(score, 4)},
            })

        return {
            "method": "TF-IDF (Exact Baseline)",
            "results": results,
            "time_ms": round(elapsed * 1000, 2),
            "peak_memory_kb": round(peak_mem / 1024, 2),
        }

    def search(self, query: str, method: str = "both", top_k: int = TOP_K) -> dict:
        """
        Run a search query.

        Args:
            query: user's question
            method: "lsh", "tfidf", or "both"
            top_k: number of results

        Returns:
            dict with retrieval results and performance metrics.
        """
        output = {"query": query, "top_k": top_k}

        if method in ("lsh", "both"):
            output["lsh"] = self._retrieve_lsh(query, top_k)

        if method in ("tfidf", "both"):
            output["tfidf"] = self._retrieve_tfidf(query, top_k)

        return output
