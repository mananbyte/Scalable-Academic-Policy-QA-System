"""
MinHash + LSH Implementation (from scratch)

MinHash: approximate Jaccard similarity via random hash functions.
LSH banding: hash signature bands to find candidate similar pairs efficiently.
"""
import hashlib
import json
import struct
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import (
    SHINGLE_SIZE,
    MINHASH_NUM_HASHES,
    LSH_NUM_BANDS,
    LSH_ROWS_PER_BAND,
    INDEX_DIR,
    TOP_K,
)


# ─── Shingling ───────────────────────────────────────────────────────────────

def text_to_shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """Convert text to a set of k-word shingles."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


# ─── MinHash Signature ──────────────────────────────────────────────────────

def _hash_shingle(shingle: str, seed: int) -> int:
    """Hash a shingle with a given seed using MD5 + struct unpack."""
    h = hashlib.md5(f"{seed}:{shingle}".encode("utf-8")).digest()
    return struct.unpack("<Q", h[:8])[0]  # unsigned 64-bit int


def compute_minhash_signature(shingles: set[str], num_hashes: int = MINHASH_NUM_HASHES) -> np.ndarray:
    """Compute the MinHash signature (array of min-hash values) for a shingle set."""
    if not shingles:
        return np.full(num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)

    signature = np.full(num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)

    for shingle in shingles:
        for i in range(num_hashes):
            h = _hash_shingle(shingle, i)
            if h < signature[i]:
                signature[i] = h

    return signature


def jaccard_from_signatures(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return float(np.sum(sig_a == sig_b)) / len(sig_a)


# ─── LSH Banding ─────────────────────────────────────────────────────────────

class MinHashLSHIndex:
    """
    Locality-Sensitive Hashing index using banding technique.

    Divide each signature into `num_bands` bands of `rows_per_band` rows.
    Hash each band to a bucket; chunks in the same bucket are candidates.
    """

    def __init__(self, num_bands: int = LSH_NUM_BANDS, rows_per_band: int = LSH_ROWS_PER_BAND):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        # band_index[band_id] = {bucket_hash: set of chunk_ids}
        self.band_buckets: list[dict[int, set[int]]] = [
            defaultdict(set) for _ in range(num_bands)
        ]
        self.signatures: dict[int, np.ndarray] = {}
        self.shingle_sets: dict[int, set[str]] = {}

    def _band_hash(self, band_values: np.ndarray) -> int:
        """Hash a band (sub-array of signature) to a bucket key."""
        return hash(band_values.tobytes())

    def add(self, chunk_id: int, signature: np.ndarray, shingles: set[str]):
        """Add a chunk's MinHash signature to the index."""
        self.signatures[chunk_id] = signature
        self.shingle_sets[chunk_id] = shingles
        for band_id in range(self.num_bands):
            start = band_id * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            bucket_key = self._band_hash(band)
            self.band_buckets[band_id][bucket_key].add(chunk_id)

    def query(self, query_signature: np.ndarray, query_shingles: set[str], top_k: int = TOP_K) -> list[tuple[int, float]]:
        """
        Find candidate chunks for a query signature, then rank by estimated Jaccard.
        Falls back to brute-force scan if LSH banding finds no candidates.
        Returns list of (chunk_id, jaccard_estimate) sorted descending.
        """
        candidates = set()
        for band_id in range(self.num_bands):
            start = band_id * self.rows_per_band
            end = start + self.rows_per_band
            band = query_signature[start:end]
            bucket_key = self._band_hash(band)
            candidates |= self.band_buckets[band_id].get(bucket_key, set())

        # Fallback: if no LSH candidates (common for short queries),
        # do a brute-force scan over all signatures
        if not candidates:
            candidates = set(self.signatures.keys())

        # Rank candidates by Jaccard estimate from signatures
        results = []
        for cid in candidates:
            jac = jaccard_from_signatures(query_signature, self.signatures[cid])
            results.append((cid, jac))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, path: Path):
        """Persist the index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "minhash_lsh_index.pkl", "wb") as f:
            pickle.dump({
                "num_bands": self.num_bands,
                "rows_per_band": self.rows_per_band,
                "band_buckets": [{k: v for k, v in band.items()} for band in self.band_buckets],
                "signatures": {k: v.tolist() for k, v in self.signatures.items()},
                # Don't save shingle sets (too large), recompute if needed
            }, f)

    @classmethod
    def load(cls, path: Path) -> "MinHashLSHIndex":
        """Load a persisted index."""
        with open(path / "minhash_lsh_index.pkl", "rb") as f:
            data = pickle.load(f)
        idx = cls(data["num_bands"], data["rows_per_band"])
        idx.band_buckets = [defaultdict(set, {k: v for k, v in band.items()}) for band in data["band_buckets"]]
        idx.signatures = {k: np.array(v, dtype=np.uint64) for k, v in data["signatures"].items()}
        return idx


def build_minhash_lsh_index(chunks: list[dict]) -> MinHashLSHIndex:
    """Build MinHash signatures and LSH index from chunk list."""
    index = MinHashLSHIndex()
    print(f"[MinHash+LSH] Building index for {len(chunks)} chunks...")
    for chunk in chunks:
        shingles = text_to_shingles(chunk["text"])
        signature = compute_minhash_signature(shingles)
        index.add(chunk["chunk_id"], signature, shingles)
    print("[MinHash+LSH] Index built.")
    return index
