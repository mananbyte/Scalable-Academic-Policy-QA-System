"""
SimHash Implementation (from scratch)

SimHash: a locality-sensitive hash for detecting near-duplicate documents.
Uses weighted term features to produce a fixed-length binary fingerprint.
Similarity measured by Hamming distance.
"""
import hashlib
import struct
import pickle
from pathlib import Path

import numpy as np

from config import SIMHASH_BITS, SIMHASH_HAMMING_THRESHOLD, INDEX_DIR, TOP_K


def _hash_token(token: str) -> int:
    """Hash a token to a SIMHASH_BITS-bit integer using SHA256."""
    h = hashlib.sha256(token.encode("utf-8")).digest()
    # Take first 16 bytes = 128 bits
    val = int.from_bytes(h[:SIMHASH_BITS // 8], byteorder="big")
    return val


def compute_simhash(text: str, weights: dict[str, float] | None = None) -> int:
    """
    Compute a SimHash fingerprint for the given text.

    Args:
        text: input text
        weights: optional {token: weight} dict (e.g. TF-IDF weights).
                 If None, all tokens get weight 1.0.

    Returns:
        An integer fingerprint of SIMHASH_BITS bits.
    """
    tokens = text.lower().split()
    if not tokens:
        return 0

    # Accumulator: one float per bit position
    v = np.zeros(SIMHASH_BITS, dtype=np.float64)

    for token in tokens:
        w = weights.get(token, 1.0) if weights else 1.0
        h = _hash_token(token)
        for i in range(SIMHASH_BITS):
            if h & (1 << (SIMHASH_BITS - 1 - i)):
                v[i] += w
            else:
                v[i] -= w

    # Convert to binary fingerprint
    fingerprint = 0
    for i in range(SIMHASH_BITS):
        if v[i] > 0:
            fingerprint |= (1 << (SIMHASH_BITS - 1 - i))

    return fingerprint


def hamming_distance(fp1: int, fp2: int) -> int:
    """Compute Hamming distance between two fingerprints."""
    xor = fp1 ^ fp2
    return bin(xor).count("1")


def hamming_similarity(fp1: int, fp2: int) -> float:
    """Compute similarity as 1 - (hamming_distance / SIMHASH_BITS)."""
    dist = hamming_distance(fp1, fp2)
    return 1.0 - (dist / SIMHASH_BITS)


class SimHashIndex:
    """
    Index that stores SimHash fingerprints and supports Hamming-distance queries.
    Uses a brute-force scan (acceptable for our corpus size ~100-300 chunks).
    """

    def __init__(self, threshold: int = SIMHASH_HAMMING_THRESHOLD):
        self.threshold = threshold
        self.fingerprints: dict[int, int] = {}  # chunk_id -> fingerprint

    def add(self, chunk_id: int, fingerprint: int):
        """Add a chunk's fingerprint to the index."""
        self.fingerprints[chunk_id] = fingerprint

    def query(self, query_fp: int, top_k: int = TOP_K) -> list[tuple[int, float]]:
        """
        Find chunks within Hamming distance threshold.
        Returns list of (chunk_id, hamming_similarity) sorted descending.
        """
        results = []
        for cid, fp in self.fingerprints.items():
            sim = hamming_similarity(query_fp, fp)
            dist = hamming_distance(query_fp, fp)
            if dist <= self.threshold:
                results.append((cid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def query_all_scored(self, query_fp: int) -> dict[int, float]:
        """Return Hamming similarity for ALL chunks (for re-ranking)."""
        return {cid: hamming_similarity(query_fp, fp) for cid, fp in self.fingerprints.items()}

    def save(self, path: Path):
        """Persist the index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "simhash_index.pkl", "wb") as f:
            pickle.dump({
                "threshold": self.threshold,
                "fingerprints": self.fingerprints,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "SimHashIndex":
        """Load a persisted index."""
        with open(path / "simhash_index.pkl", "rb") as f:
            data = pickle.load(f)
        idx = cls(data["threshold"])
        idx.fingerprints = data["fingerprints"]
        return idx


def build_simhash_index(chunks: list[dict], tfidf_weights: dict[int, dict[str, float]] | None = None) -> SimHashIndex:
    """
    Build SimHash fingerprints and index from chunk list.

    Args:
        chunks: list of chunk dicts
        tfidf_weights: optional {chunk_id: {token: weight}} for weighted SimHash
    """
    index = SimHashIndex()
    print(f"[SimHash] Building index for {len(chunks)} chunks...")
    for chunk in chunks:
        cid = chunk["chunk_id"]
        weights = tfidf_weights.get(cid) if tfidf_weights else None
        fp = compute_simhash(chunk["text"], weights)
        index.add(cid, fp)
    print("[SimHash] Index built.")
    return index
