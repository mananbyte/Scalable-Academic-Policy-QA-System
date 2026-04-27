"""
TF-IDF + Cosine Similarity Baseline

This is the exact (non-approximate) retrieval method required for comparison.
Uses scikit-learn's TfidfVectorizer for vectorization and cosine_similarity for ranking.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import TFIDF_MAX_FEATURES, TOP_K, INDEX_DIR


class TFIDFBaseline:
    """
    Exact retrieval baseline using TF-IDF vectorization + cosine similarity.
    """

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            lowercase=True,
            sublinear_tf=True,  # log-scaled TF
        )
        self.tfidf_matrix = None
        self.chunk_ids: list[int] = []

    def fit(self, chunks: list[dict]):
        """Fit the TF-IDF vectorizer on all chunks and transform them."""
        texts = [chunk["text"] for chunk in chunks]
        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"[TF-IDF] Built matrix: {self.tfidf_matrix.shape} "
              f"(chunks × features)")

    def query(self, query_text: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        """
        Retrieve top-k chunks by cosine similarity to the query.
        Returns list of (chunk_id, cosine_score) sorted descending.
        """
        query_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            cid = self.chunk_ids[idx]
            score = float(scores[idx])
            if score > 0:
                results.append((cid, score))

        return results

    def query_all_scored(self, query_text: str) -> dict[int, float]:
        """Return cosine similarity for ALL chunks (for re-ranking)."""
        query_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return {self.chunk_ids[i]: float(scores[i]) for i in range(len(self.chunk_ids))}

    def get_tfidf_weights(self, chunks: list[dict]) -> dict[int, dict[str, float]]:
        """
        Extract per-chunk TF-IDF weights for use in SimHash weighting.
        Returns {chunk_id: {token: tfidf_weight}}.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        weights = {}
        for i, chunk in enumerate(chunks):
            cid = chunk["chunk_id"]
            row = self.tfidf_matrix[i].toarray().flatten()
            nonzero = np.nonzero(row)[0]
            weights[cid] = {feature_names[j]: float(row[j]) for j in nonzero}
        return weights

    def save(self, path: Path):
        """Persist the TF-IDF model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tfidf_baseline.pkl", "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "chunk_ids": self.chunk_ids,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "TFIDFBaseline":
        """Load a persisted TF-IDF model."""
        with open(path / "tfidf_baseline.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls()
        model.vectorizer = data["vectorizer"]
        model.tfidf_matrix = data["tfidf_matrix"]
        model.chunk_ids = data["chunk_ids"]
        return model


def build_tfidf_baseline(chunks: list[dict]) -> TFIDFBaseline:
    """Build and return the TF-IDF baseline model."""
    model = TFIDFBaseline()
    model.fit(chunks)
    return model
