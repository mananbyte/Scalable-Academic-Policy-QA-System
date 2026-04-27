"""
Recommendation-based Re-ranking Extension

Combines multiple retrieval signals (Jaccard from MinHash, Hamming from SimHash,
Cosine from TF-IDF, and section importance) into a unified relevance score.
This acts as a "recommendation engine" that re-ranks candidates for optimal top-k.
"""
import re
from config import (
    RERANK_WEIGHT_JACCARD,
    RERANK_WEIGHT_COSINE,
    RERANK_WEIGHT_HAMMING,
    RERANK_WEIGHT_SECTION,
    TOP_K,
)


# Section importance heuristic: chapters with academic rules score higher
SECTION_IMPORTANCE = {
    "chapter 2": 1.0,   # Scheme of Studies, Exams, Academic Standards
    "chapter 3": 1.0,   # Award of Degree & Academic Deficiencies
    "chapter 4": 0.9,   # Architecture Degrees
    "chapter 5": 0.9,   # Management/Social Sciences Degrees
    "chapter 6": 1.0,   # Academic Provisions & Flexibilities
    "chapter 7": 0.7,   # Issuance of Degrees & Transcripts
    "chapter 8": 0.3,   # Clubs & Societies
    "chapter 9": 0.4,   # International Students
    "chapter 10": 0.3,  # Social Media & IT
    "chapter 11": 0.6,  # Code of Conduct
    "chapter 12": 0.4,  # Living on Campus
}


def get_section_score(section_title: str) -> float:
    """
    Compute a section importance score based on the chapter.
    Academic/policy chapters get higher scores.
    """
    if not section_title:
        return 0.5  # default mid-range score

    lower = section_title.lower()
    for key, score in SECTION_IMPORTANCE.items():
        if key in lower:
            return score

    return 0.5


def rerank_chunks(
    candidate_ids: set[int],
    chunks_by_id: dict[int, dict],
    jaccard_scores: dict[int, float],
    cosine_scores: dict[int, float],
    hamming_scores: dict[int, float],
    top_k: int = TOP_K,
) -> list[tuple[int, float, dict]]:
    """
    Re-rank candidate chunks using a weighted combination of multiple signals.

    Args:
        candidate_ids: set of chunk IDs to consider
        chunks_by_id: {chunk_id: chunk_dict}
        jaccard_scores: {chunk_id: Jaccard estimate from MinHash}
        cosine_scores: {chunk_id: cosine similarity from TF-IDF}
        hamming_scores: {chunk_id: Hamming similarity from SimHash}
        top_k: number of results to return

    Returns:
        List of (chunk_id, combined_score, score_breakdown) sorted descending.
    """
    results = []

    for cid in candidate_ids:
        chunk = chunks_by_id.get(cid, {})
        section = chunk.get("section_title", "")

        jac = jaccard_scores.get(cid, 0.0)
        cos = cosine_scores.get(cid, 0.0)
        ham = hamming_scores.get(cid, 0.0)
        sec = get_section_score(section)

        combined = (
            RERANK_WEIGHT_JACCARD * jac
            + RERANK_WEIGHT_COSINE * cos
            + RERANK_WEIGHT_HAMMING * ham
            + RERANK_WEIGHT_SECTION * sec
        )

        breakdown = {
            "jaccard": round(jac, 4),
            "cosine": round(cos, 4),
            "hamming": round(ham, 4),
            "section": round(sec, 4),
            "combined": round(combined, 4),
        }

        results.append((cid, combined, breakdown))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
