"""
Configuration file for the Scalable Academic Policy QA System.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

UG_HANDBOOK_PDF = PROJECT_ROOT / "28-Aug-2024-Undergraduate-Handbook.pdf"
UG_HANDBOOK_TXT = RAW_DIR / "ug_handbook.txt"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
INDEX_DIR = PROCESSED_DIR / "indices"

# ─── Chunking ────────────────────────────────────────────────────────────────
CHUNK_MIN_WORDS = 200
CHUNK_MAX_WORDS = 500
CHUNK_OVERLAP_WORDS = 50  # overlap between consecutive chunks

# ─── Shingling (for MinHash) ─────────────────────────────────────────────────
SHINGLE_SIZE = 3  # k-shingles (word n-grams)

# ─── MinHash ─────────────────────────────────────────────────────────────────
MINHASH_NUM_HASHES = 128  # number of hash functions

# ─── LSH Banding ─────────────────────────────────────────────────────────────
LSH_NUM_BANDS = 16  # bands (MINHASH_NUM_HASHES must be divisible by this)
LSH_ROWS_PER_BAND = MINHASH_NUM_HASHES // LSH_NUM_BANDS

# ─── SimHash ─────────────────────────────────────────────────────────────────
SIMHASH_BITS = 128  # fingerprint bit-length
SIMHASH_HAMMING_THRESHOLD = 10  # max Hamming distance for "similar"

# ─── TF-IDF ──────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10000

# ─── Retrieval ───────────────────────────────────────────────────────────────
TOP_K = 5  # default number of chunks to return

# ─── Recommendation Re-ranking weights ───────────────────────────────────────
RERANK_WEIGHT_JACCARD = 0.30
RERANK_WEIGHT_COSINE = 0.35
RERANK_WEIGHT_HAMMING = 0.20
RERANK_WEIGHT_SECTION = 0.15

# ─── Gemini LLM ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAjYWux3afvQc5uCaLJRciLRwjlvaAKbEk")
GEMINI_MODEL = "gemini-3-flash-preview"

# ─── NLTK Data ───────────────────────────────────────────────────────────────
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
