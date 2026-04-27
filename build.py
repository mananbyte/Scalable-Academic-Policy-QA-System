"""
Build script — runs ingestion + index building in one go.
Run this once to prepare the system.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion import run_ingestion
from query_engine import QueryEngine
from config import CHUNKS_FILE


def build():
    # Step 1: Ingest (extract + chunk)
    if not CHUNKS_FILE.exists():
        print("=" * 50)
        print("STEP 1: Data Ingestion")
        print("=" * 50)
        run_ingestion()
    else:
        print(f"[Build] Chunks already exist at {CHUNKS_FILE}, skipping ingestion.")

    # Step 2: Build all indices
    print("\n" + "=" * 50)
    print("STEP 2: Building Indices")
    print("=" * 50)
    engine = QueryEngine()
    engine.load_chunks()
    engine.build_all_indices()

    # Step 3: Quick test query
    print("\n" + "=" * 50)
    print("STEP 3: Test Query")
    print("=" * 50)
    test_q = "What is the minimum GPA requirement?"
    results = engine.search(test_q, method="both", top_k=3)

    print(f"\nQuery: {test_q}")
    if "lsh" in results:
        print(f"\n--- LSH Results ({results['lsh']['time_ms']}ms) ---")
        for r in results["lsh"]["results"]:
            print(f"  [Chunk {r['chunk_id']}] Score={r['score']:.4f} | Page {r['start_page']} | {r['section_title'][:50]}")
            print(f"    {r['text'][:120]}...")

    if "tfidf" in results:
        print(f"\n--- TF-IDF Results ({results['tfidf']['time_ms']}ms) ---")
        for r in results["tfidf"]["results"]:
            print(f"  [Chunk {r['chunk_id']}] Score={r['score']:.4f} | Page {r['start_page']} | {r['section_title'][:50]}")
            print(f"    {r['text'][:120]}...")

    print("\n[OK] Build complete! Run 'streamlit run app.py' to start the UI.")


if __name__ == "__main__":
    build()
