"""
Experiments Module

Runs required experimental comparisons:
1. Exact vs Approximate Retrieval
2. Parameter Sensitivity Analysis
3. Scalability Test
4. Evaluation Metrics (Precision@k, query latency)
"""
import json
import time
import copy
import tracemalloc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DIR,
    INDEX_DIR,
    TOP_K,
    MINHASH_NUM_HASHES,
    LSH_NUM_BANDS,
    SIMHASH_HAMMING_THRESHOLD,
)
from minhash_lsh import (
    text_to_shingles,
    compute_minhash_signature,
    MinHashLSHIndex,
)
from simhash import compute_simhash, SimHashIndex
from tfidf_baseline import TFIDFBaseline, build_tfidf_baseline
from recommender import rerank_chunks


# ─── Sample Queries (from project spec) ──────────────────────────────────────
SAMPLE_QUERIES = [
    "What is the minimum GPA requirement?",
    "What happens if a student fails a course?",
    "What is the attendance policy?",
    "How many times can a course be repeated?",
    "What are the credit hour requirements for graduation?",
    "What is the fee refund policy?",
    "How is the semester GPA calculated?",
    "What is the policy on plagiarism?",
    "Can a student change their program?",
    "What are the hostel rules?",
    "What is the grading system?",
    "How can a student apply for re-checking of papers?",
    "What are the rules for dropping a course?",
    "What is the medal and prize policy?",
    "What happens if a student's GPA falls below minimum?",
]


def experiment_exact_vs_approximate(engine, output_dir: Path):
    """
    Experiment 1: Compare TF-IDF (exact) vs LSH (approximate).
    Measures accuracy overlap, time, and memory.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Exact vs Approximate Retrieval")
    print("="*60)

    results = []
    for query in SAMPLE_QUERIES:
        search = engine.search(query, method="both", top_k=TOP_K)

        lsh_ids = {r["chunk_id"] for r in search["lsh"]["results"]}
        tfidf_ids = {r["chunk_id"] for r in search["tfidf"]["results"]}

        overlap = len(lsh_ids & tfidf_ids)
        precision_at_k = overlap / TOP_K if TOP_K > 0 else 0

        results.append({
            "query": query,
            "lsh_time_ms": search["lsh"]["time_ms"],
            "tfidf_time_ms": search["tfidf"]["time_ms"],
            "lsh_memory_kb": search["lsh"]["peak_memory_kb"],
            "tfidf_memory_kb": search["tfidf"]["peak_memory_kb"],
            "overlap": overlap,
            "overlap_ratio": precision_at_k,
            "lsh_candidates": search["lsh"].get("num_candidates", 0),
        })

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp1_exact_vs_approx.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot: Time comparison
    queries_short = [q[:30] + "..." for q in SAMPLE_QUERIES]
    lsh_times = [r["lsh_time_ms"] for r in results]
    tfidf_times = [r["tfidf_time_ms"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Time comparison
    x = np.arange(len(SAMPLE_QUERIES))
    axes[0].bar(x - 0.2, lsh_times, 0.4, label="LSH", color="#4CAF50")
    axes[0].bar(x + 0.2, tfidf_times, 0.4, label="TF-IDF", color="#2196F3")
    axes[0].set_xlabel("Query")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Query Latency: LSH vs TF-IDF")
    axes[0].legend()
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Q{i+1}" for i in x], rotation=45)

    # Memory comparison
    lsh_mem = [r["lsh_memory_kb"] for r in results]
    tfidf_mem = [r["tfidf_memory_kb"] for r in results]
    axes[1].bar(x - 0.2, lsh_mem, 0.4, label="LSH", color="#4CAF50")
    axes[1].bar(x + 0.2, tfidf_mem, 0.4, label="TF-IDF", color="#2196F3")
    axes[1].set_xlabel("Query")
    axes[1].set_ylabel("Memory (KB)")
    axes[1].set_title("Peak Memory: LSH vs TF-IDF")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Q{i+1}" for i in x], rotation=45)

    # Overlap
    overlaps = [r["overlap_ratio"] for r in results]
    axes[2].bar(x, overlaps, color="#FF9800")
    axes[2].set_xlabel("Query")
    axes[2].set_ylabel("Overlap Ratio")
    axes[2].set_title(f"Result Overlap (Top-{TOP_K})")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"Q{i+1}" for i in x], rotation=45)
    axes[2].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / "exp1_exact_vs_approx.png", dpi=150)
    plt.close()

    # Print summary
    avg_lsh_time = np.mean(lsh_times)
    avg_tfidf_time = np.mean(tfidf_times)
    avg_overlap = np.mean(overlaps)
    print(f"  Avg LSH time:    {avg_lsh_time:.2f} ms")
    print(f"  Avg TF-IDF time: {avg_tfidf_time:.2f} ms")
    print(f"  Avg overlap:     {avg_overlap:.2%}")

    return results


def experiment_parameter_sensitivity(chunks, output_dir: Path):
    """
    Experiment 2: Parameter sensitivity analysis.
    - MinHash: vary num_hashes
    - LSH: vary num_bands
    - SimHash: vary Hamming threshold
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Parameter Sensitivity")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)
    test_query = "What is the minimum GPA requirement?"

    # Build a TF-IDF baseline once for comparison (ground truth)
    tfidf = build_tfidf_baseline(chunks)
    tfidf_results = tfidf.query(test_query, top_k=10)
    ground_truth_ids = {cid for cid, _ in tfidf_results[:TOP_K]}

    # ── 2a: Vary MinHash num_hashes ──────────────────────────────────────
    hash_counts = [32, 64, 128, 256]
    minhash_results = []

    for nh in hash_counts:
        print(f"  Testing MinHash num_hashes={nh}...")
        # Build index with different num_hashes
        from minhash_lsh import compute_minhash_signature as compute_mh
        idx = MinHashLSHIndex(num_bands=max(1, nh // 8), rows_per_band=min(8, nh))

        start = time.perf_counter()
        for chunk in chunks:
            shingles = text_to_shingles(chunk["text"])
            sig = compute_mh(shingles, num_hashes=nh)
            idx.add(chunk["chunk_id"], sig, shingles)

        build_time = time.perf_counter() - start

        # Query
        q_shingles = text_to_shingles(test_query)
        q_sig = compute_mh(q_shingles, num_hashes=nh)
        start = time.perf_counter()
        results = idx.query(q_sig, q_shingles, top_k=TOP_K)
        query_time = time.perf_counter() - start

        result_ids = {cid for cid, _ in results}
        overlap = len(result_ids & ground_truth_ids) / TOP_K

        minhash_results.append({
            "num_hashes": nh,
            "build_time_ms": round(build_time * 1000, 2),
            "query_time_ms": round(query_time * 1000, 2),
            "overlap_with_tfidf": overlap,
        })

    # ── 2b: Vary LSH num_bands ───────────────────────────────────────────
    band_counts = [4, 8, 16, 32, 64]
    lsh_results = []
    nh = MINHASH_NUM_HASHES

    for nb in band_counts:
        rpb = nh // nb
        if rpb < 1:
            continue
        print(f"  Testing LSH bands={nb}, rows_per_band={rpb}...")
        idx = MinHashLSHIndex(num_bands=nb, rows_per_band=rpb)

        for chunk in chunks:
            shingles = text_to_shingles(chunk["text"])
            sig = compute_mh(shingles, num_hashes=nh)
            idx.add(chunk["chunk_id"], sig, shingles)

        q_shingles = text_to_shingles(test_query)
        q_sig = compute_mh(q_shingles, num_hashes=nh)
        start = time.perf_counter()
        results = idx.query(q_sig, q_shingles, top_k=TOP_K)
        query_time = time.perf_counter() - start

        result_ids = {cid for cid, _ in results}
        overlap = len(result_ids & ground_truth_ids) / TOP_K
        num_candidates = len(set().union(*[
            set().union(*band.values()) for band in idx.band_buckets
        ]))

        lsh_results.append({
            "num_bands": nb,
            "rows_per_band": rpb,
            "query_time_ms": round(query_time * 1000, 2),
            "overlap_with_tfidf": overlap,
            "total_candidates": num_candidates,
        })

    # ── 2c: Vary SimHash Hamming threshold ───────────────────────────────
    thresholds = [3, 5, 10, 20, 40]
    simhash_results = []

    simhash_idx = SimHashIndex()
    tfidf_weights = tfidf.get_tfidf_weights(chunks)
    for chunk in chunks:
        cid = chunk["chunk_id"]
        w = tfidf_weights.get(cid)
        fp = compute_simhash(chunk["text"], w)
        simhash_idx.add(cid, fp)

    q_fp = compute_simhash(test_query)

    for th in thresholds:
        print(f"  Testing SimHash threshold={th}...")
        simhash_idx.threshold = th
        start = time.perf_counter()
        results = simhash_idx.query(q_fp, top_k=TOP_K)
        query_time = time.perf_counter() - start

        result_ids = {cid for cid, _ in results}
        overlap = len(result_ids & ground_truth_ids) / TOP_K
        all_within = sum(1 for cid, fp in simhash_idx.fingerprints.items()
                         if hamming_distance_calc(q_fp, fp) <= th)

        simhash_results.append({
            "threshold": th,
            "query_time_ms": round(query_time * 1000, 2),
            "overlap_with_tfidf": overlap,
            "chunks_within_threshold": all_within,
        })

    # Save all results
    with open(output_dir / "exp2_parameter_sensitivity.json", "w") as f:
        json.dump({
            "minhash_num_hashes": minhash_results,
            "lsh_num_bands": lsh_results,
            "simhash_threshold": simhash_results,
        }, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # MinHash
    axes[0].plot([r["num_hashes"] for r in minhash_results],
                 [r["overlap_with_tfidf"] for r in minhash_results],
                 "o-", color="#4CAF50", linewidth=2)
    axes[0].set_xlabel("Number of Hash Functions")
    axes[0].set_ylabel("Overlap with TF-IDF (Top-5)")
    axes[0].set_title("MinHash: Hash Functions vs Accuracy")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)

    # LSH bands
    axes[1].plot([r["num_bands"] for r in lsh_results],
                 [r["overlap_with_tfidf"] for r in lsh_results],
                 "o-", color="#2196F3", linewidth=2)
    axes[1].set_xlabel("Number of Bands")
    axes[1].set_ylabel("Overlap with TF-IDF (Top-5)")
    axes[1].set_title("LSH: Bands vs Accuracy")
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)

    # SimHash threshold
    axes[2].plot([r["threshold"] for r in simhash_results],
                 [r["overlap_with_tfidf"] for r in simhash_results],
                 "o-", color="#FF9800", linewidth=2)
    ax2 = axes[2].twinx()
    ax2.plot([r["threshold"] for r in simhash_results],
             [r["chunks_within_threshold"] for r in simhash_results],
             "s--", color="#9C27B0", linewidth=2, alpha=0.7)
    axes[2].set_xlabel("Hamming Threshold")
    axes[2].set_ylabel("Overlap with TF-IDF", color="#FF9800")
    ax2.set_ylabel("Chunks Within Threshold", color="#9C27B0")
    axes[2].set_title("SimHash: Threshold vs Accuracy")
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp2_parameter_sensitivity.png", dpi=150)
    plt.close()

    print("  Parameter sensitivity analysis complete.")


def experiment_scalability(chunks, output_dir: Path):
    """
    Experiment 3: Scalability test.
    Duplicate corpus 1x, 2x, 5x, 10x and measure build + query time.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Scalability Test")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)
    test_query = "What is the minimum GPA requirement?"
    multipliers = [1, 2, 5, 10]
    results = []

    for mult in multipliers:
        print(f"  Testing {mult}x corpus size...")
        # Create scaled corpus
        scaled_chunks = []
        cid = 0
        for _ in range(mult):
            for chunk in chunks:
                new_chunk = {**chunk, "chunk_id": cid}
                scaled_chunks.append(new_chunk)
                cid += 1

        num_chunks = len(scaled_chunks)

        # Build TF-IDF
        start = time.perf_counter()
        tfidf = build_tfidf_baseline(scaled_chunks)
        tfidf_build = time.perf_counter() - start

        # Build MinHash+LSH
        start = time.perf_counter()
        from minhash_lsh import build_minhash_lsh_index
        minhash_idx = build_minhash_lsh_index(scaled_chunks)
        minhash_build = time.perf_counter() - start

        # Query TF-IDF
        start = time.perf_counter()
        tfidf.query(test_query, top_k=TOP_K)
        tfidf_query = time.perf_counter() - start

        # Query MinHash+LSH
        q_shingles = text_to_shingles(test_query)
        q_sig = compute_minhash_signature(q_shingles)
        start = time.perf_counter()
        minhash_idx.query(q_sig, q_shingles, top_k=TOP_K)
        minhash_query = time.perf_counter() - start

        results.append({
            "multiplier": mult,
            "num_chunks": num_chunks,
            "tfidf_build_ms": round(tfidf_build * 1000, 2),
            "minhash_build_ms": round(minhash_build * 1000, 2),
            "tfidf_query_ms": round(tfidf_query * 1000, 2),
            "minhash_query_ms": round(minhash_query * 1000, 2),
        })

    with open(output_dir / "exp3_scalability.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    chunks_counts = [r["num_chunks"] for r in results]

    # Build time
    axes[0].plot(chunks_counts, [r["tfidf_build_ms"] for r in results],
                 "o-", label="TF-IDF", color="#2196F3", linewidth=2)
    axes[0].plot(chunks_counts, [r["minhash_build_ms"] for r in results],
                 "o-", label="MinHash+LSH", color="#4CAF50", linewidth=2)
    axes[0].set_xlabel("Number of Chunks")
    axes[0].set_ylabel("Build Time (ms)")
    axes[0].set_title("Index Build Time vs Corpus Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Query time
    axes[1].plot(chunks_counts, [r["tfidf_query_ms"] for r in results],
                 "o-", label="TF-IDF", color="#2196F3", linewidth=2)
    axes[1].plot(chunks_counts, [r["minhash_query_ms"] for r in results],
                 "o-", label="MinHash+LSH", color="#4CAF50", linewidth=2)
    axes[1].set_xlabel("Number of Chunks")
    axes[1].set_ylabel("Query Time (ms)")
    axes[1].set_title("Query Time vs Corpus Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp3_scalability.png", dpi=150)
    plt.close()

    print("  Scalability test complete.")
    return results


# Helper for simhash threshold experiment
def hamming_distance_calc(fp1: int, fp2: int) -> int:
    """Compute Hamming distance between two fingerprints."""
    xor = fp1 ^ fp2
    return bin(xor).count("1")


# ─── Import helpers ──────────────────────────────────────────────────────────
from minhash_lsh import text_to_shingles, compute_minhash_signature
from simhash import compute_simhash


def run_all_experiments(engine, chunks, output_dir: Path = PROCESSED_DIR / "experiments"):
    """Run all three required experiments."""
    print("\n" + "#"*60)
    print("# RUNNING ALL EXPERIMENTS")
    print("#"*60)

    experiment_exact_vs_approximate(engine, output_dir)
    experiment_parameter_sensitivity(chunks, output_dir)
    experiment_scalability(chunks, output_dir)

    print(f"\n[Experiments] All results saved to {output_dir}")
