"""
Experiments Module

Runs required experimental comparisons:
1. Exact vs Approximate Retrieval (TF-IDF vs LSH)
2. Parameter Sensitivity (num_hashes, num_bands, hamming_threshold)
3. Scalability Test (1x to 10x corpus)
4. Evaluation Metrics (Precision@k, latency, qualitative eval)
"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DIR,
    TOP_K,
    MINHASH_NUM_HASHES,
    LSH_NUM_BANDS,
    LSH_ROWS_PER_BAND,
    SHINGLE_SIZE,
)
from minhash_lsh import (
    text_to_shingles,
    compute_minhash_signature,
    jaccard_from_signatures,
    MinHashLSHIndex,
)
from simhash import compute_simhash, SimHashIndex
from tfidf_baseline import build_tfidf_baseline


# ─── Sample Queries ──────────────────────────────────────────────────────────
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


def _exact_jaccard(set_a: set, set_b: set) -> float:
    """Compute exact Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _get_exact_jaccard_top_k(query: str, chunks: list[dict], k: int) -> list[tuple[int, float]]:
    """Brute-force exact Jaccard ranking (ground truth for MinHash)."""
    q_shingles = text_to_shingles(query)
    scores = []
    for chunk in chunks:
        c_shingles = text_to_shingles(chunk["text"])
        jac = _exact_jaccard(q_shingles, c_shingles)
        scores.append((chunk["chunk_id"], jac))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Exact vs Approximate Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_exact_vs_approximate(engine, output_dir: Path):
    """
    Compare TF-IDF (exact) vs MinHash+LSH (approximate).
    Evaluate: accuracy (relevance), time, memory.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Exact vs Approximate Retrieval")
    print("="*60)

    results = []
    for i, query in enumerate(SAMPLE_QUERIES):
        search = engine.search(query, method="both", top_k=TOP_K)

        lsh_ids = {r["chunk_id"] for r in search["lsh"]["results"]}
        tfidf_ids = {r["chunk_id"] for r in search["tfidf"]["results"]}
        overlap = len(lsh_ids & tfidf_ids)
        overlap_ratio = overlap / TOP_K if TOP_K > 0 else 0

        # Average relevance scores
        lsh_avg_score = np.mean([r["score"] for r in search["lsh"]["results"]]) if search["lsh"]["results"] else 0
        tfidf_avg_score = np.mean([r["score"] for r in search["tfidf"]["results"]]) if search["tfidf"]["results"] else 0

        results.append({
            "query": query,
            "lsh_time_ms": search["lsh"]["time_ms"],
            "tfidf_time_ms": search["tfidf"]["time_ms"],
            "lsh_memory_kb": search["lsh"]["peak_memory_kb"],
            "tfidf_memory_kb": search["tfidf"]["peak_memory_kb"],
            "overlap": overlap,
            "overlap_ratio": overlap_ratio,
            "lsh_avg_score": round(lsh_avg_score, 4),
            "tfidf_avg_score": round(tfidf_avg_score, 4),
            "lsh_top_chunks": [r["chunk_id"] for r in search["lsh"]["results"]],
            "tfidf_top_chunks": [r["chunk_id"] for r in search["tfidf"]["results"]],
        })
        print(f"  Q{i+1}: LSH={search['lsh']['time_ms']}ms, TF-IDF={search['tfidf']['time_ms']}ms, Overlap={overlap}/{TOP_K}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp1_exact_vs_approx.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(SAMPLE_QUERIES))
    labels = [f"Q{i+1}" for i in x]

    # Time comparison
    axes[0].bar(x - 0.2, [r["lsh_time_ms"] for r in results], 0.4,
                label="LSH (Approx)", color="#4CAF50", alpha=0.85)
    axes[0].bar(x + 0.2, [r["tfidf_time_ms"] for r in results], 0.4,
                label="TF-IDF (Exact)", color="#2196F3", alpha=0.85)
    axes[0].set_xlabel("Query")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Query Latency Comparison")
    axes[0].legend()
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)

    # Memory comparison
    axes[1].bar(x - 0.2, [r["lsh_memory_kb"] for r in results], 0.4,
                label="LSH (Approx)", color="#4CAF50", alpha=0.85)
    axes[1].bar(x + 0.2, [r["tfidf_memory_kb"] for r in results], 0.4,
                label="TF-IDF (Exact)", color="#2196F3", alpha=0.85)
    axes[1].set_xlabel("Query")
    axes[1].set_ylabel("Memory (KB)")
    axes[1].set_title("Peak Memory Usage")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)

    # Overlap + scores
    axes[2].bar(x - 0.2, [r["lsh_avg_score"] for r in results], 0.4,
                label="LSH Score", color="#4CAF50", alpha=0.85)
    axes[2].bar(x + 0.2, [r["tfidf_avg_score"] for r in results], 0.4,
                label="TF-IDF Score", color="#2196F3", alpha=0.85)
    axes[2].set_xlabel("Query")
    axes[2].set_ylabel("Avg Relevance Score")
    axes[2].set_title("Retrieval Quality (Avg Score)")
    axes[2].legend()
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "exp1_exact_vs_approx.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    avg_lsh = np.mean([r["lsh_time_ms"] for r in results])
    avg_tfidf = np.mean([r["tfidf_time_ms"] for r in results])
    avg_overlap = np.mean([r["overlap_ratio"] for r in results])
    print(f"\n  Summary:")
    print(f"    Avg LSH time:    {avg_lsh:.2f} ms")
    print(f"    Avg TF-IDF time: {avg_tfidf:.2f} ms")
    print(f"    Avg overlap:     {avg_overlap:.2%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Parameter Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_parameter_sensitivity(chunks, output_dir: Path):
    """
    Analyze impact of:
    - Number of hash functions (MinHash)
    - Number of bands (LSH)
    - Hamming threshold (SimHash)

    Uses exact Jaccard as ground truth for MinHash/LSH evaluations.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Parameter Sensitivity")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use multiple test queries for robustness
    test_queries = SAMPLE_QUERIES[:5]

    # Precompute exact Jaccard ground truth for each query
    print("  Computing ground truth (exact Jaccard)...")
    ground_truths = {}
    for q in test_queries:
        ground_truths[q] = _get_exact_jaccard_top_k(q, chunks, TOP_K)

    # Precompute all shingle sets once
    chunk_shingles = {}
    for chunk in chunks:
        chunk_shingles[chunk["chunk_id"]] = text_to_shingles(chunk["text"])

    # ── 2a: Vary MinHash num_hashes ──────────────────────────────────────
    hash_counts = [16, 32, 64, 128, 256]
    minhash_results = []

    for nh in hash_counts:
        print(f"  Testing MinHash num_hashes={nh}...")

        nb = max(2, nh // 8)  # bands
        rpb = nh // nb         # rows per band

        # Build index
        t0 = time.perf_counter()
        idx = MinHashLSHIndex(num_bands=nb, rows_per_band=rpb)
        for chunk in chunks:
            shingles = chunk_shingles[chunk["chunk_id"]]
            sig = compute_minhash_signature(shingles, num_hashes=nh)
            idx.add(chunk["chunk_id"], sig, shingles)
        build_ms = (time.perf_counter() - t0) * 1000

        # Query across all test queries
        recalls = []
        query_times = []
        jaccard_errors = []

        for q in test_queries:
            gt_ids = {cid for cid, _ in ground_truths[q]}

            q_shingles = text_to_shingles(q)
            q_sig = compute_minhash_signature(q_shingles, num_hashes=nh)

            t0 = time.perf_counter()
            results = idx.query(q_sig, q_shingles, top_k=TOP_K)
            query_times.append((time.perf_counter() - t0) * 1000)

            result_ids = {cid for cid, _ in results}
            recall = len(result_ids & gt_ids) / TOP_K if TOP_K > 0 else 0
            recalls.append(recall)

            # Measure Jaccard approximation error
            for cid in list(result_ids)[:3]:
                if cid in idx.signatures:
                    est = jaccard_from_signatures(q_sig, idx.signatures[cid])
                    exact = _exact_jaccard(q_shingles, chunk_shingles.get(cid, set()))
                    if exact > 0:
                        jaccard_errors.append(abs(est - exact))

        minhash_results.append({
            "num_hashes": nh,
            "build_time_ms": round(build_ms, 1),
            "avg_query_time_ms": round(np.mean(query_times), 2),
            "avg_recall": round(np.mean(recalls), 4),
            "avg_jaccard_error": round(np.mean(jaccard_errors), 4) if jaccard_errors else 0,
        })

    # ── 2b: Vary LSH num_bands ───────────────────────────────────────────
    band_configs = [
        (2, 64), (4, 32), (8, 16), (16, 8), (32, 4), (64, 2),
    ]
    lsh_results = []
    nh = MINHASH_NUM_HASHES  # fixed at 128

    # Pre-compute all signatures with fixed num_hashes
    all_sigs = {}
    for chunk in chunks:
        shingles = chunk_shingles[chunk["chunk_id"]]
        all_sigs[chunk["chunk_id"]] = compute_minhash_signature(shingles, num_hashes=nh)

    for nb, rpb in band_configs:
        if nb * rpb != nh:
            continue
        print(f"  Testing LSH bands={nb}, rows_per_band={rpb}...")

        # Build index
        idx = MinHashLSHIndex(num_bands=nb, rows_per_band=rpb)
        for chunk in chunks:
            sig = all_sigs[chunk["chunk_id"]]
            idx.add(chunk["chunk_id"], sig, chunk_shingles[chunk["chunk_id"]])

        # Count actual banding candidates (before fallback)
        recalls = []
        banding_candidates_counts = []
        query_times = []

        for q in test_queries:
            gt_ids = {cid for cid, _ in ground_truths[q]}
            q_shingles = text_to_shingles(q)
            q_sig = compute_minhash_signature(q_shingles, num_hashes=nh)

            # Count banding candidates manually
            candidates = set()
            for band_id in range(nb):
                start = band_id * rpb
                end = start + rpb
                band = q_sig[start:end]
                bucket_key = hash(band.tobytes())
                candidates |= idx.band_buckets[band_id].get(bucket_key, set())
            banding_candidates_counts.append(len(candidates))

            t0 = time.perf_counter()
            results = idx.query(q_sig, q_shingles, top_k=TOP_K)
            query_times.append((time.perf_counter() - t0) * 1000)

            result_ids = {cid for cid, _ in results}
            recall = len(result_ids & gt_ids) / TOP_K if TOP_K > 0 else 0
            recalls.append(recall)

        lsh_results.append({
            "num_bands": nb,
            "rows_per_band": rpb,
            "avg_query_time_ms": round(np.mean(query_times), 2),
            "avg_recall": round(np.mean(recalls), 4),
            "avg_banding_candidates": round(np.mean(banding_candidates_counts), 1),
        })

    # ── 2c: Vary SimHash Hamming threshold ───────────────────────────────
    thresholds = [3, 5, 10, 15, 20, 30, 40, 50, 64]
    simhash_results = []

    # Build SimHash index
    simhash_idx = SimHashIndex()
    for chunk in chunks:
        fp = compute_simhash(chunk["text"])
        simhash_idx.add(chunk["chunk_id"], fp)

    for th in thresholds:
        print(f"  Testing SimHash threshold={th}...")
        simhash_idx.threshold = th

        chunks_within = []
        query_times = []
        result_counts = []

        for q in test_queries:
            q_fp = compute_simhash(q)

            t0 = time.perf_counter()
            results = simhash_idx.query(q_fp, top_k=TOP_K)
            query_times.append((time.perf_counter() - t0) * 1000)

            result_counts.append(len(results))

            # Count all chunks within threshold
            count = sum(1 for cid, fp in simhash_idx.fingerprints.items()
                        if bin(q_fp ^ fp).count("1") <= th)
            chunks_within.append(count)

        simhash_results.append({
            "threshold": th,
            "avg_query_time_ms": round(np.mean(query_times), 2),
            "avg_results_found": round(np.mean(result_counts), 1),
            "avg_chunks_within_threshold": round(np.mean(chunks_within), 1),
            "selectivity_pct": round(np.mean(chunks_within) / len(chunks) * 100, 1),
        })

    # Save all results
    all_results = {
        "minhash_num_hashes": minhash_results,
        "lsh_num_bands": lsh_results,
        "simhash_threshold": simhash_results,
    }
    with open(output_dir / "exp2_parameter_sensitivity.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Recall / Quality metrics
    # MinHash: Recall vs num_hashes
    x_mh = [r["num_hashes"] for r in minhash_results]
    axes[0][0].plot(x_mh, [r["avg_recall"] for r in minhash_results],
                    "o-", color="#4CAF50", linewidth=2, markersize=8, label="Recall@5")
    axes[0][0].set_xlabel("Number of Hash Functions")
    axes[0][0].set_ylabel("Recall@5 (vs Exact Jaccard)")
    axes[0][0].set_title("MinHash: Hash Functions vs Recall")
    axes[0][0].set_ylim(-0.05, 1.05)
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # LSH: Recall + Candidates vs bands
    x_lsh = [r["num_bands"] for r in lsh_results]
    ax_lsh = axes[0][1]
    ax_lsh.bar(x_lsh, [r["avg_recall"] for r in lsh_results],
               color="#2196F3", alpha=0.7, label="Recall@5")
    ax_lsh.set_xlabel("Number of Bands")
    ax_lsh.set_ylabel("Recall@5", color="#2196F3")
    ax_lsh.set_title("LSH: Bands vs Recall & Candidates")
    ax_lsh.set_ylim(-0.05, 1.05)

    ax_lsh2 = ax_lsh.twinx()
    ax_lsh2.plot(x_lsh, [r["avg_banding_candidates"] for r in lsh_results],
                 "D--", color="#FF5722", linewidth=2, markersize=7, label="Banding Candidates")
    ax_lsh2.set_ylabel("Avg Banding Candidates", color="#FF5722")
    ax_lsh.legend(loc="upper left")
    ax_lsh2.legend(loc="upper right")
    ax_lsh.grid(True, alpha=0.3)

    # SimHash: Selectivity vs threshold
    x_sh = [r["threshold"] for r in simhash_results]
    ax_sh = axes[0][2]
    ax_sh.plot(x_sh, [r["selectivity_pct"] for r in simhash_results],
               "s-", color="#9C27B0", linewidth=2, markersize=8, label="Selectivity %")
    ax_sh.set_xlabel("Hamming Threshold")
    ax_sh.set_ylabel("% Corpus Within Threshold")
    ax_sh.set_title("SimHash: Threshold vs Selectivity")
    ax_sh.legend()
    ax_sh.grid(True, alpha=0.3)

    # Row 2: Time metrics
    # MinHash: Build time + query time
    axes[1][0].bar(np.arange(len(x_mh)) - 0.2,
                   [r["build_time_ms"] / 1000 for r in minhash_results], 0.4,
                   label="Build Time (s)", color="#4CAF50", alpha=0.7)
    axes[1][0].bar(np.arange(len(x_mh)) + 0.2,
                   [r["avg_query_time_ms"] for r in minhash_results], 0.4,
                   label="Query Time (ms)", color="#8BC34A", alpha=0.7)
    axes[1][0].set_xlabel("Number of Hash Functions")
    axes[1][0].set_ylabel("Time")
    axes[1][0].set_title("MinHash: Hash Functions vs Time")
    axes[1][0].set_xticks(np.arange(len(x_mh)))
    axes[1][0].set_xticklabels([str(v) for v in x_mh])
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)

    # LSH: Query time vs bands
    axes[1][1].bar(x_lsh, [r["avg_query_time_ms"] for r in lsh_results],
                   color="#2196F3", alpha=0.7)
    axes[1][1].set_xlabel("Number of Bands")
    axes[1][1].set_ylabel("Avg Query Time (ms)")
    axes[1][1].set_title("LSH: Bands vs Query Time")
    axes[1][1].grid(True, alpha=0.3)

    # SimHash: Query time vs threshold
    axes[1][2].plot(x_sh, [r["avg_query_time_ms"] for r in simhash_results],
                    "s-", color="#9C27B0", linewidth=2, markersize=8)
    ax_sh3 = axes[1][2].twinx()
    ax_sh3.bar(x_sh, [r["avg_chunks_within_threshold"] for r in simhash_results],
               color="#CE93D8", alpha=0.3, width=3)
    axes[1][2].set_xlabel("Hamming Threshold")
    axes[1][2].set_ylabel("Query Time (ms)", color="#9C27B0")
    ax_sh3.set_ylabel("Chunks Within Threshold", color="#CE93D8")
    axes[1][2].set_title("SimHash: Threshold vs Time")
    axes[1][2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp2_parameter_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary
    print("\n  Parameter Sensitivity Summary:")
    print(f"    MinHash Recall@5: {[r['avg_recall'] for r in minhash_results]}")
    print(f"    LSH Recall@5:     {[r['avg_recall'] for r in lsh_results]}")
    print(f"    SimHash Select%:  {[r['selectivity_pct'] for r in simhash_results]}")
    print("  Parameter sensitivity analysis complete.")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Scalability Test
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_scalability(chunks, output_dir: Path):
    """
    Simulate larger datasets and measure how performance changes.
    Duplicate corpus 1x-10x and measure build + query time for both methods.
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
        scaled_chunks = []
        cid = 0
        for _ in range(mult):
            for chunk in chunks:
                scaled_chunks.append({**chunk, "chunk_id": cid})
                cid += 1

        num_chunks = len(scaled_chunks)

        # Build TF-IDF
        t0 = time.perf_counter()
        tfidf = build_tfidf_baseline(scaled_chunks)
        tfidf_build = (time.perf_counter() - t0) * 1000

        # Build MinHash+LSH
        from minhash_lsh import build_minhash_lsh_index
        t0 = time.perf_counter()
        minhash_idx = build_minhash_lsh_index(scaled_chunks)
        minhash_build = (time.perf_counter() - t0) * 1000

        # Query TF-IDF (average of 3 runs)
        tfidf_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            tfidf.query(test_query, top_k=TOP_K)
            tfidf_times.append((time.perf_counter() - t0) * 1000)
        tfidf_query = np.mean(tfidf_times)

        # Query MinHash+LSH (average of 3 runs)
        q_shingles = text_to_shingles(test_query)
        q_sig = compute_minhash_signature(q_shingles)
        minhash_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            minhash_idx.query(q_sig, q_shingles, top_k=TOP_K)
            minhash_times.append((time.perf_counter() - t0) * 1000)
        minhash_query = np.mean(minhash_times)

        results.append({
            "multiplier": mult,
            "num_chunks": num_chunks,
            "tfidf_build_ms": round(tfidf_build, 2),
            "minhash_build_ms": round(minhash_build, 2),
            "tfidf_query_ms": round(tfidf_query, 2),
            "minhash_query_ms": round(minhash_query, 2),
        })

        print(f"    Chunks={num_chunks}, TF-IDF build={tfidf_build:.0f}ms query={tfidf_query:.1f}ms, "
              f"LSH build={minhash_build:.0f}ms query={minhash_query:.1f}ms")

    with open(output_dir / "exp3_scalability.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = [r["num_chunks"] for r in results]

    axes[0].plot(x, [r["tfidf_build_ms"] for r in results],
                 "o-", label="TF-IDF", color="#2196F3", linewidth=2, markersize=8)
    axes[0].plot(x, [r["minhash_build_ms"] for r in results],
                 "o-", label="MinHash+LSH", color="#4CAF50", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Chunks")
    axes[0].set_ylabel("Build Time (ms)")
    axes[0].set_title("Index Build Time vs Corpus Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, [r["tfidf_query_ms"] for r in results],
                 "o-", label="TF-IDF", color="#2196F3", linewidth=2, markersize=8)
    axes[1].plot(x, [r["minhash_query_ms"] for r in results],
                 "o-", label="MinHash+LSH", color="#4CAF50", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Chunks")
    axes[1].set_ylabel("Query Time (ms)")
    axes[1].set_title("Query Time vs Corpus Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp3_scalability.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Scalability test complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Run all
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_experiments(engine, chunks, output_dir: Path = PROCESSED_DIR / "experiments"):
    """Run all three required experiments."""
    print("\n" + "#"*60)
    print("# RUNNING ALL EXPERIMENTS")
    print("#"*60)

    experiment_exact_vs_approximate(engine, output_dir)
    experiment_parameter_sensitivity(chunks, output_dir)
    experiment_scalability(chunks, output_dir)

    print(f"\n[Experiments] All results saved to {output_dir}")
