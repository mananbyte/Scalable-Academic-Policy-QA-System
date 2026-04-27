"""
Streamlit UI for the Scalable Academic Policy QA System.

Features:
- Query input
- Method selector (LSH / TF-IDF / Both)
- Answer panel with LLM-generated response
- Evidence panel with top-k chunks and source references
- Performance metrics and side-by-side comparison
"""
import sys
import json
from pathlib import Path

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CHUNKS_FILE, INDEX_DIR, TOP_K
from query_engine import QueryEngine
from answer_gen import generate_answer


# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NUST Academic Policy QA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .header-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .header-container h1 {
        background: linear-gradient(135deg, #818cf8, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-container p {
        color: #cbd5e1;
        font-size: 1rem;
    }

    .result-card {
        background: rgba(30, 27, 75, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        transition: border-color 0.3s;
    }
    .result-card:hover {
        border-color: rgba(99, 102, 241, 0.6);
    }

    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .metric-card {
        background: rgba(30, 27, 75, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card h3 {
        color: #a78bfa;
        font-size: 1.5rem;
        margin: 0;
    }
    .metric-card p {
        color: #94a3b8;
        font-size: 0.8rem;
        margin: 0;
    }

    .answer-box {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #e2e8f0;
        line-height: 1.7;
    }

    .source-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a78bfa;
        padding: 0.15rem 0.5rem;
        border-radius: 8px;
        font-size: 0.75rem;
        margin-right: 0.5rem;
    }

    .sidebar .stSelectbox label,
    .sidebar .stSlider label {
        color: #cbd5e1;
    }

    div[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Engine (cached) ─────────────────────────────────────────────
@st.cache_resource
def get_engine():
    """Load the query engine (cached across reruns)."""
    engine = QueryEngine()
    engine.load_chunks()
    try:
        engine.load_indices()
    except FileNotFoundError:
        st.warning("⚠️ Indices not found. Building from scratch (this may take a minute)...")
        engine.build_all_indices()
    return engine


@st.cache_data
def get_chunks():
    """Load chunk list for display purposes."""
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1>🎓 NUST Academic Policy QA System</h1>
    <p>Scalable retrieval using MinHash+LSH, SimHash, and TF-IDF with Recommendation-based Re-ranking</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    method = st.selectbox(
        "Retrieval Method",
        ["Both (Side-by-Side)", "LSH (Approximate)", "TF-IDF (Exact Baseline)"],
        index=0,
    )

    top_k = st.slider("Top-K Results", min_value=1, max_value=20, value=TOP_K)

    use_llm = st.checkbox("Generate Answer (Gemini API)", value=True)

    if use_llm:
        api_key = st.text_input("Gemini API Key", type="password",
                                 help="Enter your Google Gemini API key")
    else:
        api_key = ""

    st.markdown("---")
    st.markdown("### 📊 System Info")
    try:
        chunks = get_chunks()
        st.markdown(f"**Chunks indexed:** {len(chunks)}")
        word_counts = [c['word_count'] for c in chunks]
        st.markdown(f"**Avg chunk size:** {np.mean(word_counts):.0f} words")
        st.markdown(f"**Source:** UG Handbook")
    except Exception:
        st.markdown("*Loading...*")

    st.markdown("---")
    st.markdown("### 💡 Sample Queries")
    sample_queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
        "What are the credit hours for graduation?",
    ]
    selected_sample = st.selectbox("Quick pick:", [""] + sample_queries)


# ─── Main Content ────────────────────────────────────────────────────────────
engine = get_engine()

# Query input
query = st.text_input(
    "🔍 Ask a question about NUST academic policies:",
    value=selected_sample if selected_sample else "",
    placeholder="e.g., What is the minimum GPA requirement?",
)

if query:
    method_key = {
        "Both (Side-by-Side)": "both",
        "LSH (Approximate)": "lsh",
        "TF-IDF (Exact Baseline)": "tfidf",
    }[method]

    with st.spinner("🔎 Searching..."):
        search_results = engine.search(query, method=method_key, top_k=top_k)

    # ─── Answer Generation ────────────────────────────────────────────────
    answer_chunks = []
    if "lsh" in search_results:
        answer_chunks = search_results["lsh"]["results"]
    elif "tfidf" in search_results:
        answer_chunks = search_results["tfidf"]["results"]

    if answer_chunks:
        with st.spinner("Generating answer..."):
            answer_data = generate_answer(
                question=query,
                retrieved_chunks=answer_chunks,
                api_key=api_key if use_llm else "",
                use_llm=use_llm and bool(api_key),
            )

        st.markdown("### 💬 Answer")

        # Show which method was used
        if answer_data.get("method") == "gemini":
            st.caption("🤖 Generated by Google Gemini")
        else:
            st.caption("📝 Extractive answer (from retrieved chunks)")
            if answer_data.get("llm_error"):
                st.warning(f"Gemini API failed — using extractive fallback. Error: {answer_data['llm_error'][:200]}")

        st.markdown(answer_data["answer"])

    # ─── Performance Metrics ─────────────────────────────────────────────
    st.markdown("### 📊 Performance Metrics")
    metric_cols = st.columns(4)

    if "lsh" in search_results:
        with metric_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{search_results['lsh']['time_ms']}</h3>
                <p>LSH Time (ms)</p>
            </div>
            """, unsafe_allow_html=True)
        with metric_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{search_results['lsh']['peak_memory_kb']}</h3>
                <p>LSH Memory (KB)</p>
            </div>
            """, unsafe_allow_html=True)

    if "tfidf" in search_results:
        with metric_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{search_results['tfidf']['time_ms']}</h3>
                <p>TF-IDF Time (ms)</p>
            </div>
            """, unsafe_allow_html=True)
        with metric_cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{search_results['tfidf']['peak_memory_kb']}</h3>
                <p>TF-IDF Memory (KB)</p>
            </div>
            """, unsafe_allow_html=True)

    # ─── Results Display ─────────────────────────────────────────────────
    def display_results(results_data: dict, label: str):
        """Display retrieval results in styled cards."""
        st.markdown(f"### 📄 {label} — {results_data['method']}")
        st.caption(f"⏱️ {results_data['time_ms']}ms | 💾 {results_data['peak_memory_kb']}KB")

        for i, result in enumerate(results_data["results"]):
            score = result.get("score", 0)
            breakdown = result.get("score_breakdown", {})
            section = result.get("section_title", "Unknown")
            page = result.get("start_page", "?")

            with st.expander(
                f"**#{i+1}** | Score: {score:.4f} | Page {page} | {section[:50]}",
                expanded=(i < 2),
            ):
                # Score breakdown
                if len(breakdown) > 1:
                    cols = st.columns(len(breakdown))
                    for j, (key, val) in enumerate(breakdown.items()):
                        cols[j].metric(key.capitalize(), f"{val:.4f}")

                # Chunk text
                st.markdown(f"**Source:** Page {page} — {section}")
                st.markdown(f'<div class="result-card">{result["text"][:800]}</div>',
                            unsafe_allow_html=True)

    if method_key == "both":
        col1, col2 = st.columns(2)
        with col1:
            if "lsh" in search_results:
                display_results(search_results["lsh"], "LSH Results")
        with col2:
            if "tfidf" in search_results:
                display_results(search_results["tfidf"], "TF-IDF Results")

        # Overlap analysis
        if "lsh" in search_results and "tfidf" in search_results:
            lsh_ids = {r["chunk_id"] for r in search_results["lsh"]["results"]}
            tfidf_ids = {r["chunk_id"] for r in search_results["tfidf"]["results"]}
            overlap = len(lsh_ids & tfidf_ids)
            st.markdown("---")
            st.markdown(f"### 🔗 Result Overlap: **{overlap}/{top_k}** chunks in common "
                        f"({overlap/top_k*100:.0f}%)")
    else:
        key = "lsh" if method_key == "lsh" else "tfidf"
        if key in search_results:
            display_results(search_results[key], "Results")

else:
    # Welcome message
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #94a3b8;">
        <h2 style="color: #a78bfa;">Welcome!</h2>
        <p>Ask any question about NUST academic policies, and the system will retrieve
        relevant information from the Undergraduate Handbook using both approximate (LSH)
        and exact (TF-IDF) methods.</p>
        <p style="font-size: 0.85rem; margin-top: 1rem;">
            💡 Try picking a sample query from the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
