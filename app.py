"""
Streamlit UI for the Scalable Academic Policy QA System.
Premium, showcase-ready dashboard with polished design.
"""
import sys
import json
import time
from pathlib import Path

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CHUNKS_FILE, INDEX_DIR, TOP_K, GEMINI_API_KEY
from query_engine import QueryEngine
from answer_gen import generate_answer


# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NUST Policy QA | Scalable Retrieval System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base ──────────────────────────────────────────────── */
    :root {
        --bg-primary: #0a0a1a;
        --bg-secondary: #111127;
        --bg-card: rgba(17, 17, 39, 0.7);
        --bg-card-hover: rgba(25, 25, 55, 0.85);
        --border-subtle: rgba(99, 102, 241, 0.15);
        --border-active: rgba(129, 140, 248, 0.4);
        --accent-1: #818cf8;
        --accent-2: #a78bfa;
        --accent-3: #f472b6;
        --accent-4: #34d399;
        --accent-5: #38bdf8;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --glow-purple: rgba(139, 92, 246, 0.15);
        --glow-green: rgba(52, 211, 153, 0.12);
    }

    .main { font-family: 'Inter', -apple-system, sans-serif; }
    .stApp { background: var(--bg-primary); }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Sidebar ────────────────────────────────────────────── */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d24 0%, #111130 100%);
        border-right: 1px solid var(--border-subtle);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-secondary);
        font-size: 0.88rem;
    }

    /* ── Hero Header ───────────────────────────────────────── */
    .hero {
        position: relative;
        background: linear-gradient(135deg,
            rgba(99, 102, 241, 0.06) 0%,
            rgba(168, 85, 247, 0.08) 40%,
            rgba(244, 114, 182, 0.06) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.5rem;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 40%, rgba(99,102,241,0.08) 0%, transparent 50%),
                    radial-gradient(circle at 70% 60%, rgba(244,114,182,0.06) 0%, transparent 50%);
        animation: pulse-bg 8s ease-in-out infinite alternate;
    }
    @keyframes pulse-bg {
        0% { transform: scale(1) rotate(0deg); opacity: 1; }
        100% { transform: scale(1.1) rotate(3deg); opacity: 0.7; }
    }
    .hero-content { position: relative; z-index: 1; }
    .hero h1 {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 30%, #f472b6 70%, #fb923c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 0.5rem 0;
    }
    .hero p {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.5;
    }
    .hero-badges {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid var(--border-subtle);
        color: var(--accent-1);
        padding: 0.3rem 0.75rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    /* ── Answer Box ─────────────────────────────────────────── */
    .answer-container {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.04), rgba(16, 185, 129, 0.06));
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .answer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #34d399, #10b981, #059669);
        border-radius: 4px 0 0 4px;
    }
    .answer-container .answer-label {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(52, 211, 153, 0.1);
        color: #34d399;
        padding: 0.25rem 0.7rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }
    .answer-container .answer-text {
        color: var(--text-primary);
        font-size: 0.95rem;
        line-height: 1.8;
    }

    /* ── Metric Cards ───────────────────────────────────────── */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0;
    }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: var(--border-active);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .metric-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.4rem;
    }

    /* ── Result Cards ──────────────────────────────────────── */
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .result-card:hover {
        border-color: var(--border-active);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
    }
    .result-card .rank-badge {
        position: absolute;
        top: 0;
        right: 0;
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.3rem 0.8rem;
        border-radius: 0 14px 0 10px;
    }
    .result-card .chunk-meta {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 0.75rem;
    }
    .chunk-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    .chunk-tag.page {
        background: rgba(56, 189, 248, 0.1);
        color: var(--accent-5);
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .chunk-tag.section {
        background: rgba(167, 139, 250, 0.1);
        color: var(--accent-2);
        border: 1px solid rgba(167, 139, 250, 0.2);
    }
    .chunk-tag.score {
        background: rgba(52, 211, 153, 0.1);
        color: var(--accent-4);
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .result-card .chunk-text {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.7;
        max-height: 120px;
        overflow: hidden;
        mask-image: linear-gradient(180deg, #000 60%, transparent 100%);
        -webkit-mask-image: linear-gradient(180deg, #000 60%, transparent 100%);
    }

    /* ── Score Breakdown Bar ───────────────────────────────── */
    .score-bar-container {
        display: flex;
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        margin: 0.75rem 0 0.5rem;
        background: rgba(255,255,255,0.03);
    }
    .score-bar-segment {
        height: 100%;
        transition: width 0.5s ease;
    }

    /* ── Section Headers ───────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0 0.75rem;
    }
    .section-header .icon {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
        border-radius: 8px;
        font-size: 0.9rem;
    }
    .section-header h3 {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .section-header .count-badge {
        background: rgba(99,102,241,0.1);
        color: var(--accent-1);
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 100px;
        border: 1px solid var(--border-subtle);
    }

    /* ── Overlap Indicator ─────────────────────────────────── */
    .overlap-bar {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .overlap-bar .overlap-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .overlap-bar .overlap-label span {
        color: var(--text-secondary);
        font-size: 0.82rem;
    }
    .overlap-bar .overlap-label strong {
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    .overlap-track {
        height: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
        overflow: hidden;
    }
    .overlap-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Welcome ─────────────────────────────────────────── */
    .welcome {
        text-align: center;
        padding: 4rem 2rem;
    }
    .welcome h2 {
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .welcome p { color: var(--text-muted); line-height: 1.7; }
    .welcome .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 2rem;
        text-align: left;
    }
    .welcome .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.2rem;
        transition: all 0.3s;
    }
    .welcome .feature-card:hover {
        border-color: var(--border-active);
        transform: translateY(-2px);
    }
    .welcome .feature-card h4 {
        color: var(--text-primary);
        font-size: 0.9rem;
        margin: 0.5rem 0 0.3rem;
    }
    .welcome .feature-card p {
        font-size: 0.78rem;
        margin: 0;
    }

    /* ── Tab Styling ──────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 0.25rem;
        border: 1px solid var(--border-subtle);
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        color: var(--text-muted);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.15) !important;
        color: var(--accent-1) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Expander ────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 0.88rem !important;
    }

    /* ── Method comparison header ─────────────────────────── */
    .method-header {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .method-header .method-name {
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 600;
    }
    .method-header .method-stats {
        display: flex;
        gap: 1rem;
    }
    .method-header .method-stat {
        color: var(--text-muted);
        font-size: 0.75rem;
    }
    .method-header .method-stat strong {
        color: var(--accent-1);
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Pipeline vis ─────────────────────────────────────── */
    .pipeline {
        display: flex;
        align-items: center;
        gap: 0;
        padding: 0.5rem 0;
        overflow-x: auto;
    }
    .pipeline-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.3rem;
        min-width: 80px;
    }
    .pipeline-step .step-icon {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        font-size: 1rem;
    }
    .pipeline-step .step-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        font-weight: 500;
        text-align: center;
    }
    .pipeline-arrow {
        color: var(--text-muted);
        font-size: 0.7rem;
        padding: 0 0.2rem;
        opacity: 0.4;
    }

    /* ── Animations ──────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Engine (cached) ─────────────────────────────────────────────
@st.cache_resource
def get_engine():
    engine = QueryEngine()
    engine.load_chunks()
    try:
        engine.load_indices()
    except FileNotFoundError:
        engine.build_all_indices()
    return engine


@st.cache_data
def get_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-size: 2rem;">🎓</span>
        <p style="color: #818cf8; font-size: 0.85rem; font-weight: 600; margin: 0.3rem 0 0; letter-spacing: 0.05em;">
            NUST POLICY QA
        </p>
        <p style="color: #64748b; font-size: 0.68rem; font-weight: 400; margin: 0;">
            Scalable Academic Retrieval
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Page navigation
    st.markdown("<p style='color:#94a3b8;font-size:0.75rem;font-weight:600;letter-spacing:0.05em;margin-bottom:0.5rem;'>NAVIGATION</p>", unsafe_allow_html=True)
    current_page = st.radio("Page", ["🔍 QA System", "📈 Experiments"], label_visibility="collapsed")

    st.markdown("---")

    st.markdown("<p style='color:#94a3b8;font-size:0.75rem;font-weight:600;letter-spacing:0.05em;margin-bottom:0.5rem;'>RETRIEVAL METHOD</p>", unsafe_allow_html=True)
    method = st.selectbox(
        "Retrieval Method",
        ["Both (Side-by-Side)", "LSH (Approximate)", "TF-IDF (Exact Baseline)"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<p style='color:#94a3b8;font-size:0.75rem;font-weight:600;letter-spacing:0.05em;margin:1rem 0 0.5rem;'>TOP-K RESULTS</p>", unsafe_allow_html=True)
    top_k = st.slider("Top-K", min_value=1, max_value=15, value=5, label_visibility="collapsed")

    st.markdown("---")

    st.markdown("<p style='color:#94a3b8;font-size:0.75rem;font-weight:600;letter-spacing:0.05em;margin-bottom:0.5rem;'>ANSWER GENERATION</p>", unsafe_allow_html=True)
    answer_mode = st.radio(
        "Mode",
        ["Gemini LLM", "Extractive (No API)", "Disabled"],
        index=0,
        label_visibility="collapsed",
    )

    use_llm = answer_mode == "Gemini LLM"
    api_key = ""
    if use_llm:
        api_key = st.text_input("Gemini API Key", type="password", value=GEMINI_API_KEY,
                                 label_visibility="collapsed",
                                 placeholder="Enter API key...")

    st.markdown("---")

    # System stats
    try:
        chunks = get_chunks()
        word_counts = [c['word_count'] for c in chunks]
        total_words = sum(word_counts)

        st.markdown(f"""
        <div style="padding:0.5rem 0;">
            <p style="color:#64748b;font-size:0.68rem;font-weight:600;letter-spacing:0.08em;margin-bottom:0.75rem;">CORPUS STATISTICS</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.12);border-radius:8px;padding:0.6rem;text-align:center;">
                    <p style="font-family:'JetBrains Mono';font-size:1.1rem;font-weight:700;color:#818cf8;margin:0;">{len(chunks)}</p>
                    <p style="font-size:0.6rem;color:#64748b;margin:0;margin-top:0.15rem;">CHUNKS</p>
                </div>
                <div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.12);border-radius:8px;padding:0.6rem;text-align:center;">
                    <p style="font-family:'JetBrains Mono';font-size:1.1rem;font-weight:700;color:#34d399;margin:0;">{total_words:,}</p>
                    <p style="font-size:0.6rem;color:#64748b;margin:0;margin-top:0.15rem;">WORDS</p>
                </div>
                <div style="background:rgba(244,114,182,0.06);border:1px solid rgba(244,114,182,0.12);border-radius:8px;padding:0.6rem;text-align:center;">
                    <p style="font-family:'JetBrains Mono';font-size:1.1rem;font-weight:700;color:#f472b6;margin:0;">{np.mean(word_counts):.0f}</p>
                    <p style="font-size:0.6rem;color:#64748b;margin:0;margin-top:0.15rem;">AVG SIZE</p>
                </div>
                <div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.12);border-radius:8px;padding:0.6rem;text-align:center;">
                    <p style="font-family:'JetBrains Mono';font-size:1.1rem;font-weight:700;color:#38bdf8;margin:0;">118</p>
                    <p style="font-size:0.6rem;color:#64748b;margin:0;margin-top:0.15rem;">PAGES</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("---")

    # Sample queries
    st.markdown("<p style='color:#64748b;font-size:0.68rem;font-weight:600;letter-spacing:0.08em;margin-bottom:0.5rem;'>SAMPLE QUERIES</p>", unsafe_allow_html=True)
    sample_queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
        "What is the grading system?",
        "Can a student change their program?",
        "What is the fee refund policy?",
        "What is the policy on plagiarism?",
    ]
    selected_sample = st.selectbox("Pick a query", [""] + sample_queries, label_visibility="collapsed")


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <h1>Scalable Academic Policy QA</h1>
        <p>Intelligent retrieval over the NUST Undergraduate Handbook using Big Data techniques — MinHash+LSH, SimHash, and TF-IDF with recommendation-based re-ranking.</p>
        <div class="hero-badges">
            <span class="hero-badge">⚡ MinHash + LSH</span>
            <span class="hero-badge">🔏 SimHash</span>
            <span class="hero-badge">📊 TF-IDF Baseline</span>
            <span class="hero-badge">🏆 Re-ranking Extension</span>
            <span class="hero-badge">🤖 Gemini LLM</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Pipeline visualization
st.markdown("""
<div class="pipeline" style="justify-content:center;margin-bottom:1rem;">
    <div class="pipeline-step">
        <div class="step-icon" style="background:rgba(56,189,248,0.12);color:#38bdf8;">📄</div>
        <span class="step-label">PDF Input</span>
    </div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step">
        <div class="step-icon" style="background:rgba(167,139,250,0.12);color:#a78bfa;">✂️</div>
        <span class="step-label">Chunking</span>
    </div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step">
        <div class="step-icon" style="background:rgba(244,114,182,0.12);color:#f472b6;">#️⃣</div>
        <span class="step-label">Hashing</span>
    </div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step">
        <div class="step-icon" style="background:rgba(52,211,153,0.12);color:#34d399;">🔍</div>
        <span class="step-label">Retrieval</span>
    </div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step">
        <div class="step-icon" style="background:rgba(251,146,60,0.12);color:#fb923c;">🏆</div>
        <span class="step-label">Re-rank</span>
    </div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step">
        <div class="step-icon" style="background:rgba(99,102,241,0.12);color:#818cf8;">💬</div>
        <span class="step-label">Answer</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Page Routing ────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent / "data" / "processed" / "experiments"

if current_page == "📈 Experiments":
    # ═══════════════════════════════════════════════════════════════════════
    #  EXPERIMENTS PAGE
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("""<div style="text-align:center;margin-bottom:1.5rem;">
<h2 style="background:linear-gradient(135deg,#818cf8,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.8rem;font-weight:700;">Experimental Analysis</h2>
<p style="color:#64748b;font-size:0.9rem;">Three required experiments evaluating retrieval quality, parameter sensitivity, and scalability.</p>
</div>""", unsafe_allow_html=True)

    exp1_img = EXPERIMENTS_DIR / "exp1_exact_vs_approx.png"
    exp2_img = EXPERIMENTS_DIR / "exp2_parameter_sensitivity.png"
    exp3_img = EXPERIMENTS_DIR / "exp3_scalability.png"
    exp1_json = EXPERIMENTS_DIR / "exp1_exact_vs_approx.json"
    exp3_json = EXPERIMENTS_DIR / "exp3_scalability.json"

    if not exp1_img.exists():
        st.warning("Experiment results not found. Run experiments first: `python -c \"...\"` or use the Run Experiments button below.")
        if st.button("Run All Experiments"):
            with st.spinner("Running experiments (this may take a few minutes)..."):
                from experiments import run_all_experiments
                engine = get_engine()
                chunks = get_chunks()
                run_all_experiments(engine, chunks)
            st.rerun()
    else:
        # ── Experiment 1: Exact vs Approximate ────────────────────────
        st.markdown("""<div style="background:rgba(17,17,39,0.7);border:1px solid rgba(99,102,241,0.15);border-radius:14px;padding:1.5rem;margin-bottom:1rem;">
<h3 style="color:#f1f5f9;font-size:1.1rem;margin:0 0 0.3rem;">Experiment 1: Exact vs Approximate Retrieval</h3>
<p style="color:#64748b;font-size:0.82rem;margin:0;">Compares LSH (approximate) against TF-IDF (exact) across 15 sample queries — evaluating latency, memory usage, and result overlap.</p>
</div>""", unsafe_allow_html=True)

        st.image(str(exp1_img), use_container_width=True)

        # Show summary stats
        if exp1_json.exists():
            exp1_data = json.loads(exp1_json.read_text(encoding="utf-8"))
            avg_lsh_time = np.mean([q["lsh_time_ms"] for q in exp1_data])
            avg_tfidf_time = np.mean([q["tfidf_time_ms"] for q in exp1_data])
            avg_overlap = np.mean([q["overlap_ratio"] for q in exp1_data])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg LSH Time", f"{avg_lsh_time:.1f} ms")
            c2.metric("Avg TF-IDF Time", f"{avg_tfidf_time:.1f} ms")
            c3.metric("Avg Overlap", f"{avg_overlap:.0%}")
            c4.metric("Queries Tested", f"{len(exp1_data)}")

        st.markdown("---")

        # ── Experiment 2: Parameter Sensitivity ───────────────────────
        st.markdown("""<div style="background:rgba(17,17,39,0.7);border:1px solid rgba(99,102,241,0.15);border-radius:14px;padding:1.5rem;margin-bottom:1rem;">
<h3 style="color:#f1f5f9;font-size:1.1rem;margin:0 0 0.3rem;">Experiment 2: Parameter Sensitivity Analysis</h3>
<p style="color:#64748b;font-size:0.82rem;margin:0;">Analyzes impact of MinHash hash functions, LSH band config, and SimHash threshold on Recall@5 (vs exact Jaccard ground truth), build time, and query latency.</p>
</div>""", unsafe_allow_html=True)

        st.image(str(exp2_img), use_container_width=True)

        st.markdown("""<div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.15);border-radius:10px;padding:1rem;margin:0.5rem 0;">
<p style="color:#94a3b8;font-size:0.82rem;margin:0;line-height:1.6;"><strong style="color:#34d399;">Key Findings:</strong> MinHash Recall@5 improves from 0.80 to 0.88 as hash functions increase (16→256), with build time scaling linearly. LSH banding finds 0 candidates for short queries — the brute-force fallback maintains 0.88 recall regardless of band config. SimHash selectivity jumps from 0% to 95% as Hamming threshold reaches 64, showing the sensitivity/selectivity tradeoff.</p>
</div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Experiment 3: Scalability ─────────────────────────────────
        st.markdown("""<div style="background:rgba(17,17,39,0.7);border:1px solid rgba(99,102,241,0.15);border-radius:14px;padding:1.5rem;margin-bottom:1rem;">
<h3 style="color:#f1f5f9;font-size:1.1rem;margin:0 0 0.3rem;">Experiment 3: Scalability Test</h3>
<p style="color:#64748b;font-size:0.82rem;margin:0;">Measures index build time and query latency as corpus size grows from 88 to 880 chunks (1x to 10x), comparing TF-IDF vs MinHash+LSH scaling behavior.</p>
</div>""", unsafe_allow_html=True)

        st.image(str(exp3_img), use_container_width=True)

        # Show scalability table
        if exp3_json.exists():
            exp3_data = json.loads(exp3_json.read_text(encoding="utf-8"))
            st.markdown("<p style='color:#94a3b8;font-size:0.82rem;font-weight:600;margin-top:0.5rem;'>Scalability Data:</p>", unsafe_allow_html=True)

            table_data = {
                "Corpus Size": [f"{d['multiplier']}x ({d['num_chunks']} chunks)" for d in exp3_data],
                "TF-IDF Build (ms)": [f"{d['tfidf_build_ms']:.0f}" for d in exp3_data],
                "MinHash Build (ms)": [f"{d['minhash_build_ms']:.0f}" for d in exp3_data],
                "TF-IDF Query (ms)": [f"{d['tfidf_query_ms']:.1f}" for d in exp3_data],
                "MinHash Query (ms)": [f"{d['minhash_query_ms']:.1f}" for d in exp3_data],
            }
            st.dataframe(table_data, use_container_width=True)

        st.markdown("""<div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.15);border-radius:10px;padding:1rem;margin:0.5rem 0;">
<p style="color:#94a3b8;font-size:0.82rem;margin:0;line-height:1.6;"><strong style="color:#38bdf8;">Key Finding:</strong> At 10x corpus size (880 chunks), LSH query time (29.8ms) is faster than TF-IDF (19.4ms) at small scale but both methods scale similarly for query. The critical difference is in build time: MinHash+LSH build grows steeply (429s at 10x) while TF-IDF remains lightweight (1.3s), reflecting the cost of computing 128 hash functions per shingle.</p>
</div>""", unsafe_allow_html=True)

else:
    # ═══════════════════════════════════════════════════════════════════════
    #  QA SYSTEM PAGE
    # ═══════════════════════════════════════════════════════════════════════

    # ─── Main ────────────────────────────────────────────────────────────────────
    engine = get_engine()

    query = st.text_input(
        "query_input",
        value=selected_sample if selected_sample else "",
        placeholder="Ask a question about NUST academic policies...",
        label_visibility="collapsed",
    )

    if query:
        method_key = {
            "Both (Side-by-Side)": "both",
            "LSH (Approximate)": "lsh",
            "TF-IDF (Exact Baseline)": "tfidf",
        }[method]

        with st.spinner(""):
            t0 = time.time()
            search_results = engine.search(query, method=method_key, top_k=top_k)
            total_time = (time.time() - t0) * 1000

        # ─── Answer ──────────────────────────────────────────────────────
        if answer_mode != "Disabled":
            answer_chunks = []
            if "lsh" in search_results:
                answer_chunks = search_results["lsh"]["results"]
            elif "tfidf" in search_results:
                answer_chunks = search_results["tfidf"]["results"]

            if answer_chunks:
                answer_data = generate_answer(
                    question=query,
                    retrieved_chunks=answer_chunks,
                    api_key=api_key if use_llm else "",
                    use_llm=use_llm,
                )

                method_label = "Gemini LLM" if answer_data.get("method") == "gemini" else "Extractive"
                if answer_data.get("llm_error"):
                    st.warning(f"Gemini API failed, using extractive fallback: {answer_data['llm_error'][:150]}")

                st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(52,211,153,0.04),rgba(16,185,129,0.06));border:1px solid rgba(52,211,153,0.2);border-radius:16px;padding:1.75rem;margin:1rem 0;border-left:4px solid #34d399;">
<div style="display:inline-flex;align-items:center;gap:0.4rem;background:rgba(52,211,153,0.1);color:#34d399;padding:0.25rem 0.7rem;border-radius:100px;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1rem;">{method_label} Answer</div>
<div style="color:#f1f5f9;font-size:0.95rem;line-height:1.8;">{answer_data['answer']}</div>
</div>""", unsafe_allow_html=True)

        # ─── Performance Metrics ─────────────────────────────────────────
        lsh_data = search_results.get("lsh", {})
        tfidf_data = search_results.get("tfidf", {})

        metric_cols = st.columns(6)
        col_idx = 0
        if lsh_data:
            metric_cols[col_idx].metric("LSH Time", f"{lsh_data['time_ms']} ms")
            col_idx += 1
            metric_cols[col_idx].metric("LSH Memory", f"{lsh_data['peak_memory_kb']} KB")
            col_idx += 1
        if tfidf_data:
            metric_cols[col_idx].metric("TF-IDF Time", f"{tfidf_data['time_ms']} ms")
            col_idx += 1
            metric_cols[col_idx].metric("TF-IDF Memory", f"{tfidf_data['peak_memory_kb']} KB")
            col_idx += 1
        if lsh_data:
            metric_cols[col_idx].metric("LSH Candidates", f"{lsh_data.get('num_candidates', '-')}")
            col_idx += 1
        metric_cols[min(col_idx, 5)].metric("Total", f"{total_time:.0f} ms")

        # ─── Results ─────────────────────────────────────────────────────
        def render_result_card(result: dict, rank: int):
            """Render a single result card."""
            import html as html_mod
            score = result.get("score", 0)
            breakdown = result.get("score_breakdown", {})
            section = result.get("section_title", "")
            page = result.get("start_page", "?")
            raw_text = result.get("text", "")[:400]
            text = html_mod.escape(raw_text)

            section_tag = f'<span class="chunk-tag section">📑 {html_mod.escape(section[:40])}</span>' if section else ""

            legend_html = ""
            if len(breakdown) > 1:
                colors = {"jaccard": "#818cf8", "cosine": "#38bdf8", "hamming": "#f472b6", "section": "#fb923c", "combined": "#34d399"}
                parts = []
                for k, v in breakdown.items():
                    if k == "combined":
                        continue
                    c = colors.get(k, "#666")
                    parts.append(f'<span style="color:{c};font-size:0.7rem;font-family:JetBrains Mono,monospace;">● {k}: {v:.3f}</span>')
                legend_html = '<div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin:0.5rem 0 0;">' + " ".join(parts) + '</div>'

            st.markdown(f"""<div style="background:rgba(17,17,39,0.7);border:1px solid rgba(99,102,241,0.15);border-radius:14px;padding:1.4rem;margin-bottom:0.75rem;position:relative;">
<div style="position:absolute;top:0;right:0;background:linear-gradient(135deg,#818cf8,#a78bfa);color:white;font-size:0.7rem;font-weight:700;padding:0.3rem 0.8rem;border-radius:0 14px 0 10px;">#{rank}</div>
<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.5rem;">
<span class="chunk-tag page">📄 Page {page}</span>
{section_tag}
<span class="chunk-tag score">⭐ {score:.4f}</span>
</div>
{legend_html}
<div style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin-top:0.5rem;">{text}...</div>
</div>""", unsafe_allow_html=True)

        def render_results_section(data: dict, label: str):
            if not data or not data.get("results"):
                st.info(f"No {label} results found.")
                return

            method_name = data.get("method", label)
            time_ms = data.get("time_ms", "?")
            mem_kb = data.get("peak_memory_kb", "?")
            candidates = data.get("num_candidates", "")
            stats = f'Time: <strong>{time_ms}ms</strong> · Memory: <strong>{mem_kb}KB</strong>'
            if candidates:
                stats += f' · Candidates: <strong>{candidates}</strong>'

            st.markdown(f"""<div style="background:rgba(17,17,39,0.7);border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:0.8rem 1.2rem;margin-bottom:0.75rem;display:flex;align-items:center;justify-content:space-between;">
<span style="color:#f1f5f9;font-size:0.95rem;font-weight:600;">{method_name}</span>
<span style="color:#64748b;font-size:0.75rem;">{stats}</span>
</div>""", unsafe_allow_html=True)

            for i, r in enumerate(data["results"]):
                render_result_card(r, i + 1)

            for i, r in enumerate(data["results"]):
                with st.expander(f"Full text — Chunk #{r.get('chunk_id', '?')} (Page {r.get('start_page', '?')})"):
                    st.text(r.get("text", ""))

        if method_key == "both":
            if lsh_data and tfidf_data:
                lsh_ids = {r["chunk_id"] for r in lsh_data.get("results", [])}
                tfidf_ids = {r["chunk_id"] for r in tfidf_data.get("results", [])}
                overlap = len(lsh_ids & tfidf_ids)
                pct = (overlap / top_k * 100) if top_k > 0 else 0
                color = "#34d399" if pct >= 60 else "#fb923c" if pct >= 30 else "#ef4444"

                st.markdown(f"""<div style="background:rgba(17,17,39,0.7);border:1px solid rgba(99,102,241,0.15);border-radius:14px;padding:1.2rem;margin:1rem 0;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
<span style="color:#94a3b8;font-size:0.82rem;">Result Overlap (LSH vs TF-IDF)</span>
<strong style="color:{color};font-family:JetBrains Mono,monospace;">{overlap}/{top_k} chunks · {pct:.0f}%</strong>
</div>
<div style="height:8px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;">
<div style="height:100%;width:{pct}%;background:linear-gradient(90deg,{color},{color}88);border-radius:4px;"></div>
</div>
</div>""", unsafe_allow_html=True)

            tab_lsh, tab_tfidf = st.tabs(["⚡ LSH Results (Approximate)", "📊 TF-IDF Results (Exact)"])
            with tab_lsh:
                render_results_section(lsh_data, "LSH")
            with tab_tfidf:
                render_results_section(tfidf_data, "TF-IDF")
        else:
            key = "lsh" if method_key == "lsh" else "tfidf"
            data = search_results.get(key, {})
            render_results_section(data, key.upper())

    else:
        # ─── Welcome Screen ──────────────────────────────────────────────
        st.markdown("""<div style="text-align:center;padding:3rem 2rem;">
<h2 style="background:linear-gradient(135deg,#818cf8,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.8rem;font-weight:700;">Ask anything about NUST academic policies</h2>
<p style="color:#64748b;line-height:1.7;">Type a question above or pick a sample from the sidebar. The system will search through the Undergraduate Handbook using both approximate and exact retrieval methods.</p>
</div>""", unsafe_allow_html=True)

        features = [
            ("⚡", "MinHash + LSH", "Approximate near-neighbor search using locality-sensitive hashing with configurable bands and hash functions."),
            ("🔏", "SimHash", "128-bit fingerprinting with Hamming distance for fast near-duplicate detection."),
            ("📊", "TF-IDF Baseline", "Exact cosine similarity retrieval for ground-truth comparison against LSH methods."),
            ("🏆", "Re-ranking", "Multi-signal recommendation engine combining Jaccard, cosine, Hamming and section importance scores."),
            ("🤖", "Gemini LLM", "Grounded answer generation via Google Gemini with automatic extractive fallback."),
            ("📈", "Experiments", "Built-in parameter sensitivity, scalability, and exact vs. approximate comparison analysis."),
        ]

        cols = st.columns(3)
        for idx, (icon, title, desc) in enumerate(features):
            with cols[idx % 3]:
                st.markdown(f"""<div style="background:rgba(17,17,39,0.7);border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:1.2rem;margin-bottom:0.75rem;">
<div style="font-size:1.5rem;">{icon}</div>
<h4 style="color:#f1f5f9;font-size:0.9rem;margin:0.5rem 0 0.3rem;">{title}</h4>
<p style="color:#64748b;font-size:0.78rem;margin:0;line-height:1.5;">{desc}</p>
</div>""", unsafe_allow_html=True)

