<div align="center">

<h1>Scalable Academic Policy QA System</h1>

<p>
  Natural-language question answering over the NUST Undergraduate Handbook<br/>
  via MinHash&nbsp;+&nbsp;LSH · SimHash · TF-IDF · Gemini LLM
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Gemini-API-4285F4?style=flat-square&logo=google&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/BDA-Project-6C63FF?style=flat-square" alt="BDA Project"/>
</p>

<p>
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-features">Features</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-experiments">Experiments</a> ·
  <a href="#-project-structure">Structure</a>
</p>

</div>

---

## Overview

This project implements a fully **from-scratch** retrieval pipeline for querying NUST's 118-page Undergraduate Handbook in natural language. Three complementary retrieval methods are fused through a multi-signal re-ranking engine, and answers are grounded by the **Google Gemini LLM** with an automatic extractive fallback.

Built as a **Big Data Analytics** course project at SEECS, NUST — designed to be reproducible, modular, and experimentally rigorous.

---

## ✨ Features

| Feature | Description |
|---|---|
| **MinHash + LSH** | Approximate nearest-neighbor via locality-sensitive hashing (128 hash functions · 16 bands) |
| **SimHash** | 128-bit Hamming-distance fingerprinting for near-duplicate detection |
| **TF-IDF Baseline** | Exact cosine similarity — serves as ground truth for experiment evaluation |
| **Re-ranking Engine** | Multi-signal score fusion: `0.30×Jaccard + 0.35×Cosine + 0.20×Hamming + 0.15×Section` |
| **Gemini LLM** | Grounded answer generation with strict context-only prompting |
| **Extractive Fallback** | Automatic offline mode when API key is unavailable |
| **Streamlit Dashboard** | Live side-by-side method comparison with performance metrics |
| **Experiment Suite** | Parameter sensitivity · Scalability · Exact vs Approximate analysis |

---

## ⚡ Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/<your-username>/BDA_Project.git
cd BDA_Project
pip install -r requirements.txt
```

### 2 — Configure your API key

```bash
# Copy the environment template
cp .env.example .env
```

Open `.env` and paste your key — get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey):

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Note:** If no key is set the system still works — it falls back to extractive mode automatically.

### 3 — Build indices *(one-time)*

```bash
python build.py
```

This extracts, chunks, and indexes the handbook. Takes ~60 seconds on first run.

### 4 — Launch the dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

---

## 🏗️ Architecture

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                    Query Engine                         │
│                                                         │
│   ┌──────────────┐  ┌──────────┐  ┌────────────────┐  │
│   │ MinHash+LSH  │  │ SimHash  │  │    TF-IDF      │  │
│   │  (Jaccard)   │  │(Hamming) │  │   (Cosine)     │  │
│   └──────┬───────┘  └────┬─────┘  └──────┬─────────┘  │
│          │               │               │             │
│          └───────────────┴───────────────┘             │
│                          │                             │
│                ┌─────────▼──────────┐                  │
│                │  Re-ranking Engine  │                  │
│                │  (multi-signal fusion)│                │
│                └─────────┬──────────┘                  │
└──────────────────────────┼──────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   Gemini LLM        │
                │   (or Extractive)   │
                └─────────────────────┘
                           │
                         Answer
```

### Re-ranking Formula

$$\text{Score} = 0.30 \times J + 0.35 \times C + 0.20 \times H + 0.15 \times S$$

where **J** = Jaccard (MinHash), **C** = Cosine (TF-IDF), **H** = Hamming (SimHash), **S** = Section importance.

---

## 📁 Project Structure

```
BDA_Project/
│
├── app.py                      # Streamlit dashboard (QA + Experiments tabs)
├── build.py                    # One-time index builder
├── .env.example                # Environment variable template
├── requirements.txt
│
├── src/
│   ├── config.py               # All parameters and paths
│   ├── ingestion.py            # PDF extraction + text chunking (88 chunks)
│   ├── minhash_lsh.py          # MinHash + LSH — built from scratch
│   ├── simhash.py              # SimHash 128-bit — built from scratch
│   ├── tfidf_baseline.py       # TF-IDF cosine retrieval
│   ├── recommender.py          # Multi-signal re-ranking engine
│   ├── query_engine.py         # Orchestration layer
│   ├── answer_gen.py           # Gemini LLM + extractive fallback
│   └── experiments.py          # Evaluation suite (3 experiments)
│
└── data/
    └── processed/
        ├── chunks.json          # 88 extracted text chunks
        ├── indices/             # Serialized indices (pkl)
        └── experiments/         # Plots + results CSVs
```

---

## 🔬 Experiments

Three rigorous experiments are included, all runnable from the dashboard's **Experiments** tab or via script:

```python
from src.query_engine import QueryEngine
from src.experiments import run_all_experiments
import json

engine = QueryEngine()
engine.load_chunks()
engine.build_all_indices()
chunks = json.load(open("data/processed/chunks.json", encoding="utf-8"))
run_all_experiments(engine, chunks)
# → Results saved to data/processed/experiments/
```

### Experiment 1 — Exact vs Approximate

Compares LSH and TF-IDF across 15 queries on latency, memory usage, and retrieval quality.

| Metric | LSH (Approx) | TF-IDF (Exact) |
|---|---|---|
| Avg Query Time | 36.88 ms | 8.99 ms |
| Peak Memory | ~300 KB | ~300 KB |
| Recall@5 (vs Jaccard GT) | **88%** | — |
| Result Overlap | 26.7% | — |

### Experiment 2 — Parameter Sensitivity

Evaluates three parameters using exact Jaccard as ground truth:

- **MinHash hash count** (16 → 256): recall plateaus at 0.88 from n=64 onward
- **LSH bands**: fallback maintains 0.88 recall regardless of band config
- **SimHash threshold**: threshold ≥ 50 required for meaningful selection

### Experiment 3 — Scalability

Corpus duplicated 1× → 10× (88 → 880 chunks). Both methods sustain < 30 ms query latency at 10×.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Retrieval | MinHash+LSH · SimHash · TF-IDF *(all from scratch)* |
| LLM | Google Gemini API (`google-genai`) |
| Dashboard | Streamlit |
| PDF Parsing | PyMuPDF |
| ML/Math | scikit-learn · NumPy |
| NLP | NLTK |
| Config | python-dotenv |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Optional | Gemini API key for LLM answers. Falls back to extractive mode if unset. |

Set via `.env` file (recommended) or export directly:

```bash
# PowerShell
$env:GEMINI_API_KEY = "your-key"

# Bash / Zsh
export GEMINI_API_KEY="your-key"
```

---

## 📊 Results Summary

- **88% Recall@5** vs exact Jaccard ground truth
- **8 / 10** test queries answered correctly (qualitative evaluation)
- **26.7%** overlap between LSH and TF-IDF results — confirming they are complementary
- **64 hash functions** is the optimal MinHash configuration
- Both methods scale to **< 30 ms query latency** at 10× corpus size

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built for Big Data Analytics · SEECS, NUST · April 2026</sub>
</div>
