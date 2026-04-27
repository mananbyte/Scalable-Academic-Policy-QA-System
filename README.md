# Scalable Academic Policy QA System

A Big Data Analytics project that enables natural-language question answering over the NUST Undergraduate Handbook using MinHash+LSH, SimHash, and TF-IDF retrieval with Gemini LLM answer generation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build indices (one-time)
python build.py

# Run the dashboard
streamlit run app.py
```

## Features

- **MinHash + LSH** — Approximate nearest-neighbor retrieval via locality-sensitive hashing
- **SimHash** — 128-bit fingerprinting with Hamming distance
- **TF-IDF Baseline** — Exact cosine similarity for ground-truth comparison
- **Re-ranking** — Multi-signal recommendation engine (Jaccard + Cosine + Hamming + Section)
- **Gemini LLM** — Grounded answer generation with extractive fallback
- **Experiments** — Parameter sensitivity, scalability, and exact vs approximate analysis

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── build.py                # Index builder
├── PROJECT_REPORT.md       # Full project report
├── src/
│   ├── config.py           # Configuration
│   ├── ingestion.py        # PDF extraction + chunking
│   ├── minhash_lsh.py      # MinHash + LSH (from scratch)
│   ├── simhash.py          # SimHash (from scratch)
│   ├── tfidf_baseline.py   # TF-IDF retrieval
│   ├── recommender.py      # Re-ranking engine
│   ├── query_engine.py     # Query orchestration
│   ├── answer_gen.py       # LLM + extractive answers
│   └── experiments.py      # Evaluation experiments
├── data/
│   └── processed/
│       ├── chunks.json     # 88 text chunks
│       ├── indices/        # Serialized indices
│       └── experiments/    # Experiment results + plots
└── requirements.txt
```

## Running Experiments

```bash
python -c "
import sys; sys.path.insert(0,'src')
from query_engine import QueryEngine
from experiments import run_all_experiments
import json

e = QueryEngine()
e.load_chunks()
e.build_all_indices()
chunks = json.load(open('data/processed/chunks.json','r',encoding='utf-8'))
run_all_experiments(e, chunks)
"
```

Results saved to `data/processed/experiments/`.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Retrieval | MinHash+LSH, SimHash, TF-IDF (from scratch) |
| LLM | Google Gemini API |
| UI | Streamlit |
| PDF | PyPDF2 |
| ML | scikit-learn, numpy |
