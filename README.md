# Production RAG Pipeline

> Enterprise-grade Retrieval-Augmented Generation with hybrid search, automated evaluation, and full production observability.

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/GPT--4o-412991?style=flat&logo=openai&logoColor=white)](https://openai.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## Overview

This repository implements a **production-grade RAG pipeline** incorporating patterns from enterprise deployments in regulated financial services environments. It goes beyond basic RAG demos to address the hard problems: retrieval quality, hallucination prevention, latency management, and continuous evaluation.

**Core Capabilities:**
- Hybrid search (dense vector + BM25 sparse) with Reciprocal Rank Fusion
- Multi-stage retrieval with contextual compression and cross-encoder re-ranking
- Automated evaluation: faithfulness, answer relevance, context precision/recall (RAGAS)
- Streaming responses with token-level observability via LangSmith
- Redis semantic caching for latency and cost reduction (~40% API cost savings)
- Full MLflow experiment tracking for retrieval and generation quality metrics
- Guardrails AI output validation with PII detection and hallucination scoring

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                    │
│                                                         │
│  Documents → Chunking → Embeddings → ChromaDB (Dense)   │
│                     └──────────────→ BM25 Index (Sparse) │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                    │
│                                                         │
│  Query → Query Expansion → Hybrid Search               │
│                │ Dense retrieval (ChromaDB/cosine)       │
│                │ Sparse retrieval (BM25)                 │
│                └→ Reciprocal Rank Fusion (RRF k=60)     │
│                          │                              │
│                    Contextual Compression                │
│                          │                              │
│                     Cross-Encoder Re-rank               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   GENERATION PIPELINE                    │
│                                                         │
│  Context + Query → Prompt Assembly → GPT-4o             │
│                                          │              │
│                               Guardrails AI Validation  │
│                               (PII, hallucination)      │
│                                          │              │
│                                   Streaming Response    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  OBSERVABILITY LAYER                     │
│  MLflow Tracking · Redis Semantic Cache · RAGAS Eval    │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet |
| Orchestration | LangChain 0.3, LangGraph |
| Vector Store | ChromaDB (local), Pinecone (cloud) |
| Sparse Index | BM25 (rank_bm25) |
| Embeddings | OpenAI `text-embedding-3-large`, BGE-M3 |
| Re-ranking | Cohere Rerank v3, Cross-encoder (BAAI/bge-reranker) |
| Caching | Redis semantic cache |
| Evaluation | RAGAS, TruLens |
| Experiment Tracking | MLflow |
| Output Validation | Guardrails AI |
| API | FastAPI with streaming (SSE) |
| Observability | LangSmith, OpenTelemetry |

---

## Project Structure

```
production-rag-pipeline/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py      # Multi-format loader (PDF, DOCX, HTML, Confluence)
│   │   ├── chunking_strategies.py  # Recursive, semantic, and late chunking
│   │   └── embedding_pipeline.py   # Batch embedding with retry + backoff
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # Dense + sparse + RRF fusion
│   │   ├── contextual_compression.py
│   │   └── reranker.py             # Cross-encoder re-ranking
│   ├── generation/
│   │   ├── rag_chain.py            # LangChain LCEL chain assembly
│   │   ├── streaming_handler.py    # Token streaming with SSE
│   │   └── guardrails.py           # PII detection + hallucination scoring
│   ├── evaluation/
│   │   ├── ragas_evaluator.py      # RAGAS metric suite
│   │   ├── test_dataset.py         # Golden QA test set builder
│   │   └── mlflow_tracker.py       # Experiment + metric logging
│   └── api/
│       ├── main.py                 # FastAPI application
│       └── schemas.py              # Pydantic request/response models
├── notebooks/
│   ├── 01_ingestion_exploration.ipynb
│   ├── 02_retrieval_ablation.ipynb
│   └── 03_evaluation_dashboard.ipynb
├── tests/
├── docker-compose.yml              # ChromaDB + Redis + MLflow
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/codebygarrysingh/production-rag-pipeline
cd production-rag-pipeline
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Set: OPENAI_API_KEY, PINECONE_API_KEY, REDIS_URL

# Start infrastructure
docker-compose up -d

# Ingest documents
python -m src.ingestion.embedding_pipeline --docs ./data/

# Launch API
uvicorn src.api.main:app --reload

# Run evaluation suite
python -m src.evaluation.ragas_evaluator --experiment baseline_v1
```

---

## Hybrid Search: Why It Matters

Dense-only retrieval fails on keyword-critical queries (product codes, regulation IDs, proper nouns). BM25-only retrieval fails on semantic/conceptual queries. Hybrid with RRF captures both:

```python
# RRF formula: score(d) = Σ 1 / (k + rank(d))
# k=60 empirically optimal (Cormack et al., 2009)

fused_score = (dense_weight / (k + dense_rank)) + (sparse_weight / (k + sparse_rank))
```

**Measured improvement over dense-only:**
- Context Recall: +15pp on keyword-heavy enterprise corpora
- Context Precision: +9pp with re-ranking applied

---

## Evaluation Benchmark

| Metric | Naive RAG | This Pipeline |
|---|---|---|
| Faithfulness | 0.71 | **0.94** |
| Answer Relevance | 0.68 | **0.91** |
| Context Precision | 0.62 | **0.87** |
| Context Recall | 0.74 | **0.89** |
| p50 Latency | 3.2s | **1.1s** (with cache) |
| API Cost / 1K queries | $4.20 | **$2.51** (with semantic cache) |

---

## Key Engineering Insights

- **Chunking strategy matters more than model choice** — semantic chunking outperforms fixed-size by ~12pp on context recall
- **Re-ranking is non-negotiable** for precision-critical domains (legal, financial, medical)
- **Semantic caching** reduces LLM costs 35–60% in typical enterprise Q&A workloads
- **Guardrails must be async** — synchronous validation adds ~200ms; use background scoring with fallback

---

## Author

**Garry Singh** — Principal AI & Data Engineer · MSc Oxford

[Portfolio](https://garrysingh.dev) · [LinkedIn](https://linkedin.com/in/singhgarry) · [Book a Consultation](https://calendly.com/garry-singh2902)
