# 🌾 Agribot — Agricultural RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot built for farmers in Jharkhand, India. Agribot answers crop, pest, and agricultural queries grounded in district-level agricultural data using a hybrid retrieval pipeline.

---

## Architecture

```
User Query
    │
    ▼
Input Validation + Prompt Injection Filter
    │
    ▼
Hybrid Retriever
    ├── FAISS (Dense) ──────── Semantic similarity via sentence-transformers
    └── BM25 (Sparse) ──────── Keyword matching via rank_bm25
    │   Ensemble weights: [0.7, 0.3]
    ▼
MMR Reranking (fetch_k=20 → k=5)
    │
    ▼
LLM (Llama 3.1 8B via Groq)
    │
    ▼
Response + Source Citations
    │
    ▼
User Feedback → Supabase
```

---

## Evaluation Results

### Retrieval Quality (Custom Eval Scripts)

| Metric | Score |
|--------|-------|
| Precision@2 | 0.82 |
| Recall@2 | 0.84 |
| MRR | 0.85 |

### End-to-End RAG Quality (RAGAS)

| Metric | Score | What it measures |
|--------|-------|-----------------|
| Faithfulness | 0.72 | Is the answer grounded in retrieved context? |
| Answer Relevancy | 0.80 | Does the answer address the question? |
| Context Recall | 0.83 | Did retrieval find the right chunks? |
| Context Precision | 0.79 | Are retrieved chunks actually useful? |

**Notes:**
- Faithfulness of 0.72 is expected for a quantised 8B model — the LLM occasionally adds general agricultural knowledge beyond the retrieved context. Mitigation: stricter prompt grounding and a larger model would improve this.
- Answer relevancy of 0.80 confirms the hybrid retriever consistently surfaces relevant context.
- All retrieval metrics at 0.85 validate the hybrid search approach over pure dense retrieval.

---

## Key Design Decisions

### Why Hybrid Search (FAISS + BM25)?

Dense retrieval (FAISS) fails on exact keyword queries — local crop names, specific disease terms, or district names that the embedding model hasn't seen frequently. Sparse retrieval (BM25) fails on semantic queries where the user paraphrases ("leaves turning yellow" vs "chlorosis"). Combining both with 0.7/0.3 weighting covers both failure modes.

### Why MMR over Similarity Search?

Maximal Marginal Relevance (`fetch_k=20, k=5, lambda=0.7`) retrieves diverse chunks rather than the top-5 most similar. Without MMR, retrieval returns near-duplicate chunks from the same document section, wasting the context window. MMR balances relevance with diversity.

### Why `all-mpnet-base-v2`?

Better semantic accuracy than `all-MiniLM-L6-v2` at the cost of slightly higher latency. Since agricultural queries require precise domain understanding (e.g. distinguishing between crop varieties), accuracy was prioritised over speed. A domain-fine-tuned embedding model on agricultural text would further improve retrieval quality.

### Why Chunk Size 384?

Aligned with `all-mpnet-base-v2`'s maximum token limit (384 tokens). Larger chunks risk truncation during embedding which degrades vector quality. Overlap of 50 tokens preserves semantic continuity at chunk boundaries.

### Why Session-Isolated Memory with TTL?

`ConversationBufferWindowMemory` is created per `session_id` rather than shared across requests — preventing conversation history from leaking between users. A 1-hour TTL cleans up inactive sessions to prevent unbounded memory growth. For production scale, this would be replaced with Redis.

---

## Features

- **Hybrid Retrieval** — FAISS (dense) + BM25 (sparse) with configurable ensemble weights
- **Session Memory** — per-user conversation history with 1-hour TTL
- **Document Deduplication** — MD5-based doc ID tracking prevents duplicate ingestion
- **Prompt Injection Defence** — input sanitisation + output filtering + hardened system prompt
- **User Feedback Loop** — thumbs up/down logged to Supabase for continuous improvement
- **Dual Interface** — Streamlit UI for farmers, FastAPI for programmatic access
- **Incremental Indexing** — add new documents without rebuilding the full vector store
- **Structured Logging** — request/response logging via Python logging module
- **Input Validation** — Pydantic-based query validation (length, empty input)
- **Containerised** — Docker + dockerignore for consistent deployment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Llama 3.1 8B (via Groq API) |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Vector Store | FAISS (faiss-cpu) |
| Sparse Retrieval | BM25 (rank_bm25) |
| Framework | LangChain |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Feedback Store | Supabase (PostgreSQL) |
| Evaluation | RAGAS + custom retrieval scripts |
| Containerisation | Docker |

---

## Project Structure

```
├── src/
│   ├── main.py              # FastAPI app, lifespan, middleware
│   ├── api.py               # Chat endpoint
│   ├── qa_chain.py          # LLM, retriever, memory, prompt
│   ├── pipeline.py          # Orchestrates doc loading → retriever
│   ├── vector_store.py      # FAISS create/load/update
│   ├── document_loader.py   # PDF/TXT loading + chunking
│   ├── models.py            # Pydantic schemas
│   ├── config.py            # Paths, model names, chunk config
│   └── limiter.py           # Rate limiter (Nginx in production)
├── utils/
│   ├── documentIDUtils.py   # MD5 doc ID generation + deduplication
│   └── supabase_client.py   # Supabase client
├── eval_scripts/
│   └── retrievers/
│       ├── precision.py     # Precision@k evaluation
│       ├── recall.py        # Recall@k evaluation
│       ├── mrr.py           # Mean Reciprocal Rank evaluation
│       └── ragas_eval.py    # End-to-end RAG evaluation
├── tests/
│   └── test_api.py          # API tests
├── data/
│   └── dicra/               # Jharkhand district agricultural data
├── chat.py                  # Streamlit UI
├── Dockerfile
└── requirements.txt
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Groq API key (free tier works)
- Supabase project (for feedback logging)

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/agribot
cd agribot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your keys
```

### Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Run

```bash
# FastAPI server
uvicorn src.main:app --reload

# Streamlit UI (separate terminal)
streamlit run chat.py
```

### Docker

```bash
docker build -t agribot .
docker run -p 8000:8000 --env-file .env agribot
```

---

## API Usage

### POST `/chat/`

```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What crops grow well in Ranchi?", "session_id": "user123"}'
```

**Response:**
```json
{
  "response": "In Ranchi, crops such as rice, maize, and pulses are commonly cultivated...",
  "sources": ["Ranchi district receives average annual rainfall of 1400mm..."]
}
```

---

## Evaluation

### Run retrieval eval
```bash
cd eval_scripts/retrievers
python precision.py
python recall.py
python mrr.py
```

### Run RAGAS eval
```bash
python eval_scripts/ragas_eval.py
```

---

## Known Limitations & Future Work

| Limitation | Planned Fix |
|-----------|-------------|
| Faithfulness 0.72 — LLM adds knowledge beyond context | Larger model or fine-tuned smaller model |
| In-memory session store | Replace with Redis for horizontal scaling |
| FAISS single-node | Migrate to Pinecone/Weaviate for scale |
| BM25 in-memory | Replace with Elasticsearch for large corpora |
| Rate limiting at app level | Move to Nginx reverse proxy (production pattern) |
| English only | Add Hindi/regional language support |
| Static knowledge base | Async document ingestion pipeline (Celery/Kafka) |

---

## Data Source

District-level agricultural data for 24 Jharkhand districts sourced from [DICRA (Data in Climate-Resilient Agriculture)](https://dicra.nabard.org/) — a UNEP initiative providing open agricultural datasets for India.

---

## License

MIT
