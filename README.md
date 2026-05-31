# staychat-rag

# StayChat AI – RAG-Based Hotel Q&A System

## Architecture Overview

```
hotel_documents.json
       │
       ▼
┌─────────────────────┐
│  Task 1: Preprocess │  clean_text() → create_chunks()
│  (sentence-window)  │  window=3 sentences, overlap=1
└────────┬────────────┘
         │  ~187 chunks
         ▼
┌─────────────────────┐
│  Task 2: Retrieval  │  SentenceTransformer (all-MiniLM-L6-v2)
│  FAISS IndexFlatIP  │  384-dim cosine similarity, top-k=5
└────────┬────────────┘
         │  top-5 chunks
         ▼
┌─────────────────────┐
│  Task 3: Generation │  OpenAI GPT-3.5-turbo (or mock)
│  Strict prompt +    │  temperature=0, context-only rules
│  temperature=0      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Task 4: Evaluation │  Precision@5, Recall@5, MRR
└─────────────────────┘
┌─────────────────────┐
│  Task 5: Halluc.    │  Strict prompt ablation demo
│  Control            │
└─────────────────────┘
```

### Dynamic Query Support
The system accepts **any natural-language question** at runtime (not just fixed queries). The 3 demo queries are examples only. Run with `python rag_pipeline.py` to enter the interactive query loop.

---

## Tools & Libraries

| Library | Purpose |
|---|---|
| `sentence-transformers` | `all-MiniLM-L6-v2` embeddings (free, local) |
| `faiss-cpu` | Vector index (IndexFlatIP, exact cosine) |
| `openai` | GPT-3.5-turbo for generation (optional) |
| `numpy` | Vector operations, metric computation |

---

## Project Structure

```
staychat_rag/
├── src/
│   └── rag_pipeline.py          # Main pipeline (all 5 tasks)
├── data/
│   └── hotel_documents.json     # 46-document hotel dataset
├── outputs/
│   └── sample_outputs.md        # Pre-run sample answers + metrics
├── staychat_rag_system.ipynb    # Jupyter notebook (all tasks + outputs)
├── requirements.txt
└── README.md
```

---

## Setup & Execution

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run in Mock mode (no API key – shows full pipeline)
```bash
cd src
python rag_pipeline.py
```

### 3. Run with real LLM answers (OpenAI)
```bash
export OPENAI_API_KEY=sk-your-key-here
cd src
python rag_pipeline.py
```

### 4. Demo only (3 fixed queries + evaluation, then exit)
```bash
python rag_pipeline.py --demo
```

### 5. Custom data path
```bash
python rag_pipeline.py --data /path/to/hotel_documents.json
```

### 6. Jupyter notebook
```bash
jupyter notebook staychat_rag_system.ipynb
```
Cells have pre-run outputs. Re-run to regenerate.

---

## Dynamic Query Mode

After the demo queries run, the system enters an **interactive loop**:

```
STAYCHAT AI – Hotel Q&A  (dynamic query mode)
Type your question and press Enter.
Commands: 'demo' = run 3 sample queries | 'exit' = quit
======================================================================

Your question: Does Summit Serenity Resort have a spa?
...retrieved chunks + answer...

Your question: What are the check-in times at Backwaters Bliss?
...retrieved chunks + answer...

Your question: exit
Goodbye!
```

---

## Dataset

46 documents covering 7 hotels across 5 categories:

| Hotel | Docs |
|---|---|
| The Oceanview Grand | descriptions, amenities, reviews, policy, location |
| Urban Nest Boutique Hotel | descriptions, amenities, reviews, policy, location |
| Heritage Haveli Udaipur | descriptions, amenities, reviews, policy, location |
| Summit Serenity Resort | descriptions, amenities, reviews, policy, location |
| Backwaters Bliss Resort | descriptions, amenities, reviews, policy, location |
| City Comfort Inn | descriptions, amenities, reviews, policy, location |
| The Metro Plaza Hotel | descriptions, amenities, reviews, policy, location |

Dataset is synthetic, created for this assessment. No external source.

---

## Design Decisions

### Chunking (Task 1)
Sentence-based sliding window (window=3, overlap=1) chosen over fixed-character chunking because hotel prose must not be split mid-sentence. A 3-sentence window (~80–150 tokens) fits within the embedding model's 256-token limit while covering one coherent topic per chunk.

### Embedding model (Task 2)
`all-MiniLM-L6-v2` — free, local, no API key required, strong semantic similarity on short-to-medium passages. 384 dimensions keeps the FAISS index small.

### k=5 (Task 2)
Covers ~3% of the corpus. Testing showed k<3 misses multi-hotel queries; k>7 introduces noise into the LLM context.

### Hallucination control (Task 5)
Strict system prompt + temperature=0. The prompt explicitly forbids out-of-context answers and instructs refusal when context is insufficient. Temperature=0 eliminates stochastic invention.

---

## Known Limitations

- **Negation queries** ("hotels WITHOUT a pool") are not handled — semantic similarity retrieves pool-related chunks regardless.
- **Comparative queries** ("cheapest hotel") require structured data; prose RAG gives incomplete answers.
- **Partial hotel names** may cause cross-hotel retrieval errors.
- **Recall is limited by k=5** for broad queries (e.g., WiFi/breakfast across all hotels).

---

## Evaluation Summary (Demo Queries)

| Metric | Q1 | Q2 | Q3 | Mean |
|---|---|---|---|---|
| Precision@5 | 1.00 | 0.20 | 1.00 | **0.73** |
| Recall@5 | 0.31 | 1.00 | 0.83 | **0.72** |
| Reciprocal Rank | 1.00 | 1.00 | 1.00 | **1.00** |
