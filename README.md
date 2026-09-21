# Local Semantic Search Engine

A fully offline tool to index and search PDF documents by **meaning**, not just keywords — built with local vector embeddings and a from-scratch implementation of the HNSW (Hierarchical Navigable Small World) algorithm. Optionally, a Groq-hosted LLM can turn the retrieved passages into a synthesized answer (RAG).

No external APIs are required for search itself — indexing, embedding, and retrieval all run locally. The LLM summarization step is optional and is the only part that talks to an external API (Groq).

## Why this project

Most semantic search tutorials wrap an existing vector database (FAISS, Pinecone, Weaviate). This project implements the core nearest-neighbor search algorithm — HNSW — from scratch, to understand what's actually happening under the hood of every modern vector database.

## How it works

```
PDFs  →  Text extraction & chunking  →  Local embeddings  →  HNSW index  →  Query  →  (optional) LLM summary
```

1. **Extraction & chunking** — PDFs are parsed with `pdfplumber` and split into overlapping word chunks, so context isn't lost at chunk boundaries.
2. **Embeddings** — each chunk is converted into a 384-dimensional, L2-normalized vector using `sentence-transformers` (`all-MiniLM-L6-v2`), run entirely locally.
3. **HNSW index** — vectors are inserted into a hand-built multi-layer graph structure, enabling approximate nearest-neighbor search in sub-linear time instead of brute-force comparison against every vector.
4. **Persistence** — the index and chunk metadata are cached to disk (`pickle`), with a folder-fingerprint check that automatically rebuilds the index only when the source PDFs change.
5. **Query** — a search query is embedded the same way, then the HNSW graph is traversed to retrieve the most semantically relevant chunks.
6. **Summarization (optional)** — `llm_search.py` sends the retrieved chunks plus your question to a Groq-hosted LLM, which answers strictly from that context.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To also use the LLM summarizer, set your Groq API key (get one at [console.groq.com](https://console.groq.com)):

```bash
export GROQ_API_KEY="your-key-here"

# Optional: override the default model. If unset or unavailable on your
# account, the script auto-falls-back to another available model.
export GROQ_MODEL="your-model"
```

Without `GROQ_API_KEY` set, `llm_search.py` still works — it just skips the summary and shows retrieved passages only (RAG-only mode).

## Usage

Place your PDFs in a `pdfs/` folder in the project root, then:

```bash
python main.py                # retrieval only: builds the index on first run, loads it on later runs
python main.py --rebuild      # force a full rebuild of the index
python main.py -k 10          # return 10 results per query instead of the default 5

python llm_search.py          # retrieval + LLM-synthesized answer (same flags as above)
```

Example session (`llm_search.py`):

```
> what is renaissance?

Top 5 retrieved sources:

1. [The Renaissance PDF.pdf]
   bankers and merchants who thrived through trade. This created a new social hierarchy...
...

Generating AI summary...

--- AI Response ---
The Renaissance was a period of profound cultural and intellectual revival in Europe,
beginning in the 14th century and peaking in the 15th-16th centuries...
-------------------
```

## Project structure

```
.
├── main.py            # CLI entry point: build/load index, run retrieval-only search loop
├── llm_search.py       # CLI entry point: retrieval + Groq LLM answer synthesis (RAG)
├── extract_text.py      # PDF text extraction and chunking
├── embedding.py           # Local embedding generation (sentence-transformers)
├── hnsw.py                 # HNSW graph construction and search, implemented from scratch
├── persistance.py            # Index + metadata persistence (pickle)
├── requirements.txt            # Python dependencies
├── index.pkl                    # Cached index (generated, not tracked in git)
└── pdfs/                          # Put your source PDFs here (not tracked in git)
```

## Design notes

- **Chunking** uses a sliding window (default: 180 words, 30-word overlap) so sentences spanning a chunk boundary aren't lost from either side. The 180-word size is intentionally kept under `all-MiniLM-L6-v2`'s 256-token limit — longer chunks get silently truncated by the model during encoding, so only part of the text would actually be embedded.
- **Embeddings are L2-normalized** (both chunks and queries) so that `hnsw.py`'s euclidean-distance search ranks results identically to cosine similarity — for unit vectors, `‖a-b‖² = 2 - 2·(a·b)`, which preserves ranking order.
- **HNSW parameters**: `M=16` (neighbors per node), `ef_construction=200` (candidate pool size while building), `M_max0=32` (layer-0 neighbor cap) — standard defaults from the original HNSW paper.
- **Approximate, not exact**: HNSW trades a small amount of recall for large gains in search speed versus brute-force comparison — the whole point of using it at scale.
- **LLM summarization** (`llm_search.py`) is strictly grounded: the model is instructed to answer only from the retrieved chunks and to say so explicitly if the answer isn't in the context, rather than filling gaps from its own training data.

## Known limitations

- Text extraction can produce missing spaces or jumbled column order on some PDFs (especially multi-column or scanned/OCR'd documents), depending on how the source document encodes character spacing and layout.
- Search returns relevant *passages*, and `llm_search.py` synthesizes an answer from them — but both are only as good as what the retrieved chunks actually contain.
- Only tested with English embeddings (`all-MiniLM-L6-v2`); for other languages, swap in a multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) and rebuild the index — embeddings from different models are not compatible with each other.
- If you change the chunking or embedding logic, delete `index.pkl` (or run with `--rebuild`) — the fingerprint check only detects changes to the PDFs, not to the code that processes them.

