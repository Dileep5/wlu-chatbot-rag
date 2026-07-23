# WLU Chatbot

Hybrid RAG chatbot for Wilfrid Laurier University, built with Streamlit,
ChromaDB, and SQLite.

## Dependency Management

Dependencies are split across three manifests instead of one monolithic
file, so an install only pulls in what that task actually needs.

| Manifest | Purpose | Installs |
|---|---|---|
| `requirements-runtime.txt` | Serve the chatbot (`streamlit run src/app.py`) | `chromadb`, `sentence-transformers`, `streamlit`, `openai` |
| `requirements-ingestion.txt` | Run the scraper/loader/vector-db-build pipeline | Everything in `requirements-runtime.txt`, plus `pandas`, `beautifulsoup4`, `requests` |
| `requirements-dev.txt` | Run the evaluation suite (`python3 src/evaluate.py`) and the ad hoc dev utilities | Everything in `requirements-runtime.txt` and `requirements-ingestion.txt` |

**Dependency hierarchy**: `requirements-ingestion.txt` includes
`requirements-runtime.txt` via `-r`, and `requirements-dev.txt` includes
both. Every shared package (`chromadb`, `sentence-transformers`) is
pinned in exactly one place - `requirements-runtime.txt` - so the two
other manifests can never drift out of sync with it.

All versions are exact-pinned (`==`), not ranged, so an install today and
an install next month resolve to the identical set of direct
dependencies.

### Installation

```bash
# Just serving the chatbot
pip install -r requirements-runtime.txt

# Running the scraper/ingestion pipeline
pip install -r requirements-ingestion.txt

# Running the evaluation suite or development utilities
pip install -r requirements-dev.txt
```

`rank-bm25` is deliberately not included in any manifest - it was only
ever imported by two unused, unimported experimental scripts
(`bm25_test.py`, `hybrid_retrieval.py`), not by any code the app or
evaluation suite actually runs.
