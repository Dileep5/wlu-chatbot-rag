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

## Docker

The serving image installs only `requirements-runtime.txt` - the
scraper-only packages in `requirements-ingestion.txt` never enter it.
`data/` and the Hugging Face model cache are mounted as volumes, not
baked into the image, since both are working artifacts that change
independently of application code.

### Environment variables

Create a `.env` file in the project root (never committed - already
git-ignored) containing:

```
OPENAI_API_KEY=sk-...
```

`docker-compose.yml` loads this via `env_file`, so the container sees
the same `OPENAI_API_KEY` the app already reads via `os.getenv(...)`
when run outside Docker.

### Expected directory structure

`data/` must exist next to the Dockerfile before starting the
container - it's mounted in, not built by the image:

```
data/
├── courses.db
├── faculty.db
├── programs.db
├── departments.db
└── vector_db/
```

These are produced by the ingestion pipeline (`requirements-ingestion.txt`)
run outside Docker; the serving container only ever reads them.

### Build and run

```bash
# Build the image
docker build -t wlu-chatbot .

# Build and start via compose (recommended - wires up both volumes
# and the .env file automatically)
docker compose up --build
```

The chatbot is then reachable at `http://localhost:8501`.
