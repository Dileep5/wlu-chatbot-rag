# WLU Hybrid RAG Assistant

## Project Overview

WLU Hybrid RAG Assistant is an AI chatbot for Wilfrid Laurier University
that answers questions about courses, programs, faculty, admissions,
tuition, scholarships, and student services.

It is built as a genuine **Hybrid Retrieval-Augmented Generation
(Hybrid RAG)** system: rather than relying on a language model's
general knowledge, every answer is grounded in real, scraped WLU data
(the academic calendar and faculty directory). The assistant first
tries deterministic, structured retrieval against a set of SQLite
databases; if that finds nothing, it falls back to semantic vector
search over the scraped WLU website; only when neither is a clean,
already-complete answer does an LLM synthesize a response, and even
then only from the retrieved data - never from its own general
knowledge. Questions outside WLU's domain are declined gracefully
rather than answered or fabricated.

---

## Features

- **Hybrid Retrieval** - structured SQL lookups are always tried
  first; vector search only runs when nothing structured matches.
- **ChromaDB Vector Search** - semantic search over scraped WLU web
  pages, used as the fallback for open-ended questions with no exact
  structured match (e.g. tuition, scholarships, student services).
- **Structured SQLite Search** - deterministic lookups across
  dedicated course, program, faculty, and department databases.
- **Deterministic Course Cards** - course questions bypass the LLM
  entirely and render directly from the structured course record, so
  the answer is exactly what's in the database, every time.
- **Deterministic Faculty Cards** - the same deterministic treatment
  for individual faculty profiles (title, department, contact info,
  research interests).
- **Deterministic Program Cards** - program questions are similarly
  deterministic, with long program descriptions automatically split
  into clear sections (Overview, Required Courses, Recommended
  Schedule, Program Regulations, Additional Information) instead of
  one long block of text.
- **Source Citations** - every grounded answer includes a link back to
  the exact WLU page it was drawn from.
- **Conversation Memory** - multi-turn context resolution for
  pronouns, follow-ups, and ordinal references ("Does it have
  prerequisites?", "Who teaches it?", "Tell me about the first one").
- **Off-topic Detection** - questions unrelated to WLU are declined
  gracefully instead of being answered or fabricated.
- **Streamlit UI** - a clean, native chat interface with a project
  sidebar and a proper conversation history.
- **Suggested Questions** - clickable example prompts shown before the
  first message, so a new user (or a demo) has an immediate, working
  starting point.

---

## Architecture

```
User
   ↓
Intent Detection
   ↓
Hybrid Retrieval
   ↓
Structured / Vector Search
   ↓
Response Generation
   ↓
Card Renderer
   ↓
Streamlit UI
```

- **Intent Detection** - the incoming message is first checked for a
  greeting, ordinary conversation, or an out-of-domain topic, so those
  cases are handled directly without ever reaching retrieval.
- **Hybrid Retrieval** - a real, in-domain question is run through
  deterministic structured search first (and, for follow-up questions,
  through contextual-reference resolution using conversation memory).
- **Structured / Vector Search** - if structured search finds nothing,
  the question falls through to ChromaDB vector search over scraped
  WLU pages as a fallback.
- **Response Generation** - responses already known to be complete and
  correct (courses, programs, faculty, prerequisites, coordinators,
  research) are shown exactly as retrieved, with no LLM involved;
  everything else is synthesized by an LLM from the retrieved context.
- **Card Renderer** - the response is handed to a dedicated rendering
  layer that displays it as a structured Course/Faculty/Program card
  when applicable, or as a plain conversational reply otherwise.
- **Streamlit UI** - the final rendered response, along with its
  source citation, is displayed in the chat interface.

---

## Technologies

- Python
- Streamlit
- SQLite
- ChromaDB
- Sentence Transformers
- OpenAI GPT
- BeautifulSoup
- Requests

---

## Project Structure

```
WLU ChatBot/
├── src/
│   ├── app.py              # Entry point: Streamlit UI, session state, routing
│   ├── retriever.py        # Structured + vector retrieval, conversation memory
│   ├── renderer.py         # Course/Faculty/Program card rendering
│   ├── conversation.py     # Small-talk / greeting detection
│   ├── domain_guard.py     # Out-of-domain (off-topic) detection
│   ├── evaluate.py         # Automated regression suite (see Evaluation below)
│   ├── scrape*.py, get_*.py, load_*.py, build_*.py
│   │                       # Offline ingestion pipeline: scrapes WLU's
│   │                       # website, builds the SQLite databases and
│   │                       # the ChromaDB vector store
│   └── create_*_table.py   # One-off database schema creation scripts
├── data/
│   ├── courses.db          # Course catalog
│   ├── programs.db         # Program/requirement catalog
│   ├── faculty.db          # Faculty directory
│   ├── departments.db      # Department directory
│   └── vector_db/          # ChromaDB persistent vector store
├── requirements-runtime.txt    # Dependencies to serve the chatbot
├── requirements-ingestion.txt  # + dependencies to run the scraper pipeline
├── requirements-dev.txt        # + dependencies to run the evaluation suite
├── Dockerfile / docker-compose.yml
└── README.md
```

`data/` is produced entirely by the ingestion pipeline (the
`scrape_*`/`get_*`/`load_*`/`build_*` scripts in `src/`) and is treated
as a build artifact - it's git-ignored where appropriate and only ever
read by the live chatbot, never written to at runtime.

---

## Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-runtime.txt

# 3. Set your OpenAI API key (required for any non-deterministic
#    response, e.g. vector-search fallback or conversational replies)
export OPENAI_API_KEY=sk-...    # or place it in a .env file

# 4. Run the app
streamlit run src/app.py
```

The app will be available at `http://localhost:8501`.

If you also want to run the evaluation suite or the ingestion/scraper
pipeline, install `requirements-dev.txt` or `requirements-ingestion.txt`
instead (see **Dependency Management** below for what each includes).

---

## Evaluation

The project includes a complete, automated regression suite:

```bash
python3 src/evaluate.py
```

This drives the real Streamlit app end-to-end (via
`streamlit.testing.v1.AppTest`) across every capability - courses,
programs, faculty, coordinators, prerequisites, research, multi-turn
conversation memory, and out-of-domain detection - plus a set of direct
data-integrity checks against the scraped databases. In total it runs
349 assertions and reports a pass/fail summary by category.

**Note**: two of the data-integrity checks (`URL normalization: legacy
URL template` and `URL normalization: stale current-pattern path`) make
a live HTTP request to the real wlu.ca website to verify redirect
behavior. Their pass/fail outcome depends on WLU's website at the
moment the suite is run, not on this application's code - if WLU
changes a page's redirect behavior, these two checks can fail (or
pass) independently of any change made here. Every other check is
fully deterministic and does not depend on external network state.

---

## Example Queries

**Course**
- "What is CP312?"
- "Tell me about CP317."

**Faculty**
- "Who is Tripat Gill?"
- "Tell me about Ammara Mahmood."

**Program**
- "Tell me about the Honours BSc Computer Science program."
- "What is the Master of Applied Politics program?"

**Admissions**
- "What are the admission requirements?"

**Scholarships**
- "What scholarships are available?"

**Student Services**
- "What student services does WLU offer?"

---

## Future Work

- Extend deterministic, structured card rendering to faculty/department
  list views and department profiles (currently still LLM-synthesized).
- Move from markdown-based cards to true native Streamlit widgets
  (`st.container`, `st.columns`, `st.metric`) for a richer visual
  layout.
- Automate the ingestion pipeline (currently a manual sequence of
  scripts) into a single scheduled or one-command refresh job.
- Verify and harden the Docker image for a real deployment target, not
  just local development.
- Add authentication and rate-limiting before any deployment beyond a
  local or classroom demo.

---

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

---

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
