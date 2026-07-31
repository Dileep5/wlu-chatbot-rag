# Serving image only (Phase 12G) - installs requirements-runtime.txt,
# not requirements-ingestion.txt/requirements-dev.txt, so the scraper-only
# stack (pandas/beautifulsoup4/requests) never enters this image. The
# ingestion pipeline is a separate, offline process (Phase 12D finding)
# and isn't containerized here.
FROM python:3.13-slim

# libgomp1 - torch's CPU wheel needs it for OpenMP-based threading at
# import time; curl - used by the HEALTHCHECK below to probe Streamlit's
# own health endpoint.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Every sqlite3.connect(...) call in retriever.py uses a path relative to
# the project root ("data/courses.db", etc.), and chromadb.PersistentClient
# is opened at "data/vector_db" - WORKDIR must be the project root, not
# src/, for those relative paths to resolve.
WORKDIR /app

# Runtime-only dependencies, pinned exactly (Phase 12E/12F). sentence-
# transformers pulls in torch transitively - the extra index resolves its
# CPU-only wheel instead of a much larger CUDA build this app never uses.
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-runtime.txt

# Exactly the live serving import chain, re-verified by grepping every
# top-level import in this chain (not assumed from an earlier pass, which
# is how renderer.py went missing before): app.py -> retriever.py,
# conversation.py, domain_guard.py, renderer.py, citation.py (Phase 3);
# retriever.py -> hybrid_rerank.py (Phase 1 hybrid retrieval).
# hybrid_rerank.py, conversation.py, domain_guard.py, renderer.py, and
# citation.py have no further local imports of their own. Neither
# memory.py nor intent_classifier.py is imported anywhere in this chain,
# so neither is copied in.
COPY src/app.py src/retriever.py src/hybrid_rerank.py src/conversation.py src/domain_guard.py src/renderer.py src/citation.py ./src/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# --server.address=0.0.0.0: Streamlit's default (localhost) is unreachable
# from outside the container. --server.headless=true: skips the
# interactive first-run email prompt, which would otherwise hang startup.
CMD ["streamlit", "run", "src/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
