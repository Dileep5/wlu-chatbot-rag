"""Hybrid retrieval support for retriever.py's hybrid_search(): BM25
sparse search, Reciprocal Rank Fusion of dense + BM25 candidates, and
cross-encoder reranking.

This module never decides whether a query is answerable at all - that
stays retriever.py's existing, untouched dense-distance confidence gate
(_VECTOR_SEARCH_DISTANCE_THRESHOLD), evaluated before anything here runs.
Everything below only decides WHICH already-approved page best answers
the question.
"""

import re
import threading
from collections import deque

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# -----------------------------
# Cross-encoder - loaded once at import time, same lazy-download-on-
# first-run pattern retriever.py already uses for its SentenceTransformer
# embedding model. cross-encoder/ms-marco-MiniLM-L-6-v2: small (~80MB),
# fast enough on CPU to rerank a ~20-candidate pool per query, and
# already installable via the sentence-transformers package this project
# has pinned - no new heavy dependency.
# -----------------------------

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

# -----------------------------
# Tunables
# -----------------------------

BM25_TOP_K = 15
RRF_K = 60
FUSION_POOL_SIZE = 20
RERANK_TOP_K = 10


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


# -----------------------------
# BM25 index - lazily (re)built from the live ChromaDB collection, not
# from outputs/chunks.csv. outputs/ is an ingestion-only build artifact
# that never ships to the runtime/Docker image (Dockerfile and
# docker-compose.yml only copy/mount data/), so indexing straight from
# the collection keeps this self-contained and guarantees BM25 can never
# drift out of sync with what dense search actually queries.
#
# Staleness is detected via collection.count() - one cheap metadata call,
# checked on every query. refresh_pipeline.py's weekly rebuild re-scrapes
# and re-chunks the whole corpus from scratch, which changes the total
# chunk count in virtually every real run, so the very next query after a
# refresh transparently rebuilds the in-memory BM25 index with no app
# restart required. (A refresh that happened to produce the exact same
# chunk count would be missed by this check; catching that too would mean
# fetching and hashing the whole corpus on every single query, which
# defeats the point of a cheap check for a corpus this size that only
# refreshes weekly.)
# -----------------------------

_bm25_lock = threading.Lock()

_bm25_state = {
    "index": None,
    "chunk_ids": [],
    "documents": [],
    "metadatas": [],
    "known_count": -1,
}


def _rebuild_bm25_index(collection, current_count):

    raw = collection.get(include=["documents", "metadatas"])

    _bm25_state["chunk_ids"] = raw["ids"]
    _bm25_state["documents"] = raw["documents"]
    _bm25_state["metadatas"] = raw["metadatas"]
    _bm25_state["index"] = BM25Okapi(
        [_tokenize(document) for document in raw["documents"]]
    )
    _bm25_state["known_count"] = current_count


def _get_bm25_state(collection):

    with _bm25_lock:

        current_count = collection.count()

        if (
            _bm25_state["index"] is None
            or current_count != _bm25_state["known_count"]
        ):
            _rebuild_bm25_index(collection, current_count)

        return _bm25_state


def bm25_search(collection, question, top_k=BM25_TOP_K):

    state = _get_bm25_state(collection)

    if not state["chunk_ids"]:
        return []

    scores = state["index"].get_scores(_tokenize(question))

    top_positions = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    candidates = []

    for rank, position in enumerate(top_positions, start=1):

        metadata = state["metadatas"][position] or {}

        candidates.append({
            "id": state["chunk_ids"][position],
            "document": state["documents"][position],
            "url": metadata.get("url"),
            "title": metadata.get("title"),
            "bm25_score": float(scores[position]),
            "bm25_rank": rank,
        })

    return candidates


def bm25_search_in_section(collection, question, url_fragment, top_k):

    """Restrict the existing BM25 index to chunks whose URL contains
    `url_fragment` (a canonical-section path segment like
    "/support-and-wellness/") and return the top-`top_k` scoring chunks.
    Used by retriever.hybrid_search() to pull a canonical section's pages
    into the fused pool when the dense/BM25 top-k never retrieved them
    (e.g. the disability-justice-and-accessibility page behind
    STUDENTSVC_004/014). Reuses the same lazily-rebuilt index as
    bm25_search(), so it stays consistent with refresh_pipeline.py
    without a fresh Chroma query or a second metadata filter pass."""

    state = _get_bm25_state(collection)

    if not state["chunk_ids"]:
        return []

    scores = state["index"].get_scores(_tokenize(question))

    section_positions = []

    for position, metadata in enumerate(state["metadatas"]):

        url = (metadata or {}).get("url") or ""

        if url_fragment in url:
            section_positions.append(position)

    section_positions.sort(key=lambda i: scores[i], reverse=True)

    candidates = []

    for position in section_positions[:top_k]:

        metadata = state["metadatas"][position] or {}

        candidates.append({
            "id": state["chunk_ids"][position],
            "document": state["documents"][position],
            "url": metadata.get("url"),
            "title": metadata.get("title"),
            "bm25_score": float(scores[position]),
            "bm25_section_rank": len(candidates) + 1,
        })

    return candidates


# -----------------------------
# Dense candidates for fusion - a plain per-chunk conversion of the
# results search_vector() already fetched (no second Chroma query, no
# per-URL dedup). Per-URL dedup is _rerank_vector_candidates()'s job for
# the confidence gate specifically, and stays completely separate and
# untouched; fusion wants the raw per-chunk dense ranking so it can fuse
# at the same chunk-level granularity BM25 operates at.
# -----------------------------

def dense_candidates_from_results(results):

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):

        candidates.append({
            "id": chunk_id,
            "document": document,
            "url": metadata.get("url"),
            "title": metadata.get("title"),
            "distance": distance,
            "dense_rank": rank,
        })

    return candidates


# -----------------------------
# Reciprocal Rank Fusion - merges by RANK POSITION, not raw score, since
# dense distance and BM25 score live on incompatible scales (bounded
# embedding distance vs. unbounded term-frequency score) that would
# otherwise need ad hoc normalization. k=60 is the standard RRF default
# and needs no calibration. A chunk present in only one ranker's list
# still gets a fused score from that ranker alone.
# -----------------------------

def reciprocal_rank_fusion(
    dense_candidates, bm25_candidates, k=RRF_K, top_n=FUSION_POOL_SIZE
):

    by_id = {}

    for candidate in dense_candidates:

        entry = by_id.setdefault(candidate["id"], dict(candidate))
        entry["dense_rank"] = candidate["dense_rank"]
        entry["distance"] = candidate["distance"]
        entry["fused_score"] = (
            entry.get("fused_score", 0.0) + 1.0 / (k + candidate["dense_rank"])
        )

    for candidate in bm25_candidates:

        entry = by_id.setdefault(candidate["id"], dict(candidate))
        entry.setdefault("url", candidate["url"])
        entry.setdefault("title", candidate["title"])
        entry.setdefault("document", candidate["document"])
        entry["bm25_rank"] = candidate["bm25_rank"]
        entry["bm25_score"] = candidate["bm25_score"]
        entry["fused_score"] = (
            entry.get("fused_score", 0.0) + 1.0 / (k + candidate["bm25_rank"])
        )

    fused = sorted(by_id.values(), key=lambda c: c["fused_score"], reverse=True)

    return fused[:top_n]


# -----------------------------
# Cross-encoder reranking - the final relevance judgment over the fused
# candidate pool. Scores every (question, chunk) pair directly (a strictly
# stronger relevance signal than embedding distance or BM25 score alone),
# batched in one predict() call.
# -----------------------------

def cross_encoder_rerank(question, candidates, top_k=RERANK_TOP_K):

    if not candidates:
        return []

    pairs = [(question, candidate["document"]) for candidate in candidates]

    scores = _cross_encoder.predict(pairs)

    reranked = []

    for candidate, score in zip(candidates, scores):

        entry = dict(candidate)
        entry["cross_encoder_score"] = float(score)
        reranked.append(entry)

    reranked.sort(key=lambda c: c["cross_encoder_score"], reverse=True)

    if top_k is None:
        # Full-pool scoring: retriever.hybrid_search() passes top_k=None
        # for canonical-section-intent questions so a section-preference
        # boost (see _apply_canonical_section_preference in retriever.py)
        # can also see candidates that land at ranks 11+ here - ranks that
        # the default top_k=RERANK_TOP_K truncation would otherwise hide
        # from every subsequent boost/penalty. Callers that want the
        # default truncation keep passing no argument.
        return reranked

    return reranked[:top_k]


# -----------------------------
# Debug trace - internal only, never surfaced in the Streamlit UI. Records
# the most recent N hybrid_search() calls' full retrieval trace
# (structured-retrieval flag, gate outcome, dense/BM25/fused/reranked
# candidate lists, final selection) for manual inspection during
# development, e.g. from a Python shell:
#
#   >>> from hybrid_rerank import get_last_debug_trace
#   >>> get_last_debug_trace()
# -----------------------------

_debug_lock = threading.Lock()
_debug_trace_history = deque(maxlen=20)


def record_debug_trace(entry):

    with _debug_lock:
        _debug_trace_history.append(entry)


def get_last_debug_trace():

    with _debug_lock:
        return _debug_trace_history[-1] if _debug_trace_history else None


def get_debug_trace_history():

    with _debug_lock:
        return list(_debug_trace_history)
