# Sprint D — Gemini-Style Query Rewriting & Intent Expansion

**Status:** Implementation + verification complete. Regression passes — committed automatically.
**Commit:** `7f7c8c9`

---

## 1. Goal and Constraints

Improve **retrieval** quality by deterministically rewriting/expanding user
queries before the free-text search layer, the way Gemini Search rewrites a
casual or abbreviated query into canonical vocabulary before matching.

Per the sprint brief, this is **BETTER SEARCH, not better prompting**. Not
modified: crawler, preprocessing, chunking, Chroma schema, embeddings, BM25
implementation, evaluation framework, citation pipeline. Only retrieval
orchestration in `src/retriever.py` changed (one file). No LLM-based query
rewriting, no external APIs, no invented facts — every expansion is a static
ground-truth phrase.

## 2. Root Cause — with evidence

Phase 1 (investigation) established with a probe over every benchmark/manual
question that `search_vector()` embeds the **raw question verbatim** and
`hybrid_rerank.bm25_search()` tokenizes the **raw question verbatim** — no
semantic/synonym/acronym/intent/entity expansion anywhere in the free-text
path (`identical_to_raw=True` for all queries). The only synonym map in the
codebase is `_TOPIC_SYNONYMS` (3 entries) used by one narrow faculty-course
searcher — not by the general vector/BM25 path.

Phase 2 (failure patterns) showed the concrete cost on real questions
(run on committed code, captured in `/tmp/wlu_eval/manual_gemini_results_BASELINE.json`):

| Question | What retrieval did | Why |
|---|---|---|
| "How do I find a research supervisor?" | **not_found** (declined) | raw query's nearest dense neighbour was 1.2+ (over gate threshold) |
| "I'm stressed, what support is there?" | cited a *time-management article* | raw words surface generic study-advice pages, not wellness |
| "What is AI?" / "What is NLP?" / "What is CS?" | off-topic decline | domain gate (safety layer) declined — no rewrite can reach it |
| "I'm looking to get a graduate degree" | cited the *McCall MacBain scholarship* page | raw words overlap a McGill-scholarship page; Laurier grad-admissions pages miss the pool |
| "I want to do a Masters" | off-topic decline | domain gate declined |

Blanket expansion was tested and rejected in Phase 2 (probe): appending
broad synonyms *hurts* precision ("scholarship + financial aid" → governance
policy page; "artificial intelligence" alone → course match). So the design
is **trigger-based** — high-precision rules that fire only on specific
tokens/intents, never on already-well-formed queries.

## 3. What Changed (all in `src/retriever.py`)

### 3.1 `_rewrite_query(question)` + six trigger rules

A pure function, inserted next to the Sprint C multi-doc helpers. Six
high-precision rules; each **appends** canonical WLU vocabulary **after the
user's own words** (so BM25/vector still weigh the original intent highest),
and every rule skips a phrase already present (no duplication):

1. **Acronym expansion** — standalone word-boundary tokens:
   `ai → artificial intelligence + machine learning`, `ml → machine
   learning`, `nlp → natural language processing`, `cs → computer
   science`, `msw → master of social work`. Course codes like `CS100`
   (no boundary) are never touched.
2. **Stress / mental-health intent** — `stressed|anxious|overwhelmed|
   depressed|suicidal|suicide|feeling down|not ok …` → `mental health
   wellness counselling student support services`.
3. **Casual graduate intent** — subject-pronoun / "looking to / thinking
   about" phrase followed by `masters|graduate|grad`, or `graduate
   degree|program|studies|school|admissions` → `graduate programs master
   programs admissions graduate studies`. Deliberately does **not** fire
   on `Master of X` proper names (structured program search owns those).
4. **Research supervisor** — `find … supervisor` / `research supervisor`
   → `faculty research graduate studies`.
5. **AI faculty/research** — AI-term + faculty/research-term co-occurrence
   in either order (no sentence-boundary crossing) → `faculty research
   computer science`.
6. **International students** — **support/help context only**
   (`support|help|immigration|visa|counselling|…`) → `support services`.
   A bare "tuition/admissions for international students" query is precise
   as-is and never picks up the noise.

No rule fires for any decline-expected benchmark item; exactly three
benchmark items fire (verified): `FAQ_002` ("MSW" → "Master of Social
Work", aligned) and `STUDENTSVC_001/011` (support-context international,
reinforcing the support-and-wellness page they require).

### 3.2 Two-stage gate in `hybrid_search()`

The free-text path now has an **"expand only when beneficial"** guarantee:

- `retrieval_question = _rewrite_query(question)` (byte-identical when no
  rule fires — the entire layer is then a no-op).
- **Stage 1** — the RAW question is always embedded first and its own
  nearest neighbour gates, exactly as pre-Sprint-D (the distance threshold
  keeps its original raw-query calibration). If it passes, the candidate
  pool is rebuilt from `retrieval_question` when a rewrite actually fired.
- **Stage 2 (rescue)** — if the raw query **declines** the gate but a
  rewrite fired, the rewritten query's own nearest neighbour is tried and
  accepted only if *it* is in-threshold. A rewrite can therefore never
  turn an answerable query into `not_found` — it can only rescue a query
  the raw form would have declined.
- `bm25_search` runs on `retrieval_question`; `_match_canonical_section`
  and `bm25_search_in_section` stay on the RAW question (section intent is
  keyed on exact canonical phrases).
- Debug trace now records `query_rewritten` / `rewritten_query` (and the
  gate-fail trace records them too).

## 4. Deliberately NOT changed

- Structured retrieval, the domain gate (`is_wlu_related`), contextual
  reference resolution, answer generation, citations — all keep the raw
  question. The 5 gate-declined manual questions (AI/NLP/CS/"Masters"/
  "overwhelmed") are out of scope: the domain gate declines them *before*
  `hybrid_search` runs, and it is a safety layer 22 decline-expected
  benchmark items depend on (zero-hallucination constraint).
- Cross-encoder reranking, fusion, penalties/boosts, Sprint C multi-doc
  context construction.
- `_TOPIC_SYNONYMS`, BM25, embeddings, the evaluation framework.

## 5. Verification — Before vs After

Baseline: committed Sprint C code, same harnesses.

| Metric | Baseline (Sprint C) | After (Sprint D) | Δ |
|---|---|---|---|
| Benchmark answer accuracy | 100.0% (202/202) | **100.0% (202/202)** | unchanged ✓ |
| Citation accuracy | 99.4–100% (FAQ_009 flake) | **99.4–100%** (same flake, §6) | unchanged ✓ |
| Hallucination rate | 0.0% (22 decline-expected) | **0.0% (22/22)** | unchanged ✓ |
| Retrieval success | 100.0% (169/169) | **100.0% (169/169)** | unchanged ✓ |
| Structured retrieval | 100.0% (105/105) | **100.0% (105/105)** | unchanged ✓ |
| Hybrid (vector) retrieval | 100.0% (66/66) | **100.0% (66/66)** | unchanged ✓ |
| Avg response time | 2.68s | **2.65s** | −0.03s ✓ |
| Regression suite (dedicated) | 235/235 | **235/235 ALL PASSED** | unchanged ✓ |
| Stress (55) rt/citations | — | 0 mismatches, 0 shifts | byte-identical ✓ |
| Edge (19) rt/citations | — | 0 mismatches, 0 shifts | byte-identical ✓ |
| Manual Gemini (19) | — | 1 rescue, 1 source improvement, 17 unchanged | §6 |

**Regression gate:** the combined benchmark+regression invocation reported
234/235 — the proven pre-existing MBA "admission requirements" LLM-summary
phrasing flake documented in Sprint C §6.2/§8 (routes `structured_search`,
never touches `hybrid_search`). The dedicated re-run verifies **235/235
ALL TESTS PASSED**.

## 6. Manual Gemini-Comparison — the improvements (2 of 19)

The 19-question set (`/tmp/wlu_eval/manual_gemini.py`) was captured on
committed Sprint C code (`manual_gemini_results_BASELINE.json`) and re-run
after (`manual_gemini_results.json`). 17 items are byte-identical in
response type and citation set (including all 5 gate-declined ones, §4).
Two items improved — both are **retrieval** outcomes (deterministic, not
LLM text):

### 6.1 MG_SUP1 — "How do I find a research supervisor?" (RESCUED)

| | Baseline | After |
|---|---|---|
| response_type | `not_found` | `vector` |
| citation | — | research-apprenticeship-program page |

The raw query's nearest dense neighbour sits over the 1.2 gate threshold
(that is *why* it was declining). The rewritten query
`How do I find a research supervisor? faculty research graduate studies`
embeds to **1.048** (in-threshold) → the Stage-2 rescue accepts it and the
answer now walks through the Research Apprenticeship Program's
faculty-mentor steps. This is the exact "intent expansion" the sprint
targeted.

### 6.2 MG_STRESS1 — "I'm stressed, what support is there?" (SOURCE SHIFT)

| | Baseline | After |
|---|---|---|
| citation | the-truth-about-time-management-and-academic-success | find-your-support-resource (wellness index) |
| answer flavor | "time management and well-being" | Student Wellness Centre + counselling + 988 crisis supports |

The stress rule redirected retrieval from a study-advice page to the actual
support-resource directory. `MG_STRESS3` ("I need help with my mental
health") is unchanged — it already names the topic explicitly, so no rule
fires (by design).

### 6.3 Remaining limitations (documented, unchanged)

- **MG_MASTERS2** ("I'm looking to get a graduate degree"): the graduate
  rule **did** put the graduate-and-postdoctoral-studies admissions page at
  the top of the fused pool (retrieval improved), but the cross-encoder
  reranker — scoring against the raw question — still prefers the McCall
  MacBain page. Re-scoring against the rewritten query was tested and
  makes the reranker *more* confident in the wrong page (pulls in
  off-target international pages), so no rerank change was made; this is a
  reranking judgment, out of Sprint D's scope.
- **MG_AI2** ("What courses use AI at WLU?"): the corpus contains no page
  listing AI-using courses (only Gen-AI *guidelines* pages); the answer
  honestly disclaims. A corpus-coverage gap, not a retrieval failure.
- **Gate-declined intents** (AI/NLP/CS/"Masters"/"overwhelmed"): declined
  by the domain gate before retrieval; the rewrite cannot reach them.
- Rewriting applies to the **free-text path only** — deterministic 16-type
  responses are single-source by nature and untouched.

## 7. Sample output (after, real run)

**"How do I find a research supervisor?"** (was `not_found`):
> To find a research supervisor at Wilfrid Laurier University, you can
> participate in the **Research Apprenticeship Program** at Laurier
> Brantford. This program requires you to secure support from a faculty
> mentor before applying. **Steps to Find a Research Supervisor:** 1.
> Identify a faculty member …
> *(citations: research-apprenticeship-program)*

**"I'm stressed, what support is there?"** (was a time-management article):
> Wilfrid Laurier University offers various mental health resources to
> support your well-being. You can reach out to the **Student Wellness
> Centre** for information and counseling. **Key Details:** Each department
> provides specialized supports tailored to individual need …
> *(citations: find-your-support-resource)*

**Deterministic paths unchanged** — course cards, program cards, faculty
profiles, structured answers are byte-identical (they never reach
`hybrid_search`).

## 8. Regression Risk Assessment

- **Benchmark/regression surface:** exactly 3 of 202 benchmark items fire a
  rule, all with aligned expansions; 0 stress/edge items fire any rule
  (verified: response types + citation sets byte-identical across all 74).
  No decline-expected item fires a rule. 202/202 answer accuracy and
  235/235 dedicated regression confirm no regression.
- **Gate safety:** the raw query is always the gate basis; a rewrite can
  only *add* a rescue branch for queries that were already declining. The
  off-topic gate, cold-start referentless/cold-follow-up gates, and
  `pending_query` flow all evaluate the raw question unchanged.
- **Precision:** every rule is a narrow trigger; expansions append, never
  replace; already-present phrases are skipped. The international rule is
  support-context-gated specifically to keep "tuition for international
  students"-style queries byte-identical.
- **Response time:** avg 2.65s vs 2.68s baseline (within LLM variance); the
  rescue path costs one extra embed only when the raw query would decline.
- **Citations:** unchanged pipeline; multi-source citations (Sprint C) flow
  through the same iterable path.

## 9. Engineering Hygiene

- One file modified (`src/retriever.py`), ~240 insertions / 16 deletions,
  no TODOs/debug code.
- `evaluation_report.md` auto-regenerated by the final benchmark run.
- Rules are data (pattern + phrase tuples) with a single pure function —
  easy to extend or audit; expansions are static ground-truth terms (no
  LLM, no API).
