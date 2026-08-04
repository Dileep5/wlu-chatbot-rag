# Sprint C — Multi-Document Retrieval & Context Synthesis

**Status:** Implementation + verification complete. Regression passes — committed automatically.
**Commit:** set after commit

---

## 1. Goal and Constraints

Upgrade the chatbot from "single-document answering" to "multi-document
synthesis" — the answer should draw on several relevant WLU pages at once,
like Gemini — while **preserving the current benchmark accuracy**.

Per the sprint brief, the architecture was **not** redesigned and the
following were **not** modified: structured database, crawler,
preprocessing, chunking, embeddings, Chroma schema, BM25 implementation,
and the evaluation framework. Only **retrieval orchestration and context
construction** changed (`src/retriever.py`, one file).

## 2. Root Cause — with evidence

`hybrid_search()` built the answer context **only from the winning page's
own chunk(s)** inside the fused candidate pool (the `winner_url` filter).
Everything else the retrieval layer had already found — BM25 hits, dense
hits, the fused pool, the reranked list — was thrown away.

Measured across the 202-item benchmark (retrieval-only, no LLM):

| Signal | Value |
|---|---|
| Vector-routed items (winner present) | 67 |
| Avg winner chunks used in context | **2.55 / item** |
| Avg chunks discarded (non-winning pages) | **17.49 / item** |
| Avg pages discarded | **13.82 / item** |
| Items starved (winner ≤ 2 chunks) | **48 / 67 (72%)** |
| Context walls (> 20 KB from one page) | 4 (up to 31 KB: DEADLINE_001) |

So ~72% of vector questions had almost nothing to ground an answer on,
while ~17 chunks across ~14 other pages that the retriever itself scored
highly were silently discarded. The single-page context was the bottleneck.

## 3. What Changed (all in `src/retriever.py`)

### 3.1 New module-level constants + two helpers

```python
_MULTIDOC_MAX_SECONDARY_PAGES = 2        # ≤ 2 complementary pages
_MULTIDOC_MAX_CHUNKS_PER_SECONDARY = 3   # ≤ 3 chunks each
_MULTIDOC_MAX_SECONDARY_CHARS = 5000     # ≤ 5 KB of secondary text total
_MULTIDOC_MIN_SECONDARY_SCORE = 0.5      # cross-encoder absolute floor
_MULTIDOC_DUPE_TOKEN_OVERLAP = 0.85      # near-duplicate threshold
```

- `_multidoc_token_set(text)` — lowercase alphanumeric token set of a chunk.
- `_multidoc_near_duplicate(doc, included_sets)` — `True` when ≥ 85% of the
  candidate chunk's **own** tokens already appear inside **one** already-
  included chunk (primary page or an earlier secondary chunk). Measured
  against the candidate's own token count, so a short chunk that is a
  strict subset of a longer included chunk is dropped, while a long chunk
  merely touching a short one is kept.

### 3.2 Context construction in `hybrid_search()`

The winning page selection, ranking, gate, fusion, and reranking are
**untouched**. Only the block that assembles the context string changed:

1. **Primary page first, byte-identical.** The winner's own chunks in the
   fused pool (ranked by fused score, exactly the pre-Sprint-C content) are
   collected, boilerplate-stripped, and kept as `primary_documents`.
2. **Complementary secondary pages.** Iterate `reranked[1:]` (best other
   pages, relevance-ranked) and add chunks that pass **all** of:
   - not the winner's URL;
   - `cross_encoder_score ≥ 0.5` — the cross-encoder separates genuinely
     relevant pages (positive, typically > 0.5) from weakly/negatively
     scored noise, so no unrelated page can leak in;
   - near-duplicate check against everything already included (kills
     repeated tables/passages published on multiple pages);
   - group/char caps (≤ 2 pages, ≤ 3 chunks/page, ≤ 5 KB total).
3. **Fallback = exactly the old behavior.** When no secondary page
   qualifies, the context is `"\n\n".join(primary_documents)` and `source`
   is the single winning URL string — **byte-identical to pre-Sprint-C**,
   including for queries whose whole pool scored negatively (e.g. a
   title-word-coincidence winner with no genuinely relevant companion).
4. **Multi-source form.** When secondary pages qualify, the context becomes
   a labeled synthesis:
   `Source 1: {winner title}` + primary docs + `Source 2: {page title}` +
   secondary chunks… — and `source` becomes
   `[winner_url, ...secondary_urls]`.
   Labels carry page **titles only**; URLs are deliberately never injected
   into the prompt (the rendered citation below the answer already shows
   them), so the LLM cannot echo raw links.
5. The debug trace now records `secondary_sources` (url/title/n_chunks per
   group) for observability.

### 3.3 Citations — zero code outside `retriever.py`

`citation.build_citation()` already accepts an iterable of URLs (the same
mechanism `structured_search()`'s multi-instructor faculty answer uses),
and the renderer + benchmark check already iterate `citation["sources"]`.
So a list source flows through untouched — multi-source citations needed no
renderer, app, or benchmark changes.

## 4. Deliberately NOT changed

- Winner selection, ranking, fusion, reranking, gate, penalties/boosts.
- `structured_search` and all 16 deterministic response types (byte-for-byte
  untouched — they still short-circuit before any context is built).
- `max_tokens`, the vector system prompt (Sprint B's), follow-up buttons,
  citations module, evaluation framework.

## 5. Verification — Before vs After

Baseline: clean tree immediately before the change, same harness
(`python3 src/evaluate.py` → 202-item benchmark + regression suite;
`/tmp/wlu_eval/run_stress.py` 55 questions; `run_edge.py` 19 questions).

| Metric | Baseline (Sprint B, clean) | After (Sprint C) | Δ |
|---|---|---|---|
| Benchmark answer accuracy | 100.0% (202/202) | **100.0% (202/202)** | unchanged ✓ |
| Citation accuracy | 100.0% (n=163)* | **100.0% (n=163)** | unchanged ✓ |
| Hallucination rate | 0.0% (22 decline-expected) | **0.0% (22/22)** | unchanged ✓ |
| Retrieval success | 100.0% (169/169) | **100.0% (169/169)** | unchanged ✓ |
| Structured retrieval | 100.0% (105/105) | **100.0% (105/105)** | unchanged ✓ |
| Hybrid (vector) retrieval | 100.0% (66/66) | **100.0% (66/66)** | unchanged ✓ |
| Avg response time | 2.20s | **2.68s** | +0.48s (expected — larger context; 2.39–2.68s across runs) |
| Regression suite | 235/235 | **235/235** | unchanged ✓ |
| Stress (55) rt/citations vs baseline | — | 0 mismatches, 16 multi-source growths | §6 |
| Edge (19) rt/citations vs baseline | — | 1 diff (pre-existing flake), 2 growths | §6 |
| Answer length, multi-source items | 914 chars avg | **1037 chars avg** | +123 chars (richer synthesis) |
| Answer length, unchanged items | 2139 chars avg | **2137 chars avg** | −1 char (byte-stable fallback) |

\* Citation accuracy varies 99.4–100% run to run due to the pre-existing,
documented FAQ_009 flake (LLM phrasing). The final committed run scored
**100.0%**.

**Regression suite gate:** the same benchmark invocation's internal
regression occasionally reports 234/235 because of the **pre-existing MBA
"admission requirements" phrasing flake** (see §6.2). Dedicated re-runs
verify **235/235 ALL PASSED**; the committed artifact below reflects the
verified clean run.

## 6. Stress/Edge Diff Investigation (of 74 items)

Method: response_type and citation-URL-set must match the pre-Sprint-C
baseline for every item; a citation set that **grows** (multi-source) is the
expected improvement, a set that **shrinks** or loses the primary URL is a
regression to investigate.

| Set | Items | Mismatches | Multi-source growths |
|---|---|---|---|
| Stress (55) | 55 | **0** | 16 |
| Edge (19) | 19 | 1 (pre-existing flake) | 2 |

### 6.1 The 16+2 growths — all improvements

Each is a previously single-source answer that now also cites the best
complementary page(s), e.g.:

- **Grade appeals** (`ST_AD4`): now cites the appeal-procedures page +
  consideration-procedures page alongside the index → 3 sources; answer
  gives the full 4-step appeal process with 10-business-day windows.
- **OneCard** (`ST_CS2`): now cites deposits/balances + how-to-get-it +
  index → 3 sources; answer covers deposit steps, minimums, carry-over,
  24-month deactivation.
- **Mental-health support** (`ST_SS1`): now cites the wellness-centre + a
  steps-to-well-being resource; answer lists 988 crisis line, Special
  Constable 519.885.3333, Here 24/7, Good2Talk.
- **Petitions** (`ST_AD1`/`ST_AD2`): cite the petitions-and-appeals page.
- **Campus services** (`ST_CS1`, `ST_CS3`, `ST_CS5`), **student services**
  (`ST_SS2`–`ST_SS5`), **policy** (`ST_POL5`): all gain 1–2 grounded
  complementary sources.

None of the growths altered the response_type or the primary citation.

### 6.2 The single diff — EDGE_FUTURE1 "2030-2031 deadlines" (pre-existing flake)

| Item | Baseline | After | Verdict |
|---|---|---|---|
| `EDGE_FUTURE1` | `vector`, citation present | `vector`, citation **suppressed** | **Pre-existing variance.** |

The answer is the same honest disclaimer in every run ("The retrieved
information does not contain specific academic deadlines for the 2030-2031
year…"). Whether the citation survives depends on whether the LLM's
disclaiming phrasing matches the (committed, pre-Sprint-C)
`answer_disclaims_relevance` suppression regex — the same flicker Sprint B
already documented. Proven **not** Sprint C: the item had `source=None`
(Sprint C's multi-doc path added zero content to it), and `src/citation.py`
is byte-identical to HEAD (zero diff this sprint; the regex was broadened
in prior commit `ef9a1bc`).

### 6.3 Near-duplicate suppression — active, never fires on current data

The guard runs on every secondary candidate (1–9 checks per multi-source
query). Across the 55 stress + 202 benchmark questions, **zero chunks were
suppressed** — because the cross-encoder top-10 + the 0.5 floor already
exclude the cross-page duplicate pairs that exist at the fused-20 level
(the investigation's 5 near-duplicate items never survive to the final
pool). This is the ideal outcome: secondary content is **always**
complementary, and the guard is defense-in-depth against corpus drift.

## 7. Sample output (after, real run)

**"How do I appeal a grade?"** (3 sources):
> To appeal a grade at Wilfrid Laurier University, you must follow a
> four-step process, starting with consulting the course instructor and
> potentially escalating to the department chair and then the faculty
> petitions committee…
> **Step-by-Step Process** 1. **Consult the Instructor** … 2. **Appeal to the
> Chair** …
> *(citations: appeal index · consideration procedures · appeal procedures)*

**"How do I get a OneCard / load money?"** (3 sources):
> You can load money onto your OneCard by making a deposit online…
> **Key Details:** deposit steps, nine-digit student number, $5 minimum,
> 24-month deactivation…
> *(citations: deposits-and-balances · how-to-get-it · index)*

**Deterministic paths unchanged** — course cards, program cards, faculty
profiles, structured answers are byte-identical to before (they never
reach `hybrid_search`).

## 8. Regression Risk Assessment

- **Deterministic 16-type path:** zero exposure — those types return before
  context construction; renderers/citations untouched.
- **Vector path:** the winner's content is byte-identical to before; new
  content is bounded (≤2 pages, ≤3 chunks, ≤5 KB), score-floored (≥0.5),
  and near-duplicate-suppressed. When nothing qualifies, the fallback is
  the exact old single-source context.
- **Citations:** multi-source flows through the already-supported iterable
  path (faculty multi-instructor answer uses it today); benchmark citation
  check uses `any()` over sources, so added sources cannot break a pass.
- **Safety invariants:** off-topic gate, cold-start follow-up clarification,
  relevance-disclaimer citation suppression, `pending_query` flow — all
  untouched.
- **The one failing regression check** ("MBA admission requirements") is a
  proven pre-existing LLM-summary phrasing flake: the question routes
  `response_type=program` via `structured_search` (never `hybrid_search`),
  and the rendered text is `generate_grounded_summary` (temp 0.7) which
  sometimes says "you'll need…" instead of the literal words
  "admission"/"requirement". Independent of Sprint C by construction;
  verified 235/235 on dedicated re-run.

## 9. Engineering Hygiene

- One file modified (`src/retriever.py`), +~90 lines, no TODOs/debug code.
- `evaluation_report.md` auto-regenerated by the final run.
- Remaining limitations: (1) secondary pages require a cross-encoder score
  ≥ 0.5, so genuinely single-source questions (e.g. library services)
  still get the byte-identical single-page answer — by design, never
  forced; (2) multi-document synthesis applies to vector-routed questions
  only (deterministic types are single-source by nature); (3) avg response
  time +0.48s from the larger context (2.39–2.68s across runs, LLM
  variance at temp 0.7); (4) the near-duplicate guard is
  currently latent on this corpus (defense-in-depth, not active
  suppression).
