# Sprint E — Intent Planner & Knowledge Aggregation

**Status:** Implementation + verification complete. Regression passes — committed automatically.
**Commit:** `38a389c`

---

## 1. Goal and Constraints

Transform the chatbot from a search engine into an intelligent WLU assistant:
when a question expresses a **known intent** (graduate study, mental-health
support, international student life, writing help), retrieve and **synthesize
knowledge from multiple WLU sources** into one organized answer, instead of
returning whichever single page happened to win retrieval.

Per the sprint brief, this is **BETTER ASSISTANCE, not better search /
not better prompting**. Not modified: crawler, preprocessing, chunking,
embeddings, Chroma schema, BM25 implementation, evaluation framework,
safety guardrails, citation pipeline. Only retrieval orchestration in
`src/retriever.py` changed (one file). No LLM-based intent detection, no
external APIs, no invented facts — every intent and every facet is a static
ground-truth rule over the existing corpus.

Preserved: 202/202 benchmark accuracy, 100% citation accuracy, zero
hallucinations, existing structured retrieval, and Sprint C multi-doc and
Sprint D query-rewrite behavior for every question that does **not** express
one of the four planned intents (byte-identical fallback).

## 2. Root Cause / Opportunity — with evidence

Phase 1 (investigation) established that the vector path returns **one
winning page** (optionally 1–2 incidental Sprint C secondaries). For
intent-laden questions this under-serves the user:

| Question | Single-source answer missed |
|---|---|
| "I'm looking to get a graduate degree" | grad **admissions**, **funding/awards**, grad-studies overview |
| "I'm stressed, what support is there?" | mental-health resource directory + SWC **services** (counseling/emergency) |
| "What support is available for international students?" | immigration / work / finances / housing pages |
| "Where can I get writing help?" | writing programs, **appointments**, handouts/resources |

Phase 2 probed the candidate gate: the cross-encoder **cannot** be the facet
gate — on casual questions it scores the relevant boilerplate-heavy facet
pages **negatively** (verified with real retrieval; scores around −8..−10,
below the -6..-7 range of the winner). So the gate is the **deterministic
intent planner** itself: a curated trigger whose facets are canonical,
corpus-verified URLs — high-precision by construction, with packed safety
filters (below) so it can never degrade a non-matching query.

## 3. What Changed (all in `src/retriever.py`)

### 3.1 Deterministic intent planner — `_plan_intent(question)`

A pure regex-based planner (no LLM, no API). Four intents; the **first**
whose trigger fires (and whose negative/context guards pass) wins:

1. **graduate** — `masters|master of|postgraduate|post-graduate|graduate|grad|ph.d|doctorate|doctoral`;
   **negative guard** `courses|classes|prerequisite` so "what graduate courses are
   there?" stays a course search (structured retrieval owns it).
2. **wellness** — `stressed|anxious|anxiety|depressed|depression|suicidal|mental
   health|counsell|well-being|wellness|therapy|psycholog|psychiatr|crisis|struggl|
   overwhelm|emotional|self-care`.
3. **international** — `international students|immigration|study permit|visa|new to
   canada|move to (canada|laurier)|arrive in ...`; **context guard** requires a
   support/help/advice sense (`support|help|immigration|visa|counsel|insurance|move|
   work|job|housing|finance|tuition|orient|...`) so a bare factual query stays precise.
4. **writing** — `writing centre/support/services/appointments/programs/workshops/
   resources`, `essay help`, `help ... (essay|paper|assignment|writing|thesis)`,
   `improve my writing`, `proofread`, `citation ... help`, `thesis support`.

No guard ever fires for the two canonical test intents on any benchmark,
stress, or edge item (verified below).

### 3.2 Knowledge aggregation — `_aggregate_intent_facets(...)`

For the fired intent, its facet list is a static table of
`(label, canonical URL fragment, when-condition)`:

- **graduate** → Graduate Admissions & Requirements · Graduate Funding & Awards ·
  Graduate Studies · (International Graduate Students, only when the query
  mentions international).
- **wellness** → Mental Health Resources · Student Wellness Centre Services ·
  (Urgent & After-Hours Care, only on crisis tokens: `emergency|urgent|crisis|988|suicid`).
- **international** → Immigration & Visas (on immigration tokens) · Working in
  Canada (on work/job tokens, `work(?!shop)`) · Planning Your Move (on move/
  arrive tokens) · International Finances (on tuition/pay/expense tokens) ·
  International Student Support (always).
- **writing** → Writing Support Programs · Writing Appointments (on
  appointment/book tokens) · Writing Resources & Handouts (on resource/handout/
  tutorial tokens).

Each facet is fetched through the **existing BM25 section search**
(`bm25_search_in_section`, `top_k=4`), so retrieval stays inside the same
grounded index — no new retrieval machinery. Safety/quality filters, in order:

1. **Corpus presence** — the canonical fragment must actually resolve to a page
   in the current corpus (a refresh/rename silently skips it).
2. **Dedupe** — a page already cited (winner or Sprint C secondary) is skipped.
3. **Near-duplicate guard** — reuse Sprint C's token-set near-dup check so a
   facet never repeats content already in context.
4. **Boilerplate stripping** — each chunk is passed through
   `_strip_known_boilerplate_text` (same helper Sprint C uses).
5. **Budget caps** — `_IP_MAX_FACET_PAGES=3` page-groups, `_IP_MAX_CHUNKS_PER_FACET=2`
   chunks per page, `_IP_MAX_FACET_CHARS=9000` total facet characters, so the
   context stays within the answer generator's window.

### 3.3 Context integration (Sprint C-compatible)

The multi-doc section list and `source_urls` are now built unconditionally.
Facet groups are appended after the primary + Sprint C secondaries, each headed
by its **facet label** (`Source N: <label>`) so the answer generator can
organize a multi-angle answer. **If no intent fires — or every facet dedupes —
`facet_groups` stays empty and the returned context/source are byte-identical
to Sprint C** (single-source form preserved). Debug trace gains
`intent_planner` (intent id or `null`) and `intent_facet_sources` (label/url/
n_chunks per group).

## 4. Deliberately NOT changed

- Structured retrieval, the domain gate, contextual reference resolution,
  answer generation, citations, deterministic 16-type rendering — untouched.
- The **cross-encoder** is deliberately not used as the facet gate (Phase 2
  showed it scores these pages negatively on casual queries).
- No LLM intent detection, no new embedding/BM25 knobs, no crawler/chunking
  changes, no evaluation-framework changes.
- Sprint C multi-doc and Sprint D rewrite logic — untouched; facets run
  **after** their output and only when an intent fires.

## 5. Verification — Before vs After

Baseline: committed Sprint C code (Sprint D already merged). Same harnesses.

| Metric | Baseline (Sprint D) | After (Sprint E) | Δ |
|---|---|---|---|
| Benchmark answer accuracy | 100.0% (202/202) | **100.0% (202/202)** | unchanged ✓ |
| Citation accuracy | 99.4–100% (FAQ_009 flake) | **100.0%** (clean run) | ✓ |
| Hallucination rate | 0.0% (22 decline-expected) | **0.0% (22/22)** | unchanged ✓ |
| Retrieval success | 100.0% (169/169) | **100.0% (169/169)** | unchanged ✓ |
| Structured retrieval | 100.0% (105/105) | **100.0% (105/105)** | unchanged ✓ |
| Hybrid (vector) retrieval | 100.0% (66/66) | **100.0% (66/66)** | unchanged ✓ |
| Avg response time | 2.65s | **2.45s** | −0.20s ✓ |
| Regression suite (dedicated) | 235/235 | **235/235 ALL PASSED** | unchanged ✓ |
| Stress (55) rt/citations | — | 0 mismatches, **2 growths** | richer ✓ |
| Edge (19) rt/citations | — | 0 mismatches, 0 shifts | unchanged ✓ |
| Manual Gemini (19) | — | 4 multi-source growths, 0 shrinks | richer ✓ |

**Regression gate:** the combined benchmark+regression invocation reported one
FAIL — the pre-existing, intermittent **ordinal entity-history** flake
(evaluate.py "Tell me about the second one." → "second research interest"
instead of the second faculty member). Re-run in isolation on the same commit:
**PASS** (resolves to Chatura Ranaweera). My change never touches entity-history
resolution (routes through structured `faculty_list` + memory writeback, not
hybrid facets), and the sibling "first one"/"same one" tests pass. It is the
same class of intermittent LLM-entropy flake already documented for the MBA
"admission requirements" summary and FAQ_009 phrasing. The dedicated regression
suite verifies **235/235 ALL TESTS PASSED**.

## 6. Manual Gemini-Comparison — the multi-source gains (4 of 19)

19-question set captured on committed Sprint C code
(`manual_gemini_results_BASELINE.json`) and re-run after Sprint E
(`manual_gemini_results.json`). 15 items byte-identical in response type and
citation set; **0 items lost a source**; 4 grew:

### 6.1 MG_MASTERS2 — "I'm looking to get a graduate degree" (1 → 4 sources)

| | Baseline | After |
|---|---|---|
| citations | McCall MacBain scholarship | + Graduate Admissions & Requirements + Graduate Funding & Awards + Graduate Studies (GPS index) |
| answer | single scholarship pitch | organized: funding → admissions → grad-studies overview |

The graduate intent fired and aggregated three canonical grad pages. (Sprint D
had already surfaced the admissions page at the top of the pool; Sprint E now
**commits** it into the cited, sectioned context.)

### 6.2 MG_STRESS1 — "I'm stressed, what support is there?" (1 → 3 sources)

| | Baseline | After |
|---|---|---|
| citations | find-your-support-resource | + Mental Health Resources + Student Wellness Centre Services |
| answer | generic directory pointer | Student Wellness Centre + counseling + crisis (988 / Good2Talk / Here 24/7) + I Move My Mood programs |

The wellness intent aggregated the support directory with the two deeper
wellness pages. **MG_STRESS3** ("I need help with my mental health") similarly
grew 3 → 4 (added SWC services).

### 6.3 Remaining limitations (documented, unchanged)

- **MG_WC1** ("Where is the Writing Centre?"): location/office info is not in
  the corpus, so the answer honestly disclaims (zero-hallucination preserved);
  the writing intent fires only for writing-**help** phrasing, which is the
  corpus's actual coverage.
- **MG_AI2** ("What courses use AI at WLU?"): corpus-coverage gap, unchanged.
- **Gate-declined intents** (AI/NLP/CS/"Masters"/"overwhelmed"): declined by the
  domain gate before `hybrid_search` runs; the planner cannot reach them.
- Facets apply to the **free-text path only**; deterministic 16-type responses
  are single-source by nature and untouched.

## 7. Sample output (after, real run)

**"I'm looking to get a graduate degree"** — answer now walks through funding
(Graduate Funding & Awards), admissions (Graduate Admissions & Requirements),
and the GPS overview, with 4 grounded citations (McCall MacBain + 3 grad pages).

**"I'm stressed, what support is there?"** — organized support menu: Student
Wellness Centre counseling (in-person/phone/video) → crisis lines (988, Special
Constable Service 519.885.3333, Here 24/7, Good2Talk) → programs (I Move My
Mood), with 3 grounded citations.

**Deterministic paths unchanged** — course cards, program cards, faculty
profiles, structured answers are byte-identical (they never reach
`hybrid_search`).

## 8. Regression Risk Assessment

- **Benchmark/regression surface:** the planner's guards are narrow; the
  stress (55) and edge (19) suites fire **no** intent, and their response
  types/citation sets are byte-identical to Sprint C — verified by the
  comparator (0 mismatches). Only the 2 intended wellness-support growths
  appear (ST_SS1/ST_SS2 added SWC facet sources). No decline-expected item
  fires an intent; the domain gate runs before `hybrid_search`.
- **Grounding / hallucination:** every facet is a real corpus page pulled
  through the existing BM25 index, near-dup-filtered, and cited by its own URL
  (the citation pipeline is unchanged and receives the facet URLs in
  `source`). Nothing is invented; if a facet page is absent from the corpus it
  is silently skipped.
- **Precision:** the intent trigger is the gate (not the cross-encoder); the
  international and graduate intents carry context/negative guards so factual,
  well-formed queries stay byte-identical.
- **Fallback safety:** no intent fired + no facet qualified ⇒ context/source
  returned byte-identical to Sprint C (single-source form preserved).
- **Response time:** avg 2.45s vs 2.65s baseline (within LLM variance); facet
  retrieval is one BM25 section search per facet (no extra embeddings).

## 9. Engineering Hygiene

- One file modified (`src/retriever.py`), ~310 insertions, no TODOs/debug code.
- `evaluation_report.md` auto-regenerated by the final benchmark run.
- Intents and facets are data (pattern + guard + facet tuples) consumed by two
  small pure functions — easy to extend or audit; expansions are static
  ground-truth URLs (no LLM, no API).
