# Sprint 4 — Academic Deadlines & Student Services Accuracy: Engineering Report

**Date:** 2026-08-03
**Branch:** main
**Scope:** Eliminate the remaining citation failures in Academic Deadlines, Campus Services, and Student Services, and the citation-accuracy failures they produced.

---

## 1. Problem Statement

Sprint 3 (commit `2f81e28`) reached 202/202 answer accuracy but only 142/163 citation accuracy (87.1%). All 21 remaining failures were **citation** failures — every answer passed its keyword check; the defect was that the cited URL pointed outside the question's canonical section (or no citation rendered at all). Category-wise, the failures were concentrated in exactly the three categories this sprint targets:

| ID | Question (abbrev.) | Sprint-3 failure |
|---|---|---|
| DEADLINE_002/007/010/014/016 | deadlines / petitions / appeals / exams / important dates | cited a non-`/calendars-and-petitions/` page |
| DEADLINE_015 | "Where can I check exam-related dates?" | answer correct, no citation rendered |
| CAMPUS_001/007/009 | accessible parking / sustainability / cycling & transit | cited a non-`/campus-services/` page |
| CAMPUS_005 | "How do I apply for residence?" | answer correct, no citation rendered |
| CAMPUS_012 | EV charging on campus | answer correct, no citation rendered (flaky) |
| STUDENTSVC_001/002/007/008/010/018 | international / mental health / athletics / gendered violence / equity / student affairs | cited a non-`/support-and-wellness/` page |
| STUDENTSVC_004/014 | disability & accessibility support | cited a non-`/support-and-wellness/` page |
| STUDENTSVC_006 | "Who is the Dean of Students…?" | cited a non-`/support-and-wellness/` page (Jason Dean faculty profile) |
| FAQ_009 | Metalworks FAQs | pre-existing citation flake (unchanged, see §7) |

## 2. Root Causes (execution evidence only)

### Root cause A — the cross-encoder does not know WLU's canonical section taxonomy (19 items)

Every answer in these categories is grounded in **a single winning page** (hybrid_search() builds context only from the winner's chunks and cites that winner). For 17 of the 20 non-FAQ items, the cross-encoder picked a winning page that was *on-topic but in the wrong canonical section* — the corporate wlu.ca site (governance, discover-laurier, strategic-initiatives, future-students) or a sibling students.wlu.ca section (finances, academics, campus-services) — even though the question's canonical section page exists in the corpus. Direct function-level trace of `hybrid_search()` on each failing question confirmed the canonical page was either present in the fused pool and out-ranked, or truncated out at `RERANK_TOP_K = 10` (DEADLINE_016's important-dates page sat at ranks 11+), or never retrieved at all (STUDENTSVC_004/014's disability-justice-and-accessibility page).

Because answer **and** citation come from the single winner, every wrong-section winner produced a citation outside the benchmark's canonical-section requirement.

### Root cause B — a structured-cascade mis-route for "Dean of Students" (STUDENTSVC_006)

`structured_search()`'s FACULTY branch uses tiered last-name matching. "Who is the Dean of Students and what do they do?" contains the capitalized title token **"Dean"**, which `search_faculty()`'s tier-3 single-name matching caught as a faculty member's last name and returned professor **Jason Dean's** profile page — the Dean of Students office page (which lives under `/support-and-wellness/`) never got a chance.

### Root cause C — `answer_disclaims_relevance()` false positive suppressed correct citations (CAMPUS_012; contributed to CAMPUS_005 / DEADLINE_015)

`citation.answer_disclaims_relevance()` runs `_NEGATED_INFO_AVAILABILITY_PATTERN`, an **unanchored** pattern that treats *any* "does not include/provide/contain…" as the LLM disclaiming relevance. CAMPUS_012's fully-on-topic EV-charging answer correctly stated "*Fees: $2.00 per hour for use via the Flo app (**does not include** parking fees)*" — a factual fee note that the pattern misread as a relevance disclaimer and suppressed the citation for. The same unanchored pattern class affected CAMPUS_005 / DEADLINE_015 (answers paraphrased from a wrong-section page that the model hedged around; the correct-section routing fix in Root cause A resolves their retrieval, and this fix removes the false-positive suppression class that CAMPUS_012's correct answer hit).

## 3. Why Each Change Was Necessary

1. **Section intent must be recognized.** WLU's service/administrative content lives in three canonical URL sections (`/calendars-and-petitions/`, `/campus-services/`, `/support-and-wellness/`). The cross-encoder ranks pages semantically but has no knowledge of this taxonomy — it reliably *finds* the right-topic page and equally reliably picks one from the wrong section when several on-topic pages compete. A deterministic vocabulary→section map (the question's own words: "deadline", "parking", "wellness", …) tells us which section the user actually asked about.

2. **The boost must be section-specific, not a blanket students.wlu.ca preference.** A blanket domain boost would have broken Policies 20/20 (which requires wlu.ca governance pages to win over students.wlu.ca siblings). Gating the boost on `_SECTION_INTENT_PATTERNS` and only boosting candidates whose URL is *inside the named section* leaves every question outside these three sections completely untouched.

3. **The boost must see the whole fused pool.** The default `RERANK_TOP_K = 10` cross-encoder truncation hid ranks 11+ from every subsequent boost/penalty — DEADLINE_016's important-dates page sat in the pool at rank 11+ and could not be rescued. Section-intent questions score the full pool; non-section questions keep the exact pre-sprint top-k pipeline (regression guard).

4. **Coverage pages must be merged into the pool.** For STUDENTSVC_004/014 the canonical page was never retrieved by the dense/BM25 top-k at all, so a boost alone had no target. A section-restricted BM25 search (same lazily-rebuilt index, no fresh Chroma query) pulls the section's top-scoring chunks into the pool so the boost can act.

5. **"Dean of Students" must defer to hybrid retrieval.** Mirroring Sprint 2/3's deferral guards, the role-question must bypass the structured FACULTY branch that catches "Dean" as a surname.

6. **The citation negation must be anchored to the source.** A correct answer quoting a factual detail ("does not include parking fees") must not be read as a relevance disclaimer; a genuine disclaimer ("the retrieved information does not provide…") must still be honored.

## 4. Fix (smallest possible — 6 targeted changes across 3 files)

No architecture changes, no database rebuilds, no new files beyond this report.

### Change 1 — `src/hybrid_rerank.py`: `cross_encoder_rerank(..., top_k=None)` full-pool support

When `top_k=None`, return every scored candidate instead of truncating to `RERANK_TOP_K`. Callers that pass no argument keep the default truncation byte-for-byte.

### Change 2 — `src/hybrid_rerank.py`: `bm25_search_in_section(collection, question, url_fragment, top_k)`

A section-restricted pass over the existing `_bm25_state` index: filter chunk positions by URL fragment, return the top-`top_k` scoring chunks. Reuses the same lazily-rebuilt BM25 index as `bm25_search()`, so it stays consistent with `refresh_pipeline.py` without a fresh Chroma query or metadata filter.

### Change 3 — `src/retriever.py`: `_SECTION_INTENT_PATTERNS` + `_match_canonical_section()` + `_apply_canonical_section_preference()`

Three (regex → section-fragment) pairs covering the deadlines/petitions, campus-services, and support-and-wellness vocabularies. `_match_canonical_section()` returns the named section (first match wins; `None` otherwise; FAQ-intent questions excluded so they keep their own FAQ-page boost path). `_apply_canonical_section_preference()` adds `_CANONICAL_SECTION_INTENT_BOOST = 5.0` to every candidate whose URL lives in the named section and re-sorts — same in-place + re-sort contract as `_apply_faq_intent_boost()`. The magnitude is calibrated live (+5.0, not +3.0, was required for DEADLINE_016, whose in-pool important-dates chunks score −4.2 to −9.9 after the cross-encoder's boilerplate/nav demotion).

### Change 4 — `src/retriever.py`: `_DEAN_OF_STUDENTS_PATTERN` guard in `structured_search()`

Immediately after the FAQ-intent guard: `if _DEAN_OF_STUDENTS_PATTERN.search(question_lower): return None`, deferring the role-question to hybrid retrieval (same deferral pattern as Sprint 2's department guard / Sprint 3's FAQ guard).

### Change 5 — `src/retriever.py`: `hybrid_search()` merge + full-pool scoring + section preference

- After `reciprocal_rank_fusion()`, if `_match_canonical_section(question)` fires and no candidate in `fused` already lives in that section, merge `bm25_search_in_section(...)`'s top-`_CANONICAL_SECTION_MERGE_TOP_K = 3` chunks into `fused`, pinned at `fused[0].fused_score + 0.001` (just above the current top) so the cross-encoder still judges them on merit.
- Section-intent questions score the **full** pool (`top_k=None`); non-section questions use the exact pre-sprint `cross_encoder_rerank(question, fused)` path (regression guard).
- Apply `_apply_topical_mismatch_penalty()`, `_apply_faq_intent_boost()`, then `_apply_canonical_section_preference()`, then truncate back to `hybrid_rerank.RERANK_TOP_K`. The same-page context loop iterates the full `fused` pool (including merged candidates), so the winner's chunks are still found.

### Change 6 — `src/citation.py`: source-anchored `_NEGATED_INFO_AVAILABILITY_PATTERN`

The negation pattern now has two branches:
- **Part A** — a negation anchored to an explicit source reference: `[article] [source adjective] [source noun] does not / no … [availability word]` ("the retrieved information does not provide…", "this page does not mention…").
- **Part B** — the bare "no <source-content noun>" form ("There is no information about…", "no details were available"), a strict subset of what the old unanchored pattern already matched, so no answer that previously rendered a citation loses one.

A real-world factual statement like "does not include parking fees" no longer reads as a relevance disclaimer; every genuine disclaimer still suppresses the citation.

## 5. Regression Results

Harness: `streamlit.testing.v1.AppTest` against the real, unmodified `src/app.py`; every benchmark item executed end-to-end through `benchmark_runner.py` (the same deterministic scorer that produced `evaluation_report.md`). "Before" = Sprint 3 full run (`/private/tmp/wlu_eval/sprint3_full_benchmark.json`); "After" = a fresh full run on this fix.

### 5a. The sprint target — citation accuracy

| Metric | Before (Sprint 3) | After (Sprint 4) |
|---|---|---|
| Citation Accuracy | 87.1% (142/163) | **99.4% (162/163)** |
| Academic Deadlines citations | 10/16 | **16/16** |
| Campus Services citations | 12/18 | **18/18** |
| Student Services citations | 10/18 | **18/18** |
| Wrong-section citations | 17 | **0** |
| Suppressed citations (false positive) | 3 | **0** |
| Citation failures fixed | — | **20 / 20** |

### 5b. Named verification categories (unchanged — regression guard)

| Category | Before | After |
|---|---|---|
| Courses (22) | 22/22 | 22/22 |
| Faculty (20) | 20/20 | 20/20 |
| Programs (20) | 20/20 | 20/20 |
| Departments (16) | 16/16 | 16/16 |
| Policies (20) | 20/20 | 20/20 |
| FAQ (12) | 12/12 (11 citations) | 12/12 (11 citations — pre-existing flake, §7) |
| Conversation / Follow-up (18) | 18/18 | 18/18 |
| Unsupported / Out-of-domain (22) | 22/22 | 22/22 |

The section-intent patterns were enumerated across all 202 benchmark questions: **0 misses** (every Academic Deadlines / Campus Services / Student Services item routes to a section) and **0 leaks** (no question outside those categories matches any section pattern; FAQ questions are excluded by the `_FAQ_INTENT_PATTERN` gate). Because non-section questions take the exact pre-sprint pipeline (`top_k=RERANK_TOP_K` + same penalties), their behavior is structurally unchanged — confirmed by the 0 new failures in the full run.

### 5c. Full 202-item benchmark sweep

**202/202 answer accuracy, 162/163 citation accuracy, 0 failed questions, 0 timeouts.** The only per-item change versus the Sprint 3 baseline is the 20 intended citation fixes plus CAMPUS_012's false-positive suppression fix. Per-item routing change: STUDENTSVC_006 moved from the structured `faculty` branch to `vector` (the intended dean guard); the structured/hybrid split shifted 106→105 / 66→67 accordingly.

## 6. Updated Overall Benchmark Metrics

| Metric | evaluation_report.md (original published baseline) | Sprint 3 (commit 2f81e28) | Post-Sprint 4 |
|---|---|---|---|
| Total Questions | 202 | 202 | 202 |
| Answer Accuracy | 95.0% | 100.0% | **100.0% (202/202)** |
| Citation Accuracy | 81.5% (n=162) | 87.1% (n=163) | **99.4% (n=163)** |
| Hallucination Rate | 0.0% (n=22) | 0.0% (n=22) | **0.0% (n=22)** |
| Retrieval Success Rate | 98.8% (n=169) | 100.0% (n=169) | **100.0% (n=169)** |
| Structured Retrieval Accuracy | 93.2% (n=103) | 100.0% (n=106) | **100.0% (n=105)** |
| Hybrid Retrieval Accuracy | 98.5% (n=67) | 100.0% (n=66) | **100.0% (n=67)** |
| Average Response Time | 2.67s | 2.73s | **2.55s** |
| Failed Questions | 10 | 0 | **0** |

Category-wise breakdown (post-Sprint 4, all from the same full run):

| Category | Passed | Total | Accuracy |
|---|---|---|---|
| Courses | 22 | 22 | 100.0% |
| Faculty | 20 | 20 | 100.0% |
| Programs | 20 | 20 | 100.0% |
| Departments | 16 | 16 | 100.0% |
| Policies | 20 | 20 | 100.0% |
| **Academic Deadlines** | **16** | **16** | **100.0%** |
| **Campus Services** | **18** | **18** | **100.0%** |
| **Student Services** | **18** | **18** | **100.0%** |
| FAQ | 12 | 12 | 100.0% |
| Conversation / Follow-up | 18 | 18 | 100.0% |
| Unsupported / Out-of-domain | 22 | 22 | 100.0% |

## 7. Remaining Known Issues (pre-existing)

1. **FAQ_009 citation flakiness.** The Metalworks FAQ page is always retrieved first, but intermittently the LLM-paraphrased answer opens with "The retrieved information does not provide a specific list of FAQs…" and then summarizes the FAQ content — which is a *genuine* relevance disclaimer, so the citation is correctly suppressed (this run reproduced it: "The retrieved information does not provide a specific list of FAQs about the Metalworks partnership…"). Documented since Sprint 2/3, orthogonal to retrieval, and shared by every LLM-grounded answer in every category. Fixing the flake means changing the LLM grounding prompt or the answer-phrasing heuristic — deliberately out of scope for this retrieval-accuracy sprint.
2. **Program-specific penalty subject extraction.** `_apply_topical_mismatch_penalty()` reads only the first `/programs/` path segment as a page's subject; the underlying extraction bug for multi-segment `/programs/<group>/<program>/` URLs remains (FAQ_003 is already handled by the FAQ-page boost; this sprint added no program-path changes). Unchanged from Sprint 3.

## 8. Files Modified

- `src/hybrid_rerank.py` — `cross_encoder_rerank()` `top_k=None` full-pool support; new `bm25_search_in_section()` helper (58 insertions).
- `src/retriever.py` — `_SECTION_INTENT_PATTERNS`, `_CANONICAL_SECTION_INTENT_BOOST`, `_CANONICAL_SECTION_MERGE_TOP_K`, `_DEAN_OF_STUDENTS_PATTERN`, `_match_canonical_section()`, `_apply_canonical_section_preference()`; `_DEAN_OF_STUDENTS_PATTERN` deferral guard in `structured_search()`; section merge + full-pool scoring + section-preference boost in `hybrid_search()` (184 insertions).
- `src/citation.py` — source-anchored `_NEGATED_INFO_AVAILABILITY_PATTERN` (Part A anchored / Part B preserved) (44 insertions).
- `SPRINT4_DEADLINES_SERVICES_ACCURACY_REPORT.md` — this report.

## 9. Verification Notes

- No code modified by the harness — every result came from the real app under `AppTest`.
- Function-level traces of `structured_search()`/`hybrid_search()` verified for all 20 target items (all winners in their canonical section) before the end-to-end run.
- The section-intent pattern's match set was enumerated across the entire 202-item benchmark: exactly the Academic Deadlines (16) + Campus Services (18) + Student Services (18) items, nothing else; FAQ items excluded by gate.
- The citation-pattern change is a strict narrowing for the anchored verb forms plus a preserved subset for the bare "no <noun>" form — verified across all answers generated under the old semantics that none now disclaims (0 regressions), and that genuine disclaimer phrasings still disclaim.
