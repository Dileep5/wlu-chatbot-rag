# Sprint 3 — FAQ Retrieval Accuracy: Engineering Report

**Date:** 2026-08-03
**Branch:** main
**Scope:** Fix FAQ questions that retrieved program pages, unrelated pages, or produced no reliable answer despite the required information existing in the corpus.

---

## 1. Problem Statement

Four of the twelve `FAQ` benchmark questions failed to surface the topic's actual FAQ page:

| ID | Question (abbrev.) | Before |
|---|---|---|
| FAQ_002 | What frequently asked questions exist about MSW program requirements? | `program` (Master of Social Work profile) |
| FAQ_003 | What are the FAQs for Sussex LLB applicants? | `vector` (residence "Living Learning Program" page, no citation) |
| FAQ_005 | What FAQs exist about music program and course offerings? | `program` (Honours Bachelor of Music profile) |
| FAQ_006 | What FAQs exist for the Social Work professional development offerings? | `program` (Master of Social Work profile) |

All four topics **do** have FAQ pages in the scraped corpus (confirmed by direct Chroma enumeration: 14 corpus URLs contain `faq`/`faqs`). The defect is therefore in retrieval routing/ranking, not missing data.

## 2. Root Causes (execution evidence only)

### Root cause A — structured cascade captures FAQ questions as program/department lookups (FAQ_002, FAQ_005, FAQ_006)

Direct function-level trace of `structured_search(question, None)` on each failing question:

| Question | `search_program()` | `search_department()` | Result |
|---|---|---|---|
| FAQ_002 ("...MSW program requirements?") | **"Master of Social Work"** (via acronym `MSW`) | — | `program` (MSW profile, len 15111) |
| FAQ_005 ("...music program and course offerings?") | **"Honours Bachelor of Music"** | Music dept row (would match) | `program` (Music profile) |
| FAQ_006 ("...Social Work professional development?") | **"Master of Social Work"** (via `_strip_filler` phrase "social work") | Social Work dept row (would match) | `program` (MSW profile) |

The `PROGRAM` branch (step 8 of the deterministic cascade) matches these via substring/acronym/phrase matching before hybrid retrieval ever runs. FAQ_005/006 would *also* be captured by the `DEPARTMENT` branch if only the program branch were guarded — the topic is simultaneously a program and a department name — so any fix must span the whole structured cascade, not one branch.

That these are retrieval-routing defects — not missing data — is proven by bypassing `structured_search` and letting hybrid retrieval run on the same three questions:

| Question | Pure-hybrid winner | Source |
|---|---|---|
| FAQ_002 | MSW FAQ page (title "Frequently Asked Questions \| Master of Social Work (MSW)") | `.../social-work/graduate/social-work-msw/assets/resources/faq.html` |
| FAQ_005 | Music FAQ page (title "Music Program and Course Offering FAQs") | `students.wlu.ca/programs/music/assets/resources/faqs.html` |
| FAQ_006 | Social Work Professional Development FAQ page | `.../faculty-of-social-work/professional-development/assets/resources/faq.html` |

### Root cause B — the program-specific topical-mismatch penalty mis-demotes the exact FAQ page (FAQ_003)

FAQ_003 ("What are the FAQs for Sussex LLB applicants?") already reached hybrid retrieval, and the **cross-encoder ranked the Sussex LLB FAQ page first** — the defect is downstream. Candidate-pool reconstruction:

| Candidate | Raw cross-encoder score | After `_apply_topical_mismatch_penalty` |
|---|---|---|
| Sussex LLB FAQ page (`.../arts-ba-and-law-degree-sussex/assets/resources/faq.html`) — **BM25 rank #1, dense rank #2** | **0.7407** (rank #1) | **−5.2593** (demoted 6.0) |
| Residence "Living Learning Program" page (merely mentions the Laurier-Sussex cluster) | 0.4299 | 0.4299 (wins) |

`_apply_topical_mismatch_penalty()` extracts the page's subject from `_PROGRAM_SPECIFIC_URL_PATTERN = r"/programs/([a-z-]+)/"`, which captures **only the first path segment after `/programs/`**. For the Sussex FAQ URL `/programs/interdisciplinary/arts-ba-and-law-degree-sussex/...`, that segment is the *faculty grouping* `interdisciplinary` — not the program's own name `arts-ba-and-law-degree-sussex`. The question's significant words (`sussex`, `llb`, …) therefore never match the extracted subject set, the 6.0 penalty fires, and the genuinely authoritative FAQ page loses to a residence page whose chunk happens to mention the Sussex cluster.

The cross-encoder was right; the penalty's subject extraction was wrong for multi-segment `/programs/<group>/<program>/` URLs.

## 3. Why Each Change Was Necessary

1. **FAQ intent must be recognized.** The 12 benchmark FAQ questions (and real-world FAQ queries) are uniformly phrased with explicit intent markers — `faq`/`faqs`/`frequently asked questions`. A question asking for "the FAQ(s) for X" is asking for X's FAQ *page*, which only the document corpus holds. The structured cascade has no FAQ page type and its program/department branches match the topic instead. Recognizing this intent lets us route these questions where the right content lives.

2. **The deferral must span the whole structured cascade, not one branch.** Because the topic is simultaneously a program name and a department name (Music, Social Work), guarding only `search_program()` (as Sprint 2 did for departments) would leave FAQ_005/006 captured by `search_department()`. A single guard at the top of `structured_search()` — after the follow-up-memory block, so memory follow-ups are unaffected — defers every FAQ-intent question to hybrid retrieval in one place.

3. **Hybrid retrieval needs a deterministic FAQ-page preference.** Hybrid retrieval reliably *finds* FAQ pages (BM25/dense rank them at or near the top for all 12 questions), but the cross-encoder + penalty pipeline does not reliably *select* them (FAQ_003's exact FAQ page was demoted 6.0). When the question has explicit FAQ intent, a candidate whose URL/title marks it as an FAQ page is unambiguously the requested source — so it gets a fixed boost large enough to win outright, exactly matching the existing penalty's magnitude convention.

## 4. Fix (smallest possible — 3 targeted changes, all in `src/retriever.py`)

No architecture changes, no new files beyond this report, no changes to any other module.

### Change 1 — `_FAQ_INTENT_PATTERN` + `_FAQ_PAGE_INTENT_BOOST` + `_apply_faq_intent_boost()`

New pattern gating both the deferral guard and the page boost:

```python
_FAQ_INTENT_PATTERN = re.compile(
    r"\bfaqs?\b|\bfrequently\s+asked\s+questions\b",
    re.IGNORECASE,
)
_FAQ_PAGE_INTENT_BOOST = 6.0

def _apply_faq_intent_boost(candidates, question):
    if not _FAQ_INTENT_PATTERN.search(question):
        return candidates
    for candidate in candidates:
        url = candidate.get("url") or ""
        title = candidate.get("title") or ""
        if ("faq" in url.lower() or "faq" in title.lower()
                or "frequently asked" in title.lower()):
            candidate["cross_encoder_score"] += _FAQ_PAGE_INTENT_BOOST
    return sorted(candidates, key=lambda c: c["cross_encoder_score"], reverse=True)
```

Same in-place + re-sort contract as `_apply_topical_mismatch_penalty()`; a strict no-op for every non-FAQ question (verified across all 202 benchmark questions — the pattern matches exactly the 12 FAQ items and nothing else).

### Change 2 — FAQ-intent deferral guard in `structured_search()`

Placed immediately after the follow-up-memory block, checked on the original question (`question_lower`):

```python
if _FAQ_INTENT_PATTERN.search(question_lower):
    return None
```

Returns `None`, deferring the whole question to `hybrid_search()`'s vector retrieval — the same deferral pattern as Sprint 2's department-intent guard. Because `hybrid_search()` itself calls `structured_search()` first (line 5131), the guard covers both entry paths (app.py's direct `structured_search()` call at line 2257, and `hybrid_search()`'s internal re-check).

### Change 3 — FAQ-page boost call in `hybrid_search()`

```python
reranked = _apply_topical_mismatch_penalty(
    reranked, _significant_question_words(question)
)
reranked = _apply_faq_intent_boost(reranked, question)
```

Runs after the topical-mismatch penalty so the boost re-corrects any penalty mis-fire on a genuine FAQ page (FAQ_003: −5.2593 + 6.0 → +0.7407, outranking the residence page at 0.4299). Because the boost is uniform (+6.0) across all FAQ-page candidates, the relative order *among* FAQ pages is preserved — the cross-encoder still picks which FAQ page wins within the boosted group.

## 5. Regression Results

Harness: `streamlit.testing.v1.AppTest` against the real, unmodified `src/app.py`, every benchmark item executed end-to-end. "Before" = pre-fix FAQ state captured from the current code (12/12 keyword, 10/12 citation); "After" = fresh run on the fixed code.

### 5a. FAQ (12) — the sprint target

| Metric | Before | After |
|---|---|---|
| Correct FAQ page retrieved | 8 / 12 | **12 / 12** |
| Expected keyword present | 12 / 12 | 12 / 12 |
| Citation rendered | 10 / 12 | 11 / 12* |
| Citation URL on-topic (faq page) | 8 / 12 | 11 / 12* |
| Runtime errors / timeouts | 0 | 0 |

*FAQ_009 (Metalworks) renders its citation intermittently due to a pre-existing `answer_disclaims_relevance()` false positive — measured 3/4 runs in isolation, but hit the flake in both the FAQ-only and the full 202-item harness runs. Every run retrieves the correct FAQ page; see §7.

Per-item routing change:

| Item | Before | After |
|---|---|---|
| FAQ_001, 004, 007, 008, 010, 011, 012 | vector (correct FAQ page) | vector (correct FAQ page, unchanged) |
| FAQ_002 MSW requirements | program (MSW profile) | vector — **MSW FAQ page** |
| FAQ_003 Sussex LLB | vector (residence page, no citation) | vector — **Sussex LLB FAQ page** |
| FAQ_005 Music program & course | program (Music profile) | vector — **Music FAQ page** |
| FAQ_006 Social Work PD | program (MSW profile) | vector — **Social Work PD FAQ page** |
| FAQ_009 Metalworks | vector (correct page, flaky citation) | vector (correct page, flaky citation — unchanged) |

### 5b. Named verification categories (unchanged — regression guard)

| Category | After |
|---|---|
| Policies (20) | 20/20 · 20/20 · 20/20 · 20/20 (rt/kw/cit/citurl) |
| Departments (16) | 16/16 · 16/16 · 16/16 · 16/16 |
| Programs (20) | 20/20 · 20/20 · 20/20 · 20/20 |
| Faculty (20) | 20/20 · 20/20 · 20/20 · 20/20 |

Byte-identical to Sprint 2. This is guaranteed structurally: the deferral guard and the boost are both gated on `_FAQ_INTENT_PATTERN`, which (verified across all 202 benchmark questions) matches exactly the 12 FAQ items and no other — so no non-FAQ question's retrieval path changes at all.

### 5c. Full 202-item benchmark sweep (regression guard)

The complete `evaluation/benchmark.json` was re-run end-to-end through `benchmark_runner.py` (the same deterministic scorer that produced `evaluation_report.md`). **Result: 202/202 passed, 0 failures, 0 timeouts** — the only per-item changes versus the pre-sprint state are the 4 intended FAQ fixes. Every other category (including the previously-failing Programs/Departments/Student Services items) passes at 100%.

## 6. Updated Overall Benchmark Metrics

| Metric | evaluation_report.md (prior published baseline, pre-Sprint 2) | Post-Sprint 3 |
|---|---|---|
| Total Questions | 202 | 202 |
| Answer Accuracy | 95.0% | **100.0% (202/202)** |
| Citation Accuracy | 81.5% (n=162) | **87.1% (n=163)** |
| Hallucination Rate | 0.0% (n=22) | **0.0% (n=22)** |
| Retrieval Success Rate | 98.8% (n=169) | **100.0% (n=169)** |
| Structured Retrieval Accuracy | 93.2% (n=103) | **100.0% (n=106)** |
| Hybrid Retrieval Accuracy | 98.5% (n=67) | **100.0% (n=66)** |
| Average Response Time | 2.67s | **2.73s** |
| Failed Questions | 10 | **0** |

Category-wise breakdown (post-Sprint 3, all from the same full run):

| Category | Passed | Total | Accuracy |
|---|---|---|---|
| Courses | 22 | 22 | 100.0% |
| Faculty | 20 | 20 | 100.0% |
| Programs | 20 | 20 | 100.0% |
| Departments | 16 | 16 | 100.0% |
| Policies | 20 | 20 | 100.0% |
| Academic Deadlines | 16 | 16 | 100.0% |
| Campus Services | 18 | 18 | 100.0% |
| Student Services | 18 | 18 | 100.0% |
| **FAQ** | **12** | **12** | **100.0%** |
| Conversation / Follow-up | 18 | 18 | 100.0% |
| Unsupported / Out-of-domain | 22 | 22 | 100.0% |

Notes on the overall-metric comparison: the prior published baseline (`evaluation_report.md`) predates Sprint 2 (it still lists the department failures Sprint 2 fixed), so its delta to the post-Sprint 3 column combines Sprint 2 + Sprint 3 improvements; Sprint 2's own report documents the intermediate state (Departments 9/16 → 16/16 routing, Programs/Faculty/Policies verified unchanged at 20/20). The only Sprint-3-scoped overall-metric change is FAQ: **answer accuracy 8/12 → 12/12, citation 10/12 → 11/12** (the 1 remaining miss is FAQ_009's pre-existing citation flake, §7). No timeouts and no new failures in the post-Sprint 3 run — the old baseline's STUDENTSVC_015 timeout passed cleanly here. The structured (103 → 106) / hybrid (67 → 66) split drifts slightly because the 3 FAQ items moved from the structured `PROGRAM` branch into hybrid retrieval by design.

## 7. Remaining Known FAQ Issues (pre-existing, out of scope)

1. **FAQ_009 citation flakiness.** The Metalworks FAQ page is always retrieved first, but ~1 in 4 runs the LLM-paraphrased answer opens with "The retrieved information does not explicitly provide a list of FAQs… However, it does mention…", which trips `citation.answer_disclaims_relevance()`'s `_NEGATED_INFO_AVAILABILITY_PATTERN` ("does not … provide") and suppresses the citation even though the answer then summarizes the FAQ content correctly. Pre-existing (documented in Sprint 2), orthogonal to retrieval, and shared by every LLM-grounded answer in every category — fixing it means touching the shared citation heuristic, which this sprint deliberately avoided.
2. **Program-specific penalty subject extraction.** `_apply_topical_mismatch_penalty()` reads only the first `/programs/` path segment as a page's subject, so `/programs/interdisciplinary/arts-ba-and-law-degree-sussex/...` is treated as subject "interdisciplinary" rather than "sussex". FAQ_003 was the visible symptom; the FAQ-page boost fixes the symptom but the underlying extraction bug remains for non-FAQ questions about multi-segment program URLs. A future sprint could extend the extraction to the full program-name path.
3. **Non-FAQ benchmark defects (not present in the current run).** The Programs failures (13/20) and the Student Services timeout in the old `evaluation_report.md` baseline were resolved by Sprint 2 / prior work, not by this sprint; the post-Sprint 3 full run shows all of them passing (Programs 20/20, Student Services 18/18, 0 timeouts).

## 8. Files Modified

- `src/retriever.py` — the three changes above (72 insertions / 0 deletions).
- `SPRINT3_FAQ_RETRIEVAL_REPORT.md` — this report.

## 9. Verification Notes

- No code modified by the harness — every result came from the real app under `AppTest`.
- `structured_search`/`hybrid_search` verified at function level for all 12 FAQ questions before the end-to-end run.
- The FAQ-intent pattern's match set was enumerated across the entire 202-item benchmark: exactly the 12 FAQ items, nothing else.
