# Sprint 2 — Department Retrieval Accuracy: Engineering Report

**Date:** 2026-08-03
**Branch:** main
**Scope:** Fix department questions that retrieved the matching academic *program* page instead of the *department* page.

---

## 1. Problem Statement

Seven of the sixteen `Departments` benchmark questions answered with a **program profile** (`response_type = "program"`) instead of a **department overview** (`response_type = "department_profile"`):

| ID | Question (abbrev.) | Before |
|---|---|---|
| DEPT_001 | Tell me about the Ancient Studies department. | `program` |
| DEPT_005 | Tell me about the Communication Studies department. | `program` |
| DEPT_006 | Tell me about the Criminology Minor (Faculty of Human and Social Sciences) department. | `program` |
| DEPT_010 | Tell me about the Film Studies department. | `program` |
| DEPT_011 | Tell me about the General BSc without Designation department. | `program` |
| DEPT_013 | Tell me about the International Public Policy department. | `program` |
| DEPT_014 | Tell me about the Management Option (LSBE) department. | `program` |

The keyword metric masked the defect: every program profile contains the department's subject words (e.g. the *Honours BA Ancient Studies* program page contains "Ancient Studies"), so keyword matching passed 16/16 even while 7/16 were the wrong page type.

## 2. Root Cause

Two independent defects, confirmed by direct function-level evidence (`search_program` vs `search_department` on each failing question):

### 2a. Cascade ordering — the primary cause (7/7 failures)

`structured_search()` tries `search_program()` (step 7 of the deterministic cascade) **before** `search_department()` (step 11). `search_program()`'s matching is substring/normalized-phrase based: the stored program name *"Honours BA Ancient Studies"* normalizes (via `_strip_filler`) to `ancient studies`, which is a substring of the question *"Tell me about the Ancient Studies department."*. So the PROGRAM branch matched and returned a program profile, and the department branch was never reached.

The pre-existing comment above `_PROGRAM_SUBJECT_SIGNAL_PATTERN` documents the *intended* routing for single-word subjects:

> "a question naming both a subject and the word 'department' … is asking about the DEPARTMENT, not the program, and must be left for `search_department()` to handle instead"

That deferral only protected **single-word** program subjects (the academic-signal gate in `_subject_match_is_safe`). **Multi-word** subjects (`Ancient Studies`, `Communication Studies`, …) bypass the signal gate entirely, so the guard never fired and the program branch captured the query.

### 2b. `_department_name_matches()` trailing-`\b` bug — the blocker for parenthesized names (DEPT_006 exact row, DEPT_014)

The whole-phrase match used `\b{name}\b`. A name ending in a non-word character (**parentheses**: `Management Option (LSBE)`, `Criminology Minor (Faculty of Human and Social Sciences)`) can *never* match with a trailing `\b`: the character after `)` is a space or end-of-text, and neither forms a word boundary. Result: `search_department("Tell me about the Management Option (LSBE) department.")` returned `None`, and the exact `Criminology Minor (...)` row was unmatchable (so the broader `Criminology` row matched instead). 24 stored department names end in `)` (e.g. `Geography (GG/ES)`, `French (Lang/Lit)`) and were all silently unresolvable by name. Same class of bug as the escaped-period note in `_strip_person_titles`.

### 2c. First-match row ordering — the wrong-row cause (DEPT_006)

`search_department()` returned the **first** row whose name matched, in DB order. Distinct department rows share whole-word prefixes — `Criminology` (row 10) sorts before `Criminology Minor (Faculty of Human and Social Sciences)` (row 41) — so once 2b was fixed, first-match would still resolve DEPT_006 to the broader `Criminology` page.

## 3. Why It Occurred

1. **The single-word guard was never extended to multi-word names.** The academic-signal gate correctly defers single-word program subjects, but the same department-vs-program ambiguity exists for every multi-word subject that is *both* a program name and a department name — an unhandled gap.
2. **Trailing-`\b` assumes names end in word characters.** The whole-phrase regex was written for typical department names and never validated against parenthesized names; the escaping/`\b` interaction silently made a whole class of names unresolvable, which the benchmark's substring-keyword check could not detect.
3. **First-match assumed names are unique rows.** The departments table carries both distinct near-duplicate rows and duplicate rows for different academic-calendar versions, so "first match" is an arbitrary pick rather than the most specific referent.

## 4. Fix (smallest possible — 3 targeted changes, all in `src/retriever.py`)

No architecture changes, no new files beyond this report, no changes to any other module.

### Change 1 — department-intent guard in `search_program()`

Added `_DEPARTMENT_INTENT_PATTERN = re.compile(r"\bdepartments?\b", re.IGNORECASE)` and a guard at the top of `search_program()`:

```python
if _DEPARTMENT_INTENT_PATTERN.search(question_lower):
    if search_department(question, None) is not None:   # probe only — no memory side effects
        return None
```

When the question explicitly names a department **and** that department resolves, the program branch defers and the cascade reaches the existing DEPARTMENT branch. This is the *minimal* version of "route department-named questions to the department branch": it does not reorder the cascade, does not duplicate the ~40-line department context builder, and only fires when both conditions hold, so questions like *"What programs are offered in the Computer Science department?"* still route correctly. `memory` is deliberately not passed to the probe call so the department entity is recorded exactly once — by the DEPARTMENT branch that actually answers.

### Change 2 — trailing-`\b` fix in `_department_name_matches()`

```python
if re.search(r"\w$", name):
    name_pattern = rf"\b{re.escape(name)}\b"
else:
    name_pattern = rf"\b{re.escape(name)}"
```

Only require a trailing word boundary when the name actually ends in a word character; otherwise match the literal tail. This makes parenthesized department names resolvable. Strictly additive: it only *adds* matches that previously never matched.

### Change 3 — most-specific-match in `search_department()`

Replaced first-match with longest-name-match (a row only enters consideration when its *full* name appears in the question, so the longest match is the most specific referent the user named):

```python
best_row = None
best_name_len = -1
for row in rows:
    if _department_name_matches(row[0], question_lower):
        if len(row[0]) > best_name_len:
            best_row, best_name_len = row, len(row[0])
```

`Criminology Minor (Faculty of Human and Social Sciences)` now beats `Criminology`; duplicate rows of equal name keep first-row behavior (identical to before).

## 5. Regression Results

Harness: `streamlit.testing.v1.AppTest` against the real, unmodified `src/app.py`, every benchmark item executed end-to-end. "Before" = pre-fix capture of the same 202 items; "After" = fresh run on the fixed code.

### 5a. Departments (16) — the sprint target

| Metric | Before | After |
|---|---|---|
| Correct routing (`department_profile`) | 9 / 16 | **16 / 16** |
| Expected keyword present | 16 / 16 | 16 / 16 |
| Citation rendered | 16 / 16 | 16 / 16 |
| Citation URL on-topic (academic-calendar.wlu.ca) | 16 / 16 | 16 / 16 |
| Runtime errors / timeouts | 0 | 0 |

Per-item routing change:

| Item | Before | After |
|---|---|---|
| DEPT_001 Ancient Studies | program | department_profile |
| DEPT_005 Communication Studies | program | department_profile |
| DEPT_006 Criminology Minor (Faculty of Human and Social Sciences) | program | department_profile (exact row) |
| DEPT_010 Film Studies | program | department_profile |
| DEPT_011 General BSc without Designation | program | department_profile |
| DEPT_013 International Public Policy | program | department_profile |
| DEPT_014 Management Option (LSBE) | program | department_profile (exact row) |
| DEPT_002, 003, 004, 007, 008, 009, 012, 015, 016 | department_profile | department_profile (**byte-identical**, no change) |

### 5b. Named verification categories (unchanged)

| Category | Before | After |
|---|---|---|
| Programs (20) | 20/20 | **20/20** |
| Faculty (20) | 20/20 | **20/20** |
| Policies (20) | 20/20 | **20/20** |

The only non-Departments benchmark item containing the word "department" is an out-of-scope question (`UNSUPPORTED_016` — "Department of Time Travel Studies"); no such department exists, so the guard does not fire and it still declines gracefully.

### 5c. Full 202-item benchmark sweep (regression guard)

Every benchmark item was re-run end-to-end against the fixed code and diffed per-item against the pre-fix capture. **Routing result: 0 regressions across all 202 items.** The only intended routing changes are the 7 department fixes above.

Per-category signals (kw = expected keyword present, cit = citation rendered, citurl = on-topic citation URL; `rt_ok` only applies to the 4 target categories):

| Category | Before (kw/cit/citurl) | After (kw/cit/citurl) | rt_ok before→after |
|---|---|---|---|
| Academic Deadlines (16) | 16/16 · 16/16 · 10/16 | 16/16 · 16/16 · 10/16 | — |
| Campus Services (18) | 18/18 · 17/18 · 14/18 | 18/18 · 17/18 · 14/18 | — |
| Conversation / Follow-up (18) | 13/13 · 7/18 · 7/18 | 13/13 · 7/18 · 7/18 | — |
| Courses (22) | 22/22 · 22/22 · 22/22 | 22/22 · 22/22 · 22/22 | — |
| **Departments (16)** | 16/16 · 16/16 · 16/16 | 16/16 · 16/16 · 16/16 | **9/16 → 16/16** |
| FAQ (12) | 12/12 · 9/12 · 9/12 | 12/12 · 10/12 · 10/12 | — |
| **Faculty (20)** | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 | 20/20 → 20/20 |
| **Policies (20)** | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 | 20/20 → 20/20 |
| **Programs (20)** | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 | 20/20 → 20/20 |
| Student Services (18) | 18/18 · 18/18 · 9/18 | 18/18 · 18/18 · 9/18 | — |
| Unsupported / Out-of-domain (22) | 22/22 graceful declines | 22/22 graceful declines | — |

Notes on the handful of per-item differences outside Departments:

- **Vector answers** (Campus Services, Deadlines, Student Services, FAQ, Conversation) are LLM-paraphrased; their character length and occasionally their rendered citation vary run-to-run. Re-running any of them in isolation reproduces the same keyword/citation signals (e.g. Academic Deadlines is 16/16 on every metric when run alone — identical to before). This is pre-existing answer nondeterminism, not a Sprint 2 change.
- **FAQ_009** ("Metalworks partnership FAQs", a vector answer) lost its citation in one full-run pass. Run 4× in isolation on the fixed code it rendered 3/4 times — the citation drops when the LLM paraphrase trips the `answer_disclaims_relevance()` negation heuristic. Flaky before and after; unchanged by this sprint.
- **FAQ_002 / FAQ_006** actually improved (citation now rendered, 9→10 and 12/12), from the Sprint 1 deterministic-citation gate.
- **Policies** now carry full bodies (Sprint 1 improvement); the 20/20 keyword/citation pass is unchanged.
- **UNSUPPORTED_016** ("Department of Time Travel Studies") still declines gracefully — the guard requires a *resolvable* department, and no such row exists.

**Errors/timeouts: 0 in all 202 post-fix runs.**

## 6. Files Modified

- `src/retriever.py` — the three changes above (69 insertions / 7 deletions).
- `SPRINT2_DEPARTMENT_RETRIEVAL_REPORT.md` — this report.

## 7. Remaining Issues (out of scope for this sprint, pre-existing)

1. **Corpus boilerplate in thin department rows.** The `departments.db` rows for `Criminology Minor (Faculty of Human and Social Sciences)` and `Management Option (LSBE)` store mostly navigation chrome ("Department Information on this page … © 2026 Wilfrid Laurier University") with little substantive prose. This is a data-quality issue in the scraped corpus (the source pages genuinely contain little text), not a retrieval-routing defect; it predates this sprint and does not affect any other department.
2. **Duplicate calendar-version rows.** Several departments (Criminology, Communication Studies, Biology, Management Option (LSBE), …) have duplicate rows for different `academic-calendar` versions (graduate vs undergraduate). The longest-match rule returns the first row on ties, which for e.g. `Biology` is the graduate-calendar row (`Level: Graduate`, "Master of Science in Integrative Biology"). Pre-existing behavior; a future sprint could prefer the undergraduate row when both exist.
3. **Keyword metric blindness.** The benchmark's keyword check cannot distinguish "the program page that mentions the department subject" from "the department page". The Departments category should assert `response_type = department_profile` (the routing criterion this sprint fixed) in addition to keywords.

## 8. Verification Notes

- No code modified by the harness — every result came from the real app under `AppTest`.
- The 9 already-passing department questions are byte-identical before/after (same content length, same response type, same citation URL).
- `search_department`/`search_program` verified at function level for all 16 department questions before the end-to-end run.
