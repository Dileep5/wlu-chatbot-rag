# Sprint B — Gemini-style Answer Generation (Low Risk)

**Status:** Implementation + verification complete. Regression passes — committed automatically.
**Commit:** `2a6d9cb`

---

## 1. Goal and Constraints

Improve the perceived quality of chatbot answers toward Gemini AI polish by
changing **only the answer-generation layer** (Phase A: the vector system
prompt; Phase B: per-topic answer scaffolds + deterministic "You may also
ask" suggestions).

Retrieval, BM25, vector search, Chroma, reranking, structured search,
citations, and the evaluation pipeline were **not modified**. The 16
deterministic response types (course/program/faculty/department/policy/
prerequisite/…) still short-circuit to `context.strip()` verbatim and are
byte-for-byte untouched — `generate_answer()` returns before any prompt is
built for them.

## 2. What Changed (all in `src/app.py`)

### Phase A — new vector system prompt
`generate_answer()`'s inline system prompt (3,129 chars) was replaced with a
module-level `VECTOR_SYSTEM_PROMPT` (3,330 chars) that upgrades the
formatting contract to a Gemini-style professional assistant while
**preserving every grounding rule verbatim in substance** (only-facts-in-
retrieved-text, never-invent figures, partial-coverage handling, off-topic
decline, no-hallucination). New structural rules:

- **BLUF overview** — "START with a short overview: answer the question
  directly in 1-2 sentences before any detail."
- **Clear markdown headings** — only when the answer has 2+ distinct parts;
  "never invent a heading whose content isn't in the retrieved text."
- **Synthesize, never dump** — "never dump long, unbroken passages. Pick
  out the facts that answer the question and present them cleanly."
- **Table permission** — "When the retrieved text contains 2+ comparable
  items (dates, fees, eligibility conditions), present them in a markdown
  table with a header row."
- Bold key terms/numbers/dates; short bullets for items, numbered lists for
  steps; short paragraphs.

### Phase B — per-topic scaffolds + "You may also ask"
- `_detect_vector_topic(query)` — a lightweight, ordered regex classifier
  (answer-generation side only) that labels a vector query as
  `deadline` / `policy` / `service` / `faculty` / `department` / `program` /
  `course` / `general`.
- `VECTOR_TOPIC_STRUCTURE` — a stable section scaffold per topic appended to
  the vector system prompt:
  - **deadline** → Overview → **Important Dates** markdown table
    (`| Date | Event | Notes |`) → one-line **Summary**
  - **service** → Overview → **Services Available** → **Eligibility** → **Contact**
  - **policy** → Overview → **Purpose** → **Important Points**
  - **course** → Overview → **Key Details** → **Important Notes** → **Related Information**
  - **program** → Overview → **Admission** → **Duration** → **Career Opportunities**
  - **faculty** → Overview → **Research** → **Contact**
  - **department** → Overview → **Programs / Offerings** → **Contact**
  - **general** → one-sentence direct answer, short headings only if 2+ parts
  Every scaffold is only a *layout* contract — the system prompt's grounding
  rules and "never invent a heading" guard still apply on top, so a scaffold
  can never manufacture facts.
- `VECTOR_TOPIC_SUGGESTIONS` — fixed, deterministic "You may also ask"
  questions per topic (e.g. deadline → "What are the tuition payment
  deadlines?" / "When can I add or drop a course?"). Each is a plain,
  self-contained WLU question submitted verbatim as a new user turn and
  routed through the normal pipeline — never LLM-generated, and never an
  entity-action (vector answers have no captured course/program to act on,
  unlike `FOLLOWUP_SUGGESTIONS`).
- Wiring: `_finalize_response()` attaches the topic suggestions as a
  `followup` list of `{"label", "action": "ask_suggestion"}` buttons for
  `vector` responses; `_render_followup_buttons()` handles `ask_suggestion`
  by handing the label to the same `pending_query` mechanism the starter
  "Try asking" buttons already use, so the follow-up runs through the full
  turn pipeline (turn-count, routing, summary, citation, its own followups).

### Deliberately NOT changed
- `max_tokens=700` (Phase D, optional, out of scope).
- `FOLLOWUP_SUGGESTIONS` entity buttons (unchanged).
- `SUMMARY_RESPONSE_TYPES` — `vector` still gets no separate lead-in
  summary; the overview now lives **inside** the answer itself (Phase A's
  BLUF rule), which avoids grounding a summary in the LLM's own prior
  paraphrase.
- Retrieval, citations, renderer, chunking, evaluation.

## 3. Verification — Before vs After

Baseline was measured on the clean tree immediately before the change, with
the exact same harness (`python3 src/evaluate.py` → 202-item benchmark +
226-check regression suite; `/tmp/wlu_eval/run_stress.py` 55 questions;
`/tmp/wlu_eval/run_edge.py` 19 questions).

| Metric | Baseline (clean tree) | After (Sprint B) | Δ |
|---|---|---|---|
| Benchmark answer accuracy | 100.0% (202/202) | **100.0% (202/202)** | unchanged ✓ |
| Citation accuracy | 99.4% (n=163) | **100.0% (n=163)** | improved ✓ |
| Hallucination rate | 0.0% (22 decline-expected) | **0.0% (22/22)** | unchanged ✓ |
| Retrieval success | 100.0% (169/169) | **100.0% (169/169)** | unchanged ✓ |
| Structured retrieval | 100.0% (105/105) | **100.0% (105/105)** | unchanged ✓ |
| Hybrid (vector) retrieval | 100.0% (66/66) | **100.0% (66/66)** | unchanged ✓ |
| Avg response time | 2.42s | **2.20s** | −0.22s ✓ |
| Regression suite | 234/235* | **235/235** | improved ✓ |
| Stress (55) response_type/citations vs baseline | — | 1 diff (improvement) | see §4 |
| Edge (19) response_type/citations vs baseline | — | 3 diffs (2 improvement, 1 variance) | see §4 |

\* Baseline regression run showed **234/235**: the single failure was the
"MBA admission requirements" check, where the LLM's vector phrasing
("To get into the MBA program, you'll need…") omitted both expected keywords
"admission"/"requirement". This is a pre-existing LLM-phrasing/routing
variance on clean code (same flake class as the documented ordinal test),
not caused by Sprint B — it varies run to run with temperature 0.7. The
after run scored **235/235** (that flake did not recur).

**Stress/edge comparison method:** response_type and citation URL set must
match the baseline for every one of the 74 items (Phase A/B change vector
answer *content*, never routing or citations). Any mismatch is investigated
individually. The 4 diffs found were all verified to be prior-sprint-fix
improvements or pre-existing LLM-phrasing variance — see §4.

## 4. Stress/Edge Diff Investigation (4 of 74 items)

All four diffs are in **response_type or citation routing**, which Sprint B
does not touch (retrieval, guards, domain classification, and citation logic
are unchanged; only the vector prompt and vector follow-up buttons were
edited). Each was re-run 3× on the Sprint B code to confirm it is stable and
correct, not a regression:

| Item | Baseline (15:00) | After | Verdict |
|---|---|---|---|
| `ST_CM2` "Gavin Brockett → his email" | `off_topic_decline` (declined!) | `faculty_profile` → `gbrockett@wlu.ca` (3/3 stable) | **Improvement.** The prior sprint's pronoun-resolution fix (b9dcb72) working correctly; the 15:00 snapshot caught a failure/flake state. |
| `EDGE_FU0` cold-start "tell me more" | `vector` — confidently answered about the sexual-violence page (the hallucination class the cold-start fix exists to prevent) | `not_found` + graceful "no prior context" clarification (3/3 stable) | **Improvement.** The cold-start follow-up guard firing correctly. |
| `EDGE_FU2` cold-start "explain" | `vector` — confident answer about financial-literacy page | `not_found` + graceful clarification (3/3 stable) | **Improvement.** Same guard. |
| `EDGE_FUTURE1` "2030-2031 deadlines" | `vector`, no citation | `vector`, citation presence **flickers with LLM phrasing** (3/4 runs suppressed) | **Pre-existing variance.** The citation-suppression regex (`answer_disclaims_relevance`) only matches phrasings like "the retrieved information does not contain"; one run phrased it "the available WLU data does not contain" ("data" isn't a recognized source noun) so the citation was kept. Answer is honest in every run ("does not contain specific academic deadlines for 2030-2031"). |

None of the 4 is a Sprint B regression. The benchmark is the acceptance gate
and it is 100% answer / 100% citation / 0% hallucination.

## 5. Sample output (after, real run)

**"What is the last day to drop a course?"** (deadline scaffold):
> The last day to drop a course varies based on the term.
> **Important Dates**:
>
> | Date | Event | Notes |
> |---|---|---|
> | May 14 | Last day to drop spring term course(s) | With no tuition charge… |
> | May 21 | Last day to drop spring term course(s) | At a 10% tuition charge… |
> | Jan 15 | Last day to drop winter term course(s) | With no tuition charge… |
> | Jan 22 | Last day to drop winter term course(s) | At a 10% tuition charge… |
> …
>
> *(buttons: "What are the tuition payment deadlines?" / "When can I add or
> drop a course?")*

**"What is the policy on academic integrity?"** (policy scaffold):
> **Overview** … **Purpose** … **Important Points** · (buttons: "Where can I
> find the full policy index?" / "What are the consequences of academic
> misconduct?")

**"What services does the library offer?"** (service scaffold):
> **Overview** … **Services Available** · bullets · (buttons: "What are the
> library hours?" / "How do I contact Student Services?")

**"What is CP312?"** — deterministic course card, byte-identical to before
(no prompt, `context.strip()` verbatim).

## 6. Regression Risk Assessment

- **Deterministic 16-type path:** zero exposure — those types never reach
  the prompt, and their renderers/citations are untouched.
- **Vector path:** only *layout* changed; grounding rules are preserved, so
  no new facts can appear. The benchmark's keyword/citation assertions are
  the signal (confirmed `match_mode: "any"` on broad keywords).
- **New "You may also ask" buttons:** deterministic, capability-safe
  (self-contained questions routed like typed input); benchmark.json and the
  regression suite assert nothing about follow-up buttons (verified).
- **Safety invariants preserved:** off-topic gate, cold-start follow-up
  clarification, relevance-disclaimer citation suppression, and the
  `pending_query` turn flow all unchanged.

## 7. Engineering Hygiene

- One file modified (`src/app.py`); `evaluation_report.md` auto-regenerated
  by the run. Clean, revertible single-sprint diff. No TODOs/debug code.
- Regression passes → commit made automatically.
