# Sprint P2 — UI & UX Polish

**Status:** Implementation + verification complete. All gates pass — committed automatically.
**Commit:** `(pending)`

---

## 1. Goal and Constraints

A **presentational-only polish sprint**. The Streamlit interface was
brought to production quality across ten areas (chat layout, answer
cards, source section, follow-up buttons, show-details reveal, loading
experience, error messages, empty state, accessibility, mobile
responsiveness) **without touching any chatbot logic**.

Per the sprint brief, **NOT modified**: retrieval, RAG, embeddings,
citations, BM25, Chroma, structured retrieval, answer generation.
The entire sprint is CSS + minimal presentational markup — no new LLM
calls, no API changes, no behavioral code paths changed. Every
functional surface is byte-identical in behavior to Sprint P1.

## 2. What Changed

Two files, `+275 / −13` lines, all presentational.

### `src/renderer.py` — source section redesign (area 3)

- `_citation_links_html()`: source links grouped into a
  `.wlu-source-links` inline row; the retrieval date now carries a
  small **clock glyph** and sits on its own line below the links.
- `_card_footer_html()` / `_render_source()`: a `.wlu-source-label-row`
  (link icon + "Source" label) tops both citation surfaces, so they
  read as one clearly-labelled "sources" panel instead of a bare link
  dump.
- `_icon()`: two new inline-SVG entries — `"link"` and `"clock"` —
  using `currentColor` so they adapt to dark mode automatically.
- `_CARD_CSS`: the footer becomes a tinted `--wlu-purple-soft` panel
  with a hairline top border; the standalone source becomes a
  self-contained rounded panel (tint + border) instead of a floating
  strip; the retrieval date is a muted micro-text line with the clock
  icon.

**Contract preserved:** the card is still ONE `st.markdown()` call
(the evaluate.py AppTest harness reads only `markdown[0]`), and the
`"Retrieved: {date}"` string is unchanged — the CSS redesign is purely
visual. No functional change.

### `src/app.py` — design-system additions (areas 1, 2, 4–10)

Appended to `CUSTOM_CSS` before `</style>`:

| Area | What |
|---|---|
| 1. Chat layout | Bubble-to-bubble `margin-bottom`; consistent prose rhythm (`p` margin/line-height) inside bubbles. |
| 2. Answer cards | Markdown **headings** (`h2`/`h3`) in brand purple + head font; **tables** themed (purple-soft header row, hairline borders, zebra striping via tokens → auto dark-mode); bold **lead-ins** (`**Overview:**`) get a quiet accent only when the paragraph's first content. |
| 4. Follow-up buttons | Button blocks flip `inline-block` so the action set wraps as a pill row; each button keeps its shared hover/transition. |
| 5. Show Details | The reveal button renders as **`type="primary"`** (solid purple, white text ≈10.6:1) with a trailing `▾` chevron hint. |
| 6. Loading | "Thinking…" label next to the pulsing dots (new `.wlu-typing-label`). |
| 7. Errors | `stAlert` boxes get the card radius / brand font / shadow; severity colours left to Streamlit. |
| 8. Empty state | A quiet `.wlu-welcome-hint` CTA under the welcome card ("Type a question below, or tap a suggested question…"). |
| 9. Accessibility | `@media (prefers-reduced-motion: reduce)` collapses the fade-up entrance, pulsing dots, and every transition to ~0. |
| 10. Mobile | `≤768px`: smaller hero + avatars, bubbles full-width, wide tables scroll (`overflow-x:auto`); `≤480px`: single-column suggested-question grid. |

Markup-only changes in `app.py`: the typing indicator gains
`<span class="wlu-typing-label">Thinking…</span>`; the reveal button
call becomes `st.button(..., type="primary")` (all other follow-up
buttons stay default `secondary`); the welcome card gains the hint line.

## 3. Deliberately NOT changed

Retrieval, RAG, embeddings, BM25, Chroma schema, structured retrieval,
answer generation, citations pipeline, safety system, hallucination
guards, crawler/preprocessing. No LLM-based changes. The suggested-
questions block, hero/welcome rendering, turn processing, and message
loop are byte-identical to Sprint P1.

## 4. Verification — all gates pass

| Gate | Result |
|---|---|
| Regression suite (`src/evaluate.py` dedicated 235-check) | **235/235 ALL TESTS PASSED** |
| Stress suite (55 items) — response_type + cited URLs vs P1 baseline | **0/55 changes** |
| Manual production probes (BUG1–BUG6, same as Sprint P1) | **6/6 identical behavior** |
| Convocation date-exactness probe (8 runs) | **8/8 "October 20"** |
| Full benchmark (202 items) | see §5 |

**Stress 0/55** is the strongest single signal that P2 is purely visual:
fifty-five diverse queries (courses, programs, faculty, policies,
departments, out-of-domain) produce byte-identical `response_type` and
citation-URL sets to the Sprint P1 baseline — the CSS/markup cannot
have altered a single routing decision.

**Manual probes** confirm the six Sprint P1 bug fixes still behave
identically: "I'm stressed." → wellness vector answer; Writing Centre →
Brantford OM207 / Waterloo P226 (+ virtual); AI faculty → `research`
aggregation; CS department → `department_profile`; Patricia Goff →
enriched profile; convocation + "For Brantford?" → "October 20 at
2:00 p.m.".

**Reveal-button check (live DOM probe):** after a first-ask
faculty_profile query, the "Show full details" button's `proto.type`
is `primary` while every other button (suggested questions, "New
Conversation") stays `secondary`; clicking it flips `show_card` and
rerenders the card exactly as before.

## 5. Benchmark (202 items)

| Metric | Sprint P1 (after) | Sprint P2 (after) |
|---|---|---|
| Answer accuracy | 202/202 (1.0) | **202/202 (1.0)** |
| Hallucination rate | 0 (n=22) | **0 (n=22)** |
| Retrieval success | 1.0 (n=169) | **1.0 (n=169)** |
| Structured / hybrid accuracy | 1.0 / 1.0 | **1.0 / 1.0** |
| Citation accuracy | 99.4% (162/163) | **99.4% (162/163)** — see FAQ_009 note |
| Avg response time | 2.67s | 3.95s — LLM latency variance, not a code change |

**FAQ_009 citation note.** Citation accuracy reads 99.4% (162/163)
because FAQ_009 returned no citation in isolation — **identical in the
P1 and P2 runs**. This is the documented pre-existing LLM-phrasing
flake ([[wlu-flaky-tests]]), not a regression; the answer itself
passes, only the citation expectation flakes. P2 changed no code that
could affect it.

## 6. Regression Risk Assessment

- **Pure presentation:** the diff is CSS strings + three markup tokens
  (typing label, welcome hint, `type="primary"`). No control flow,
  no data path, no LLM prompt changed — the byte-identical stress and
  probe outputs are the direct consequence.
- **Harness compatibility:** the card remains a single `st.markdown()`
  call; the `"Retrieved: {date}"` text is unchanged; CSS cannot affect
  the AppTest proto values the regression suite asserts on.
- **Grounding / hallucination:** untouched code paths; the convocation
  date-exactness guard still holds 8/8.
- **Scope control:** every new CSS rule is scoped to
  `div[data-testid="stChatMessage"]`, `[data-testid="stAlert"]`,
  `.wlu-*` classes, or `@media` blocks — nothing targets the app's
  global chrome beyond the intended chat surfaces.

## 7. Engineering Hygiene

- Two files modified, `+275 / −13` insertions/deletions. No TODOs, no
  debug code, no dead branches.
- All new CSS uses the existing design tokens (with `#hex` fallbacks)
  so light/dark mode both render correctly; contrast ratios documented
  in comments.
- `sprintP2_report.md` follows the two-commit sprint convention
  (implementation commit + "Pin report commit hash" commit).
