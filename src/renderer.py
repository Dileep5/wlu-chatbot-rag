import html
import re
import uuid

import streamlit as st

# Phase 13B: coarse response_type -> renderer grouping. This is a safe
# refactor only - these groups decide which of the four renderers below
# handles a given response_type, but every renderer still shows exactly
# the same answer/source text as before (Phase 13A). Anything not listed
# here (research, vector, and the None used by greeting/conversation/
# off-topic/clarification paths) falls through to render_generic(),
# which is byte-identical to today's plain rendering.
_COURSE_RESPONSE_TYPES = {
    "course",
    "prerequisite",
    "course_instructors",
}

_FACULTY_RESPONSE_TYPES = {
    "faculty_profile",
    "faculty_list",
    "faculty_topic_courses",
    "department_faculty_list",
}

_PROGRAM_RESPONSE_TYPES = {
    "program",
    "undergraduate_requirements",
    "graduate_requirements",
    "undergraduate_program_list",
    "coordinator",
}

# Phase 18: "department_profile" previously had no dedicated card at all
# and fell through to render_generic() - a real presentation gap, not a
# deliberate omission (structured_search()'s DEPARTMENT branch, unlike
# every other structured branch, never had a renderer of its own).
_DEPARTMENT_RESPONSE_TYPES = {
    "department_profile",
}


# -----------------------------
# Phase 18: shared HTML card styling and building blocks.
#
# Everything in this section is presentation-only: it consumes the
# fields each _parse_*_fields() function already extracts (all
# unchanged below) and lays them out as an HTML card instead of a flat
# "**Label**\nvalue" markdown list. No parsing, extraction, or
# label/field selection logic is touched by anything in this section.
#
# Colors reference the CSS custom properties app.py's Phase 17 redesign
# already defines on :root (--wlu-purple, --wlu-gold, etc.) so cards
# stay visually consistent with the rest of the app without duplicating
# the palette here - the var(..., fallback) form keeps this file
# self-contained (falls back to the same values) if it's ever rendered
# before that CSS block runs.
# -----------------------------

_CARD_CSS = """
<style>
.wlu-card {
    background: var(--wlu-card-bg, var(--wlu-paper, #FAF8FC));
    border: 1px solid var(--wlu-border, #E4DCEF);
    border-radius: var(--wlu-radius, 16px);
    overflow: hidden;
    margin: var(--wlu-sp-1, 0.5rem) 0;
    box-shadow: var(--wlu-shadow-rest, 0 4px 14px rgba(26, 21, 38, 0.12));
    transition: box-shadow var(--wlu-transition, 180ms ease);
}
.wlu-card:hover {
    box-shadow: var(--wlu-shadow-hover, 0 10px 28px rgba(26, 21, 38, 0.18));
}
.wlu-card-header {
    position: relative;
    background: linear-gradient(135deg, var(--wlu-purple, #522687) 0%, var(--wlu-purple-dark, #3E1C66) 100%);
    padding: var(--wlu-sp-2, 1rem) var(--wlu-sp-3, 1.5rem);
}
.wlu-card-header::after {
    content: '';
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--wlu-gold, #FCC707), rgba(252, 199, 7, 0) 85%);
}
.wlu-card-header .wlu-eyebrow {
    font-size: var(--wlu-fs-micro, 0.75rem);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--wlu-gold, #FCC707);
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.wlu-card-header .wlu-card-title {
    font-family: var(--wlu-font-head, 'Poppins', 'Inter', sans-serif);
    font-size: var(--wlu-fs-h2, 1.375rem);
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.3;
    margin: 0;
}
.wlu-card-header .wlu-card-subtitle {
    font-size: var(--wlu-fs-body-sm, 0.9375rem);
    color: rgba(255, 255, 255, 0.85);
    margin-top: 0.2rem;
}
.wlu-card-body {
    padding: var(--wlu-sp-3, 1.5rem) var(--wlu-sp-3, 1.5rem) 0.2rem;
}
.wlu-meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: var(--wlu-sp-1, 0.5rem) var(--wlu-sp-3, 1.5rem);
}
.wlu-meta-item .wlu-meta-label {
    font-size: var(--wlu-fs-micro, 0.75rem);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--wlu-ink-muted, #6B5F7A);
    font-weight: 600;
    margin-bottom: 0.15rem;
}
.wlu-meta-item .wlu-meta-value {
    font-size: var(--wlu-fs-body-sm, 0.9375rem);
    color: var(--wlu-ink, #1A1526);
    font-weight: 500;
    line-height: 1.45;
}
.wlu-section {
    margin-top: var(--wlu-sp-2, 1rem);
    padding-top: var(--wlu-sp-2, 1rem);
    padding-bottom: var(--wlu-sp-2, 1rem);
    border-top: 1px solid var(--wlu-border, #E4DCEF);
}
.wlu-card-body > .wlu-section:first-child {
    margin-top: 0;
    padding-top: 0;
    border-top: none;
}
.wlu-section-title {
    font-size: var(--wlu-fs-micro, 0.75rem);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--wlu-purple-text, #522687);
    font-weight: 700;
    margin-bottom: var(--wlu-sp-1, 0.5rem);
}
.wlu-section p {
    margin: 0 0 var(--wlu-sp-1, 0.5rem);
    line-height: 1.6;
    color: var(--wlu-ink, #1A1526);
    font-size: var(--wlu-fs-body-sm, 0.9375rem);
}
.wlu-section p:last-child {
    margin-bottom: 0;
}
.wlu-section.wlu-highlight {
    background: var(--wlu-paper, #FAF8FC);
    border: 1px solid var(--wlu-border, #E4DCEF);
    border-left: 3px solid var(--wlu-gold, #FCC707);
    border-radius: var(--wlu-radius, 16px);
    padding: var(--wlu-sp-2, 1rem);
}
.wlu-schedule-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: var(--wlu-sp-1, 0.5rem);
}
.wlu-schedule-item {
    background: var(--wlu-paper, #FAF8FC);
    border-radius: var(--wlu-radius, 16px);
    padding: var(--wlu-sp-1, 0.5rem) var(--wlu-sp-2, 1rem);
}
.wlu-schedule-item .wlu-meta-label {
    color: var(--wlu-purple-text, #522687);
}
.wlu-schedule-item .wlu-meta-value {
    font-weight: 400;
}
.wlu-card-footer {
    background: var(--wlu-paper, #FAF8FC);
    border-top: 1px solid var(--wlu-border, #E4DCEF);
    padding: var(--wlu-sp-1, 0.5rem) var(--wlu-sp-3, 1.5rem);
    font-size: var(--wlu-fs-caption, 0.8125rem);
}
.wlu-card-footer .wlu-source-label,
.wlu-standalone-source .wlu-source-label {
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: var(--wlu-fs-micro, 0.75rem);
    font-weight: 700;
    color: var(--wlu-ink-muted, #6B5F7A);
    margin-right: 0.3rem;
}
.wlu-card-footer a,
.wlu-standalone-source a {
    color: var(--wlu-purple-text, #522687);
    font-weight: 600;
    text-decoration: none;
    word-break: break-all;
    transition: color var(--wlu-transition, 180ms ease);
}
.wlu-card-footer a:hover,
.wlu-standalone-source a:hover {
    color: var(--wlu-purple-text, #522687);
    text-decoration: underline;
}
.wlu-card-footer a:focus-visible,
.wlu-standalone-source a:focus-visible {
    outline: none;
    box-shadow: var(--wlu-focus-ring, 0 0 0 2px #FAF8FC, 0 0 0 4px #FCC707);
    border-radius: 4px;
}
.wlu-standalone-source {
    font-size: var(--wlu-fs-caption, 0.8125rem);
    padding: var(--wlu-sp-1, 0.5rem) 0;
}
.wlu-card-footer a,
.wlu-standalone-source a {
    margin-right: var(--wlu-sp-1, 0.5rem);
}
.wlu-retrieval-date {
    color: var(--wlu-ink-muted, #6B5F7A);
    font-size: var(--wlu-fs-micro, 0.75rem);
}
.wlu-icon {
    display: inline-flex;
    vertical-align: -2px;
    margin-right: 0.35em;
}
.wlu-icon svg {
    width: 1em;
    height: 1em;
}

/* Generic/vector answer card (render_generic()) - the same card shell
   (background/border/radius/shadow) as .wlu-card above, applied to a
   st.container(key=...) instead of a raw HTML wrapper - see
   render_generic()'s own comment for why. The selector matches on a
   substring rather than a fixed class: the container's key carries a
   fresh uuid suffix per render (Streamlit requires unique keys within
   one script run, and the chat-history replay loop can render several
   of these in a single run), so no single fixed key exists to select. */
[class*="st-key-wlu-generic-card-"] {
    background: var(--wlu-card-bg, #FFFFFF);
    border: 1px solid var(--wlu-border, #E4DCEF);
    border-radius: var(--wlu-radius, 16px);
    box-shadow: var(--wlu-shadow-rest, 0 4px 14px rgba(26, 21, 38, 0.12));
    padding: var(--wlu-sp-3, 1.5rem);
    margin: var(--wlu-sp-1, 0.5rem) 0;
    transition: box-shadow var(--wlu-transition, 180ms ease);
}
[class*="st-key-wlu-generic-card-"]:hover {
    box-shadow: var(--wlu-shadow-hover, 0 10px 28px rgba(26, 21, 38, 0.18));
}
.wlu-generic-eyebrow {
    display: flex;
    align-items: center;
    font-size: var(--wlu-fs-micro, 0.75rem);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--wlu-purple-text, #522687);
    font-weight: 700;
    margin-bottom: var(--wlu-sp-2, 1rem);
    padding-bottom: var(--wlu-sp-1, 0.5rem);
    border-bottom: 1px solid var(--wlu-border, #E4DCEF);
}
[class*="st-key-wlu-generic-card-"] .wlu-standalone-source {
    margin-top: var(--wlu-sp-1, 0.5rem);
    padding-top: var(--wlu-sp-1, 0.5rem);
    border-top: 1px solid var(--wlu-border, #E4DCEF);
}
</style>
"""


# Production polish: replaces the emoji this card system originally
# shipped with (📘/📄/👨‍🏫/📇/🔬/📝/👤/📥/📚/📅/📋/ℹ️/🏛️/🎓) - emoji render
# as full-color pictures with inconsistent weight/style across
# platforms, which reads as a mismatched icon language next to the
# card system's own deliberate, monochrome design. Each icon below is
# a small inline SVG using currentColor for its stroke/fill, so it
# always exactly matches whatever text color the surrounding label
# already has (--wlu-gold in a card eyebrow, --wlu-purple-text in a
# section title) - including automatically adapting to dark mode,
# with no separate dark-mode variant needed, unlike a raster image or
# a data-URI background-image baked to one fixed color. The person and
# cap shapes deliberately reuse the exact path data from app.py's chat
# avatar icons, so the same concept (a person; graduation/academic)
# looks identical everywhere it appears in the app.
def _icon(name):

    paths = {
        "document": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M6 3h9l5 5v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/>'
            '<path d="M9 12h6M9 16h6M9 8h3"/></svg>'
        ),
        "person": (
            '<svg viewBox="0 0 24 24" fill="currentColor">'
            '<circle cx="12" cy="8" r="4"/>'
            '<path d="M4,21 A8,8 0 0,1 20,21 Z"/></svg>'
        ),
        "cap": (
            '<svg viewBox="0 0 24 24" fill="currentColor">'
            '<path d="M12,4 L22,9 L12,14 L2,9 Z"/>'
            '<rect x="8" y="10" width="8" height="5" rx="1" opacity="0.85"/>'
            '<line x1="19" y1="9" x2="19" y2="16" stroke="currentColor" stroke-width="1.5"/>'
            '<circle cx="19" cy="16.5" r="1.3"/></svg>'
        ),
        "book": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>'
            '<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>'
        ),
        "mail": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<rect x="2" y="4" width="20" height="16" rx="2"/>'
            '<path d="M2 6l10 7 10-7"/></svg>'
        ),
        "flask": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M9 3h6M10 3v6l-5.5 9.5A1 1 0 0 0 5.4 20h13.2a1 1 0 0 0 '
            '.9-1.5L14 9V3"/></svg>'
        ),
        "building": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M3 21h18M4 21V9l8-5 8 5v12M9 21v-6h6v6"/></svg>'
        ),
        "calendar": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<rect x="3" y="4" width="18" height="18" rx="2"/>'
            '<path d="M16 2v4M8 2v4M3 10h18"/></svg>'
        ),
        "info": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<circle cx="12" cy="12" r="9"/>'
            '<path d="M12 11v6M12 7.5v.01"/></svg>'
        ),
        "inbox": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M22 12h-6l-2 3h-4l-2-3H2"/>'
            '<path d="M5.45 5.11 2 12v7a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-7l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg>'
        ),
        "message": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
        ),
    }

    return f'<span class="wlu-icon">{paths[name]}</span>'


def _esc(value):
    """HTML-escapes a value for safe interpolation into the card markup
    below - the value itself is exactly what the unchanged _parse_*_
    fields()/_extract_labeled_field() functions already produced from
    the raw context text; this only prevents literal "&"/"<"/">"/quote
    characters occasionally present in scraped text from being
    misread as markup."""

    return html.escape(str(value)) if value else ""


def _text_html(value):
    """Escapes a value, then reflows it into <p> blocks split on blank
    lines (real paragraph breaks) - presentation only, the text itself
    is unchanged. A single newline within a paragraph is deliberately
    left as-is rather than forced into a <br>: HTML already collapses a
    lone newline to whitespace the same way Markdown's own soft-break
    rule would have, which matters here because the scraped academic-
    calendar text is full of single mid-sentence line breaks (e.g.
    "MA102\\nprior to completing\\nMA103") that would otherwise render
    as a needlessly choppy, broken-up paragraph instead of flowing
    text."""

    if not value:
        return ""

    escaped = _esc(value)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", escaped) if p.strip()]

    if not paragraphs:
        return ""

    return "".join(f"<p>{para}</p>" for para in paragraphs)


def _meta_grid_html(items):
    """items: an iterable of (label, value) pairs. Entries with a falsy
    value are skipped entirely (the same "hide missing fields"
    convention _parse_*_fields() callers already relied on). Renders as
    a responsive multi-column grid via CSS auto-fit, not fixed markup -
    the actual column count adapts to available width."""

    cells = "".join(
        f'<div class="wlu-meta-item">'
        f'<div class="wlu-meta-label">{_esc(label)}</div>'
        f'<div class="wlu-meta-value">{_esc(value)}</div>'
        f"</div>"
        for label, value in items
        if value
    )

    return f'<div class="wlu-meta-grid">{cells}</div>' if cells else ""


def _section_html(title, body_html, icon="", highlight=False):
    """Wraps already-built body HTML (from _text_html()/_meta_grid_html()/
    a custom block like the program schedule grid) in a titled, visually
    separated section. Returns "" - and is safe to include unfiltered in
    a sections list - whenever body_html is empty, so a missing field
    simply produces no section rather than an empty heading.

    `icon`, if given, is an _icon() name - resolved to its trusted SVG
    here rather than accepted as raw text, so a caller can never pass
    something that needs escaping through this parameter."""

    if not body_html:
        return ""

    css_class = "wlu-section wlu-highlight" if highlight else "wlu-section"
    heading = f"{_icon(icon)}{_esc(title)}" if icon else _esc(title)

    return (
        f'<div class="{css_class}">'
        f'<div class="wlu-section-title">{heading}</div>'
        f"{body_html}"
        f"</div>"
    )


def _citation_links_html(citation):
    """Phase 3: one <a> per source (title as link text, falling back to
    the bare URL when no title could be resolved), followed by the
    citation's single retrieval date - shared by _card_footer_html() and
    _render_source() so both citation surfaces stay in sync."""

    links = "".join(
        f'<a href="{_esc(entry["url"])}" target="_blank" rel="noopener">'
        f'{_esc(entry["title"] or entry["url"])}</a>'
        for entry in citation["sources"]
    )

    return (
        f"{links}"
        f'<span class="wlu-retrieval-date">Retrieved: {_esc(citation["date"])}</span>'
    )


def _card_footer_html(citation):

    if not citation:
        return ""

    return (
        f'<div class="wlu-card-footer">'
        f'<span class="wlu-source-label">Source</span>'
        f"{_citation_links_html(citation)}"
        f"</div>"
    )


def _card_html(eyebrow, title, subtitle, body_sections, source, eyebrow_icon=None):
    """Assembles one complete card (header + body sections + footer) as
    a single HTML string, so every card is still exactly one
    st.markdown() call - unchanged from before Phase 18, and required
    for the evaluate.py AppTest harness, which reads only the first
    markdown element of the last chat message.

    eyebrow_icon: an _icon() name (trusted, static SVG), kept as a
    separate parameter rather than baked into `eyebrow` itself, since
    `eyebrow` is always HTML-escaped below (consistent with every
    other field in this file) and escaping a pre-built <svg> string
    would mangle it into visible markup instead of rendering the
    icon."""

    subtitle_html = (
        f'<div class="wlu-card-subtitle">{_esc(subtitle)}</div>' if subtitle else ""
    )
    eyebrow_icon_html = _icon(eyebrow_icon) if eyebrow_icon else ""

    return (
        f"{_CARD_CSS}"
        f'<div class="wlu-card">'
        f'<div class="wlu-card-header">'
        f'<div class="wlu-eyebrow">{eyebrow_icon_html}{_esc(eyebrow)}</div>'
        f'<div class="wlu-card-title">{_esc(title)}</div>'
        f"{subtitle_html}"
        f"</div>"
        f'<div class="wlu-card-body">{"".join(body_sections)}</div>'
        f"{_card_footer_html(source)}"
        f"</div>"
    )


def _render_source(citation):

    if citation:
        st.markdown(
            f"{_CARD_CSS}"
            f'<div class="wlu-standalone-source">'
            f'<span class="wlu-source-label">Source</span>'
            f"{_citation_links_html(citation)}"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_summary(summary):
    """The natural-language lead-in app.py's generate_grounded_summary()
    produces for course/faculty_profile/program/department_profile
    answers, shown as a plain paragraph ABOVE the card - deliberately a
    separate st.markdown() call on its own, never concatenated into the
    `answer` string the four render_X() functions below parse for their
    actual labeled fields, so it can never be mistaken for card content
    by _extract_labeled_field(). A no-op (renders nothing) whenever
    summary is falsy - every other response_type, and any card whose
    summary generation failed or was skipped, looks exactly as it did
    before this feature existed."""

    if summary:
        st.markdown(summary)


# Phase 13C: known labels the underlying context text can contain
# (search_course()'s own f-string in retriever.py, plus the Sprint 10D
# metadata section) - used as lookahead boundaries so a field's value
# is captured up to whichever label comes next, not a fixed line count.
# This list is only used to recognize where one field's value ends, not
# to change what retriever.py produces.
_COURSE_FIELD_LABELS = [
    "Course Code",
    "Course Name",
    "Credits",
    "Department",
    "Level",
    "Description",
    "Prerequisites",
    "Corequisites",
    "Exclusions",
    "Location",
    "Notes",
]

_COURSE_FIELD_ALTERNATION = "|".join(
    re.escape(label) for label in _COURSE_FIELD_LABELS
)

# Fields shown on the card, in display order (Requirement 3) - not the
# same order they appear in the source text. Corequisites/Exclusions/
# Location/Notes (Phase 13D) come after the original five fields - since
# "course" now bypasses the LLM entirely, these Sprint 10D metadata
# fields would otherwise never reach the user at all (they used to be
# folded into the LLM's paraphrase; the card is now the only rendering
# path, so it has to cover everything the raw text can carry, not just
# the original five).
_COURSE_CARD_FIELDS = [
    "Course Code",
    "Course Name",
    "Credits",
    "Prerequisites",
    "Description",
    "Corequisites",
    "Exclusions",
    "Location",
    "Notes",
]


def _extract_labeled_field(label, text, label_alternation):
    r"""Captures a label's value up to whichever known label comes next
    (or end of string) - shared by course and faculty parsing, since both
    answer shapes are the same pattern: a fixed set of "Label: value"
    lines, some of which (Description/Biography) continue over multiple
    lines rather than ending at the first newline.

    The whitespace immediately after "Label:" is matched with [ \t]*,
    not \s* - deliberately same-line-only. \s* would also consume the
    newline that follows an EMPTY value (e.g. "Phone: \nOffice: ..."),
    and the lookahead below requires that exact newline to recognize
    the next label as a boundary; with it already consumed, the capture
    would instead swallow the entire next "Label: value" pair looking
    for a later boundary (confirmed live: an empty "Phone:" absorbed
    the following "Office: ..." whole). [ \t]* stops at the newline
    every time, so an empty value is always captured as empty and never
    absorbs whatever comes after it."""

    pattern = (
        rf"{re.escape(label)}:[ \t]*(.*?)"
        rf"(?=\n\s*(?:{label_alternation}):|\Z)"
    )

    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    value = match.group(1).strip()

    return value or None


def _parse_course_fields(answer):
    """Best-effort parse of the existing label:value course text. Returns
    None (parsing "failed") unless both Course Code and Course Name are
    found - the two fields that actually identify a course - since the
    answer text for a "course" response_type is normally LLM-paraphrased
    prose (not the raw labeled context: "course" isn't one of app.py's
    deterministic response_types), so these labels frequently won't be
    present at all. That's expected, not an error - render_course() falls
    back to the original rendering whenever this returns None."""

    fields = {
        label: _extract_labeled_field(label, answer, _COURSE_FIELD_ALTERNATION)
        for label in _COURSE_CARD_FIELDS
    }

    if not fields["Course Code"] or not fields["Course Name"]:
        return None

    return fields


def _render_course_fallback(answer, source, summary=None):

    _render_summary(summary)
    st.markdown(f"Course\n\n{answer}")
    _render_source(source)


def render_course(answer, source, summary=None):

    fields = _parse_course_fields(answer)

    if not fields:
        _render_course_fallback(answer, source, summary)
        return

    _render_summary(summary)

    meta_items = [
        ("Credits", fields.get("Credits")),
        ("Prerequisites", fields.get("Prerequisites")),
        ("Corequisites", fields.get("Corequisites")),
        ("Exclusions", fields.get("Exclusions")),
        ("Location", fields.get("Location")),
        ("Notes", fields.get("Notes")),
    ]

    body_sections = [
        _meta_grid_html(meta_items),
        _section_html("Description", _text_html(fields.get("Description")), icon="document"),
    ]

    st.markdown(
        _card_html(
            eyebrow="Course",
            eyebrow_icon="book",
            title=f"{fields['Course Code']} · {fields['Course Name']}",
            subtitle=None,
            body_sections=[s for s in body_sections if s],
            source=source,
        ),
        unsafe_allow_html=True,
    )


# Phase 13E: known labels the raw faculty_profile context can contain
# (search_faculty()'s own f-string in retriever.py) - "Contact" is a bare
# section heading with no value of its own, but still has to be listed
# here so it's recognized as a boundary when capturing the field right
# before it (Department).
_FACULTY_FIELD_LABELS = [
    "Name",
    "Title",
    "Faculty",
    "Department",
    "Contact",
    "Email",
    "Phone",
    "Office",
    "Biography",
    "Research Interests",
    "Website",
    "Office Hours",
]

_FACULTY_FIELD_ALTERNATION = "|".join(
    re.escape(label) for label in _FACULTY_FIELD_LABELS
)

# Fields shown on the card, in display order. Name/Title/Department/
# Email/Office/Phone/Research Interests are Requirement 3's named
# fields; Faculty and Biography are additional - both appear in the
# existing raw context and dropping them would be real information
# loss (the same mistake Phase 13D found and fixed for the Course
# Card). Website/Office Hours have no corresponding column in
# faculty.db at all (confirmed directly against the schema) - they're
# still listed so parsing would pick them up if that data ever exists,
# but today they'll always be absent and simply hidden, per
# Requirement 3's "hide missing fields."
_FACULTY_CARD_FIELDS = [
    "Name",
    "Title",
    "Faculty",
    "Department",
    "Email",
    "Office",
    "Phone",
    "Research Interests",
    "Biography",
    "Website",
    "Office Hours",
]


def _parse_faculty_fields(answer):
    """Mirrors _parse_course_fields() - best-effort parse of the existing
    label:value faculty text. Returns None (parsing "failed") unless Name
    is found - the one field that actually identifies a person - since
    "faculty_profile" answer text can still be LLM-paraphrased prose in
    older stored session history from before this phase, with no labels
    to find at all. render_faculty() falls back to the original
    rendering whenever this returns None."""

    fields = {
        label: _extract_labeled_field(label, answer, _FACULTY_FIELD_ALTERNATION)
        for label in _FACULTY_CARD_FIELDS
    }

    if not fields["Name"]:
        return None

    return fields


def _render_faculty_fallback(answer, source, summary=None):

    _render_summary(summary)
    st.markdown(f"Faculty\n\n{answer}")
    _render_source(source)


def render_faculty(answer, source, summary=None):

    fields = _parse_faculty_fields(answer)

    if not fields:
        _render_faculty_fallback(answer, source, summary)
        return

    _render_summary(summary)

    profile_items = [
        ("Faculty", fields.get("Faculty")),
        ("Department", fields.get("Department")),
    ]

    contact_items = [
        ("Email", fields.get("Email")),
        ("Phone", fields.get("Phone")),
        ("Office", fields.get("Office")),
        ("Website", fields.get("Website")),
        ("Office Hours", fields.get("Office Hours")),
    ]

    body_sections = [
        _meta_grid_html(profile_items),
        _section_html("Contact", _meta_grid_html(contact_items), icon="mail"),
        _section_html(
            "Research Interests",
            _text_html(fields.get("Research Interests")),
            icon="flask",
            highlight=True,
        ),
        _section_html("Biography", _text_html(fields.get("Biography")), icon="document"),
    ]

    st.markdown(
        _card_html(
            eyebrow="Faculty Profile",
            eyebrow_icon="person",
            title=fields["Name"],
            subtitle=fields.get("Title"),
            body_sections=[s for s in body_sections if s],
            source=source,
        ),
        unsafe_allow_html=True,
    )


# Phase 13F: known labels the raw "program" context can contain
# (search_program()'s own f-string in retriever.py) - same lookahead-
# boundary pattern as course/faculty parsing.
#
# "Department", "Program Coordinator", and "Department Coordinator" are
# included here too (production-polish fix): the shared "coordinator"
# response_type is produced by either search_program() or
# search_department() in retriever.py, and render_response()'s dispatch
# always routes that response_type to render_program() - never
# render_department() - regardless of which one produced it (see the
# comment above _DEPARTMENT_FIELD_LABELS below). Before this fix,
# "Program Coordinator"/"Department Coordinator" weren't recognized
# label boundaries at all, so a program-coordinator answer's "Program:"
# field had no lookahead boundary to stop at and swallowed the entire
# coordinator text into the card title, while a department-coordinator
# answer (text starting "Department:", never "Program:") failed to
# parse at all and fell back to plain text under a mislabeled "🎓
# Program" header. Both are fixed by _parse_program_fields() below
# recognizing either shape.
_PROGRAM_FIELD_LABELS = [
    "Program",
    "Department",
    "Program Coordinator",
    "Department Coordinator",
    "Level",
    "Program Type",
    "Description",
    "Admission Requirements",
    "Program Requirements",
]

_PROGRAM_FIELD_ALTERNATION = "|".join(
    re.escape(label) for label in _PROGRAM_FIELD_LABELS
)

# (display label, raw label) pairs, in display order. Program Name/
# Level/Admission Requirements/Description are Requirement 3's named
# fields the raw text actually has a label for ("Program" is renamed to
# "Program Name" to match Requirement 3's wording). Degree/Department/
# Required Courses/Electives/Credits Required/Coordinator/Duration have
# no corresponding label anywhere in this response_type's text (this is
# a real data gap, not a parsing miss - confirmed directly against
# retriever.py's PROGRAM branch, which never produces any of them), so
# they're simply never found and hidden, per Requirement 3's "hide
# missing fields." Program Type/Program Requirements are additional
# labels the raw text does contain - shown per Requirement 4 rather
# than dropped, the same lesson already applied to the Course and
# Faculty cards.
_PROGRAM_CARD_FIELD_MAP = [
    ("Program Name", "Program"),
    ("Level", "Level"),
    ("Program Type", "Program Type"),
    ("Admission Requirements", "Admission Requirements"),
    ("Program Requirements", "Program Requirements"),
    ("Description", "Description"),
]


def _parse_program_fields(answer):
    """Mirrors _parse_course_fields()/_parse_faculty_fields() - returns
    None (parsing "failed") unless Program Name is found, since that's
    the one field that actually identifies a program. Text for
    "undergraduate_requirements"/"graduate_requirements"/
    "undergraduate_program_list" response_types never has this label at
    all (they're sentences/bulleted lists, not a label:value program
    sheet), so this correctly returns None for those and render_program()
    falls back to the original rendering - unchanged from before this
    phase for all three.

    Production-polish fix: also accepts a "Department: X" identifying
    label (falling back to it only when "Program:" isn't present) and a
    "Coordinator" value from either "Program Coordinator:" or
    "Department Coordinator:" - the shared "coordinator" response_type
    is produced by either search_program() or search_department() in
    retriever.py, but render_response() always routes it through
    render_program()/this parser regardless of which one produced it.
    fields["Entity Kind"] records which shape was actually found, so
    render_program() can label the card correctly either way."""

    fields = {
        display: _extract_labeled_field(raw, answer, _PROGRAM_FIELD_ALTERNATION)
        for display, raw in _PROGRAM_CARD_FIELD_MAP
    }

    fields["Entity Kind"] = "Program"

    if not fields["Program Name"]:

        department_name = _extract_labeled_field(
            "Department", answer, _PROGRAM_FIELD_ALTERNATION
        )

        if department_name:
            fields["Program Name"] = department_name
            fields["Entity Kind"] = "Department"

    fields["Coordinator"] = (
        _extract_labeled_field("Program Coordinator", answer, _PROGRAM_FIELD_ALTERNATION)
        or _extract_labeled_field("Department Coordinator", answer, _PROGRAM_FIELD_ALTERNATION)
    )

    if not fields["Program Name"]:
        return None

    return fields


def _render_program_fallback(answer, source, summary=None):

    _render_summary(summary)
    st.markdown(f"Program\n\n{answer}")
    _render_source(source)


# Phase 13F: known labels the raw "program" context can contain
# Phase 14B: real scraped undergraduate program descriptions (confirmed
# directly against live data, e.g. "Honours BSc Computer Science") embed
# a year-by-year course breakdown and a regulations block as literal,
# standalone-line headings within the Description text itself - "Year
# 1"/"Year 2"/"Year 3"/"Year 4" and "Program Regulations". These are
# used as section boundaries below; nothing here changes what
# retriever.py produces, only how this already-parsed text gets
# reorganized for display.
_YEAR_HEADING_PATTERN = re.compile(r"(?m)^[ \t]*Year[ \t]+([1-4])[ \t]*$")

_PROGRAM_REGULATIONS_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*Program Regulations[ \t]*$"
)

# Graduate program descriptions (confirmed directly, e.g. "Master of
# Applied Politics") use a different standalone-line heading instead of
# "Program Regulations" - "Additional Information", immediately followed
# by a short page-navigation-style list of the sub-headings that repeat
# right after it (which duplicate the separately-labeled Admission
# Requirements/Program Requirements text already captured on their own).
# Recognizing it as a boundary too keeps that duplicate tail out of
# Overview, the same way "Program Regulations" already does for
# undergraduate descriptions.
_ADDITIONAL_INFO_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*Additional Information[ \t]*$"
)

# A confirmed, recurring scraped-page footer artifact (seen verbatim in
# multiple real program descriptions) - not a real "regulations" fact,
# but not discarded either: routed into Additional Information's "Other
# notes" rather than left mixed into Program Regulations.
_TRAILING_BOILERPLATE_PATTERN = re.compile(
    r"Academic\s*&\s*Related\s*Dates", re.IGNORECASE
)

# Sentence-level keywords used only to decide which sentence, within the
# text following "Program Regulations", belongs under Additional
# Information (campus/delivery mode/co-op) rather than staying under
# Program Regulations proper (GPA/graduation/progression). This is a
# best-effort classification, not exact NLP - any sentence that matches
# none of these simply stays under Program Regulations, so nothing is
# ever dropped regardless of how well a given sentence classifies.
_ADDITIONAL_INFO_KEYWORDS = [
    "campus", "campuses",
    "delivery", "in-person", "hyflex", "hybrid", "online", "remote",
    "co-op", "cooperative education", "internship", "placement",
]


def _split_regulations_and_additional(text):
    """Splits the text following the "Program Regulations" heading into
    (regulations, additional) - sentences matching a campus/delivery/
    co-op keyword go to "additional", the trailing scraped-page footer
    (if present) always goes to "additional" as a catch-all "other
    notes", and everything else stays in "regulations". Never drops
    anything: every sentence ends up in exactly one of the two."""

    if not text:
        return None, None

    boilerplate_match = _TRAILING_BOILERPLATE_PATTERN.search(text)

    if boilerplate_match:
        main_text = text[:boilerplate_match.start()].strip()
        footer_text = text[boilerplate_match.start():].strip()
    else:
        main_text = text
        footer_text = None

    chunks = re.split(r"(?<=[.!?])\s+", main_text) if main_text else []

    regulation_chunks = []
    additional_chunks = []

    for chunk in chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        lowered = chunk.lower()

        if any(keyword in lowered for keyword in _ADDITIONAL_INFO_KEYWORDS):
            additional_chunks.append(chunk)
        else:
            regulation_chunks.append(chunk)

    if footer_text:
        additional_chunks.append(footer_text)

    regulations = " ".join(regulation_chunks).strip() or None
    additional = "\n\n".join(additional_chunks).strip() or None

    return regulations, additional


def _split_program_description(description):
    """Best-effort split of the Description text into an overview,
    a year-by-year schedule, a regulations block, and an additional-
    information block - every one of which may be empty/None if this
    particular description doesn't contain that structure at all (e.g.
    a short graduate description with no year breakdown), in which case
    the whole thing simply stays as "overview". Every character of the
    input ends up in exactly one of the four returned pieces - nothing
    is ever discarded, only reorganized."""

    if not description:
        return {"overview": None, "schedule": [], "regulations": None, "additional": None}

    regulations_match = _PROGRAM_REGULATIONS_HEADING_PATTERN.search(description)
    additional_match = _ADDITIONAL_INFO_HEADING_PATTERN.search(description)

    # Whichever of the two non-year headings appears first - used both
    # to bound the schedule below and as an Overview boundary alongside
    # the year headings.
    non_year_positions = [
        m.start() for m in (regulations_match, additional_match) if m
    ]
    earliest_non_year = min(non_year_positions) if non_year_positions else None

    year_matches = list(_YEAR_HEADING_PATTERN.finditer(description))

    # Only years found BEFORE that boundary count as schedule boundaries -
    # real data never puts them after, but this keeps the boundary math
    # below unambiguous if it ever did.
    if earliest_non_year is not None:
        year_matches = [m for m in year_matches if m.start() < earliest_non_year]

    boundary_positions = [m.start() for m in year_matches]

    if earliest_non_year is not None:
        boundary_positions.append(earliest_non_year)

    first_boundary = min(boundary_positions) if boundary_positions else None

    overview = (
        description[:first_boundary].strip()
        if first_boundary is not None
        else description.strip()
    )

    schedule = []

    if year_matches:

        next_boundaries = [m.start() for m in year_matches[1:]]
        next_boundaries.append(
            earliest_non_year if earliest_non_year is not None else len(description)
        )

        for match, boundary_end in zip(year_matches, next_boundaries):

            year_text = description[match.end():boundary_end].strip()

            if year_text:
                schedule.append((f"Year {match.group(1)}", year_text))

    regulations_text = None
    additional_chunks = []

    if regulations_match:

        # Bounded by the Additional Information heading if one follows
        # it, otherwise runs to the end of the description - either way,
        # keyword-based sentence classification is always applied, so a
        # campus/delivery/co-op mention (or the trailing page footer)
        # still surfaces under Additional Information rather than
        # staying mixed into genuine GPA/graduation/progression text,
        # regardless of which heading combination this description has.
        regulations_end = (
            additional_match.start()
            if additional_match and additional_match.start() > regulations_match.start()
            else len(description)
        )
        remainder = description[regulations_match.end():regulations_end].strip()
        regulations_text, extra_additional = _split_regulations_and_additional(remainder)

        if extra_additional:
            additional_chunks.append(extra_additional)

    if additional_match:

        additional_remainder = description[additional_match.end():].strip()

        if additional_remainder:
            additional_chunks.append(additional_remainder)

    additional_text = "\n\n".join(additional_chunks).strip() or None

    return {
        "overview": overview or None,
        "schedule": schedule,
        "regulations": regulations_text,
        "additional": additional_text,
    }


def render_program(answer, source, summary=None):

    fields = _parse_program_fields(answer)

    if not fields:
        _render_program_fallback(answer, source, summary)
        return

    try:
        sections = _split_program_description(fields.get("Description"))
    except Exception:
        _render_program_fallback(answer, source, summary)
        return

    _render_summary(summary)

    meta_items = [
        ("Level", fields.get("Level")),
        ("Program Type", fields.get("Program Type")),
    ]

    schedule_html = ""

    if sections["schedule"]:

        schedule_cards = "".join(
            f'<div class="wlu-schedule-item">'
            f'<div class="wlu-meta-label">{_esc(year_label)}</div>'
            f'<div class="wlu-meta-value">{_text_html(year_text)}</div>'
            f"</div>"
            for year_label, year_text in sections["schedule"]
        )
        schedule_html = f'<div class="wlu-schedule-grid">{schedule_cards}</div>'

    body_sections = [
        _meta_grid_html(meta_items),
        # Coordinator placed first (production-polish fix): for a
        # fact-lookup "coordinator" answer this is the entire point of
        # the response - every other section below is empty/absent for
        # that shape and gets filtered out by the `if s` below, so this
        # is what actually shows. For a full program/department answer
        # that also happens to include coordinator info, leading with it
        # is still a reasonable, consistent position.
        _section_html(
            "Coordinator", _text_html(fields.get("Coordinator")), icon="person",
        ),
        _section_html(
            "Admission Requirements",
            _text_html(fields.get("Admission Requirements")),
            icon="inbox",
        ),
        _section_html("Overview", _text_html(sections["overview"]), icon="document"),
        _section_html(
            "Required Courses",
            _text_html(fields.get("Program Requirements")),
            icon="book",
        ),
        _section_html("Recommended Schedule", schedule_html, icon="calendar"),
        _section_html(
            "Program Regulations", _text_html(sections["regulations"]), icon="document"
        ),
        _section_html(
            "Additional Information", _text_html(sections["additional"]), icon="info"
        ),
    ]

    # Production-polish fix: the card is correctly labeled "Department"
    # when this answer came from search_department() rather than
    # search_program() (see _parse_program_fields()'s "Entity Kind"),
    # instead of always showing "Program" regardless of source.
    is_department = fields.get("Entity Kind") == "Department"

    st.markdown(
        _card_html(
            eyebrow="Department" if is_department else "Program",
            eyebrow_icon="building" if is_department else "cap",
            title=fields["Program Name"],
            subtitle=None,
            body_sections=[s for s in body_sections if s],
            source=source,
        ),
        unsafe_allow_html=True,
    )


# Phase 18: known labels the raw "department_profile" context can
# contain (structured_search()'s DEPARTMENT branch, retriever.py) - same
# lookahead-boundary pattern as course/faculty/program parsing. The
# sibling "coordinator" response_type (used for both program- and
# department-coordinator answers) is intentionally left routed through
# render_program() exactly as before - unchanged, since retargeting it
# would mean deciding, from parsed text alone, which of the two a given
# "coordinator" answer actually is, which is routing logic this phase
# doesn't touch.
_DEPARTMENT_FIELD_LABELS = [
    "Department",
    "Faculty",
    "Level",
    "Programs",
    "Description",
]

_DEPARTMENT_FIELD_ALTERNATION = "|".join(
    re.escape(label) for label in _DEPARTMENT_FIELD_LABELS
)

_DEPARTMENT_CARD_FIELDS = [
    "Department",
    "Faculty",
    "Level",
    "Programs",
    "Description",
]


def _parse_department_fields(answer):
    """Mirrors _parse_course_fields()/_parse_faculty_fields()/
    _parse_program_fields() - best-effort parse of the existing
    label:value department text. Returns None (parsing "failed") unless
    Department is found - the one field that actually identifies a
    department - since department_profile answer text can still be
    LLM-paraphrased prose in older stored session history, with no
    labels to find at all. render_department() falls back to the
    original (pre-Phase-18, render_generic-equivalent) rendering
    whenever this returns None."""

    fields = {
        label: _extract_labeled_field(label, answer, _DEPARTMENT_FIELD_ALTERNATION)
        for label in _DEPARTMENT_CARD_FIELDS
    }

    if not fields["Department"]:
        return None

    return fields


def _render_department_fallback(answer, source, summary=None):

    _render_summary(summary)
    st.markdown(f"Department\n\n{answer}")
    _render_source(source)


def render_department(answer, source, summary=None):

    fields = _parse_department_fields(answer)

    if not fields:
        _render_department_fallback(answer, source, summary)
        return

    _render_summary(summary)

    meta_items = [
        ("Faculty", fields.get("Faculty")),
        ("Level", fields.get("Level")),
    ]

    body_sections = [
        _meta_grid_html(meta_items),
        _section_html("Programs", _text_html(fields.get("Programs")), icon="book"),
        _section_html("Overview", _text_html(fields.get("Description")), icon="document"),
    ]

    st.markdown(
        _card_html(
            eyebrow="Department",
            eyebrow_icon="building",
            title=fields["Department"],
            subtitle=None,
            body_sections=[s for s in body_sections if s],
            source=source,
        ),
        unsafe_allow_html=True,
    )


def render_generic(answer, source, summary=None):
    """The vector/policy/research answer path - in practice the most
    common real response type, since it's what a free-text question
    that doesn't match a structured course/faculty/program/department
    lookup falls through to. Previously a completely bare st.markdown()
    call with none of the structured cards' visual treatment; now
    wrapped in the same card family via a CSS-keyed st.container(),
    with a minimal eyebrow so it reads as a recognized member of that
    family rather than an unstyled box.

    Deliberately NOT built as one raw _card_html() string the way the
    structured cards are: `answer` here is LLM-generated Markdown
    (**bold**, numbered/bulleted lists) that needs Streamlit's real
    Markdown-to-HTML pass to render correctly - confirmed live that
    inline emphasis inside a raw HTML block is NOT processed (stays as
    literal asterisks) even with unsafe_allow_html=True, unlike the
    structured cards' body text, which is already-scraped plain prose
    with no Markdown syntax of its own. st.container(key=...) instead
    wraps a normal st.markdown(answer) call - full native Markdown
    rendering - inside a div CSS can style as a card shell. The key
    must be unique per call within a single script run (Streamlit
    enforces this) since the chat-history replay loop can render
    several generic answers in one run; a fresh uuid4 per call
    guarantees that without needing to thread a stable id down from
    app.py."""

    _render_summary(summary)

    with st.container(key=f"wlu-generic-card-{uuid.uuid4().hex[:10]}"):

        st.markdown(
            f"{_CARD_CSS}"
            f'<div class="wlu-generic-eyebrow">{_icon("message")}Answer</div>',
            unsafe_allow_html=True,
        )

        st.markdown(answer)
        _render_source(source)


def render_response(
    response_type, answer, source, summary=None, show_card=True
):
    """show_card=False (app.py's answer-first, card-on-request redesign -
    only ever passed False for the four dense entity-profile types:
    course/faculty_profile/program/department_profile) shows just the
    grounded summary on its own, skipping the matching render_X() call
    entirely - app.py renders a "Show full details" BUTTON immediately
    after calling this function, which is what shows the full card next
    turn if clicked. Every other response_type always passes
    show_card=True (app.py's contract, not enforced here), so this
    never affects anything besides those four types.

    Follow-up suggestions (FOLLOWUP_SUGGESTIONS, app.py) are no longer
    rendered here at all - they're real st.button()s now, which need
    direct access to st.session_state/st.rerun() to handle a click, so
    app.py renders them itself immediately after calling this function
    rather than this module reaching into session state (this module
    stays presentation-only, same as before)."""

    if not show_card:
        _render_summary(summary)

    elif response_type in _COURSE_RESPONSE_TYPES:
        render_course(answer, source, summary)

    elif response_type in _FACULTY_RESPONSE_TYPES:
        render_faculty(answer, source, summary)

    elif response_type in _PROGRAM_RESPONSE_TYPES:
        render_program(answer, source, summary)

    elif response_type in _DEPARTMENT_RESPONSE_TYPES:
        render_department(answer, source, summary)

    else:
        render_generic(answer, source, summary)
