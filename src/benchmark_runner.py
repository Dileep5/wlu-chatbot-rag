"""Phase 4: benchmark evaluation runner.

Drives evaluation/benchmark.json through the real, unmodified app (same
streamlit.testing.v1.AppTest harness evaluate.py's own 226-check suite
already uses - not a shortcut around retriever.py) and computes the
metrics required for the evaluation framework. Every expectation
(expected_keywords, expect_decline, expected_source_contains, ...) is
already stored in benchmark.json - nothing here calls an LLM to judge or
generate an expected answer; correctness is decided by plain
deterministic string matching, the same technique evaluate.py's existing
226 checks already use.

Deliberately does not import or alter anything in retriever.py -
retrieval, ranking, prompting, and the hallucination gate are exercised
exactly as a real user would exercise them, never modified or bypassed.
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
APP_PATH = str(SRC_DIR / "app.py")
BENCHMARK_FILE = BASE_DIR / "evaluation" / "benchmark.json"

# Every response_type retriever.py's structured_search() cascade can
# produce (mirrors app.py's own DETERMINISTIC_RESPONSE_TYPES plus the
# list-shaped/relational types that also bypass the LLM but were never
# added to that set) - anything else that isn't "vector" and isn't
# None/"not_found" would be a new response_type this list hasn't been
# taught about yet, and is still safely bucketed as "other" below rather
# than mis-counted as either structured or hybrid.
_STRUCTURED_RESPONSE_TYPES = {
    "course", "program", "coordinator", "faculty_profile", "department_profile",
    "policy", "prerequisite", "undergraduate_requirements", "graduate_requirements",
    "research", "course_instructors", "faculty_topic_courses", "faculty_list",
    "department_faculty_list", "course_clarify", "faculty_clarify",
}

# Exact, deterministic sentinel phrases - each one is fixed text
# retriever.py's own code emits directly (never passed through the LLM),
# so a plain substring check is fully precise for these; no need for
# pattern matching here.
_DECLINE_MARKERS = [
    "couldn't find reliable information",
    "i couldn't find a course named",
    "i couldn't find a program called",
    "i couldn't find a faculty member named",
    "i couldn't identify a specific faculty member",
    "no course named",
    "no faculty-taught record",
]

# LLM-phrased declines (generate_answer()'s grounding prompt, hybrid_search
# path: vector search found *something* that passed the confidence gate,
# but the LLM correctly noticed the retrieved content doesn't actually
# answer the question and said so in its own words) have no fixed
# wording - temperature=0.7 means the exact phrasing is never the same
# twice, so a growing list of literal phrases can never have full
# coverage (confirmed: real declines like "does not explicitly detail
# what 'policy 0.0' entails... does not define or outline 'policy 0.0'
# itself" were missed by the previous fixed-phrase list even after
# multiple rounds of adding more phrases to it).
#
# Instead of enumerating wordings, this matches the underlying SEMANTIC
# STRUCTURE every grounded decline shares: a negation next to a word
# about information being present/documented/stated ("does not
# contain/mention/specify/define/outline/detail/state/indicate...",
# "no information/details available", "doesn't cover"). Deliberately
# narrow: the word list is restricted to meta-statements about the
# SOURCE MATERIAL's content, not the fabricated entity's own properties
# - "does not require a thesis" or "does not offer co-op" would NOT
# match (those verbs aren't in the list), since a real hallucination
# inventing specific facts about a fabricated program is exactly what
# this must still catch as a failure, not excuse as a valid decline.
_INFO_AVAILABILITY_WORDS = (
    r"contain(?:s|ed|ing)?|include[sd]?|specify|specifie[sd]|"
    r"define[sd]?|outline[sd]?|mention(?:s|ed)?|detail(?:s|ed)?|"
    r"provide[sd]?|address(?:es|ed)?|state[sd]?|"
    r"indicate[sd]?|available|found|information|details|data|specifics?"
)

# Word gap deliberately tight (0-2 words, and "cover(s/ed)" deliberately
# excluded from the word list above) - confirmed empirically that a wider
# gap and "cover" both let the pattern reach across an unrelated clause
# and false-positive on genuinely confident answers, e.g. "CP312 does
# not require any prerequisites and covers algorithm design." (a real,
# correct, substantive answer, not a decline).
_NEGATED_INFO_AVAILABILITY_PATTERN = re.compile(
    r"\b(?:does\s+not|doesn't|do\s+not|don't|did\s+not|didn't|"
    r"cannot|can't|unable\s+to|no)\b"
    rf"(?:\s+\w+){{0,2}}\s+(?:{_INFO_AVAILABILITY_WORDS})\b",
    re.IGNORECASE
)

# The off-topic branch (app.py) used to assign a fixed OFF_TOPIC_MESSAGE
# constant directly, with no LLM involvement - production polish
# (warmer tone for social/emotional off-topic messages vs. genuine
# factual ones) replaced that with two LLM-generated paths,
# generate_offtopic_decline()/generate_offtopic_social_response()
# (app.py), so the literal constant is now only their no-API-key
# fallback.
#
# response_type is the real, primary signal now (see
# _looks_like_decline() below): app.py tags every response from those
# two functions with "off_topic_decline"/"off_topic_social", read
# straight from session state - deterministic, regardless of exact
# LLM wording. _OFFTOPIC_SCOPE_PATTERN below started as the PRIMARY
# mechanism and was repeatedly insufficient: temperature (0.7-0.8)
# means the scope statement both prompts require ("I'm all about WLU" /
# "my focus is on Wilfrid Laurier University" / "I'm here for WLU
# questions", ...) gets phrased differently every run, and confirmed-
# live gaps kept surfacing under real testing - "I'm here specifically
# for..." (extra adverb before "for"), "my focus is WLU" (missing
# "on"), "I focus specifically on..." (adverb before "on"), "I'm here
# to chat specifically about..." (different verb/preposition
# entirely). Each fix closed the specific gap found and immediately
# revealed another - the same "can never have full coverage" problem
# _NEGATED_INFO_AVAILABILITY_PATTERN's own comment above already
# documents for LLM-phrased text in general, which is what motivated
# switching to response_type as the primary signal instead. Kept only
# as a fallback for content this file's checks might exercise without
# response_type available. Kept identical to evaluate.py's own copy of
# this same pattern (not imported - this file already deliberately
# avoids importing retriever.py/app.py internals beyond the one lazy
# OFF_TOPIC_MESSAGE import below, to stay a self-contained check
# against the real app's actual output, not its internals).
_OFFTOPIC_SCOPE_PATTERN = re.compile(
    r"\bfocus(?:ed)?\s+(?:is\s+)?on\b.{0,60}\b(?:wlu|wilfrid\s+laurier)\b"
    r"|\bfocus\s+is\s+(?:wlu|wilfrid\s+laurier)\b"
    r"|\bhere\s+(?:specifically\s+|solely\s+|primarily\s+|mainly\s+|"
    r"really\s+|just\s+)?(?:for|to\s+help)\b"
    r".{0,60}\b(?:wlu|wilfrid\s+laurier)\b"
    r"|\ball\s+about\b.{0,60}\b(?:wlu|wilfrid\s+laurier)\b",
    re.IGNORECASE
)


def load_benchmark():
    with BENCHMARK_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_decline(content, response_type):
    """True if the response is a graceful decline rather than a
    confident, specific answer. response_type is checked first and is
    the reliable signal for the response_type-tagged paths
    ("not_found", "off_topic_decline", "off_topic_social" - all
    assigned directly in app.py, never inferred from wording); the
    LLM-phrased grounding-prompt path (no dedicated response_type) is
    checked via _NEGATED_INFO_AVAILABILITY_PATTERN, which generalizes
    across wordings instead of requiring every variant to be
    enumerated."""

    if response_type in ("not_found", "off_topic_decline", "off_topic_social"):
        return True

    from app import OFF_TOPIC_MESSAGE

    if OFF_TOPIC_MESSAGE in content:
        return True

    if _OFFTOPIC_SCOPE_PATTERN.search(content):
        return True

    lower = content.lower()

    if "i'm not sure" in lower:
        return True

    if any(marker in lower for marker in _DECLINE_MARKERS):
        return True

    return bool(_NEGATED_INFO_AVAILABILITY_PATTERN.search(content))


def _contains_any(text, keywords):
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


def _contains_all(text, keywords):
    lower = text.lower()
    return all(k.lower() in lower for k in keywords)


def run_benchmark_item(item):

    result = {
        "id": item["id"],
        "category": item["category"],
        "question": item["turns"][-1],
        "expect_decline": item["expect_decline"],
        "requires_retrieval": item["requires_retrieval"],
        "passed": False,
        "response_type": None,
        "content": "",
        "elapsed_seconds": 0.0,
        "citation_checked": False,
        "citation_correct": None,
        "retrieval_success": None,
        "failure_reason": None,
    }

    start = time.perf_counter()

    try:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=90)

        for turn in item["turns"]:
            at.chat_input[0].set_value(turn).run(timeout=90)

        msg = at.session_state.messages[-1]

        content = msg.get("content", "") or ""
        source = msg.get("source")
        response_type = msg.get("response_type")

    except Exception as e:
        result["elapsed_seconds"] = time.perf_counter() - start
        result["failure_reason"] = f"error: {e}"
        return result

    result["elapsed_seconds"] = time.perf_counter() - start
    result["content"] = content
    result["response_type"] = response_type

    if item["expect_decline"]:
        answer_correct = _looks_like_decline(content, response_type)
        if not answer_correct:
            result["failure_reason"] = (
                "expected a graceful decline, got a confident answer "
                "(possible hallucination)"
            )
    else:
        keywords = item["expected_keywords"]
        if not keywords:
            answer_correct = bool(content.strip())
        elif item["match_mode"] == "all":
            answer_correct = _contains_all(content, keywords)
        else:
            answer_correct = _contains_any(content, keywords)
        if not answer_correct:
            result["failure_reason"] = (
                f"expected keywords {keywords} ({item['match_mode']}) "
                f"not found in response"
            )

    result["passed"] = answer_correct

    if item["requires_retrieval"]:
        result["retrieval_success"] = response_type not in (None, "not_found")

    if item["expect_citation"]:
        result["citation_checked"] = True

        if not source:
            result["citation_correct"] = False
            if answer_correct and not result["failure_reason"]:
                result["failure_reason"] = (
                    "answer correct but no citation was rendered"
                )
        else:
            expected_fragment = item.get("expected_source_contains")
            if expected_fragment:
                result["citation_correct"] = any(
                    expected_fragment.lower() in s["url"].lower()
                    for s in source["sources"]
                )
                if not result["citation_correct"] and not result["failure_reason"]:
                    result["failure_reason"] = (
                        f"citation present but source didn't contain "
                        f"'{expected_fragment}'"
                    )
            else:
                result["citation_correct"] = True

    return result


def run_benchmark(items=None, progress=True):

    items = items if items is not None else load_benchmark()

    results = []

    for i, item in enumerate(items, start=1):

        if progress:
            print(f"[{i}/{len(items)}] {item['category']}: {item['turns'][-1][:70]}")

        result = run_benchmark_item(item)
        results.append(result)

        if progress:
            print("  PASS" if result["passed"] else f"  FAIL - {result['failure_reason']}")

    return summarize(results)


def summarize(results):

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    citation_checked = [r for r in results if r["citation_checked"]]
    citation_correct = [r for r in citation_checked if r["citation_correct"]]

    decline_expected = [r for r in results if r["expect_decline"]]
    hallucinated = [r for r in decline_expected if not r["passed"]]

    retrieval_expected = [r for r in results if r["requires_retrieval"]]
    retrieval_succeeded = [r for r in retrieval_expected if r["retrieval_success"]]

    structured = [r for r in results if r["response_type"] in _STRUCTURED_RESPONSE_TYPES]
    structured_correct = [r for r in structured if r["passed"]]

    hybrid = [r for r in results if r["response_type"] == "vector"]
    hybrid_correct = [r for r in hybrid if r["passed"]]

    avg_response_time = (
        sum(r["elapsed_seconds"] for r in results) / total if total else 0.0
    )

    category_breakdown = defaultdict(lambda: [0, 0])
    for r in results:
        bucket = category_breakdown[r["category"]]
        bucket[1] += 1
        if r["passed"]:
            bucket[0] += 1

    failed = [r for r in results if not r["passed"]]

    def _rate(numerator, denominator):
        return (numerator / denominator) if denominator else None

    return {
        "total_questions": total,
        "answer_accuracy": _rate(passed, total),
        "citation_accuracy": _rate(len(citation_correct), len(citation_checked)),
        "citation_checked_count": len(citation_checked),
        "hallucination_rate": _rate(len(hallucinated), len(decline_expected)),
        "decline_expected_count": len(decline_expected),
        "retrieval_success_rate": _rate(len(retrieval_succeeded), len(retrieval_expected)),
        "retrieval_expected_count": len(retrieval_expected),
        "structured_retrieval_accuracy": _rate(len(structured_correct), len(structured)),
        "structured_count": len(structured),
        "hybrid_retrieval_accuracy": _rate(len(hybrid_correct), len(hybrid)),
        "hybrid_count": len(hybrid),
        "average_response_time": avg_response_time,
        "category_breakdown": {
            category: {"passed": p, "total": t, "accuracy": _rate(p, t)}
            for category, (p, t) in category_breakdown.items()
        },
        "failed_questions": [
            {
                "id": r["id"],
                "category": r["category"],
                "question": r["question"],
                "response_type": r["response_type"],
                "failure_reason": r["failure_reason"],
                "response_snippet": r["content"][:200],
            }
            for r in failed
        ],
        "results": results,
    }
