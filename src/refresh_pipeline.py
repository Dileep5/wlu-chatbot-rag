"""Weekly data refresh pipeline.

Orchestrates the existing ingestion scripts end-to-end, in dependency
order, and writes a timestamped log (logs/refresh_<UTC timestamp>.log):

  Step 1  crawler.py                     -> urls.txt
  Step 2  scrape.py                      -> outputs/raw_pages.csv
  Step 3  clean.py                       -> outputs/clean_pages.csv
  Step 4  chunk.py                       -> outputs/chunks.csv
  Step 5  Rebuild all structured databases (schema reset, then
          undergraduate calendar sync, graduate calendar sync, and
          faculty directory sync)
  Step 6  Rebuild ChromaDB vector databases (content + faculty research)

No retrieval, ranking, prompting, renderer, or evaluation code is
imported or touched here - this script only launches the existing
standalone ingestion scripts as subprocesses, exactly as a human
operator would run them by hand.

Run manually:
    python3 src/refresh_pipeline.py

Exit code is 0 on success, 1 if any step fails (the pipeline stops at
the first failing step, since every later step depends on an earlier
one's output).
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"


def count_csv_rows(path: Path):
    """Row count excluding the header, or None if the file doesn't exist."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return max(sum(1 for _ in f) - 1, 0)


def count_lines(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


# Each step reuses an existing script unmodified (aside from the path-
# portability fix applied to scrape.py/chunk.py/build_vector_db.py/
# build_faculty_vector_db.py so they no longer hardcode one developer's
# machine). count_fn/count_label surface a "number of pages processed"
# style figure in the log wherever the step's output makes that
# derivable from the resulting file, without parsing each script's own
# print statements (their formats aren't uniform).
STEPS = [
    dict(
        name="crawl", label="Step 1: Crawl WLU site", script="crawler.py",
        count_fn=lambda: count_lines(BASE_DIR / "urls.txt"),
        count_label="URLs discovered",
    ),
    dict(
        name="scrape", label="Step 2: Scrape page content", script="scrape.py",
        count_fn=lambda: count_csv_rows(OUTPUTS_DIR / "raw_pages.csv"),
        count_label="pages scraped",
    ),
    dict(
        name="clean", label="Step 3: Clean scraped text", script="clean.py",
        count_fn=lambda: count_csv_rows(OUTPUTS_DIR / "clean_pages.csv"),
        count_label="pages cleaned",
    ),
    dict(
        name="chunk", label="Step 4: Chunk cleaned text", script="chunk.py",
        count_fn=lambda: count_csv_rows(OUTPUTS_DIR / "chunks.csv"),
        count_label="chunks produced",
    ),

    # Step 5: Rebuild all structured databases.
    # 5a. Schema reset - each create_*_table.py does DROP TABLE IF EXISTS
    # then CREATE TABLE, so it is safe to re-run and gives every refresh
    # a genuinely rebuilt schema, not an incrementally patched one.
    dict(name="create_courses_table", label="Step 5: Reset courses schema",
         script="create_courses_table.py"),
    dict(name="create_departments_table", label="Step 5: Reset departments schema",
         script="create_departments_table.py"),
    dict(name="create_faculty_table", label="Step 5: Reset faculty schema",
         script="create_faculty_table.py"),
    dict(name="create_faculty_courses_table", label="Step 5: Reset faculty_courses_taught schema",
         script="create_faculty_courses_table.py"),
    dict(name="create_programs_table", label="Step 5: Reset programs schema",
         script="create_programs_table.py"),
    dict(name="create_course_prerequisite_refs_table", label="Step 5: Reset course_prerequisite_refs schema",
         script="create_course_prerequisite_refs_table.py"),
    dict(name="create_program_course_requirements_table", label="Step 5: Reset program_course_requirements schema",
         script="create_program_course_requirements_table.py"),

    # 5b. Undergraduate calendar. get_undergraduate_program_links.py reads
    # outputs/undergraduate_departments.csv, which sync_undergraduate.py is
    # about to overwrite with this week's data - it runs first here so
    # load_programs (called inside sync_undergraduate.py) has an
    # undergraduate_program_links.csv to read at all. That means program
    # links trail the department list by one refresh cycle, which is a
    # non-issue since department structure changes far slower than weekly.
    dict(
        name="undergrad_program_links", label="Step 5: Refresh undergraduate program links",
        script="get_undergraduate_program_links.py",
        count_fn=lambda: count_csv_rows(OUTPUTS_DIR / "undergraduate_program_links.csv"),
        count_label="undergraduate program links",
    ),
    dict(name="sync_undergraduate", label="Step 5: Sync undergraduate calendar (faculties, departments, programs, courses)",
         script="sync_undergraduate.py"),

    # 5c. Graduate calendar (cal=3, year=94 - baked into each script's own
    # __main__ block already, reused here as-is).
    dict(
        name="grad_faculties", label="Step 5: Scrape graduate faculties", script="get_faculties.py",
        count_fn=lambda: count_csv_rows(OUTPUTS_DIR / "faculties.csv"),
        count_label="graduate faculties",
    ),
    dict(
        name="grad_departments", label="Step 5: Scrape graduate departments", script="save_departments.py",
        count_fn=lambda: count_csv_rows(OUTPUTS_DIR / "departments.csv"),
        count_label="graduate departments",
    ),
    dict(
        name="grad_programs", label="Step 5: Scrape graduate programs", script="save_graduate_programs.py",
        count_fn=lambda: count_csv_rows(OUTPUTS_DIR / "graduate_programs.csv"),
        count_label="graduate programs",
    ),
    dict(
        name="grad_course_links", label="Step 5: Scrape graduate course links", script="get_all_course_links.py",
        count_fn=lambda: count_csv_rows(OUTPUTS_DIR / "course_links.csv"),
        count_label="graduate course links",
    ),
    dict(name="grad_load_departments", label="Step 5: Load graduate departments into departments.db",
         script="load_departments.py"),
    dict(name="grad_load_programs", label="Step 5: Load graduate programs into programs.db",
         script="load_programs.py"),
    dict(name="grad_load_courses", label="Step 5: Load graduate courses into courses.db",
         script="laod_courses.py"),

    # 5d. Faculty directory (calendar-independent).
    dict(
        name="faculty_links", label="Step 5: Scrape faculty directory", script="get_faculty_links.py",
        count_fn=lambda: count_csv_rows(OUTPUTS_DIR / "faculty_links.csv"),
        count_label="faculty profiles",
    ),
    dict(name="load_faculty", label="Step 5: Load faculty into faculty.db",
         script="load_faculty.py"),

    # Step 6: Rebuild ChromaDB vector databases.
    dict(name="build_vector_db", label="Step 6: Rebuild content vector DB",
         script="build_vector_db.py"),
    dict(name="build_faculty_vector_db", label="Step 6: Rebuild faculty research vector DB",
         script="build_faculty_vector_db.py"),
]


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_step(step: dict, log_lines: list[str]) -> bool:
    script_path = SRC_DIR / step["script"]
    print(f"\n--- {step['label']} ({step['script']}) ---")

    start = datetime.now(timezone.utc)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    duration = (datetime.now(timezone.utc) - start).total_seconds()

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    ok = result.returncode == 0
    status = "OK" if ok else "FAIL"

    count_note = ""
    if ok and step.get("count_fn") is not None:
        count = step["count_fn"]()
        if count is not None:
            count_note = f"  ({count} {step['count_label']})"

    log_lines.append(
        f"[{status}] {step['label']:<70} {duration:6.1f}s{count_note}"
    )
    if not ok:
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        log_lines.append(f"       Error (exit code {result.returncode}):")
        for line in stderr_tail.splitlines():
            log_lines.append(f"         {line}")

    return ok


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_start = datetime.now(timezone.utc)
    timestamp = pipeline_start.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"refresh_{timestamp}.log"

    log_lines: list[str] = []
    success = True

    for step in STEPS:
        try:
            ok = run_step(step, log_lines)
        except Exception as exc:
            log_lines.append(f"[FAIL] {step['label']:<70}    -    ")
            log_lines.append(f"       Exception: {exc}")
            ok = False

        # Write the log incrementally so a partial run still leaves a
        # useful record behind if the pipeline crashes hard.
        _write_log(log_path, pipeline_start, log_lines, finished=False)

        if not ok:
            success = False
            log_lines.append("\nPipeline halted after first failure.")
            break

    pipeline_end = datetime.now(timezone.utc)
    _write_log(log_path, pipeline_start, log_lines, finished=True,
               end=pipeline_end, success=success)

    print(f"\nRefresh {'SUCCEEDED' if success else 'FAILED'}. Log written to {log_path}")
    return 0 if success else 1


def _write_log(log_path: Path, start: datetime, log_lines: list[str],
                finished: bool, end: datetime | None = None, success: bool | None = None):
    header = [
        "WLU Chatbot -- Data Refresh Pipeline Log",
        "=" * 42,
        f"Run started (UTC):  {start.isoformat(timespec='seconds')}",
    ]
    if finished:
        duration = (end - start).total_seconds()
        header.append(f"Run finished (UTC): {end.isoformat(timespec='seconds')}")
        header.append(f"Total duration:     {format_duration(duration)}")
        header.append(f"Overall result:     {'SUCCESS' if success else 'FAILURE'}")
    else:
        header.append("Run finished (UTC): (in progress)")

    header.append("")
    header.append("Step-by-step results:")
    header.append("-" * 42)

    log_path.write_text("\n".join(header + log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
