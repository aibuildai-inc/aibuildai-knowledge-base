"""Per-repo LLM extraction of awesome-list entries.

Takes the in-scope repos from filter_out/annotated.yaml, spawns one
ClaudeSDK agent per repo with Read/Write/Bash in a dedicated work
directory, and lets the agent write/run/self-check an extraction
script until its output looks reasonable.

Output layout (under the shared work dir resolved by resolve_work_dir):

    <run-dir>/
        extraction_out/
            checkpoint.json             # atomic, per-repo completion state
            <owner>__<repo>/
                output.json             # final StructuredOutput {entries, notes}
                work/
                    README.md           # pre-staged by this driver
                    <agent-authored files>  # extract.py, entries.json, notes...
        steps/<owner>__<repo>/          # rendered SDK transcript (execution.md, ...)

CLI:
    python -m sourcing.awesome_lists.extract --run-id <id>
    # reads  <run-dir>/filter_out/annotated.yaml
    # writes <run-dir>/extraction_out/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from sourcing.awesome_lists.parse_readme import fetch_readme
from sourcing.utils.agent_calls import AgentCalls, PermanentError, sum_costs
from sourcing.utils.work_dir import (
    load_checkpoint,
    resolve_work_dir,
    save_checkpoint,
    step_dir,
)

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint.json"

ENTRIES_FILENAME = "entries.json"


class PaperEntry(BaseModel):
    """Minimal schema for one paper. Only title + pdf_url required.

    Sub-agent writes a list[PaperEntry] to work/entries.json. Extra fields
    pass through (extra='allow') so humans can audit richer context when
    the sub-agent captured it.
    """

    model_config = {"extra": "allow"}

    title: str
    pdf_url: str


class ExtractionNotes(BaseModel):
    """Only structured payload the sub-agent sends back. Entries themselves
    live in work/entries.json on disk.
    """

    model_config = ConfigDict(extra="forbid")

    notes: str


_PROMPT = """\
You are extracting paper entries from a GitHub `awesome-*` list README.

INPUT
- Repo: {repo}
- Repo description: {description}
- README file: {work_dir}/README.md
- Your work directory: {work_dir}  (writable; Bash/Read/Write allowed here)

REQUIRED OUTPUT -- TWO PARTS
1. Write {work_dir}/entries.json -- a JSON array where each element is an
   object with AT MINIMUM:
     - "title":   string, the paper title (not a link text like "Paper",
                  "here", or a GitHub repo name)
     - "pdf_url": string, a link to the paper itself (arXiv abs/pdf URL,
                  DOI, conference proceedings page, or the paper's project
                  page). NEVER a GitHub repo URL, shields.io badge,
                  in-page anchor (#...), or the README itself.
   You MAY include additional fields for human review:
     - "authors", "venue", "year", "category", "subcategory",
       "description", "code_url", etc.
   Extra fields are preserved but not validated.

2. Return a StructuredOutput with a single field `notes` -- a free-form
   paragraph describing:
     - What the README format looks like (bullets / table / <details> /
       per-entry headers).
     - How you decided which URLs count as `pdf_url`.
     - Any sections you skipped and why.
     - Any uncertainty about edge cases a reviewer should know.

COMPLETENESS CONTRACT -- NON-NEGOTIABLE
Write every paper that matches the criteria into entries.json. This is
not a demo, a sample, or a preview. The downstream pipeline expects
the complete extraction. Partial, sampled, demonstrative, or
"representative" outputs are FAILURES -- they waste the whole run.

You must NOT:
- Stop after N "representative" entries and call the extraction complete.
- Say "for demonstration I extracted X out of Y" and leave Y unfilled.
- Truncate because the list "looks long enough" or "gets the idea across".
- Skip sections because they seem redundant with earlier sections.
- Hold back entries to keep output brief.

The entries live on disk, written directly by your script. There is NO
size limit on entries.json, NO tool-parameter cap, NO reason to stop
early. If your script parses 2000 papers from the README, write 2000
papers to entries.json. If it parses 20, write 20. If 0, write `[]`.

If you are uncertain whether a line is a paper, INCLUDE it rather
than exclude it -- downstream filters will handle borderline cases.

DO NOT attempt to pass the entries array through the StructuredOutput
tool. The entries live on disk. StructuredOutput receives only `notes`.

APPROACH
Write an extraction script in your work directory, run it, inspect the
output JSON, iterate until the count on disk matches what you believe
the README contains. Python stdlib only (no pip install). Keep
intermediate scripts and dumps in the work directory for later
inspection.

SCOPE -- WRITE `entries.json` AS `[]` (EMPTY ARRAY) IF:
- The README is mostly tutorials, courses, websites, blog posts,
  cheatsheets, videos, books, or general-purpose tools/libraries
  rather than a curated list of research papers.
- The README is a list of other awesome-lists (a list-of-lists).
- Papers, if any, are only an afterthought compared to non-paper
  resources.

Mixed lists with a few papers among many tools should also write `[]`.
When writing `[]`, explain in `notes` what the list actually contains
so we can audit the decision. Writing `[]` for an out-of-scope README
is the correct action -- it is NOT the same as a partial extraction.

QUALITY HINTS
- Titles like "here", "Paper", "pull requests", raw `<img>` tags,
  single letters are extraction errors -- fix the script.
- `pdf_url` pointing to shields.io, star-history, in-page anchor,
  or README file means you picked the wrong link on that line.
- Many entries with empty category or one giant catch-all category
  suggests the script missed section headers -- fix the script.
- An entry whose only link is a GitHub repo with no paper link is
  probably not a paper entry -- skip it.

Awesome lists come in many flavors: `-`/`*`/`+` bullets, numbered
lists, `<details>` blocks, tables, badge-image links
`[![badge](img)](real-url)`, or `## name` headers with `- key: <url>`
underneath -- each needs a different approach, and you are responsible
for adapting the script until it extracts every paper in the README.
"""


def _slug(full_name: str) -> str:
    return full_name.replace("/", "__")


def _is_done(entry: dict | None) -> bool:
    """Treat only successful runs as done; re-try on fetch/SDK failures."""
    return bool(entry) and "entries_count" in entry


def _init_ck(raw: dict) -> dict:
    """Normalize checkpoint to {'summary': {...}, 'repos': {slug: ...}} shape.

    Accepts either the new shape or a legacy flat {slug: entry} dict.
    """
    if "repos" in raw and isinstance(raw.get("repos"), dict):
        raw.setdefault("summary", {})
        return raw
    return {"summary": {}, "repos": raw}


def _recompute_summary(ck: dict) -> None:
    repos = ck["repos"]
    done = [v for v in repos.values() if _is_done(v)]
    with_entries = [v for v in done if v.get("entries_count", 0) > 0]
    # An entry holds cost_usd only once a call was made; an entry without it
    # (a failed fetch, a repo not reached yet) spent nothing.
    total_cost = sum_costs(*(v["cost_usd"] for v in repos.values() if "cost_usd" in v))
    ck["summary"] = {
        "total_repos_processed": len(done),
        "total_repos_with_entries": len(with_entries),
        "total_repos_empty": len(done) - len(with_entries),
        "total_repos_errored": sum(1 for v in repos.values() if "error" in v),
        "total_entries": sum(v.get("entries_count", 0) for v in done),
        "total_cost_usd": None if total_cost is None else round(total_cost, 6),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _save(ck_path: Path, ck: dict) -> None:
    _recompute_summary(ck)
    save_checkpoint(ck_path, ck)


def _prepare_work_dir(repo_dir: Path, full_name: str) -> Path:
    """Create <repo_dir>/work and pre-stage README.md; return the work path."""
    work = repo_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    readme = fetch_readme(full_name)
    (work / "README.md").write_text(readme, encoding="utf-8")
    return work


def _read_entries_file(work: Path) -> tuple[list[dict], str | None]:
    """Load and validate work/entries.json.

    Returns (entries, None) on success, ([], reason) on any failure.
    Failure reasons: missing file, invalid JSON, non-array root, or any
    item failing PaperEntry validation. The reason goes back into the
    agent's own session through the verify loop, so the session that
    wrote the bad file fixes it before the call returns.
    """
    entries_path = work / ENTRIES_FILENAME
    if not entries_path.exists():
        return [], f"missing {ENTRIES_FILENAME}"
    try:
        raw = json.loads(entries_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], f"invalid json in {ENTRIES_FILENAME}: {exc}"
    if not isinstance(raw, list):
        return [], f"{ENTRIES_FILENAME} is not a JSON array"
    validated: list[dict] = []
    for i, item in enumerate(raw):
        try:
            entry = PaperEntry.model_validate(item)
        except Exception as exc:
            return [], f"{ENTRIES_FILENAME}[{i}] invalid: {exc}"
        validated.append(entry.model_dump())
    return validated, None


async def _one_repo(
    calls: AgentCalls,
    repo: dict,
    output_dir: Path,
    wd: Path,
    sem: asyncio.Semaphore,
    ck: dict,
    ck_path: Path,
    model: str,
) -> None:
    full_name = repo["full_name"]
    slug = _slug(full_name)
    repos = ck["repos"]
    if _is_done(repos.get(slug)):
        logger.info("[skip] %s (already done)", full_name)
        return

    entry = repos.setdefault(slug, {})
    repo_dir = output_dir / slug
    try:
        work = _prepare_work_dir(repo_dir, full_name)
    except Exception as exc:
        logger.exception("[fetch-fail] %s", full_name)
        entry["error"] = f"fetch: {exc}"
        _save(ck_path, ck)
        return

    prompt = _PROMPT.format(
        repo=full_name,
        description=(repo.get("description") or "").strip() or "(none)",
        work_dir=str(work),
    )

    async def verify(_result: dict) -> "str | None":
        _entries, file_err = _read_entries_file(work)
        return file_err

    async with sem:
        call_dir = step_dir(wd, slug)
        spent_before = calls.spent.get(call_dir, 0.0)
        try:
            async with calls.conversation(
                name=slug,
                model=model,
                tools=("Read", "Write", "Bash", "Glob", "Grep"),
                cwd=str(work),
                output_schema=ExtractionNotes.model_json_schema(),
                max_turns=60,
                add_dirs=[str(work)],
            ) as session:
                rec = await session.ask(
                    user_prompt=prompt,
                    step_dir=call_dir,
                    verify=verify,
                )
        except PermanentError as exc:
            # Hard failures (retry ceiling, auth/billing, invalid request): do
            # not poison the checkpoint with a bogus per-repo error. Stop and
            # leave the repo pending for next run.
            logger.error("[halt] %s: %s: %s", full_name, type(exc).__name__, exc)
            raise
        except Exception as exc:
            logger.exception("[fail] %s", full_name)
            entry["error"] = str(exc)
            return
        finally:
            if call_dir in calls.spent:
                entry["cost_usd"] = sum_costs(entry.get("cost_usd", 0.0), calls.spent[call_dir], None if spent_before is None else -spent_before)
            _save(ck_path, ck)

    assert isinstance(rec.result, dict)
    notes = (rec.result.get("notes") or "").strip()
    entries, file_err = _read_entries_file(work)
    assert file_err is None, file_err  # verify accepted this exact file

    # Driver-owned canonical output: downstream consumers read this, not the
    # sub-agent's scratch files under work/.
    (repo_dir / "output.json").write_text(
        json.dumps({"entries": entries, "notes": notes}, indent=2),
        encoding="utf-8",
    )

    repos[slug] = {
        "entries_count": len(entries),
        "notes": notes,
        "cost_usd": entry["cost_usd"],
    }
    _save(ck_path, ck)
    logger.info(
        "[done] %s -> %d entries (cost USD: %s)",
        full_name, repos[slug]["entries_count"], repos[slug]["cost_usd"],
    )


async def _amain(
    calls: AgentCalls,
    annotated: Path,
    output_dir: Path,
    wd: Path,
    *,
    model: str,
    workers: int,
    limit: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ck_path = output_dir / CHECKPOINT_NAME
    ck = _init_ck(load_checkpoint(ck_path))

    annotated_data = yaml.safe_load(annotated.read_text())
    repos = [repo for repo in annotated_data.get("repos", []) if repo.get("verdict")]
    if limit:
        repos = repos[:limit]

    pending = [
        r for r in repos if not _is_done(ck["repos"].get(_slug(r["full_name"])))
    ]
    done = len(repos) - len(pending)
    logger.info(
        "[extract] total=%d done=%d pending=%d workers=%d",
        len(repos), done, len(pending), workers,
    )

    sem = asyncio.Semaphore(workers)
    try:
        await asyncio.gather(
            *(
                _one_repo(calls, r, output_dir, wd, sem, ck, ck_path, model)
                for r in pending
            )
        )
    except PermanentError as exc:
        _save(ck_path, ck)
        logger.error(
            "[extract] halted by %s: %s. "
            "Remaining pending repos preserved in checkpoint; rerun after recovery.",
            type(exc).__name__, exc,
        )
        raise SystemExit(2) from exc
    _save(ck_path, ck)  # ensure summary is current even if nothing ran
    s = ck["summary"]
    logger.info(
        "[extract] done. repos=%d entries=%d cost USD=%s -> %s",
        s["total_repos_processed"], s["total_entries"], s["total_cost_usd"], ck_path,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Per-repo LLM extraction driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotated", type=Path, default=None,
                   help="annotated.yaml (default: <run-dir>/filter_out/annotated.yaml)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="extraction_out dir (default: <run-dir>/extraction_out)")
    p.add_argument("--workers", type=int, default=4, help="Parallel LLM calls")
    p.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model used to extract repository entries",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N in-scope repos (for smoke tests).",
    )
    p.add_argument("--run-id", default=None,
                   help="Shared run id; the three awesome_lists CLIs MUST share it.")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Override the resolved work dir entirely.")
    args = p.parse_args()
    wd = resolve_work_dir("awesome_lists", run_id=args.run_id, override=args.work_dir)
    annotated = args.annotated or (wd / "filter_out" / "annotated.yaml")
    output_dir = args.output_dir or (wd / "extraction_out")
    calls = AgentCalls()
    asyncio.run(
        _amain(
            calls,
            annotated,
            output_dir,
            wd,
            model=args.model,
            workers=args.workers,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
