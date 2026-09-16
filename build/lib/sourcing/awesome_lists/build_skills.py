"""Convert extraction_out/<slug>/output.json into plugin skill directories.

Reads LLM-extracted entries from extract.py's output and renders
SKILL.md + references/*.md for each in-scope repo. No S2 enrichment;
extra fields flow through to frontmatter.

Input (under the shared work dir resolved by resolve_work_dir):
    <run-dir>/extraction_out/checkpoint.json     # selects repos with entries_count > 0
    <run-dir>/extraction_out/<slug>/output.json  # {entries: [...], notes: "..."}
    <run-dir>/filter_out/annotated.yaml          # upstream description per repo

Output (the served corpus indexed by kb):
    data/corpus/plugins/aibuildai-modeling-marketplace/aibuildai-awesome-lists/skills/
        <slug>/
            SKILL.md
            references/<entry-slug>.md

CLI:
    python -m sourcing.awesome_lists.build_skills --run-id <id>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from sourcing.awesome_lists.parse_readme import OUTPUT_DIR
from sourcing.utils.work_dir import resolve_work_dir

logger = logging.getLogger(__name__)

def slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:80].rstrip("-") or "untitled"


def _fm_value(v: Any) -> str:
    """YAML-safe single-line frontmatter value."""
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return str(v)
    s = str(v).replace("\n", " ").strip()
    # Always quote to avoid YAML parsing surprises (colons, brackets, leading #).
    esc = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{esc}"'


REFERENCE_FIELDS = [
    "category",
    "subcategory",
    "venue",
    "year",
    "authors",
    "code_url",
    "page_url",
    "description",
]


def _render_reference(entry: dict, upstream: str) -> str:
    title = entry.get("title", "").strip()
    pdf_url = entry.get("pdf_url", "").strip()
    fm = [
        "---",
        f"title: {_fm_value(title)}",
        "entry_type: paper",
        f"source: {_fm_value(pdf_url)}",
        f"upstream_list: {_fm_value(upstream)}",
    ]
    for k in REFERENCE_FIELDS:
        v = entry.get(k)
        if v in (None, "", []):
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        fm.append(f"{k}: {_fm_value(v)}")
    fm.append("---")

    body = ["", f"# {title}", ""]
    if pdf_url:
        body.append(f"**Source**: [{pdf_url}]({pdf_url})")
        body.append("")
    if entry.get("code_url"):
        code = entry["code_url"]
        body.append(f"**Code**: [{code}]({code})")
        body.append("")
    if entry.get("page_url"):
        page = entry["page_url"]
        body.append(f"**Page**: [{page}]({page})")
        body.append("")
    meta = []
    for k in ("year", "venue", "category", "subcategory"):
        v = entry.get(k)
        if v:
            meta.append(f"**{k.capitalize()}**: {v}")
    if meta:
        body.append(" | ".join(meta))
        body.append("")
    if entry.get("authors"):
        authors = entry["authors"]
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors)
        body.append(f"**Authors**: {authors}")
        body.append("")
    if entry.get("description"):
        body += ["## Description", "", str(entry["description"]).strip(), ""]
    return "\n".join(fm + body).rstrip() + "\n"


def _render_skill_md(
    *, slug: str, description: str, upstream: str, entries: list[dict]
) -> str:
    entry_rows = []
    by_cat: dict[str, list[dict]] = {}
    for i, e in enumerate(entries, 1):
        cat = (e.get("category") or "").strip() or "Uncategorized"
        by_cat.setdefault(cat, []).append(e)
        code = "yes" if e.get("code_url") else ""
        year = e.get("year") or ""
        title = (e.get("title") or "").replace("|", "\\|")
        cat_cell = cat.replace("|", "\\|")
        fname = e["_ref_filename"]
        entry_rows.append(
            f"| {i} | {title} | {cat_cell} | {year} | {code} "
            f"| references/{fname} |"
        )

    parts = [
        "---",
        "description: >-",
        f"  {description or upstream}",
        "---",
        "",
        f"# {slug.replace('__', ' / ')} Knowledge Base",
        "",
        f"**Source**: [{upstream}](https://github.com/{upstream}) "
        f"({len(entries)} entries)",
        "",
        "## Entry Index",
        "",
        "| # | Title | Category | Year | Code | File |",
        "|---|-------|----------|------|------|------|",
        *entry_rows,
        "",
        "## Categories",
        "",
    ]
    for cat in sorted(by_cat):
        parts.append(f"### {cat}")
        parts.append("")
        for e in by_cat[cat]:
            title = e.get("title") or ""
            parts.append(f"- [{title}](references/{e['_ref_filename']})")
        parts.append("")
    return "\n".join(parts)


def _assign_ref_filenames(entries: list[dict]) -> None:
    """Mutate each entry in-place, adding `_ref_filename` with de-duplication."""
    seen: set[str] = set()
    for e in entries:
        base = slugify(e.get("title", ""))
        name = base
        n = 2
        while name in seen:
            name = f"{base}-{n}"
            n += 1
        seen.add(name)
        e["_ref_filename"] = f"{name}.md"


def _build_one(
    slug: str,
    entries: list[dict],
    description: str,
    output_dir: Path,
) -> tuple[Path, int]:
    """Write SKILL.md + references/*.md for one repo. Returns (skill_dir, count)."""
    upstream = slug.replace("__", "/", 1)
    skill_dir = output_dir / slug
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)

    _assign_ref_filenames(entries)
    for e in entries:
        (refs_dir / e["_ref_filename"]).write_text(
            _render_reference(e, upstream), encoding="utf-8"
        )
    (skill_dir / "SKILL.md").write_text(
        _render_skill_md(
            slug=slug, description=description, upstream=upstream, entries=entries
        ),
        encoding="utf-8",
    )
    return skill_dir, len(entries)


def _load_descriptions(annotated_path: Path) -> dict[str, str]:
    data = yaml.safe_load(annotated_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for r in data.get("repos", []):
        full = r.get("full_name", "")
        desc = (r.get("description") or "").strip()
        if full:
            out[full.replace("/", "__")] = desc
    return out


def run(
    extraction_out: Path,
    annotated: Path,
    output_dir: Path,
    min_entries: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ck = json.loads(
        (extraction_out / "checkpoint.json").read_text(encoding="utf-8")
    )
    descriptions = _load_descriptions(annotated)

    eligible = [
        slug for slug, v in ck.get("repos", {}).items()
        if v.get("entries_count", 0) >= min_entries
    ]
    eligible.sort()
    logger.info("[build_skills] eligible=%d (min_entries=%d)", len(eligible), min_entries)

    built = 0
    total_entries = 0
    failed: list[tuple[str, str]] = []
    for i, slug in enumerate(eligible, 1):
        src = extraction_out / slug / "output.json"
        if not src.exists():
            failed.append((slug, "missing output.json"))
            continue
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            entries = data.get("entries", [])
            if not entries:
                failed.append((slug, "entries array empty"))
                continue
            desc = descriptions.get(slug, "")
            skill_dir, n = _build_one(slug, entries, desc, output_dir)
        except Exception as exc:
            failed.append((slug, str(exc)))
            logger.exception("[build_skills] %s", slug)
            continue
        built += 1
        total_entries += n
        if i % 50 == 0 or i == len(eligible):
            logger.info(
                "[build_skills] %d/%d built=%d entries=%d",
                i, len(eligible), built, total_entries,
            )

    stats = {
        "eligible": len(eligible),
        "built": built,
        "total_entries": total_entries,
        "failed": len(failed),
        "failures": failed,
        "output_dir": str(output_dir),
    }
    logger.info(
        "[build_skills] done: built=%d entries=%d failed=%d",
        built, total_entries, len(failed),
    )
    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Build plugin skills from extraction_out")
    p.add_argument("--extraction-out", type=Path, default=None,
                   help="extraction_out dir (default: <run-dir>/extraction_out)")
    p.add_argument("--annotated", type=Path, default=None,
                   help="annotated.yaml (default: <run-dir>/filter_out/annotated.yaml)")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--min-entries", type=int, default=1)
    p.add_argument("--run-id", default=None,
                   help="Shared run id; the three awesome_lists CLIs MUST share it.")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Override the resolved work dir entirely.")
    args = p.parse_args()
    wd = resolve_work_dir("awesome_lists", run_id=args.run_id, override=args.work_dir)
    extraction_out = args.extraction_out or (wd / "extraction_out")
    annotated = args.annotated or (wd / "filter_out" / "annotated.yaml")
    stats = run(extraction_out, annotated, args.output_dir, args.min_entries)
    print(json.dumps({k: v for k, v in stats.items() if k != "failures"}, indent=2))
    if stats["failures"]:
        print(f"\nfailures ({len(stats['failures'])}):")
        for slug, reason in stats["failures"][:20]:
            print(f"  {slug}: {reason}")
        if len(stats["failures"]) > 20:
            print(f"  ... +{len(stats['failures']) - 20} more")


if __name__ == "__main__":
    main()
