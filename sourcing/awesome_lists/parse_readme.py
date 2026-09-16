"""Single-file awesome-list ingestion pipeline.

Flow: YAML config -> fetch README -> regex parse -> (optional) S2 enrichment
  -> write skill directory (SKILL.md + references/*.md).

Data cache at data/<name>/entries/<id>.json enables enrichment resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

logger = logging.getLogger("awesome_lists")

PIPELINE_DIR = Path(__file__).parent
CONFIGS_DIR = PIPELINE_DIR / "configs"
DATA_DIR = PIPELINE_DIR / "data"
OUTPUT_DIR = (
    PIPELINE_DIR.parent.parent
    / "data"
    / "corpus"
    / "plugins"
    / "aibuildai-modeling-marketplace"
    / "aibuildai-awesome-lists"
    / "skills"
)


@dataclass
class CanonicalEntry:
    id: str
    title: str
    entry_type: str  # "paper" | "repo" | "package" | "tool"
    source_url: str
    category: str
    subcategory: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    enrichments: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "CanonicalEntry":
        d = json.loads(path.read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_ARXIV_SIGS = ("arxiv.org", "arxiv.abs", "arxiv.pdf")


def classify_url(url: str) -> str:
    u = url.lower()
    if any(s in u for s in _ARXIV_SIGS):
        return "paper"
    if "semanticscholar.org" in u or "doi.org" in u:
        return "paper"
    if "proceedings" in u or "openreview.net" in u:
        return "paper"
    if "pypi.org" in u or "npmjs.com" in u:
        return "package"
    if "github.com" in u:
        return "repo"
    return "tool"


def slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:80].rstrip("-")


_ENTRY_RE = re.compile(
    r"^[\s]*[-*]\s+"
    r"(?:\*\*)?(?:\[)?"
    r"\[([^\]]+)\]"
    r"\(([^)]+)\)"
    r"(?:\])?(?:\*\*)?"
    r"(?:\s*[-\u2013:]?\s*(.*))?"
)
_NUMBERED_RE = re.compile(
    r"^[\s]*\d+\.\s+"
    r"(?:\[([^\]]*)\]\s*)?"
    r"\*\*([^*]+)\*\*"
    r"(.*)"
)
_BRACKET_LINK_RE = re.compile(r"\[\[([^\]]*)\]\]\(([^)]+)\)")
_TABLE_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")


def parse_readme(text: str, config: dict) -> list[CanonicalEntry]:
    entries: list[CanonicalEntry] = []
    seen: set[str] = set()
    sections: dict[int, str] = {}
    exclude = {s.lower() for s in config.get("exclude_sections", [])}
    include_types = set(config.get("include_types", []))

    for line in text.split("\n"):
        h = _HEADING_RE.match(line)
        if h:
            level = len(h.group(1))
            sections[level] = h.group(2).strip()
            for k in list(sections):
                if k > level:
                    del sections[k]
            continue

        if any(s.lower() in exclude for s in sections.values()):
            continue

        title, url, desc = None, None, ""

        m = _ENTRY_RE.match(line)
        if m:
            title = m.group(1).strip()
            url = m.group(2).strip()
            desc = (m.group(3) or "").strip()

        if not title:
            m = _NUMBERED_RE.match(line)
            if m:
                venue = (m.group(1) or "").strip()
                title = m.group(2).strip()
                rest = m.group(3)
                links = _BRACKET_LINK_RE.findall(rest)
                if links:
                    for label, href in links:
                        if label.lower() in ("paper", "arxiv", "pdf"):
                            url = href
                            break
                    if not url:
                        url = links[0][1]
                if venue:
                    desc = venue

        if not title and ("[[" in line or ("[" in line and "](" in line)):
            for ltitle, lurl in _TABLE_LINK_RE.findall(line):
                if (
                    lurl.startswith("http")
                    and "badge" not in lurl
                    and "shield" not in lurl
                    and "img." not in lurl
                ):
                    title, url = ltitle.strip(), lurl.strip()
                    break

        if not title or not url:
            continue
        if url.startswith("#") or not url.startswith("http"):
            continue

        etype = classify_url(url)
        if include_types and etype not in include_types:
            continue

        levels = sorted(sections.keys())
        category = sections.get(levels[0], "Uncategorized") if levels else "Uncategorized"
        subcategory = sections.get(levels[-1], "") if len(levels) > 1 else ""

        eid = slugify(title)
        if eid in seen:
            n = 2
            while f"{eid}-{n}" in seen:
                n += 1
            eid = f"{eid}-{n}"
        seen.add(eid)

        entries.append(
            CanonicalEntry(
                id=eid,
                title=title,
                entry_type=etype,
                source_url=url,
                category=category,
                subcategory=subcategory,
                metadata={"description": desc, "source_repo": config["repo"]},
            )
        )
    return entries


README_CANDIDATES = ("README.md", "readme.md", "Readme.md", "README.MD")


def fetch_readme(repo: str) -> str:
    """Fetch a repo's README via raw.githubusercontent.com.

    Tries (branch, filename) combinations in order: branches main/master,
    filenames README.md / readme.md / Readme.md / README.MD. Returns on
    first HTTP 200. On a single 429 or rate-limit signal, aborts early
    with a dedicated message. Otherwise aggregates every attempt's HTTP
    status into the RuntimeError so the caller can tell 404 from 429
    without reading the stack trace.
    """
    attempts: list[tuple[str, str, int]] = []
    for branch in ("main", "master"):
        for name in README_CANDIDATES:
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{name}"
            resp = requests.get(url, timeout=30)
            attempts.append((branch, name, resp.status_code))
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 403):
                retry_after = resp.headers.get("Retry-After", "?")
                rl_remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                rl_reset = resp.headers.get("X-RateLimit-Reset", "?")
                raise RuntimeError(
                    f"rate-limited fetching {repo}/{branch}/{name}: "
                    f"HTTP {resp.status_code} "
                    f"(Retry-After={retry_after}, X-RateLimit-Remaining={rl_remaining}, "
                    f"X-RateLimit-Reset={rl_reset})"
                )
    detail = ", ".join(f"{b}/{n}={s}" for b, n, s in attempts)
    raise RuntimeError(f"Could not fetch README for {repo}: tried [{detail}]")


def collect(config: dict) -> list[CanonicalEntry]:
    readme_url = config.get("readme_url")
    if readme_url:
        resp = requests.get(readme_url, timeout=30)
        resp.raise_for_status()
        text = resp.text
    else:
        text = fetch_readme(config["repo"])
    entries = parse_readme(text, config)
    logger.info("Collected %d entries from %s", len(entries), config["repo"])
    return entries


S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,year,citationCount,authors,venue,tldr,externalIds,url"
S2_DELAY = 3.0
S2_MAX_RETRIES = 3

_ARXIV_ID_RES = [
    re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)"),
    re.compile(r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)"),
    re.compile(r"arxiv\.org/abs/([\w.-]+/\d{7})"),
]


def extract_arxiv_id(url: str) -> str | None:
    for r in _ARXIV_ID_RES:
        m = r.search(url)
        if m:
            return m.group(1)
    return None


def _s2_get(path: str, params: dict | None = None) -> dict | None:
    url = f"{S2_API}{path}"
    for attempt in range(S2_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException:
            logger.exception("S2 request error (attempt %d)", attempt + 1)
            time.sleep(3)
            continue
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            logger.warning("S2 rate limited, waiting %ds", wait)
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            return None
        return resp.json()
    return None


def _s2_fetch(entry: CanonicalEntry) -> dict | None:
    arxiv_id = extract_arxiv_id(entry.source_url)
    if arxiv_id:
        data = _s2_get(f"/paper/ARXIV:{arxiv_id}", {"fields": S2_FIELDS})
        if data:
            return data
    data = _s2_get(
        "/paper/search",
        {"query": entry.title, "fields": S2_FIELDS, "limit": 1},
    )
    if data and data.get("data"):
        return data["data"][0]
    return None


def _s2_extract(data: dict) -> dict:
    authors = [a.get("name", "") for a in (data.get("authors") or [])]
    tldr = data.get("tldr")
    tldr_text = tldr.get("text", "") if isinstance(tldr, dict) else ""
    ext = data.get("externalIds") or {}
    return {
        "abstract": data.get("abstract") or "",
        "year": data.get("year"),
        "citation_count": data.get("citationCount", 0),
        "authors": authors,
        "venue": data.get("venue") or "",
        "tldr": tldr_text,
        "arxiv_id": ext.get("ArXiv") or "",
        "doi": ext.get("DOI") or "",
        "s2_url": data.get("url") or "",
        "s2_paper_id": data.get("paperId") or "",
    }


def enrich_papers(entries: list[CanonicalEntry]) -> list[CanonicalEntry]:
    papers = [e for e in entries if e.entry_type == "paper"]
    others = [e for e in entries if e.entry_type != "paper"]
    logger.info(
        "Enriching %d papers (skipping %d non-papers)", len(papers), len(others)
    )
    for i, entry in enumerate(papers):
        if entry.enrichments.get("s2_enriched"):
            continue
        logger.info("[%d/%d] %s", i + 1, len(papers), entry.title[:60])
        data = _s2_fetch(entry)
        if data:
            entry.enrichments.update(_s2_extract(data))
            entry.enrichments["s2_enriched"] = True
        else:
            entry.enrichments["s2_enriched"] = False
            logger.warning("  not found on S2: %s", entry.title[:60])
        time.sleep(S2_DELAY)
    return papers + others


def _render_reference(entry: CanonicalEntry, source_repo: str) -> str:
    from sourcing.utils.skill_output import AwesomeListEntry, dump_frontmatter

    e = entry.enrichments
    schema_entry = AwesomeListEntry(
        entry_type="awesome-list",
        title=entry.title,
        source=entry.source_url,
        tags=e.get("tags", []),
        tldr=e.get("tldr", ""),
        categories=[entry.category] if entry.category else [],
        upstream_list=source_repo,
        arxiv_id=e.get("arxiv_id"),
        doi=e.get("doi"),
    )
    fm = dump_frontmatter(schema_entry).rstrip()
    parts = ["---", fm, "---", "", f"# {entry.title}", ""]

    if e.get("authors"):
        authors = e["authors"]
        shown = ", ".join(authors[:10])
        if len(authors) > 10:
            shown += f" (+{len(authors) - 10} more)"
        parts += [f"**Authors**: {shown}", ""]

    meta = []
    if e.get("year"):
        meta.append(f"**Year**: {e['year']}")
    if e.get("venue"):
        meta.append(f"**Venue**: {e['venue']}")
    if e.get("citation_count"):
        meta.append(f"**Citations**: {e['citation_count']}")
    if meta:
        parts += [" | ".join(meta), ""]

    parts += [f"**Source**: [{entry.source_url}]({entry.source_url})", ""]
    if e.get("s2_url"):
        parts += [f"**Semantic Scholar**: [{e['s2_url']}]({e['s2_url']})"]
    parts.append("")

    if e.get("tldr"):
        parts += ["## TLDR", "", e["tldr"], ""]
    if e.get("abstract"):
        parts += ["## Abstract", "", e["abstract"], ""]

    return "\n".join(parts).rstrip() + "\n"


def _render_skill_md(
    name: str, description: str, source_repo: str, entries: list[CanonicalEntry]
) -> str:
    parts = [
        "---",
        "description: >-",
        f"  {description}",
        "---",
        "",
        f"# {name.replace('-', ' ').title()} Knowledge Base",
        "",
        f"**Source**: [{source_repo}](https://github.com/{source_repo})"
        f" ({len(entries)} entries)",
        "",
        "## Entry Index",
        "",
        "| # | Title | Year | Citations | Type | Category | TLDR | File |",
        "|---|-------|------|-----------|------|----------|------|------|",
    ]
    for i, entry in enumerate(entries, 1):
        e = entry.enrichments
        year = e.get("year", "\u2014")
        cites = e.get("citation_count", "\u2014")
        tldr = (e.get("tldr") or "")
        if len(tldr) > 100:
            tldr = tldr[:97] + "..."
        tldr = tldr.replace("|", "\\|")
        parts.append(
            f"| {i} | {entry.title} | {year} | {cites} "
            f"| {entry.entry_type} | {entry.category} "
            f"| {tldr} | references/{entry.id}.md |"
        )
    parts.append("")

    by_cat: dict[str, list[CanonicalEntry]] = {}
    for entry in entries:
        by_cat.setdefault(entry.category or "Uncategorized", []).append(entry)
    parts += ["## Categories", ""]
    for cat in sorted(by_cat):
        parts.append(f"### {cat}")
        parts.append("")
        for entry in by_cat[cat]:
            parts.append(f"- [{entry.title}](references/{entry.id}.md)")
        parts.append("")
    return "\n".join(parts)


def build_skill(
    entries: list[CanonicalEntry],
    output_dir: Path,
    name: str,
    description: str,
    source_repo: str,
) -> Path:
    skill_dir = output_dir / name
    refs = skill_dir / "references"
    refs.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (refs / f"{entry.id}.md").write_text(_render_reference(entry, source_repo))
    (skill_dir / "SKILL.md").write_text(
        _render_skill_md(name, description, source_repo, entries)
    )
    logger.info(
        "Built skill %s: %d entries -> %s", name, len(entries), skill_dir
    )
    return skill_dir


def save_entries(entries: list[CanonicalEntry], config: dict) -> Path:
    data_dir = DATA_DIR / config["name"]
    entries_dir = data_dir / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        entry.save(entries_dir / f"{entry.id}.json")
    logger.info("Saved %d entries to %s", len(entries), data_dir)
    return data_dir


def load_entries(config: dict) -> list[CanonicalEntry]:
    entries_dir = DATA_DIR / config["name"] / "entries"
    if not entries_dir.exists():
        raise FileNotFoundError(f"No canonical data at {entries_dir}")
    return [CanonicalEntry.load(p) for p in sorted(entries_dir.glob("*.json"))]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(
    config_path: Path,
    output_dir: Path,
    *,
    skip_enrich: bool,
) -> Path:
    config = load_config(config_path)
    name = config["name"]
    logger.info("=== Pipeline: %s ===", name)

    entries = collect(config)
    save_entries(entries, config)

    papers = [e for e in entries if e.entry_type == "paper"]
    if papers and not skip_enrich:
        entries = enrich_papers(entries)
        save_entries(entries, config)
        enriched = sum(1 for e in entries if e.enrichments.get("s2_enriched"))
        logger.info("  %d/%d papers enriched", enriched, len(papers))
    else:
        logger.info("  skipping enrichment")

    skill_dir = build_skill(
        entries, output_dir, name, config.get("description", ""), config["repo"]
    )
    logger.info("=== Done: %s -> %s ===", name, skill_dir)
    return skill_dir


def _cmd_collect(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    entries = collect(config)
    save_entries(entries, config)
    print(f"Collected {len(entries)} entries")


def _cmd_enrich(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    entries = load_entries(config)
    todo = [
        e for e in entries
        if e.entry_type == "paper" and not e.enrichments.get("s2_enriched")
    ]
    if not todo:
        print("All entries already enriched")
        return
    print(f"Enriching {len(todo)} entries...")
    entries = enrich_papers(entries)
    save_entries(entries, config)
    print(f"Enriched {sum(1 for e in entries if e.enrichments.get('s2_enriched'))}")


def _cmd_build(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    entries = load_entries(config)
    out_dir = Path(args.output) if args.output else OUTPUT_DIR
    path = build_skill(
        entries, out_dir, config["name"],
        config.get("description", ""), config["repo"],
    )
    print(f"Built skill at {path}")


def _cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output) if args.output else OUTPUT_DIR
    run(args.config, out_dir, skip_enrich=args.skip_enrich)


def _cmd_run_all(args: argparse.Namespace) -> None:
    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    if not configs:
        print(f"No configs in {CONFIGS_DIR}")
        return
    print(f"Found {len(configs)} configs")
    for path in configs:
        try:
            out_dir = Path(args.output) if args.output else OUTPUT_DIR
            run(path, out_dir, skip_enrich=args.skip_enrich)
        except Exception as e:
            logger.exception("Failed %s", path.name)
            print(f"  ERROR: {e}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Awesome-list ingestion pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    p_col = sub.add_parser("collect")
    p_col.add_argument("config", type=Path)

    p_enr = sub.add_parser("enrich")
    p_enr.add_argument("config", type=Path)

    p_bld = sub.add_parser("build")
    p_bld.add_argument("config", type=Path)
    p_bld.add_argument("--output", "-o")

    p_run = sub.add_parser("run")
    p_run.add_argument("config", type=Path)
    p_run.add_argument("--output", "-o")
    p_run.add_argument("--skip-enrich", action="store_true")

    p_all = sub.add_parser("run-all")
    p_all.add_argument("--output", "-o")
    p_all.add_argument("--skip-enrich", action="store_true")

    args = p.parse_args()
    handlers = {
        "collect": _cmd_collect,
        "enrich": _cmd_enrich,
        "build": _cmd_build,
        "run": _cmd_run,
        "run-all": _cmd_run_all,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
