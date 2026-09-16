"""github — GitHub-repo sourcing pipeline.

Single-file pipeline: GitHub search -> classify repos into PwC tasks -> emit
SKILL.md. Each repo is narrowed to its top-K candidate PwC tasks by embedding,
then a single Haiku call picks 1-3 verbatim; the orthogonal entry_type/install/
tags/tldr axis is kept. All pipeline logic, schemas, paths, and CLI live in this
module.

LLM calls go through the shared backend call (`sourcing.utils.agent_calls`): the batch
PwC classify via `AgentCalls.run_batch` (inside `sourcing.utils.classify`), the
per-category describe via `AgentConversation.ask` — so every call writes its transcript under the
run's work dir, and the per-stage caches live under the shared work dir
(`<runtime-root>/sourcing/github/<run-id>/`) instead of an in-repo `.cache/`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Literal

import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field

from sourcing.utils import classify
from sourcing.utils.agent_calls import AgentCalls, CallRecord
from sourcing.utils.skill_output import RepoEntry, dump_frontmatter
from sourcing.utils.work_dir import (
    resolve_run_id,
    resolve_work_dir,
    step_dir,
)

log = logging.getLogger("github")

# Filesystem layout.

PACKAGE_DIR = Path(__file__).parent
_PLUGIN_DIR = (
    PACKAGE_DIR.parent.parent
    / "data" / "corpus" / "plugins" / "aibuildai-modeling-marketplace"
    / "aibuildai-github" / "skills"
)


def cache_path(run_id: str, stage: str) -> Path:
    """Return `<work-dir>/cache/<stage>.json` for this run, creating the dir.

    The per-stage JSON filename (`repos.json` / `classified.json`) and the per-run
    isolation are preserved; the cache now lives under the shared work dir
    (`<runtime-root>/sourcing/github/<run-id>/cache/`) instead of the in-repo `.cache/`.
    """
    d = resolve_work_dir("github", run_id=run_id) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stage}.json"


def output_dir() -> Path:
    """Return the plugin `aibuildai-github/skills/` directory that stage 4 writes into."""
    return _PLUGIN_DIR


# Schemas.

EntryType = Literal[
    "training framework",
    "dataset SDK",
    "benchmark",
    "utility library",
    "eval toolkit",
    "model zoo",
    "pretrained model",
    "cli tool",
    "notebook collection",
    "other",
]


class Repo(BaseModel):
    """Stage 1 output: one GitHub repo record."""
    model_config = ConfigDict(extra="forbid")

    full_name: str
    url: str
    stars: int
    language: str | None = None
    license: str | None = None
    description: str = ""
    topics: list[str] = Field(default_factory=list)
    default_branch: str = "main"
    search_query: str
    readme: str | None = None  # filled by stage 2

    @property
    def slug(self) -> str:
        return self.full_name.replace("/", "-").lower()


class RepoClassification(BaseModel):
    """Stage 3 single-repo LLM output."""
    model_config = ConfigDict(extra="forbid")

    categories: list[str] = Field(min_length=1, max_length=3)
    entry_type: EntryType
    install: str = Field(description="shell command to install, e.g. 'pip install chemprop'")
    tags: list[str] = Field(default_factory=list, max_length=6)
    tldr: str


class RepoClassificationItem(BaseModel):
    """One verdict inside a BatchRepoClassificationVerdicts response."""
    model_config = ConfigDict(extra="forbid")

    repo_full_name: str
    categories: list[str] = Field(min_length=1, max_length=3)
    entry_type: EntryType
    install: str
    tags: list[str] = Field(default_factory=list, max_length=6)
    tldr: str


class BatchRepoClassificationVerdicts(BaseModel):
    """Stage 3 batch output container."""
    model_config = ConfigDict(extra="forbid")

    verdicts: list[RepoClassificationItem]


class CategoryDescription(BaseModel):
    """Stage 4 LLM output: 1-2 sentence description of a category."""
    model_config = ConfigDict(extra="forbid")

    description: str


# Stage 1: search GitHub for repo metadata.

SEARCH_URL = "https://api.github.com/search/repositories"


def search_repos(query: str, max_repos: int, token: str | None) -> list[Repo]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    out: list[Repo] = []
    per_page = min(100, max_repos)
    page = 1
    while len(out) < max_repos:
        r = requests.get(
            SEARCH_URL,
            params={"q": query, "sort": "stars", "order": "desc", "per_page": per_page, "page": page},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        items = r.json().get("items") or []
        if not items:
            break
        for it in items:
            lic = (it.get("license") or {}).get("spdx_id")
            out.append(Repo(
                full_name=it["full_name"],
                url=it["html_url"],
                stars=it.get("stargazers_count", 0),
                language=it.get("language"),
                license=lic,
                description=it.get("description") or "",
                topics=it.get("topics") or [],
                default_branch=it.get("default_branch") or "main",
                search_query=query,
            ))
            if len(out) >= max_repos:
                break
        page += 1
        if len(items) < per_page:
            break
    return out


def search(run_id: str, search_queries_path: Path, force: bool = False) -> list[Repo]:
    cache = cache_path(run_id, "repos")
    if cache.exists() and not force:
        log.info("stage1 cache hit %s", cache)
        return [Repo.model_validate(o) for o in json.loads(cache.read_text())]

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.warning("GITHUB_TOKEN not set; subject to 60 req/hr limit")

    search_queries = yaml.safe_load(search_queries_path.read_text())["search_queries"]
    seen: dict[str, Repo] = {}
    for s in search_queries:
        log.info("stage1 search query=%r max=%d", s["query"], s["max_repos"])
        for r in search_repos(s["query"], s["max_repos"], token):
            seen.setdefault(r.full_name, r)
    repos = list(seen.values())
    cache.write_text(json.dumps([r.model_dump() for r in repos], indent=2))
    log.info("stage1 wrote %d unique repos -> %s", len(repos), cache)
    return repos


# Stage 2: fetch README (enriches each repo's text for PwC classification).


def fetch_readme(repo: Repo, token: str | None) -> str:
    r = requests.get(
        (
            f"https://raw.githubusercontent.com/{repo.full_name}/"
            f"{repo.default_branch}/README.md"
        ),
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=30,
    )
    if r.status_code != 200:
        return ""
    return r.text[:6000]


# Stage 3: classify repos into their own PwC task candidates.

_ENTRY_TYPES = (
    "training framework, dataset SDK, benchmark, utility library, "
    "eval toolkit, model zoo, pretrained model, cli tool, "
    "notebook collection, other"
)


def _build_batch_prompt(batch: list[dict]) -> str:
    blocks = []
    for r in batch:
        cands = "\n".join(f"  - {c}" for c in r["candidates"])
        blocks.append(
            f"[repo_full_name={r['repo_full_name']}]\n"
            f"description: {r['description']}\n"
            f"language: {r['language']}  license: {r['license']}\n"
            f"topics: {', '.join(r['topics'])}\n"
            f"readme_head:\n{(r.get('readme') or '')[:1500]}\n"
            f"Candidate tasks (choose 1-3 verbatim from THESE):\n{cands}"
        )
    return (
        "For each GitHub repository, assign 1-3 categories chosen VERBATIM from "
        "ITS OWN candidate task list, choose the most specific entry_type, extract "
        "the install command (single shell line), produce up to 6 tags, and a "
        "one-sentence tldr.\n\n"
        f"Allowed entry_type values: {_ENTRY_TYPES}.\n\n"
        "Echo repo_full_name exactly. One verdict per repository.\n\n"
        + "\n\n".join(blocks)
    )


async def _run_batch_async(
    calls: AgentCalls,
    items: list[dict],
    *,
    model: str,
    batch_size: int,
    workers: int,
    calls_dir: Path,
) -> list[dict]:
    return await calls.run_batch(
        items=items,
        make_prompt=_build_batch_prompt,
        key_of=lambda x: x["repo_full_name"],
        output_schema=BatchRepoClassificationVerdicts,
        system_prompt="",
        model=model,
        calls_dir=calls_dir,
        batch_size=batch_size,
        workers=workers,
        verify_verdict=classify.verify_verbatim_picks("categories"),
    )


def classify_repos(
    calls: AgentCalls,
    repos: list[dict],
    *,
    model: str,
    top_n: int,
    k: int,
    batch_size: int,
    workers: int,
    calls_dir: Path,
) -> list[dict]:
    """Classify each repo into 1-3 PwC tasks from its own embedding candidates;
    keep the orthogonal entry_type/install/tags/tldr axis."""
    labels, _area = classify.load_label_set(top_n)
    texts = [f"{r.get('description','')}\n{' '.join(r.get('topics', []))}\n{(r.get('readme') or '')[:1500]}"
             for r in repos]
    cand = classify.candidate_lists(texts, labels, k)
    items = [
        {"repo_full_name": r["full_name"], "description": r.get("description", ""),
         "language": r.get("language") or "", "license": r.get("license") or "",
         "topics": r.get("topics", []), "readme": r.get("readme") or "", "candidates": c}
        for r, c in zip(repos, cand, strict=True)
    ]
    verdicts_raw = asyncio.run(
        _run_batch_async(
            calls,
            items,
            model=model,
            batch_size=batch_size,
            workers=workers,
            calls_dir=calls_dir,
        )
    )
    by_name = {v["repo_full_name"]: v for v in verdicts_raw}
    enriched: list[dict] = []
    for r in repos:
        v = by_name[r["full_name"]]
        enriched.append({**r, "categories": v["categories"], "entry_type": v["entry_type"],
                         "install": v["install"], "tags": v.get("tags", []), "tldr": v.get("tldr", "")})
    return enriched


def classify_stage(
    calls: AgentCalls,
    run_id: str,
    *,
    model: str,
    top_n: int,
    k: int,
    batch_size: int,
    workers: int,
    force: bool,
) -> list[dict]:
    classified_path = cache_path(run_id, "classified")
    if classified_path.exists() and not force:
        log.info("stage3 cache hit %s", classified_path)
        return json.loads(classified_path.read_text(encoding="utf-8"))
    repos_raw = json.loads(cache_path(run_id, "repos").read_text(encoding="utf-8"))
    # fill readmes so they enrich each repo's classification text
    token = os.environ.get("GITHUB_TOKEN")
    for r in repos_raw:
        r["readme"] = fetch_readme(Repo.model_validate(r), token)
    calls_dir = resolve_work_dir("github", run_id=run_id) / "classify" / "calls"
    enriched = classify_repos(
        calls,
        repos_raw,
        model=model,
        top_n=top_n,
        k=k,
        batch_size=batch_size,
        workers=workers,
        calls_dir=calls_dir,
    )
    classified_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    log.info("stage3 wrote %d classified repos -> %s", len(enriched), classified_path)
    return enriched


# Stage 4: emit SKILL.md + references/<repo>.md.


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9-]", "", text.lower().replace("/", "-").replace(" ", "-"))
    return s or "uncategorized"


def _describe_category(
    calls: AgentCalls,
    run_id: str,
    category: str,
    repos: list[dict],
    *,
    model: str,
) -> str:
    names = "\n".join(
        f"- {r['full_name']} — {r.get('tldr','')}" for r in repos[:15]
    )
    prompt = (
        f"Given these GitHub repositories grouped under '{category}', write a "
        "1-2 sentence description of what this collection provides. Be concrete "
        "about methods, tasks, and applications. No fluff:\n"
        f"{names}"
    )
    wd = resolve_work_dir("github", run_id=run_id)
    call_dir = step_dir(wd, "category_describe", _slug(category))
    async def invoke() -> CallRecord:
        async with calls.conversation(
            name="category-describe",
            model=model,
            cwd=str(call_dir.resolve()),
            output_schema=CategoryDescription.model_json_schema(),
            max_turns=1,
        ) as session:
            rec = await session.ask(
                user_prompt=prompt,
                step_dir=call_dir,
            )
        return rec
    rec = asyncio.run(invoke())
    return CategoryDescription.model_validate(rec.result).description.strip()


def _skill_md(category: str, repos: list[dict], description: str) -> str:
    rows = "\n".join(
        f"| {i+1} | {r['full_name']} | {r.get('entry_type','')} | "
        f"{r.get('language','')} | {', '.join((r.get('tags') or [])[:3])} | "
        f"references/{_slug(r['full_name'])}.md |"
        for i, r in enumerate(repos)
    )
    return (
        f"---\n"
        f"description: >-\n"
        f"  {description}\n"
        f"---\n\n"
        f"# {category.replace('-', ' ').title()} — GitHub Repositories\n\n"
        f"{description}\n\n"
        f"## Entry Index\n\n"
        f"| # | Name | Kind | Language | Tags | File |\n"
        f"|---|------|------|----------|------|------|\n"
        f"{rows}\n"
    )


def _reference_md(repo: dict) -> str:
    entry = RepoEntry(
        entry_type="repo",
        title=repo["full_name"],
        source=repo.get("url", ""),
        tags=repo.get("tags", []),
        tldr=repo.get("tldr", ""),
        categories=repo.get("categories", []),
        language=repo.get("language"),
        license=repo.get("license"),
        install=repo.get("install", ""),
    )
    fm = dump_frontmatter(entry)
    body = (
        f"# {repo['full_name']}\n\n"
        f"**Source**: [{repo['full_name']}]({repo.get('url','')})\n\n"
        f"## Description\n\n{repo.get('description','')}\n\n"
        f"## TLDR\n\n{repo.get('tldr','')}\n\n"
    )
    if entry.install:
        body += f"## Installation\n\n```bash\n{entry.install}\n```\n"
    return f"---\n{fm}---\n\n{body}"


def generate(calls: AgentCalls, run_id: str, *, model: str) -> None:
    classified_path = cache_path(run_id, "classified")
    rows = json.loads(classified_path.read_text())
    out_root = output_dir()
    out_root.mkdir(parents=True, exist_ok=True)

    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        for cat in r.get("categories") or ["other"]:
            by_cat.setdefault(cat, []).append(r)

    for cat, members in by_cat.items():
        cat_dir = out_root / _slug(cat)
        ref_dir = cat_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        desc = _describe_category(calls, run_id, cat, members, model=model)
        (cat_dir / "SKILL.md").write_text(_skill_md(cat, members, desc), encoding="utf-8")
        for r in members:
            (ref_dir / f"{_slug(r['full_name'])}.md").write_text(_reference_md(r), encoding="utf-8")
    log.info("stage4 emitted %d categories -> %s", len(by_cat), out_root)


# Orchestration + CLI.


def run_one(
    calls: AgentCalls,
    run_id: str,
    search_queries: Path,
    *,
    model: str,
    top_n: int,
    k: int,
    batch_size: int,
    workers: int,
    force: bool,
) -> None:
    if force:
        for stage in ("repos", "classified"):
            cache_path(run_id, stage).unlink(missing_ok=True)
    search(run_id, search_queries, force=force)
    classify_stage(
        calls,
        run_id,
        model=model,
        top_n=top_n,
        k=k,
        batch_size=batch_size,
        workers=workers,
        force=force,
    )
    generate(calls, run_id, model=model)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m sourcing.github",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--search-queries", type=Path,
                        default=PACKAGE_DIR / "configs" / "search_queries.yaml")
    parser.add_argument("--workers", type=int, default=4, help="Parallel LLM calls")
    parser.add_argument("--batch-size", type=int, default=50, help="Repos per LLM call")
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model used for classification and category text",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=300,
        help="Number of taxonomy tasks to load",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=50,
        help="Candidate tasks given to the model",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_id = resolve_run_id(args.run_id)
    calls = AgentCalls()
    run_one(
        calls,
        run_id,
        args.search_queries,
        model=args.model,
        top_n=args.top_n,
        k=args.candidate_count,
        batch_size=args.batch_size,
        workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
