"""Build the playbook-skill corpus from the faridrashidi index + local Meta Kaggle.

Data flow:
  index (faridrashidi competitions.yml)
    -> resolve text (local Meta Kaggle: discussion topic_id join / writeup title match)
    -> classify each competition into the shared PwC task taxonomy (sourcing.utils.classify)
    -> group competitions by PwC task (thin tasks fold to their PwC primary area)
    -> per-task workspace
    -> summarize one playbook skill per PwC-task group (writer LLM, fixed creator prompt)
    -> atomic replace into the live corpus.

This module owns "how the source becomes corpus": parsing, local text resolution,
PwC classification + grouping, workspace building, the writer call (the shared
`AgentCalls` driver), and run(). What ONE skill IS lives in playbook.py.

A skill key is the PwC task (the shared topic vocabulary every pipeline classifies
into); per-pipeline LAYOUT (one skill per topic group, the writer's principle index)
is unchanged. There is no per-pipeline topic table any more.

Entry: `python -m sourcing.playbooks` -> __main__ -> run().
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from sourcing.utils import classify, skill_output
from sourcing.utils.agent_calls import AgentCalls, sum_costs
from sourcing.utils.work_dir import resolve_work_dir, setup_logging, step_dir

from .playbook import (
    PlaybookOutput,
    WriteupRecord,
    build_writer_prompt,
    strip_images,
    write_skill_directory,
)

log = logging.getLogger("playbooks")


# ---------- Paths / config ----------

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]

INDEX_NAME = "kaggle-solutions.competitions.yml"
EXCLUDED_COMPETITIONS = PIPELINE_DIR / "excluded_competitions.txt"
COMPETITIONS_CSV = "Competitions.csv"
FORUM_TOPICS_CSV = "ForumTopics.csv"
FORUM_MESSAGES_CSV = "ForumMessages.csv"

# Live corpus the run replaces (atomic replace target). UNCHANGED: the corpus output
# stays under data/corpus/, NOT under the analysis/ work dir.
SKILLS_DIR = (
    REPO_ROOT
    / "data"
    / "corpus"
    / "plugins"
    / "aibuildai-modeling-marketplace"
    / "aibuildai-playbooks"
    / "skills"
)

# Regenerable build scratch (workspace/steps/logs) lives under the shared per-run
# work dir, resolved inside run() -- NOT a module constant (avoids an import-time mkdir).

# Title-match tie-break (spec section 2).
JACCARD_FLOOR = 0.6
JACCARD_MARGIN = 0.2


class PlaybookGenerationConfig(BaseModel):
    """Reviewed source selection plus the writer route and output locations."""

    model_config = ConfigDict(extra="forbid")

    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    selection_csv: Path
    workspace_dir: Path
    output_dir: Path
    work_dir: Path
    groups: list[str]


# ---------- Index parsing ----------


@dataclass
class Solution:
    rank: int
    link: str
    kind: str


@dataclass
class Competition:
    slug: str
    title: str
    metric: str
    team: str
    kind: str
    solutions: list[Solution]


def load_excluded_competitions() -> frozenset[str]:
    """The competitions of the benchmarks we run; no playbook reads their writeups."""
    lines = EXCLUDED_COMPETITIONS.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip() and not line.startswith("#"))


def load_index(yml_path: Path) -> dict[str, Competition]:
    """Parse the faridrashidi index: each competition -> rank-sorted solution links.

    A competition in excluded_competitions.txt is dropped here, before any text is
    resolved, so no later phase can place it in a skill."""
    excluded = load_excluded_competitions()
    data = yaml.safe_load(yml_path.read_text())
    out: dict[str, Competition] = {}
    for comp in data.get("competitions", []):
        match = re.search(r"/c/([^/?#]+)", comp.get("link") or "")
        slug = match.group(1) if match else ""
        if not slug or slug in excluded:
            continue
        solutions: list[Solution] = []
        for s in comp.get("solutions") or []:
            rank_raw = s.get("rank")
            slink = s.get("link") or ""
            if rank_raw is None or not slink:
                continue
            try:
                rank = int(str(rank_raw).strip())
            except ValueError:
                continue
            solutions.append(Solution(rank=rank, link=slink, kind=s.get("kind") or ""))
        if not solutions:
            continue
        solutions.sort(key=lambda x: x.rank)
        out[slug] = Competition(
            slug=slug,
            title=comp.get("title") or slug,
            metric=comp.get("metric") or "",
            team=comp.get("team") or "",
            kind=comp.get("kind") or "",
            solutions=solutions,
        )
    return out


# ---------- Meta Kaggle loading ----------


@dataclass
class CompMeta:
    slug: str
    forum_id: int | None
    title: str
    overview: str | None
    data_description: str | None
    eval_name: str | None
    eval_is_max: bool | None
    total_teams: int | None


@dataclass
class Topic:
    id: int
    title: str
    first_message_id: int | None


def load_competitions_meta(meta_dir: Path) -> dict[str, CompMeta]:
    df = pd.read_csv(
        meta_dir / COMPETITIONS_CSV,
        usecols=[
            "Slug",
            "Title",
            "ForumId",
            "Overview",
            "DatasetDescription",
            "EvaluationAlgorithmName",
            "EvaluationAlgorithmIsMax",
            "TotalTeams",
        ],
    )
    out: dict[str, CompMeta] = {}
    for row in df.itertuples(index=False):
        out[row.Slug] = CompMeta(
            slug=row.Slug,
            forum_id=int(row.ForumId) if pd.notna(row.ForumId) else None,
            title=row.Title if isinstance(row.Title, str) else row.Slug,
            overview=row.Overview if isinstance(row.Overview, str) else None,
            data_description=row.DatasetDescription
            if isinstance(row.DatasetDescription, str)
            else None,
            eval_name=row.EvaluationAlgorithmName
            if isinstance(row.EvaluationAlgorithmName, str)
            else None,
            eval_is_max=bool(row.EvaluationAlgorithmIsMax)
            if pd.notna(row.EvaluationAlgorithmIsMax)
            else None,
            total_teams=int(row.TotalTeams) if pd.notna(row.TotalTeams) else None,
        )
    return out


def load_forum_topics(meta_dir: Path) -> dict[int, list[Topic]]:
    df = pd.read_csv(
        meta_dir / FORUM_TOPICS_CSV,
        usecols=["Id", "ForumId", "Title", "FirstForumMessageId"],
    )
    by_forum: dict[int, list[Topic]] = defaultdict(list)
    for row in df.itertuples(index=False):
        if pd.isna(row.ForumId):
            continue
        by_forum[int(row.ForumId)].append(
            Topic(
                id=int(row.Id),
                title=row.Title if isinstance(row.Title, str) else "",
                first_message_id=int(row.FirstForumMessageId)
                if pd.notna(row.FirstForumMessageId)
                else None,
            )
        )
    return dict(by_forum)


def fetch_messages(meta_dir: Path, message_ids: set[int]) -> dict[int, str]:
    """One scan of the 1.6GB ForumMessages.csv: message_id -> RawMarkdown (or Message)."""
    if not message_ids:
        return {}
    lookup: dict[int, str] = {}
    for chunk in pd.read_csv(
        meta_dir / FORUM_MESSAGES_CSV,
        chunksize=100000,
        usecols=["Id", "RawMarkdown", "Message"],
    ):
        hit = chunk[chunk["Id"].isin(message_ids)]
        for _, row in hit.iterrows():
            mid = int(row["Id"])
            raw = str(row["RawMarkdown"]) if pd.notna(row["RawMarkdown"]) else ""
            html = str(row["Message"]) if pd.notna(row["Message"]) else ""
            lookup[mid] = raw if raw.strip() else html
        if len(lookup) >= len(message_ids):
            break
    return lookup


# ---------- Text resolution ----------


@dataclass
class PendingMatch:
    rank: int
    topic_id: int
    topic_title: str
    link: str
    kind: str
    message_id: int


_DISCUSSION_RE = re.compile(r"/c/[^/]+/discussion/(\d+)(?:#(\d+))?")
_WRITEUP_RE = re.compile(r"/c/[^/]+/writeups/([^/?#]+)")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _tokens(slug: str) -> set[str]:
    return {t for t in slug.split("-") if t}


def token_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_writeup(writeup_slug: str, topics: list[Topic]) -> Topic | None:
    """Title-match a /writeups/<slug> link to a forum topic in the same competition.

    Accept (a) exact slugify(title) == writeup_slug, or (b) a unique winner with
    token-Jaccard >= 0.6 AND (best - second) >= 0.2 among the >= 0.6 candidates
    (a single >= 0.6 candidate has second = 0, so its margin = best -> accepted by
    design). Otherwise drop -- never guess.
    """
    target = slugify(writeup_slug)
    for t in topics:
        if t.first_message_id is not None and slugify(t.title) == target:
            return t

    target_tokens = _tokens(target)
    scored = [
        (token_jaccard(target_tokens, _tokens(slugify(t.title))), t)
        for t in topics
        if t.first_message_id is not None
    ]
    candidates = sorted(
        [(score, t) for score, t in scored if score >= JACCARD_FLOOR],
        key=lambda x: -x[0],
    )
    if not candidates:
        return None
    best = candidates[0][0]
    second = candidates[1][0] if len(candidates) > 1 else 0.0
    if best - second >= JACCARD_MARGIN:
        return candidates[0][1]
    return None


def match_solutions(
    solutions: list[Solution], topics: list[Topic]
) -> list[PendingMatch]:
    """Map each solution link to (topic, message_id) using local Meta Kaggle topics.

    Discussion links: /discussion/<topic_id>(#<comment_id>) -> the topic's
    FirstForumMessageId (or the comment_id when anchored). Writeup links:
    /writeups/<slug> -> title match. Other link kinds (github/kernel/external) drop.
    """
    by_id = {t.id: t for t in topics}
    matches: list[PendingMatch] = []
    for sol in solutions:
        disc = _DISCUSSION_RE.search(sol.link)
        if disc:
            topic_id = int(disc.group(1))
            comment_id = int(disc.group(2)) if disc.group(2) else None
            topic = by_id.get(topic_id)
            if topic is None:
                continue
            message_id = (
                comment_id if comment_id is not None else topic.first_message_id
            )
            if message_id is None:
                continue
            matches.append(
                PendingMatch(
                    rank=sol.rank,
                    topic_id=topic_id,
                    topic_title=topic.title,
                    link=sol.link,
                    kind=sol.kind,
                    message_id=message_id,
                )
            )
            continue
        wu = _WRITEUP_RE.search(sol.link)
        if wu:
            topic = match_writeup(wu.group(1), topics)
            if topic is None or topic.first_message_id is None:
                continue
            matches.append(
                PendingMatch(
                    rank=sol.rank,
                    topic_id=topic.id,
                    topic_title=topic.title,
                    link=sol.link,
                    kind=sol.kind,
                    message_id=topic.first_message_id,
                )
            )
            continue
        # github / kernel / external -> drop (deferred online fallback).
    return matches


def attach_text(
    comp_slug: str, matches: list[PendingMatch], messages: dict[int, str]
) -> list[WriteupRecord]:
    """Attach resolved text to matches, dropping any with empty/missing text."""
    records: list[WriteupRecord] = []
    for m in matches:
        text = messages.get(m.message_id, "")
        if not text or not text.strip():
            continue
        records.append(
            WriteupRecord(
                comp_slug=comp_slug,
                rank=m.rank,
                topic_id=m.topic_id,
                topic_title=m.topic_title,
                link=m.link,
                kind=m.kind,
                text=text,
            )
        )
    records.sort(key=lambda r: r.rank)
    return records


def resolve_text(
    comp_slug: str,
    solutions: list[Solution],
    topics: list[Topic],
    messages: dict[int, str],
) -> list[WriteupRecord]:
    """Resolve a competition's solution links to ranked writeup records (all local)."""
    return attach_text(comp_slug, match_solutions(solutions, topics), messages)


# ---------- PwC classification + grouping ----------


def _competition_text(slug: str, meta: CompMeta | None) -> str:
    """The text the classifier reads to place a competition into a PwC task: its
    title plus the overview / data description when Meta Kaggle carries them."""
    if meta is None:
        return slug.replace("-", " ")
    parts = [meta.title]
    if meta.overview:
        parts.append(strip_images(meta.overview))
    if meta.data_description:
        parts.append(strip_images(meta.data_description))
    return "\n".join(parts)


def classify_competitions(
    calls: AgentCalls,
    comps_with_text: dict[str, list[WriteupRecord]],
    comp_meta: dict[str, CompMeta],
    work_dir: Path,
    *,
    model: str,
    top_n: int,
    k: int,
    batch_size: int,
    workers: int,
) -> list[dict]:
    """Classify each competition that has resolved writeups into 1-3 PwC tasks via
    the shared classify mechanism (embed -> top-K candidates -> one Haiku pick),
    exactly like papers/github. Cached to ``<work_dir>/classified.json`` so a resume
    reuses the verdicts."""
    cache = work_dir / "classified.json"
    if cache.exists():
        log.info("classify cache hit %s", cache)
        return json.loads(cache.read_text(encoding="utf-8"))
    items = [
        {"slug": slug, "text": _competition_text(slug, comp_meta.get(slug))}
        for slug in comps_with_text
    ]
    classified = classify.classify_items(
        items,
        calls=calls,
        text_of=lambda it: it["text"],
        id_of=lambda it: it["slug"],
        model=model,
        top_n=top_n,
        k=k,
        batch_size=batch_size,
        workers=workers,
        calls_dir=work_dir / "classify" / "calls",
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(classified, indent=2), encoding="utf-8")
    log.info("wrote %d classified competitions -> %s", len(classified), cache)
    return classified


def group_into_skills(
    classified: list[dict],
    comps_with_text: dict[str, list[WriteupRecord]],
    *,
    min_items: int,
    top_n: int,
) -> dict[str, dict[str, list[WriteupRecord]]]:
    """Group classified competitions into one skill per PwC task; a task with fewer
    than ``min_items`` competitions folds into its PwC primary area (group_by_task).
    The skill key is the shared-helper slug of the task (or area) name. Returns
    {skill_key: {comp_slug: writeups}}."""
    _names, area_of = classify.load_label_set(top_n)
    by_task = classify.group_by_task(
        classified,
        skill_key=skill_output.slugify,
        area_of=area_of,
        min_items=min_items,
    )
    return {
        skill_key: {m["slug"]: comps_with_text[m["slug"]] for m in members}
        for skill_key, members in by_task.items()
    }


# ---------- Workspace building ----------


def _format_competition_overview(slug: str, meta: CompMeta | None) -> str:
    if meta is None:
        return f"# {slug}\n"
    parts = [f"# {meta.title}"]
    if meta.eval_name:
        parts.append(f"Evaluation metric: {meta.eval_name}")
    if meta.eval_is_max is not None:
        parts.append(f"Higher is better: {meta.eval_is_max}")
    if meta.total_teams:
        parts.append(f"Total teams: {meta.total_teams}")
    if meta.overview:
        parts.append(f"\n## Task Overview\n{strip_images(meta.overview)}")
    if meta.data_description:
        parts.append(f"\n## Data Description\n{strip_images(meta.data_description)}")
    return "\n".join(parts)


def build_workspace(
    skill_key: str,
    comps_with_text: dict[str, list[WriteupRecord]],
    comp_meta: dict[str, CompMeta],
    workspace_root: Path,
) -> Path:
    """Write the per-task workspace the writer reads (overview + per-comp writeups)."""
    display = skill_key.replace("-", " ")
    task_dir = workspace_root / skill_key
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    slugs = list(comps_with_text.keys())
    (task_dir / "overview.md").write_text(
        f"# {display}\n\n{len(slugs)} competitions: {', '.join(slugs)}\n"
    )
    for slug, records in comps_with_text.items():
        comp_dir = task_dir / "competitions" / slug
        comp_dir.mkdir(parents=True, exist_ok=True)
        (comp_dir / "overview.md").write_text(
            _format_competition_overview(slug, comp_meta.get(slug))
        )
        index_lines = []
        for rec in sorted(records, key=lambda r: r.rank):
            text = strip_images(rec.text)
            title_slug = slugify(rec.topic_title)[:50].strip("-")
            filename = f"writeup_{title_slug}_{rec.topic_id}.md"
            (comp_dir / filename).write_text(
                f"# {rec.topic_title}\n\nRank: #{rec.rank}\n\n{text}"
            )
            index_lines.append(
                f"- {filename} | {rec.topic_title} | rank #{rec.rank} | {len(text)} chars"
            )
        (comp_dir / "index.md").write_text(
            f"# {slug} writeups ({len(records)} total)\n\n"
            + "\n".join(index_lines)
            + "\n"
        )
    return task_dir


# ---------- Summarization ----------


async def summarize_one(
    calls: AgentCalls,
    skill_key: str,
    comps_with_text: dict[str, list[WriteupRecord]],
    comp_meta: dict[str, CompMeta],
    workspace_root: Path,
    skill_out_root: Path,
    model: str,
    work_dir: Path,
) -> float | None:
    """Build the workspace, run the writer, and write the skill directory for one
    PwC-task group. Returns the settled cost (USD) of the writer call."""
    task_dir = build_workspace(skill_key, comps_with_text, comp_meta, workspace_root)
    return await summarize_workspace_one(
        calls,
        skill_key,
        comps_with_text,
        task_dir,
        skill_out_root,
        model,
        work_dir,
        effort=None,
    )


async def summarize_workspace_one(
    calls: AgentCalls,
    skill_key: str,
    comps_with_text: dict[str, list[WriteupRecord]],
    task_dir: Path,
    skill_out_root: Path,
    model: str,
    work_dir: Path,
    *, effort: str | None,
) -> float | None:
    """Run the writer against an already-reviewed per-task workspace."""
    item = {
        "skill_key": skill_key,
        "display_name": skill_key.replace("-", " "),
        "train_slugs": list(comps_with_text.keys()),
        "cwd": task_dir,
    }
    user_prompt, system_prompt, _label = build_writer_prompt(item)
    call_dir = step_dir(work_dir, skill_key)
    async with calls.conversation(
        name=f"writer-{skill_key}",
        instructions=system_prompt,
        model=model,
        tools=("Read", "Glob", "Grep"),
        cwd=str(task_dir),
        output_schema=PlaybookOutput.model_json_schema(),
        max_turns=30,
        effort=effort,
    ) as session:
        rec = await session.ask(
            user_prompt=user_prompt,
            step_dir=call_dir,
        )
    structured = rec.result
    cost = rec.cost_usd
    if not structured:
        raise AssertionError(
            f"[{skill_key}] writer produced no structured PlaybookOutput"
        )
    assert isinstance(structured, dict)
    skill_dir = skill_out_root / skill_key
    if skill_dir.exists():
        shutil.rmtree(skill_dir)
    write_skill_directory(structured, skill_dir, skill_key, comps_with_text)
    return cost


def load_generation_config(path: Path) -> PlaybookGenerationConfig:
    config = PlaybookGenerationConfig.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    for field in ("selection_csv", "workspace_dir", "output_dir", "work_dir"):
        value = getattr(config, field)
        if not value.is_absolute():
            raise AssertionError(f"generation config {field} must be absolute: {value}")
    if not config.groups:
        raise AssertionError("generation config groups must not be empty")
    if len(config.groups) != len(set(config.groups)):
        raise AssertionError("generation config groups must be unique")
    return config


def load_reviewed_selection(
    selection_csv: Path,
    workspace_dir: Path,
) -> dict[str, dict[str, list[WriteupRecord]]]:
    """Recover writer records from the reviewed CSV and its prepared workspace.

    Rows of a competition in excluded_competitions.txt are dropped, as in load_index."""
    excluded = load_excluded_competitions()
    by_group: dict[str, dict[str, list[WriteupRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with selection_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"group", "competition", "rank", "topic_id", "title", "source"}
    if not rows:
        raise AssertionError(f"reviewed selection is empty: {selection_csv}")
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise AssertionError(
            f"reviewed selection missing columns: {sorted(missing_columns)}"
        )

    for row in rows:
        group = row["group"]
        competition = row["competition"]
        if competition in excluded:
            continue
        rank = int(row["rank"])
        topic_id = int(row["topic_id"])
        comp_dir = workspace_dir / group / "competitions" / competition
        matches = list(comp_dir.glob(f"writeup_*_{topic_id}.md"))
        if len(matches) != 1:
            raise AssertionError(
                f"expected one workspace writeup for {group}/{competition}/{topic_id}, "
                f"found {len(matches)}"
            )
        prefix = f"# {row['title']}\n\nRank: #{rank}\n\n"
        workspace_text = matches[0].read_text(encoding="utf-8")
        if not workspace_text.startswith(prefix):
            raise AssertionError(
                f"workspace metadata differs from selection CSV: {matches[0]}"
            )
        by_group[group][competition].append(
            WriteupRecord(
                comp_slug=competition,
                rank=rank,
                topic_id=topic_id,
                topic_title=row["title"],
                link=row["source"],
                kind="",
                text=workspace_text[len(prefix) :],
            )
        )

    return {
        group: {competition: records for competition, records in competitions.items()}
        for group, competitions in by_group.items()
    }


async def run_reviewed_selection(
    calls: AgentCalls,
    config_path: Path,
    only_group: str | None = None,
) -> None:
    """Generate only the reviewed selection without replacing the live corpus."""
    config = load_generation_config(config_path)
    setup_logging(config.work_dir)
    selected = load_reviewed_selection(config.selection_csv, config.workspace_dir)
    selected_groups = set(selected)
    configured_groups = set(config.groups)
    if selected_groups != configured_groups:
        raise AssertionError(
            "generation config groups differ from selection CSV: "
            f"missing={sorted(selected_groups - configured_groups)}, "
            f"extra={sorted(configured_groups - selected_groups)}"
        )
    groups = [only_group] if only_group else config.groups
    if only_group and only_group not in configured_groups:
        raise AssertionError(f"group is not present in generation config: {only_group}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    total_cost_usd: float | None = 0.0
    built: list[str] = []
    cached: list[str] = []
    for skill_key in groups:
        if (config.output_dir / skill_key / "SKILL.md").exists():
            cached.append(skill_key)
            continue
        comps = selected[skill_key]
        n_writeups = sum(len(records) for records in comps.values())
        log.info(
            "[%s] summarizing reviewed selection from %d writeups across %d competitions ...",
            skill_key,
            n_writeups,
            len(comps),
        )
        cost = await summarize_workspace_one(
            calls,
            skill_key,
            comps,
            config.workspace_dir / skill_key,
            config.output_dir,
            config.model,
            config.work_dir,
            effort=config.effort,
        )
        total_cost_usd = sum_costs(total_cost_usd, cost)
        built.append(skill_key)
        log.info(
            "[%s] done (cost USD: %s; running total USD: %s).",
            skill_key,
            cost,
            total_cost_usd,
        )
    log.info(
        "Reviewed-selection build complete: %d built, %d cached. Total writer cost USD: %s.",
        len(built),
        len(cached),
        total_cost_usd,
    )


# ---------- Atomic replace ----------


def _atomic_replace(staging: Path, live: Path) -> None:
    """Replace the live skills dir with staging via renames (no half-written corpus)."""
    backup = live.parent / (live.name + ".backup")
    if backup.exists():
        shutil.rmtree(backup)
    if live.exists():
        os.rename(live, backup)
    os.rename(staging, live)
    if backup.exists():
        shutil.rmtree(backup)


# ---------- Data presence ----------


def assert_data_present(meta_dir: Path) -> None:
    missing = [
        name
        for name in (INDEX_NAME, COMPETITIONS_CSV, FORUM_TOPICS_CSV, FORUM_MESSAGES_CSV)
        if not (meta_dir / name).exists()
    ]
    if missing:
        raise AssertionError(
            f"Meta Kaggle inputs missing in {meta_dir}: {missing}. "
            f"Pass --meta-kaggle-dir with the folder that holds them."
        )


# ---------- Orchestration ----------


async def run(
    calls: AgentCalls,
    *,
    run_id: str | None,
    work_dir_override: str | os.PathLike | None,
    classify_model: str,
    writer_model: str,
    meta_dir: Path,
    min_writeups: int,
    min_task_competitions: int,
    top_n: int,
    k: int,
    batch_size: int,
    workers: int,
) -> None:
    work_dir = resolve_work_dir("playbooks", run_id=run_id, override=work_dir_override)
    setup_logging(work_dir)
    workspace_dir = work_dir / "workspace"
    assert_data_present(meta_dir)

    log.info("Loading faridrashidi index + Meta Kaggle tables ...")
    index = load_index(meta_dir / INDEX_NAME)
    comp_meta = load_competitions_meta(meta_dir)
    topics_by_forum = load_forum_topics(meta_dir)
    log.info(
        f"Loaded: {len(index)} competitions with solutions, "
        f"{len(comp_meta)} competition metadata rows, "
        f"{sum(len(v) for v in topics_by_forum.values())} forum topics."
    )

    # Phase A: match links -> message ids (in memory) for every competition with solutions.
    matches_by_comp: dict[str, list[PendingMatch]] = {}
    needed_message_ids: set[int] = set()
    for slug, comp in index.items():
        meta = comp_meta.get(slug)
        topics = (
            topics_by_forum.get(meta.forum_id, [])
            if (meta and meta.forum_id is not None)
            else []
        )
        m = match_solutions(comp.solutions, topics)
        matches_by_comp[slug] = m
        needed_message_ids.update(x.message_id for x in m)
    log.info(
        f"Phase A: matched {sum(len(v) for v in matches_by_comp.values())} solution links across "
        f"{len(index)} competitions -> {len(needed_message_ids)} unique messages to fetch."
    )

    # Phase B: one ForumMessages scan.
    log.info("Phase B: scanning ForumMessages.csv (one pass) ...")
    messages = fetch_messages(meta_dir, needed_message_ids)
    log.info(
        f"Phase B: fetched {len(messages)}/{len(needed_message_ids)} message bodies."
    )

    # Phase C: resolve writeups per competition; keep the competitions that have any.
    comps_with_text: dict[str, list[WriteupRecord]] = {}
    for slug in index:
        records = attach_text(slug, matches_by_comp.get(slug, []), messages)
        if records:
            comps_with_text[slug] = records
    log.info(f"Phase C: {len(comps_with_text)} competitions have resolved writeups.")

    # Phase D: classify every resolved competition into the shared PwC task taxonomy,
    # then group by PwC task (thin tasks fold to their PwC primary area).
    classified = classify_competitions(
        calls,
        comps_with_text,
        comp_meta,
        work_dir,
        model=classify_model,
        top_n=top_n,
        k=k,
        batch_size=batch_size,
        workers=workers,
    )
    skills_by_task = group_into_skills(
        classified,
        comps_with_text,
        min_items=min_task_competitions,
        top_n=top_n,
    )
    log.info(f"Phase D: {len(skills_by_task)} PwC-task skill groups.")

    # Phase E: per PwC-task group -> workspace -> summarize -> staging (resumable).
    staging = SKILLS_DIR.parent / (SKILLS_DIR.name + ".staging")
    staging.mkdir(parents=True, exist_ok=True)

    built: list[str] = []
    skipped_floor: list[tuple[str, int]] = []
    skipped_cached: list[str] = []
    total_cost_usd: float | None = 0.0

    for skill_key, comps in skills_by_task.items():
        if (staging / skill_key / "SKILL.md").exists():
            skipped_cached.append(skill_key)
            continue
        n_writeups = sum(len(v) for v in comps.values())
        if n_writeups < min_writeups:
            skipped_floor.append((skill_key, n_writeups))
            log.warning(
                "[%s] only %d writeups (< %d); skipping (no skill).",
                skill_key,
                n_writeups,
                min_writeups,
            )
            continue
        log.info(
            f"[{skill_key}] summarizing from {n_writeups} writeups across {len(comps)} competitions ..."
        )
        cost = await summarize_one(
            calls,
            skill_key,
            comps,
            comp_meta,
            workspace_dir,
            staging,
            writer_model,
            work_dir,
        )
        total_cost_usd = sum_costs(total_cost_usd, cost)
        built.append(skill_key)
        log.info(
            f"[{skill_key}] done (cost USD: {cost}; running total USD: {total_cost_usd})."
        )

    log.info(
        f"Build complete: {len(built)} built, {len(skipped_cached)} cached, "
        f"{len(skipped_floor)} skipped (< floor): {skipped_floor}. "
        f"Total writer cost USD: {total_cost_usd}."
    )

    _atomic_replace(staging, SKILLS_DIR)
    log.info(f"Atomic replace done -> {SKILLS_DIR}")
