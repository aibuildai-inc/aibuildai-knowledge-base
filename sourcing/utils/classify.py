# sourcing/utils/classify.py
"""Shared: classify producer items into the PwC research task taxonomy.

Replaces per-pipeline LLM clustering. Embedding narrows each item to its top-K
candidate tasks (drawn from the top-N most paper-frequent PwC tasks); one Haiku
call then picks 1-3 verbatim from THAT item's candidates. Embedding only narrows;
the LLM decides. Both `papers` and `github` reuse `load_label_set` +
`candidate_lists`; `papers` also uses `classify_items` + `group_by_task`.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sourcing.utils.agent_calls import AgentCalls

TAXONOMY_PATH = Path(__file__).parent / "research_taxonomy.json"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_label_set(top_n: int, *, path: Path = TAXONOMY_PATH) -> tuple[list[str], dict[str, str]]:
    """Return (top-N task display names by paper_freq, {task: primary_area})."""
    records = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    records = sorted(records, key=lambda r: r["paper_freq"], reverse=True)[:top_n]
    names = [r["task"] for r in records]
    area = {r["task"]: r["primary_area"] for r in records}
    return names, area


def _embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # lazy: heavy, GPU-first
    model = SentenceTransformer(EMBED_MODEL)
    return np.asarray(model.encode(
        texts, show_progress_bar=False, batch_size=64, normalize_embeddings=True))


def candidate_lists(item_texts: list[str], labels: list[str], k: int) -> list[list[str]]:
    """Per item, the top-k label names by cosine (labels embedded once, normalized)."""
    label_emb = _embed(labels)
    item_emb = _embed(item_texts)
    sims = item_emb @ label_emb.T
    out: list[list[str]] = []
    for row in sims:
        idx = np.argsort(-row)[:k]
        out.append([labels[int(i)] for i in idx])
    return out


# append to sourcing/utils/classify.py

class _Pick(BaseModel):
    """One item's verdict: 1-3 tasks chosen verbatim from its candidates."""
    model_config = ConfigDict(extra="forbid")
    item_id: str
    tasks: list[str] = Field(min_length=1, max_length=3)
    tags: list[str] = Field(default_factory=list, max_length=5)
    tldr: str = ""


class _PickVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[_Pick]


def _build_pick_prompt(batch: list[dict]) -> str:
    blocks = []
    for it in batch:
        cands = "\n".join(f"  - {c}" for c in it["candidates"])
        blocks.append(
            f"[item_id={it['item_id']}]\n{it['text'][:1500]}\nCandidate tasks:\n{cands}"
        )
    return (
        "For each item, choose the 1-3 research tasks from ITS OWN candidate list "
        "that best describe it. Use the task names VERBATIM; choose only from that "
        "item's candidates; most-relevant first. Also give up to 5 topical tags and "
        "a one-sentence tldr. One verdict per item; echo item_id exactly.\n\n"
        + "\n\n".join(blocks)
    )


def verify_verbatim_picks(field: str) -> Callable[[dict, dict], "str | None"]:
    """The caller rule both classifiers hand to `run_batch`: every entry of the
    verdict's ``field`` is a verbatim member of that item's own candidate list.
    The product's verify loop sends the reason back to the same session, so a
    weak pick is repaired there instead of being replaced by a default here."""
    def verify(item: dict, verdict: dict) -> "str | None":
        off = [t for t in verdict[field] if t not in item["candidates"]]
        if off:
            return (
                f"these {field} are not in this item's own candidate task "
                "list; choose only from its candidates, verbatim: "
                + ", ".join(off)
            )
        return None
    return verify


def classify_items(
    items: list[dict],
    *,
    calls: AgentCalls,
    text_of: Callable[[dict], str],
    id_of: Callable[[dict], str],
    model: str,
    top_n: int,
    k: int,
    batch_size: int,
    workers: int,
    calls_dir: Path,
) -> list[dict]:
    """Annotate each item with `primary_task` (first pick), `task_tags` (the
    rest), `tags`, `tldr`. Embedding narrows to top-K candidates; Haiku picks via
    the shared backend call, which renders one product transcript per batch under
    `calls_dir`."""
    labels, _area = load_label_set(top_n)
    cand = candidate_lists([text_of(it) for it in items], labels, k)
    llm_items = [
        {"item_id": id_of(it), "text": text_of(it), "candidates": c}
        for it, c in zip(items, cand, strict=True)
    ]
    verdicts = asyncio.run(calls.run_batch(
        items=llm_items,
        make_prompt=_build_pick_prompt,
        key_of=lambda x: x["item_id"],
        output_schema=_PickVerdicts,
        system_prompt="",
        model=model,
        calls_dir=calls_dir,
        batch_size=batch_size,
        workers=workers,
        verify_verdict=verify_verbatim_picks("tasks"),
    ))
    by_id = {v["item_id"]: v for v in verdicts}
    out: list[dict] = []
    for it in items:
        v = by_id[id_of(it)]
        picks = v["tasks"]
        out.append({
            **it, "primary_task": picks[0], "task_tags": picks[1:],
            "tags": v.get("tags", []), "tldr": v.get("tldr", ""),
        })
    return out


def group_by_task(
    items: list[dict],
    *,
    skill_key: Callable[[str], str],
    area_of: dict[str, str],
    min_items: int,
) -> dict[str, list[dict]]:
    """Group items by `primary_task` into skills; any task group smaller than
    `min_items` folds into a coarse group keyed by its `primary_area` (this
    parent-area fallback is internal behavior, not named in the function). Returns
    {skill_dir_key: items}. Each item lands in exactly one skill."""
    by_task: dict[str, list[dict]] = {}
    for it in items:
        by_task.setdefault(it["primary_task"], []).append(it)
    skills: dict[str, list[dict]] = {}
    for task, group in by_task.items():
        target = task if len(group) >= min_items else area_of.get(task, "Miscellaneous")
        skills.setdefault(skill_key(target), []).extend(group)
    return skills
