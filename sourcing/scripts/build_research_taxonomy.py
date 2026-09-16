# sourcing/scripts/build_research_taxonomy.py
"""One-time builder: extract a clean PwC task taxonomy into research_taxonomy.json.

Source: PwC's `evaluation-tables.json` (frozen 2025-07 snapshot, CC-BY-SA 4.0) —
the only lossless task table; no clean standalone file exists. We recurse it,
drop the `datasets`/`description`/`source_link` chrome, keep `task`/`categories`/
`subtasks`/`synonyms`, apply basic normalization (NO semantic merge), drop a
denylist of garbage labels, and record each task's paper frequency from the
papers-with-abstracts snapshot. Run offline; commit the JSON it writes.

  python -m sourcing.scripts.build_research_taxonomy
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import urllib.request
from collections import Counter
from pathlib import Path

logger = logging.getLogger("sourcing.taxonomy.build")

EVAL_TABLES_URL = (
    "https://media.githubusercontent.com/media/World-Snapshot/papers-with-code/"
    "main/data/evaluation-tables.json.gz"
)
PAPERS_URL = (
    "https://huggingface.co/datasets/pwc-archive/papers-with-abstracts/resolve/"
    "main/data/train-{i:05d}-of-00004.parquet"
)
# The builder lives in sourcing/scripts/, but the tracked taxonomy it writes stays
# in sourcing/utils/ (classify.py reads it from there).
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "utils" / "research_taxonomy.json"

GARBAGE = {
    "model", "sentence", "object", "decoder", "diversity", "prediction",
    "deep learning", "general classification", "big-bench machine learning",
    "big bench machine learning",
}


def normalize_name(s: str) -> str:
    """Lowercase, hyphen==space, collapse whitespace, strip surrounding punctuation."""
    s = s.replace("-", " ").lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" .,:;\"'`()[]")
    return s


def _is_garbage(norm: str) -> bool:
    return (not norm) or norm in GARBAGE or norm.startswith("(deleted task")


def extract_tasks(eval_tables: list[dict]) -> dict[str, dict]:
    """Recurse the DAG; return {normalized_task: record}. Identical-after-normalize
    names collapse to one record, keeping the other display strings as synonyms.
    `primary_area` = the first `categories` value seen, inheriting from the nearest
    ancestor that has one."""
    out: dict[str, dict] = {}

    def visit(node: dict, inherited_area: str | None) -> None:
        raw = node.get("task") or ""
        norm = normalize_name(raw)
        cats = [c for c in (node.get("categories") or []) if normalize_name(c)]
        area = cats[0] if cats else inherited_area
        if norm and not _is_garbage(norm):
            rec = out.get(norm)
            if rec is None:
                out[norm] = {
                    "task": raw, "primary_area": area or "Miscellaneous",
                    "areas": list(dict.fromkeys(cats)), "synonyms": [],
                }
            else:
                if raw != rec["task"] and raw not in rec["synonyms"]:
                    rec["synonyms"].append(raw)
                if not rec["areas"] and cats:
                    rec["areas"] = list(dict.fromkeys(cats))
                    rec["primary_area"] = cats[0]
        for child in node.get("subtasks") or []:
            visit(child, area)

    for top in eval_tables:
        visit(top, None)
    return out


def build(eval_tables: list[dict], freq: dict[str, int]) -> dict:
    """Assemble the SSOT dict: 16 areas + one record per task with paper_freq."""
    tasks = extract_tasks(eval_tables)
    areas: list[str] = []
    records: list[dict] = []
    for norm, rec in tasks.items():
        rec = {**rec, "paper_freq": int(freq.get(norm, 0))}
        records.append(rec)
        if rec["primary_area"] not in areas:
            areas.append(rec["primary_area"])
    records.sort(key=lambda r: r["paper_freq"], reverse=True)
    return {"areas": sorted(areas), "tasks": records}


def _download_eval_tables() -> list[dict]:
    logger.info("downloading %s", EVAL_TABLES_URL)
    with urllib.request.urlopen(EVAL_TABLES_URL, timeout=120) as r:
        data = gzip.decompress(r.read())
    return json.loads(data)


def _paper_frequencies() -> dict[str, int]:
    """Count distinct papers referencing each normalized task across the 4 shards."""
    import pyarrow.parquet as pq  # builder-only dep

    counts: Counter[str] = Counter()
    for i in range(4):
        url = PAPERS_URL.format(i=i)
        logger.info("reading %s", url)
        local = Path(f"/tmp/pwc_papers_{i}.parquet")
        if not local.exists():
            urllib.request.urlretrieve(url, local)
        tbl = pq.read_table(local, columns=["tasks"])
        for task_list in tbl.column("tasks").to_pylist():
            seen = {normalize_name(t) for t in (task_list or [])}
            for norm in seen:
                if norm:
                    counts[norm] += 1
    return dict(counts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    eval_tables = _download_eval_tables()
    freq = _paper_frequencies()
    ssot = build(eval_tables, freq)
    OUTPUT_PATH.write_text(json.dumps(ssot, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %d tasks, %d areas -> %s", len(ssot["tasks"]), len(ssot["areas"]), OUTPUT_PATH)


if __name__ == "__main__":
    main()
