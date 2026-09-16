"""LLM-classified filter for scraped awesome-* repos.

Input JSONL (one repo per line) -> YAML annotated with per-item
`verdict` (bool) and `reason` (str) fields.

Uses AgentCalls.run_batch for batching + repair rounds + per-batch transcript
dumps + checkpointed resume.

CLI:
    python -m sourcing.awesome_lists.filter --input tmp/awesome_repos.jsonl --run-id <id>
    # writes <run-dir>/filter_out/annotated.yaml (+ checkpoint.json, calls/)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from sourcing.utils.agent_calls import AgentCalls
from sourcing.utils.work_dir import (
    load_checkpoint,
    resolve_work_dir,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).parent
DEFAULT_INPUT = PIPELINE_DIR.parent.parent / "tmp" / "awesome_repos.jsonl"
DESCRIPTION_LIMIT = 500


class RepoVerdictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str
    verdict: bool
    reason: str


class BatchRepoVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[RepoVerdictItem]


_PROMPT_HEADER = """\
You are classifying GitHub `awesome-*` lists for a Machine Learning Engineering
(MLE) registry.

PRINCIPLE
A repo is IN SCOPE if its list supports a practitioner who BUILDS, TRAINS,
EVALUATES, or DEPLOYS machine learning models. Two patterns count as IN:
  (a) core-mle: ML methods/architectures, training or inference frameworks,
      datasets, benchmarks, MLOps tooling, or surveys of ML techniques.
  (b) domain-applied-ml: ML work targeted at a scientific or engineering
      domain (chemistry, biology, physics, medicine, robotics, materials,
      earth science, ai-for-science).

A repo is OUT OF SCOPE if it is:
  - A general programming language, framework, OS, editor, or tooling list
    unrelated to ML.
  - Devops, cloud infra, blockchain, cybersecurity, or web-dev resources with
    no ML training/serving focus.
  - A CONSUMER AI PRODUCT, PROMPT-ENGINEERING, CHATBOT, or AI-POWERED-DEV-TOOL
    list. A practitioner building ML models does not use these to construct
    models.
  - An AGENT / SKILL / MCP list aimed at end-user assistants rather than
    model training or inference.
  - A cheatsheet, interview-question list, CV/resume template, or education
    portal.

RESOLVING AMBIGUITY
Ambiguous cases default to OUT. Mark IN only if you can name the concrete MLE
activity (training, inference, evaluation, deployment, data curation, or
domain-applied ML research) that the list supports. Reason from the PRINCIPLE,
not from keyword matches.

EXAMPLES — IN (core-mle)
  - mrgloom/awesome-semantic-segmentation     (CV method survey)
  - opendilab/awesome-RLHF                    (RLHF training resources)
  - Duan-JM/awesome-papers-fewshot            (few-shot learning surveys)
  - Efficient-ML/Awesome-Model-Quantization   (inference compression)

EXAMPLES — IN (domain-applied-ml)
  - ai-boost/awesome-ai-for-science           (AI applied across sciences)
  - lmmentel/awesome-python-chemistry         (cheminformatics stack)
  - amorehead/awesome-molecular-generation    (molecular generation ML)

EXAMPLES — OUT (even though awesome-*)
  - jaywcjlove/awesome-mac                    (macOS apps)
  - vsouza/awesome-ios                        (iOS dev)
  - akullpp/awesome-java                      (language)
  - yjjnls/awesome-blockchain                 (non-ML infra)
  - ai-boost/awesome-prompts                  (prompt-engineering product)
  - promptslab/Awesome-Prompt-Engineering     (product layer, not MLE)
  - hesreallyhim/awesome-claude-code          (agentic coding product)
  - punkpeye/awesome-mcp-servers              (MCP protocol tooling)
  - spring-ai-community/awesome-spring-ai     (Java GenAI app framework)
  - jdorfman/awesome-json-datasets            (general datasets, not ML)

For each repo below, return `full_name` verbatim plus `verdict` (bool) and a
one-sentence `reason`.

REPOS:
"""


def _build_batch_prompt(batch: list[dict]) -> str:
    return _PROMPT_HEADER + json.dumps(batch, indent=2)


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run(
    calls: AgentCalls,
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    *,
    model: str,
    workers: int,
    batch_size: int,
    max_batches: int | None,
) -> dict:
    filter_out_dir = work_dir / "filter_out"
    filter_out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = filter_out_dir / "checkpoint.json"
    calls_dir = filter_out_dir / "calls"
    ck: dict[str, dict] = load_checkpoint(checkpoint)

    all_repos = _load_jsonl(input_path)

    pending = [r for r in all_repos if r["full_name"] not in ck]
    items = [
        {
            "full_name": repo["full_name"],
            "description": (
                (repo.get("description") or "")[:DESCRIPTION_LIMIT] or "none"
            ),
            "topics": ", ".join(repo.get("topics") or []) or "none",
            "stars": repo.get("stars", 0),
            "language": repo.get("language") or "unknown",
            "archived": repo.get("archived", False),
        }
        for repo in pending
    ]

    logger.info(
        "[filter] total=%d already_classified=%d pending=%d",
        len(all_repos), len(ck), len(pending),
    )

    def _on_batch_complete(verdicts: list[dict]) -> None:
        for v in verdicts:
            ck[v["full_name"]] = v
        save_checkpoint(checkpoint, ck)

    if items:
        asyncio.run(calls.run_batch(
            items=items,
            make_prompt=_build_batch_prompt,
            key_of=lambda x: x["full_name"],
            output_schema=BatchRepoVerdicts,
            system_prompt="",
            model=model,
            calls_dir=calls_dir,
            batch_size=batch_size,
            workers=workers,
            on_batch_complete=_on_batch_complete,
            max_batches=max_batches,
        ))

    annotated: list[dict] = []
    missing: list[str] = []
    for repo in all_repos:
        name = repo["full_name"]
        v = ck.get(name)
        if v is None:
            missing.append(name)
            continue
        annotated.append({
            **repo,
            "verdict": bool(v["verdict"]),
            "reason": v["reason"],
        })

    annotated.sort(key=lambda r: (-int(r["verdict"]), -r["stars"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = sum(1 for r in annotated if r["verdict"])
    yaml.safe_dump(
        {
            "source": "scrape+llm-filter",
            "input_jsonl": str(input_path),
            "total": len(all_repos),
            "classified": len(annotated),
            "in_scope": kept,
            "out_of_scope": len(annotated) - kept,
            "repos": annotated,
        },
        output_path.open("w", encoding="utf-8"),
        sort_keys=False,
        allow_unicode=True,
    )

    stats = {
        "total": len(all_repos),
        "classified": len(annotated),
        "missing_verdicts": len(missing),
        "in_scope": kept,
        "out_of_scope": len(annotated) - kept,
        "output": str(output_path),
    }
    logger.info("[filter] %s", stats)
    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="MLE-scope filter for scraped awesome-* repos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=None,
                   help="annotated.yaml (default: <run-dir>/filter_out/annotated.yaml)")
    p.add_argument("--workers", type=int, default=4, help="Parallel LLM calls")
    p.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model used to classify repositories",
    )
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Process at most N batches (for smoke tests).")
    p.add_argument("--run-id", default=None,
                   help="Shared run id; the three awesome_lists CLIs MUST share it.")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Override the resolved work dir entirely.")
    args = p.parse_args()
    work_dir = resolve_work_dir("awesome_lists", run_id=args.run_id, override=args.work_dir)
    output_path = args.output or (work_dir / "filter_out" / "annotated.yaml")
    calls = AgentCalls()
    stats = run(
        calls,
        input_path=args.input,
        output_path=output_path,
        work_dir=work_dir,
        model=args.model,
        workers=args.workers,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
