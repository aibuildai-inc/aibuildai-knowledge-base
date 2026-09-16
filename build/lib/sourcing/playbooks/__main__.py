import argparse
import asyncio
import os
from pathlib import Path

from sourcing.playbooks.pipeline import run, run_reviewed_selection
from sourcing.utils.agent_calls import AgentCalls


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m sourcing.playbooks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument(
        "--classify-model",
        default="claude-haiku-4-5-20251001",
        help="Model used to classify competitions",
    )
    parser.add_argument(
        "--writer-model",
        default=os.environ.get(
            "PLAYBOOKS_WRITER_MODEL",
            "claude-sonnet-4-5-20250929",
        ),
        help="Model used to write playbooks",
    )
    parser.add_argument(
        "--meta-kaggle-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "PLAYBOOKS_META_KAGGLE_DIR",
                str(Path(__file__).parent / "meta-kaggle"),
            )
        ),
    )
    parser.add_argument(
        "--min-writeups",
        type=int,
        default=int(os.environ.get("PLAYBOOKS_MIN_WRITEUPS", "5")),
        help="Smallest writeup group that produces a skill",
    )
    parser.add_argument(
        "--min-task-competitions",
        type=int,
        default=int(os.environ.get("PLAYBOOKS_MIN_TASK_COMPETITIONS", "5")),
    )
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--candidate-count", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4, help="Parallel LLM calls")
    parser.add_argument("--batch-size", type=int, default=50, help="Competitions per LLM call")
    args = parser.parse_args()
    if args.config and (args.run_id or args.work_dir):
        parser.error("--run-id and --work-dir cannot be used with --config")
    if not args.config and args.group:
        parser.error("--group requires --config")
    calls = AgentCalls()
    if args.config:
        asyncio.run(
            run_reviewed_selection(
                calls,
                Path(args.config),
                only_group=args.group,
            )
        )
    else:
        asyncio.run(
            run(
                calls,
                run_id=args.run_id,
                work_dir_override=args.work_dir,
                classify_model=args.classify_model,
                writer_model=args.writer_model,
                meta_dir=args.meta_kaggle_dir,
                min_writeups=args.min_writeups,
                min_task_competitions=args.min_task_competitions,
                top_n=args.top_n,
                k=args.candidate_count,
                batch_size=args.batch_size,
                workers=args.workers,
            )
        )


if __name__ == "__main__":
    main()
