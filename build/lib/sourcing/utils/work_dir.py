"""Shared work-dir / run-id / logging helpers for every `sourcing/` pipeline.

One run of one pipeline writes everything under a single work dir:

    <runtime-root>/sourcing/<pipeline>/<run-id>/
        steps/<label>/            one render step dir per AgentConversation.ask call
        <stage>/calls/batch_NNN/  one per AgentCalls.run_batch batch
        ... caches / checkpoints / intermediate stage JSON (exact current names) ...

The runtime root keeps every runtime artifact (sourcing outputs and the Kb search
index) out of the tracked tree, so it never pollutes git and an interrupted run
resumes the same dir. `runtime_root()` is the SINGLE source for that base:
$AIBUILDAI_KB_RUNTIME_DIR when set, otherwise `runtime/` at the repository root
(ignored by git). The sourcing root is `runtime_root() / "sourcing"`.
$SOURCING_WORK_DIR overrides the whole work-dir composition with a literal work-dir.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

# sourcing/utils/work_dir.py -> parents[0]=utils, [1]=sourcing, [2]=repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def atomic_write_text(target: Path, content: str) -> None:
    """Write through a temporary file and a rename, so a reader never sees a partial file."""
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def runtime_root() -> Path:
    """The runtime root holding every aibuildai-mcp runtime artifact -- sourcing
    pipeline outputs and the Kb search index -- outside the tracked tree.

    SINGLE source for that base path; the deploy launchers import it (rather than
    re-deriving it in shell) so the layout has one definition:
    $AIBUILDAI_KB_RUNTIME_DIR when set, otherwise ``runtime/`` at the repository root.
    """
    configured = os.environ.get("AIBUILDAI_KB_RUNTIME_DIR")
    if configured:
        return Path(configured)
    return _REPO_ROOT / "runtime"


def resolve_run_id(run_id: "str | None" = None) -> str:
    """Resolve the run id: explicit arg > $SOURCING_RUN_ID > ``"default"``.

    The default is stable so an interrupted run resumes the SAME dir; the id MUST
    thread through every stage of one pipeline run so inter-stage filename
    contracts resolve to the same dir.
    """
    return run_id or os.environ.get("SOURCING_RUN_ID") or "default"


def resolve_work_dir(
    pipeline: str, *, run_id: "str | None" = None, override: "str | os.PathLike | None" = None,
) -> Path:
    """Resolve (and mkdir -p) the work dir for one pipeline run.

    Precedence: ``override`` arg > $SOURCING_WORK_DIR >
    ``<analysis-sourcing-root>/<pipeline>/<resolve_run_id()>``.
    """
    if override is not None:
        work_dir = Path(override)
    elif os.environ.get("SOURCING_WORK_DIR"):
        work_dir = Path(os.environ["SOURCING_WORK_DIR"])
    else:
        work_dir = runtime_root() / "sourcing" / pipeline / resolve_run_id(run_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def step_dir(work_dir: Path, *parts: str) -> Path:
    """Resolve (and mkdir -p) ``work_dir/steps/<*parts>`` -- the render step dir
    for one single ``AgentConversation.ask`` call."""
    d = Path(work_dir).joinpath("steps", *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(work_dir: Path) -> None:
    """Configure the root logger for one pipeline run: a DEBUG file handler under
    ``work_dir/logs/sourcing_<ts>.log`` plus an INFO stream handler. Relocated
    from the playbooks pipeline so every pipeline shares one logging setup."""
    log_dir = Path(work_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sourcing_{ts}.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)
    logging.getLogger(__name__).info("Log file: %s", log_file)


def load_checkpoint(path: Path) -> dict:
    """Read one sourcing checkpoint, returning an empty state when it is absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, data: dict) -> None:
    """Atomically write one sourcing checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2))
