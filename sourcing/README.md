# sourcing/

Offline pipelines that produce `aibuildai-*` corpus plugins.

## Running a pipeline

Every pipeline runs as a module from the repository root, in the `[producers]` environment described in the top-level README. All per-run output lands outside the tracked tree under the runtime root (see "Output placement"). Only the produced corpus plugins are written in-tree.

| Pipeline | Entry command |
|---|---|
| `awesome_lists` | `python -m sourcing.awesome_lists` (per-repo extraction: `python -m sourcing.awesome_lists.extract`) |
| `github` | `python -m sourcing.github` (set `GITHUB_TOKEN` for the GitHub API) |
| `playbooks` | `python -m sourcing.playbooks` |

## Pipelines

"Corpus plugin" is the committed `data/corpus/plugins/<marketplace>/<plugin>` a pipeline writes.

| Pipeline | Corpus plugin produced | Where the target is hardcoded |
|---|---|---|
| `awesome_lists/` | `aibuildai-modeling-marketplace/aibuildai-awesome-lists` | `parse_readme.py` `OUTPUT_DIR` |
| `github/` | `aibuildai-modeling-marketplace/aibuildai-github` | `run.py` `_PLUGIN_DIR` |
| `playbooks/` | `aibuildai-modeling-marketplace/aibuildai-playbooks` | `pipeline.py` `SKILLS_DIR`; the normal run replaces the committed skills atomically, while the reviewed-selection command writes to its configured output folder |

**Corpus plugin with no `sourcing/` producer.** This committed plugin is hand-curated; no pipeline here writes it: `aibuildai-huggingface-marketplace/aibuildai-huggingface`.

## Naming convention

Each pipeline directory is named by WHAT IT SOURCES, with no `aibuildai_` prefix. A single-output pipeline emits the plugin `aibuildai-<dir-name-as-kebab>` (e.g. `awesome_lists/` -> `aibuildai-awesome-lists`).

## Shared code (`sourcing/utils/`)

Every pipeline drives the model through one shared toolkit, so the model call, the per-call record, and the work-dir layout are not re-rolled per pipeline:

- `agent_calls.py` — `AgentCalls` owns the calls of one sourcing command and what each step dir spent. `AgentCalls.conversation(...)` opens one Claude Agent SDK client for a conversation; `AgentConversation.ask` sends one prompt, sends a rejected structured result back to the same session until `verify` accepts it, and writes `execution.md`, `trajectory.json`, `system_prompt.md`, `user_message.md`, and `output_1.json` to the step dir. `run_batch` runs one structured call per batch with a worker cap and joins each result by its key.
- `work_dir.py` — `runtime_root` / `resolve_work_dir` / `resolve_run_id` / `step_dir` / `setup_logging` / `atomic_write_text` and the checkpoint helpers; `runtime_root()` is the single source for the runtime root, and one run writes everything under `<runtime-root>/sourcing/<pipeline>/<run-id>/`.
- `classify.py` — narrows items to candidate Papers-with-Code tasks by embedding, then asks the model to pick from them. `research_taxonomy.json` is the frozen task table, rebuilt with `python -m sourcing.scripts.build_research_taxonomy`.

## Environment variables

| Variable | Read by | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | the Claude Agent SDK | Needed only when no local Claude Code login exists |
| `GITHUB_TOKEN` | `github/run.py` | Optional. Raises the GitHub API rate limit |
| `AIBUILDAI_KB_RUNTIME_DIR` | `utils/work_dir.py` | Optional. The runtime root; defaults to `runtime/` at the repository root |
| `SOURCING_RUN_ID` | `utils/work_dir.py` | Optional. The run id when `--run-id` is not given; defaults to `default` |
| `SOURCING_WORK_DIR` | `utils/work_dir.py` | Optional. A literal work dir that replaces the whole `<runtime-root>/sourcing/<pipeline>/<run-id>/` composition |
| `PLAYBOOKS_WRITER_MODEL` | `playbooks/__main__.py` | Optional. Default for `--writer-model` |
| `PLAYBOOKS_META_KAGGLE_DIR` | `playbooks/__main__.py` | Optional. Default for `--meta-kaggle-dir` |
| `PLAYBOOKS_MIN_WRITEUPS` | `playbooks/__main__.py` | Optional. Default for `--min-writeups`; 5 |
| `PLAYBOOKS_MIN_TASK_COMPETITIONS` | `playbooks/__main__.py` | Optional. Default for `--min-task-competitions`; 5 |

## Output placement

The tracked tree holds only durable content — code, configs, the frozen `utils/research_taxonomy.json`, and the corpus under `data/corpus/plugins/`. Every runtime or intermediate artifact lives under the runtime root, `$AIBUILDAI_KB_RUNTIME_DIR` or `runtime/` at the repository root:

```
<runtime-root>/
  sourcing/<pipeline>/<run-id>/             <- one run dir per pipeline run (resolve_work_dir)
    steps/<label>/                          per-call step dirs
    <stage>/calls/batch_NNN/                per-batch call dirs
    cache/  logs/
  index/                                    <- Kb txtai index (offline build product)
```

## Adding a new pipeline

A new pipeline is a sibling directory `sourcing/<name>/`:

- `__init__.py` + `__main__.py` so it runs as `python -m sourcing.<name>`. The `__main__.py` is a two-liner: `from sourcing.<name>.run import main` then `if __name__ == "__main__": main()` (see `sourcing/github/__main__.py`).
- `run.py` with a `main()` entry. Hardcode the output target inside the pipeline, pointing at `data/corpus/plugins/<marketplace>/aibuildai-<kebab-name>/skills/` (pattern: `sourcing/github/run.py` `_PLUGIN_DIR`).
- `configs/` for any pipeline config (e.g. search queries).
- Shared output schema and path helpers come from `sourcing/utils/` (`skill_output.py`, `work_dir.py`); model calls go through `sourcing/utils/agent_calls.py`.
- Register producer deps in `pyproject.toml` `[project.optional-dependencies].producers`. `sourcing/github/` is the simplest template to copy.
