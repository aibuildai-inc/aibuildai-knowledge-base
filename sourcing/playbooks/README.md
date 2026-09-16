# sourcing/playbooks/

Builds the `aibuildai-playbooks` corpus from the faridrashidi competitions index + local Meta Kaggle text: resolve each competition's text, classify every competition into the shared Papers-with-Code (PwC) task taxonomy (`sourcing/utils/classify.py`), group the competitions by PwC task, build a per-task workspace, summarize one playbook skill per group via the writer LLM (fixed creator prompt), then atomic-replace it into the live corpus.

## Topic keying (PwC tasks)

A skill is keyed by a PwC task -- the one topic vocabulary the sourcing pipelines classify into (`github`, `playbooks`). Each competition is narrowed to its top-K candidate tasks by embedding, then a single Haiku call picks 1-3 verbatim (the same `classify.classify_items` path github uses); the competitions are then grouped by their primary task. A task with fewer than `--min-task-competitions` competitions folds into its PwC primary area, so a sparse task lands in a broader, well-populated area skill instead of a thin one. There is no per-pipeline topic table; the writer's principle-index layout is unchanged.

## Excluded competitions

`excluded_competitions.txt` lists every competition of the benchmarks we run (today MLE-bench). Their top-solution writeups are the benchmark's own answers, so `load_index` drops them before any text is resolved and `load_reviewed_selection` drops their rows; a rebuild cannot add them back. Add a benchmark's competitions to that file before its first run.

## How to run

```bash
python -m sourcing.playbooks                 # full pipeline
python -m sourcing.playbooks --run-id <id>   # resume a specific run dir
python -m sourcing.playbooks --help           # models, workers, and input limits
```

The Meta Kaggle producer input (downloaded CSVs + the faridrashidi index) lives under `sourcing/playbooks/meta-kaggle/` (gitignored, regenerable).

## Output

- Corpus plugin (committed): `data/corpus/plugins/aibuildai-modeling-marketplace/aibuildai-playbooks/skills/` (written via an atomic replace from a staging dir).
- Per-run intermediates (workspaces, render steps, logs): outside the tracked tree under `<runtime-root>/sourcing/playbooks/<run-id>/` (resolved by `sourcing/utils/work_dir.py:runtime_root`).
