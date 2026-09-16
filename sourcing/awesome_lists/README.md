# sourcing/awesome_lists/

Ingests curated "awesome-*" GitHub lists and the repos they link, running a per-repo LLM extraction that summarizes each into the `aibuildai-awesome-lists` plugin -- the bulk of the corpus (~111k markdown files across ~460 skills).

## How to run

```bash
python -m sourcing.awesome_lists                     # list-fetch / parse pipeline
python -m sourcing.awesome_lists.extract --workers 4 # per-repo LLM extraction driver
```

## Output

- Corpus plugin (committed): `data/corpus/plugins/aibuildai-modeling-marketplace/aibuildai-awesome-lists/skills/`.
- Per-run extraction intermediates (per-repo work dirs, render steps): outside the tracked tree under `<runtime-root>/sourcing/awesome_lists/<run-id>/` (resolved by `sourcing/utils/work_dir.py:runtime_root`).
