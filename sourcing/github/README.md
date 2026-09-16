# sourcing/github/

Sources ML repos from GitHub search, fetches each README, classifies it into the frozen PwC task taxonomy, and emits one `SKILL.md` per repo into the `aibuildai-github` plugin.

## How to run

```bash
python -m sourcing.github                 # full pipeline (search -> fetch -> classify -> emit)
```

Export `GITHUB_TOKEN` to raise the unauthenticated 60 req/hr search ceiling to 5000.

## Output

- Corpus plugin (committed): `data/corpus/plugins/aibuildai-modeling-marketplace/aibuildai-github/skills/`.
- Per-run intermediates (caches, render steps, classify calls): outside the tracked tree under `<runtime-root>/sourcing/github/<run-id>/` (resolved by `sourcing/utils/work_dir.py:runtime_root`).
