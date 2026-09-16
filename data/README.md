# data/ — the served skill corpus

This directory holds the skill corpus the Kb service retrieves over (`corpus/plugins/`) and this README. The search index is a build artifact: it lives under the runtime root (`runtime_root()` in `sourcing/utils/work_dir.py`, which is `$AIBUILDAI_KB_RUNTIME_DIR` or `runtime/` at the repository root) and never enters the tracked tree. The Docker image copies the corpus and a built index into `/app/data/corpus/`.

```
data/
  corpus/plugins/           <- the served skill corpus (SKILL.md tree); indexed and served by the Kb
  README.md                 <- this file

<runtime-root>/
  index/                    <- txtai index over the corpus (offline build product)
  sourcing/                 <- per-run outputs of the sourcing pipelines
```

`deploy/deploy.sh` passes the runtime index to Docker as the named `kb_index` build context, and the Dockerfile copies that context into the image.

## data/corpus/plugins/

`data/corpus/plugins/` holds the full trees of these 5 plugins across 3 marketplaces:

| plugin | type |
|---|---|
| aibuildai-huggingface-marketplace/aibuildai-huggingface | Hugging Face how-to |
| aibuildai-modeling-marketplace/aibuildai-awesome-lists | academic-domain knowledge base (the bulk: about 111k markdown files across 460 skills) |
| aibuildai-modeling-marketplace/aibuildai-github | repository reference |
| aibuildai-modeling-marketplace/aibuildai-playbooks | domain ML playbooks, built without the writeups of MLE-bench competitions |
| aibuildai-posttrain-marketplace/aibuildai-posttrain | post-training workflow, datasets, methods, and frameworks |

The layout is `<marketplace>/<plugin>/skills/<skill>/SKILL.md` (+ `references/*.md`), and those three names ARE a skill's address: every retrieval tool takes the triple (marketplace, plugin, skill) together, so two plugins may hold a skill on the same topic. Every plugin needs `.claude-plugin/plugin.json` with a `description`. Anything directly under a plugin's `skills/` that is not a directory holding a `SKILL.md` is not a skill, whether it is a stray file or a directory without one: it fails the offline build, and it fails service start too, naming the exact path. Only `*.md` files inside a skill directory are indexed; other files are carried but not embedded. The index records one corpus hash, and the service refuses to start when the corpus and the index do not match.

The sourcing pipelines (`sourcing/`) write new content into `data/corpus/plugins/`; rebuild the index after any corpus change.
