# pyre-code

An interview-prep coding judge, not a post-training trainer: it grades hand-written PyTorch implementations of individual loss functions (including DPO, GRPO, PPO, and a reward-model loss) against unit tests, but it never trains a model or runs an alignment job.

**Pyre Code** describes itself as a platform where a learner writes an implementation of one of "76 problems", "a local grading service runs the tests", and the learner "see[s] what broke" [1]. The repository is owned by the GitHub account whwangovo [2][3]; its `pyproject.toml` names the package author as "kiren" [4], the LICENSE file's added copyright line reads "Copyright (c) 2024 kiren" [5], and the commit read for this card was itself authored by "kiren" per its Git commit metadata [6] - so the maintainer's working name is kiren, distinct from the GitHub account login. The project builds on the judge engine and problem set of TorchCode by GitHub user duoan, which the README credits as the basis for the `torch_judge/` package [1]. Its API shape is a browser code editor plus a FastAPI grading backend: a learner submits a Python function body over HTTP, and the backend's own `_execute_tests(code, task, ...)` function in `grading_service/main.py` `exec()`s that code into a fresh namespace, then `exec()`s each of the task's bundled test snippets against it [7]. `torch_judge/engine.py` also defines a `check(task_id)` function, and the package's own `torch_judge/__init__.py` re-exports it as the package's documented top-level entry point for a separate, Jupyter-notebook workflow - its module docstring reads "Usage: `from torch_judge import check, status`" [8][9] - but the FastAPI web backend never imports or calls it; the web app's grading path runs entirely through `_execute_tests` in `grading_service/main.py` instead [7]. It lives at https://github.com/whwangovo/pyre-code [3].

**When to pick it**: never for running an actual training job - it ships no trainer class, no optimizer loop, no checkpoint, and no distributed launcher. Pick it only as a standalone study aid for ML/RL interview prep or for someone who wants to hand-implement isolated loss functions (attention variants, normalization, DPO, GRPO, PPO, a Bradley-Terry reward-model loss, diffusion steps, GNN layers, and more) and get pass/fail feedback against reference tests, with no GPU required [1]. The shortlist row that names this repository itself records `"trains_what": null`, matching what the code shows: there is nothing here that trains a model on your data.

**Methods it ships**: not trainers - four of the 76 problems are named after alignment/RLHF methods, and each is a single pure function graded against unit tests, with no surrounding training loop [1]. At commit 7ede95b, `torch_judge/tasks/dpo_loss.py` defines a `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1)` task whose reference solution computes `-log(sigmoid(beta*((chosen-ref_chosen)-(rejected-ref_rejected)))).mean()` [10]. The same commit's `grpo_loss.py` defines a `grpo_loss(logps, rewards, group_ids, eps=1e-5)` task that normalizes rewards within each prompt group before computing per-token advantages [11]; `ppo_loss.py` defines a `ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2)` clipped-surrogate task [12]; and `reward_model.py` defines a `reward_model_loss` task built on the Bradley-Terry preference model over chosen/rejected hidden states [13]. These four sit in a "Loss & Alignment" category alongside cross-entropy, label smoothing, focal loss, and contrastive loss, and in an "Alignment & Agent Reasoning" learning path that also includes MCTS [1].

**Scale it handles**: not applicable - there is no training run to scale. The README states the problems need "No GPU" and that the whole stack (Next.js frontend, FastAPI grading backend, SQLite progress store) is meant to run on one machine via `npm run dev` or Docker Compose [1].

**Install**: `git clone` then `./setup.sh` (or the Windows `setup.bat`/`setup.ps1` equivalent) followed by `npm run dev`, or `pip install -e ".[dev]"` plus `npm install` for a manual venv/conda setup, per the README [1]. At commit 7ede95b, `pyproject.toml`'s `requires-python` sets a package-level floor of >=3.10 and pins no version floors on its runtime dependencies (`fastapi`, `uvicorn`, `torch`, `numpy`, `pydantic` are all unpinned); the `dev` extra adds `build`, `twine`, `pytest`, also unpinned [4]. The README's own "Prerequisites" section and all three of its quick-start recipes instead target Python 3.11+ (`Python 3.11+` as a stated prerequisite, and `uv venv --python 3.11` / `conda create -n pyre python=3.11` in the setup commands) [1] - a stricter floor than the 3.10 the package itself declares, so a reader should follow the README's 3.11+ over the package metadata's 3.10. `package.json` at the same commit additionally requires Node.js (the README states Node.js 18+) and lists one devDependency, `concurrently@^8.2.2` [14][1]. No CUDA or hardware minimum is stated anywhere in the docs read for this card; the README explicitly says none is needed [1]. Licensing is inconsistent across the repository: `pyproject.toml` declares `license = "MIT"` [4] and the repository's `LICENSE` file opens with "MIT License" and the standard MIT permission text, with added copyright lines for duoan (2024) and kiren (2024) [5], yet GitHub's license-detection API for this repository returns `spdx_id: "NOASSERTION"` with no further explanation of why [15] - consistent with the shortlist row's own `"licence": "NOASSERTION"` value, which the row likewise does not explain.

**Maintained by**: a single maintainer working under the name kiren (per the package author field, the LICENSE copyright line, and the Git commit-author metadata of the commit read for this card), on the GitHub account whwangovo [2][4][5][6]; the repository was created 2026-04-09 and last pushed 2026-05-12T14:55:30Z, with a README changelog documenting incremental feature additions (a GNN problem path, a UI redesign, submission history, an optional AI-hints feature) through 2026-04-20, the most recent dated entry read for this card [1][3]. Star counts are not used here to rank it.

## Quick start

There is no training script to quote - the "quick start" is standing up the practice platform itself, taken from the README [1]:

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code
./setup.sh
npm run dev
```

This starts the grading service at `http://localhost:8000` and the web app at `http://localhost:3000` [1]. A learner then opens a problem (e.g., DPO Loss) in the browser editor, writes the function body, and submits it for grading; no separate CLI form for grading a single file outside the web UI is documented in the README [1]. Docker Compose is offered as an alternative: `docker compose up --build`, then open `http://localhost:3000` [1].

## Start it

Not applicable in the training-run sense this section normally covers - there is no model, dataset, optimizer, or distributed launcher to start. What a learner "starts" is a locally-served editor and grading backend, launched with `npm run dev` (which runs the FastAPI backend and Next.js frontend concurrently) or `docker compose up --build`, both single-machine [1]. Under the hood, submitting a solution sends the code to the FastAPI `grading_service`, whose `_execute_tests` function `exec()`s the submitted code into a fresh namespace and then `exec()`s each of the task's bundled test snippets against it, e.g. checking `dpo_loss` against five hard-coded tensors (an easy pair, a hard pair, a gradient-flow check, a closed-form comparison, and a beta-scaling check for the DPO task specifically) [7][10]. There is no batch size, GPU count, or generation layout to configure, and no OOM first aid is documented, because no full model is ever loaded during a check.

## Watch it

Not applicable - there is no training loop, so there are no per-step reward, KL, or loss curves of the kind this section covers on a trainer card. What a learner sees is pass/fail per test case, described in the README as "instant feedback": submit and see pass/fail per test case in seconds [1]. The README also lists "print output capture" (2026-04-10 changelog entry): `print()` statements inside a submitted function now show up in the test results [1]. Progress (solved count, attempt history) is persisted across sessions in a local SQLite database at the configurable `DB_PATH` (default `./data/pyre.db`) [1]. No RL-specific metric names, logging backend integration (e.g., Weights & Biases, TensorBoard), or published stopping/health threshold is documented anywhere read for this card, because none of the four alignment tasks runs an actual optimization loop over steps.

## Save it

Not applicable - nothing here trains or checkpoints a model. The only persisted state is the learner's own progress: a SQLite database at `DB_PATH` (default `./data/pyre.db`) storing solved counts and submission/attempt history per problem, configurable via `web/.env.local` [1]. When run through Docker Compose, that progress is kept in a named Docker volume, and `docker compose down -v` is documented as the way to reset it [1]. There is no model checkpoint directory, no adapter/full-model distinction, and no loader handoff, because no model weights are ever produced by using this tool.

## Find it in the docs

There is no separate hosted documentation site for this project; the README (English at `README.md`, Chinese at `README_CN.md`) is the whole of the documentation read for this card [1]. Within the repository: the problem catalogue and learning-path table live in the README's "Problem Set" section; the environment-variable table (`GRADING_SERVICE_URL`, `DB_PATH`) lives in its "Configuration" section; and the top-level directory layout (`web/`, `grading_service/`, `torch_judge/`) is given in its "Project Structure" section [1]. To read a specific alignment task's own definition, test cases, hint, and reference solution, open the corresponding file under `torch_judge/tasks/` in the repository - e.g. `torch_judge/tasks/dpo_loss.py`, `grpo_loss.py`, `ppo_loss.py`, `reward_model.py` - each of which is a self-contained Python dict with `description_en`/`description_zh`, `hint`, `tests`, and `solution` fields [10][11][12][13]. The README credits TorchCode by duoan as the origin of the judge engine and problem set [1]; that upstream project was not fetched for this card and is named only as the acknowledged source. No official MCP endpoint or community tutorial curation page is documented for this project in the sources read for this card.

Honest boundary, stated where a reader would hit it: anyone arriving at this repository expecting a DPO/GRPO/PPO/reward-model trainer - the kind covered by the other cards in this deck - will not find one; the four exercises named after those methods are single-function, unit-tested coding problems meant to be solved by hand, with a bundled reference `solution` string but no code path that fine-tunes a real language model [10][11][12][13].

## Sources

[1] pyre-code README (English). https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/README.md. Fetched 2026-08-12, read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e.

[2] pyre-code repository metadata (owner login). https://api.github.com/repos/whwangovo/pyre-code. Fetched 2026-08-12.

[3] pyre-code GitHub repository. https://github.com/whwangovo/pyre-code. Fetched 2026-08-12.

[4] pyproject.toml (Python floor, dependency list, declared license, package author field). https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/pyproject.toml. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[5] LICENSE file text (MIT-style text with duoan/kiren copyright lines). https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/LICENSE. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[6] Git commit metadata for commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e (commit author name "kiren"). https://api.github.com/repos/whwangovo/pyre-code/commits/7ede95b47437798c74023a0ffae8fb33bbeebc2e. Fetched 2026-08-12.

[7] grading_service/main.py, the FastAPI backend's `_execute_tests` function that `exec()`s a submission and its bundled test snippets. https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/grading_service/main.py. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[8] torch_judge/engine.py, defining the `check(task_id)` Jupyter-notebook helper (unused by the FastAPI web backend specifically). https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/torch_judge/engine.py. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[9] torch_judge/__init__.py, the package's top-level module that re-exports `check`/`hint` from `engine.py` and documents them as the Jupyter-notebook usage pattern in its module docstring. https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/torch_judge/__init__.py. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[10] torch_judge/tasks/dpo_loss.py. https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/torch_judge/tasks/dpo_loss.py. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[11] torch_judge/tasks/grpo_loss.py. https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/torch_judge/tasks/grpo_loss.py. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[12] torch_judge/tasks/ppo_loss.py. https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/torch_judge/tasks/ppo_loss.py. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[13] torch_judge/tasks/reward_model.py. https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/torch_judge/tasks/reward_model.py. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[14] package.json (Node devDependency, npm scripts). https://raw.githubusercontent.com/whwangovo/pyre-code/7ede95b47437798c74023a0ffae8fb33bbeebc2e/package.json. Read at commit 7ede95b47437798c74023a0ffae8fb33bbeebc2e, fetched 2026-08-12.

[15] GitHub license-detection API result for this repository (NOASSERTION). https://api.github.com/repos/whwangovo/pyre-code/license?ref=7ede95b47437798c74023a0ffae8fb33bbeebc2e. Fetched 2026-08-12.
