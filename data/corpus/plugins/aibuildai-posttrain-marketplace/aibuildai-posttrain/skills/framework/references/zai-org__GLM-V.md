# GLM-V

The GLM-V repository is a vision-language model release, not an RL trainer; the one post-training-relevant component it ships is `glmv_reward`, a CPU-side reward/verifier library that plugs into an external RL loop.

GLM-V "contains our `GLM-4.6V`, `GLM-4.5V` and `GLM-4.1V` series models" [1], built and maintained by Zai (formerly Zhipu AI); the reward system's own README names its "Related Project" as `THUDM/GLM-4.1V-Thinking` [2], and a maintainer's issue reply separately points there for the RL-related code [3]. Bundled inside it is the **VLM Reward System** (package name `glmv_reward`), described in its own README as "a core component of our vision-language model training infrastructure" that "powers the reinforcement learning training of our GLM-4.1V series models" and is built to "Works with any RL training pipeline" [4]. Its API is a single class, `RewardSystem`, loaded from a YAML config, whose `get_reward(prompts, answers, gt_answers, datasources, ...)` method returns a list of float scores [4][5]. No model-training trainer class, launcher, or checkpoint format ships anywhere in this repository — GLM-V's own README says model fine-tuning goes through the third-party LLaMA-Factory project instead [1]. It lives at https://github.com/zai-org/GLM-V [6].

**When to pick it**: not a general post-training framework choice at all — pick `glmv_reward` specifically when you need the same rule-based-plus-LLM-judge verifiers (math, chart, OCR, VQA, GUI-agent, ...) that Zai used to reward-score rollouts for GLM-4.1V-Thinking, and you already have your own RL trainer (e.g. verl, trl, or a custom loop) to consume the scores it returns; it ships no trainer, no distributed launcher, and no GPU code, so it cannot replace verl/trl for running the actual policy update. Whole-model fine-tuning (SFT, not RL) on GLM-4.5V/GLM-4.1V-9B-Thinking is documented as going through the external LLaMA-Factory project instead [1].

**Methods it ships**: one, REWARD — the `glmv_reward` verifier system. It registers 16 `verifier_type` values: `math`, `general`, `biology`, `chemistry`, `physics`, `geography`, `geoquest`, `liberal_arts`, `chart`, `ocr`, `mmsi`, `multi_image`, `vqa`, `counting`, `language_mix`, and `file_based`, plus three agent-evaluation aliases (`androidworld`, `osworld`, `webvoyager`) that resolve to the same `FileBasedVerifier` class — 19 registry entries in total [7]. The math verifier combines symbolic checking (via `sympy`) with an LLM-judge fallback; the README documents this general pattern as "Hybrid Verification: Combines rule-based verifiers with LLM-as-a-judge" [4]. No method here is experimental/stable-tagged the way a trainer library tags its trainers — there is no such taxonomy published; the source of truth for the verifier list is `src/glmv_reward/verifiers/__init__.py` at commit 726dac56ddde6d33f72bd62967322e15f61a8471 [7]. The RL algorithm this reward system was built to feed — RLCS (Reinforcement Learning with Curriculum Sampling) — is named in the top-level README as the training paradigm behind GLM-4.1V-Thinking, but its own trainer/GRPO code is not published in this repository; a maintainer confirmed this directly (see Find it in the docs) [1][3].

**Scale it handles**: single process only, no distributed launcher of any kind. Internally, `RewardSystem.get_reward()` fans a batch of prompt/answer/ground-truth triples out over a `ThreadPoolExecutor` sized `min(128, len(prompts))` — I/O-bound concurrency for the LLM-judge HTTP calls, not GPU parallelism [5]. The package has no `torch` dependency at all [8], so it runs on CPU; scaling a reward-scoring workload beyond one host is left entirely to whatever RL framework calls it, and no such mechanism is documented in this repository.

**Install**: no PyPI package exists for `glmv_reward` — a PyPI JSON-API lookup for `glmv-reward` and `glmv_reward` both returned HTTP 404 on 2026-08-11 — so the only install path is from source: `git clone https://github.com/zai-org/GLM-V && cd GLM-V/glmv_reward && pip install -e .` [4]. Read at the pinned commit 726dac56ddde6d33f72bd62967322e15f61a8471 (the repository's most recent push at scan time, matching `last_push` 2026-07-21T16:03:29Z; there is no GitHub release or tag to resolve against — the repo has zero releases and zero tags) [9]: Python `>=3.10`; version `0.1.0`; dependencies pinned with `~=` compatible-release ranges — `numpy~=2.2`, `sympy~=1.14`, `openai~=1.95`, `pillow~=11.3`, `requests~=2.32`, `msgspec~=0.19.0`, `ruamel-yaml~=0.18.14`, `editdistance~=0.8.1` — and no `torch`, `transformers`, or CUDA dependency anywhere in this subpackage [8]. Licence is Apache-2.0 for the whole GLM-V repository [6]. No CUDA/GPU/hardware minimum is stated anywhere for `glmv_reward`, consistent with it having no GPU dependency [8].

**Maintained by**: `glmv_reward` is developed by the CogVLM Team at Zhipu AI (now Zai), per its own README's "About" section [2]; the wider GLM-V repository is maintained by Zai/Zhipu AI staff, evidenced by MEMBER-associated replies from the same maintainer account across multiple issues [3][10]. Dated signs of life: the repository's most recent push was 2026-07-21T16:03:29Z [6], and the top-level README's own project-update timeline records the reward system's open-sourcing on 2025-07-16 [1].

## Quick start

From the `glmv_reward` README's own quick-start steps [4]:

```bash
cd glmv_reward
pip install -e .
export ZHIPUAI_API_KEY='your_api_key_here'
cp examples/configs/example.yaml.template examples/configs/example.yaml
# edit example.yaml, then:
python examples/reward_system_demo.py
```

The demo script's core call, reduced to the reward-scoring step itself [5][11]:

```python
from glmv_reward import RewardSystem

reward_system = RewardSystem("examples/configs/example.yaml")

rewards = reward_system.get_reward(
    prompts=["What is 15 + 27?"],
    answers=["<think>15 + 27 = 42</think><answer><|begin_of_box|>42<|end_of_box|></answer>"],
    gt_answers=["<think>15 + 27 = 42</think><answer><|begin_of_box|>42<|end_of_box|></answer>"],
    datasources=["math"],
)
print(rewards[0])  # 1.0 for a correct answer
```

## Start it

- One process is the only supported form — there is no `accelerate launch` / `torchrun` / Ray equivalent here; `RewardSystem(config_file)` is instantiated once per process and `get_reward()` is called per batch, internally parallelized across up to 128 threads for the LLM-judge calls [5].
- Config surface is a YAML file mapped onto `RewardSystemConfig`: `reward_log_dir` (default `"logs"`), `datasource_reward_config_mapping` (datasource name → named verifier config), `reward_configs` (named verifier configs), and `enable_mix_verifier` (default `True`, turns on a language-mixing check applied before any verifier runs) [12]. Each verifier config also carries its own `llm_api_key` / `llm_judge_url` / `llm_model` / `llm_max_tokens` / `llm_temperature` / `llm_top_p` / `llm_judge_prompt_template` fields when an LLM-judge fallback is enabled, as shown in both the minimal math-only template and the larger `configs/full_config.yaml` reference, which wires up 17 datasource names in total — including `ocr_ignore_case` as a distinct entry from `ocr`, and the agent tasks `AndroidWorld`, `WebVoyager`, and `OSWorld` [13][14].
- Answer-format gate: before any verifier runs, `RewardSystem` requires the model answer and ground truth to match a `^<think>...</think>\s*<answer>...</answer>$` pattern, with at most one `<|begin_of_box|>`/`<|end_of_box|>` pair and no legacy `\boxed{}` syntax inside the answer span; anything that fails this check is scored at the verifier's `min_reward` (default negative infinity, later clamped) rather than judged [5].
- There is no GPU, no batch/micro-batch size to tune, and no generation engine here, so the library's own out-of-memory first aid does not apply; the only capacity knob documented is the thread-pool cap of 128 concurrent workers in `get_reward()` [5].

## Watch it

This section covers only how `glmv_reward` emits data — whether a given reward number is healthy for training is a question for the RL trainer's/method's own card, not this one.

- No metrics backend (no W&B/TensorBoard/trackio integration) is wired into this package; logging goes through Python's standard `logging` module via a thin `get_logger()` wrapper that returns a logger truncated to two name segments, with no handler or formatter configured by the package itself — messages are only visible if the calling process configures logging [15].
- Sample-level logging of rollouts is opt-in via `get_reward(..., log_reward_judge=True, save_dir=..., current_iteration=N)`. It writes newline-delimited JSON under `<save_dir or reward_log_dir>/<datasource>/`, in four files split by outcome: `rollout_reward_pass@k.jsonl` / `rollout_reward_not_pass@k.jsonl` (split on whether any reward in the batch exceeds 0.75) and `rollout_reward_correct.jsonl` / `rollout_reward_incorrect.jsonl` (split per-item on `reward > 0`). Each line carries `current_iteration`, `prompt`, `image_file`, `answer`, `gt_answer`, `reward`, `answer_token_length`, `reward_sum_of_this_prompt`, and `uuid` [5].
- No evaluation-during-training loop is published here — `get_reward()` is a stateless scoring call the caller invokes from inside its own eval loop; `glmv_reward` has no scheduling concept of its own.
- Stopping rule: no threshold, patience, or stopping criterion is published for `glmv_reward`. Search run 2026-08-11 over the two pages that would carry one — the package README [4] and `reward_system.py` itself [5] — returns no early-stopping or convergence-threshold field; the only fixed numeric cutoff in the source is the pass@k split point (`reward > 0.75`), which is a logging-bucket boundary, not a stopping rule [5].

## Save it

`glmv_reward` writes no model weights and defines no checkpoint format — it is a scoring function, not a trainer, so there is nothing here analogous to a checkpoint directory. The only artifacts it writes to disk are the reward-judgment JSONL logs described above, gated behind `log_reward_judge=True` [5]. Saving, resuming, and reloading the policy model being trained is entirely the responsibility of whichever external RL framework calls `get_reward()`; that framework's own card covers its checkpoint/resume/loader contract. GLM-V's own README documents a separate, unrelated save path for SFT fine-tuning through LLaMA-Factory, which is out of scope for this reward-system component [1].

## Find it in the docs

There is no hosted documentation site for `glmv_reward` or GLM-V — everything lives in the GitHub repository itself, so "the docs" means these files, read at commit 726dac56ddde6d33f72bd62967322e15f61a8471 [9]:

- Top-level `README.md` (and `README_zh.md`) at the repo root: model downloads, serving frameworks, and the LLaMA-Factory fine-tuning pointer [1].
- `glmv_reward/README.md` (and `README_zh.md`): the reward system's own quick start, supported-verifier list, and a minimal config example [4].
- `glmv_reward/configs/full_config.yaml`: the full configuration reference the package README points to, wiring all 17 datasource names to named verifier configs with real (if placeholder) API keys and judge prompts — read this file directly for any config field, rather than the trimmed template [14].
- `glmv_reward/examples/configs/example.yaml.template` + `glmv_reward/examples/reward_system_demo.py`: the runnable end-to-end example, math-only [13][11].
- `glmv_reward/src/glmv_reward/verifiers/__init__.py` is the authoritative verifier registry (`_VERIFIER_REGISTRY`) — the README's prose list can drift from it, so check this file for the exact `verifier_type` strings a `datasource_reward_config_mapping` entry must resolve to [7].
- `glmv_reward/tests/`: per-verifier test directories (`chart`, `counting`, `geoquest`, `language_mix`, `mmsi`, `multi_image`, `ocr`, `vqa`), a `cogagent` directory covering the three agent verifiers, a `general` directory with five test files, a `stem` directory covering chemistry, physics, math, AND logic (`test_math_verifier.py`, `test_logic_verifier.py` alongside chemistry/physics), and a `utils` directory — runnable with `pytest tests/` — the closest thing to known-good smoke tests for each verifier [9].
- Trap (from a closed GitHub issue, maintainer reply, original in Chinese): asked directly whether the GRPO/RL training code behind RLCS would be released, a MEMBER-associated maintainer replied that there is no such plan for now, and pointed to `https://github.com/THUDM/GLM-4.1V-Thinking/tree/main/glmv_reward` as the publicly available RL-related code (issue #70, 2025-07-27) — i.e. `glmv_reward` is confirmed to be the entire public surface of the RL training stack, not a partial view of a larger published trainer [3]. A separate MEMBER reply in issue #5 (2025-07-02) said only that the team would open-source in the near future and was evaluating related open-source work — a general commitment, with no statement about sequencing relative to policy-training code [10].
- No official MCP endpoint for these docs was found or is claimed anywhere in the repository.

Honest boundary: this repository ships no RL trainer, no distributed launcher, and no GPU code for post-training at all. If what you need is the thing that actually updates GLM-V's weights via RL, it is not here and — per the maintainer reply above — is not planned to be published; `glmv_reward` only supplies the reward signal for a policy-training loop you must bring yourself.

## Sources

[1] GLM-V top-level README, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/README.md. Fetched 2026-08-11.

[2] glmv_reward README, "Related Project" link and "About" section naming the CogVLM Team at Zhipu AI as developer. Same fetch as [4].

[3] GLM-V GitHub issue #70 (MEMBER reply on RLCS/RL training code release). https://api.github.com/repos/zai-org/GLM-V/issues/70 and its comments endpoint. Fetched 2026-08-11.

[4] glmv_reward README, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/glmv_reward/README.md. Fetched 2026-08-11.

[5] glmv_reward `reward_system.py` source, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/glmv_reward/src/glmv_reward/reward_system.py. Fetched 2026-08-11.

[6] GLM-V GitHub repository / API record (name, licence, stars, push/creation dates, zero releases and zero tags). https://github.com/zai-org/GLM-V and https://api.github.com/repos/zai-org/GLM-V. Fetched 2026-08-11.

[7] glmv_reward verifier registry, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/glmv_reward/src/glmv_reward/verifiers/__init__.py. Fetched 2026-08-11.

[8] glmv_reward `pyproject.toml`, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/glmv_reward/pyproject.toml. Fetched 2026-08-11. Cross-checked against PyPI JSON API lookups for `glmv-reward` and `glmv_reward`, both HTTP 404, fetched 2026-08-11.

[9] GLM-V repository tree and release/tag listing at commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://api.github.com/repos/zai-org/GLM-V/git/trees/726dac56ddde6d33f72bd62967322e15f61a8471?recursive=1, https://api.github.com/repos/zai-org/GLM-V/releases, https://api.github.com/repos/zai-org/GLM-V/tags. Fetched 2026-08-11.

[10] GLM-V GitHub issue #5 (MEMBER reply on future open-sourcing plans). https://api.github.com/repos/zai-org/GLM-V/issues/5 and its comments endpoint. Fetched 2026-08-11.

[11] glmv_reward demo script, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/glmv_reward/examples/reward_system_demo.py. Fetched 2026-08-11.

[12] glmv_reward `RewardSystemConfig` struct, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/glmv_reward/src/glmv_reward/configs/reward_system.py. Fetched 2026-08-11.

[13] glmv_reward example config template, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/glmv_reward/examples/configs/example.yaml.template. Fetched 2026-08-11.

[14] glmv_reward full configuration reference, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/glmv_reward/configs/full_config.yaml. Fetched 2026-08-11.

[15] glmv_reward logging utility, commit 726dac56ddde6d33f72bd62967322e15f61a8471. https://raw.githubusercontent.com/zai-org/GLM-V/726dac56ddde6d33f72bd62967322e15f61a8471/glmv_reward/src/glmv_reward/utils/logging.py. Fetched 2026-08-11.
