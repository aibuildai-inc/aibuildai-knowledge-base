# OpenClaw-RL

A repository of RL recipes and launch scripts, built on a full vendored copy of THUDM's slime, that turns live OpenClaw agent conversations (or terminal/GUI/SWE/tool-call rollouts) into training signal for a policy model, with no PyPI package and no tagged releases.

**OpenClaw-RL** is "a fully asynchronous reinforcement learning framework that turns everyday conversations into training signals for personalized AI agents, and supports training general agents with large-scale environment parallelization" [1]. It is maintained under the Gen-Verse GitHub organization [2], with the accompanying technical report authored by Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, and Ling Yang [3]. The repository wraps a policy model behind an OpenAI-compatible proxy (via the OpenClaw agent client), and its top-level entry points are per-method shell scripts (e.g. `openclaw-rl/run_qwen3_4b_openclaw_rl.sh`) that call `python3 train_async.py` through Ray on top of a vendored slime + Megatron-LM + SGLang stack [1][4]. It lives at https://github.com/Gen-Verse/OpenClaw-RL [2].

**When to pick it**: pick this repo specifically to reproduce or extend the OpenClaw-RL paper's three methods (Binary RL/GRPO with PRM scoring, On-Policy Distillation, and their Hybrid combination) on top of slime, or to reuse its Track-2 scripts for terminal/GUI/SWE/tool-call agentic RL [1][5]. The paper's own reference run shows why the Hybrid method is the one this repo defaults to: measuring the minimum number of conversation sessions needed to reach the target personalization effect, averaged over 5 trials with Qwen3-4B-Thinking-2507, Hybrid RL needs about 10.3 sessions under joint multi-user optimization versus 14.1 for GRPO alone and 29.7 for OPD alone (15.0 vs 21.1 and 29.4 respectively under separate-user optimization) [6]. Do not pick it as a general-purpose RL library: it ships no installable package, no PyPI entry, and no GitHub releases or tags [7][8] - every method folder is a self-contained set of shell scripts and Python files layered on a full copy of slime and Megatron-LM checked into the repo, not imported as dependencies (`slime/` and `Megatron-LM/` are top-level directories in the tree, not submodule pointers) [9]. Choose slime directly instead if you want the general Megatron+SGLang RL framework without OpenClaw-RL's method-specific scripts.

**Methods it ships**: three method families, each its own top-level folder with its own README (method math and the defining paper are on the methodology cards, not restated here) [1][5][10]:
- `openclaw-rl/` - Binary RL: GRPO with PRM (Process Reward Model) scoring of the next conversational state, PPO-style clipped surrogate loss [10].
- (OPD folder, referenced from the root README as `openclaw-opd`) - On-Policy Distillation: a judge model extracts a hint from the next state, and the teacher-vs-student token-level log-prob gap becomes a per-token advantage [1].
- `openclaw-combine/` - Hybrid: a weighted sum of the GRPO and OPD per-token losses (`L_i = w_RL * L_i^GRPO + w_OPD * L_i^OPD`, both weights default to 1.0), with an "overlap-guided hint selection" step that picks the hint whose induced teacher top-k token set overlaps most with the student's, controlled by `OPENCLAW_TOPK_HINT_SELECTION` (`sequence_optimal` default, `token_optimal`, or `shortest`) and `OPENCLAW_TOPK_SUBSET_MODE` [5].
- Track 2 general-agent settings reuse GRPO (optionally PRM-scored) against different environments: `terminal-rl/` (Docker sandbox, camel-ai-based agent) [11], `gui-rl/`, `swe-rl/` (vendors mini-swe-agent), and `toolcall-rl/` (based on slime's Retool implementation, with its own SFT-then-RL pipeline over the ReTool-SFT and DAPO-Math-17k Hub datasets) [1][12].
- None of these are marked experimental or stable in the docs read for this card; the repo carries no taxonomy page, only the root README's roadmap checklist [1].

**Scale it handles**: single machine with 8 GPUs is the documented default for the personal-agent Track 1 path, with GPU counts split into `ACTOR_GPUS`/`ROLLOUT_GPUS`/`PRM_GPUS` and configurable via env vars [4][13]. Multi-node is documented with real published scripts: `toolcall-rl/retool_qwen25_32b_4nodes_rl.sh` (4 nodes, 32 GPUs) and a 5-node PRM variant that dedicates one full node to the PRM judge to avoid GPU contention with training [12]; `swe-rl/scripts/run_swe_rl_4b_4nodes_colocate.sh` is a working 4-node example, while the root README's documented 8-node SWE path (`swe-rl/run_swe_rl_32b_remote_8nodes.sh`) 404s at the pinned commit - treat the root README's Track-2 quick-start paths as stale and use the scripts under each method's own subtree [1]. All multi-node launches follow the same Ray pattern: `ray start --head` on a master node, `ray start --address=...` on each worker, then `ray job submit --address=http://<master>:8265` [14]. No throughput or scaling benchmark is published in the docs read for this card.

**Install**: `git clone https://github.com/Gen-Verse/OpenClaw-RL.git`, then follow `instructions/README.md`; there is no PyPI package, and `tags`/`releases` are both empty for this repository, so there is no version number or dated release to pin - only the commit [7][8]. The documented environment is `conda create --name openclaw-rl python=3.12`, CUDA 12.9 [15][4]. `instructions/README.md`'s pip sequence at the pinned commit installs `torch==2.9.1+cu129` from the PyTorch cu129 wheel index first, then `pip install -r requirements.txt`, a 293-line pip-freeze that load-bearingly pins `transformers==4.57.1`, `ray==2.54.0`, `wandb==0.25.0`, and an SGLang build from a pinned commit (`git+https://github.com/sgl-project/sglang.git@d566816d838ce92d3ae044209f7d67eaa58ce74a`) plus `sglang-router==0.3.2` [16][15]. After that, `instructions/README.md` builds six packages from source (via `pip install --no-build-isolation` or an equivalent local build) rather than installing them as prebuilt wheels: DeepEP, an `int4_qat` kernel package, NVIDIA apex, `flash-attn==2.7.4.post1`, `megatron-bridge` pinned to a specific git commit (`pip install "megatron-bridge @ git+..." --no-build-isolation`), and `transformer_engine[pytorch,core_cu12]==2.10.0`; one more package, `flashinfer-jit-cache==0.6.3`, is installed as a prebuilt wheel from the flashinfer cu129 wheel index rather than built locally [15]. Optionally upgrading to `transformers==5.3.0` for Qwen3.5 support is documented as contradicting the base `transformers==4.57.1` pin from `requirements.txt` [15]. Licence is Apache-2.0 [17]. No CUDA/GPU-model minimum is stated beyond "CUDA 12.9" in the setup doc and the `+cu129` wheel selection [15][4].

**Maintained by**: the Gen-Verse GitHub organization, with the technical report authors listed above [2][3]; the repository has no tagged releases, so "maintained" here means commit activity - the pinned commit is dated 2026-05-23, and the root README's own News list runs from a 2026-02-26 v1 release through Fireworks AI integration credited 2026-04-15 [8][18][1]. 5,626 GitHub stars are recorded by the API at fetch time, not used here as a ranking signal [8].

## Quick start

The root README's documented flow is: start the RL server, then point an OpenClaw client at it. For the Hybrid method [1]:

```bash
cd slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh
```

For Binary RL (GRPO) alone [10]:

```bash
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

Once running, the policy is served as an OpenAI-compatible API at `http://<HOST_IP>:30000/v1`; the README shows wiring this endpoint into an OpenClaw `openclaw.json` provider entry (model id, `baseUrl`, `apiKey` matching `SGLANG_API_KEY`) so that live conversations are proxied through the trained model and continuously optimized in the background [1]. There is no pip-installable quickstart - `run_qwen3_4b_openclaw_rl.sh` is the smallest complete run, and it expects `HF_CKPT`, `REF_LOAD`, and `SAVE_CKPT` to be set to real local model/checkpoint paths before it is invoked [4].

## Start it

- One machine, one GRPO run: `openclaw-rl/run_qwen3_4b_openclaw_rl.sh`, invoked from inside `slime/` as shown above. It defaults `NUM_GPUS=8` split as `ACTOR_GPUS=4`, `ROLLOUT_GPUS=2`, `PRM_GPUS=2` (all overridable env vars, and the script asserts their sum does not exceed `NUM_GPUS`) [4].
- The launcher itself starts Ray (`ray start --head --num-gpus "${NUM_GPUS}" --dashboard-port=8265`) and submits the training job with `ray job submit --address="http://127.0.0.1:8265" -- python3 train_async.py ...` [4]. Multi-node scenarios repeat this pattern across machines: `ray start --head` on the master, `ray start --address=<master>:6379` on each worker, then one `ray job submit` from the master, as shown in `swe-rl/scripts/run_swe_rl_4b_4nodes_colocate.sh` and in `toolcall-rl`'s 4- and 5-node scripts [14][12].
- Config templates live as one script per scenario, not a separate YAML layer: e.g. `openclaw-rl/run_qwen3_4b_openclaw_rl.sh`, `openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh`, `terminal-rl/terminal-rl_qwen3-8b.sh`/`terminal-rl_qwen3-8b_prm_2nodes.sh`, `toolcall-rl/retool_qwen3_4b_rl.sh`/`retool_qwen25_32b_4nodes_rl.sh`/`retool_qwen25_32b_prm_5nodes_rl.sh` [4][5][11][12].
- Effective batch is set by slime's Megatron-side args exposed on each script: `run_qwen3_4b_openclaw_rl.sh`'s `ROLLOUT_ARGS` sets `--rollout-batch-size 16`, `--n-samples-per-prompt 1`, `--num-steps-per-rollout 1`; `PERF_ARGS` additionally enables `--use-dynamic-batch-size --max-tokens-per-gpu 32768` rather than a fixed per-device batch size [4].
- Generation is a separate SGLang engine, not colocated by default in the layout diagram: `SGLANG_ARGS` sets `--rollout-num-gpus-per-engine 2`, `--sglang-mem-fraction-static 0.85`, `--sglang-context-length 32768` [4]. The GRPO block (`GRPO_ARGS`) sets `--eps-clip 0.2 --eps-clip-high 0.28 --entropy-coef 0.00 --disable-rewards-normalization`, and turns on `--use-kl-loss --kl-loss-coef 0.0` - the script's own KL coefficient is 0.0 (KL loss active but weighted to zero), which is a different value from the 0.02 that `openclaw-rl/README.md`'s method write-up gives for beta_KL; the two sources disagree and this card reports both rather than picking one [10][4].
- No precision default is stated as changed from slime's own default in the docs read for this card; the script instead sets `--attention-softmax-in-fp32` and `--accumulate-allreduce-grads-in-fp32` alongside `--attention-backend flash` [4].
- Out-of-memory first aid is not published as a dedicated troubleshooting section in the docs read for this card; the closest documented levers are the GPU-partition env vars above and the `--sglang-mem-fraction-static`/`--max-tokens-per-gpu` knobs already listed [4].

## Watch it

This section is the mechanics only; what a metric shape means for GRPO or OPD training health lives on those methods' own cards, not here.

- **Enable it**: logging is Weights & Biases only, and it is conditionally on: `run_qwen3_4b_openclaw_rl.sh` sets `USE_WANDB=${USE_WANDB:-1}` but only appends `--use-wandb --wandb-project ... --wandb-key ...` to the launch command when `USE_WANDB=1` AND a non-empty `WANDB_KEY`/`WANDB_API_KEY` is supplied; without a key, no logging backend is active even though `USE_WANDB` defaults on [4]. `terminal-rl` and `toolcall-rl` scripts take the same `WANDB_KEY` env var [11][12].
- **Metric names**: none of the fetched READMEs or launch scripts in this repository publish their own metric-name list; metrics are whatever the vendored slime training loop logs to W&B, so consult slime's own docs (out of scope for this card - see the slime methodology/framework card) rather than this repository [1][4].
- **Sample-level logging**: `openclaw-rl` records raw interaction traffic, not training-loop generations - `run_qwen3_4b_openclaw_rl.sh` sets `OPENCLAW_RECORD_ENABLED=1` by default and writes to `OPENCLAW_RECORD_FILE` (a `.jsonl` under `results/`), which is the proxy's request/response log rather than a completions sampler [4]. `openclaw-tinker`'s API server layer keeps its own "record management" per the architecture description, but the exact fields are not enumerated in the docs read for this card [19].
- **Evaluation-during-training**: `run_qwen3_4b_openclaw_rl.sh` declares an empty `EVAL_ARGS=()` array with no fields set, so this launch script runs no periodic evaluation out of the box; no eval cadence flags are documented elsewhere in the docs read for this card [4].
- **Stopping**: no RL-specific stopping rule, threshold, or patience value is published in any of the READMEs or launch scripts read for this card (root README [1], `openclaw-rl/README.md` [10], `openclaw-combine/README.md` [5], `openclaw-tinker/README.md` [19], `terminal-rl/README.md` [11], `toolcall-rl/README.md` [12]); `--num-rollout 100000000` in `run_qwen3_4b_openclaw_rl.sh` is a de facto "run forever" cap, not a convergence rule [4].

## Save it

- Local-GPU path (slime/Megatron): checkpoints are written via `--save "${SAVE_CKPT}" --save-interval 100` in the CKPT_ARGS block, in Megatron's own `torch_dist` sharded format, not directly an HF-loadable directory [4]. The repository ships separate conversion tools under `slime/tools/`: `convert_torch_dist_to_hf.py` and `convert_hf_to_torch_dist.py` (the two directions used by the quick-start scripts), plus `convert_fsdp_to_hf.py`, `convert_hf_to_fp8.py`, `convert_hf_to_int4.py`, `convert_hf_to_int4_direct.py`, `convert_to_hf.py`, and `merge_lora_adapter.py` [20]. `toolcall-rl/README.md` shows the `convert_hf_to_torch_dist.py` call pattern used before training starts: it takes the per-model `MODEL_ARGS` sourced from `slime/scripts/models/<model>.sh`, `--hf-checkpoint`, and `--save` [12]. None of the READMEs read for this card state what a reduced-retention flag (e.g. dropping optimizer state) does to resumability for the Megatron path - only the file inventory and conversion direction are documented here.
- No file-level layout of a `torch_dist` checkpoint directory (which file holds which shard) is enumerated in the docs read for this card; inspect one on disk before assuming a structure.
- Tinker cloud path (`openclaw-tinker/`): checkpointing is cloud-managed through the Tinker API, not local files. `trainer.py` calls `save_state_async(name=f"step_{step:04d}")` for full training-state checkpoints and `save_weights_and_get_sampling_client_async()` to push updated LoRA weights into a fresh sampling client for rollout [21][19]. `--save-interval`/`SAVE_INTERVAL` (default 20 steps) controls cadence, and `--resume-from-ckpt`/`RESUME_FROM_CKPT` takes a checkpoint path to resume from [19].
- The Tinker path trains LoRA adapters only ("Tinker only supports LoRA" per the root README's now-commented option-1 text, and `openclaw-tinker/README.md`'s `--lora-rank`/`LORA_RANK` default of 32) [1][19]; `slime/tools/merge_lora_adapter.py` exists in the vendored slime tree for merging an adapter into a full model, but its exact CLI contract is not documented in the pages read for this card [20].
- Loader handoff: whether an evaluator can load a saved local-GPU checkpoint directly depends on which conversion step was run - a raw `torch_dist` directory under `SAVE_CKPT` is NOT HF-loadable until passed through `convert_torch_dist_to_hf.py`; a Tinker LoRA checkpoint is not a standalone loadable model either until merged or paired with the base model via the sampling-client handoff Tinker itself provides. Neither path was verified end-to-end in the docs read for this card.

## Find it in the docs

There is no separate hosted documentation site for OpenClaw-RL; the docs ARE the repository - the root `README.md` plus one `README.md` per method folder, all reached only by browsing the GitHub tree at a given ref, since there are no release tags to build a versioned docs URL from [7][8].

- Address pattern: `https://github.com/Gen-Verse/OpenClaw-RL/blob/<ref>/<path>`, where `<ref>` is `main` (there is no other branch or tag documented) or a commit SHA for a pinned reading, e.g. `.../blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/openclaw-rl/README.md` [8][7].
- Page map by folder, each with its own `README.md`: `openclaw-rl/` (Binary RL/GRPO) [10], `openclaw-combine/` (Hybrid) [5], `openclaw-tinker/` (Tinker cloud path) [19], `terminal-rl/`, `gui-rl/`, `swe-rl/`, `toolcall-rl/` (Track 2 general-agent settings) [11][12][1]; the root README's `openclaw-opd` link and `openclaw-fireworks/README.md` reference are named in the README but were not independently fetched for this card [1]. Environment setup is `instructions/README.md` [15].
- Runnable references beyond the READMEs: `terminal-rl/data_utils/download.py seta_env` pulls the SETA-env terminal-agent dataset [11]; `toolcall-rl` names `JoeYing/ReTool-SFT`, `Qwen/Qwen3-4B-Instruct-2507`, `BytedTsinghua-SIA/DAPO-Math-17k`, and `zhuzilin/aime-2024` as its known-good Hub datasets/checkpoints, plus a pre-trained SFT checkpoint at `font-info/qwen3-4b-sft-SGLang-RL` to skip the SFT stage [12].
- Community layer: the root README links two community tutorial videos under its 2026/3/3 News entry, with no further curation page; the README gives no author attribution beyond the links themselves, so treat them as unvetted third-party content rather than a maintainer-curated tutorials page [1].
- No official MCP endpoint for these docs is named anywhere in the pages read for this card.
- Documented trap: the root README's own Track-2 quick-start command for SWE (`swe-rl/run_swe_rl_32b_remote_8nodes.sh`) returns a 404 at the pinned commit - confirmed by fetching that raw path directly, which returned a 14-byte "404: Not Found" body - so the working reference for multi-node SWE runs is `swe-rl/scripts/run_swe_rl_4b_4nodes_colocate.sh` instead, found by browsing the `swe-rl/scripts/` subtree rather than following the README [1][22].
- Honest boundary: this repository ships no evaluation-during-training defaults (`EVAL_ARGS=()` in the flagship script) [4], no published stopping rule, no PyPI package, and no versioned release - a reader who needs any of those must build them on top of what is here, not find them shipped.

## Sources

[1] OpenClaw-RL root README. https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/README.md. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[2] Gen-Verse GitHub organization (repository owner). https://github.com/Gen-Verse/OpenClaw-RL. Fetched 2026-08-10 via the GitHub API.

[3] OpenClaw-RL technical report abstract page, arXiv:2603.10165, "OpenClaw-RL: Train Any Agent Simply by Talking", authors Wang, Chen, Jin, Wang, Yang; submitted 2026-03-10, this version online 2026-05-11. https://arxiv.org/abs/2603.10165. Fetched 2026-08-10.

[4] `openclaw-rl/run_qwen3_4b_openclaw_rl.sh`. https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/openclaw-rl/run_qwen3_4b_openclaw_rl.sh. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[5] `openclaw-combine/README.md`. https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/openclaw-combine/README.md. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[6] OpenClaw-RL technical report, full text (Table 3, "Optimization efficiency of different methods across settings", and its Section 4.6 discussion), arXiv:2603.10165v2. https://arxiv.org/html/2603.10165v2. Fetched 2026-08-10.

[7] GitHub API tags and releases listings for the repository (both empty). https://api.github.com/repos/Gen-Verse/OpenClaw-RL/tags and https://api.github.com/repos/Gen-Verse/OpenClaw-RL/releases. Fetched 2026-08-10.

[8] GitHub API repository metadata (stars, pushed_at, default branch, topics, license, description, homepage). https://api.github.com/repos/Gen-Verse/OpenClaw-RL. Fetched 2026-08-10.

[9] Repository file tree at the pinned commit (`git/trees` API, recursive), confirming `slime/` and `Megatron-LM/` are checked-in directories rather than submodule pointers. https://api.github.com/repos/Gen-Verse/OpenClaw-RL/git/trees/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3?recursive=1. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[10] `openclaw-rl/README.md` (Binary RL/GRPO method description, loss formula, eps-clip and beta_KL values, run command, file structure). https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/openclaw-rl/README.md. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[11] `terminal-rl/README.md`. https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/terminal-rl/README.md. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[12] `toolcall-rl/README.md`. https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/toolcall-rl/README.md. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[13] Root README, "Personal Agent Optimization Quick Start" deployment-requirements section (8x GPU default, CUDA 12.9, Python 3.12). https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/README.md. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[14] `swe-rl/scripts/run_swe_rl_4b_4nodes_colocate.sh` (the working 4-node Ray launch pattern). https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/swe-rl/scripts/run_swe_rl_4b_4nodes_colocate.sh. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[15] `instructions/README.md` (full environment setup: conda, torch wheel, requirements.txt, from-source builds for DeepEP/apex/flash-attn/flashinfer/megatron-bridge/transformer_engine, Qwen3.5 transformers upgrade note). https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/instructions/README.md. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[16] `requirements.txt` (pip-freeze dependency pins: transformers, ray, wandb, torch, sglang git commit, sglang-router). https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/requirements.txt. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[17] `LICENSE` file (Apache License 2.0 full text). https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/LICENSE. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[18] Root README News section (dated entries from 2026-02-26 through 2026-04-15). https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/README.md. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[19] `openclaw-tinker/README.md` (Tinker cloud path: CLI flags, env vars, architecture, checkpoint cadence, resume flag, LoRA-rank default). https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/openclaw-tinker/README.md. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[20] Repository file tree at the pinned commit, `slime/tools/` subtree (checkpoint-conversion script names). https://api.github.com/repos/Gen-Verse/OpenClaw-RL/git/trees/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3?recursive=1. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[21] `openclaw-tinker/trainer.py` (Tinker API calls: `save_state_async`, `save_weights_and_get_sampling_client_async`, `load_state_async`). https://github.com/Gen-Verse/OpenClaw-RL/blob/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/openclaw-tinker/trainer.py. Read at commit f48ac358adf9873b5cb2210f1cb234a52ed8a8a3.

[22] Raw fetch of `swe-rl/run_swe_rl_32b_remote_8nodes.sh` at the pinned commit, returning a "404: Not Found" body, confirming the root README's documented path is stale. https://raw.githubusercontent.com/Gen-Verse/OpenClaw-RL/f48ac358adf9873b5cb2210f1cb234a52ed8a8a3/swe-rl/run_swe_rl_32b_remote_8nodes.sh. Fetched 2026-08-10.
