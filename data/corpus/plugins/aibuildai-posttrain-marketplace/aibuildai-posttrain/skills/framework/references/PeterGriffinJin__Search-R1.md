# Search-R1

A research codebase for one specific recipe: RL-training an LLM to interleave reasoning with search-engine calls, built as a frozen fork of an early veRL snapshot plus a retriever server - pick it to reproduce that recipe, not as a general post-training framework.

**Search-R1** is described by its own README as "a reinforcement learning framework designed for training reasoning-and-searching interleaved LLMs" - language models that learn to reason and call tools such as search engines in a coordinated manner [1]. It is authored by Bowen Jin and coauthors (Search-R1's own citation lists Jin, Zeng, Yue, Yoon, Arik, Wang, Zamani, Han) [1][2], and the README states it is "Built upon veRL" [1], vendoring a copy of that framework's Python package (still named `verl`, version `0.1`) directly inside the repository rather than depending on it as an external package [3][4]. Its API shape is a set of bash launcher scripts (`train_ppo.sh`, `train_grpo.sh`) that set environment variables and Hydra config overrides and invoke `python3 -m verl.trainer.main_ppo`, alongside a separate local or remote retriever HTTP server that the training loop calls during rollouts [1][5]. It lives at https://github.com/PeterGriffinJin/Search-R1 [6].

**When to pick it**: reproducing or extending the specific Search-R1 recipe - RL (PPO or GRPO) on a base LLM with a rule-based exact-match reward and an interleaved retriever call, on datasets like NQ/HotpotQA - when you want the paper's own scripts and wandb logs as a reference point [1][7]. Its vendored veRL snapshot pins `transformers<4.48` and `vllm<=0.6.3` [4], so it is not a vehicle for using current veRL features or model support; for that, or for a general-purpose RL post-training framework, use veRL itself (cross-reference; not covered here). Search-R1 is a single-repository research project, not a framework the way trl or veRL are - there is no separate methods taxonomy page or plugin system.

**Methods it ships**: the README states support for "different RL methods (e.g., PPO, GRPO, reinforce)" [1], and the repo ships runnable launcher scripts only for PPO (`train_ppo.sh`) and GRPO (`train_grpo.sh`), both invoking the same `verl.trainer.main_ppo` entry point with `algorithm.adv_estimator` set to `gae` or `grpo` respectively [5]. A vendored `SFTTrainer` (`verl/trainer/fsdp_sft_trainer.py`) and its Hydra config (`verl/trainer/config/sft_trainer.yaml`) exist in the tree from the underlying veRL fork, but the README and quickstart never mention SFT - it is present as unused inherited code, not a documented Search-R1 workflow [1][4]. A vendored Megatron-strategy PPO config (`verl/trainer/config/ppo_megatron_trainer.yaml`, `strategy: megatron`) and a vendored `RewardModelWorker` (`verl/workers/reward_model/megatron/reward_model.py`) also exist, but `main_ppo.py` loads `config_name='ppo_trainer'` - the FSDP config - by default, and that config sets `reward_model.enable: False` [4][8]. The reward Search-R1 actually trains with is a rule-based exact-match scorer, `qa_em.compute_score_em`, selected by dataset name (`nq`, `triviaqa`, `popqa`, `hotpotqa`, `2wikimultihopqa`, `musique`, `bamboogle`) in `verl/trainer/main_ppo.py` [9] - the vendored reward-model worker is dead code for the documented workflow.

**Scale it handles**: single node (the quickstart and `train_ppo.sh`/`train_grpo.sh` default to `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, i.e. 8 GPUs, `trainer.n_gpus_per_node=8`, `trainer.nnodes=1`) up to multi-node via Ray, documented in `docs/multinode.md` with a head-node/worker-node `ray start` walkthrough and `ray job submit --runtime-env=verl/trainer/runtime_env.yaml` [5][10]. The repo ships an actual multi-node reference: `example/multinode/train_ppo_multinode_32b.sh` launches a Qwen2.5-32B PPO run with `N_NODES=4` [11], and the maintainer states in a closed issue this configuration runs in practice: "I can successfully run the 32B LLM training with four 8*H100 nodes, each with 80GB memory" (issue #106, PeterGriffinJin, OWNER, 2025-05-25) [12] - a published reference point, not a formal benchmark table.

**Install**: no PyPI package and no GitHub Releases or tags exist for this repository (both the Releases and Tags API endpoints return empty) [13][14], so the only install path is from source at a commit; the README's own environment recipe is `conda create -n searchr1 python=3.9`, then `pip install torch==2.4.0 --index-url .../cu121`, `pip3 install vllm==0.6.3` (older 0.5.4/0.4.2/0.3.1 also noted as installable), `pip install -e .` (this installs the vendored package, named `verl`, version `0.1`, from `setup.py`/`pyproject.toml` in this repo, not the upstream veRL package), then `pip3 install flash-attn --no-build-isolation` and `pip install wandb` [1][3][4]. Apache-2.0 licence [15]. `requirements.txt` at this commit pins `transformers<4.48`, `vllm<=0.6.3`, `tensordict<0.6`, with `accelerate`, `ray`, `hydra-core`, `flash-attn` unpinned [3]; `pyproject.toml` additionally states `requires-python = ">=3.8"` even though the README's own conda recipe uses 3.9 [4]. A separate, optional retriever conda environment is documented for local dense/sparse search (`transformers`, `datasets`, `pyserini`, `faiss-gpu=1.8.0`, `uvicorn`, `fastapi`) [1]. No CUDA or GPU-memory minimum is stated in the README or install docs; the closest published numbers are from maintainer issue replies (see Start it).

**Maintained by**: Bowen Jin (GitHub handle PeterGriffinJin) and coauthors [1][2]; the repository is not archived, its most recent push is 2025-11-13, and its README News section is updated as late as October 2025 (a note that Search-R1 is used in Thinking Machines Lab's Tinker cookbook) [6][1]. Community activity signs: as of this reading the repo has attracted 20+ downstream "Awesome work powered or inspired by Search-R1" entries linked from its own README, spanning 2025-03 through at least 2025-10 [1].

## Quick start

From the README's own NQ-with-e5-and-Wikipedia walkthrough, run in order [1]:

```bash
# (1) Download the indexing and corpus
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz

# (2) Process the NQ dataset
python scripts/data_process/nq_search.py

# (3) Launch a local retrieval server
conda activate retriever
bash retrieval_launch.sh

# (4) Run RL training (PPO) with Llama-3.2-3b-base
conda activate searchr1
bash train_ppo.sh
```

Inference with a trained checkpoint, also from the README [1]:

```bash
conda activate retriever
bash retrieval_launch.sh
# in another shell:
conda activate searchr1
python infer.py   # edit the `question` variable on line 7
```

## Start it

- One node: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_ppo.sh` (or `train_grpo.sh`) is the default, hard-coded to 8 GPUs in the script itself; edit the exported `CUDA_VISIBLE_DEVICES` and `trainer.n_gpus_per_node` override to change device count [5].
- Multiple nodes go through Ray: start a head node with `ray start --head --dashboard-host=0.0.0.0`, register worker nodes with `ray start --address=<gcs-address>`, launch the retrieval server on every node, then from the head node run `ray job submit --address=<dashboard-address>:8265 --runtime-env=verl/trainer/runtime_env.yaml -- python3 -m verl.trainer.main_ppo ...` with `trainer.nnodes=$N_NODES` [10]. Config templates for this are the example scripts themselves - `example/multinode/train_ppo_multinode_32b.sh` and `train_ppo_multinode_72b.sh`, `train_grpo_multinode_32b.sh` [11] - there is no separate Accelerate/DeepSpeed-style YAML template layer; the launch command is the template.
- Effective batch: `data.train_batch_size=512` split into `actor_rollout_ref.actor.ppo_mini_batch_size=256` PPO-update chunks, each further split into `ppo_micro_batch_size=64` per-forward-pass chunks in the quickstart scripts [5]; GRPO additionally samples `actor_rollout_ref.rollout.n_agent=5` completions per prompt for the group [5]. `data.max_prompt_length` (4096 in the quickstart) is derived from `max_start_length + max_response_length*(max_turns-1) + max_obs_length*max_turns`, per a comment in `train_ppo.sh` itself [5].
- Config surface is the vendored `verl/trainer/config/ppo_trainer.yaml`, loaded by `main_ppo.py`'s `@hydra.main(config_path='config', config_name='ppo_trainer')` [4][8]; the Megatron-strategy sibling config (`ppo_megatron_trainer.yaml`) is not the one this entry point loads [4][8]. Search-R1-specific fields on top of veRL's own PPO fields: `max_turns` (rollout/search turn budget, default 10 in the yaml, overridden to 2-4 in the launcher scripts), `do_search: true`, and `retriever.url` / `retriever.topk` [4]. The yaml's own `algorithm` block sets `gamma: 1.0` and `lam: 1.0` explicitly [4] - unchanged GAE defaults, not a Search-R1-specific tuning.
- Out-of-memory first aid is not published as a docs page but recurs consistently in maintainer replies to closed issues: lower `actor_rollout_ref.actor.ppo_micro_batch_size`, `actor_rollout_ref.rollout.log_prob_micro_batch_size`, and `actor_rollout_ref.ref.log_prob_micro_batch_size` (issue #45 and #62, PeterGriffinJin, OWNER) [16][17]; the retrieval server itself can OOM independently of training and the maintainer recommends 80GB VRAM to launch it (issue #45) [16]. In a separate issue the maintainer attributes a mid-training GPU memory spike to GRPO instability causing runaway-long generations, and recommends switching from GRPO to PPO for more stable memory behavior on Qwen-base models, or using a smaller model on 32GB GPUs (issue #41, PeterGriffinJin, OWNER, 2025-03-25 to 2025-03-27) [18].

## Watch it

This section covers only the mechanics of what is logged and how to turn it on; what a given metric's value or trend means for PPO/GRPO training health lives on that method's own methodology card.

- **Enable it**: the vendored `Tracking` class (`verl/utils/tracking.py`) supports three backends - `wandb`, `mlflow`, `console` - selected via `trainer.logger` in the Hydra config [19]. The yaml default is `logger: ['console', 'wandb']` [4], but both quickstart launcher scripts override it to `trainer.logger=['wandb']` only [5] - following the README's own scripts as written, a run with `WANDB_API_KEY` unset will still call `wandb.init` and log nothing to console.
- **Metric names**: `compute_data_metrics` in `verl/trainer/ppo/ray_trainer.py` emits, every logged step: `critic/score/{mean,max,min}`, `critic/rewards/{mean,max,min}`, `critic/advantages/{mean,max,min}`, `critic/returns/{mean,max,min}`, and (only `use_critic=True`, i.e. PPO) `critic/values/{mean,max,min}` and `critic/vf_explained_var`; `response_length/{mean,max,min,clip_ratio}` and `prompt_length/{mean,max,min,clip_ratio}` [20]. On top of these inherited veRL fields, Search-R1 adds its own interleaved-search metrics when the corresponding rollout metadata is present: `env/number_of_actions/{mean,max,min}`, `env/finish_ratio`, `env/number_of_valid_action`, `env/ratio_of_valid_action`, and `env/number_of_valid_search` [20].
- **Sample-level generations**: within `ray_trainer.py` as read for this card, no dedicated flag or call was found that logs raw generated text to the tracker (contrast with trl's `log_completions`); the closest is the reward function's own `num_examine`/`already_print_data_sources` console-print guard in `main_ppo.py`, which prints the first few scored examples per data source to stdout, not to wandb [9].
- **Evaluate during training**: validation runs on `val_reward_fn` when `trainer.test_freq > 0` and `global_steps % trainer.test_freq == 0`, logged under the same `logger.log(data=val_metrics, ...)` call used for training metrics [21]; `test_freq=50` in the quickstart scripts [5]. There is no separate held-out-eval config surface beyond `data.val_files` and `trainer.test_freq`.
- **Stopping**: no early-stopping, patience, or reward-threshold field was found in `ppo_trainer.yaml`, `ray_trainer.py`, or the README/docs pages fetched for this card (search performed 2026-08-10); the loop's only stopping conditions are `trainer.total_epochs` and `trainer.total_training_steps`, both fixed at construction time [4][21]. Shapes to watch (the `env/*` and `critic/*` names above) are published; a stopping threshold on any of them is not.

## Save it

- `_save_checkpoint` in `ray_trainer.py` writes to `{trainer.default_local_dir}/actor/global_step_{N}/` (and, when a critic is used, `.../critic/global_step_{N}/`), calling each worker's own `save_checkpoint(local_path, hdfs_path)` [22].
- Inside the FSDP actor and critic workers (`verl/workers/fsdp_workers.py`), `save_checkpoint` gathers a full (unsharded) state dict and calls `actor_module.save_pretrained(local_path, state_dict=state_dict)` plus `tokenizer.save_pretrained(local_path)` [23] - so a `global_step_N/` directory is a standard Hugging Face model directory (weights + tokenizer files), directly loadable via `from_pretrained`. Neither the actor nor the critic `save_checkpoint` call saves optimizer or scheduler state at all [23] - this is not a retention flag that can be toggled, it is how every checkpoint is written in this fork.
- Correspondingly, `ray_trainer.py` hard-codes `self.global_steps = 0` at training-loop setup [22] and, within the file as read for this card, contains no call that loads a previous checkpoint or step count back in - there is no PPO/GRPO resume-from-checkpoint mechanism in this fork's trainer loop, unlike the vendored (but unused) SFT trainer's config, which does carry a `trainer.resume_path` field [4]. Treat every `bash train_ppo.sh` / `train_grpo.sh` invocation as starting fresh from `actor_rollout_ref.model.path`.
- No `push_to_hub` or Hub-upload call was found in the files read for this card; `hdfs_path`, when `trainer.default_hdfs_dir` is set, is the only built-in remote-copy path (via `hdfs_io.copy`) [22][23].
- No PEFT/LoRA saving path applies to the PPO/GRPO trainers read for this card - `lora_rank` appears only in the unused vendored `sft_trainer.yaml` [4].
- Loader handoff: because `global_step_N/` is a plain HF `from_pretrained`-loadable directory with no optimizer state, an evaluator can load it directly with `transformers.AutoModelForCausalLM.from_pretrained(local_path)` - there is no adapter-merge step to reason about here, since nothing in the read scope saves adapters.

## Find it in the docs

There is no separate hosted docs site for Search-R1: the README (`README.md`) and three files under `docs/` (`docs/retriever.md`, `docs/multinode.md`, `docs/experiment_log.md`) on the `main` branch are the whole of the documentation, addressed as ordinary GitHub blob URLs, e.g. `https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md` - verified working at this fetch [1][7][10]. Because there are no tags or releases, `main` (or a specific commit SHA in place of `main`) is the only version selector; there is no `v<X.Y.Z>` form to guess [13][14].

- Retriever choice and setup (local sparse/BM25, local dense/e5 with flat or ANN indexing, or online Google/SerpAPI search) is `docs/retriever.md`, which also states a decision rule: private/domain corpus with no good embedding model -> BM25; private corpus with a good embedding model and enough GPUs -> flat e5; not enough GPUs -> ANN (HNSW64); general-purpose with budget -> SerpAPI over Google Search API, because "Google Search API has a hard 10k monthly quota" [7].
- Multi-node Ray setup is `docs/multinode.md`, and itself links out to veRL's own multinode guide for the base `ray start` mechanics [10].
- Experiment provenance (which wandb project corresponds to which paper/version) is `docs/experiment_log.md`, pointing at four separate wandb projects: `Search-R1-open` (preliminary), `Search-R1-nq_hotpotqa_train` (v0.1), `Search-R1-v0.2`, `Search-R1-v0.3` [24].
- Runnable references beyond the docs: `example/retriever/` (one launch script per retriever backend) and `example/multinode/` (PPO/GRPO at 32B and 72B) [11]; `scripts/download.py` and `scripts/data_process/nq_search.py` are the concrete data/index preparation used by the quickstart [1]; `example/corpus.jsonl` is a tiny worked example of the expected corpus JSONL shape [1].
- Community layer: the README curates a running list under "Awesome work powered or inspired by Search-R1" - among them DeepResearcher, ZeroSearch, StepSearch, SimpleTIR, and SkyRL, each linked with its own repo [1]; a 2025-06 README News entry also documents that Search-R1's own retrieval-tool pattern was folded into upstream veRL, pointed at from veRL's own docs (`sglang_multiturn/search_tool_example.html`) plus an external Chinese/English tutorial pair from a third-party "Awesome-ML-SYS-Tutorial" repo [1]. There is no official curated-tutorials page beyond this README section, and no MCP endpoint was found for this project.
- Traps found in closed issues, all replies from PeterGriffinJin as OWNER: retrieval-server OOM commonly needs ~80GB VRAM (#45, 2025-03-31) [16]; per-GPU-memory OOM during training is addressed by lowering the three micro-batch-size fields above (#45, #62) [16][17]; GRPO-specific training collapse (sudden very long generations, memory spikes) is a known instability the maintainer recommends addressing by switching to PPO on Qwen-base models or using a smaller model (#41) [18]; 4-node x 8xH100(80GB) is confirmed working for a 32B model (#106) [12].

## Sources

All GitHub file contents and API responses below were fetched 2026-08-10 at commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3` (the shortlist row's pinned commit, which is also this repository's most recent push - there is no separate Release or tag to resolve, so the commit pin and the "latest available" state coincide here) [13][14][6]. GitHub Issues content is live and dated per-comment as cited inline.

[1] Search-R1 README, at commit 598e61b. https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/README.md. Fetched 2026-08-10.

[2] Search-R1 arXiv citation block (author list), quoted within [1]. https://arxiv.org/abs/2503.09516 (link target named in [1], not separately fetched). Cited via [1].

[3] Search-R1 `requirements.txt`, at commit 598e61b. https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/requirements.txt. Fetched 2026-08-10.

[4] Search-R1 `setup.py`, `pyproject.toml`, `verl/version/version`, and the Hydra configs `verl/trainer/config/ppo_trainer.yaml` and `verl/trainer/config/sft_trainer.yaml`, all at commit 598e61b. https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/{setup.py,pyproject.toml,verl/version/version,verl/trainer/config/ppo_trainer.yaml,verl/trainer/config/sft_trainer.yaml}. Fetched 2026-08-10.

[5] Search-R1 `train_ppo.sh` and `train_grpo.sh`, at commit 598e61b. https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/{train_ppo.sh,train_grpo.sh}. Fetched 2026-08-10.

[6] Search-R1 GitHub repository (item home; also GitHub REST API repo metadata: `pushed_at`, `archived`, `license`, `stargazers_count`, `default_branch`). https://github.com/PeterGriffinJin/Search-R1. Fetched 2026-08-10.

[7] Search-R1 retriever guide, `docs/retriever.md`, at commit 598e61b. https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/docs/retriever.md. Fetched 2026-08-10.

[8] Search-R1 `verl/trainer/main_ppo.py`, at commit 598e61b (Hydra `config_name='ppo_trainer'`; `_select_rm_score_fn`). https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/trainer/main_ppo.py. Fetched 2026-08-10.

[9] Search-R1 `verl/trainer/main_ppo.py`, same file as [8] (rule-based EM reward selection and console print guard). Fetched 2026-08-10.

[10] Search-R1 multinode guide, `docs/multinode.md`, at commit 598e61b. https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/docs/multinode.md. Fetched 2026-08-10.

[11] Search-R1 `example/multinode/train_ppo_multinode_32b.sh`, and repo tree listing of `example/`, at commit 598e61b (GitHub Git Trees API, recursive). https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/example/multinode/train_ppo_multinode_32b.sh ; https://api.github.com/repos/PeterGriffinJin/Search-R1/git/trees/598e61bd1d36895726d28a8d06b3a15bed19f5d3?recursive=1. Fetched 2026-08-10.

[12] Search-R1 GitHub issue #106, comment by PeterGriffinJin (OWNER), 2025-05-25. https://github.com/PeterGriffinJin/Search-R1/issues/106. Fetched 2026-08-10.

[13] Search-R1 GitHub Releases API (empty list). https://api.github.com/repos/PeterGriffinJin/Search-R1/releases. Fetched 2026-08-10.

[14] Search-R1 GitHub Tags API (empty list). https://api.github.com/repos/PeterGriffinJin/Search-R1/tags. Fetched 2026-08-10.

[15] Search-R1 GitHub License API. https://api.github.com/repos/PeterGriffinJin/Search-R1/license. Fetched 2026-08-10.

[16] Search-R1 GitHub issue #45, comments by PeterGriffinJin (OWNER), 2025-03-26 to 2025-03-31. https://github.com/PeterGriffinJin/Search-R1/issues/45. Fetched 2026-08-10.

[17] Search-R1 GitHub issue #62, comment by PeterGriffinJin (OWNER), 2025-04-07. https://github.com/PeterGriffinJin/Search-R1/issues/62. Fetched 2026-08-10.

[18] Search-R1 GitHub issue #41, comments by PeterGriffinJin (OWNER), 2025-03-25 to 2025-03-27. https://github.com/PeterGriffinJin/Search-R1/issues/41. Fetched 2026-08-10.

[19] Search-R1 `verl/utils/tracking.py`, at commit 598e61b. https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/utils/tracking.py. Fetched 2026-08-10.

[20] Search-R1 `verl/trainer/ppo/ray_trainer.py`, at commit 598e61b (`compute_data_metrics`, `env/*` fields). https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/trainer/ppo/ray_trainer.py. Fetched 2026-08-10.

[21] Search-R1 `verl/trainer/ppo/ray_trainer.py`, same file as [20] (training loop: `global_steps`, `test_freq`, `save_freq`, `total_epochs`/`total_training_steps`). Fetched 2026-08-10.

[22] Search-R1 `verl/trainer/ppo/ray_trainer.py`, same file as [20] (`_save_checkpoint`). Fetched 2026-08-10.

[23] Search-R1 `verl/workers/fsdp_workers.py`, at commit 598e61b (`ActorRolloutRefWorker.save_checkpoint`, `CriticWorker.save_checkpoint`). https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/workers/fsdp_workers.py. Fetched 2026-08-10.

[24] Search-R1 experiment log, `docs/experiment_log.md`, at commit 598e61b. https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/598e61bd1d36895726d28a8d06b3a15bed19f5d3/docs/experiment_log.md. Fetched 2026-08-10.

Ecosystem tools named in passing (veRL, Ray, vLLM, FSDP, FAISS, BM25/pyserini, SerpAPI, Hydra, wandb) are reached only through the sources above and are deliberately not enumerated as separate references. Method names (PPO, GRPO) are deliberately cited to nothing here; their defining papers live on the methodology cards.
