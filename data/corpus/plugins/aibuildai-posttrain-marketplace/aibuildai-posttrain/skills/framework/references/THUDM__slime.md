# slime

An SGLang-native RL post-training framework: Megatron trains, SGLang generates, Ray glues the two together - config-driven bash launcher scripts, not a Python trainer-class API. Repository: https://github.com/THUDM/slime

**slime** is described in its own README as "an LLM post-training framework for RL scaling, providing two core capabilities: 1. High-Performance Training: Supports efficient training in various modes by connecting Megatron with SGLang; 2. Flexible Data Generation: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines" [1]. It is built and maintained by THUDM, a GitHub organization [2][3], with the BibTeX entry naming Zilin Zhu, Chengxing Xie, Xin Lv and "slime Contributors" as authors [1]. Its API shape is a set of shell scripts that assemble Megatron and SGLang command-line arguments plus slime-specific flags into a `python3 train.py ...` call, submitted to a Ray cluster with `ray job submit` [4] - there is no `Trainer.train()`-style Python class.

**When to pick it**: RL post-training at Megatron-training / SGLang-rollout scale, when the target model already has a Megatron checkpoint path or you need Megatron's tensor/pipeline/context parallelism for a large dense or MoE model, and you want the rollout engine's own SGLang argument surface exposed via a `--sglang-` pass-through prefix rather than wrapped [5]. It uses Ray to place actor and rollout workers, colocated on the same GPUs or disaggregated onto separate pools [6] - for a comparison against Ray-based alternatives (e.g. verl) or against Hugging Face's Accelerate-launched trainer classes (trl), weigh the sibling cards in this deck; that comparison is not carried by slime's own docs and is not repeated here.

**Methods it ships**: `--advantage-estimator` selects among `grpo`, `gspo`, `cispo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, and `ppo`, each documented with its defining arXiv id on the usage page (method math and training-signal semantics belong on each method's own card, not restated here) [7]. GRPO runs critic-free with group-relative advantages (`--n-samples-per-prompt`, `--normalize-advantages`); PPO adds a critic that requests its own GPU allocation via `--critic-num-nodes` / `--critic-num-gpus-per-node`, defaulting to the actor's topology if unset [7]. On-policy distillation is documented as orthogonal to the estimator choice, toggled with `--use-opd` and `--opd-kl-coef` [7]. SFT is supported as a rollout/data-generation mode (an SFT data-processing example lives at `examples/retool/sft_data_processing.py` and an SFT rollout path at `slime/rollout/sft_rollout.py`) [8]. No LoRA, adapter, or other parameter-efficient fine-tuning flag appears in the arguments module, the README, the quick-start guide, or the usage guide - the only mention of LoRA anywhere in the pages read for this card is that a third-party derivative, Miles, adds "LoRA, TITO, and low-precision training" on top of slime, implying slime itself trains full parameters only in the paths this card checked [1][18]; the low-precision, speculative-decoding, and beyond-Megatron architecture-support pages were not fetched for this card and were not checked for a LoRA path.

**Scale it handles**: single multi-GPU node up to multi-node, launched through Ray - `ray start --head --node-ip-address <addr> --num-gpus <n>` on the head node, `ray start --address=<addr>:6379 --num-gpus <n>` on each worker, then `ray job submit --address="http://127.0.0.1:8265" ... -- python3 train.py ...` [4]. The one worked example in the docs at v0.3.1 is 8xH100 for a 9B model, not a single-GPU run [9]. Actor/rollout placement is either disaggregated (`--actor-num-gpus-per-node`, `--rollout-num-gpus` on separate GPU pools) or colocated on the same pool via `--colocate` [6]; sharding is Megatron's own tensor/pipeline/context/sequence parallelism plus `--use-dynamic-batch-size` with `--max-tokens-per-gpu` for length-packed dynamic micro-batching [5]. Hardware support is documented per generation: H100/H200 carry official support with comprehensive CI testing and stable performance, B200 is fully supported with the same setup steps as H-series GPUs, B-series GPUs more broadly are called stable for development and testing but currently without CI coverage, and AMD is covered by a separate tutorial page not read for this card [10]. No published multi-node throughput benchmark accompanies the multi-node instructions - the section documents mechanism only [4].

**Install**: there is no working `pip install slime` - the PyPI package registered under the name `slime` is an unrelated statistical-modeling package (its listed summary describes a simple linear mixed-effects model with priors and constraints), not THUDM's framework [11]. The documented path is Docker-first: `docker pull slimerl/slime:latest`, run it with `docker run --rm --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 -it slimerl/slime:latest /bin/bash`, then inside the container `git clone https://github.com/THUDM/slime.git && cd slime && pip install -e . --no-deps` [4]. The card pins to release v0.3.1 (published 2026-08-06T11:38:46Z, tag resolves to commit `a6272da0d4f3d0a08520c99a2f3b4f6c887960dc`) [12][13]. `setup.py` at that commit sets `python_requires=">=3.10"` and lists no upper Python bound [14]; the package license is Apache-2.0 [2]. `requirements.txt` at that commit lists `sglang-router>=0.3.0` as its only pinned floor, alongside unpinned `accelerate`, `datasets`, `ray[default]`, `transformers`, `wandb`, `tensorboard`, and others - notably `torch`, `sglang`, and `megatron` are absent from `requirements.txt` because they ship inside the `slimerl/slime` Docker image or are built separately, not installed via this file or any pip extra (`extras_require` is empty in `setup.py`) [14][15]. No CUDA or torch version floor is stated in `setup.py` or `pyproject.toml` beyond a bare "Environment :: GPU :: NVIDIA CUDA" classifier [14][16]; the closest hardware floor is the docs' H100/H200/B200 support statement above [10].

**Maintained by**: THUDM, a GitHub organization [2][3][17]. Nine tagged releases from v0.1.0 through v0.3.1 as of this card, at an irregular cadence - a roughly three-month gap between v0.1.0 (2025-08-31) and v0.2.0 (2025-11-28), then gaps ranging from a few days to about nine weeks between the remaining releases, most recently v0.3.1 published 2026-08-06 [13]; 446 open issues at fetch time [17].

## Quick start

The smallest complete worked example in the docs at v0.3.1 is the GLM4-9B-on-8xH100 walkthrough, run inside the `slimerl/slime:latest` container [9]:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime/
pip install -e . --no-deps
```

```bash
# hf checkpoint
hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414
# train data
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
# eval data
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
```

```bash
# convert the HF checkpoint to a Megatron-loadable torch_dist checkpoint
cd /root/slime
source scripts/models/glm4-9B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist
```

```bash
cd /root/slime
bash scripts/run-glm4-9B.sh
```

That script's ROLLOUT_ARGS show the GRPO-style shape end to end: `--rm-type deepscaler`, `--num-rollout 3000`, `--rollout-batch-size 32`, `--n-samples-per-prompt 8`, `--rollout-max-response-len 8192`, one training step per rollout (`--num-steps-per-rollout 1`) [9].

## Start it

- Every run - one node or many - goes through Ray: the docs show no bare single-process launch form; even the single-node example is a `ray job submit ... -- python3 train.py ...` call [4][9].
- Actor and rollout placement: disaggregated by default (`--actor-num-nodes`, `--actor-num-gpus-per-node`, `--rollout-num-gpus` size separate pools), or colocated on one pool by adding `--colocate`, which by default makes the actor and rollout GPU counts equal unless `--rollout-num-gpus` is set explicitly [6]. In colocate mode the docs recommend `--sglang-mem-fraction-static 0.8` to leave headroom for training [5].
- Multi-node: `ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats` on the head node, `ray start --address=${MASTER_ADDR}:6379 --num-gpus 8` on each worker, then submit the job from the head node with `ray job submit --address="http://127.0.0.1:8265" --runtime-env-json='{...}' -- python3 train.py ...` [4].
- Effective batch arithmetic is explicit in the docs: `rollout-batch-size x n-samples-per-prompt = global-batch-size x num-steps-per-rollout` [5].
- SGLang arguments not natively exposed by slime pass through with a `--sglang-` prefix (e.g. `--sglang-log-level INFO` forwards `--log-level INFO` to SGLang) [5]; Megatron arguments pass through directly, unprefixed [1].
- Config surface is CLI flags parsed by `slime/utils/arguments.py`, not a config-class object; the file mixes native Megatron args, native (prefixed) SGLang args, and slime-specific args in one namespace [1][18]. slime always trains through data packing and, per the docs, "strictly ensures that per sample loss or per token loss is correct" regardless of this setting; the docs then recommend also turning on dynamic batching (`--use-dynamic-batch-size` with `--max-tokens-per-gpu`) because it "will not affect loss calculation" and is "strongly recommended" purely for the efficiency of packing variable-length data into GPU-sized chunks - it is an opt-in performance knob, not a correctness requirement [5][19].
- Out-of-memory first aid, from the FAQ page [19]: for `max_tokens_per_gpu` OOM, the initial suggestion is `rollout_max_response_len / cp_size` as a starting value (only active under `--use-dynamic-batch-size`), and to raise `--context-parallel-size` if it still OOMs; for SGLang-side "illegal memory access" errors during generation, reduce `--sglang-mem-fraction-static`, per SGLang's own FAQ [19]; for sglang port conflicts on multi-engine-per-node setups, reduce the number of SGLang servers per machine (e.g. by raising tensor parallelism per engine) [19].

## Watch it

This section covers only the plumbing - which signals exist and how to turn them on. What a given curve should look like for a given method belongs on that method's own card.

- **Enable it**: logging is opt-in. `--use-wandb` defaults to `False`; when set, `--wandb-mode` (`online`/`offline`/`disabled`), `--wandb-project`, `--wandb-group`, `--wandb-team`, `--wandb-key`, `--wandb-host` configure the run, and a random 6-character suffix is appended to the run name unless `--disable-wandb-random-suffix` is passed [18]. `--use-tensorboard` (default `False`) with `--tb-project-name` / `--tb-experiment-name` is the alternative [18]. With neither flag set, a run produces no W&B or TensorBoard record [18].
- **Metric namespaces**, read directly from source at the v0.3.1 tag (commit `a6272da0d4f3d0a08520c99a2f3b4f6c887960dc`): training-step metrics are logged under `train/` from `slime/backends/megatron_utils/model.py`, including `train/loss`, `train/pg_loss`, `train/entropy_loss`, `train/pg_clipfrac`, `train/ppo_kl`, `train/grad_norm`, `train/lr-pg_<id>`, `train/global_batch_size`, stepped by `train/step` [20]. The loss-computation module additionally names `value_loss`, `value_clipfrac` for PPO's critic and `tis`, `tis_clipfrac`, `tis_abs` for truncated-importance-sampling variants, which model.py logs under the same `train/` prefix [21]. Rollout- and eval-time metrics come from `slime/ray/rollout.py`: per-rollout metrics are logged under `rollout/` (via `compute_metrics_from_samples`, covering response-length statistics, repetition and truncation fractions) and `perf/` (via `compute_perf_metrics_from_samples`, covering rollout wall time and tokens/sec), stepped by `rollout/step`; per-eval-dataset metrics are logged under `eval/<dataset>` (mean reward) and `eval/<dataset>/...`, stepped by `eval/step` [22].
- **SGLang-side serving metrics are a separate path, by design**: the docs state plainly that slime's "default observability path is intentionally small" - W&B/TensorBoard keep receiving reward, loss, KL, entropy, and eval metrics, plus per-rollout-step aggregated request-timing summaries under `perf/request/*`, `perf/prefill/*`, `perf/decode/*` (e.g. `perf/request/e2e_latency/mean`, `perf/decode/throughput/mean`); but the high-frequency SGLang Prometheus metrics (`sglang:num_queue_reqs`, `sglang:num_running_reqs`, `sglang:kv_transfer_speed_gb_s_bucket`, and others) are explicitly "no longer uploaded to W&B" and instead require running a separate Prometheus scraping SGLang's `/engine_metrics` endpoint - without Prometheus running, those serving-side metrics have no history at all [23]. The detailed `perf/prefill/*` / `perf/decode/*duration` breakdown only appears when SGLang returns `pd_*` fields, i.e. under PD disaggregation [23].
- **Sample-level logging**: `--log-multi-turn` (default `False`), `--log-passrate` (default `False`, logs pass@n over the rollout's responses), `--log-reward-category` (default `None`, logs why a reward function failed a sample, keyed by a given dict key), and `--log-correct-samples` (default `False`) are opt-in flags on the arguments module [18]. Full per-sample generations and SGLang request traces are captured separately via `--save-debug-rollout-data`, replayable with a trace viewer (`python tools/trace_timeline_viewer.py /path/to/debug/rollout_0.pt`), which the docs say needs no separate logging service to work [24][23].
- **Evaluation during training**: `--eval-interval`, `--eval-prompt-data`, `--n-samples-per-eval-prompt`, `--eval-max-response-len`, and `--eval-top-p` override the rollout sampling settings for a held-out eval dataset [9].
- **Stopping / health limits**: no RL-specific automatic stopping threshold is published anywhere read for this card. The one quantitative limit found is a CI-only assertion, not a runtime health check: `slime/backends/megatron_utils/model.py` asserts `train/train_rollout_logprob_abs_diff <= 0.1` but only when `args.ci_test` is set, i.e. it fires in the project's own CI, not in a normal training run [20]. The debugging guide instead gives qualitative precision checks to run by hand at the first and second training steps - e.g. confirming `log_probs` and `ref_log_probs` are exactly equal at step one, and that `grad_norm` and KL stay small when `num_steps_per_rollout == 1` - rather than a numeric threshold to alert on [25]. The fault-tolerance page separately documents rollout-engine health-check timers with defaults (`--rollout-health-check-first-wait` 300s, `--rollout-health-check-interval` 10s, `--rollout-health-check-timeout` 5s), but these gate SGLang server restarts, not training convergence [26].

## Save it

- Three checkpoint flags: `--ref-load` (frozen reference model), `--load` (actor's resume-from checkpoint - falls back to `--ref-load` if unset or invalid), `--save` (actor's save directory), with `--save-interval` controlling cadence in the quick-start example (`--save-interval 20`) [9][27].
- Two on-disk formats. Megatron `torch_dist` (the recommended format, supporting automatic re-sharding across parallelism layouts) writes a `latest_checkpointed_iteration.txt` file plus per-iteration `iter_XXXXXXX/` directories containing `.distcp` shard files - the format produced by `tools/convert_hf_to_torch_dist.py` and consumed by `--ref-load`/`--load` [27]. A separate HuggingFace-format export is available via `--save-hf <path>`, whose help text states: "Path to save the model in HuggingFace format when using Megatron backend. The model will be saved to `save_hf.format(rollout_id)`. Weights are saved with the same quantization config as `--hf-checkpoint`" [18]; the saver module writes safetensors shards and copies the HF config/tokenizer assets alongside them [28].
- **Retention flag that kills resume**: `--no-save-optim`'s help text states, "If set, do not save the optimizer state when saving checkpoints. This reduces checkpoint size but disables training resumption from the saved checkpoint" [18].
- Resume: point `--load` at the same directory as a prior `--save` and restart training; the FAQ confirms this is the supported resume path [19].
- No LoRA/adapter saving path appears in the pages checked for this card - every save described above is a full-parameter Megatron checkpoint or a full HuggingFace-format export, not an adapter delta (see Methods it ships, and its caveat about pages not fetched) [1][18].
- Loader handoff: the `--save-hf` HuggingFace-format export is what an external evaluator or `transformers`/vLLM loader would consume directly; the `torch_dist`/`torch` Megatron checkpoint under `--save` is Megatron-native and needs the repo's own conversion tooling (the inverse of `tools/convert_hf_to_torch_dist.py`) before a non-Megatron loader can read it - this card does not verify that inverse conversion path in detail.

## Find it in the docs

The docs site at https://thudm.github.io/slime/ is unversioned Sphinx output built from the `main` branch - its own "Edit on GitHub" links point at `github.com/THUDM/slime/edit/main/...`, so there is no per-release version-tag URL form to swap in; the page you fetch always reflects `main` at fetch time, and the pinned-tag claims in this card are read from `raw.githubusercontent.com/.../v0.3.1/...` instead, not from this site [29].

- Address pattern for the raw pinned source: `https://raw.githubusercontent.com/THUDM/slime/<tag>/<path>`, e.g. `.../v0.3.1/docs/en/get_started/quick_start.md` - verified working for this card [9]. The rendered (unpinned) site mirrors the same paths under `https://thudm.github.io/slime/<section>/<page>.html`, e.g. `get_started/quick_start.html`, `advanced/observability.html`, `developer_guide/debug.html` [29].
- Page-slug map, from the site's own navigation [29]: setup and first run under `get_started/` (`quick_start`, `usage`, `qa`, `customization`, `agent`); scale and engine-integration features under `advanced/` (`fault-tolerance`, `observability`, `on-policy-distillation`, `pd-disaggregation`, `delta-weight-sync`, `external-rollout-engines`, `sglang-config`, `megatron-config`, `low-precision`, `speculative-decoding`, `reproducibility`, `arch-support-beyond-megatron`); worked model recipes under `examples/` (`glm4-9B`, `glm4.7-30B-A3B`, `glm4.7-355B-A32B`, `glm5.2-744B-A40B`, `qwen3-30B-A3B`, `qwen3-4B`, `qwen3-4b-base-openhermes`, `deepseek-r1`); contributor process under `developer_guide/` (`ci`, `debug`, `trace`, `profiling`); origin-story writeups under `blogs/` (`introducing_slime`, `release_v0.1.0`) [29].
- Question-to-slug map: "why did my rollout come out garbled / what's my KL at step 1" -> `developer_guide/debug` [25]; "what do I watch besides the loss curve" -> `advanced/observability` [23]; "how do I resume / what does my checkpoint directory look like" -> `get_started/usage` [27]; "how do I run across nodes" -> `get_started/quick_start`'s Multi-Node Training section [4]; general troubleshooting (OOM, port conflicts, NaN gradients) -> `get_started/qa` [19].
- Runnable references beyond the docs: the `examples/` tree in the GitHub repo itself (agentic-RL recipes: `multi_agent`, `search-r1`, `fully_async`, `coding_agent_rl`, `retool`), and the `scripts/` tree with per-model launch scripts and their matching `scripts/models/*.sh` config files, e.g. `scripts/run-glm4-9B.sh` paired with `scripts/models/glm4-9B.sh` [1][9].
- Community layer: this card found no dedicated curated-tutorials page comparable to trl's `community_tutorials`. The closest curated door is the README's own "Blogs" section, which links slime's own origin post (first published on `lmsys.org`, mirrored into the docs as `blogs/introducing_slime`) and its `v0.1.0` release note; both are first-party, not third-party practitioner posts [1][30]. No community blog layer was verified for this card.
- No official MCP endpoint for these docs was found in the pages read for this card; a third-party "Ask DeepWiki" badge links to `deepwiki.com/THUDM/slime` in the README, but DeepWiki is not slime's own project and was not queried for this card [1].
- Honest boundaries, stated where a reader would hit them: no LoRA/PEFT fine-tuning path is documented in the arguments module, README, quick-start guide, or usage guide checked for this card - the README attributes LoRA support only to the third-party derivative Miles, not to slime itself [1][18]; the low-precision, speculative-decoding, and beyond-Megatron architecture-support pages were not fetched for this card and were not checked for a LoRA path (see Methods it ships). Fault tolerance is scoped to the rollout engine only: the fault-tolerance page states explicitly that cluster-level preemption, trainer-rank failure, and full-job resume are the job of the cluster scheduler plus Ray's restart policy plus slime's own checkpointing, not something `--use-fault-tolerance` covers [26]. This card searched the FAQ and debugging pages but found no closed-issue, maintainer-attributed trap (with issue number, author association, and date) meeting this deck's stricter bar for that category - the FAQ entries cited above [19] are first-party documentation, not issue-thread testimony, and are presented here as general troubleshooting rather than as traps.

## Sources

All GitHub API and raw-file reads are pinned to release tag v0.3.1, resolved to commit `a6272da0d4f3d0a08520c99a2f3b4f6c887960dc` [12][13], except the repository/organization metadata [2][3][17], which reflects the live GitHub API at fetch time and is not itself version-pinned, and the docs site [29], which is an unversioned build of `main`. Fetch date for every source below is 2026-08-10 unless noted. Method names (SFT, GRPO, GSPO, CISPO, REINFORCE++, PPO, on-policy distillation) are deliberately cited to nothing here beyond the page that lists them; their defining papers belong on their own method cards. Ecosystem tools named in passing (Ray, Megatron-LM, SGLang, sgl-router, DeepWiki) are reached through slime's own links and are deliberately not enumerated as separate references.

[1] slime README at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/README.md.

[2] slime repository metadata (GitHub REST API). https://api.github.com/repos/THUDM/slime.

[3] slime GitHub repository (item home). https://github.com/THUDM/slime.

[4] slime Quick Start guide, Multi-Node Training and Docker sections, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/get_started/quick_start.md.

[5] slime Quick Start guide, PERF_ARGS and Native Engine Pass-Through / SGLang argument sections, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/get_started/quick_start.md.

[6] slime Quick Start guide, Colocated Actor and Rollout section, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/get_started/quick_start.md.

[7] slime Usage guide, Hyperparameters for RL Training / GRPO Algorithm / PPO Algorithm sections, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/get_started/usage.md.

[8] slime repository tree at tag v0.3.1 (file paths for the SFT data-processing example and SFT rollout module). https://api.github.com/repos/THUDM/slime/git/trees/v0.3.1?recursive=1.

[9] slime GLM4-9B-with-8xH100 example, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/examples/glm4-9B.md.

[10] slime Quick Start guide, Hardware Support section, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/get_started/quick_start.md.

[11] PyPI package registered under the name "slime" (an unrelated statistical-modeling project, not THUDM's framework). https://pypi.org/pypi/slime/json.

[12] slime tags list (GitHub REST API), resolving tag v0.3.1 to its commit. https://api.github.com/repos/THUDM/slime/tags.

[13] slime releases list (GitHub REST API), giving v0.3.1's publish date and the full release history. https://api.github.com/repos/THUDM/slime/releases.

[14] slime setup.py at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/setup.py.

[15] slime requirements.txt at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/requirements.txt.

[16] slime pyproject.toml at tag v0.3.1 (build-system and lint/test tool config only; no dependency pins). https://raw.githubusercontent.com/THUDM/slime/v0.3.1/pyproject.toml.

[17] slime repository metadata (GitHub REST API): owner type and open-issue count. https://api.github.com/repos/THUDM/slime.

[18] slime arguments module (`slime/utils/arguments.py`) at tag v0.3.1 (W&B/TensorBoard flags, sample-logging flags, `--save-hf`, `--no-save-optim`). https://raw.githubusercontent.com/THUDM/slime/v0.3.1/slime/utils/arguments.py.

[19] slime FAQ (Q&A) page, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/get_started/qa.md.

[20] slime Megatron training-step model module (`slime/backends/megatron_utils/model.py`) at tag v0.3.1 (`train/*` metric keys and the CI-only logprob-diff assertion). https://raw.githubusercontent.com/THUDM/slime/v0.3.1/slime/backends/megatron_utils/model.py.

[21] slime Megatron loss module (`slime/backends/megatron_utils/loss.py`) at tag v0.3.1 (per-loss-type metric keys). https://raw.githubusercontent.com/THUDM/slime/v0.3.1/slime/backends/megatron_utils/loss.py.

[22] slime Ray rollout module (`slime/ray/rollout.py`) at tag v0.3.1 (`rollout/*`, `perf/*`, `eval/*` metric keys). https://raw.githubusercontent.com/THUDM/slime/v0.3.1/slime/ray/rollout.py.

[23] slime Observability guide, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/advanced/observability.md.

[24] slime arguments module (`slime/utils/arguments.py`) at tag v0.3.1: `--debug-rollout-only`, `--save-debug-rollout-data`, `--load-debug-rollout-data` help text. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/slime/utils/arguments.py.

[25] slime Debugging guide (developer guide), at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/developer_guide/debug.md.

[26] slime Fault Tolerance guide, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/advanced/fault-tolerance.md.

[27] slime Usage guide, Loading Megatron Checkpoints section, at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/get_started/usage.md.

[28] slime HuggingFace checkpoint saver module (`slime/backends/megatron_utils/hf_checkpoint_saver.py`) at tag v0.3.1. https://raw.githubusercontent.com/THUDM/slime/v0.3.1/slime/backends/megatron_utils/hf_checkpoint_saver.py.

[29] slime documentation site (unversioned Sphinx build of the `main` branch). https://thudm.github.io/slime/.

[30] slime "Introducing slime" blog post, at tag v0.3.1 (mirrored from its original publication on lmsys.org). https://raw.githubusercontent.com/THUDM/slime/v0.3.1/docs/en/blogs/introducing_slime.md.
