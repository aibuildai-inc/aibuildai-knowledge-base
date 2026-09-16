# simple_GRPO

A ~200-line, two-process reference implementation of GRPO built for reading and hacking, not for production training runs.

**simple_GRPO** is described by its own README as "a very simple GRPO implement for reproducing r1-like LLM thinking", built to be a minimal, readable codebase rather than a general framework [1]. It lives at https://github.com/lsdefine/simple_GRPO [1]. The project is led by Dr. Jiaqing Liang and Professor Yanghua Xiao of KnowledgeWorks Lab, Fudan University, with a core development team of Jinyi Han and Xinyi Wang and other contributors [1]; GitHub's own contributor list shows four accounts with commits (lsdefine, JinyiHan99, ViviqwerAsd, Jiangzs1028) [2]. Its API is two plain Python scripts, not a package or trainer class: `ref_server.py` runs a standalone reference-model HTTP server, and `grpo_vllm_one.py` is launched directly with `deepspeed` to train, spawning its own vLLM generation subprocess [3][4].

**When to pick it**: pick this when you want to read or modify a complete, working GRPO training loop in one sitting - loss, KL penalty, clipping, and the generation/training split are all in two files with no framework abstraction layer between you and the code [1]. It ships no CLI, no config file, no logging integration, and no training-resume path (see Start it, Watch it, Save it) [4][5]. It is not a multi-method or multi-node framework; for those needs weigh the trl or verl cards instead (cross-reference; not covered here). The maintainer's own successor project, LSRL, is pointed to from this README as a more robust, better-organized rewrite for users who outgrow this repo [1][6].

**Methods it ships**: GRPO is the documented, primary method, in the top-level `grpo_vllm_one.py` [1][4]. A REINFORCE++ variant was added per the README changelog entry dated 2025/03/24, shipped as `rf++_vllm_one.py` in the `simple-reinforce++/` folder, with the same usage pattern as GRPO [1][7]. A second GRPO script, `simple_grpo_v1/grpo_ref_split.py`, offers a synchronous ref/train split described in its own folder readme as needing "JUST two py files" [8][9]; that script references a commented-out Triton loss kernel (`from kernel.ce_kernel import fast_log_softmax_gather`) whose `kernel/` module does not exist anywhere in the repository tree, so it cannot be enabled as shipped [8][3]. A fourth folder, `Auto_Program/`, ships five more files (`config.py`, `hjy_grpo_program.py`, `ref_server.py`, and two system-prompt text files) that are not mentioned anywhere in the README; this card does not describe their behavior since it was not read [10]. There is no methods taxonomy page to check against - the README itself is the only index, and it is a changelog list, not a maintained table [1].

**Scale it handles**: single machine, multiple GPUs, launched with `deepspeed` directly on `grpo_vllm_one.py`; the deepspeed config is an inline Python dict in the script (ZeRO stage 2, optimizer offloaded to CPU), not a separate launcher template [4]. The reference-model process is decoupled over HTTP and the README states it can run "on a different machine with 4090" hardware [1]; there is no multi-node training launcher for the trainer process itself, and no FSDP support in the code. The only published scale benchmark is two GPUs: the README reports Qwen2.5-3B and Qwen2.5-7B each completing 60 optimization steps on 2xA800 (80GB) in 12m34s and 16m40s respectively [1] - there is no larger published run to check a bigger-scale claim against. The same benchmark reports that this loop actually converges at that scale: Qwen2.5-3B's accuracy stabilizes above 60% after 5 steps and peaks around 70%, with format compliance reaching about 100% by step 30, while Qwen2.5-7B's accuracy stays above 90% throughout training with format compliance hitting 100% within 30 steps [1].

**Install**: no PyPI package exists; install is `git clone` plus `pip install -r requirements.txt` [11]. There are no GitHub releases and no tags on this repository [12][13], so there is no version number or release date to report, and the commit this card reads, `30f252ce368c03556662eeaf0023809c38cff6f9`, is simultaneously the repository's newest push and the only available pin [14][15]. Python floor and CUDA/hardware minimum are not stated anywhere in the repo; the README states only "At least two GPUs are needed" [1]. `requirements.txt` lists bottle, math-verify, safetensors, sentencepiece, tokenizers, torch, tornado, transformers, and deepspeed with no version bounds at all, and one pinned package: `vllm==0.10.1.1` [11]. That vLLM pin is load-bearing and deliberate: in a closed issue, the maintainer (repo owner, association OWNER) confirms that later vLLM releases silently broke `prompt_token_ids` support in `vllm_gen.generate()`, and calls pinning to `vllm==0.10.1.1` "a reasonable workaround" until a more robust integration lands [16]. Licence is Apache-2.0 [17].

**Maintained by**: KnowledgeWorks Lab, Fudan University (Dr. Jiaqing Liang and Professor Yanghua Xiao), with Jinyi Han and Xinyi Wang as core developers [1]. The repository was last pushed 2025-11-21T01:03:17Z [15], the same day the maintainer replied to a newly opened compatibility issue with a concrete fix plan [16] - a dated sign of active maintenance, not a claim about trend over time.

## Quick start

The README's usage section, run as two terminals from a clone of the repo [1]:

```bash
CUDA_VISIBLE_DEVICES=7 python ref_server.py
```

This starts the reference-model HTTP server on one GPU [1]. In `grpo_vllm_one.py`, set the generation device index relative to the visible devices of the next command:

```python
gen_device = 1
```

Then, in a second terminal:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed grpo_vllm_one.py
```

This runs training across the listed GPUs, using GSM8K loaded live via `datasets.load_dataset("openai/gsm8k", "main", split="train")` and a Qwen model at a hardcoded local path (`model_path = "/data2/Qwen/Qwen2.5-7B"` in the script) that must be edited before the run will find a model [1][4]. There is no CLI form; every value above is a module-level variable in the script itself.

## Start it

- One GPU is not a supported layout: the README states plainly that at least two GPUs are needed, because the reference-model server and the training/generation processes are separate processes that must not share a device [1].
- Multi-GPU: `ref_server.py` is launched as a plain Python process pinned to one GPU via `CUDA_VISIBLE_DEVICES`, with no distributed launcher [1][5]. `grpo_vllm_one.py` is launched with `deepspeed grpo_vllm_one.py` across the remaining GPUs; deepspeed's config is the inline `ds_config` dict in the script - ZeRO stage 2, `overlap_comm`, `contiguous_gradients`, and `offload_optimizer` to CPU, with `bf16.enabled: True` forced on - not a separate YAML template [4]. There is no config file to point the launcher at.
- Generation runs as its own spawned subprocess (`mp.Process(target=gen_worker, ...)`) that loads vLLM and is pinned to a single GPU index, `gen_device`, which the code comment warns must be excluded from the training launch's `CUDA_VISIBLE_DEVICES` set [4].
- Effective batch: the deepspeed config sets `train_micro_batch_size_per_gpu = train_batch_size` (8) and `gradient_accumulation_steps = 4` [4], so the effective batch per optimizer step is `train_batch_size x (number of training GPUs) x gradient_accumulation_steps`.
- Config surface: there is no Config class - every hyperparameter is a bare module-level variable at the top of `grpo_vllm_one.py`: `beta` (KL coefficient, default 0.04), `all_steps` (1000, the fixed step budget), `Q_batch_size` (5 prompts per generation batch), `num_pre_Q` (8 generations per prompt), `train_batch_size` (8), `gen_update_steps` (16, how often trained weights are pushed to the vLLM generation worker), `save_steps` (200), and `clip_param` (0.2, the PPO-style clip range) [4].
- Changed default worth flagging: `bf16` is hardcoded on in both the deepspeed config and the `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)` call - a bf16-capable GPU is assumed and there is no flag to switch it; changing precision means editing the script [4].
- vLLM's memory share is hardcoded too: `LLM(model=model_path, gpu_memory_utilization=0.5)` inside `gen_worker` [4].
- OOM first aid comes from the maintainer's own reply in a closed compatibility issue, not from a published tuning guide: reduce vLLM's `max_model_len` from the model's full context (around 128k) down to the task's actual needs - the maintainer gives 4k-8k as a reasonable range for GSM8K - to shrink the KV cache, and add an explicit `del state_dict` plus `torch.cuda.empty_cache()` after the `Q.put(state_dict)` call in the training loop for long runs, since the maintainer's own testing found the reference held by that queue eventually gets reclaimed but is not cleaned up promptly [16]. A separate, unconfirmed community report in a different closed issue (no maintainer reply) says lowering `num_pre_Q` from 8 to 6 avoided OOM across two A100 40GB training GPUs (plus a third A100 40GB dedicated to inference) when training Qwen2.5-3B [18].

## Watch it

This section is mechanics only - what a loss or reward number means for GRPO as a method lives on the GRPO methodology card, not here.

- There is no logging integration anywhere in the code: no `wandb`, `tensorboard`, or equivalent import exists in `grpo_vllm_one.py` or `ref_server.py` [4][5]. The only run record is stdout, and nothing is written to disk unless the operator redirects it themselves.
- The training process prints one scalar via `tqdm`: `progress.set_description(f"Loss: {loss.item():.6f}")`, where `loss` is the GRPO clipped-ratio objective combined with the `beta`-scaled KL penalty computed in `GRPO_step` [4].
- The generation process prints, per batch, the wall-clock generation time and the raw per-completion `rewards` tensor (`print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )`), and prints one full sample completion every 5th generation iteration (`if it % 5 == 0: print('answers:', answers[0])`) - this is the only sample-level visibility the code provides [4].
- No completion-length, KL, or clip-fraction scalar is logged separately from the single combined loss; a reader who wants those would have to add the logging.
- No evaluation-during-training exists: the generation worker samples only from the GSM8K train split (`random.sample(QAs, Q_batch_size)`), and there is no `eval_dataset` or held-out evaluation loop in either script [4].
- No published health limit or stopping rule was found; the only search available for this repo is its two Python scripts and their two readmes, since there is no separate docs site - and none of them define a threshold, patience value, or automatic stop [4][5][1][9]. The only stopping condition is the fixed `all_steps = 1000`, a hyperparameter set by the operator, not a health check [4]. The one documented failure mode that looks like a hang rather than a stop is the training loop's "waiting for batch..." print, which repeats forever if the reference-server or generation process has died or lost its network connection to `ref_server`; the maintainer's own reply in a closed issue traces this exact symptom to a crashed or disconnected generation process and recommends checking whether the generation subprocess died, or trying the maintainer's newer, more robust async implementation in the LSRL repository instead [6].

## Save it

- Checkpoints land in `./step_{step}/`, created every `save_steps` (default 200) optimizer steps, using the training process's own `step` counter [4].
- The save call is `engine.module.save_pretrained(save_name, state_dict=state_dict)` followed by `tokenizer.save_pretrained(save_name)`, where `state_dict` is `engine.module.state_dict()` moved to CPU first - this writes a standard Hugging Face model directory (config plus weight files plus tokenizer files) directly through the underlying `transformers` model, not through any deepspeed-specific format [4].
- No optimizer or scheduler state is ever saved: `deepspeed`'s own `engine.save_checkpoint()` call is never invoked anywhere in `grpo_vllm_one.py` or `simple_grpo_v1/grpo_ref_split.py` - only the bare model weights leave the process [4][8].
- There is no resume path: neither script contains a `resume_from_checkpoint` call, a deepspeed `load_checkpoint` call, or any other code that reads a saved `./step_N/` directory back into a running training process - a direct search of both files for "resume" and "checkpoint"-loading logic returns no match [4][8]. A saved directory can only be used as a fresh starting model for a new `from_pretrained` load, not to continue the same training run.
- No PEFT or adapter path exists; every save is the full model's weights, so there is no adapter-vs-merged distinction to make here.
- Loader handoff: because the save call goes straight through the underlying `transformers` model's own `save_pretrained`, a saved `./step_N/` directory is directly loadable by a plain `AutoModelForCausalLM.from_pretrained("./step_N/")` in any evaluator - the loader's contract is met for evaluation, even though the same directory cannot restore training state.

## Find it in the docs

There is no hosted documentation site for this project - the README.md at the repository root is the only maintained index, and two folders carry their own local readme: `simple_grpo_v1/readme.md` documents the synchronous split-reference variant [1][9]. The `simple-reinforce++/` and `Auto_Program/` folders have no folder-local readme; their own source files are the only description of what they do, and this card does not summarize either beyond noting they exist [7][10].

- Lookup recipe: there is nothing to guess a URL pattern for - read the top-level `README.md` first, then the folder you are using, in the GitHub file tree at https://github.com/lsdefine/simple_GRPO [1].
- Runnable reference beyond the README: the quickstart script itself loads a known-good smoke-test dataset live from the Hub, `openai/gsm8k` (`"main"` split) [4] - there is no separate examples tree.
- The community/successor layer, curated by the maintainer directly in the README: a "NEW" entry dated 2025/07/23 in the changelog points readers to LSRL, described there as "our next-generation RL framework with custom optimizer and streamlined architecture for better performance" [1]. This is the maintainer's own recommended upgrade path, not a third-party curation list - when a user hit the "waiting for batch" hang described above, the maintainer's own closed-issue reply pointed them specifically at LSRL's `examples/rl_gsm8k_DAPO.py` and `rl_gsm8k_7b_1gpu.py` as more robust async reimplementations of this repo's idea [6]. LSRL itself is a separate project and is not covered by this card.
- No official MCP endpoint or docs-query tool is published for this repository.

The honest boundary: this repository trains one model architecture family at a time via a single hardcoded `model_path`, supports GRPO and REINFORCE++ as documented methods, requires at least two GPUs to run at all, and provides no config file, no logging integration, no evaluation loop, and no training-resume path - a reader who needs any of those should expect to add them, or use a general framework instead [1][4][5].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. All GitHub file, API, and issue reads below are pinned to commit `30f252ce368c03556662eeaf0023809c38cff6f9`, the repository's `main`-branch HEAD at fetch time, since no release or tag exists for this repository (see [12], [13]).

[1] simple_GRPO README.md at commit 30f252c. https://raw.githubusercontent.com/lsdefine/simple_GRPO/30f252ce368c03556662eeaf0023809c38cff6f9/README.md. Fetched 2026-08-12.

[2] simple_GRPO contributors (GitHub API). https://api.github.com/repos/lsdefine/simple_GRPO/contributors. Fetched 2026-08-12.

[3] simple_GRPO repository root file listing at commit 30f252c (GitHub API). https://api.github.com/repos/lsdefine/simple_GRPO/contents/. Fetched 2026-08-12.

[4] `grpo_vllm_one.py` at commit 30f252c. https://raw.githubusercontent.com/lsdefine/simple_GRPO/30f252ce368c03556662eeaf0023809c38cff6f9/grpo_vllm_one.py. Fetched 2026-08-12.

[5] `ref_server.py` at commit 30f252c. https://raw.githubusercontent.com/lsdefine/simple_GRPO/30f252ce368c03556662eeaf0023809c38cff6f9/ref_server.py. Fetched 2026-08-12.

[6] simple_GRPO issue #49, "waiting for batch", and maintainer (lsdefine, association OWNER) replies. https://api.github.com/repos/lsdefine/simple_GRPO/issues/49/comments. Fetched 2026-08-12.

[7] `simple-reinforce++/` folder file listing at commit 30f252c (GitHub API). https://api.github.com/repos/lsdefine/simple_GRPO/contents/simple-reinforce%2B%2B. Fetched 2026-08-12.

[8] `simple_grpo_v1/grpo_ref_split.py` at commit 30f252c. https://raw.githubusercontent.com/lsdefine/simple_GRPO/30f252ce368c03556662eeaf0023809c38cff6f9/simple_grpo_v1/grpo_ref_split.py. Fetched 2026-08-12.

[9] `simple_grpo_v1/readme.md` at commit 30f252c. https://raw.githubusercontent.com/lsdefine/simple_GRPO/30f252ce368c03556662eeaf0023809c38cff6f9/simple_grpo_v1/readme.md. Fetched 2026-08-12.

[10] `Auto_Program/` folder file listing at commit 30f252c (GitHub API). https://api.github.com/repos/lsdefine/simple_GRPO/contents/Auto_Program. Fetched 2026-08-12.

[11] `requirements.txt` at commit 30f252c. https://raw.githubusercontent.com/lsdefine/simple_GRPO/30f252ce368c03556662eeaf0023809c38cff6f9/requirements.txt. Fetched 2026-08-12.

[12] simple_GRPO releases (GitHub API, empty). https://api.github.com/repos/lsdefine/simple_GRPO/releases. Fetched 2026-08-12.

[13] simple_GRPO tags (GitHub API, empty). https://api.github.com/repos/lsdefine/simple_GRPO/tags. Fetched 2026-08-12.

[14] simple_GRPO commits list (GitHub API), showing commit 30f252c as HEAD at fetch time. https://api.github.com/repos/lsdefine/simple_GRPO/commits. Fetched 2026-08-12.

[15] simple_GRPO repository metadata (GitHub API): licence, pushed_at, created_at, stargazers_count, description, archived, language. https://api.github.com/repos/lsdefine/simple_GRPO. Fetched 2026-08-12.

[16] simple_GRPO issue #54, "Vllm compatibility, memory leak and context length issues", and maintainer (lsdefine, association OWNER) reply. https://api.github.com/repos/lsdefine/simple_GRPO/issues/54 and https://api.github.com/repos/lsdefine/simple_GRPO/issues/54/comments. Fetched 2026-08-12.

[17] simple_GRPO `LICENSE` file at commit 30f252c (Apache License 2.0 header). https://raw.githubusercontent.com/lsdefine/simple_GRPO/30f252ce368c03556662eeaf0023809c38cff6f9/LICENSE. Fetched 2026-08-12.

[18] simple_GRPO issue #11, "OOM 问题" - community troubleshooting thread with no maintainer reply, cited as unconfirmed. https://api.github.com/repos/lsdefine/simple_GRPO/issues/11/comments. Fetched 2026-08-12.
