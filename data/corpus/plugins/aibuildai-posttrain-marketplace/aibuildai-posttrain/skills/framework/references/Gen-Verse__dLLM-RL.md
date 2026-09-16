# dLLM-RL

A research codebase for post-training diffusion language models (dLLMs): SFT, RL (TraceRL, Random Masking RL, Coupled RL), and RLHF, driven by hand-written subprocess scripts rather than a trainer-class API.

**dLLM-RL** is the official code release described in its GitHub description as "Official code for TraceRL: Revolutionizing post-training for Diffusion LLMs, powering the SOTA TraDo series" [1], and its README calls itself the "Most comprehensive framework for dLLM's and multimodal dLLM's post-training" [2]. It is built by the authors of the TraceRL paper (Yinjie Wang, Ling Yang, Bowen Li, Ye Tian, Ke Shen, Mengdi Wang) [3] under the Gen-Verse GitHub organization, at https://github.com/Gen-Verse/dLLM-RL [1]. There is no trainer class: an outer Python script (`rl.py`) drives one RL step at a time as a sequence of `subprocess.run` calls to separate rollout, reward, and `accelerate launch` training scripts, and SFT is a single `accelerate launch train/sft_<model>.py` invocation with no orchestration loop around it [4][5].

**When to pick it**: pick this repo specifically to reproduce the TraceRL paper or to post-train one of the seven diffusion-LLM families it names support for (TraDo, SDAR, Dream, LLaDA, MMaDA, LLaDA-V, Diffu-Coder) [2], not as a general post-training library - it has no trainer-class API, no PyPI package, and its multi-step RL loop is a set of Python scripts glued together with `subprocess.run` rather than a framework surface like trl's or verl's (cross-reference; not covered here) [4]. Weigh it against those only if your target model is autoregressive; for the diffusion-LLM model families it lists, it is the paper's own reference implementation.

**Methods it ships** [2]:
- RL: TraceRL (trajectory-aware policy optimization, with an optional diffusion-based value model for variance reduction), Random Masking RL (a PPO-like objective on randomly masked sampled data), Coupled RL (adds the complement of each random-masking sample as an extra training example). The `training.method` config field selects one of `"TraceRL"`, `"random_masking"`, or `"coupled"` [6].
- SFT: Block SFT, semi-AR SFT, random masking SFT, and trace fine-tuning, with the applicable subset depending on model architecture - block-attention models (TraDo, SDAR) support semi-autoregressive or trace fine-tuning, adapted full-attention models (Dream, Diffu-Coder) support semi-autoregressive (sliced data), random-masking, or standard AR SFT, and pretrained full-attention models (LLaDA, MMaDA) support semi-autoregressive or random-masking SFT [2].
- Reward: a math-accuracy reward function (`reward/reward.py`, exact-match via `math_utils.is_equal`) [7] and a coding reward path (`rl_code_reward.py`) reachable through `rl.py`'s `is_code_task` branch [4]; a process-reward/RLHF path exists via configs named `*_process_reward_rlhf_with_value.yaml`, described in `configs/readme.md` as "Multimodal Process Reward / Fine-Grained Alignment" [8], and its `process_reward` step in `rl.py` runs three scripts through a separate `conda run -n CURE2` environment not otherwise documented [4].
- There is no separate methods-index page; the method list above is read from the README's Features and RL Methods sections [2] and the RL/SFT method names in `configs/readme.md` [8], both live at the repo's default branch and subject to change with new commits.

**Scale it handles**: single GPU up to a documented 16-node, 8-GPU-per-node layout - `accelerate_configs/` ships DeepSpeed ZeRO-2 and ZeRO-3 templates named `{1,2,4,8,16}_node_8_gpus_deepspeed_{zero2,zero3}.yaml` [9]. Both SFT and the per-RL-step training subprocess launch through `accelerate launch --config_file accelerate_configs/<template>.yaml train/<script>.py` [5][4]; multi-node RL and eval additionally have `multinode_rl.py`/`multinode_eval.py` entry points that read `MLP_ROLE_INDEX`, `MLP_WORKER_NUM`, `MLP_WORKER_0_HOST`, `MLP_WORKER_0_PORT` and run only on the rank-0 "controller" node, with other ranks idling on `tail -f /dev/null` [2]. This is documented mechanism with named templates for up to 16 nodes; no published benchmark or throughput number for any of these node counts appears in the README or configs docs.

**Install**: `conda create --name dllm-rl python=3.10`, then `pip install torch==2.6.0`, a pinned flash-attention wheel (`flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`), then `pip install -r requirements.txt` (or `requirements_v.txt` for multimodal settings, per the README) [2]. Apache-2.0 licence [10]. There are no GitHub Releases and no tags on this repository (both the releases and tags API endpoints return empty lists) [11][12], so there is no version/date to report beyond the commit itself: the card is pinned to commit `10b4fd1cba83e4496c20585dc2fd01dc02f20255`, dated 2026-01-28, which is also the repository's newest push at fetch time [13][1]. `requirements.txt` at that commit hard-pins `torch==2.6.0` and `transformers==4.52.4`; `accelerate`, `deepspeed`, `datasets`, and `peft` are listed unpinned [14]. The README's own multimodal install line names `requirements_v.txt`, but no file of that name exists in the repository tree at this commit - a repo/docs mismatch to check before following that instruction [15][9]. No CUDA or hardware minimum is stated anywhere in the README or configs docs beyond the flash-attention wheel's `cu12` build tag, which implies CUDA 12 [2].

**Maintained by**: the Gen-Verse GitHub organization, corresponding to the TraceRL paper's authors [1][3]; the README's own changelog records active work through 2026-01-26, when it notes the TraceRL paper's acceptance to ICLR 2026, after adding RLHF, process-reward, and multimodal RL/SFT support on 2025-12-07 [2]. 512 GitHub stars at the time this row was compiled, not a ranking signal.

## Quick start

The README's inference snippet is the only quoted "quick start" the repo publishes - there is no minimal training example shorter than a full RL or SFT config [2]:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import block_diffusion_generate

model_name = "Gen-Verse/TraDo-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype="float16", device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "What's the solution of x^2 - 2x + 1 = 0\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

tokens = tokenizer.batch_encode_plus([text], return_tensors='pt', padding=True, truncation=True, max_length=200)
tokens = {k: v.to(model.device) for k, v in tokens.items()}

output_ids = block_diffusion_generate(
    model, prompt=tokens, mask_id=151669, gen_length=200,
    block_length=4, denoising_steps=4,
    temperature=1.0, top_k=0, top_p=1.0,
    remasking_strategy="low_confidence_dynamic", confidence_threshold=0.9
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
```

For training, the smallest complete forms are single commands against a pre-selected config: `python rl.py config=configs/rl_trado.yaml` for RL, and `accelerate launch --num_machines 1 --machine_rank 0 --main_process_ip 127.0.0.1 --main_process_port 8888 --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml train/sft_trado.py config=configs/sft_trado.yaml` for SFT [2].

The paper's own reference numbers show what the RL side buys: on MATH500 with static sampling, TraDo-4B-Instruct (TraceRL-trained) scores 75.6 versus 70.2 for its SDAR-4B-Chat starting point (+5.4), and TraDo-8B-Instruct scores 78.5 versus 74.3 for SDAR-8B-Chat (+4.2); the long-CoT TraDo-8B-Thinking variant (TraceRL plus long-CoT SFT) reaches 87.4 on MATH500, 35.5 on AIME2024, and 36.0 on LiveBench (dynamic sampling), each row reporting the delta over its own un-RL'd baseline in the same table [16].

## Start it

- One GPU: run `python rl.py config=configs/rl_trado.yaml` directly, or the equivalent `python eval.py config=configs/trado_eval.yaml` for inference-only [2]. `rollout.tensor_parallel_size: 1` is the single-GPU rollout default in the shipped TraDo RL config [6].
- Multiple GPUs, one node: `accelerate_configs/1_node_8_gpus_deepspeed_zero{2,3}.yaml` selected via the config's `experiment.deepspeed_file` field; `rl.py` itself invokes `accelerate launch --num_machines 1 --machine_rank 0 --main_process_ip 127.0.0.1 --main_process_port 8888 --config_file accelerate_configs/{deepspeed_file}.yaml train/{script}.py config=configs/{project}.yaml experiment.current_epoch={i}` for every training call inside the RL loop [4][9].
- Multiple nodes: the README's SFT multi-node form reads cluster-assigned environment variables directly into the `accelerate launch` flags - `--num_machines $MLP_WORKER_NUM --machine_rank $MLP_ROLE_INDEX --main_process_ip $MLP_WORKER_0_HOST --main_process_port $MLP_WORKER_0_PORT` - and RL multi-node uses `multinode_rl.py` with a rank-0-only launch guard (`if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then ...; else exec tail -f /dev/null; fi`) [2].
- A generation-layout choice for RL: with a value model configured (`model.value_base_model` set), `rl.py` runs a separate `accelerate launch` for the value-model training script and one for the policy script every RL step (`train(i, target="value")` then `train(i, target="policy")`), instead of the single combined training call used when no value model is configured [4].
- Effective batch arithmetic for RL rollouts: the TraDo RL config sets `rollout.num_task_per_step: 64` and `rollout.num_response_per_task: 16`, so 1,024 rollout samples feed each RL step's reward and training stage, divided into gradient-accumulated minibatches by `training.gradient_accumulation_steps: 32` on the training side [6].
- Config surface: one OmegaConf YAML per project under `configs/`, merged with CLI overrides via `OmegaConf.from_cli()` + `OmegaConf.merge` - every script (`rl.py`, `eval.py`, `train/*.py`) takes `config=configs/<name>.yaml` plus dotted-key overrides on the command line [4]. `training.mixed_precision: "bf16"` is the value set in the shipped TraDo RL and SFT configs, a bf16-capable-GPU assumption the repo does not flag as a change from any base default because there is no separate base library underneath it [6][17].
- Out-of-memory first aid, from `configs/readme.md`'s tips section: for large block sizes, lower `rollout.max_active` (e.g. 16 for a block size of 64); if inference stagnates, decrease `max_active` further; if RL training is unstable, raise `gradient_accumulation_steps`, `num_task_per_step`, and `num_response_per_task` together - the same page notes that even Open-Reasoner-Zero's GRPO runs with `num_task_per_step=128` and `num_response_per_task=64` can still be unstable without a value model [18].

## Watch it

This section is the mechanics only - no method card exists yet for TraceRL/Random Masking RL/Coupled RL's training-signal semantics in this corpus, so read the shapes below as plumbing, not as guidance on what a healthy curve looks like.

- **No experiment tracker is wired up by default, and none can be enabled without editing code.** Both `train/rl_trado.py` and `train/sft_trado.py` construct `Accelerator(..., log_with=None, ...)` and never call `accelerator.log()` or `wandb.log()` anywhere in their training loops, even though `accelerator.init_trackers()` is called with a wandb config built from the YAML's `wandb.entity`/`wandb.resume` fields, at commit `10b4fd1cba83e4496c20585dc2fd01dc02f20255` [19][5]. The only per-step console signal is `print(loss_lm)` for the first 10 optimizer steps of a run [19].
- **Reward and accuracy go to plain-text files, not a metrics store.** `reward/reward.py` and `reward/rl_reward.py` compute accuracy and average response length and write them with a `cprint`-then-append helper to `<project>/results/results-<name>.txt`, one line per evaluation call, at the same commit [7][19]. A maintainer (GitHub user `yinjjiew`, association MEMBER) confirmed this is the intended interface in issue #41 ("Visualization of reward and training loss, with logging", opened 2025-11-12, closed): "You can check ./project_name(your rl config name)/results/. The reward and acc during training are all within the txt files. If you want to add more training logs, you can check ./reward and modify the rl_reward.py code" [20].
- **No metric names are published on a page**: there is no docs site or metrics-list page for this repo (README and `configs/readme.md` are the only prose docs), so there is no live page to point at the way `trl`'s trainer pages do; the field and file names above are read directly from the source.
- **No sample-level generation logging** is wired into the RL scripts beyond the plain-text results files; there is no `log_completions`-style flag in the configs read for this card [6][19].
- **Evaluation during RL training is built into the outer loop, not a Config flag**: `rl.py`'s main loop runs a full evaluation pass - sampling, execution (for code tasks), and reward - every `experiment.eval_every` steps, looping over the config's `evaluation.block_size`/`top_k`/`remasking_strategy` lists and writing results the same way as training rewards [4].
- **Stopping**: no early-stopping, patience, or reward-threshold field appears in any config read for this card (`configs_rl_trado.yaml`, `configs_rl_trado_with_value.yaml`) or in `configs/readme.md`'s tips section; the RL loop is a fixed `while i <= experiment.total_step` counter with no other exit condition read from configuration [4][6][18].
- **A maintainer's own stability guidance, from a separate closed issue** (#45, "Oscillations in the LOSS curve during training", opened 2025-12-04, closed): in response to a reported oscillating loss curve during RL, `yinjjiew` (MEMBER) replied "This is very common for loss in RL training, which is different from SFT. We usually directly check ACC/reward during RL training" [21], and when the reporter followed up that accuracy itself plateaued and then declined, the same maintainer replied that a fluctuating, non-improving curve signals the RL run has hit a bottleneck, suggesting either scaling training to thousands of steps or first collecting SFT data to fine-tune the model before resuming RL [21].

## Save it

- Checkpoints land under `<project>/ckpt/<name>/`, written by a shared `save_checkpoint` helper duplicated in `train/rl_trado.py` and `train/sft_trado.py`: it calls `model_to_save.save_pretrained(save_dir, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model), safe_serialization=True)` (safetensors weights) plus `tokenizer.save_pretrained(...)`, then copies the custom architecture's `modeling_*.py`/`configuration_*.py`/`tokenization_*.py`/`processing_*.py` source files into the same directory so `trust_remote_code=True` loading works standalone, and writes a `metadata.json` with a save timestamp, all at commit `10b4fd1cba83e4496c20585dc2fd01dc02f20255` [19].
- **No optimizer, scheduler, or RNG state is ever saved.** Neither training script's `save_checkpoint` writes any optimizer or scheduler state; a `checkpoints_total_limit` config field prunes older checkpoint directories by count, and there is no equivalent of a "save only model" flag to disable, because optimizer state was never part of the save contract to begin with [19].
- Resume works at the RL-orchestration level, not through `accelerator`/Trainer-style mid-epoch resume: `configs/readme.md` states plainly, "How to resume your training? Simply set `experiment.start_from_scratch = False` and `experiment.current_epoch` as your last RL step" [18] - this restarts the outer sample-reward-train loop in `rl.py` from a saved policy (and value, if configured) checkpoint at the given step, re-running rollout and reward from scratch for that step rather than resuming a paused optimizer.
- `train/rl_trado.py` is re-launched as a fresh subprocess once per RL step by `rl.py`'s outer `while i <= config.experiment.total_step` loop, and its unconditional final `save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)` call overwrites the same `optimized_name` checkpoint directory at the end of every one of those per-step subprocesses, not only once at the end of the run [19][4]. Separately, an `experiment.save_every`-gated call inside the script writes a distinct, non-overwritten `epoch-<current_epoch>` checkpoint directory every `save_every` steps [19].
- No PEFT/LoRA adapter-only save path was found in the training scripts read for this card (`train/rl_trado.py`, `train/sft_trado.py`); every save is a full model directory under `ckpt/`, so there is no "adapter is not a full model" distinction to flag here the way there is for PEFT-integrated libraries.
- Loader handoff: because `save_pretrained` writes a complete safetensors model directory plus tokenizer and the model's own dynamic-module source files, a saved checkpoint directory is loadable directly with `AutoModelForCausalLM.from_pretrained(<ckpt_dir>, trust_remote_code=True)`, mirroring the README's own inference snippet's load call [2][19] - whether a downstream evaluator accepts that call is that evaluator's contract, not this repo's.

## Find it in the docs

There is no hosted docs site for this repository - the README [2] and `configs/readme.md` [8] are the only prose documentation, both read at the pinned commit and both living on the default branch, so their content moves with every push rather than being version-tagged.

- Repository root: https://github.com/Gen-Verse/dLLM-RL - browse `README.md` for the feature overview, quick start, and method descriptions, and `configs/readme.md` for the config-selection guide and the "Some tips" operational notes [2][8].
- `configs/readme.md` is the practical reference for "which config do I use": it maps task (eval / SFT / RL, with or without a value model, with or without process-reward/RLHF) and model family to a specific YAML filename in `configs/`, and states the three fields every config must set (model name, dataset, node count) [8]. It separately documents two environment requirements: multimodal (MMaDA/LLaDA-V) training needs two separate Conda environments, one for general training/inference and one running a vLLM build that supports Qwen3-VL-MoE for evaluation and generation; process-reward/RLHF work on TraDo or SDAR models instead needs a single environment where vLLM and JetEngine coexist, with an example known-working combination of `vllm==0.8.5.post1, torch==2.6.0, python3.10, triton==3.2.0, flash-attn==2.7.4.post1, flashinfer-python==0.3.1` [8].
- `data/` (not fully read for this card beyond its top-level README) documents `download_data.py --dataset <name>` for both evaluation sets (MATH500, GSM8K, AIME2024, LiveCodeBench, LiveBench, MBPP, HumanEval) and training sets (MATH_train, PrimeIntellect for RL; demon_openr1math for SFT, requiring a `preprocess_sft.ipynb` preprocessing step first), plus the JSON schema a custom dataset must match for math and coding tasks [22].
- Runnable references beyond the README: `sample/trace_viewer.html`, a static HTML trajectory viewer opened directly in a browser, with `sample/get_trace_viewer.py` to generate a trajectory for a custom run - the README's own instructions name it `./sample/trace.viewer.html`, but the file in the repository tree at this commit is `trace_viewer.html` (underscore, no second dot), a naming mismatch worth checking before following the README literally [2][9].
- Community layer: none found - there is no curated tutorials page, and no recurring third-party blog coverage was located while researching this card; the project's own blog post (linked from the README's badge row) at `https://yinjjiew.github.io/projects/dllmrl/` is maintainer-authored, not community content, and was not fetched for this card.
- No MCP endpoint or official chat/search interface over these docs was found or expected, given there is no hosted docs site.
- Honest boundary: this is a paper-release research codebase, not a packaged library - there is no PyPI package, no version tags, no stable API contract, and (per the Install field above) at least one README instruction (`requirements_v.txt`) that does not match the repository tree at this commit [11][12][15]. It targets diffusion language models specifically and states no support for autoregressive model architectures.

## Sources

All GitHub pages and raw files are read at commit `10b4fd1cba83e4496c20585dc2fd01dc02f20255` (the repository's default-branch HEAD at fetch time) unless a different resource is named; fetched 2026-08-12, except the issue #45 read, fetched 2026-08-12. There is no docs site, no PyPI package, and no version tag for this repository, so no "pinned release" claim is made anywhere in this card - see the Install field for why.

[1] GitHub repository API for Gen-Verse/dLLM-RL. https://api.github.com/repos/Gen-Verse/dLLM-RL. Fetched 2026-08-12.

[2] README.md at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/README.md. Fetched 2026-08-12.

[3] arXiv abstract page for the TraceRL paper. https://arxiv.org/abs/2509.06949. Fetched 2026-08-12.

[4] rl.py at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/rl.py. Fetched 2026-08-12.

[5] train/sft_trado.py at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/train/sft_trado.py. Fetched 2026-08-12.

[6] configs/rl_trado.yaml at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/configs/rl_trado.yaml. Fetched 2026-08-12.

[7] reward/reward.py at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/reward/reward.py. Fetched 2026-08-12.

[8] configs/readme.md at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/configs/readme.md. Fetched 2026-08-12.

[9] Git tree listing at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255 (recursive). https://api.github.com/repos/Gen-Verse/dLLM-RL/git/trees/10b4fd1cba83e4496c20585dc2fd01dc02f20255?recursive=1. Fetched 2026-08-12.

[10] LICENSE at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/LICENSE. Fetched 2026-08-12.

[11] GitHub Releases API for Gen-Verse/dLLM-RL (empty). https://api.github.com/repos/Gen-Verse/dLLM-RL/releases. Fetched 2026-08-12.

[12] GitHub Tags API for Gen-Verse/dLLM-RL (empty). https://api.github.com/repos/Gen-Verse/dLLM-RL/tags. Fetched 2026-08-12.

[13] Commit API for 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://api.github.com/repos/Gen-Verse/dLLM-RL/commits/10b4fd1cba83e4496c20585dc2fd01dc02f20255. Fetched 2026-08-12.

[14] requirements.txt at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/requirements.txt. Fetched 2026-08-12.

[15] Direct fetch of requirements_v.txt at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255 (returns 404, confirming absence). https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/requirements_v.txt. Fetched 2026-08-12.

[16] assets/maintable.png at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255 (Table 2, main benchmark results). https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/assets/maintable.png. Fetched 2026-08-12.

[17] configs/sft_trado.yaml at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/configs/sft_trado.yaml. Fetched 2026-08-12.

[18] configs/readme.md, "Some tips" section, same fetch as [8]. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/configs/readme.md. Fetched 2026-08-12.

[19] train/rl_trado.py at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/train/rl_trado.py. Fetched 2026-08-12.

[20] GitHub issue #41, "Visualization of reward and training loss, with logging" (opened 2025-11-12, closed), maintainer reply by yinjjiew (association: MEMBER), 2025-11-15. https://api.github.com/repos/Gen-Verse/dLLM-RL/issues/41 and https://api.github.com/repos/Gen-Verse/dLLM-RL/issues/41/comments. Fetched 2026-08-12.

[21] GitHub issue #45, "Oscillations in the LOSS curve during training" (opened 2025-12-04, closed), maintainer replies by yinjjiew (association: MEMBER), 2025-12-05 and 2025-12-07. https://api.github.com/repos/Gen-Verse/dLLM-RL/issues/45 and https://api.github.com/repos/Gen-Verse/dLLM-RL/issues/45/comments. Fetched 2026-08-12.

[22] data/readme.md at commit 10b4fd1cba83e4496c20585dc2fd01dc02f20255. https://raw.githubusercontent.com/Gen-Verse/dLLM-RL/10b4fd1cba83e4496c20585dc2fd01dc02f20255/data/readme.md. Fetched 2026-08-12.
