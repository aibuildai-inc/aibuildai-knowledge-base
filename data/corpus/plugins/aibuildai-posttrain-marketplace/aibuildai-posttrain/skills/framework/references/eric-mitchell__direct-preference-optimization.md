# direct-preference-optimization

Eric Mitchell's original reference implementation of DPO: a single-file, single-repo research script (no PyPI package, no docs site) that runs SFT then DPO on one GPU or a naively-parallel/FSDP-sharded multi-GPU box, with Hydra configs and optional wandb logging.

**direct-preference-optimization** is described by its own README as "a reference implementation of the DPO algorithm for training language models from preference data, as described in the paper Direct Preference Optimization: Your Language Model is Secretly a Reward Model" [1]. It is authored and maintained by Eric Mitchell, one of the DPO paper's authors [1][2]. The API is a single CLI entry point, `train.py`, driven entirely by Hydra config composition (`config/config.yaml` plus a `model=` and `loss=` group) rather than by importable trainer classes meant for reuse in other projects [1][3]. It lives at https://github.com/eric-mitchell/direct-preference-optimization [2].

**When to pick it**: the historical reference to check a DPO re-implementation against, or to run the exact two-stage SFT-then-DPO recipe from the original paper on a Pythia/GPT-2/Llama model with minimal machinery - not a library to build a production or multi-method pipeline on. It ships far fewer methods than trl (DPO, conservative DPO, IPO, and SFT only, versus trl's 16 trainers [1][4]) and, unlike trl or verl, has no PyPI package, no Hub trainer-class integration, and no documented multi-node path (cross-reference; not covered here). The repository has had no new commits since 2023-12-14 [5], so treat it as a frozen artifact rather than an actively developed framework.

**Methods it ships**: DPO (`loss=dpo`, config at `config/loss/dpo.yaml`, with `loss.beta` required and `loss.label_smoothing` giving "conservative" DPO - a noisy-preference variant - at nonzero values) and SFT (`loss=sft`, config at `config/loss/sft.yaml`, which holds only a `name: sft` key) [3][6]. The README also documents an IPO loss (`loss=ipo`, plus `loss.beta`) and says conservative DPO needs no separate config beyond `loss.label_smoothing` [1]; the `preference_loss` function in `trainers.py` does implement the IPO loss term as an `ipo: bool` branch [7]. But no `config/loss/ipo.yaml` file exists in the repository at the pinned commit, and the commit that added the IPO and conservative-DPO code ("Added conservative DPO and IPO losses.", 2023-11-25) modified README.md, `config/loss/dpo.yaml`, `train.py`, and `trainers.py` without adding any new file under `config/loss/` [8][3]. Hydra's config-group override (`loss=ipo`) needs a matching YAML in `config/loss/`, so the README's own `loss=ipo` command is not runnable as written against this repo state - a reader wanting IPO must write `config/loss/ipo.yaml` themselves (`name: ipo`, plus a `beta` key) before the command in [1] will work.

**Scale it handles**: single GPU, or multiple GPUs on one machine only. `BasicTrainer` naively splits the model layer-by-layer across available GPUs "effectively offering N times available memory, but without any parallel computation" [1]. `FSDPTrainer` shards each transformer block (named via `model.block_name`, e.g. `GPT2Block`, `GPTNeoXLayer`, `LlamaDecoderLayer`) across GPUs with PyTorch FSDP and is the recommended choice for real speedups, needing `ulimit -n 64000` before launch [1]. `TensorParallelTrainer` shards linear layers via the third-party `tensor_parallel` package and is marked experimental, with slow sampling [1]. There is no multi-node launcher: `train.py` spawns one process per local GPU with `torch.multiprocessing.spawn` and calls `init_distributed(rank, world_size, port=config.fsdp_port)` without passing a `master_addr`, so it falls back to that function's own default of `master_addr='localhost'` defined in `utils.py`, which only reaches processes on one machine [9][20]. A user-filed issue on multi-node training (#33) shows a community member patching in `torchrun`-based multi-node support themselves; it was never merged, and the maintainer's own reply only asked whether the single-node fix carried over, without endorsing multi-node as supported [10].

**Install**: no PyPI package - clone the repo and run `pip install -r requirements.txt` inside a fresh virtualenv; the README states "Python 3.8+ is recommended" [1]. Licence: Apache-2.0 [2]. `requirements.txt` at the pinned commit pins exact versions with `==` throughout: `torch==2.0.1`, `transformers==4.29.2`, `datasets==2.12.0`, `tokenizers==0.13.3`, `wandb==0.15.3`, `hydra-core==1.3.2`, `numpy==1.24.3`, `beautifulsoup4==4.12.2`, `ipykernel==6.23.1`, `tqdm==4.65.0`, and `tensor-parallel==1.2.4` (needed only for `TensorParallelTrainer`) [11]. No CUDA or GPU-count minimum is stated anywhere in the README or requirements file; the README's own worked example was run "on a machine with 4 80GB A100s", but that is a reported example run, not a documented floor [1].

**Maintained by**: Eric Mitchell, individually, not an organization [2]. No commits since 2023-12-14T19:47:13Z, the "Updated bibtex entry for NeurIPS" commit that is also the repository's current HEAD and the pinned commit for this card [5]; the GitHub API's `pushed_at` shows a later timestamp (2024-08-11T00:24:06Z) with no corresponding new commit on `main` and no tags, so that later push did not change the code [12][5].

## Quick start

Complete example commands quoted from the README's "complete example" walkthrough, training Pythia-2.8B on Anthropic-HH [1]:

Step 1 - install:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Step 2 - run SFT (single command, no separate script needed):
```
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
```

Step 3 - run DPO from the SFT checkpoint:
```
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/sft/LATEST/policy.pt
```

The README reports this example took "about 1hr 30min" for SFT and "about 2hrs 45min" for DPO on 4x80GB A100s [1]. Sample wandb output for both runs is linked from the README under the `readme-example` tag [1].

## Start it

- Single GPU: run `train.py` with `trainer=BasicTrainer` (the config default); `train.py` sets `device_map='balanced'` whenever `config.trainer == 'BasicTrainer'` (regardless of GPU count), letting `transformers` spread the model across whatever devices are visible [3][13].
- Multiple GPUs, one machine: pass `trainer=FSDPTrainer` (recommended for speed, needs `model.block_name` and `ulimit -n 64000` first) or `trainer=BasicTrainer` (naive layer split across GPUs, no compute speedup) or the experimental `trainer=TensorParallelTrainer` [1]. There is no config template system for launcher flags beyond these three `trainer=` choices and the Hydra config itself; the "launcher" is Python's own `torch.multiprocessing.spawn`, one worker per local GPU, invoked automatically inside `train.py` when `'FSDP' in config.trainer` [9].
- Effective batch: `batch_size` is the global per-step batch; for FSDP, the per-GPU batch is `batch_size / (gradient_accumulation_steps * num_gpus)` [3][14].
- Config surface: all training hyperparameters live in `config/config.yaml` (batch sizes, `lr=5e-7`, `warmup_steps=150`, `max_grad_norm=10.0`, `max_length=512`, `n_epochs=1`, `eval_every=20_000` examples, optimizer `RMSprop`) [14]; model choice and dtypes in `config/model/*.yaml`; loss choice in `config/loss/*.yaml`. The library's own non-default choice worth flagging: it uses `RMSprop`, not Adam, as the default optimizer, with the config comment explaining "we use RMSprop because it works about as well as Adam and is more memory-efficient" [14].
- Precision: `model.policy_dtype` defaults to `float32` and `model.reference_dtype` to `float16` in every shipped model config [15]. A closed maintainer-answered issue (#19) confirms that setting the policy to `float16` commonly produces `nan` metrics because of overflow, and recommends `bfloat16` instead - "You might have better luck with bfloat16" (eric-mitchell, OWNER, 2023-07-19) [16]. FSDP mixed precision is set separately via `model.fsdp_policy_mp` (only `bfloat16` has been tested per the README) [1].
- Out-of-memory first aid: increase `gradient_accumulation_steps` to shrink the per-GPU microbatch, and/or set `activation_checkpointing=true` (FSDP only) [1]. A closed, maintainer-answered issue (#11) found that `Killed`/OOM crashes under `FSDPTrainer` were often not GPU memory but the CPU RAM reserved by the job scheduler: "the issue is not the amount of RAM free on the machine, it's the amount of RAM your slurm job actually requests," and that "very conservatively, running llama on 4 GPUs should be fine with ~200GB RAM reserved" (eric-mitchell, OWNER, 2023-07-07) [17].

## Watch it

This section is mechanics only; what a metric value means for DPO training - healthy reward margins, accuracy trends - lives on the DPO methodology card.

- Enable it: `wandb.enabled: true` by default in `config/config.yaml`, logging to the `direct-preference-optimization` project unless overridden; `wandb.entity` must be set for a team account [14]. Setting `config.debug=true` disables both wandb `init` and `log` calls entirely [9]. There is no non-wandb tracker integration; if `wandb.enabled=false` and `debug` is unset, training still runs but nothing is logged to an external dashboard - only `rank0_print` console output remains [9].
- Metric names, read directly from `trainers.py`'s `get_batch_metrics` and `train` methods [7]: for DPO/IPO, `rewards_train/chosen`, `rewards_train/rejected`, `rewards_train/accuracies`, `rewards_train/margins` (and the `_eval/` equivalents during evaluation), plus `logps_train/rejected` and `logps_train/chosen`; for both SFT and DPO/IPO, `loss/train` (or `loss/eval`); and per training step, `examples_per_second`, `grad_norm`, `counters/examples`, `counters/updates`.
- Sample-level logging: when `sample_during_eval=true` (the config default), each evaluation pass generates `n_eval_model_samples` completions from the policy (and, for DPO/IPO, the reference model too) and logs them to wandb as `wandb.Table` objects named `policy_samples` and `reference_samples`, plus prints the first 10 to the console [7][14].
- Evaluation during training: controlled by `eval_every` (in examples, default 20,000; `main()` rounds it down to a multiple of `batch_size` if needed, printing a warning) and `n_eval_examples` (default 256) [14][9]. `do_first_eval=true` by default runs one evaluation pass before any training step [14].
- No stopping rule is published. Neither `config/config.yaml` nor `train.py` nor `trainers.py` defines an early-stopping flag, patience value, or reward/loss threshold that halts training automatically; `n_epochs` (default 1) or `n_examples` are the only ways training ends, and the loop otherwise runs to the end of the configured data [14][9]. Search performed 2026-08-11 over these three files, the only ones defining the training loop and its config.

## Save it

- Checkpoints land under the run directory (`config.local_run_dir`, itself under the first existing path in `local_dirs: [/scr-ssd, /scr, .cache]`, per-user subfolder) [14][3]. During training, periodic checkpoints go to `<run_dir>/step-<example_counter>/`; `save()` with no explicit `output_dir` writes to `<run_dir>/LATEST/` [7]. Each checkpoint directory holds three separate files, one per state dict: `policy.pt`, `optimizer.pt`, `scheduler.pt`, each a `torch.save`'d dict with keys `step_idx`, `state`, and `metrics` [7]. There is no retention-limiting flag (no equivalent of a `save_total_limit`) - every evaluation-triggered save is kept unless the caller manages disk space externally [7].
- `debug=true` disables checkpoint saving entirely ("skipping save in debug mode") [7].
- `FSDPTrainer.save()` gathers the full (unsharded) state dict to rank 0 via `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` before writing, so `policy.pt` from an FSDP run is a standard, non-sharded state dict [7]. `TensorParallelTrainer.save()` similarly un-shards via `tp.save_tensor_parallel` before writing only `policy.pt` (no optimizer/scheduler save) [7].
- To resume or reuse a checkpoint, pass `model.archive=/path/to/step-XXXX/policy.pt` (or `.../LATEST/policy.pt`); `train.py` loads it with `torch.load(...)['state']` into `policy.load_state_dict(...)`, and into the reference model too when doing DPO/IPO [9]. This only restores model weights, not optimizer or scheduler state or the step counter as training state - a fresh `train.py` invocation with `model.archive` set starts a new run (new `exp_name`, new step counter from 0), it does not resume the original run's counters [9].
- There is no adapter/PEFT path in this repo - every save is a full model state dict, and no LoRA or adapter-only save option exists in `trainers.py` or `config/model/*.yaml` [7][15].
- Loader handoff: `policy.pt`'s `state` dict is a raw `nn.Module.state_dict()`, loadable only via `model.load_state_dict(...)` on a matching `AutoModelForCausalLM` instance already constructed with the same `name_or_path` - there is no `save_pretrained`/`from_pretrained`-compatible directory produced by this repo, so a Hub-style loader cannot load a checkpoint here without first writing that conversion step yourself [7][9].

## Find it in the docs

There is no separate docs site for this repository; the README is the documentation, and the code itself is the second source.

- Primary reference: the README at https://github.com/eric-mitchell/direct-preference-optimization/blob/main/README.md, covering install, the two-stage SFT-then-DPO recipe, the three trainer classes, adding new datasets, and multi-GPU tuning tips [1].
- Config lookup: every tunable lives in one of `config/config.yaml` (global), `config/model/*.yaml` (per-model dtype/block settings, one file per pre-defined model - `blank_model.yaml`, `gpt2-large.yaml`, `gpt2-xl.yaml`, `gptj.yaml`, `llama7b.yaml`, `pythia28.yaml`, `pythia69.yaml`), or `config/loss/{dpo,sft}.yaml`, each with inline comments explaining the field [3][14][15][6].
- Adding a custom model: use `model=blank_model model.name_or_path=NAME_OR_PATH`, or add a new YAML file under `config/model/` following the existing examples [1].
- Adding a custom dataset: implement a `get_<name>()` function in `preference_datasets.py` returning, per prompt, a dict with `responses` (list of strings), `pairs` (list of chosen/rejected index tuples), and `sft_target` (the string used for SFT) - follow the three shipped examples, `get_hh()` (loads `Anthropic/hh-rlhf`), `get_shp()` (loads `stanfordnlp/SHP`), and `get_se()` (loads `HuggingFaceH4/stack-exchange-preferences`) - then register the name in `get_dataset()` [1][18]. These three Hub dataset ids are the known-good smoke-test data, selectable via `datasets=[hh]`, `datasets=[shp]`, `datasets=[se]`, or any combination [18][14].
- Trainer-class internals (the wrapping/sharding logic itself, not just how to select one) are documented only in the docstrings inside `trainers.py`'s `BasicTrainer`, `FSDPTrainer`, and `TensorParallelTrainer` classes [7].
- Community layer: no curated tutorials page exists for this repo. The awesome-list entries found for it are one-line pointers with no added tutorial content beyond the README itself (e.g., AbdelStark/awesome-ai-safety and drivetosouth/Awesome-SafeRL-and-RLHF both link straight to this repo as "the" DPO reference implementation) [19]. Two closed, maintainer-answered issues stand in for a FAQ: #11 on CPU-RAM-driven OOM under FSDP [17] and #19 on `float16` NaNs [16], both cited above.
- No MCP endpoint or API is published for this repository; it is source code plus a README, not a hosted docs product.

The honest boundary: this repo supports only DPO, SFT, and (via undocumented manual config) IPO/conservative DPO on a single machine; it has no multi-node launcher, no PEFT/adapter saving, no non-wandb tracker, and (per the "Maintained by" field above) no commits since 2023-12-14 - anyone needing active maintenance, more methods, or cluster-scale training should look at trl or verl instead (cross-reference; not covered here).

## Sources

[1] direct-preference-optimization README. https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/README.md, read at commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f. Fetched 2026-08-11.

[2] direct-preference-optimization GitHub repository (metadata: license, owner, URL). https://github.com/eric-mitchell/direct-preference-optimization ; https://api.github.com/repos/eric-mitchell/direct-preference-optimization. Fetched 2026-08-11.

[3] `config/config.yaml`. https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/config/config.yaml, read at commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f. Fetched 2026-08-11.

[4] trl documentation index (method count, for the comparative claim only). https://huggingface.co/docs/trl/main/en/index. Fetched 2026-08-11 (cross-reference; not covered here).

[5] repository commit history on `main`. https://api.github.com/repos/eric-mitchell/direct-preference-optimization/commits?sha=main. Fetched 2026-08-11.

[6] `config/loss/sft.yaml` and `config/loss/dpo.yaml`. https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/config/loss/sft.yaml ; https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/config/loss/dpo.yaml, read at commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f. Fetched 2026-08-11.

[7] `trainers.py`. https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/trainers.py, read at commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f. Fetched 2026-08-11.

[8] commit d4fff4967e4c47c4f50ee9e18b0fd926b3a023cf, "Added conservative DPO and IPO losses." (file list). https://api.github.com/repos/eric-mitchell/direct-preference-optimization/commits/d4fff4967e4c47c4f50ee9e18b0fd926b3a023cf. Fetched 2026-08-11.

[9] `train.py`. https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/train.py, read at commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f. Fetched 2026-08-11.

[10] Issue #33, "Multi-node optimizer save error" (comments). https://api.github.com/repos/eric-mitchell/direct-preference-optimization/issues/33/comments. Fetched 2026-08-11.

[11] `requirements.txt`. https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/requirements.txt, read at commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f. Fetched 2026-08-11.

[12] repository metadata `pushed_at` field. https://api.github.com/repos/eric-mitchell/direct-preference-optimization. Fetched 2026-08-11.

[13] `train.py` (device_map logic). Same source as [9].

[14] `config/config.yaml`. Same source as [3].

[15] `config/model/blank_model.yaml`, `config/model/pythia28.yaml`, `config/model/llama7b.yaml`, `config/model/gpt2-xl.yaml`. https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/config/model/blank_model.yaml (and sibling files under the same path), read at commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f. Fetched 2026-08-11.

[16] Issue #19 comments (float16 NaN, maintainer reply). https://api.github.com/repos/eric-mitchell/direct-preference-optimization/issues/19/comments. Fetched 2026-08-11.

[17] Issue #11 comments (CPU RAM vs. GPU OOM, maintainer reply). https://api.github.com/repos/eric-mitchell/direct-preference-optimization/issues/11/comments. Fetched 2026-08-11.

[18] `preference_datasets.py`. https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/preference_datasets.py, read at commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f. Fetched 2026-08-11.

[19] Awesome-list entries citing this repository, as recorded in the shortlist evidence for this card (AbdelStark/awesome-ai-safety, drivetosouth/Awesome-SafeRL-and-RLHF). Not independently re-fetched; content quoted is limited to what the shortlist evidence already shows.

[20] `utils.py` (`init_distributed`'s `master_addr='localhost'` default). https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/main/utils.py, read at commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f. Fetched 2026-08-11.
