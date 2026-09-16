# Colossal-AI / ColossalChat

A large-scale parallel-training system whose RLHF application, ColossalChat, wraps SFT, reward modeling, PPO, GRPO, DPO/SimPO, ORPO and KTO trainers around Colossal-AI's own sharding plugins.

**Colossal-AI** describes itself on its GitHub README tagline as "Making large AI models cheaper, faster, and more accessible" [1]. It is built and maintained by HPC-AI Tech (the ColossalAI Team) [1][2]; the RLHF post-training layer, ColossalChat, describes itself as "a project to implement LLM with RLHF, powered by the Colossal-AI" [3]. The API shape is two-layered: a generic `Booster` class wraps a model/optimizer pair with a chosen parallelism `Plugin` [4], and on top of that ColossalChat exposes one trainer per method, each launched from a per-method shell script and Python training script (module names and citations in Methods it ships and Quick start, below). It lives at https://github.com/hpcaitech/ColossalAI [1].

**When to pick it**: pick it when the parallelism strategy itself is the reason to choose a framework - Colossal-AI's own plugin-selection guidance runs from data-parallel-only up to a documented, GPU-count-scaled tensor+pipeline+ZeRO combination for 60B+ parameter models on thousand-card clusters [5]. Weigh this against a real limitation stated by the project's own Algorithm Lead: as of a 2025-02-21 issue reply, the RLHF (PPO/GRPO) pipeline was still "under intensive development" with documentation the team called insufficiently detailed for reproduction [6], and the shipped GRPO/PPO code is explicitly marked experimental (below). Pick ColossalChat when you specifically need Colossal-AI's tensor/pipeline-parallel sharding for SFT or reward modeling, which carry no such maturity warning; weigh the PPO/GRPO immaturity above before picking it for online RL specifically.

**Methods it ships**, grouped by ColossalChat's own README table of contents [7]:
- RLHF stage 1 - SFT: `SFTTrainer` (`coati/trainer/sft.py`) [8].
- RLHF stage 2 - reward modeling: `RewardModelTrainer` (`coati/trainer/rm.py`) [8].
- RLHF stage 3 - online RL: `PPOTrainer` (`coati/trainer/ppo.py`); alternative `GRPOTrainer` (`coati/trainer/grpo.py`) [8][7].
- Offline alignment alternatives: `DPOTrainer` (`coati/trainer/dpo.py`, also drives SimPO by setting `loss_type` to `simpo_loss`) [7]; `ORPOTrainer` (`coati/trainer/orpo.py`); `KTOTrainer` (`coati/trainer/kto.py`) [8][7].
- The README states plainly that the PPO and GRPO pipelines are "still under extensive development (integration with Ray and the inference engine)", that rollout uses "a naive generation approach without any acceleration", and that the released GRPO experiment is "focused solely on verifying the correctness of the GRPO algorithm" rather than being a production-ready fast path [9]. The maturity warning lives in the README text quoted above, not in a separate package path or namespace - `PPOTrainer` and `GRPOTrainer` import from the same top-level `coati.trainer` module as the stable trainers [8] - so recheck [7] and [9] before relying on PPO/GRPO speed or docs completeness.

**Scale it handles**: single GPU, single node with multiple GPUs, and multi-node, all through the `colossalai run` launcher (a wrapper over the PyTorch distributed launch utility) with a `--hostfile` for multi-node [10][5]. Parallelism is chosen by picking a Booster plugin: `TorchDDPPlugin` for models under ~2B parameters; `TorchFSDPPlugin`/`LowLevelZeroPlugin` (ZeRO stage 1/2) for under ~10B; `GeminiPlugin` (chunk-based ZeRO-3) for 10B+ on small-to-medium clusters; `HybridParallelPlugin` (tensor + pipeline + ZeRO 1/2 combined) for 60B+ on clusters of "a thousand cards or more" [5] - this last figure is the library's own scale guidance, not a measured benchmark, so treat it as documented mechanism rather than a published multi-node run. The one published reference-run table that does exist is single/few-GPU VRAM, not multi-node throughput (below).

**Install**: `pip install colossalai`; PyPI lists version 0.5.0 uploaded 2025-06-04 [11]; the GitHub release for the same tag was published 2025-06-04T06:00:47Z [12]; licence Apache-2.0 [1][11]. At the `v0.5.0` tag (commit `d322ff8`) [13], `requirements/requirements.txt` pins `torch>=2.2.0,<=2.5.1`, `transformers==4.51.3`, `peft>=0.7.1,<=0.13.2`, `bitsandbytes>=0.39.0`, `diffusers==0.29.0` [14]; the ColossalChat application adds its own `applications/ColossalChat/requirements.txt` pinning `transformers==4.39.3` (older than the core package's own floor above - the two requirement files disagree on transformers, so install ColossalChat's own file when running it) and `torch>=2.1.0` [15]. The current `main` branch, at commit `4f9953b` (2026-04-09, the latest commit on the default branch as of this reading) [16], still reports `version.txt` as `0.5.0` and carries the identical `requirements/requirements.txt` pins listed above [17] - so despite the repository having pushed commits well after the v0.5.0 tag, no new release has superseded it and the install line above is current. Colossal-AI's own installation page states hardware/software floors that PyPI's metadata does not: PyTorch >= 2.1, Python >= 3.7, CUDA >= 11.0, NVIDIA GPU compute capability >= 7.0 (V100/RTX20 and higher), and Linux only [18] (read 2026-08-10, unversioned live page).

**Maintained by**: HPC-AI Tech / the ColossalAI Team [1]; the repository is not archived and its default branch was still receiving commits as of 2026-04-09 [16]; ColossalChat's README names an eight-person core author list plus two PhD-student contributors, including the "Algorithm Lead" who personally answered a GRPO reproducibility issue in 2025 [19][6].

## Quick start

Smallest complete runs, from ColossalChat's examples README and training-script arguments [20]:

```bash
colossalai run --nproc_per_node 4 --master_port 28534 --hostfile ./hostfile train_sft.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --dataset ${dataset[@]} \
    --save_interval 5000 \
    --save_path $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --plugin gemini \
    --batch_size 4 \
    --max_epochs 1 \
    --accumulation_steps 1 \
    --lr 2e-5 \
    --max_len 2048 \
    --use_wandb
```

For PPO, the unique arguments on top of the shared training configuration are the reward-model path and the RL episode/batch schedule:

```bash
--pretrain $PRETRAINED_MODEL_PATH \
--rm_pretrain $PRETRAINED_MODEL_PATH \
--tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
--rm_checkpoint_path $REWARD_MODEL_PATH \
--prompt_dataset ${prompt_dataset[@]} \
--conversation_template_config $CONVERSATION_TEMPLATE_CONFIG_PATH \
--pretrain_dataset ${ptx_dataset[@]} \
--ptx_batch_size 1 \
--ptx_coef 0.0 \
--num_episodes 2000 \
--num_collect_steps 1 \
--num_update_steps 1 \
--experience_batch_size 8 \
--train_batch_size 4 \
--accumulation_steps 2
```
[20]

Each of these is a shell wrapper (`applications/ColossalChat/examples/training_scripts/train_sft.sh`, `train_ppo.sh`, `train_grpo.sh`, `train_dpo.sh`, `train_orpo.sh`) around the corresponding `train_*.py` script and Booster-based trainer class [20].

## Start it

- One process, one GPU: run the training script directly; `colossalai run` is only needed once you have more than one process.
- Single-node, multi-GPU: `colossalai run --nproc_per_node 4 train.py` (default port 29500, override with `--master_port`) [10].
- Multi-node: `colossalai run --nproc_per_node <per-node-GPUs> --hostfile <hostfile> train.py`, where the hostfile lists one hostname per line and the master node must be able to SSH to every node (including itself) without a password [21][10]. `colossalai run` is documented as a wrapper of the torch distributed launch utility "enhanced with the capability of launching multi-node jobs easily" [10].
- Effective batch/experience-buffer arithmetic for PPO: without tensor parallelism, `experience buffer size = num_process * num_collect_steps * experience_batch_size = train_batch_size * accumulation_steps * num_process`; with tensor parallelism of degree `tp`, replace `num_process` with `num_process / tp` on both sides [20]. GRPO's rollout multiplies this further by `num_generations` (the group size sampled per prompt, "usually greater than 8") [22].
- Configuration surface is CLI flags per training script, not a single Config class: `--plugin` selects the Booster plugin (`ddp`, `gemini`, `gemini_auto`, `zero2`, `zero2_cpu`, tensor-parallel `HybridParallelPlugin` variants) [23]; `--mixed_precision` supports `'fp16'` and `'bf16'`, with the README warning that "some devices may not support the 'bf16' option"; the flag is opt-in per run rather than defaulted by the library [24]. Gemini and Gemini-Auto plugins explicitly do not support local gradient accumulation, so `--accumulation_steps` must stay at 1 under those plugins [23].
- GRPO-specific generation knobs: `--num_generations` (rollouts per prompt), `--inference_batch_size` (rollout batch size), `--logits_forward_batch_size` (batch size for the separate logits pass), `--initial_temperature`/`--final_temperature` for a temperature-annealing schedule; the README notes rebatching happens "to prevent out of memory both before roll out and before calculating logits", and to tune `inference_batch_size` and `logits_forward_batch_size` for your device [22].
- OOM first aid: the README's PPO hardware-requirements table is the published reference (7B model, llama2-7B-hf, sequence length 2048, rollout length 512, H800 80GB GPUs) - tensor-parallel degree 8 stays under 65GB up to batch size 30 (18485MB at bs=1 to 64047MB at bs=30), while tensor-parallel degree 4 needs 42934-56779MB at bs=1-16 and fails outright at bs=30, showing tp=8 as the OOM fix over tp=4 at this model size [25]. For SFT, the same table shows `zero2-cpu` using well under half the VRAM of plain `zero2` at equal batch size - on the 2-GPU row, micro batch size 4: 22458MB for `zero2_cpu` versus 72391MB for `zero2` (roughly a third); on the 4-GPU row, micro batch size 8: 19413MB for `zero2_cpu` versus 43446MB for `zero2` (roughly 45%) - so switching `--plugin` to a `_cpu` ZeRO-2 variant is the documented lever, with the exact savings ratio varying by GPU count and batch size [25].

## Watch it

This section is mechanics only; what a given curve or ratio means for a specific method belongs on that method's own card.

- **Enable it**: pass `--use_wandb` on the training script to log to Weights & Biases; the SFT trainer's constructor asserts a `log_dir` must also be given when `use_wandb` is set, and independently of wandb, passing `log_dir` alone turns on a TensorBoard `SummaryWriter` under `<log_dir>/sft/<timestamp>` [26]. Reading the trainer source directly (commit `d322ff8`, the v0.5.0 tag): `SFTTrainer.__init__` calls `wandb.init(project="Coati-sft", sync_tensorboard=True)` when enabled [26]. Without `--use_wandb` and without `--log_dir`, a ColossalChat run writes no training-curve record at all.
- **Metric names by trainer** (read from `coati/trainer/*.py` at commit `d322ff8`, the v0.5.0 tag [27]):
  - SFT (`sft.py`): `train/loss`, `train/lr`.
  - Reward model (`rm.py`): `train/loss`, `train/lr`, `train/dist` (chosen-minus-rejected reward), `train/reward_chosen`, `train/reward_reject`, `train/acc`.
  - PPO (`ppo.py`): `train/max_ratio`, `train/skip_ratio`, `train/actor_loss`, `train/lr_actor`, `train/lr_critic`, `train/critic_loss`, `train/ptx_loss` (only when `ptx_coef != 0`), `reward`, `approx_kl`, `value`, `advantages`.
  - GRPO (`grpo.py`): `train/max_ratio`, `train/skip_ratio`, `train/actor_loss`, `train/lr_actor`, `train/ptx_loss` (when `ptx_coef != 0`), `reward`, `token_cost`, `approx_kl`, `advantages` - no critic/value metrics, since GRPO has no critic.
  - DPO (`dpo.py`) and KTO (`kto.py`): `train/loss`, `train/lr`, plus per-file reward/margin scalars (read the file directly for the exact set at your commit; not fully enumerated here since this card stops at what was read for PPO/GRPO/RM/SFT).
  - These names are not published on any docs page - the live docs site's Booster/CLI/plugin pages do not list trainer metrics, so this list is read from source, not from documentation, and should be re-read from the same files if you pin a different commit.
- **Sample-level logging**: PPO and GRPO do it, SFT and RM do not. Every 10 training steps (`self.num_train_step % 10 == 0`), `PPOTrainer` and `GRPOTrainer` batch-decode the sampled response sequences, append each response's numeric reward to its decoded text, and log the batch as a `wandb.Table` under the key `"sample_response"` when a wandb run is active; with only a TensorBoard writer active (no wandb), the same decoded-response-plus-reward lines are printed to stdout on the master process instead of being written to TensorBoard [27]. No such call exists in the SFT or reward-model trainer files read for this card, which only make scalar `add_scalar`/`wandb` calls [27].
- **Evaluation during training**: not documented on the pages read for this card; the training-script flags enumerated in the README's "Training Configuration" section list save/log/precision/batch controls but no `--eval_*` flags [24].
- **Published health limit**: the README's reward-model section states a concrete acceptance check before moving to PPO/GRPO: "check the following list to ensure that your reward model is stable and robust... The mean reward for chosen data is much higher than those for rejected data... The accuracy is larger than 0.5 by a significant margin (usually should be greater than 0.6)" [28]. The PPO section separately publishes an FAQ of four named failure shapes with fixes - negative reward (expected if the reward model itself is negative, but should still rise), negative actor loss (normal, not a bug), reward that decreases (check the reward model and hyperparameters), and degenerate ("garbage") generation from over-training, fixed by setting `--ptx_coef` to a nonzero value to blend in an SFT loss term [29]. No numeric early-stopping threshold or patience value is published anywhere in the README or the docs pages read; the honest boundary is qualitative shapes only, not thresholds.
- **A documented immaturity trap**: in closed issue #6211 ("能否提供一个可以直接运行的grpo数据集" / "can you provide a directly-runnable GRPO dataset"), a user reported `prepare_prompt_dataset.sh` failing immediately when following the GRPO README; TongLi3701, ColossalChat's own Algorithm Lead per the Authors list [19], replied on 2025-02-21 that "we are under intensive development and will release a new version soon including more concise documentation and speed optimization" [6] - i.e., as of that date the maintainers themselves acknowledged the GRPO quickstart path was not reliably reproducible from the README alone.

## Save it

- Checkpoint I/O for ColossalChat's RL trainers goes through `coati/utils/ckpt_io.py`'s `save_checkpoint`/`load_checkpoint` helpers (read at commit `d322ff8`, the v0.5.0 tag) [30]. `save_checkpoint` writes to `<save_dir>/epoch-<epoch>_step-<step>/`, containing a `modeling/` subfolder (`booster.save_model(..., shard=True)`), an `optimizer/` subfolder (`booster.save_optimizer(..., shard=True)`), an `lr_scheduler` file, and a `running_states.json` recording `epoch`, `step`, and `sample_start_index` [30]. `PPOTrainer._save_checkpoint` calls this twice per save, once for the actor and once for the critic, into separate `actor_save_dir`/`critic_save_dir` trees [31].
- `Booster.save_model(..., shard=True)` is documented as producing "a folder with the same format as Huggingface transformers checkpoint" [32] - so the `modeling/` subfolder above is directly `from_pretrained`-loadable.
- Cadence: the shared `--save_interval` flag controls how often "the model weights as well as optimizer/scheduler states" are saved [24].
- LoRA/PEFT: the `--merge_lora_weights` flag controls "whether to merge lora weights before saving the model" [24] - the README does not spell out what an unmerged adapter save looks like on disk beyond this one line, so verify the directory contents directly if you train with `--lora_rank` and skip merging.
- Resume: the training-script docs state `--checkpoint_path`, "if provided, will load weights from the checkpoint_path" [24]; at the code level, `load_checkpoint(load_dir, booster, model, optimizer, lr_scheduler)` restores model, optimizer, and LR-scheduler state and returns `(epoch, step, sample_start_index)` read back out of `running_states.json`, so resume restores optimizer state, not just weights [30].
- Loader handoff: because the sharded model save is documented as HuggingFace-`from_pretrained`-compatible [32], the `modeling/` output of a ColossalChat save can be loaded directly by any evaluator that accepts a standard `transformers` checkpoint directory; the optimizer/scheduler/`running_states.json` alongside it are for resuming training, not for evaluation loading.

## Find it in the docs

The docs at `colossalai.org` are unversioned and live - fetching `https://colossalai.org/docs/v0.5.0/get_started/installation` returns 404, confirming there is no per-release doc tree to pin to; every docs claim on this card is dated to the fetch, not to a release tag [33].

- Address pattern: `https://colossalai.org/docs/<section>/<page>` (English) or `https://colossalai.org/zh-Hans/docs/<section>/<page>` (Chinese); confirmed working forms include `/docs/get_started/installation`, `/docs/basics/command_line_tool`, `/docs/basics/launch_colossalai`, `/docs/basics/booster_plugins`, `/docs/basics/booster_checkpoint` [18][34][10][5][32] (all fetched 2026-08-10).
- Question-to-page map: "how do I install / what hardware do I need" -> `get_started/installation` [18]; "how do I launch on N GPUs or multiple nodes" -> `basics/launch_colossalai`, with `basics/command_line_tool` as the CLI-flags overview [10][34]; "which plugin/parallelism strategy should I use" -> `basics/booster_plugins`, which is also where the parameter-count scale guidance quoted above lives [5]; "how do I save/load a checkpoint" -> `basics/booster_checkpoint` [32].
- Runnable references beyond the docs: the RLHF-specific quickstarts, hardware-requirement tables, and troubleshooting FAQ all live in `applications/ColossalChat/examples/README.md`, a separate file from the top-level `applications/ColossalChat/README.md` [3][7] - the examples file, not the docs site, is where the operational content in this card's Quick start, Start it, and Watch it sections comes from (read at the `v0.5.0` tag) [20]; the `applications/ColossalChat/examples/training_scripts/` and `applications/ColossalChat/examples/data_preparation_scripts/` directories hold the runnable shell wrappers and dataset-prep scripts that file references [20].
- Community layer: the repo's own README curates a "Blog" link (`hpc-ai.com/blog`) and a GitHub Discussions forum as its official pointers [1]. No curated community-tutorials hub was found linked from the docs nav (nav items observed on every docs page fetched: "Tutorials", "Examples", "Blogs" [18]), so the vendor blog and Discussions are the curated door here.
- No official MCP endpoint for querying these docs was found on the pages read for this card.
- Honest boundary, stated where a reader would hit it: the PPO/GRPO pipeline itself is the boundary - both are explicitly flagged as under active development with a "naive generation approach without any acceleration" [9], and the maintainers' own 2025-02-21 issue reply confirms the RLHF README was not reliably reproducible as written at that time [6]. For stable, well-documented offline methods (SFT, reward modeling, DPO/ORPO/KTO) this caveat does not apply to the same degree.

## Sources

All docs pages are unversioned/live and were fetched 2026-08-10 unless a different date is given; code citations name the commit read. Method names (SFT, DPO, GRPO, ORPO, KTO, PPO, SimPO) are deliberately cited to nothing here - their defining papers live on the methodology cards.

[1] Colossal-AI GitHub repository README. https://github.com/hpcaitech/ColossalAI. Fetched 2026-08-10.

[2] Colossal-AI repository metadata (owner, description, license, archived status). https://api.github.com/repos/hpcaitech/ColossalAI. Fetched 2026-08-10.

[3] ColossalChat README, "What is ColossalChat?" section, read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/README.md. Fetched 2026-08-10.

[4] Colossal-AI Booster API docs, description of Booster as "a high-level API for training neural networks" that "provides a unified interface" over plugins. https://colossalai.org/docs/basics/booster_api/. Fetched 2026-08-10.

[5] Colossal-AI Booster Plugins docs (plugin descriptions and the parameter-count/cluster-size selection guidance). https://colossalai.org/docs/basics/booster_plugins/. Fetched 2026-08-10.

[6] GitHub issue #6211 comment thread, reply from TongLi3701 (author association: CONTRIBUTOR per the GitHub API), 2025-02-21. https://github.com/hpcaitech/ColossalAI/issues/6211. Fetched 2026-08-10.

[7] ColossalChat README table of contents and per-method sections, read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/README.md. Fetched 2026-08-10.

[8] `applications/ColossalChat/coati/trainer/` directory listing at the `v0.5.0` tag (file names: `sft.py`, `rm.py`, `ppo.py`, `grpo.py`, `dpo.py`, `orpo.py`, `kto.py`, `base.py`, `utils.py`, `callbacks/`). https://api.github.com/repos/hpcaitech/ColossalAI/contents/applications/ColossalChat/coati/trainer?ref=v0.5.0. Fetched 2026-08-10.

[9] ColossalChat examples README, "GRPO Training and DeepSeek R1 reproduction" section, development-status note, read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/examples/README.md. Fetched 2026-08-10.

[10] Colossal-AI "Launch Colossal-AI" docs (`colossalai run`, single- and multi-node launch forms). https://colossalai.org/docs/basics/launch_colossalai/. Fetched 2026-08-10.

[11] `colossalai` package on PyPI (version, upload date, license classifier). https://pypi.org/pypi/colossalai/json. Fetched 2026-08-10.

[12] Colossal-AI GitHub releases list (v0.5.0 published date). https://api.github.com/repos/hpcaitech/ColossalAI/releases. Fetched 2026-08-10.

[13] Colossal-AI `v0.5.0` git tag resolved to its commit SHA (`d322ff8...`). https://api.github.com/repos/hpcaitech/ColossalAI/git/refs/tags/v0.5.0. Fetched 2026-08-10.

[14] `requirements/requirements.txt` at the `v0.5.0` tag. https://raw.githubusercontent.com/hpcaitech/ColossalAI/v0.5.0/requirements/requirements.txt. Fetched 2026-08-10.

[15] `applications/ColossalChat/requirements.txt` at the `v0.5.0` tag. https://raw.githubusercontent.com/hpcaitech/ColossalAI/v0.5.0/applications/ColossalChat/requirements.txt. Fetched 2026-08-10.

[16] Colossal-AI `main` branch HEAD commit (`4f9953b`, dated 2026-04-09). https://api.github.com/repos/hpcaitech/ColossalAI/commits/main. Fetched 2026-08-10.

[17] `version.txt` and `requirements/requirements.txt` on the `main` branch, compared against the `v0.5.0` tag versions. https://raw.githubusercontent.com/hpcaitech/ColossalAI/main/version.txt and https://raw.githubusercontent.com/hpcaitech/ColossalAI/main/requirements/requirements.txt. Fetched 2026-08-10.

[18] Colossal-AI "Setup" / installation docs (hardware and software floors). https://colossalai.org/docs/get_started/installation/. Fetched 2026-08-10.

[19] ColossalChat README "Authors" section, read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/README.md. Fetched 2026-08-10.

[20] ColossalChat examples README, PPO "Step 3: Training" section (unique PPO arguments, experience-buffer arithmetic), read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/examples/README.md. Fetched 2026-08-10.

[21] ColossalChat examples README, "Get Start with ColossalRun" section (hostfile format, passwordless-SSH requirement), read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/examples/README.md. Fetched 2026-08-10.

[22] ColossalChat examples README, GRPO "Step 2: Training" section (generation knobs, effective-batch arithmetic with `num_generations`), read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/examples/README.md. Fetched 2026-08-10.

[23] ColossalChat examples README, "Training Configuration" section, Gemini/Gemini-Auto plugin subsections, read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/examples/README.md. Fetched 2026-08-10.

[24] ColossalChat examples README, "Training Configuration" flag-glossary list (`save_interval`, `mixed_precision`, `merge_lora_weights`, `checkpoint_path`, `log_dir`, `use_wandb`, etc.), read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/examples/README.md. Fetched 2026-08-10.

[25] ColossalChat examples README, "Hardware Requirements" section (SFT and PPO VRAM-by-plugin/tp-size/batch-size tables), read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/examples/README.md. Fetched 2026-08-10.

[26] `applications/ColossalChat/coati/trainer/sft.py`, `SFTTrainer.__init__` (wandb/TensorBoard setup), read at commit `d322ff8` (the `v0.5.0` tag). https://raw.githubusercontent.com/hpcaitech/ColossalAI/v0.5.0/applications/ColossalChat/coati/trainer/sft.py. Fetched 2026-08-10.

[27] `applications/ColossalChat/coati/trainer/{sft,rm,ppo,grpo}.py`, `add_scalar`/`wandb` logging call sites, read at commit `d322ff8` (the `v0.5.0` tag). https://raw.githubusercontent.com/hpcaitech/ColossalAI/v0.5.0/applications/ColossalChat/coati/trainer/. Fetched 2026-08-10.

[28] ColossalChat examples README, "Note on Reward Model Training" section, read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/examples/README.md. Fetched 2026-08-10.

[29] ColossalChat examples README, "Note on PPO Training" Q&A section, read at the `v0.5.0` tag. https://github.com/hpcaitech/ColossalAI/blob/v0.5.0/applications/ColossalChat/examples/README.md. Fetched 2026-08-10.

[30] `applications/ColossalChat/coati/utils/ckpt_io.py`, `save_checkpoint`/`load_checkpoint` functions, read at commit `d322ff8` (the `v0.5.0` tag). https://raw.githubusercontent.com/hpcaitech/ColossalAI/v0.5.0/applications/ColossalChat/coati/utils/ckpt_io.py. Fetched 2026-08-10.

[31] `applications/ColossalChat/coati/trainer/ppo.py`, `PPOTrainer._save_checkpoint`, read at commit `d322ff8` (the `v0.5.0` tag). https://raw.githubusercontent.com/hpcaitech/ColossalAI/v0.5.0/applications/ColossalChat/coati/trainer/ppo.py. Fetched 2026-08-10.

[32] Colossal-AI "Booster Checkpoint" docs (`save_model`/`load_model` signatures, HuggingFace-compatible sharded format). https://colossalai.org/docs/basics/booster_checkpoint/. Fetched 2026-08-10.

[33] Confirmation that a versioned docs path 404s (`https://colossalai.org/docs/v0.5.0/get_started/installation`), establishing the docs as unversioned/live. Fetched 2026-08-10.

[34] Colossal-AI "Command Line Tool" docs (`colossalai check`, `colossalai run` overview). https://colossalai.org/docs/basics/command_line_tool/. Fetched 2026-08-10.
