# DeepSpeed

Repository: https://github.com/deepspeedai/DeepSpeed

A memory- and communication-optimized distributed training/inference engine for PyTorch models, wrapped around a model with `deepspeed.initialize()`, that scales from a single GPU to trillion-parameter, multi-node jobs through its ZeRO family of sharding stages.

DeepSpeed is "a deep learning optimization library that makes distributed training and inference easy, efficient, and effective" [1]. It is built and maintained by the DeepSpeed Team [1][2], having been renamed from `microsoft/DeepSpeed` to `deepspeedai/DeepSpeed` [3]. The API centers on `deepspeed.initialize()`, which takes a PyTorch `nn.Module`, an optimizer, and a JSON config, and returns a `model_engine` wrapper whose `forward`, `backward()`, and `step()` calls replace the corresponding PyTorch calls while DeepSpeed handles gradient averaging, loss scaling, and the learning-rate schedule automatically [4].

**When to pick it**: general-purpose PyTorch distributed training/inference when the deciding constraint is GPU memory rather than a specific post-training method - ZeRO lets a single GPU run models that plain data parallelism cannot fit at all, and DeepSpeed's own docs state the crossover plainly: "For models of up to 13 billion parameters, you can use ZeRO-powered data parallelism conveniently without requiring model parallelism, while in contrast, standard data parallelism will run out of memory for models with more than 1.4 billion parameters" [5]. DeepSpeed itself ships no RLHF/GRPO/DPO trainer classes; the shortlist row's "RLHF" evidence is a diagram inside a ZeRO++ blog post showing ZeRO++ speeding up the separate `DeepSpeedExamples` repo's `applications/DeepSpeed-Chat` RLHF pipeline (SFT, reward model, then PPO), not a method DeepSpeed trains on its own [6][7]. Choose it as the sharding/memory-optimization layer underneath a trainer (Transformers, Accelerate, DeepSpeed-Chat, or a custom loop), not as a post-training method library in itself [1][8].

**Methods it ships**: none of the post-training methods (SFT/DPO/PPO/GRPO) that the other cards in this deck cover - DeepSpeed is a training/inference systems library, not a method library. What it ships are memory and parallelism techniques: ZeRO Stages 1-3 (optimizer-state, then +gradient, then +parameter partitioning) [9], ZeRO-Offload and ZeRO-Infinity (CPU/NVMe offload of optimizer and parameter state) [10], ZeRO++ (hierarchical partitioning and quantized communication) [8], 3D parallelism combining data, tensor (model), and pipeline parallelism, with Automatic Tensor Parallelism (AutoTP) and Automatic Expert Parallelism (AutoEP) as automated layout modes [11][12], sparse-attention kernels, a model-compression toolkit, and communication-efficient optimizers (1-bit Adam, 0/1 Adam, 1-bit LAMB) [11]. Mixture-of-Experts (DeepSpeed MoE) is a separate documented feature, with its own tutorial page [11][13]. This taxonomy lives on the docs site's Tutorials index and moves as features are added - check the live page [13] rather than this list alone.

**Scale it handles**: single GPU (ZeRO-powered data parallelism, no model parallelism needed up to roughly 13B parameters per the ratio above [5]) up to multi-node clusters, launched through the bundled `deepspeed` CLI entry point over a hostfile (`worker-1 slots=4` format, compatible with OpenMPI/Horovod, defaulting to `/job/hostfile` if `--hostfile` is not given) [14], with `--num_nodes`/`--num_gpus`/`--include`/`--exclude` to restrict the resource set, and a no-SSH mode (`--no_ssh --node_rank=<n> --master_addr=<addr>`) for container orchestration such as Kubernetes [14]. 3D parallelism (data + tensor + pipeline) is documented as reaching trillion-parameter models, and ZeRO-powered data parallelism combined with model parallelism is documented as giving up to 10x throughput versus model parallelism alone across the 1.5B-to-hundred-billion-parameter range - both are the library's own published figures, not third-party benchmarks [11]. Long-sequence support (sparse attention kernels) is documented as giving an order-of-magnitude longer sequence length at up to 6x faster execution than dense attention, and 1.5-3x faster than other sparse implementations the docs compare against [11].

**Install**: `pip install deepspeed`; latest release v0.19.4, published 2026-08-06 [15][16], tag `v0.19.4` resolves to commit `c455031422641a825588c926455458c82b27b6c0`, dated 2026-08-06 [17] - this is the release the install line delivers, and it is a different, later commit than the shortlist row's `617061d6b98a1199ec67291abba86dc81bf1c4a9` (dated 2026-07-31): comparing the two shows the release commit is 22 commits ahead of the row's commit, not behind it, so the row's screening commit is stale relative to what `pip install deepspeed` actually delivers [16][29]. At that release commit, `requirements/requirements.txt` pins `torch>=2.0.0` with no upper bound, plus `einops`, `hjson`, `msgpack`, `ninja`, `numpy`, `packaging>=20.0`, `psutil`, `py-cpuinfo`, `pydantic>=2.0.0`, `tqdm` - none of the other dependencies carry an upper bound either [18]. `setup.py` at that commit declares no `python_requires` and only advertises Python 3.8-3.12 via classifiers [19]; PyPI's own metadata for 0.19.4 likewise reports no `requires_python` floor [15]. Licence is Apache-2.0 [1][19]. Extras are listed in `setup.py`'s `extras_require` (`1bit`, `1bit_mpi`, `autotuning`, `autotuning_ml`, `sparse_attn`, `sparse`, `inf`, `sd`, `triton`, `deepcompile`, `torchembed`, `dev`, `readthedocs`, and an `all` aggregate) with each extra's own pinned requirements file under `requirements/` [19]. The README states PyTorch >=2.0 is recommended, a C++/CUDA (or ROCm) compiler is needed to build/install DeepSpeed's op extensions, and lists NVIDIA Pascal/Volta/Ampere/Hopper and AMD MI100/MI200 as well-tested GPUs, with Huawei Ascend NPU, Intel Gaudi 2 HPU, Intel Xeon CPU, Intel Data Center GPU Max XPU, and Tecorigin SDAA as community-contributed hardware support - no single documented CUDA-version floor is stated [20].

**Maintained by**: the DeepSpeed Team (formerly under Microsoft, now the `deepspeedai` GitHub organization) [1][3], with public monthly office hours on the last Tuesday of each month [16]; the README's own dated news feed runs through 2026/05, including a Muon-optimizer integration post and an AMD-GPU SDMA offload feature, and the GitHub releases API separately shows a newer v0.19.4 release published 2026-08-06 [15][16].

## Quick start

The training-loop pattern from the Getting Started guide, using the model, optimizer, and config you already have [4]:

```python
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                       model=model,
                                                       model_parameters=params)

for step, batch in enumerate(data_loader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

A minimal `ds_config.json`, from the same page [4]:

```json
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": { "lr": 0.00015 }
  },
  "fp16": { "enabled": true },
  "zero_optimization": true
}
```

CLI launch, single node, all local GPUs [14]:

```bash
deepspeed <client_entry.py> <client args> \
  --deepspeed --deepspeed_config ds_config.json
```

## Start it

One GPU: run the script above directly (`--num_gpus=1` or letting DeepSpeed auto-detect local slots) [14][4]. Several GPUs or nodes go through the bundled `deepspeed` launcher and a hostfile - `worker-1 slots=4` / `worker-2 slots=4` lines, passed with `--hostfile=myhostfile`, or `/job/hostfile` by default; `--num_nodes`/`--num_gpus` restrict the count, `--include="worker-2:0,1"` / `--exclude="worker-2:0@worker-3:0,1"` restrict to specific devices [14]. For SSH-less orchestration (e.g. Kubernetes), run the same command on every node with `--hostfile=myhostfile --no_ssh --node_rank=<n> --master_addr=<addr> --master_port=<port>` - the hostnames no longer need to be reachable via passwordless SSH, but the docs state the hostfile "is still required for the launcher to collect information about the environment, such as the number of nodes and the number of GPUs per node," so no-SSH mode does not remove the hostfile requirement [14]. Effective batch size is `train_batch_size` in the config, which DeepSpeed derives from per-GPU micro-batch x data-parallel world size x `gradient_accumulation_steps` - the quickstart config sets `train_batch_size` directly rather than the components [4].

ZeRO is configured under the `zero_optimization` block, with `"stage"` selecting 0/1/2/3; Stage 3 adds `offload_param` and `offload_optimizer` sub-blocks to move parameters and optimizer state to CPU (or NVMe), plus stage-3-only tuning fields `stage3_max_live_parameters`, `stage3_max_reuse_distance`, `stage3_prefetch_bucket_size`, and `stage3_param_persistence_threshold` [21]. `pin_memory` under `offload_param`/`offload_optimizer` now defaults to `true` (previously `false`) to enable full-bandwidth async CPU offload overlapped with backward compute - the docs flag this as a source of new out-of-memory errors on hosts with a low `ulimit -l` memlock limit, and give the fix as explicitly setting `"pin_memory": false` [21] - this is a DeepSpeed-changed default worth flagging before assuming a config from an older run still fits the same host. Documented OOM first aid: enable `offload_optimizer` (optionally `offload_param` too) to move state off-GPU, and if still short on memory, reduce `stage3_max_live_parameters` / `stage3_prefetch_bucket_size` and set `pin_memory: false` if the host's memlock limit is tight [21].

## Watch it

Mechanics only - what a given metric means for a specific training method belongs on that method's own card, not here.

- **Enable it**: the Monitor module logs to TensorBoard, WandB, Comet, or plain CSV; each backend is its own config block (`"tensorboard": {"enabled": true, "output_path": ...}`, `"wandb": {"enabled": true, "team": ..., "project": "deepspeed", ...}`), and TensorBoard/WandB logging additionally require the corresponding Python package installed [22].
- **Auto-logged metric names**: the config-json reference's Monitoring Module section publishes the exact field names DeepSpeed writes once a backend is enabled: `Train/Samples/train_loss`, `Train/Samples/lr`, `Train/Samples/loss_scale` (only when fp16 is enabled), `Train/Eigenvalues/ModelBlockParam_{i}` (only when `eigenvalue` is enabled), and the `Train/Samples/elapsed_time_ms_{forward,backward,backward_inner,backward_allreduce,step}` timing family (only when `flops_profiler.enabled` or `wall_clock_breakdown` is set) [22]. Without one of the Monitor backends enabled, DeepSpeed logs none of these anywhere.
- **Communication logging** is a separate opt-in block, `"comms_logger": {"enabled": true, "verbose": ..., "prof_all": ..., "debug": ...}`; retrieving the summary is a client-side call, `deepspeed.comm.log_summary()` [23].
- **Stopping-rule search**: read on 2026-08-10, the ZeRO tutorial [9], the config-json reference [21][22][23], and the Getting Started page [4] carry no RL-style early-stopping, patience, or reward-threshold field. The config-json reference does document one field named `tuner_early_stopping` [30], but it belongs to the unrelated Autotuning module: it is "the number of experiments to run beyond the current best experiment" before the Autotuner stops searching for a batch/micro-batch/ZeRO-stage configuration, and has nothing to do with stopping a training run on a reward or loss signal; DeepSpeed exposes gradient-clipping (`"gradient_clipping": 1.0`) and the ZeRO tuning knobs above, but no RL-style stopping rule was found in the pages searched.

## Save it

- The base checkpoint call is `model_engine.save_checkpoint(save_dir, ckpt_id, client_sd=client_sd)`, where `ckpt_id` uniquely identifies the checkpoint (any value convertible to string, e.g. a loss value) and `client_sd` is an arbitrary caller-supplied state dict returned unchanged by the matching load call [4]. The docs warn explicitly: "all processes must call this method and not just the process with rank 0... This method will hang waiting to synchronize with other processes if it's called just for the process with rank 0" [4].
- Load with `model_engine.load_checkpoint(load_dir, ckpt_id)`, returning `(_, client_sd)` [4].
- On disk, a ZeRO checkpoint directory holds per-rank model-state files named `zero_pp_rank_<N>_mp_rank_<NN>_model_states.pt`, alongside separate optimizer-shard files [24][25]. AutoEP (expert-parallel) checkpoints add a further split by ZeRO stage: under Stage 1 or Stage 2, DeepSpeed writes the routed expert weights into per-expert files named `layer_<moe_layer_id>_expert_<global_expert_id>_mp_rank_<NN>_model_states.pt`; under Stage 3, AutoEP checkpoints are partition-native instead - no per-expert files are produced, and the same `zero_pp_rank_*_model_states.pt` and optimizer shard files are used together with recorded partition-group metadata [24]. Saving a checkpoint also drops a `zero_to_fp32.py` script into the top-level checkpoint directory automatically [25].
- Under ZeRO-2, `state_dict` already contains full fp16 model weights, savable with plain `torch.save` [25]. Under ZeRO-3, `state_dict` holds only placeholders because weights are partitioned across GPUs; setting `"stage3_gather_16bit_weights_on_model_save": true` in the ZeRO config and calling `model_engine.save_16bit_model(output_dir, output_file)` consolidates and saves the full fp16 weights - the docs warn this requires gathering all weights onto one GPU and so "can be slow and memory demanding", and that if the flag is left `false`, no consolidated weights are saved at all [25].
- For fp32 weights offline, without GPUs, run the auto-generated `./zero_to_fp32.py . pytorch_model.bin` inside the checkpoint directory, or call `deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)` / `load_state_dict_from_zero_checkpoint(model, checkpoint_dir)` in Python; the docs note this script uses roughly 2x the final checkpoint's size in host RAM, and that a model updated via `load_state_dict_from_zero_checkpoint` is good for saving but "no longer good for continuing the training" without a fresh `deepspeed.initialize()` [25].
- **Universal Checkpointing** converts a saved ZeRO checkpoint (dense, AutoTP, or AutoEP) into a parallelism-topology-independent format with `deepspeed/checkpoint/ds_to_universal.py`, then resumes with `checkpoint.load_universal` enabled - this is what lets a checkpoint saved under one GPU/parallelism layout be reloaded under a different one [12].
- **A documented trap**: GitHub issue #7549 (opened 2025-09-09, closed 2025-10-01), "[BUG] save_checkpoint race when consolidating NVMe offloaded tensors → FileExistsError", reports that saving a ZeRO Stage 3 checkpoint with NVMe offloading enabled from `accelerator.save_state` can raise `FileExistsError`, because DeepSpeed's `shutil.copytree` consolidation of per-rank NVMe offload directories into a shared `offloaded_tensors` destination races when every rank calls it concurrently; collaborator sfc-gh-truwase (COLLABORATOR) guided the discussion toward saving into separate per-rank directories that mirror the offload layout instead [26]. In the same thread, contributor H1manshu21 (CONTRIBUTOR) later reported that after adopting rank-scoped saves, resuming via `accelerator.prepare()` followed by `accelerator.load_state()` deadlocked or segfaulted at `accelerator.backward(loss)` with no error printed; contributor therealnaveenkamal (CONTRIBUTOR) traced this secondary issue to `_load_zero_checkpoint()` assuming NVMe-offloaded optimizer tensors are already in memory and purging the swap state on load [26]. Treat NVMe-offloaded ZeRO-3 save and resume as fragile until verified on your own version.
- Loader handoff: a plain ZeRO-2 `state_dict` or a `save_16bit_model()`/`zero_to_fp32.py` output is an ordinary PyTorch state dict or checkpoint file, loadable outside DeepSpeed; a raw ZeRO-3 checkpoint directory without one of those consolidation steps is NOT directly loadable by a plain PyTorch or Transformers loader - whether a given evaluator can load it directly depends on which of the above save paths produced it [25].

## Find it in the docs

The docs (deepspeed.ai) are a Jekyll site, not tag-versioned by DeepSpeed release: fetched 2026-08-10, `https://www.deepspeed.ai/getting-started/`, `/training/`, `/tutorials/`, `/tutorials/zero/`, `/tutorials/universal-checkpointing/`, and `/docs/config-json/` all return the current live page with no per-release URL form - there is no `vX.Y.Z` path segment to swap in for this site (unlike readthedocs, below). Page slugs follow topic names directly under `/tutorials/<slug>/`; the Tutorials index [13] lists every slug currently published, grouped as Training / Inference / Compression / Science.

- Getting started and the core `initialize()`/checkpoint API: `/getting-started/` [4].
- Feature overview and scaling numbers (3D parallelism, ZeRO, sparse attention, MoE, Monitor): `/training/` [11].
- ZeRO stage semantics and weight-extraction recipes: `/tutorials/zero/` [9][25].
- Universal Checkpointing conversion flow: `/tutorials/universal-checkpointing/` [12].
- Every `ds_config.json` field, including ZeRO sub-fields and the Monitor/`comms_logger` blocks: `/docs/config-json/` [21][22][23].
- API-generated reference (docstrings) lives on a separate, versioned host, readthedocs.io: `https://deepspeed.readthedocs.io/en/latest/` tracked the in-development `master` branch (titled "DeepSpeed 0.19.5 documentation" when fetched 2026-08-10, one version ahead of the 0.19.4 release), while `https://deepspeed.readthedocs.io/en/stable/` served the 0.19.4 release docs matching the current `pip install deepspeed` - a `v0.19.4`-style path segment 404s on this host, so `stable` (not a version tag) is the pinned-to-release form here [27].
- Runnable references beyond the docs: the repo's own `blogs/` and `examples/` trees carry worked configs and write-ups (the ZeRO++ Chinese blog post that the shortlist row's RLHF evidence traces to is `blogs/zeropp/chinese/README.md`) [6]; the companion `deepspeedai/DeepSpeedExamples` repository holds runnable Applications/Training/Inference/Compression/Benchmarks trees, including `applications/DeepSpeed-Chat` for the 3-step (SFT, reward model, PPO/RLHF) pipeline built on top of DeepSpeed [7][28].
- Community layer: the README's own "Community Tutorials" section curates three third-party videos - Mark Saroufim's "DeepSpeed: All the tricks to scale to gigantic models", Yannic Kilcher's "Turing-NLG, DeepSpeed and the ZeRO optimizer", and The AI Epiphany's "Ultimate Guide To Scaling ML Models" - alongside a "Further Reading" table and dated news feed linking to the docs, API docs (readthedocs), tutorials, and dated blog posts [16].
- No official MCP endpoint for querying these docs was found in the pages read for this card.

## Sources

All pages are live/unpinned unless a commit or tag is stated, and are 2026-08-10 readings unless noted. Method names covered elsewhere in this deck (SFT, DPO, PPO, GRPO) are deliberately cited to nothing here.

[1] DeepSpeed GitHub repository metadata (description, homepage, licence, maintainer org). https://api.github.com/repos/deepspeedai/DeepSpeed. Fetched 2026-08-10.

[2] DeepSpeed `setup.py` author field, at release commit c455031422641a825588c926455458c82b27b6c0 (tag v0.19.4). https://raw.githubusercontent.com/deepspeedai/DeepSpeed/c455031422641a825588c926455458c82b27b6c0/setup.py. Fetched 2026-08-10.

[3] DeepSpeed GitHub repository (former-name redirect from microsoft/DeepSpeed, confirmed via the shortlist row's own `former_names` field and the live `deepspeedai` org). https://github.com/deepspeedai/DeepSpeed. Fetched 2026-08-10.

[4] DeepSpeed Getting Started guide. https://www.deepspeed.ai/getting-started/. Fetched 2026-08-10.

[5] DeepSpeed Training Overview and Features page, "Good Usability" section. https://www.deepspeed.ai/training/. Fetched 2026-08-10.

[6] DeepSpeed ZeRO++ blog post (Chinese), `blogs/zeropp/chinese/README.md` on the `master` branch, RLHF-with-ZeRO++ section. https://raw.githubusercontent.com/deepspeedai/DeepSpeed/master/blogs/zeropp/chinese/README.md. Fetched 2026-08-10.

[7] DeepSpeedExamples repository README (Applications/Training/Inference/Compression/Benchmarks structure). https://raw.githubusercontent.com/deepspeedai/DeepSpeedExamples/master/README.md. Fetched 2026-08-10.

[8] DeepSpeed Training Overview and Features page (communication-efficiency section covering ZeRO++). https://www.deepspeed.ai/training/. Fetched 2026-08-10.

[9] DeepSpeed ZeRO tutorial, stage definitions. https://www.deepspeed.ai/tutorials/zero/. Fetched 2026-08-10.

[10] DeepSpeed Training Overview and Features page (ZeRO-Offload feature description). https://www.deepspeed.ai/training/. Fetched 2026-08-10.

[11] DeepSpeed Training Overview and Features page (feature list, 3D parallelism, sparse-attention, MoE, 10x/6x/2x/26x figures). https://www.deepspeed.ai/training/. Fetched 2026-08-10.

[12] DeepSpeed Universal Checkpointing tutorial. https://www.deepspeed.ai/tutorials/universal-checkpointing/. Fetched 2026-08-10.

[13] DeepSpeed Tutorials index page (live taxonomy of feature/tutorial slugs). https://www.deepspeed.ai/tutorials/. Fetched 2026-08-10.

[14] DeepSpeed Getting Started guide, Launching DeepSpeed Training / Resource Configuration sections. https://www.deepspeed.ai/getting-started/. Fetched 2026-08-10.

[15] deepspeed PyPI JSON API (version 0.19.4, `requires_python`, licence, upload time). https://pypi.org/pypi/deepspeed/json. Fetched 2026-08-10.

[16] DeepSpeed GitHub releases list and repository README (release dates, office hours, dated news feed). https://api.github.com/repos/deepspeedai/DeepSpeed/releases and https://raw.githubusercontent.com/deepspeedai/DeepSpeed/master/README.md. Fetched 2026-08-10.

[17] GitHub API resolution of tag `v0.19.4` to its commit SHA. https://api.github.com/repos/deepspeedai/DeepSpeed/git/refs/tags/v0.19.4. Fetched 2026-08-10.

[18] DeepSpeed core requirements file, at release commit c455031422641a825588c926455458c82b27b6c0. https://raw.githubusercontent.com/deepspeedai/DeepSpeed/c455031422641a825588c926455458c82b27b6c0/requirements/requirements.txt. Fetched 2026-08-10.

[19] DeepSpeed `setup.py`, at release commit c455031422641a825588c926455458c82b27b6c0 (extras_require, classifiers, licence). https://raw.githubusercontent.com/deepspeedai/DeepSpeed/c455031422641a825588c926455458c82b27b6c0/setup.py. Fetched 2026-08-10.

[20] DeepSpeed repository README, Installation section (PyTorch/compiler prerequisites, tested and community-contributed hardware). https://raw.githubusercontent.com/deepspeedai/DeepSpeed/master/README.md. Fetched 2026-08-10.

[21] DeepSpeed config-json reference, ZeRO Optimizations section (stage field, offload sub-blocks, stage-3 tuning fields, pin_memory default-change note). https://www.deepspeed.ai/docs/config-json/. Fetched 2026-08-10.

[22] DeepSpeed config-json reference, Monitoring Module section. https://www.deepspeed.ai/docs/config-json/. Fetched 2026-08-10.

[23] DeepSpeed config-json reference, Communication Logging section. https://www.deepspeed.ai/docs/config-json/. Fetched 2026-08-10.

[24] DeepSpeed Universal Checkpointing tutorial, AutoEP checkpoint file-naming section. https://www.deepspeed.ai/tutorials/universal-checkpointing/. Fetched 2026-08-10.

[25] DeepSpeed ZeRO tutorial, "Extracting weights" section (state_dict behavior by stage, save_16bit_model, zero_to_fp32.py usage and memory note, resumability warning). https://www.deepspeed.ai/tutorials/zero/. Fetched 2026-08-10.

[26] DeepSpeed GitHub issue #7549, "[BUG] save_checkpoint race when consolidating NVMe offloaded tensors → FileExistsError" and its comment thread (collaborator sfc-gh-truwase; contributors H1manshu21 and therealnaveenkamal). https://github.com/deepspeedai/DeepSpeed/issues/7549. Fetched 2026-08-10.

[27] DeepSpeed API docs on Read the Docs, `latest` vs `stable` version pages. https://deepspeed.readthedocs.io/en/latest/ and https://deepspeed.readthedocs.io/en/stable/. Fetched 2026-08-10.

[28] DeepSpeed-Chat application README, `applications/DeepSpeed-Chat/README.md` in the DeepSpeedExamples repository (3-step SFT/reward-model/RLHF pipeline description). https://raw.githubusercontent.com/deepspeedai/DeepSpeedExamples/master/applications/DeepSpeed-Chat/README.md. Fetched 2026-08-10.

[29] GitHub API commit lookups and compare, establishing that release commit c455031422641a825588c926455458c82b27b6c0 (2026-08-06T00:48:13Z) is 22 commits ahead of the shortlist row's commit 617061d6b98a1199ec67291abba86dc81bf1c4a9 (2026-07-31T20:48:55Z). https://api.github.com/repos/deepspeedai/DeepSpeed/commits/c455031422641a825588c926455458c82b27b6c0, https://api.github.com/repos/deepspeedai/DeepSpeed/commits/617061d6b98a1199ec67291abba86dc81bf1c4a9, and https://api.github.com/repos/deepspeedai/DeepSpeed/compare/617061d6b98a1199ec67291abba86dc81bf1c4a9...c455031422641a825588c926455458c82b27b6c0. Fetched 2026-08-10.

[30] DeepSpeed config-json reference, Autotuning Module section (`tuner_early_stopping` field). https://www.deepspeed.ai/docs/config-json/. Fetched 2026-08-10.
