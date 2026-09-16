# DAMO-ConvAI

A multi-paper research monorepo from Alibaba DAMO, not a post-training library: pick a subdirectory for one paper's code, not the repo for a reusable trainer API.

DAMO-ConvAI is described by its own README as "the official repository which contains the codebase for Alibaba DAMO Conversational AI" [1]. It is built and maintained by Alibaba DAMO Academy / Alibaba Research [1][2], and it lives at https://github.com/AlibabaResearch/DAMO-ConvAI [2]. Its root holds 49 independent, mostly unrelated project directories (50 top-level directory entries minus the non-project `.github` directory) - dialogue systems, text-to-SQL (`bird`, `r2sql`, `s2sql`), and several recent LLM post-training papers - each with its own README, its own environment, and its own run scripts; there is no shared package, `import damo_convai`, or repo-level API [3]. The methods this card's shortlist row detected (DDPO, DPO, KTO, ORPO, GRPO, reward modeling, PPO) do not belong to a DAMO-ConvAI trainer: they sit inside three of those subdirectories as vendored, and in one case modified, copies of trainer code from the Hugging Face **trl** library and from **LLaMA-Factory**, pulled in as an implementation detail of one paper's experiments each [4][5][6].

**When to pick it**: do not pick DAMO-ConvAI itself for a training run - there is no top-level install command, Config surface, or launcher that spans the repo. Pick it only to reproduce one specific paper, by entering that paper's own subdirectory and following its own README; for a reusable trainer, use the upstream library that subdirectory vendors (trl or LLaMA-Factory - both covered on their own cards) directly instead of the copy embedded here.

**Methods it ships**: none as a repo-level API. The methods the shortlist row found trace to three unrelated subprojects, each vendoring a different upstream trainer file, all as of commit `483554e` [7]:
- `MMLatentAction/` (a multimodal latent-action paper) carries its own copy of trl's trainer classes under `MMLatentAction/trl/trainer/` - `ddpo_trainer.py`, `orpo_trainer.py`, `reward_trainer.py`, `ppo_trainer.py` - plus a top-level `grpo_trainer.py`. Diffing `MMLatentAction/trl/trainer/grpo_trainer.py` against the upstream `trl` `v0.18.1` tag (the version its own `requirements.txt` pins) shows it is not an unmodified copy: it adds VLM-specific imports (`AutoProcessor`, `ProcessorMixin`, `prepare_multimodal_messages`, `split_pixel_values_by_grid`) that do not exist in that trl release, so it is a project-specific fork, not a passthrough [4][8].
- `EPO/LLaMA-Factory/` (a strategic-reasoning RL paper) is a vendored copy of the LLaMA-Factory repository; its DPO and KTO trainer files sit under `EPO/LLaMA-Factory/src/llamafactory/train/{dpo,kto}/trainer.py`, and EPO's own README documents driving it entirely through LLaMA-Factory's own `llamafactory-cli train` command [5][9].
- `EvoTrainer/mcore_adapter/` (an agentic-RL paper) vendors `dpo_trainer.py` from `mcore_adapter`, a Megatron-Core adapter module; EvoTrainer's own README states it is "built on top of the ROLL framework" (a separate cluster RL framework, covered on its own card if shortlisted) and is not itself a general-purpose trainer [6].
- `RPM-Generalization/estimator_training/grpo_trainer.py` is a GRPO trainer used only to train that paper's own R-EMID metric estimator, not a policy-training entry point [10].

None of these four is documented, versioned, or supported as a standalone library by DAMO-ConvAI; each is read only as far as the file the shortlist cited plus its immediate subdirectory README.

**Scale it handles**: no repo-level answer - scale is whatever the vendored launcher underneath each subproject supports, and each is documented at a different point on that spectrum. `MMLatentAction` launches through Accelerate. Its vendored trl copy carries a general template set - `single_gpu`, `multi_gpu`, `fsdp1`, `fsdp2`, `zero1`, `zero2`, `zero3` - under `MMLatentAction/trl/accelerate_configs/`, but `run_MMRole_RL.sh` itself resolves `--config_file` against a separate, sibling directory, `MMLatentAction/accelerate_configs/`, which at this commit holds exactly one file, `multi_gpu_4gpu.yaml` - so the RL quickstart run in this card is fixed to that one 4-GPU layout, not a free choice among the trl template set [11]. `EPO` launches through LLaMA-Factory's own `llamafactory-cli`, selecting the compute device via `CUDA_VISIBLE_DEVICES` or `ASCEND_RT_VISIBLE_DEVICES` [9]. `EvoTrainer` documents an "Agentic Training" launch step and lists Megatron and DeepSpeed among its third-party backend adapters, implying multi-GPU/multi-node reach through ROLL and mcore_adapter, but this card's reading of its README did not reach a documented node-count or benchmark [6].

**Install**: no single install line for the repo. Each relevant subproject pins its own environment: `MMLatentAction/requirements.txt` (Python 3.10 per its README) pins `torch==2.6.0`, `transformers==4.57.0`, `trl==0.18.1`, `accelerate~=1.10.1`, `peft==0.14.0`, `vllm==0.10.0` [4][11]. `EPO/LLaMA-Factory` is installed with `pip install -r requirements.txt` followed by `pip install -e .` inside that subdirectory, using its own vendored `setup.py`/`requirements.txt` rather than the upstream LLaMA-Factory release [9]. `EvoTrainer` has its own `mcore_adapter/` install path not read in full for this card. The repo overall is MIT-licensed [1][2] (LLaMA-Factory content vendored under `EPO/LLaMA-Factory/` carries its own upstream licence, not re-stated here). No repo-level CUDA or hardware floor is stated anywhere in the root README [1].

**Maintained by**: Alibaba DAMO Academy / Alibaba Research [1][2]. The root README's own news feed only lists items through 2024-02 [1], but the repository is not stale: GitHub reports a push to the default branch on 2026-06-10, and the newest subprojects surfaced by this card's search (`EvoTrainer`, `RPM-Generalization`, `MMLatentAction`) cite 2026 arXiv IDs, so activity has continued past the README's last edit without the README being updated to match [2][4][6][10].

## Quick start

There is no repo-level quick start. Each subproject's own smallest run, quoted or reproduced from its own README:

`MMLatentAction` (pretrain the latent-action module, then run RL on the MMRole task) [11]:
```bash
bash pretrain.sh
bash run_MMRole_RL.sh
```
which itself calls, for the RL stage:
```bash
accelerate launch --config_file "accelerate_configs/multi_gpu_4gpu.yaml" \
     grpo_vlm_MMRole.py \
     --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
     --num_generations 8 --learning_rate 1e-6 --lr_scheduler_type constant \
     --max_prompt_length 1024 --max_completion_length 1024 \
     --lm_mode DAPO-VLMActionRL --beta 0.01
```

`EPO` (train on the SOTOPIA-PI RL data through the vendored LLaMA-Factory CLI) [9]:
```bash
cd EPO/LLaMA-Factory
llamafactory-cli train examples/train_epo/llama3_sotopia_pi_rl.yaml
```

`RPM-Generalization` (train the R-EMID estimator via GRPO) [10]:
```bash
cd RPM-Generalization/estimator_training
bash run_co_evolve.sh
```

## Start it

Launch mechanics are entirely subproject-specific; there is no shared launcher.
- `MMLatentAction`'s `run_MMRole_RL.sh` passes `--config_file "accelerate_configs/multi_gpu_4gpu.yaml"` to `accelerate launch`, which resolves to `MMLatentAction/accelerate_configs/multi_gpu_4gpu.yaml` - a single 4-GPU template, not one of the seven general templates (`single_gpu.yaml`, `multi_gpu.yaml`, `fsdp1.yaml`, `fsdp2.yaml`, `zero1.yaml`, `zero2.yaml`, `zero3.yaml`) that ship separately under the vendored `MMLatentAction/trl/accelerate_configs/`; swapping in one of those requires pointing `--config_file` at that other directory yourself. The RL stage batches with `--per_device_train_batch_size 8` and `--num_generations 8` in the example above [11].
- `EPO` selects the device set via `CUDA_VISIBLE_DEVICES` (GPU) or `ASCEND_RT_VISIBLE_DEVICES` (NPU) before invoking `llamafactory-cli train`; the actual Config surface, defaults, and OOM guidance belong to the LLaMA-Factory card, not here, because EPO's copy is used as-is through that CLI [9].
- Neither `EvoTrainer`'s nor `RPM-Generalization`'s launch-time knobs, effective-batch arithmetic, or out-of-memory guidance were read in enough depth for this card to state them; their own subdirectory READMEs are the place to look [6][10].

## Watch it

No subproject README read for this card documents its own metric names, logging backend, or a published stopping-rule/threshold; because `MMLatentAction`'s and `EPO`'s trainers are vendored copies of trl and LLaMA-Factory respectively, whatever those libraries log by default is what a run here would log, but this card did not verify that the vendored copies were left unmodified on the logging path (the `MMLatentAction` GRPO trainer is confirmed modified elsewhere, see Methods it ships) [4][9]. Treat this as unanswered rather than "none published": the search that would answer it (each subproject's own docs, past the top-level READMEs already fetched) was not carried out for this card.

## Save it

Not documented at the repo level. `MMLatentAction`'s example RL command writes to `--output_dir "${PRETRAIN_CKPT}-${TASK_NAME}"` via the Accelerate-launched script, but the checkpoint directory's internal layout, retention flags, and resume call were not read for this card [11]. Since `MMLatentAction`'s trainer and `EPO`'s LLaMA-Factory copy are forks/vendored copies of libraries with their own save/resume contracts, do not assume the on-disk layout matches the upstream library's own card without checking the vendored code directly.

## Find it in the docs

DAMO-ConvAI has no versioned documentation site or version-tag URL pattern - it is a plain GitHub repository, browsed by path [2][3]. To find anything: open the repo root [2], pick the subdirectory whose name matches the paper or arXiv ID you want, and read that subdirectory's own `README.md` (linked inline above for `MMLatentAction`, `EPO`, `EvoTrainer`, `RPM-Generalization`); several subdirectories additionally provide a `README_zh.md` [6]. For the vendored trainer internals (trl's GRPOTrainer/PPOTrainer/RewardTrainer/ORPOTrainer/DDPO, or LLaMA-Factory's DPO/KTO trainers), the upstream library's own docs are the authoritative reference, not this repo - go to the trl or LLaMA-Factory card. `EvoTrainer`'s own upstream, ROLL (`https://github.com/alibaba/ROLL`), is named directly in its README as the framework it is built on [6]. No official MCP endpoint or curated community-tutorials page for DAMO-ConvAI as a whole was found in the pages read for this card.

Honest boundary: this repo is not built, packaged, or supported as a general-purpose post-training library, and no subproject's README claims that role for it - each is scoped to reproducing one paper's own experiments [1][4][5][6][10]. No maintainer replies in closed GitHub issues about the post-training subprojects specifically (`MMLatentAction`, `EPO`, `EvoTrainer`, `RPM-Generalization`) were searched for this card; none are cited here as a result.

## Sources

Sources [1], [3]-[11] are pinned to commit `483554eae102996f5ec1f4feab4e78ef29c2a394`: [1] and [4]-[11] are `raw.githubusercontent.com` URLs with that commit SHA in the path, and [3] is a GitHub API contents call made with `?ref=483554eae102996f5ec1f4feab4e78ef29c2a394`, so its 49-project-directory count is a fact about that commit and will not change under the same URL. Source [2] is the sole live, unpinned fetch here: the repository-metadata endpoint (`api.github.com/repos/...`) takes no revision parameter, so the push date, star count, and archived status it reports reflect whatever GitHub shows on the fetch date and can differ on a later read. All fetches below are dated 2026-08-12.

[1] DAMO-ConvAI root README. https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/483554eae102996f5ec1f4feab4e78ef29c2a394/README.md. Fetched 2026-08-12.

[2] DAMO-ConvAI GitHub repository (metadata: license, push date, creation date, star count, archived status). https://github.com/AlibabaResearch/DAMO-ConvAI. Fetched 2026-08-12 via the GitHub API (`api.github.com/repos/AlibabaResearch/DAMO-ConvAI`).

[3] DAMO-ConvAI root directory listing at commit 483554e. https://github.com/AlibabaResearch/DAMO-ConvAI/tree/483554eae102996f5ec1f4feab4e78ef29c2a394. Fetched 2026-08-12 via the GitHub API contents endpoint.

[4] `MMLatentAction/requirements.txt` and its vendored `trl/trainer/grpo_trainer.py`, diffed against upstream trl tag `v0.18.1`. https://github.com/AlibabaResearch/DAMO-ConvAI/tree/483554eae102996f5ec1f4feab4e78ef29c2a394/MMLatentAction/trl. Fetched 2026-08-12; diffed against https://raw.githubusercontent.com/huggingface/trl/v0.18.1/trl/trainer/grpo_trainer.py, fetched 2026-08-12.

[5] `EPO/LLaMA-Factory/src/llamafactory/train/{dpo,kto}/trainer.py` (vendored LLaMA-Factory trainer files; directory only, not diffed against upstream for this card). https://github.com/AlibabaResearch/DAMO-ConvAI/tree/483554eae102996f5ec1f4feab4e78ef29c2a394/EPO/LLaMA-Factory. Fetched 2026-08-12.

[6] `EvoTrainer/README.md` (states it is built on ROLL; lists Megatron/DeepSpeed/vLLM/SGLang as third-party backend adapters; `mcore_adapter/src/mcore_adapter/trainer/dpo_trainer.py` file header). https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/483554eae102996f5ec1f4feab4e78ef29c2a394/EvoTrainer/README.md and https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/483554eae102996f5ec1f4feab4e78ef29c2a394/EvoTrainer/mcore_adapter/src/mcore_adapter/trainer/dpo_trainer.py. Fetched 2026-08-12.

[7] Shortlist row's `methods_seen` file paths for this repository (pipeline-internal evidence, not an independently fetched page): DDPO, DPO, KTO, ORPO, GRPO, reward-model, and PPO trainer file locations at commit 483554e.

[8] trl `v0.18.1` release tag, `trl/trainer/grpo_trainer.py`. https://raw.githubusercontent.com/huggingface/trl/v0.18.1/trl/trainer/grpo_trainer.py. Fetched 2026-08-12.

[9] `EPO/README.md` and `EPO/LLaMA-Factory/README.md` (setup, `llamafactory-cli train` commands, device-selection environment variables). https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/483554eae102996f5ec1f4feab4e78ef29c2a394/EPO/README.md and https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/483554eae102996f5ec1f4feab4e78ef29c2a394/EPO/LLaMA-Factory/README.md. Fetched 2026-08-12.

[10] `RPM-Generalization/README.md` (R-EMID estimator training command; arXiv 2512.17270). https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/483554eae102996f5ec1f4feab4e78ef29c2a394/RPM-Generalization/README.md. Fetched 2026-08-12.

[11] `MMLatentAction/README.md` and `MMLatentAction/run_MMRole_RL.sh` (setup, pretrain/RL commands, accelerate config selection); directory listings of both `MMLatentAction/accelerate_configs/` (one file, `multi_gpu_4gpu.yaml` - the directory the script actually resolves `--config_file` against) and the separate `MMLatentAction/trl/accelerate_configs/` (seven general templates, vendored with the trl copy). https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/483554eae102996f5ec1f4feab4e78ef29c2a394/MMLatentAction/README.md, https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/483554eae102996f5ec1f4feab4e78ef29c2a394/MMLatentAction/run_MMRole_RL.sh, https://github.com/AlibabaResearch/DAMO-ConvAI/tree/483554eae102996f5ec1f4feab4e78ef29c2a394/MMLatentAction/accelerate_configs, and https://github.com/AlibabaResearch/DAMO-ConvAI/tree/483554eae102996f5ec1f4feab4e78ef29c2a394/MMLatentAction/trl/accelerate_configs. Fetched 2026-08-12.
