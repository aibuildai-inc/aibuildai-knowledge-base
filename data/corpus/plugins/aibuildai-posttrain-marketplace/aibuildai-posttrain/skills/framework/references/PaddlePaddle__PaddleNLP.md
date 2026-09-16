# PaddleNLP

https://github.com/PaddlePaddle/PaddleNLP

A PaddlePaddle-native LLM development suite: unified-script SFT/LoRA/Prefix-Tuning, DPO/KTO/RL alignment, and reward-model training, with 4D-parallel and multi-hardware support baked into the same config-driven trainers.

**PaddleNLP** "is a Large Language Model (LLM) development suite based on the PaddlePaddle deep learning framework, supporting efficient large model training, lossless compression, and high-performance inference on various hardware devices" [1]. It is built and maintained by the PaddlePaddle organization (Baidu's deep-learning framework team) [2][3]. Its API is script- and config-driven: each method has a runner script (e.g. `run_finetune.py`, `run_dpo.py`, `run_rl.py`) that takes a JSON/YAML config file naming the model, dataset, and method hyperparameters, alongside a Hugging-Face-TRL-styled `SFTTrainer`/`SFTConfig` class pair for no-clone use [3].

**When to pick it**: PaddlePaddle-framework shops - PaddleNLP's trainers are built on `paddle.distributed`, not PyTorch, so the model zoo, checkpoints, and custom ops (FlashMask, fused_ln) are PaddlePaddle-specific and do not interoperate with a PyTorch-based stack [3][4]. Inside that ecosystem it is broad: one repo covers SFT, several PEFT variants, DPO, KTO, reward modeling, and PPO/GRPO/REINFORCE++ RL, plus multi-hardware backends (NVIDIA GPU, Kunlun XPU, Ascend NPU, Enflame GCU, Hygon DCU) [3]. Weigh against trl or verl (cross-reference; not covered here) if the stack is PyTorch-based - this card makes no cross-framework speed or scale claim, since none of the fetched PaddleNLP sources measure against another library.

**Methods it ships** [4][5]:
- Fine-tuning (`llm/run_finetune.py`, config-selected): full-parameter SFT, LoRA, Prefix Tuning, plus (per the fine-tuning tutorial and the top-level README) LoKr, VeRA, MoRA, ReFT, rsLoRA, LoRA+, PiSSA, MoSLoRA [5][4]. The no-clone path imports `SFTConfig`/`SFTTrainer` from `paddlenlp.trl`, which also holds `DPOTrainer`, `KTOTrainer`, `EmbeddingTrainer`, and `DisLoRA` support [6].
- Alignment: DPO and KTO run through `llm/alignment/dpo/run_dpo.py` and `llm/alignment/kto/run_kto.py`; a loss_type switch inside the DPO config additionally reaches SimPO and ORPO without a separate trainer [7]. GRPO and REINFORCE++ share the entry point `llm/alignment/rl/run_rl.py`, selected by the `rl_algorithm` config field documented as accepting `grpo` or `reinforce_plus_plus` [8]; the same script's code also branches on a third value, `rl_algorithm == "ppo"`, wiring in a `PPOTrainer` from `paddlenlp.rl.trainer.ppo_trainer` - a third, code-level-only path not named in the RL README's own config glossary [9]. The underlying trainer classes live in `paddlenlp/rl/trainer/` (`actor_trainer.py`, `critic_trainer.py`, `ppo_trainer.py`, `reward_trainer.py`, `rl_trainer.py`) [10]. A standalone reward-model trainer, also named `RewardTrainer` but defined separately (subclassing the base `Trainer` rather than `RLTrainer`), lives at `llm/alignment/rm/reward_trainer.py`, distinct from the RL-internal `RewardTrainer` under `paddlenlp/rl/trainer/reward_trainer.py` used to score rollouts during PPO/GRPO/REINFORCE++ [11][12].
- The fetched sources give no experimental/stable split for these methods (unlike trl's index) and no single live taxonomy page collects the whole list - `llm/README.md` and `llm/docs/algorithm_overview.md` are the closest, and both can drift as the repo's `develop` branch moves [4][13].

**Scale it handles**: single GPU up to multi-node through PaddlePaddle's own launcher, `python -u -m paddle.distributed.launch --devices "0,1,...,7"` (or `--gpus`), invoked identically for SFT, DPO, KTO, RM, and RL [5][7][14][8]. Sharding (stage1/stage2), tensor-parallel, pipeline-parallel (with a virtual-pipeline degree), and sequence-parallel degrees are all config fields on the same `llm/config/qwen/grpo_argument.yaml`-style config, documented for the GRPO/REINFORCE++ path in the RL README's glossary [8]. The RL README publishes one documented multi-node shape, not a benchmark: a commented-out example command for a Qwen-32B run across "9台8x80G" (9 nodes of 8x80GB GPUs) with a 2k-prompt/30k-response length, alongside two reproducibility logs on Weights & Biases for the 4-GPU single-node GRPO and REINFORCE++ runs actually run in the README [8].

**Install**: `pip install --upgrade paddlenlp==3.0.0b4`, as the pinned-commit README's own install line specifies [3]; PyPI's bare `pip install paddlenlp` instead resolves to 2.8.1 (uploaded 2024-06-20), a pre-3.0 release almost a year older, because pip prefers a non-prerelease version over 3.0.0b4 unless it is pinned exactly as the README does [15]. The `v3.0.0-beta4` tag resolves to commit `a286abc1063e516ed56b746fcca33bedce5fcef3` (confirmed via the GitHub tags API); reading the build metadata at that commit: Python floor `>=3.8`, licence "Apache 2.0", `datasets>=2.0.0`, `huggingface_hub>=0.19.2`, `numpy<=1.26.4`, `tokenizers<=0.20.3` on Python <=3.8 or `tokenizers>=0.21,<0.22` above that, `dill<0.3.5`, `multiprocess<=0.70.12.2` - the full pinned list is `requirements.txt` at that same tag/commit [16][17]. `paddlepaddle` itself, the load-bearing deep-learning core, is NOT in that requirements file and must be installed separately; the README's environment section states the floor as `paddlepaddle >= 3.0.0rc1` and points to the PaddlePaddle site for hardware-specific wheels, so no CUDA/hardware minimum is stated in the fetched sources beyond that version floor [3]. Every other code claim on this card (Quick start, Methods it ships, Start it, Save it) is read at the shortlist's pinned commit `3f87dac9b719f75399f92f8bf634ae2ef0611832`, which is newer than the `v3.0.0-beta4` release commit `a286abc1063e516ed56b746fcca33bedce5fcef3` that `pip install paddlenlp==3.0.0b4` actually delivers - code and config paths described elsewhere on this card may not all exist yet in the pip-installed release. RL training additionally needs two custom-op builds from source - `PaddleNLP/csrc/setup_cuda.py install` for `paddlenlp_ops`, and `PaddleNLP/slm/model_zoo/gpt-3/external_ops/setup.py install` for `fused_ln`/`fast_ln` - both marked required (必需/必须) in the RL README [8].

**Maintained by**: the PaddlePaddle organization; about 12,962 GitHub stars, read live on 2026-08-10 [2]. The repository's most recent push is dated 2026-05-23 [2]; its most recent GitHub Release is the terse, non-prerelease `rl-v1.0.0` (2025-05-21, release notes reading only "GRPO、RF++ ready", built against the `develop` branch rather than a fixed commit) [18], and its most recent substantive, version-numbered prerelease is `v3.0.0-beta4` (2025-03-12), whose changelog adds DeepSeek V3/R1/R1-Distill and QwQ-32B support, FP8/INT8/4-bit quantized inference, and a MergeKit model-merging tool [18].

## Quick start

Text generation, from the pinned-commit README [3]:

```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype="float16")
input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
outputs = model.generate(**input_features, max_new_tokens=128)
print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
```

No-clone SFT, using the Hub `datasets` library directly [3]:

```python
from paddlenlp.trl import SFTConfig, SFTTrainer
from datasets import load_dataset

dataset = load_dataset("ZHUI/alpaca_demo", split="train")
training_args = SFTConfig(output_dir="Qwen/Qwen2.5-0.5B-SFT", device="gpu")
trainer = SFTTrainer(args=training_args, model="Qwen/Qwen2.5-0.5B-Instruct", train_dataset=dataset)
trainer.train()
```

The clone-and-script CLI form for SFT: `python -u run_finetune.py ./config/qwen/sft_argument_0p5b.json`, after fetching the demo `AdvertiseGen.tar.gz` archive [3].

## Start it

- One GPU is the base form for every method: drop the `paddle.distributed.launch` wrapper and call the script directly, e.g. `python run_finetune.py ./config/qwen/lora_argument_0p5b.json` [5].
- Several GPUs (or nodes) go through PaddlePaddle's own launcher: `python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py ./config/qwen/sft_argument.json`, and the identical pattern is used for DPO (`./alignment/dpo/run_dpo.py`), KTO (`./alignment/kto/run_kto.py`), reward modeling (`llm/alignment/rm/run_reward.py`), and RL (`llm/alignment/rl/run_rl.py`) - only the script and config path change [5][7][14][8]. RL additionally requires starting a `reward_server.py` process before launching training when not using a local reward model [8].
- `llm/README.md` publishes VRAM shapes as comments next to its example commands, not as a benchmark table, and the shapes are not directly comparable to each other because each one names a different model-size config, not just a different GPU count [4]: full SFT on a single GPU with the Qwen 0.5B config (`sft_argument_0p5b.json`) needs about 12G (16G for the "best-practice" variant, `sft_argument_0p5b_best.json`)†, while full SFT across 8 GPUs with a larger Qwen config (`sft_argument.json`) needs about 45G†‡; LoRA needs about 9G on a single GPU with the 0.5B config (`lora_argument_0p5b.json`) or about 29G on a single GPU with a larger config (`lora_argument.json`)†; Prefix Tuning needs about 10G or 30G under the same single-GPU 0.5B-vs-larger-config pairing (`pt_argument_0p5b.json` / `pt_argument.json`)†; full-parameter DPO needs about 26G on a single GPU with the Qwen 0.5B config (`dpo_argument_0p5b.json`) versus about 40G across 8 GPUs with the (larger) Llama config (`dpo_argument.json`)†‡; LoRA DPO with the same Llama config needs about 52G on a single GPU (`dpo_lora_argument.json`), and the README gives no VRAM figure for LoRA DPO on 8 GPUs [4]. † marks single-GPU-vs-single-GPU pairs that still differ by model-config size (0.5B config vs an unstated larger config), and ‡ additionally marks a change in GPU count - none of these figures isolate GPU count as the only variable, so none should be read as a clean single-GPU-to-8-GPU scaling ratio.
- The RL config surface (`llm/config/qwen/grpo_argument.yaml`) is large and documented field-by-field in the RL README: `rl_algorithm` selects `grpo` or `reinforce_plus_plus` [8], and a third, code-level-only value `ppo` is also accepted by the script but is not named in the README's glossary [9]; the rollout side is split from the training side via `global_batch_size` (prompts sampled per rollout step), `global_mini_batch_size` (prompts per actor-model update), and `rollout_n` (responses sampled per prompt); `per_device_train_batch_size`, `per_device_logprob_batch_size`, `per_device_reward_batch_size`, and `per_device_value_batch_size` are four separate per-device batch knobs, glossed by the RL README as: the actor model's loss-and-backprop pass, the log-prob computation pass, the critic model's loss-and-backprop pass (despite its "reward" name), and the critic model's forward values pass, respectively [8].
- FlashAttention-2 (`use_flash_attention`) is documented as off by default ("默认为 False") in the RL config glossary and must be explicitly turned on [8]. The same glossary lists a `bf16` field ("使用 bfloat16 精度进行模型训练和推理") but states no default value for it, unlike `use_flash_attention` - so whether a PaddleNLP RL run trains in bf16 by default is not established from this source [8].
- Out-of-memory first aid, as documented mechanism rather than an OOM-specific troubleshooting list: `recompute` (activation recomputation) with `recompute_granularity` set to `core_attn` (faster, less memory saved) or `full` (slower, more memory saved); `use_remove_padding` to drop padding tokens from training; and, for RL specifically, lowering `per_device_*_batch_size` fields or `rollout_max_num_seqs` (max sequences handled per inference call) [8][5].

## Watch it

This section is mechanics only - what a given metric means for a method lives on that method's methodology card, not here.

- **Enable it**: the RL config's `report_to` field accepts `"all"`, `"wandb"`, `"tensorboard"`, `"visualdl"`, or `"none"`; the shipped GRPO and REINFORCE++ configs set `logging_dir` to `vdl_log`, PaddlePaddle's own VisualDL format (a TensorBoard-like tool), viewable via `visualdl --logdir vdl_log --host 0.0.0.0`; switching to Weights & Biases needs `wandb` installed, logged in, and `"logging_dir": "wandb"` in the config [8]. `logging_steps` sets the print cadence [8].
- The fetched sources give no single live page enumerating every SFT/DPO/KTO/RM logged metric name by name; the RL README's config glossary is the most complete field list found, and it documents config knobs (batch sizes, KL and clip coefficients, evaluation cadence) rather than a fixed list of scalar log keys [8]. A reader who needs exact SFT/DPO metric key strings has to read the trainer source directly - not established from the docs fetched for this card.
- RL-specific knobs visible in the same glossary: `kl_coeff` and `kl_loss_coeff` (KL penalty terms), `pg_loss_coeff` and `entropy_coeff` (policy-gradient and entropy loss weights), `clip_range_ratio` (with separate `_low`/`_high` bounds for asymmetric PPO clipping), `clip_range_score` (reward clipping) and `clip_range_value` (critic-output clipping), and `normalize_reward`/`normalize_advantage` toggles [8].
- **Evaluate during training**: the RL glossary lists `do_eval`, `evaluation_strategy` (e.g. `"steps"`), `eval_steps`, and `per_device_eval_batch_size` as config fields [8].
- **Sample-level logging of generations** and a **published stopping-rule threshold** are not stated in any of the RL README, the fine-tuning README, or `llm/docs/algorithm_overview.md` fetched for this card - none of the three documents a health limit, an early-stopping field, or a generation-preview toggle; this is what the search found, not confirmation that no such field exists in the source code.

## Save it

- Two checkpoint formats coexist. The legacy, non-unified format shards model/optimizer state by parallelism rank directly into files like `model_state.tp00_pp00.pdparams` and `optimizer.tp00_pp00.pdopt` alongside `scheduler.pdparams`, `scaler.pdparams`, `rng_state_8.pth`, and the tokenizer files, inside the trainer's `output_dir` [19].
- The **Unified Checkpoint** format, enabled with `--unified_checkpoint 1`, instead writes sharded `.safetensors` files uniformly regardless of tensor/pipeline-parallel degree: `model-000NN-of-000MM.safetensors` plus a `model.safetensors.index.json` for the weights, and separate `optimizer-*.safetensors`/`optimizer.safetensors.index.json` and `master_weights-*.safetensors`/`master_weights.safetensors.index.json` files for optimizer and FP32 master-weight state [19]. `unified_checkpoint_config` takes space-separated flags: `skip_save_model_weight` (skip weights, e.g. when only resuming optimizer state elsewhere), `master_weight_compatible`, `remove_master_weight`, and `async_save` (asynchronous, non-blocking checkpoint writes) [19].
- Retention: the RL glossary's `ignore_save_lr_and_optim` flag, and the general `unified_checkpoint_config`'s `skip_save_model_weight`/`remove_master_weight` flags, each drop part of the state needed to resume exactly - the unified-checkpoint doc states that when a checkpoint folder already holds legacy-format files, PaddleNLP loads them in the legacy way and re-saves subsequent checkpoints in Unified Checkpoint format instead [8][19].
- PEFT (LoRA/Prefix) saves are adapter-only: `model.save_pretrained('lora_path')` writes `lora_mode_state.pdparams` (the trained A/B matrices) and `lora_config.json` into `lora_path` - NOT a full model directory [20]. Reload pairs the adapter with the base model - `config = LoRAConfig.from_pretrained('lora_path')` then `model = LoRAModel.from_pretrained(model, 'lora_path')` - and `save_pretrained(save_directory, merge_tensor_parallel=True)` additionally merges tensor-parallel shards into a single trainable state dict at save time when tensor-parallel degree exceeds 1 [20].
- Cadence: `save_steps` (RL glossary) sets the checkpoint interval; the SFT/DPO/RL scripts all write into the run's `output_dir` [8].
- Whether an evaluator can load a saved checkpoint directly depends on which of the two formats and which retention flags were used - read this skill's shared `references/loading-the-result.md` before the first save, and treat an adapter directory the same way the PEFT contract above describes: it is not a loadable full model on its own.

## Find it in the docs

The docs are the live source; this section teaches the lookup rather than mirroring the content.

- Two READMEs disagree in the same repository: the Chinese `README.md`, read at the pinned commit, documents `pip install --upgrade paddlenlp==3.0.0b4` and `paddlepaddle >= 3.0.0rc1`; the English `README_en.md`, read unpinned from the `main` branch, still shows the older `paddlenlp==3.0.0b3` and `paddlepaddle >= 3.0.0b0` and a News list that stops months earlier - a live, unpinned English page can lag the pinned Chinese one by more than one release, so check which language file you are actually reading before trusting a version number [3][1].
- The default branch is `develop`, not `main` [2]; in-repo relative links (e.g. to `llm/config/llama`) resolve against `develop` on GitHub.
- The rendered documentation site is ReadTheDocs at `https://paddlenlp.readthedocs.io`, linked from the README badge and repo homepage field [3][2]; the fetched sources for this card are the raw GitHub Markdown files under `llm/docs/`, not the rendered ReadTheDocs pages, so page slugs on the rendered site were not independently verified.
- Question-to-file map inside the repo, all under `llm/docs/` at the pinned commit: fine-tuning detail and the LoKr/VeRA/MoRA/ReFT/rsLoRA/LoRA+/PiSSA/MoSLoRA list -> `finetune.md`; DPO/SimPO/ORPO -> `dpo.md`; PEFT internals (LoRA/Prefix save-and-reload API) -> `peft.md`; Unified Checkpoint format and compatibility -> `unified_checkpoint.md`; cross-cutting training-acceleration techniques (Zero Padding, FlashAttention2, FlashMask, NEFT, LoRA+, rsLoRA, PiSSA, VeRA) -> `algorithm_overview.md`; a separate, differently-organized PPO walkthrough (referencing a `./alignment/ppo` directory layout not present at the pinned commit - the current PPO code lives under `llm/alignment/rl/`, so this page describes a superseded layout) -> `rlhf.md` [5][7][20][19][13][21].
- The GRPO/REINFORCE++/PPO alignment guide is not under `llm/docs/`; it is `llm/alignment/rl/README.md`, and `llm/README.md` explicitly points there for all three RL algorithms [8][4].
- Runnable references beyond the docs: the RL smoke-test dataset is a pre-templated Knights-and-Knaves set at `https://paddlenlp.bj.bcebos.com/datasets/examples/ppo-kk.tgz` [8]; `llm/README.md`'s own clone-and-script examples fetch `alpaca_demo.gz` for SFT and `ultrafeedback_binarized_pointwise.tar` for KTO [4]; DPO and RM both fetch `ultrafeedback_binarized.tar.gz`, per the DPO doc and the RM README respectively [7][14] - all hosted on Baidu's `bcebos.com` object storage rather than the Hugging Face Hub. The RL README also links two Weights & Biases run logs as its own published reproducibility record for the GRPO and REINFORCE++ example commands [8].
- No community-tutorials page or official MCP endpoint was found in the sources fetched for this card; this is what the search found among the README, `llm/README.md`, and the `llm/docs/` files listed above, not confirmation that neither exists anywhere in the project.
- Stated boundary: PaddleNLP's trainers are PaddlePaddle-only - there is no PyTorch code path in the sources read, so a PyTorch checkpoint or PyTorch-only workflow needs the separate `torch2paddle.md` conversion guide referenced from `llm/README.md`'s table of contents, not a direct load [4].

## Sources

All GitHub file citations are read at the shortlist's pinned commit `3f87dac9b719f75399f92f8bf634ae2ef0611832` unless a release tag is named explicitly; all fetches performed 2026-08-10.

[1] PaddleNLP `README_en.md`, raw file read unpinned from the `main` branch. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/main/README_en.md. Fetched 2026-08-10.

[2] PaddleNLP GitHub repository (GitHub API). https://api.github.com/repos/PaddlePaddle/PaddleNLP. Item home page: https://github.com/PaddlePaddle/PaddleNLP. Fetched 2026-08-10.

[3] PaddleNLP README.md, raw file at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/README.md. Fetched 2026-08-10.

[4] `llm/README.md` ("飞桨大模型套件"), raw file at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/README.md. Fetched 2026-08-10.

[5] `llm/docs/finetune.md`, raw file at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/docs/finetune.md. Fetched 2026-08-10.

[6] `paddlenlp/trl/__init__.py`, raw file at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/paddlenlp/trl/__init__.py. Fetched 2026-08-10.

[7] `llm/docs/dpo.md`, raw file at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/docs/dpo.md. Fetched 2026-08-10.

[8] `llm/alignment/rl/README.md` ("GRPO && REINFORCE++"), raw file at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/alignment/rl/README.md. Fetched 2026-08-10.

[9] `llm/alignment/rl/run_rl.py`, raw file at the pinned commit (the `rl_algorithm == "ppo"` branch). https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/alignment/rl/run_rl.py. Fetched 2026-08-10.

[10] `paddlenlp/rl/trainer/` directory listing (GitHub Contents API) at the pinned commit. https://api.github.com/repos/PaddlePaddle/PaddleNLP/contents/paddlenlp/rl/trainer?ref=3f87dac9b719f75399f92f8bf634ae2ef0611832. Fetched 2026-08-10.

[11] `paddlenlp/rl/trainer/reward_trainer.py`, raw file at the pinned commit (`class RewardTrainer(RLTrainer)`, line 42). https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/paddlenlp/rl/trainer/reward_trainer.py. Fetched 2026-08-10.

[12] `llm/alignment/rm/reward_trainer.py`, raw file at the pinned commit (`class RewardTrainer(Trainer)`, line 22). https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/alignment/rm/reward_trainer.py. Fetched 2026-08-10.

[13] `llm/docs/algorithm_overview.md`, raw file at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/docs/algorithm_overview.md. Fetched 2026-08-10.

[14] `llm/alignment/rm/README.md` ("FlashMask Reward model training") and its directory listing, raw file and GitHub Contents API at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/alignment/rm/README.md ; https://api.github.com/repos/PaddlePaddle/PaddleNLP/contents/llm/alignment/rm?ref=3f87dac9b719f75399f92f8bf634ae2ef0611832. Fetched 2026-08-10.

[15] paddlenlp on PyPI (JSON API: latest version, upload dates, per-release file list). https://pypi.org/pypi/paddlenlp/json. Fetched 2026-08-10.

[16] `setup.py`, raw file at the `v3.0.0-beta4` release tag, which resolves to commit `a286abc1063e516ed56b746fcca33bedce5fcef3` per the GitHub tags API (`https://api.github.com/repos/PaddlePaddle/PaddleNLP/tags`). https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/v3.0.0-beta4/setup.py. Fetched 2026-08-10.

[17] `requirements.txt`, raw file at the `v3.0.0-beta4` release tag, same resolved commit `a286abc1063e516ed56b746fcca33bedce5fcef3` as [16]. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/v3.0.0-beta4/requirements.txt. Fetched 2026-08-10.

[18] PaddleNLP GitHub Releases (GitHub API, up to 100 most recent). https://api.github.com/repos/PaddlePaddle/PaddleNLP/releases?per_page=100. Fetched 2026-08-10.

[19] `llm/docs/unified_checkpoint.md`, raw file at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/docs/unified_checkpoint.md. Fetched 2026-08-10.

[20] `llm/docs/peft.md`, raw file at the pinned commit. https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/docs/peft.md. Fetched 2026-08-10.

[21] `llm/docs/rlhf.md`, raw file at the pinned commit (superseded PPO walkthrough, cited only to note it describes a directory layout no longer present). https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/docs/rlhf.md. Fetched 2026-08-10.
