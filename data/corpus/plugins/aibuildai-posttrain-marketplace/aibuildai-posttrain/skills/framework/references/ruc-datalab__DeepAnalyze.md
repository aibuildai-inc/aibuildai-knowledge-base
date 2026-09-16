# DeepAnalyze

Not a general post-training library: DeepAnalyze is a data-science agent model release whose "Develop Your Own DeepAnalyze" recipe reproduces its curriculum by calling vendored copies of ms-swift (SFT) and SkyRL (GRPO), not by exposing its own trainer API.

It lives at https://github.com/ruc-datalab/DeepAnalyze [1]. The repository describes DeepAnalyze as "the first agentic LLM for autonomous data science" that "can autonomously complete a wide range of data-centric tasks without human intervention" such as data preparation, analysis, modeling, visualization, and report generation [1]. It is built by researchers at Renmin University of China and Tsinghua University [2], and ships a served model (`DeepAnalyzeVLLM`, deployed via `vllm serve`) plus a chat WebUI and OpenAI-style API for using the trained agent [1] - it is not organized around a trainer-class or Config-object API for training arbitrary models. Reproducing its own training instead means installing and invoking two other frameworks that are copied whole into the repo under `deepanalyze/ms-swift/` and `deepanalyze/SkyRL/`, with no `.gitmodules` pinning them as submodules - they are vendored source trees at whatever commit was copied in [3][4].

**When to pick it**: pick DeepAnalyze if the goal is the DeepAnalyze-8B data-science agent itself (its model, inference stack, or its exact three-stage curriculum) - not as a library choice for training your own arbitrary model. For a generic post-training library, use ms-swift or SkyRL directly rather than through DeepAnalyze's vendored, path-hardcoded copies (its scripts still contain placeholder paths like `PATH_TO_MODEL_ADD_VOCAB` and `PATH_TO_SAVE_MODEL`) [5]; those upstream projects are covered on their own cards (cross-reference, not covered here).

**Methods it ships**: DeepAnalyze's own three documented training stages use exactly two methods: full-parameter SFT for two curriculum stages (single-ability, then multi-ability "cold start"), run through ms-swift's `swift sft` CLI [5], and one RL stage using GRPO (`trainer.algorithm.advantage_estimator="grpo"`), run through a vendored SkyRL via `python -m examples.deepanalyze.main_deepanalyze` inside `deepanalyze/SkyRL/skyrl-train/` [5]. For the RL stage, DeepAnalyze adds its own SkyRL Gym environment, `DeepAnalyzeEnv` in `examples/deepanalyze/deepanalyze_env.py`; its `_get_reward` method builds a dict of named reward components per task type and returns their mean (`sum(rewards.values()) / len(rewards)`), not a sum: for QA tasks the components are a table-QA accuracy score and an LLM-as-judge analysis score; for data/open-research tasks the components come from calling every function named in `reward_spec["function"]` (resolved via `globals()`) plus a code-execution-pass-rate term, and, for open-ended research specifically, an added turn-count term capped at 1.0 for 10+ assistant turns [6]. Code search also finds DPO, KTO, ORPO, PPO, CPO, GKD and Reward-model trainer classes inside `deepanalyze/ms-swift/swift/trainers/trainer_factory.py`, and a DAPO example inside `deepanalyze/SkyRL/skyrl-train/examples/algorithms/dapo/` [7][8] - these are unmodified, generic capabilities of the vendored ms-swift and SkyRL trees, not methods that any DeepAnalyze script in `scripts/` actually invokes [5][7][8].

**Scale it handles**: the three published scripts (`scripts/single.sh`, `scripts/multi_coldstart.sh`, `scripts/multi_rl.sh`) are all written for one 8-GPU node (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `NPROC_PER_NODE=8` or `NUM_GPUS=8`) with DeepSpeed ZeRO-3 for the SFT stages and FSDP2 with CPU offload for the RL stage [5]; no multi-node launch form or benchmark for DeepAnalyze's own training is published in the README or these scripts. Multi-node scale, if needed, is a property of the vendored ms-swift and SkyRL frameworks themselves, not something DeepAnalyze's own docs demonstrate.

**Install**: `git clone` then `pip install -r requirements.txt` for inference (unpinned `torch`, `transformers`, `vllm>=0.8.5`, Python 3.12 per the README's `conda create -n deepanalyze python=3.12`) [5]. Training additionally needs `(cd ./deepanalyze/ms-swift/ && pip install -e .)` and `(cd ./deepanalyze/SkyRL/ && pip install -e .)`, and the README itself recommends separating the inference and training environments to avoid dependency conflicts [5]. The repository has no GitHub releases or tags at all (both API endpoints return empty lists), so `git clone` pulls unpinned `main` HEAD - whatever the vendored-framework pins and hardware notes below say about the reviewed commit `0b54741848df5c157303eb82f80a8dca1afb438f` is not guaranteed to be what a `git clone` run today delivers, since the repo's own last-push timestamp (2026-07-01) postdates this commit [14]. At commit `0b54741848df5c157303eb82f80a8dca1afb438f`, ms-swift's `deepanalyze/ms-swift/requirements/framework.txt` pins `transformers>=4.33,<4.53`, `trl>=0.15,<0.20`, `peft>=0.11,<0.16`, `datasets>=3.0,<3.4`, and `numpy<2.0` [9]; the vendored SkyRL's top-level `deepanalyze/SkyRL/pyproject.toml` requires Python `==3.12.*` and is MIT-licensed [4], while ms-swift itself is Apache-2.0, distinct from DeepAnalyze's own MIT licence [3][10]. No CUDA version or GPU minimum is stated for the training scripts (`scripts/*.sh` assume an 8-GPU host but state no VRAM floor) [5], but the README does give an inference-serving GPU-memory table: it recommends a minimum of 16GB GPU memory (4-bit-quantized model, FP8 KV cache, `max-model-len` 49152) up to 80GB (original model, `max-model-len` 131072, no FP8) for serving DeepAnalyze-8B through vLLM [15].

**Maintained by**: RUC-DataLab (Renmin University of China's data lab), with authors also affiliated with Tsinghua University [2]; the GitHub repo was created 2025-10-11 and last pushed 2026-07-01, and its README news log shows continued activity through 2026.07 (a companion project, DeepPrep, announced as upcoming) [1].

## Quick start

The README's own example (inference only - training has no equally short "quick start", see Start it) [1]:

```python
from deepanalyze import DeepAnalyzeVLLM

prompt = """# Instruction
Generate a data science report.

# Data
File 1: {"name": "bool.xlsx", "size": "4.8KB"}
..."""

workspace = "/path/to/example/analysis_on_student_loan/"

deepanalyze = DeepAnalyzeVLLM(
    "/path/to/checkpoints/deepanalyze-8b/"
)
answer = deepanalyze.generate(prompt, workspace=workspace)
print(answer["reasoning"])
```

CLI form for serving the model: `vllm serve DeepAnalyze-8B`, then interact through the API or WebUI (`cd demo/chat/frontend && npm install && cd .. && bash start.sh`) [1].

## Start it

Training is a three-stage curriculum, each stage a single 8-GPU-node script with hardcoded placeholder paths the user must fill in [5]:

- **Stage 1 - single-ability SFT** (`scripts/single.sh`): full-parameter SFT from a base model (DeepSeek-R1-0528-Qwen3-8B, with special tokens added via `deepanalyze/add_vocab.py`) via `swift sft --train_type "full"`, 13 curriculum datasets from DataScience-Instruct-500K, `per_device_train_batch_size 8`, `gradient_accumulation_steps 4`, `learning_rate 5e-5`, 3 epochs, `deepspeed "zero3"`, `use_liger_kernel true`, `attn_impl "flash_attn"` [5].
- **Stage 2 - multi-ability "cold start" SFT** (`scripts/multi_coldstart.sh`): same `swift sft` command against the Stage-1 checkpoint, 12 different curriculum datasets, `per_device_train_batch_size 1`, `gradient_accumulation_steps 32`, `learning_rate 5e-6` [5].
- **Stage 3 - GRPO RL** (`scripts/multi_rl.sh`): `python -m examples.deepanalyze.main_deepanalyze` from inside `deepanalyze/SkyRL/skyrl-train/`, `trainer.algorithm.advantage_estimator="grpo"`, `trainer.strategy="fsdp2"` with `cpu_offload=true` for both policy and reference models, `trainer.train_batch_size=256`, `trainer.policy_mini_batch_size=256`, `generator.n_samples_per_prompt=5`, vLLM as the generation backend (`generator.backend="vllm"`, `generator.async_engine=true`, `generator.gpu_memory_utilization=0.5`), `trainer.algorithm.use_kl_loss=false` [5].
- Effective batch size for the RL stage is `trainer.train_batch_size=256` prompts per update, each with `generator.n_samples_per_prompt=5` generations, split into `trainer.policy_mini_batch_size=256`-sample minibatches and `trainer.micro_train_batch_size_per_gpu=1` per-GPU micro-batches under FSDP2 [5]; for the SFT stages effective batch is `per_device_train_batch_size x NPROC_PER_NODE x gradient_accumulation_steps` (Stage 1: 8x8x4=256; Stage 2: 1x8x32=256) [5].
- Configuration surface is inherited entirely from the two vendored frameworks - ms-swift's `swift sft` CLI flags for Stages 1-2, SkyRL's Hydra-style dotted overrides (`trainer.*`, `generator.*`, `environment.*`) for Stage 3 - DeepAnalyze's own scripts only set values, they do not define or change either framework's defaults [5].
- Out-of-memory first aid: not published as a dedicated section anywhere searched (README, docs/FAQ.md); the RL script itself demonstrates the mitigation pattern in its own values - `trainer.policy.fsdp_config.cpu_offload=true`, `trainer.ref.fsdp_config.cpu_offload=true`, and `generator.gpu_memory_utilization=0.5` - rather than documenting them as a troubleshooting guide [5].

## Watch it

**RL stage (SkyRL)**: the vendored SkyRL's own base config, `deepanalyze/SkyRL/skyrl-train/skyrl_train/config/ppo_base_config.yaml`, defaults `trainer.logger` to `"wandb"` [16]; DeepAnalyze's own `scripts/multi_rl.sh` overrides this to `trainer.logger="[\"console\",\"tensorboard\"]"`, so a run launched exactly as that script specifies logs to the console and to TensorBoard, not Weights & Biases, unless the operator changes it back [5]. `main_deepanalyze.py` is a thin wrapper that registers the `DeepAnalyzeEnv` Gym environment and then calls SkyRL's own `BasePPOExp(cfg).run()` [17] - the specific metric names that trainer writes to those logs are SkyRL's own metric surface and are not restated in any DeepAnalyze file read for this card; consult SkyRL's own card/docs for that name list.
**SFT stages (ms-swift)**: the scripts set `--logging_steps 1` but no `report_to`/tracker backend, so by default logging goes wherever `swift sft`'s underlying `transformers`-derived trainer sends it (console/log file) unless the operator adds a tracker flag; DeepAnalyze's own scripts do not set one [5].
**DeepAnalyze-specific reward signal**: the RL environment's `DeepAnalyzeEnv._get_reward` is the one training-time signal DeepAnalyze itself defines (see Methods it ships for the formula) [6]; the SkyRL logger receives whatever fields SkyRL's own trainer chooses to report about it, not a DeepAnalyze-defined metric name.
No RL-specific stopping rule, threshold, or patience value is published in the README or `docs/FAQ.md` (docs/FAQ.md's two entries, checked 2026-08-11, cover Windows vLLM deployment and a matplotlib Chinese-font fix only, nothing about training) [11].

## Save it

**SFT stages (ms-swift)**: the scripts set `--save_steps 50 --save_total_limit 3 --save_only_model false --output_dir "${MODEL_...}"` [5]. The vendored `swift/trainers/mixin.py` builds each checkpoint directory as `os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")`, the same `checkpoint-<step>` naming convention `transformers.Trainer` uses, and overrides `push_to_hub` to wrap `transformers.Trainer`'s implementation in its own hub-patching context manager (`with self.hub.patch_hub(): return super().push_to_hub(*args, **kwargs)`) [18] - DeepAnalyze's own scripts run with `save_only_model false`, i.e. optimizer/scheduler state is kept alongside the model weights in each checkpoint, which is what makes `resume_from_checkpoint` possible for these SFT runs.
**RL stage (SkyRL)**: DeepAnalyze's own `scripts/multi_rl.sh` sets `trainer.ckpt_path`, `trainer.export_path`, `trainer.hf_save_interval=1`, `trainer.ckpt_interval=1`, `trainer.resume_mode="latest"` [5]. SkyRL's own base config comments the meaning of each field: `ckpt_path` is the "Path for resumable training checkpoints (model state, optimizer state, etc.)", written every `ckpt_interval` steps (with `max_ckpts_to_keep` controlling retention); `export_path` is the "Path for exported artifacts (HF models, debug dumps, etc.)", written every `hf_save_interval` steps; `resume_mode: latest` resumes training from the most recent checkpoint under `ckpt_path` [16]. With DeepAnalyze's own values (`ckpt_interval=1`, `hf_save_interval=1`), every step writes both a full resumable checkpoint and an HF-format export.
The released model itself, `RUC-DataLab/DeepAnalyze-8B` on the Hugging Face Hub, is a standard sharded-safetensors checkpoint (`model-0000{1..4}-of-00004.safetensors` plus `model.safetensors.index.json`, `config.json`, tokenizer files), directly loadable with a standard `from_pretrained`/`vllm serve` call, not an adapter directory [12].

## Find it in the docs

DeepAnalyze publishes no versioned docs site of the kind that takes a `<version>` tag in its URL; its documentation is the GitHub README plus a small `docs/` folder, all read at the single commit `0b54741848df5c157303eb82f80a8dca1afb438f` unless noted [1][11]. Lookup recipes:

- `README.md` at the repo root is the primary reference: installation, quick start, WebUI, API, and the "Develop Your Own DeepAnalyze" training section are all sub-headings within it - use in-page search rather than separate URLs [1].
- `docs/FAQ.md` (English) and `docs/FAQ_ZH.md` (Chinese) hold troubleshooting entries; as of this commit only two entries exist (Windows vLLM, matplotlib fonts), so most operational questions are not yet covered there [11].
- `docs/DeepAnalyze_API_Key_Usage_Guide.md` covers the hosted API-key program (HeyWhale-provided), not training [1].
- Training specifics live in the three scripts under `scripts/` (`single.sh`, `multi_coldstart.sh`, `multi_rl.sh`) and in the vendored framework trees they call: `deepanalyze/ms-swift/` for the SFT CLI and `deepanalyze/SkyRL/skyrl-train/` for the RL script and the custom `examples/deepanalyze/` environment [5][6].
- The training data referenced by every script, `DataScience-Instruct-500K`, is a Hugging Face dataset at `RUC-DataLab/DataScience-Instruct-500K`; the README names its `interation/` (SFT curriculum) and `RL/` (GRPO parquet files) subfolders as the paths the scripts expect [1][5].
- The defining paper is the arXiv abs page at `https://arxiv.org/abs/2510.16872`, "DeepAnalyze: Agentic Large Language Models for Autonomous Data Science" [13]; method-level math for GRPO and SFT is not restated here - it belongs on those methods' own cards.
- Community layer: the README lists a homepage (`https://ruc-deepanalyze.github.io`) with usage-case demos, and credits a WebUI JupyterUI integration by an external contributor built on `jupyter-mcp-server`; no official MCP endpoint for querying DeepAnalyze's own docs is stated [1].
- Honest boundary: DeepAnalyze's published training path only reproduces its own DeepSeek-R1-0528-Qwen3-8B-based, 8-GPU, single-node curriculum; no documented path or benchmark supports multi-node training, alternative base-model architectures beyond the vocab-extension step, or any RL algorithm other than GRPO, within DeepAnalyze's own scripts [5].

## Sources

Read at commit `0b54741848df5c157303eb82f80a8dca1afb438f` unless a different source is named. Method names (SFT, GRPO, DPO, KTO, ORPO, PPO, CPO, GKD, DAPO) are deliberately cited to nothing beyond their appearance in DeepAnalyze's own scripts or vendored trainer code; their defining papers and math live on the methodology cards. The vendored ms-swift and SkyRL frameworks are named where DeepAnalyze's own files call or bundle them, and are otherwise left to their own cards (cross-reference, not covered here).

[1] DeepAnalyze README. https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/README.md. Fetched 2026-08-11.

[2] DeepAnalyze README, author/affiliation block. Same URL as [1]. Fetched 2026-08-11.

[3] ms-swift LICENSE inside DeepAnalyze's vendored copy (Apache-2.0). https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/deepanalyze/ms-swift/LICENSE. Fetched 2026-08-11.

[4] Vendored SkyRL top-level pyproject.toml (name "skyrl", MIT licence, `requires-python = "==3.12.*"`); absence of a `.gitmodules` file at this commit (404 on the raw URL) confirms ms-swift and SkyRL are copied source trees, not submodules. https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/deepanalyze/SkyRL/pyproject.toml ; https://raw.githubusercontent.com/ruc-datalab/DeepAnalyze/0b54741848df5c157303eb82f80a8dca1afb438f/.gitmodules. Fetched 2026-08-11.

[5] DeepAnalyze training scripts and README training/install sections: `scripts/single.sh`, `scripts/multi_coldstart.sh`, `scripts/multi_rl.sh`, and the README's Requirements and "Develop Your Own DeepAnalyze" sections. https://github.com/ruc-datalab/DeepAnalyze/tree/0b54741848df5c157303eb82f80a8dca1afb438f/scripts ; https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/README.md. Fetched 2026-08-11.

[6] DeepAnalyze's custom SkyRL Gym environment. https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/deepanalyze/SkyRL/skyrl-train/examples/deepanalyze/deepanalyze_env.py. Fetched 2026-08-11.

[7] Vendored ms-swift trainer factory, listing generic trainer classes (DPO, ORPO, KTO, CPO, RM, PPO, GRPO, GKD) unrelated to which trainers DeepAnalyze's own scripts invoke. https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/deepanalyze/ms-swift/swift/trainers/trainer_factory.py. Fetched 2026-08-11.

[8] Vendored SkyRL DAPO example directory, present in the copied SkyRL tree but not called by any DeepAnalyze script. https://github.com/ruc-datalab/DeepAnalyze/tree/0b54741848df5c157303eb82f80a8dca1afb438f/deepanalyze/SkyRL/skyrl-train/examples/algorithms/dapo. Fetched 2026-08-11.

[9] Vendored ms-swift framework dependency pins. https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/deepanalyze/ms-swift/requirements/framework.txt. Fetched 2026-08-11.

[10] DeepAnalyze's own LICENSE file (MIT, copyright RUC-DataLab). https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/LICENSE. Fetched 2026-08-11.

[11] DeepAnalyze FAQ. https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/docs/FAQ.md. Fetched 2026-08-11.

[12] RUC-DataLab/DeepAnalyze-8B model repository file listing (Hugging Face Hub API). https://huggingface.co/api/models/RUC-DataLab/DeepAnalyze-8B. Fetched 2026-08-11.

[13] DeepAnalyze arXiv abstract page. https://arxiv.org/abs/2510.16872. Fetched 2026-08-11.

[14] GitHub REST API, repository metadata (stars, license, push date, description, and empty `releases`/`tags` lists confirming no pinned release exists). https://api.github.com/repos/ruc-datalab/DeepAnalyze ; https://api.github.com/repos/ruc-datalab/DeepAnalyze/releases ; https://api.github.com/repos/ruc-datalab/DeepAnalyze/tags. Fetched 2026-08-11.

[15] DeepAnalyze README, "Memory Configuration Recommended Parameters Table" and vLLM launch command examples (inference-serving GPU-memory floors). Same URL as [1]. Fetched 2026-08-11.

[16] Vendored SkyRL base Hydra config, defining `trainer.logger` default, `ckpt_path`/`export_path`/`ckpt_interval`/`hf_save_interval`/`resume_mode` semantics. https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/deepanalyze/SkyRL/skyrl-train/skyrl_train/config/ppo_base_config.yaml. Fetched 2026-08-11.

[17] DeepAnalyze's SkyRL entrypoint wrapper, registering the `DeepAnalyzeEnv` Gym environment and delegating to SkyRL's own `BasePPOExp`. https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/deepanalyze/SkyRL/skyrl-train/examples/deepanalyze/main_deepanalyze.py. Fetched 2026-08-11.

[18] Vendored ms-swift trainer mixin, defining the `checkpoint-<global_step>` directory naming and the `push_to_hub` override. https://github.com/ruc-datalab/DeepAnalyze/blob/0b54741848df5c157303eb82f80a8dca1afb438f/deepanalyze/ms-swift/swift/trainers/mixin.py. Fetched 2026-08-11.

GitHub issue search for maintainer traps (https://api.github.com/search/issues, query `repo:ruc-datalab/DeepAnalyze is:issue training`) returned 2 open issues at fetch time - #116 (GRPO training NaN losses) and #82 (SQLite-dialect SQL generation bias) - both open, not closed, so per the closed-issue-only rule neither is reported as a trap on this card.
