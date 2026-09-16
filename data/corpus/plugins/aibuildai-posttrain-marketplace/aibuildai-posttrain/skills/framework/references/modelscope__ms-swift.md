# ms-swift

ModelScope's fine-tuning and deployment framework: a CLI (`swift sft`, `swift rlhf`, `swift infer`, `swift export`, `swift rollout`, `swift megatron`) over PEFT/full-parameter training plus a Megatron backend for large-scale runs.

**ms-swift** ("Scalable lightWeight Infrastructure for Fine-Tuning") is "a large model and multimodal large model fine-tuning and deployment framework provided by the ModelScope community", supporting pre-training, fine-tuning, human alignment, inference, evaluation, quantization, and deployment for 600+ text-only and 400+ multimodal large models [1]. It is built and maintained by the ModelScope community [1], with the packaged project listing "DAMO ModelScope teams" as its author [10]. Its API is a set of CLI subcommands, each taking a `--model` id and a `--dataset` id and producing an `output_dir` of checkpoints [3]. It lives at https://github.com/modelscope/ms-swift [2].

**When to pick it**: fine-tuning or RLHF on models pulled from the ModelScope hub (or Hugging Face, via `--use_hf true`) [3], when you want one CLI to cover LoRA/full-parameter SFT, RLHF (DPO/KTO/GRPO/PPO/RM and more), and a separate Megatron-parallelism backend for dense and MoE models too large for DeepSpeed sharding on the same hardware [4][7]. Its GRPO trainer supports multimodal rollouts and both a colocated and a dedicated-server vLLM layout [8]. Weigh against trl if the work is Hub-native and single-to-few-GPU (cross-reference; not covered here); weigh against verl if the target is Ray-scheduled cluster RL from the start (cross-reference; not covered here).

**Methods it ships**: pre-training and SFT (LoRA and full-parameter) via `swift sft`; RLHF methods via `swift rlhf --rlhf_type <type>` covering DPO, KTO, GRPO, PPO, RM (reward modeling), CPO, SimPO, ORPO, GKD; embedding, reranker, and sequence-classification training [1][5]. The Quick-start page additionally lists a wider GRPO algorithm family it integrates - GRPO, DAPO, GSPO, SAPO, CISPO, RLOO, Reinforce++ - as reinforcement-learning algorithms alongside Megatron parallelism [4]; of these, GSPO is documented as a configuration option on the GRPO trainer itself, selected with `--importance_sampling_level sequence` [19][18]. Method math and defining papers live on the method-level cards, not here.

**Scale it handles**: single GPU up to multi-node, through two separate backends. The single-node backend launches with `CUDA_VISIBLE_DEVICES` and documents device-map model partitioning, DeepSpeed ZeRO-2/ZeRO-3 sharding, and FSDP+QLoRA (with a stated example of training a 70B model on two 3090 GPUs) as memory-scaling options; multi-node runs are documented as example shell scripts covering `swift`, `torchrun`, `dlc`, `deepspeed`, and `accelerate` launchers, noting that all but the `dlc` and `deepspeed` scripts must be started on every node [6]. The Megatron-SWIFT backend (`swift megatron sft`) adds Megatron's tensor, pipeline, sequence, context, and expert parallelism and is documented to support CPT/SFT/GRPO/DPO/KTO/RM for Qwen3, Qwen3.5, DeepSeek-R1, GLM4.5, GPT-OSS, and others [7]. On one published benchmark, dense Qwen2.5-14B full-parameter training at 8K context on a single 8x A800 node ran at 9.04s/it under Megatron-LM against 10.32s/it (DeepSpeed ZeRO-2) and 10.56s/it (ZeRO-3); on MoE Qwen3-30B-A3B at 8K context across two 16x A800 nodes, Megatron-LM ran at 9.6s/it while DeepSpeed ZeRO-2 ran out of memory and ZeRO-3 needed 91.2s/it - the deciding number for choosing Megatron-SWIFT on MoE models at this scale [7]. GRPO's colocated vLLM mode shares training GPUs; its server mode runs `swift rollout` as a separate process, letting rollout and training GPUs scale independently [8].

**Install**: `pip install ms-swift`; PyPI version 4.4.2, uploaded 2026-07-21 [9]. `python_requires>=3.8.0` per package metadata [10], though the docs recommend Python 3.12 and state a minimum of 3.10 for the documented environment [11] - the two disagree and the stricter docs floor should be trusted for a working setup. Apache-2.0 licence [9][10]. At the v4.4.2 release tag (commit `f48847d`) [12], `requirements/framework.txt` pins `transformers>=4.33,<5.13.0`, `peft>=0.11,<0.20`, `datasets>=3.0,<4.8.5`, `trl>=0.15,<1.0`, `gradio>=3.40.0,<6.0`, `modelscope>=1.23` [13]; torch is not pinned by ms-swift itself and arrives transitively - the docs recommend torch>=2.0, with 2.8.0 or 2.11.0 as the tested versions, and CUDA 12.8/13.0 (unneeded for CPU/NPU/MPS) [11]. The `megatron` extra (`pip install ms-swift[megatron]`) additionally pins `megatron-core>=0.16`, `mcore-bridge>=1.5.0`, `peft>=0.15` [14]; other extras are `eval` (`evalscope>=1.0.0`), `swanlab`, and `ray` [15]. This v4.4.2 tag's commit is 19 days behind the repository's newest push (commit `3738096`, 2026-08-09) [16][2] - any claim sourced from the live docs or HEAD-of-main code past this point is not covered by this install pin. No hardware minimum is stated beyond the recommended CUDA versions above; the docs' own hardware table marks T4/V100 and Ascend NPU with "some models may encounter NAN," and lists A10/A100/H100, RTX 20/30/40, and CPU with no remarks [11].

**Maintained by**: the ModelScope community, packaged under author "DAMO ModelScope teams" with contact `contact@modelscope.cn` per package metadata [10]; the repository shows a push on 2026-08-09 and five releases from v4.3.1 (2026-06-17) through v4.4.2 (2026-07-21) in roughly two-to-four-week intervals [2][17]; the project's own README news feed shows continued activity, with its most recent entry on 2026-07-22 adding inference support for the Kimi-K3 multimodal model, following earlier 2025 entries such as Megatron GRPO support (2025-11-14) and a Megatron/Mcore-Bridge integration (2025-11-04) [1].

## Quick start

Smallest complete run, quoted from the project's own Quick-start page [3], LoRA SFT on a single 3090 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

The same page's inference step reads the saved adapter directly - `--adapters` pointed at the last checkpoint folder - and merges plus serves it with vLLM: `swift infer --adapters output/vx-xxx/checkpoint-xxx --merge_lora true --infer_backend vllm --vllm_max_model_len 8192` [3]. RLHF runs use the same CLI shape with a different subcommand and a `--rlhf_type` flag, e.g. `swift rlhf --rlhf_type dpo --model <model> --dataset <dataset> ...` [5].

## Start it

- One GPU: `CUDA_VISIBLE_DEVICES=0 swift sft ...` as above [3].
- Larger single-node jobs: device-map partitioning, DeepSpeed ZeRO-2/ZeRO-3, or FSDP+QLoRA, each documented on the Pre-training-and-Fine-tuning page with its own tradeoff (ZeRO-3 shards parameters on top of ZeRO-2 for more memory savings at more speed cost) [6].
- Multi-node: the repository ships example shell scripts for the `swift`, `torchrun`, `dlc`, `deepspeed`, and `accelerate` launchers; scripts other than the `dlc` and `deepspeed` ones must be started on every node [6].
- Megatron-SWIFT is a separate launch path, `swift megatron sft` / `swift megatron rlhf`, configured through Megatron's own parallelism flags (tensor/pipeline/sequence/context/expert parallel sizes) rather than DeepSpeed's [7].
- GRPO's generation layout is chosen at start time: `--use_vllm true --vllm_mode colocate` shares the training GPUs with generation, with a five-item memory list for tightening it (lower `vllm_gpu_memory_utilization`; `--sleep_level 1`; `--offload_optimizer true --offload_model true`; `--vllm_tensor_parallel_size`; `--move_model_batches`; `--offload_bridge true`); `--use_vllm true --vllm_mode server` instead points training at a separately launched `swift rollout` process via `--vllm_server_host`/`--vllm_server_port`/`--vllm_server_timeout` [8]. The docs note that data-parallel-only server mode combined with `--vllm_use_async_engine` can error, and link the tracking issue [8].
- Effective batch size for GRPO is worked through in the docs with a concrete example: 8 processes, `per_device_train_batch_size=4`, `gradient_accumulation_steps=8` gives a `generation_batch_size` of 512 and, with `num_generations=64`, 8 prompts are sampled to 512 total responses per update, with an update batch size of 256 [8].
- Config surface is CLI flags rather than a config-class object; `--load_args false` disables ms-swift's default behaviour of auto-reading a resumed adapter directory's saved `args.json` for `--model`/`--system` and other training args [3]. ms-swift changes the underlying Trainer default and enables `--gradient_checkpointing true` by default to save memory, at the cost of slightly slower training [6].
- Out-of-memory first aid documented in the FAQ [18]: for LoRA/full training, reduce `--per_device_train_batch_size` and raise `--gradient_accumulation_steps` to hold the effective batch size, and consider packing (`--packing true`, which must be paired with `--attn_impl flash_attn`) to cut padding waste. For GRPO specifically, lowering `vllm_gpu_memory_utilization` and enabling `--sleep_level 1`/offload flags addresses generation-side memory as noted above [8].

## Watch it

Mechanics only - what a metric shape means for a given method lives on that method's card.

- Logging is enabled with `--report_to`, default `'tensorboard'`; multiple loggers can be listed (`--report_to tensorboard wandb swanlab`, or `--report_to all`) [19]. Cadence is `--logging_steps`, default 5 [19].
- GRPO's live logged-metrics list (from the GRPO trainer page [8]) includes the `completions/*` family (mean/min/max/clipped length), per-reward-function `reward/{reward_func_name}/mean|std`, combined `reward`/`reward_std`, `kl` (only when a KL penalty is active), `clip_ratio/*`, and an entropy family that is only recorded when `--log_entropy true` is set - entropy curves are not logged by default [8][18].
- Sample-level logging is gated by `--log_completions` (default `False`): when set, generated content is logged alongside `--report_to wandb`/`swanlab` as a table of training-dynamics data (step, prompt, completion, per-reward-function score, entropy), with an `image` key auto-detected for wandb-only display and a `metrics_to_gather` dict for adding extra logged columns [8]; if `--report_to wandb`/`swanlab` is not set, enabling `--log_completions` instead writes a `completions.jsonl` file into the checkpoint directory [19].
- Evaluation during training uses `--eval_strategy` (default follows `--save_strategy`) and `--eval_steps` (default follows `--save_steps` when an eval dataset is set) [19].
- A GRPO-specific health note from the docs: in on-policy training the new and old policies are identical, so the importance-sampling ratio stays at 1 and the clip operation never takes effect - a `clip_ratio` sitting at zero is therefore expected there; the algorithm becomes off-policy once `num_iterations > 1` or `gradient_accumulation_steps % steps_per_generation != 0` [8].
- No published RL-specific stopping rule or threshold was found. The Command-line-parameters page documents `early_stop_interval`: training stops when `best_metric` shows no improvement within that many `save_steps`-based periods, and setting it auto-adds an early-stop callback, but no default numeric value is stated in that entry [20]. The FAQ and GRPO pages were also searched (2026-08-10 reading) and add no threshold beyond this [8][18].

## Save it

- Checkpoints land under `--output_dir`, in a versioned subfolder by default (`add_version`, default `True`, appends `<version>-<timestamp>` to prevent overwriting) [20], as numbered `checkpoint-<step>/` directories referenced directly by `--adapters`/`--model` in the quick-start inference step [3].
- Retention and cadence: `--save_strategy` (default `'steps'`), `--save_steps` (default 500), `--save_total_limit` (default `None`, keeps every checkpoint; set to e.g. 2 to keep only the best and the last) [20].
- `--save_only_model` (default `False`) saves only model weights, excluding optimizer and random-seed state, to reduce full-parameter-training checkpoint overhead [20].
- Resume: `--resume_from_checkpoint <path>` resumes model, optimizer, and scheduler state; `--resume_only_model` (default `False`) resumes weights only, ignoring optimizer state and random seed, when combined with `--resume_from_checkpoint`; `--ignore_data_skip` (default `False`) starts from step 0 without replaying or skipping already-seen data [20].
- For RLHF methods continuing from a LoRA SFT checkpoint, the pattern is `--adapters sft_ckpt --ref_adapters sft_ckpt`; resuming that RLHF run itself uses `--resume_from_checkpoint rlhf_ckpt --ref_adapters sft_ckpt` [20].
- LoRA saves an adapter, not a merged model - `--merge_lora true` (supported for LoRA, LlamaPro, and LongLoRA) produces the full merged model as a separate artifact [20]. ms-swift's native LoRA checkpoint format is not directly PEFT-loadable as-is: it nests weights under a `default` subfolder and does not use PEFT's plain key names; `Swift.save_to_peft_format(ckpt_dir, output_dir)` converts a saved checkpoint into PEFT-compatible form by moving the `default` folder's contents to the output root, stripping the `{tuner_name}.` segment from weight keys, and adding a `basemodel.model` prefix - and the docs note only LoRA can be converted this way [21].
- A closed-issue trap: a user who ran `swift export --adapters <ckpt> --merge_lora true` on a Qwen2.5-VL-7B LoRA checkpoint and then tried to load the merged output with `Qwen2_5_VLForConditionalGeneration.from_pretrained` hit a load error; ms-swift maintainer Jintao-Huang (COLLABORATOR) replied by pointing to the upstream `transformers` issue tracking this exact incompatibility, without further elaboration - so a merged Qwen2.5-VL checkpoint from `swift export` may still need a matching transformers version or patch from that upstream thread before it loads [22].
- Whether an evaluator can load the saved checkpoint directly depends on which artifact you point it at: a merged model or a full-parameter checkpoint is a standard `from_pretrained`-loadable directory, while a raw LoRA `checkpoint-N/` directory needs either ms-swift's own loading path (`--adapters`) or the `save_to_peft_format` conversion above before a plain PEFT/transformers loader can read it [3][21].

## Find it in the docs

The docs are the live source; this section teaches the lookup.

- Address pattern: `https://swift.readthedocs.io/en/<version>/<Section>/<Page>.html`. `<version>` is a branch-level slug, not an exact patch tag: fetching `en/v4.4.2/...` 404s, while `en/v4.4/...` returns 200 (verified 2026-08-10 against both the docs root and `GetStarted/Quick-start.html`) [23][24]. The ReadTheDocs versions API confirms the active slugs are branch-form (`v4.4`, `v4.3`, ... `v3.5`), each tracking a minor-version branch rather than a fixed patch release [25].
- Top-level sections, from the docs index [24]: `GetStarted/` (Quick-start, Web-UI, Notebook environment), `Instruction/` (Command-line-parameters, Pre-training-and-Fine-tuning, GRPO, RLHF, Inference-and-deployment, Sampling, Evaluation, Export-and-push, FAQ), `Customization/` (custom model/dataset/architecture), `Megatron-SWIFT/` (Quick-start, Command-line-arguments, LoRA-training, Multimodal, Mcore-Bridge, Megatron-GRPO), `BestPractices/` (per-model-family guides).
- Question-to-page map: CLI flag lookup -> `Instruction/Command-line-parameters.html`; RLHF method flags and `--rlhf_type` values -> `Instruction/RLHF.html`; GRPO cluster modes, memory knobs, logged metrics -> `Instruction/GRPO.html`; checkpoint merging and quantized export -> `Instruction/Export-and-push.html`; general troubleshooting (batch-size math, packing, sequence parallel, tuner restrictions, GRPO gotchas) -> `Instruction/Frequently-asked-questions.html`; Megatron parallelism sizing and benchmarks -> `Megatron-SWIFT/Quick-start.html`.
- Runnable references beyond the docs: the quick-start and RLHF examples use datasets already published on ModelScope/Hugging Face (`AI-ModelScope/alpaca-gpt4-data-zh`, `swift/self-cognition`) as smoke-test data [3].
- Community layer: no curated third-party tutorial page comparable to trl's `community_tutorials` was found on the docs index (checked 2026-08-10, no tutorial/blog/awesome links in the index page's navigation) [24]. The closest first-party analog is the maintainer-authored `BestPractices/` section itself (Qwen3, Qwen3-VL, Qwen3.5, DeepSeek-V4 training, embedding/reranker training guides) - official walkthroughs rather than curated external blogs [24].
- No official MCP endpoint was found: the docs index page was searched for "mcp" (case-insensitive, 2026-08-10) with no match [24], unlike trl's documented `huggingface.co/mcp` server.
- Honest boundary: install pins cap `peft<0.20`, `datasets<4.8.5`, and `trl<1.0` at the v4.4.2 release [13] - a project already holding newer pins of these libraries will collide on install. The GRPO trainer's own docs state channel loss is not supported, and that GRPO combined with the Liger fused loss and padding-free training is not a supported combination [18].

## Sources

Method names (SFT, DPO, KTO, GRPO, PPO, RM, GKD, ...) are deliberately cited to nothing here; their defining papers live on the methodology cards. All docs pages are `main`-branch or version-branch live pages read on the stated fetch date unless a commit is named for source code.

[1] ms-swift README, Introduction and News sections. https://github.com/modelscope/ms-swift/blob/main/README.md. Fetched 2026-08-10.

[2] ms-swift GitHub repository API metadata (stars, pushed_at, description, default branch). https://api.github.com/repos/modelscope/ms-swift. Fetched 2026-08-10.

[3] ms-swift Quick-start page (LoRA SFT command, inference/merge command, args.json auto-load note). https://swift.readthedocs.io/en/main/GetStarted/Quick-start.html. Fetched 2026-08-10.

[4] ms-swift Quick-start page, "Why Choose ms-swift?" feature list (distributed training, multimodal training mention). https://swift.readthedocs.io/en/main/GetStarted/Quick-start.html. Fetched 2026-08-10.

[5] ms-swift RLHF documentation page (`swift rlhf --rlhf_type` usage). https://swift.readthedocs.io/en/main/Instruction/RLHF.html. Fetched 2026-08-10.

[6] ms-swift Pre-training-and-Fine-tuning page (device-map/DeepSpeed/FSDP+QLoRA scaling options, multi-node launcher scripts, gradient-checkpointing default). https://swift.readthedocs.io/en/main/Instruction/Pre-training-and-Fine-tuning.html. Fetched 2026-08-10.

[7] ms-swift Megatron-SWIFT Quick-start page (supported methods and models, benchmark table for dense Qwen2.5-14B and MoE Qwen3-30B-A3B against DeepSpeed ZeRO-2/ZeRO-3). https://swift.readthedocs.io/en/main/Megatron-SWIFT/Quick-start.html. Fetched 2026-08-10.

[8] ms-swift GRPO documentation page (colocate/server cluster modes, memory-optimization flags, logged-metrics list, completions logging, effective-batch-size worked example, eval fields, entropy logging default, clip-ratio-zero explanation, unsupported channel-loss/Liger+padding-free combination). https://swift.readthedocs.io/en/main/Instruction/GRPO.html. Fetched 2026-08-10.

[9] ms-swift on PyPI (version 4.4.2, upload date, licence). https://pypi.org/pypi/ms-swift/json. Fetched 2026-08-10.

[10] ms-swift setup.py at the v4.4.2 tag (python_requires, licence, author/contact, entry points). https://raw.githubusercontent.com/modelscope/ms-swift/v4.4.2/setup.py. Fetched 2026-08-10.

[11] ms-swift Installation page, Running Environment and Supported Hardware tables. https://swift.readthedocs.io/en/main/GetStarted/Installation.html. Fetched 2026-08-10.

[12] ms-swift git tag v4.4.2 resolved to its commit. https://api.github.com/repos/modelscope/ms-swift/git/refs/tags/v4.4.2. Fetched 2026-08-10.

[13] ms-swift requirements/framework.txt at the v4.4.2 tag. https://raw.githubusercontent.com/modelscope/ms-swift/v4.4.2/requirements/framework.txt. Fetched 2026-08-10.

[14] ms-swift requirements/megatron.txt at the v4.4.2 tag. https://raw.githubusercontent.com/modelscope/ms-swift/v4.4.2/requirements/megatron.txt. Fetched 2026-08-10.

[15] ms-swift requirements/eval.txt and requirements/ray.txt at the v4.4.2 tag. https://raw.githubusercontent.com/modelscope/ms-swift/v4.4.2/requirements/eval.txt and .../requirements/ray.txt. Fetched 2026-08-10.

[16] ms-swift GitHub commit for the resolved v4.4.2 tag, showing its committer date (2026-07-21), compared against the repository's newest push commit on `main` (2026-08-09). https://api.github.com/repos/modelscope/ms-swift/commits/f48847d23dbcd72ceb15fdbc5a1482cc7eb0359d and https://api.github.com/repos/modelscope/ms-swift/commits/main. Fetched 2026-08-10.

[17] ms-swift GitHub releases API (release dates for v4.3.1 through v4.4.2). https://api.github.com/repos/modelscope/ms-swift/releases. Fetched 2026-08-10.

[18] ms-swift Frequently-asked-questions page (packing requires flash_attn, GRPO FAQ bullets: entropy logging default, channel loss unsupported, Liger+padding-free unsupported). https://swift.readthedocs.io/en/main/Instruction/Frequently-asked-questions.html. Fetched 2026-08-10.

[19] ms-swift Command-line-parameters page, logging fields (`report_to` default, `logging_steps` default, `eval_strategy`/`eval_steps` defaults, `log_completions` default and its `completions.jsonl` fallback). https://swift.readthedocs.io/en/main/Instruction/Command-line-parameters.html. Fetched 2026-08-10.

[20] ms-swift Command-line-parameters page, save/resume fields (`add_version`, `save_strategy`, `save_steps`, `save_total_limit`, `save_only_model`, `resume_from_checkpoint`, `resume_only_model`, `ignore_data_skip`, `ref_adapters`, `merge_lora`, `early_stop_interval`). https://swift.readthedocs.io/en/main/Instruction/Command-line-parameters.html. Fetched 2026-08-10.

[21] ms-swift Using-Tuners page (`Swift.save_to_peft_format` conversion contract: default-subfolder split, weight-key renaming, LoRA-only). https://swift.readthedocs.io/en/main/Instruction/Use-tuners.html. Fetched 2026-08-10.

[22] ms-swift GitHub issue #5440 (closed), reply from maintainer Jintao-Huang (COLLABORATOR), 2025-08-20, linking the upstream transformers issue for a merged Qwen2.5-VL LoRA checkpoint load failure. https://github.com/modelscope/ms-swift/issues/5440. Fetched 2026-08-10.

[23] ms-swift docs root at the v4.4 branch-slug and a page under it, used to verify the working version-tag URL form against the 404ing exact-patch-tag form. https://swift.readthedocs.io/en/v4.4/ and https://swift.readthedocs.io/en/v4.4/GetStarted/Quick-start.html. Fetched 2026-08-10.

[24] ms-swift documentation index page (full section/page slug map; searched for tutorial/blog/awesome and mcp mentions, none found). https://swift.readthedocs.io/en/main/index.html. Fetched 2026-08-10.

[25] ReadTheDocs versions API for the swift project (active version slugs are branch-form, e.g. v4.4, not exact patch tags). https://readthedocs.org/api/v3/projects/swift/versions/?active=true. Fetched 2026-08-10.
