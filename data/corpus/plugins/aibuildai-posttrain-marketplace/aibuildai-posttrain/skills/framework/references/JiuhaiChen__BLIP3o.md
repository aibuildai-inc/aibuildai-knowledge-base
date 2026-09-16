# JiuhaiChen/BLIP3o

https://github.com/JiuhaiChen/BLIP3o

Not a post-training library: this is the official code release for the BLIP3o-NEXT image-generation model, and its GRPO step runs on a hard-forked, single-purpose copy of Hugging Face's trl vendored inside the repo, not on a reusable trainer API.

The repository is described by its own GitHub metadata as the "Official implementation of BLIP3o-Series" [1], an autoregressive-plus-diffusion image-generation model built by Jiuhai Chen and collaborators, whose README states it "generates intermediate features via the autoregressive model and then conditions on these features to generate images through the diffusion model" [2]. For its reinforcement-learning stage, the README states plainly: "Using Group Relative Policy Optimization (GRPO), we train the BLIP3o-NEXT to improve prompt alignment and text rendering in image generation" [2], and the shape of that stage is not a configurable library call - it is a copy of trl's source tree at `trl/trl/` [3] with `trl/trl/trainer/grpo_trainer.py` edited in place to hardcode a PaddleOCR-based text-rendering reward, imported and instantiated at module load time [4]. There is no package published to PyPI and no version tag; the repo is pinned here at commit d0b0a1c46e15db22f2b460cee930a81f16afeef7 on the `BLIP3o-NEXT` default branch [1][5].

**When to pick it**: pick this repo only to reproduce or build directly on BLIP3o-NEXT's own pretraining, instruction-tuning, or GRPO text-rendering runs, using its exact scripts and reward. It is not a candidate on the same axis as trl, verl, or other post-training libraries: its GRPO path is a single fixed reward (OCR edit-distance against a quoted string in the prompt) burned into `_calculate_rewards`, not a function you pass in at the trl-documented `reward_funcs` call site - the standalone `train_grpo.py` script does pass `reward_funcs=reward_len` to `GRPOTrainer`, but that argument is not what actually scores completions on the OCR path, since `_calculate_rewards` in the same vendored file ignores it and calls the module-level `ocr_model.predict` directly [4][6]. Anyone wanting a general-purpose, configurable GRPO trainer should use upstream trl directly (cross-reference; not covered here) rather than this fork.

**Methods it ships**: the vendored `trl/trl/trainer/` directory carries a full copy of trl 0.21.0.dev0's trainer files - DDPO, DPO, KTO, ORPO, Reward, GRPO, and PPO among them, each importable the same way as in upstream trl [3][7] - but the README and the only training entry point that BLIP3o-NEXT documents and wires to a script (`train_grpo.py`, launched by `run.sh`) exercise GRPO alone, for the text-rendering RL stage [8][9]. The other vendored trainer files were not observed to be called from any script in the repository's root, `blip3o/`, or `trl/` top level read for this card, and no README documents using them; treat them as unused vendored code, not as methods this repo ships. GRPO's own math and semantics are not restated here; see the GRPO methodology card and, for the un-forked trainer, the trl library card.

**Scale it handles**: the shipped example targets one Slurm node with 8 GPUs, launched through Accelerate with a DeepSpeed ZeRO-1 config template at `trl/examples/accelerate_configs/deepspeed_zero1.yaml`, invoked as `accelerate launch --config_file examples/accelerate_configs/deepspeed_zero1.yaml --num_machines 1 --num_processes 8 ... train_grpo.py` inside `srun` [9]. The script sets `--nodes=1` and comments the multi-node path only through `--main_process_ip`/`--machine_rank` populated from `$SLURM_JOB_NODELIST`, so a second node is mechanically reachable by raising `--nodes`, but no multi-node run is documented or benchmarked in the README or the `trl/` subdirectory read for this card [9].

**Install**: no PyPI package and no version tag exist for this repo; it is installed from source at the commit above. Two disjoint environments are required, per the README's own version-conflict warning [2]. Pretraining/instruction-tuning: `conda create -n blip3o-next python=3.11`, then `pip install -r requirements.txt` and `pip install -e .` from the repo root, where `requirements.txt` pins `torch==2.3.0`, `transformers==4.51.3`, `accelerate==0.28.0`, `deepspeed==0.14.4`, `diffusers==0.34.0`, `flash_attn==2.6.2` [10][2]. GRPO: a separate `conda create -n grpo python=3.11`, then `cd trl && pip install -r requirements.txt && cd .. && pip install -e .`, where `trl/requirements.txt` pins `torch==2.7.1`, `transformers==4.54.1`, `accelerate>=1.4.0`, `liger-kernel==0.6.1`, `paddleocr==3.1.0`, `paddlepaddle==3.0`, `diffusers==0.34.0` [11][2]. No CUDA or GPU-count floor is stated beyond the 8-GPU example in `run.sh` [9]. No LICENSE, LICENSE.md, or LICENSE.txt file was found in the repository at this commit (each returned HTTP 404), and the GitHub API reports the repository's licence field as null [1]; the vendored `trl/trl/trainer/grpo_trainer.py` itself retains its original Apache-2.0 header from the Hugging Face Team [4], but that header covers only that file, not BLIP3o-NEXT's own code.

**Maintained by**: Jiuhai Chen and collaborators, under the `JiuhaiChen` GitHub account [1]; the repository was pushed most recently on 2025-11-29 [1], and the README links an arXiv paper (arXiv:2510.15857) as the project's write-up [2].

## Quick start

The repo documents no minimal end-to-end snippet outside its scripts; the smallest complete run is the GRPO script it ships, `trl/train_grpo.py`, which builds a prompt-only dataset from a local text file and calls the vendored `GRPOTrainer` directly [6]:

```python
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

with open("ocr_train.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

train_dataset = Dataset.from_dict({"prompt": prompts})
train_dataset.shuffle(seed=0)

training_args = GRPOConfig(
    output_dir="BLIP3o-NEXT-Text-GRPO", use_liger_loss=True,
    per_device_train_batch_size=16, num_generations=16, save_steps=50,
    lr_scheduler_type="cosine", learning_rate=1e-6, beta=0.001,
)

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

trainer = GRPOTrainer(
    model="/fsx/home/jiuhai.chen/BLIP3o-NEXT/models/debug",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

The script itself labels `reward_len` a "dummy reward for testing" in a source comment [6]; the reward that actually scores images on the documented text-rendering run is the hardcoded OCR path inside `grpo_trainer.py`, described under "Watch it" below. The model path, `/fsx/home/jiuhai.chen/BLIP3o-NEXT/models/debug`, is the author's own cluster path and is not a published checkpoint id [6]. There is no separate CLI form; the only launch path is `bash run.sh` or a direct `accelerate launch ... train_grpo.py` [8][9].

## Start it

- One process, one GPU: run `train_grpo.py` directly with `python`, or through `accelerate launch` with a single-process config; the shipped example does not demonstrate this, only the 8-GPU Slurm form [9].
- Multiple GPUs, one node: `run.sh` calls `accelerate launch --config_file examples/accelerate_configs/deepspeed_zero1.yaml --num_processes 8 train_grpo.py` under `srun`, requesting `--gres=gpu:8` in its `#SBATCH` header [9].
- Multiple nodes: `run.sh` derives `NODELIST` from `$SLURM_JOB_NODELIST` and passes `--main_process_ip ${NODELIST[0]} --machine_rank $SLURM_PROCID`, but the script as shipped sets `--nodes=1` and `--num_machines 1`, so a working multi-node launch requires editing those values; no multi-node config or run is documented [9].
- Effective batch size in the shown config is `per_device_train_batch_size=16` times device count (8 in the shipped script) times `num_generations=16` completions sampled per prompt; `use_liger_loss=True` selects trl's fused Liger GRPO loss kernel [6].
- `GRPOConfig` here is trl's own Config class, vendored unchanged as far as this card's reading of `grpo_trainer.py` went beyond the reward path; BLIP3o-NEXT does not document any config default it changes from upstream trl, only the values it passes at call time (`beta=0.001`, `learning_rate=1e-6`, `save_steps=50`) [4][6].
- Out-of-memory first aid: none is documented in the README or `trl/README.md` for this repo specifically; the generic trl knobs (lower `per_device_train_batch_size`, `num_generations`, or `vllm_gpu_memory_utilization`) apply only insofar as this vendored trainer preserves trl's upstream memory-relevant code paths, which this card's reading did not verify line-by-line [2][8].

## Watch it

- The training script prints the full `GRPOConfig` object at start (`print(training_args)`) but sets no `report_to` value in the shown config, so no external tracker (Weights & Biases, TensorBoard) is wired up in the quick-start script itself [6]; `run.sh` exports `WANDB_API_KEY` as a placeholder, implying the author's own runs used Weights & Biases, but the script that would set `report_to="wandb"` was not found in the files read for this card [9].
- The reward the documented GRPO run actually optimizes is computed in `_calculate_rewards` inside the vendored `grpo_trainer.py`: generated images are converted to BGR arrays, passed to a module-level `PaddleOCR` instance's `.predict()`, and the recognized text is compared against a quoted substring of the prompt by Levenshtein edit distance, with `reward = 1 - dist / len(prompt)` and a full penalty (`dist = len(prompt)`) on any OCR failure or when the distance exceeds the prompt length [4]. This reward is specific to the text-rendering task the README describes and is not a general image-quality or prompt-alignment score [2][4].
- The same function saves the single highest-reward image in the batch to `highest_reward_image_final.png` in the working directory on every call, as a debugging artifact rather than a configurable sample-logging feature [4].
- Beyond the metric names inherited unmodified from upstream trl's `GRPOTrainer` (this card did not diff the full 2144-line vendored file line-by-line against a specific upstream trl release, so it does not claim which of trl's documented GRPO metrics still fire unchanged), no BLIP3o-specific metric name, evaluation-during-training field, or published stopping-rule or health-limit was found in the README, `trl/README.md`, or `run.sh` read for this card [2][8][9].

## Save it

- `GRPOConfig(output_dir="BLIP3o-NEXT-Text-GRPO", ..., save_steps=50, ...)` is the only save configuration shown; per this value, a checkpoint is written every 50 steps, following trl's own `checkpoint-<step>` directory convention, which this vendored file's reading did not find altered [6].
- No BLIP3o-specific resume call, checkpoint-directory listing, or adapter-saving path is documented in the README or `trl/README.md`; the repo does not show a `resume_from_checkpoint` call for `train_grpo.py`, and none was found in the files read for this card [2][6][8].
- `trl/README.md`'s own guidance for changing the reward function directs the reader to edit source, not to pass a new argument: it tells readers to "modify [OCR reward]" at a specific line link into `grpo_trainer.py` [8] - underscoring that this fork's save/reload/resume behavior, wherever it diverges from upstream trl, is discoverable only by reading the vendored file directly, not from any doc page.
- Whether a downstream evaluator can load a checkpoint produced here depends on whatever base-model class BLIP3o-NEXT trains (an AR+diffusion architecture, per the README [2]); this card does not verify that loader contract, since no loading script was read for this card.

## Find it in the docs

There are no versioned docs pages for this repository; everything a user needs sits in the repo itself.

- Repository home: https://github.com/JiuhaiChen/BLIP3o, default branch `BLIP3o-NEXT` [1].
- Top-level `README.md` covers the pretraining/instruction-tuning install and the `inference.py` entry point [2].
- `trl/README.md` covers the GRPO-specific environment and the `bash run.sh` launch, and is the only place that names the OCR reward and points to its exact edit location in `grpo_trainer.py` by line link [8].
- The paper is at arXiv:2510.15857, linked from the README as "Arxiv" [2]; this card did not fetch the paper itself and cites it only as a pointer the README gives.
- Community layer: the README links a Discord server and a WeChat QR code for questions, with no separate tutorials or blog posts curated by the authors found in the README read for this card [2].
- No MCP endpoint, hosted docs site, or API reference distinct from the two README files above was found for this repository.
- Honest boundary: this is a single research group's release for one model family, not a general post-training library - it lacks a stable public API, a package release, a license file, and any documented multi-node or non-GRPO training path beyond what its own scripts show, and reusing its GRPO trainer for a different reward requires editing the vendored `grpo_trainer.py` source directly [2][4].

## Sources

Method names (GRPO, DPO, PPO, KTO, ORPO) are deliberately cited to nothing here; their defining papers live on the methodology cards, and trl's own generic behavior is covered by the trl library card, not restated here. All GitHub pages and raw files below were fetched 2026-08-12 at commit d0b0a1c46e15db22f2b460cee930a81f16afeef7 unless a different path is named.

[1] JiuhaiChen/BLIP3o repository metadata (GitHub REST API). https://api.github.com/repos/JiuhaiChen/BLIP3o. Fetched 2026-08-12.

[2] JiuhaiChen/BLIP3o root README. https://raw.githubusercontent.com/JiuhaiChen/BLIP3o/d0b0a1c46e15db22f2b460cee930a81f16afeef7/README.md. Fetched 2026-08-12.

[3] Directory listing of the vendored `trl/trl/` package. https://api.github.com/repos/JiuhaiChen/BLIP3o/contents/trl/trl?ref=d0b0a1c46e15db22f2b460cee930a81f16afeef7. Fetched 2026-08-12.

[4] Vendored, modified `grpo_trainer.py` (PaddleOCR reward hardcoded in `_calculate_rewards`; Apache-2.0 header retained). https://raw.githubusercontent.com/JiuhaiChen/BLIP3o/d0b0a1c46e15db22f2b460cee930a81f16afeef7/trl/trl/trainer/grpo_trainer.py. Fetched 2026-08-12.

[5] Repository root directory listing. https://api.github.com/repos/JiuhaiChen/BLIP3o/contents/?ref=d0b0a1c46e15db22f2b460cee930a81f16afeef7. Fetched 2026-08-12.

[6] `trl/train_grpo.py`, the shipped GRPO training script. https://raw.githubusercontent.com/JiuhaiChen/BLIP3o/d0b0a1c46e15db22f2b460cee930a81f16afeef7/trl/train_grpo.py. Fetched 2026-08-12.

[7] Vendored trl `__init__.py`, giving the pinned trl version (0.21.0.dev0). https://raw.githubusercontent.com/JiuhaiChen/BLIP3o/d0b0a1c46e15db22f2b460cee930a81f16afeef7/trl/trl/__init__.py. Fetched 2026-08-12.

[8] `trl/README.md`, the GRPO-specific instructions and the OCR-reward edit pointer. https://raw.githubusercontent.com/JiuhaiChen/BLIP3o/d0b0a1c46e15db22f2b460cee930a81f16afeef7/trl/README.md. Fetched 2026-08-12.

[9] `trl/run.sh`, the Slurm/Accelerate launch script. https://raw.githubusercontent.com/JiuhaiChen/BLIP3o/d0b0a1c46e15db22f2b460cee930a81f16afeef7/trl/run.sh. Fetched 2026-08-12.

[10] Root `requirements.txt` (pretraining/instruction-tuning environment pins). https://raw.githubusercontent.com/JiuhaiChen/BLIP3o/d0b0a1c46e15db22f2b460cee930a81f16afeef7/requirements.txt. Fetched 2026-08-12.

[11] `trl/requirements.txt` (GRPO environment pins). https://raw.githubusercontent.com/JiuhaiChen/BLIP3o/d0b0a1c46e15db22f2b460cee930a81f16afeef7/trl/requirements.txt. Fetched 2026-08-12.
