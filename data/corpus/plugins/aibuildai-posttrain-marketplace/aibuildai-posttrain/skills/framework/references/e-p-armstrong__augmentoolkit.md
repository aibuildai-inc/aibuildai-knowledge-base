# augmentoolkit

A dataset-generation tool that turns documents into training data, and bundles one experimental GRPO trainer plus a hand-off to Axolotl for the SFT/pretraining data it produces - not a general-purpose post-training library.

**augmentoolkit** describes itself as a project that "creates domain-expert datasets that update an AI's brain (basically, its knowledge cutoff), so that the AI becomes an expert in an area of your choosing" [1], and its own technical definition names the output as "Continued Pretraining and Supervised Fine Tuning datasets for teaching LLMs factual information" [2]. It is built and maintained by Evan Armstrong (GitHub handle e-p-armstrong) at https://github.com/e-p-armstrong/augmentoolkit [3][4]. Its API is a set of Python pipeline functions driven by YAML configs, run either through a `run_augmentoolkit.py` CLI with a `super_config.yaml` pipeline order, or through a bundled React/FastAPI web interface, each pipeline producing dataset files and, for two specific pipelines, either a trained LoRA adapter or ready-to-run Axolotl training configs [5][6][2].

**When to pick it**: pick augmentoolkit when the task is turning your own documents into a continued-pretraining-plus-SFT dataset for teaching an LLM a new factual domain, and you want the pipeline to also hand you ready-to-run Axolotl configs for that data [7]; it is not a general RL or SFT trainer in the shape of trl or verl (cross-reference; not covered here) - its only bundled training loop is one experimental GRPO pipeline for prompt-defined reward alignment, and even that pipeline delegates the actual `GRPOTrainer` step to Hugging Face's trl [8][9]. Do not pick it if you need a stable, actively-maintained trainer: the docs mark the GRPO pipeline itself as experimental, "in beta," with "no good examples yet" [9].

**Methods it ships**: one training pipeline, GRPO, at `generation/core_pipelines/do_grpo_rl_with_a_prompt/` [10]. Its `GRPOTrainerStopSequences` class subclasses trl's `GRPOTrainer` directly - `class GRPOTrainerStopSequences(GRPOTrainer):` - to add custom stop sequences and rebuild vLLM's `SamplingParams`, and the pipeline also imports Unsloth's `FastLanguageModel`, `PatchFastRL`, and `is_bfloat16_supported` [8] (commit `ec18b905f8a2fc1d64d345c04856cf2f036d9fc6`, the repository's newest push at screening time [3]). The docs title this page "GRPO (Experimental)" and expand the acronym as "Generative Reward Powered Optimization," a phrasing that differs from the standard "Group Relative Policy Optimization" name used on this deck's method cards; the README separately calls the same feature "GPRO" once, a typo [9][11]. There is no other method-card-covered trainer bundled in the repository: the flagship Complete Factual Datagen pipeline instead generates two Axolotl YAML configs (`pretraining_run/axolotl_pretraining_config.yaml`, `sft_run/sft_training_config.yaml`) that the user copies to a GPU machine and runs with `accelerate launch -m axolotl.cli.train <config>.yaml` through the separate Axolotl project, not augmentoolkit itself [7][12].

**Scale it handles**: the GRPO pipeline is single-GPU only in its own docs and config surface - the config's `model` section sets `gpu_memory_utilization` for one card's worth of colocated vLLM generation, and nothing in `docs/grpo.md` or the config schema names a multi-GPU or multi-node launch path for it [9][13]. The GRPO docs' own worked example targets one rented Linux GPU instance [9]. For dataset-generation concurrency (not training), the local-inference script `local_linux.sh` accepts a `--tensor-parallelism N` flag for splitting the vLLM datagen server across N GPUs [5]. Multi-node training is out of scope for this repository - the only training compute path documented is a single GPU for GRPO, or an external Axolotl run whose own scale is not documented here [9][7].

**Install**: `git clone https://github.com/e-p-armstrong/augmentoolkit.git`, then `pip install uv` and `uv pip install -r requirements.txt` [5]; there is no `pyproject.toml` and no PyPI package - confirmed by an empty fetch of the repository root's `pyproject.toml` at commit ec18b905 - so `git clone` plus `requirements.txt` is the only install path [3]. The README instructs "Be sure to use Python 3.11 when creating the virtual environment" for the start-script paths [11], while `docs/quickstart.md`'s separate Manual Interface Setup section lists "Python 3.9+" as the prerequisite for the fully-manual path [5] - the two docs pages state different floors for different setup paths and are cited separately here. Licence is MIT, per the repository's root `LICENSE` file ("Copyright (c) 2024 Evan Armstrong") [14]. The base `requirements.txt` (58 lines, read in full at commit ec18b905) pins only `datasets==2.15.0` and `protobuf==3.20.0`; `torch`, `transformers`, `peft`, and `wandb` are unpinned, and `trl`, `vllm`, and `unsloth` are commented out entirely - none of the three packages the GRPO pipeline needs is installed by default [15]. The GRPO pipeline's own `generation/core_pipelines/do_grpo_rl_with_a_prompt/requirements.txt` (4 lines) adds `trl`, `torch`, `unsloth`, `vllm`, each with no version specifier at all [16] - a reader who already holds a pinned torch/CUDA/vLLM stack gets no guardrail against a collision. No CUDA or hardware-minimum version is stated on the installation or quickstart pages; the GRPO docs instead state the pipeline is "Linux only" because it needs vLLM, and requires "a machine with a capable GPU (suitable for LoRA training)" without naming a specific GPU or CUDA version [9][5]. There is no tagged release current with the screening commit: the latest GitHub Release is `v3.0.0`, published 2025-06-12, at commit `9d584026027610f2ec093d9f3f7c403aec57f2b5` [17][18] - over a year behind the screening commit `ec18b905f8a2fc1d64d345c04856cf2f036d9fc6` from the repository's newest push on 2026-06-27 [3]; the v3.0.0 release notes themselves describe a smaller, three-pipeline tool ("Before we had 3 pipelines. Now we have 16") [17], so a reader relying on that tag would be missing most of the current pipeline set, including whichever changes landed on `master` after v3.0.0.

**Maintained by**: Evan Armstrong (GitHub handle e-p-armstrong), the sole maintainer named in the repository owner and licence copyright fields [3][14]. Signs of life: the repository's newest push is dated 2026-06-27, the same date as the shortlist row's screening commit [3]; the last tagged release, v3.0.0, was published 2025-06-12, with release notes describing an expansion from 3 to 16 pipelines and the addition of automatic training-config generation for whole-LLM training [17]. Axolotl-config generation specifically was announced earlier, in the v1.5.0 release notes (published 2024-07-09), which list "Axolotl training configs provided as part of the repo" [17]. There is no dedicated changelog or announcements page among the docs read for this card beyond the GitHub Releases list itself [17].

## Quick start

The README's own three-line interface quickstart, one command per OS, is the smallest complete run [5][11]:

```bash
# Linux (interface)
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash linux.sh
```

```bash
# MacOS (interface)
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash macos.sh
```

The CLI form, identical across OSes apart from venv activation syntax, is [5]:

```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -r requirements.txt
# add your API key to ./external_configs/complete_factual_datagen_example.yaml
python run_augmentoolkit.py
```

For local (no-API) dataset generation on Linux, `bash local_linux.sh normal` runs the project's own FP16 7B data-generation model via vLLM, or `bash local_linux.sh small` runs a quantized version for lower-VRAM cards [5].

## Start it

- Interface mode is the recommended path: one start script (`macos.sh`, `linux.sh`, or WSL plus the Linux script on Windows) launches the API server, task worker, and frontend together [5][11].
- CLI mode runs `python run_augmentoolkit.py` against `super_config.yaml`, which lists a `pipeline_order` of pipeline name plus config file [19]. To launch the GRPO pipeline specifically, that file's `pipeline_order` entry points at a GRPO config, e.g. `node: grpo-rl-pipeline`, `config: grpo-rl-folder:your_grpo_config.yaml` [9].
- Manual Interface Setup runs three terminals by hand: Terminal 1 activates the venv and starts the task queue with `huey_consumer tasks.huey`; Terminal 2 runs `uvicorn api:app --host 0.0.0.0 --port 8000`; Terminal 3 builds and serves the frontend from `atk-interface/` with `npm install && npm run build && npx serve -s dist --listen 5173`, after which the interface is reached at `http://localhost:5173` [5]. This path additionally requires Redis or Valkey running as Huey's message broker; `linux.sh` will build Valkey from source if no server is found and running [5].
- GRPO's config surface is a dedicated, non-standard config shape (it "lacks standard sections like `api` or `path`" that other pipelines use) [9], with `training_parameters` mapped onto "the Unsloth/TRL `GRPOConfig`" - `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `num_generations`, `max_prompt_length`, `max_completion_length`, `max_steps`, and checkpoint fields all live there; the bundled example config sets `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 1`, and `num_generations: 6`, so the effective batch per step is 1 x 1 x 6 generations on a single GPU (no multi-GPU multiplier is documented for this pipeline) [13][9]. A `model` section sets `lora_rank`, `lora_alpha`, `max_seq_length`, and `gpu_memory_utilization` (0.7 in the example, shared between training and the colocated vLLM generation server) [13].
- Out-of-memory first aid for GRPO is not separately itemized on the docs page beyond what the config exposes; the config's own knobs to shrink first are `gpu_memory_utilization` (generation-side, defaults to 0.7 in the example) and `max_completion_length` / `max_prompt_length` (train-side, 2500 and 4000 in the example) [13][9].
- The GRPO docs give a full worked setup on a fresh Ubuntu GPU rental: `apt-get update && apt-get install -y tmux nano`, clone the repo, `bash linux.sh` (documented to be allowed to error, since it "installs some dependencies"), then `source .venv/bin/activate` and `uv pip install -r generation/core_pipelines/do_grpo_rl_with_a_prompt/requirements.txt` to add the GRPO-only packages, edit `super_config.yaml`, `huggingface-cli login`, then `python run_augmentoolkit.py` [9].

## Watch it

This section is the mechanics only; augmentoolkit does not define what a GRPO reward curve or loss shape should look like - that judgment belongs on trl's GRPO method card, since the pipeline's training step is trl's own `GRPOTrainer` [8].

- Weights & Biases is the only logging path the docs mention for GRPO: after training finishes, the docs say to run `wandb login` and then "the wandb sync command it tells you to run" [9] - the docs do not name a `report_to` field or any other backend for this pipeline, and no explicit metric-name list for GRPO is published on this page or in the config schema read for this card [9][13].
- The config's `reward` section can save sample-level output: setting `reward.score_save_threshold` writes prompt/completion pairs whose combined reward exceeds that threshold to `high_scoring_examples.jsonl` in the run directory, described as "formatted for potential use as preference data" [9][13]. This is the only sample-level logging documented for GRPO; no separate flag for logging a sample of ordinary (non-high-scoring) generations is named in the pages read for this card.
- Evaluation-during-training fields are not named in the GRPO config or docs page read for this card; the `datasets` list instead defines per-dataset `eval_llm_name`, `eval_llm_base_url`, and `eval_llm_api_key`, but those configure an LLM used to grade generations as part of the reward computation itself, not a held-out evaluation split [13][9].
- No stopping-rule or reward threshold for ending training early is published: `max_steps` is the only training-length control named in the config and docs read for this card, and the docs state plainly "No Auto-Resume" as a known limitation of this pipeline rather than a stopping feature [9][13].
- Outside GRPO, augmentoolkit's dataset-generation pipelines (which are the majority of the project) have their own separate config-driven progress and cost logging (`cost` and `system` config sections), documented on `docs/config_common_fields.md`; that machinery is dataset-generation progress tracking, not training-run monitoring, and is out of scope for this "Watch it" section.

## Save it

- GRPO's trained artifact is a LoRA adapter, saved under the config's `training_parameters.output_dir` (the bundled example sets this to `outputs`); checkpoints are written according to `save_strategy` and `save_steps` from the same section (the example uses `save_strategy: steps`, `save_steps: 100`, `save_total_limit: 7`) [13][9].
- The docs describe this checkpoint directory as containing "the final adapter," to be "loaded and merged with the base model for inference" [9] - it is an adapter directory, not a full model directory, matching the general PEFT adapter-vs-full-model distinction (adapter files only, requiring the base model to reload).
- "No Auto-Resume" is stated directly in the GRPO docs as one of four numbered warnings about the pipeline, distinguishing it from "the common Augmentoolkit mainstay" of other pipelines that do auto-resume interrupted runs [9]. No resume call form or flag for continuing a stopped GRPO run from a saved checkpoint is given on the page read for this card.
- To ship the result, the docs' own post-training sequence is: `huggingface-cli login`, `wandb login`, run the wandb sync command it prints, then `huggingface-cli upload YourHFUsername/YourDesiredHFRepoWhereYouWantYourModelFiles outputs/` to push the raw adapter/checkpoint directory to the Hub, and finally `python cli_utils/merge_lora.py --base-model [base] --adapter-model [checkpoint path] --save-path [output folder]` to merge the LoRA adapter into the base model on disk [9].
- Whether an evaluator can load the result directly depends on which artifact you point it at: the unmerged `outputs/` checkpoint directory is an adapter and needs the base model plus a PEFT-aware loader, while the output of `cli_utils/merge_lora.py` is a full merged model directory, loadable on its own - the docs read for this card do not spell out the merged directory's file layout beyond naming the `--save-path` it lands in [9].
- For the Axolotl hand-off path (Complete Factual Datagen), augmentoolkit itself saves no model checkpoint at all - its output is the `pretraining_run/` and `sft_run/` folders, each holding a generated Axolotl YAML config plus training data, to be copied to a GPU machine and run with `accelerate launch -m axolotl.cli.train <config>.yaml`; checkpointing and saving for that run are Axolotl's contract, not augmentoolkit's [7][12].

## Find it in the docs

- The docs are plain Markdown files in the repository, not a hosted docs site: the README's own "Documentation Pages" section links each one by repo-relative path, e.g. `docs/quickstart.md`, `docs/grpo.md` (listed there as "GRPO (experimental)"), `docs/complete_factual_datagen.md`, `docs/axolotl_concepts.md`, `docs/config_common_fields.md`, `docs/CLI_flows.md`, `docs/project_structure.md`, `docs/vision.md` [11]. There is no versioned docs URL pattern to resolve - reading a page means reading that file at whatever commit of `master` you have checked out, or via `raw.githubusercontent.com/e-p-armstrong/augmentoolkit/<ref>/docs/<file>.md` for a specific ref.
- Question-to-file map, from the README's own index [11]: how training configs are produced -> `docs/complete_factual_datagen.md`; what an Axolotl config's fields mean -> `docs/axolotl_concepts.md`; fields shared by most pipeline configs (`api`, `path`, `system`, `cost`, `meta_datagen`) -> `docs/config_common_fields.md`; how to chain pipelines via `super_config.yaml` -> `docs/CLI_flows.md`; repo layout (`generation/core_pipelines`, `generation/core_composition`, interface architecture) -> `docs/project_structure.md`; the GRPO pipeline -> `docs/grpo.md`.
- Runnable references beyond the docs: the GRPO pipeline's own bundled example config at `generation/core_pipelines/do_grpo_rl_with_a_prompt/config.yaml` is a real, close-to-runnable file with `!!PLACEHOLDER!!` markers only for the base model name, an input dataset path, and API keys [13]; a starter Complete Factual Datagen config ships at `external_configs/_START_HERE_complete_factual.yaml`, named directly by the quickstart page as the place to start after the interface is running [5].
- Community layer: the README points to a Discord server for support and a free Substack ("Training and Datagen Tips") written by the maintainer for model-training and datagen advice, plus a YouTube channel of help videos including a "Train a Model on your Own Data in 13 Minutes" walkthrough [11] - there is no separate curated community-tutorials page distinct from these maintainer-run channels in the pages read for this card.
- No official MCP endpoint for querying these docs is named on any page read for this card.
- A trap stated where it bites: the GRPO docs' own numbered warnings say customizing reward functions "requires Python coding" (editing `reward_functions.py` and adding new prompt files for LLM-based rewards), and that "defaults and best practices are still evolving" for this pipeline, with the interface subject to change - these are maintainer-authored cautions on the docs page itself, not issue-tracker reports, and no closed-issue trap with an issue number was found in the pages read for this card [9].
- Honest boundary: the GRPO pipeline is stated to work only on Linux, because it depends on vLLM, and needs "a machine with a capable GPU (suitable for LoRA training)" - no Windows or macOS native path exists for this specific pipeline, and Windows users are directed to WSL for the rest of the project generally, not just GRPO [9][5].

## Sources

Ecosystem tools named in passing (trl, Unsloth, vLLM, Axolotl, Huey, Redis/Valkey, FastAPI, Uvicorn) are reached through the pages cited below and are not separately enumerated as references; trl's own GRPOTrainer and GRPOConfig surface is described on trl's own card (cross-reference, not covered here). All GitHub-hosted docs and source files are cited at commit `ec18b905f8a2fc1d64d345c04856cf2f036d9fc6` (the repository's newest push at screening time, 2026-06-27), which is ahead of the last tagged release `v3.0.0`; that gap is discussed explicitly in the Install field. All pages fetched and read 2026-08-11.

[1] augmentoolkit README, opening description. https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/README.md. Fetched 2026-08-11.

[2] augmentoolkit docs, Vision page, "What is Augmentoolkit? (Technical)" section. https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/docs/vision.md. Fetched 2026-08-11.

[3] augmentoolkit GitHub repository metadata (owner, html_url, pushed_at, topics, license, archived status). https://api.github.com/repos/e-p-armstrong/augmentoolkit. Fetched 2026-08-11.

[4] augmentoolkit GitHub repository. https://github.com/e-p-armstrong/augmentoolkit. Fetched 2026-08-11.

[5] augmentoolkit docs, Quickstart page (install commands per OS, local datagen, tensor-parallelism flag, Manual Interface Setup prerequisites and 3-terminal steps). https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/docs/quickstart.md. Fetched 2026-08-11.

[6] augmentoolkit docs, Project Structure page (interface architecture: atk-interface -> api.py -> Huey worker). https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/docs/project_structure.md. Fetched 2026-08-11.

[7] augmentoolkit docs, Complete Factual Datagen page ("Which files to use for training?", Axolotl config output, base-model rationale). https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/docs/complete_factual_datagen.md. Fetched 2026-08-11.

[8] augmentoolkit source, `grpo_trainer_subclass.py` (GRPOTrainerStopSequences subclassing trl's GRPOTrainer; Unsloth imports). https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/generation/core_pipelines/do_grpo_rl_with_a_prompt/grpo_trainer_subclass.py. Fetched 2026-08-11.

[9] augmentoolkit docs, GRPO page (experimental warning, config option descriptions, model/hardware requirements, output files, worked Runpod setup, wandb sync and upload/merge sequence, "No Auto-Resume"). https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/docs/grpo.md. Fetched 2026-08-11.

[10] augmentoolkit repository path listing the GRPO pipeline directory, `generation/core_pipelines/do_grpo_rl_with_a_prompt/`, as referenced in the shortlist row's methods_seen field. Fetched 2026-08-11.

[11] augmentoolkit README, Benefits section, Documentation Pages index, Python 3.11 instruction, community links. https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/README.md. Fetched 2026-08-11.

[12] augmentoolkit README, Useful Commands section (`accelerate launch -m axolotl.cli.train`). https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/README.md. Fetched 2026-08-11.

[13] augmentoolkit source, GRPO pipeline example config, `generation/core_pipelines/do_grpo_rl_with_a_prompt/config.yaml`. https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/generation/core_pipelines/do_grpo_rl_with_a_prompt/config.yaml. Fetched 2026-08-11.

[14] augmentoolkit repository root LICENSE file. https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/LICENSE. Fetched 2026-08-11.

[15] augmentoolkit repository root `requirements.txt`. https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/requirements.txt. Fetched 2026-08-11.

[16] augmentoolkit GRPO pipeline `requirements.txt`, `generation/core_pipelines/do_grpo_rl_with_a_prompt/requirements.txt`. https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/generation/core_pipelines/do_grpo_rl_with_a_prompt/requirements.txt. Fetched 2026-08-11.

[17] augmentoolkit GitHub Releases API (v3.0.0 tag, published date, release-notes text). https://api.github.com/repos/e-p-armstrong/augmentoolkit/releases. Fetched 2026-08-11.

[18] augmentoolkit GitHub Tags API (v3.0.0 tag resolved to commit `9d584026027610f2ec093d9f3f7c403aec57f2b5`). https://api.github.com/repos/e-p-armstrong/augmentoolkit/tags. Fetched 2026-08-11.

[19] augmentoolkit docs, CLI Flows page (`run_augmentoolkit.py`, `super_config.yaml`, `pipeline_order`). https://raw.githubusercontent.com/e-p-armstrong/augmentoolkit/ec18b905f8a2fc1d64d345c04856cf2f036d9fc6/docs/CLI_flows.md. Fetched 2026-08-11.
