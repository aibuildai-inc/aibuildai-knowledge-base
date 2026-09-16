# distilabel

A synthetic-data and AI-feedback pipeline framework that builds datasets for post-training - it does not itself train models, and its "DPO"/"SFT" output is a dataset shape for other trainers to consume.

**distilabel** is "the framework for synthetic data and AI feedback for engineers who need fast, reliable and scalable pipelines based on verified research papers" [1]. It is built by Argilla, and the project's own README states that its original authors have moved on and a group of community members has since joined as collaborators to maintain it and prepare the next release, directing users to the `develop` branch for the latest fixes [1]. Its API shape is a `Pipeline` context manager that chains `Step` and `Task` objects (the latter driven by an `LLM` client) into a DAG, run with `pipeline.run(...)` to produce a `Distiset` - a dict-like collection of Hugging Face `datasets.Dataset` objects, one per output branch [2][3]. It lives at https://github.com/argilla-io/distilabel [23]. This card is written for readers who want distilabel to prepare training data for a separate trainer (trl, axolotl, alignment-handbook, or similar); it does not cover choosing or operating a trainer.

**When to pick it**: pick distilabel when the task is to generate or reshape a dataset with LLM calls - instruction generation, model-judging/AI-feedback, evaluation, or converting raw generations into SFT- or DPO-shaped rows - not when the task is to run a training loop; distilabel ships no optimizer, loss, or gradient-update code anywhere in the two formatting-step files or the pipeline API read for this card [4][5][2]. Its own docs list its two purpose-built dataset-formatting steps as producing output "following the standard formatting from frameworks such as `axolotl` or `alignment-handbook`" [4][5], i.e. distilabel is the upstream data stage, and a trainer card from this same deck (trl, axolotl, ...) is the downstream stage - not a substitute for one.

**Methods it ships**: not training methods but two dataset-formatting `Step` pairs under `distilabel.steps.formatting`: `FormatTextGenerationDPO`/`FormatChatGenerationDPO`, which pick the highest- and lowest-rated of a list of `generations` as `chosen`/`rejected` and emit `prompt`, `prompt_id`, `chosen`, `chosen_rating`, `rejected`, `rejected_rating` [4]; and `FormatTextGenerationSFT`/`FormatChatGenerationSFT`, which convert an `instruction`/`messages` plus a single `generation` into a chat-format `messages` column plus `prompt`/`prompt_id` [5]. Beyond formatting, the library's Task Gallery lists 34 generative `Task`-family classes (plus 4 plain `Step` subclasses documented on the same page) for the generation/judging work upstream of formatting (e.g. `TextGeneration`, `EvolInstruct`, `UltraFeedback`, `PrometheusEval`, `SelfInstruct`, `Magpie`/`MagpieGenerator`, `CLAIR`, `MathShepherdGenerator`) [6]; eight of the docs' "Papers" tutorials reproduce specific published pipelines (DeepSeek Prover, DEITA, Instruction Backtranslation, Prometheus 2, UltraFeedback, APIGen, CLAIR, Math Shepherd) [7]. The task/step taxonomy is a live, moving page - the Task Gallery lives at the `api/task/task_gallery` slug and the broader Step Gallery is split into Argilla, Hugging Face, Columns, and Extra sub-pages [7]; recheck there. distilabel also ships a generic, no-pipeline-authoring path: `InstructionResponsePipeline` builds an SFT-shaped instruction/response dataset out of the box, but as of this docs snapshot the maintainers state a generic DPO pipeline class is not yet built, only planned [3].

**Scale it handles**: single machine via Python multiprocessing (the default `Pipeline`, with per-step `input_batch_size`/`batch_size` to bound memory) up to a multi-node Ray cluster (`pip install distilabel[ray]`), where each `Step` replica runs as a Ray Actor and GPU/replica counts are set per step via `resources={"replicas": N, "gpus": M}` [8]. Ray execution is documented for both the Ray Jobs API (a working directory plus a `runtime_env.yaml` pinning extras, submitted with `ray job submit`) and a SLURM-launched multi-node cluster (a full head/worker `ray start` batch script is given) [8]. Multi-GPU/multi-node generation with vLLM is reached by setting `tensor_parallel_size` and `distributed_executor_backend: "ray"` inside the `vLLM` LLM's `extra_kwargs` [8]. All of this is documented mechanism with no published throughput or scaling benchmark in the pages read for this card - treat it as "works, unmeasured here."

**Install**: `pip install distilabel --upgrade`; latest PyPI release `1.5.3`, released 2025-01-28 [9]; Python `>=3.9` (docs additionally cap practical use at 3.12, calling 3.12 support still in progress) [10][11]; Apache-2.0 [10]. Core dependencies carry no deep-learning framework at all - no torch or transformers in the base install. Among the LLM/data/structured-generation extras, only one carries an upper bound, `mlx-lm>=0.21.0,<0.22.0`; every other integration extra (`vllm`, `hf-transformers`, `ray`, `outlines`, `instructor`, `argilla`, ...) is a floor-only `>=` pin. The `dev` extra (not load-bearing for running a pipeline) additionally exact-pins `ruff==0.8.1`. All read from `pyproject.toml` at the `1.5.3` release tag (commit `1b6c101c3012c9d1306227566ed9ad8dd463309b`) [12]. The screening commit `313fac85b1a2472dd88db1a31c2b754599f46476` (pushed 2026-07-27, over a year past the 1.5.3 release) still reports `__version__ = "1.5.3"` in `src/distilabel/__init__.py` [13], and its `pyproject.toml` is byte-identical to the one at the release tag [12][14] - so these pins hold at both points, but no newer release exists to install from PyPI as of this reading; the extra dependency-heavy commits since then are only reachable via the source install (`pip install "distilabel @ git+...@develop"`) [11]. No CUDA or GPU-driver minimum is stated on the installation page or in the package metadata; GPU requirements flow through whichever LLM-integration extra (`vllm`, `hf-transformers`, `llama-cpp`) is installed [11].

**Maintained by**: Argilla (`argilla-io` on GitHub), with a README-stated maintenance handover from the original author team to a group of community collaborators who describe themselves as actively working toward the next release off the `develop` branch [1]; the repository is not archived and its last push (2026-07-27) is well after the last PyPI release (2025-01-28), consistent with active unreleased development [10][13].

## Quick start

The smallest complete run, from the docs quickstart - a generic pipeline needing no manual `Step` wiring, using a hosted Inference Endpoint:

```python
from distilabel.pipeline import InstructionResponsePipeline

pipeline = InstructionResponsePipeline()
dataset = pipeline.run()
```

This defaults to `InferenceEndpointsLLM` with `meta-llama/Meta-Llama-3.1-8B-Instruct` and produces an `instruction`/`response` dataset [3]. The custom-pipeline form, also from the quickstart, wires a `Pipeline` explicitly and runs it over a Hub dataset:

```python
from datasets import load_dataset
from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            generation_kwargs={"temperature": 0.7, "max_new_tokens": 512},
        ),
    )

if __name__ == "__main__":
    dataset = load_dataset("distilabel-internal-testing/instructions", split="test")
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="distilabel-example")
```
[3]

The CLI covers re-running an already-serialized pipeline, not authoring one: `distilabel pipeline run --config <path-or-URL-to-pipeline.yaml>`, e.g. against a Hub-hosted `pipeline.yaml` [15].

## Start it

- One process, no cluster: the scripts above, as-is - `Pipeline` runs locally via Python multiprocessing by default [2][3].
- Distributed execution goes through Ray: install with `pip install distilabel[ray]`, and set `resources={"replicas": N, "gpus": M}` on the steps that need it; each replica becomes a Ray Actor. The recommended launch path is the Ray Jobs API - package the pipeline script with a `runtime_env.yaml` pinning `distilabel[ray,vllm]` and any secrets (e.g. `HF_TOKEN`), then run `ray job submit --address http://localhost:8265 --working-dir <dir> --runtime-env <dir>/runtime_env.yaml -- python pipeline.py` [8]. Multi-node adds a SLURM batch script that starts a Ray head node and worker nodes via `ray start`, then submits the same job [8]. A `GlobalStep` or `use_fs_to_pass_data=True` requires a shared filesystem across all Ray nodes, set via `storage_parameters={"path": "file:///mnt/data"}` [8].
- Generation-layout choice: vLLM as the `LLM` backend takes `extra_kwargs={"tensor_parallel_size": N, "distributed_executor_backend": "ray"}` to shard one model across N GPUs/nodes under Ray [8].
- There is no training batch-size arithmetic here; the analogous knob is per-step throughput: `batch_size` on a `GeneratorStep` (e.g. `LoadDataFromHub`) sets how many rows it emits per batch, and `input_batch_size` on a downstream `Task` sets how many rows it consumes per call - tuning these two against each other is how a pipeline avoids overloading either a data source or an LLM endpoint [16].
- Config surface: individual `Step`/`Task` constructor arguments (`input_mappings`, `output_mappings`, `input_batch_size`, `resources`) plus `Pipeline.run(parameters=..., dataset=..., use_cache=..., storage_parameters=...)`; `Pipeline.dry_run(parameters=..., batch_size=1)` validates the DAG on one row before a full run [16].
- Out-of-memory first aid: lower `input_batch_size` on the memory-heavy `Task`/`Step` - the docs' own memory-issues guidance is exactly this one knob, tuned per step rather than globally [16]. A documented rate-limit trap sits next to it: against Hugging Face's Free Serverless Inference Endpoints, the default input batch size of 50 can trigger "Model is overloaded" errors, and the fix given is to set `input_batch_size=1` on the `Task` [17].

## Watch it

distilabel is a pipeline-execution framework, not a training loop, so there is no loss/reward curve to watch here - the signal is whether steps are progressing and what they are producing. What follows is mechanics only; there is nothing to interpret about training dynamics because none run inside distilabel.

- **Enable it**: no dedicated tracker integration (no `report_to`/wandb/tensorboard hook) was found in the pipeline-execution, caching, CLI, Ray, FAQ, or Distiset API pages read for this card - distilabel's only persistent execution record is a `pipeline.log` file, written automatically for every `Pipeline.run()` and reachable on the resulting `Distiset` via `Distiset.log_filename_path` [18]. Console progress during a run is via `rich`, a listed core dependency, though the docs pages read here do not spell out its exact display [12].
- **Stopping mid-run**: pressing Ctrl+C (SIGINT) once stores the outputs completed so far in the cache and lets the pipeline stop cleanly; pressing it again forces an immediate stop and can lose the batches in flight - this is the docs' own stated rule, not a threshold the user tunes [19].
- **No stopping-rule search applies**: distilabel has no reward/loss metric to threshold against, so the "shapes published, thresholds not" question that applies to trainers does not apply here; the closest analogue - `Pipeline.dry_run(parameters=..., batch_size=1)` to validate a DAG before committing a full run - is a correctness check, not a stopping rule [19].
- **Evaluation-during-training fields**: not applicable - distilabel runs a fixed DAG once per invocation; a `Task` such as `PrometheusEval` or `UltraFeedback` can score generations as part of that same DAG, but that is pipeline output, not a periodic eval loop [6].
- **No published runtime health limit** was found: the pipeline-execution guide's only cautionary note is the batch-size-vs-memory guidance above, and the FAQ page's only reliability-relevant items are the cache-on-failure guarantee and the Free-Serverless-Endpoint overload workaround, both already covered [16][17]. Files searched: the pipeline-execution guide, the caching guide, the FAQ, and the CLI guide [16][19][17][15].

## Save it

- The unit distilabel saves is a `Distiset`, not a model checkpoint: a dict subclass wrapping one `datasets.Dataset` per leaf step of the pipeline DAG [20].
- `distiset.save_to_disk(distiset_path, max_shard_size=None, num_shards=None, num_proc=None, storage_options=None, save_card=True, save_pipeline_config=True, save_pipeline_log=True)` writes the dataset plus, by default, a dataset card, the `pipeline.yaml` that produced it, and the `pipeline.log` from the run; `distiset_path` accepts a local path or an `fsspec` remote path (e.g. `s3://...`) [20]. `Distiset.load_from_disk(distiset_path, keep_in_memory=None, storage_options=None, download_dir=None)` reloads it [20].
- `distiset.push_to_hub(repo_id, private=False, token=None, generate_card=True, include_script=False, **kwargs)` uploads the dataset to the Hub, and, per the docs, also uploads the `pipeline.yaml` and `pipeline.log` alongside it, giving a Hub-hosted repo a full record of how the data was generated [20].
- Resume is pipeline-level caching, not checkpoint loading: `Pipeline.run()` stores per-step state and intermediate outputs keyed by the pipeline's cache signature, so re-running the same script resumes from the last completed step; `use_cache=False` forces a full rerun, and the FAQ confirms a Ctrl+C-interrupted run keeps its completed-step outputs in `.cache/distilabel` for that same resume path [19][17].
- There is no adapter/full-model distinction here since nothing is trained; `pipeline.save("pipeline.yaml"|"pipeline.json")` and `Pipeline.from_yaml(...)`/`Pipeline.from_json(...)` serialize and reload the DAG definition itself (steps, connections, runtime parameters), separately from the `Distiset` outputs [16].
- Loader handoff: the output is a Hugging Face `datasets.Dataset` (or a Hub dataset repo after `push_to_hub`), directly loadable with `datasets.load_dataset(...)` or `Distiset.load_from_disk(...)` by any downstream consumer - most relevantly a trainer such as trl, which reads DPO- or SFT-shaped rows exactly like the ones `FormatTextGenerationDPO`/`FormatTextGenerationSFT` produce [4][5][20]. Whether a specific trainer's loader accepts the exact column names distilabel emits is that trainer's contract, not distilabel's - check the target trainer's dataset-format expectations against the formatting step's stated output columns [4][5].

## Find it in the docs

The docs are the live source; this section teaches the lookup, not the content.

- Address pattern: `https://distilabel.argilla.io/<version>/<page>`. `<version>` is `latest` (tracks the newest published docs build) or a release tag with no leading `v`, e.g. `1.5.3` - checked 2026-08-11, both `https://distilabel.argilla.io/1.5.3/` and `https://distilabel.argilla.io/latest/sections/how_to_guides/basic/pipeline/` return 200 [21]. The site is built with mkdocs-material and versioned with `mike` [22].
- Page-slug recipe: how-to guides live under `sections/how_to_guides/basic/<topic>` or `.../advanced/<topic>` - e.g. `basic/pipeline` (this card's Start-it/Watch-it/Save-it source), `advanced/caching`, `advanced/scaling_with_ray`, `advanced/cli` (the CLI guide sits under `advanced`, not `basic`, despite covering a beginner-facing command) [22]. API reference pages live under `api/<object>/<subtopic>`, e.g. `api/pipeline/`, `api/task/task_gallery` for the full Task Gallery, `api/step_gallery/{argilla,hugging_face,columns,extra}` for step-level integrations - the DPO/SFT formatting steps documented in this card are covered on `api/step_gallery/columns` rather than a dedicated formatting page [22].
- Question-to-slug map: "how do I scale this out" -> `sections/how_to_guides/advanced/scaling_with_ray`; "what happens if my run crashes" -> `sections/how_to_guides/advanced/caching` and the FAQ page `sections/getting_started/faq` (`Pipeline.run` failure and Ctrl+C behavior are both answered there) [19][17]; "what does this task/step do" -> the relevant Gallery page under `api/` [22].
- Runnable references beyond the docs: the "Tutorials" nav holds four worked notebooks (preference-dataset generation, cleaning an existing preference dataset, sentence-pair generation for retrieval/reranking, text-classification data generation) plus eight "Papers" pages that reproduce specific published pipelines end to end (DeepSeek Prover, DEITA, Instruction Backtranslation, Prometheus 2, UltraFeedback, APIGen, CLAIR, Math Shepherd) [7]. Known-good smoke-test data used throughout the docs includes `distilabel-internal-testing/instruction-dataset-mini` and the pre-generated `distilabel-internal-testing/instruction-dataset-mini-with-generations` (used directly by the CLI guide's `pipeline info`/`pipeline run` examples) [15][16].
- Community layer: the README points to a bi-weekly community meetup, a Discord (`#argilla-general`, `#argilla-help`), and a public roadmap board, but the pages read for this card carry no separate curated community-tutorials page comparable to trl's - these are the community channels distilabel's own README names, not a curated content index [1].
- No official MCP endpoint for these docs was found in the pages read for this card.
- Honest boundary: distilabel does not train models - it has no optimizer, trainer class, or checkpoint-with-optimizer-state anywhere in the pipeline or formatting-step code read for this card [2][4][5]. As of this docs snapshot, a built-in generic DPO pipeline (parallel to `InstructionResponsePipeline` for SFT) does not exist yet - the quickstart page states it explicitly as planned, not shipped [3]. No maintainer-reply GitHub issue trap is included in this card: GitHub's REST API rate-limited every attempt to fetch issue threads during this research session, so none could be verified firsthand: for the same reason, `releases.json` and `repo_root.json` fetch attempts in this card's own working notes returned only rate-limit errors and are not cited as sources.

## Sources

Unless a specific commit or PyPI release is named, sources are `latest`-version docs or live pages read on 2026-08-11. Ecosystem tools named in passing (Ray, SLURM, vLLM, the Hugging Face Hub, `datasets`, `rich`) are reached through the cited distilabel pages and are not separately enumerated. Method-adjacent terms (DPO, SFT) are deliberately cited only to distilabel's own formatting-step docstrings here, not to their defining papers, since distilabel does not implement the training methods themselves.

[1] distilabel README, at commit 313fac85b1a2472dd88db1a31c2b754599f46476. https://raw.githubusercontent.com/argilla-io/distilabel/313fac85b1a2472dd88db1a31c2b754599f46476/README.md. Fetched 2026-08-11.

[2] distilabel pipeline-execution how-to guide (Pipeline/Step/Task/DAG shape, `Pipeline.save`/`from_yaml`/`from_json`). https://distilabel.argilla.io/latest/sections/how_to_guides/basic/pipeline/. Fetched 2026-08-11.

[3] distilabel Quickstart docs (`InstructionResponsePipeline`, custom-pipeline example, planned-but-unshipped generic DPO pipeline). https://distilabel.argilla.io/latest/sections/getting_started/quickstart/. Fetched 2026-08-11.

[4] `src/distilabel/steps/formatting/dpo.py` at commit 313fac85b1a2472dd88db1a31c2b754599f46476 (`FormatTextGenerationDPO`, `FormatChatGenerationDPO`, axolotl/alignment-handbook formatting language, input/output columns). https://raw.githubusercontent.com/argilla-io/distilabel/313fac85b1a2472dd88db1a31c2b754599f46476/src/distilabel/steps/formatting/dpo.py. Fetched 2026-08-11.

[5] `src/distilabel/steps/formatting/sft.py` at commit 313fac85b1a2472dd88db1a31c2b754599f46476 (`FormatTextGenerationSFT`, `FormatChatGenerationSFT`). https://raw.githubusercontent.com/argilla-io/distilabel/313fac85b1a2472dd88db1a31c2b754599f46476/src/distilabel/steps/formatting/sft.py. Fetched 2026-08-11.

[6] distilabel Task Gallery (API reference; 38 documented classes total, of which 34 are `Task`-family classes and 4 are plain `Step` subclasses). https://distilabel.argilla.io/latest/api/task/task_gallery/. Fetched 2026-08-11.

[7] distilabel docs navigation (`mkdocs.yml`) at commit 313fac85b1a2472dd88db1a31c2b754599f46476, naming the Tutorials/Papers nav entries and the Task/Step Gallery sub-pages. https://raw.githubusercontent.com/argilla-io/distilabel/313fac85b1a2472dd88db1a31c2b754599f46476/mkdocs.yml. Fetched 2026-08-11.

[8] distilabel "Scaling and distributing a pipeline with Ray" guide (Ray Actors, Jobs API, `runtime_env.yaml`, SLURM script, `storage_parameters`, vLLM `tensor_parallel_size`/`distributed_executor_backend`). https://distilabel.argilla.io/latest/sections/how_to_guides/advanced/scaling_with_ray/. Fetched 2026-08-11.

[9] PyPI JSON API for distilabel (release version and upload-date history). https://pypi.org/pypi/distilabel/json. Fetched 2026-08-11.

[10] distilabel `pyproject.toml` at commit 313fac85b1a2472dd88db1a31c2b754599f46476 (`requires-python`, `license`, core dependency list, optional-dependency extras, project URLs). https://raw.githubusercontent.com/argilla-io/distilabel/313fac85b1a2472dd88db1a31c2b754599f46476/pyproject.toml. Fetched 2026-08-11.

[11] distilabel Installation docs (Python 3.9-3.12 floor/ceiling note, source install command, extras list, no stated CUDA/hardware minimum, `flash-attn` recommendation). https://distilabel.argilla.io/latest/sections/getting_started/installation/. Fetched 2026-08-11.

[12] distilabel `pyproject.toml` at the `1.5.3` release-tag commit 1b6c101c3012c9d1306227566ed9ad8dd463309b (dependency floors and extras as actually shipped in the installable release). https://raw.githubusercontent.com/argilla-io/distilabel/1b6c101c3012c9d1306227566ed9ad8dd463309b/pyproject.toml. Fetched 2026-08-11.

[13] `src/distilabel/__init__.py` at commit 313fac85b1a2472dd88db1a31c2b754599f46476 (`__version__ = "1.5.3"` still reported at the screening commit). https://raw.githubusercontent.com/argilla-io/distilabel/313fac85b1a2472dd88db1a31c2b754599f46476/src/distilabel/__init__.py. Fetched 2026-08-11.

[14] Diff between [10] and [12], run locally 2026-08-11: no differences found, confirming dependency pins are unchanged between the 1.5.3 release tag and the screening commit.

[15] distilabel CLI guide (`distilabel pipeline info`/`distilabel pipeline run`, `--config`/`--script`/`--param`/`--ignore-cache` options, the `distilabel-internal-testing/instruction-dataset-mini-with-generations` example dataset). https://distilabel.argilla.io/latest/sections/how_to_guides/advanced/cli/. Fetched 2026-08-11.

[16] distilabel pipeline-execution how-to guide, batch-size/memory and `Pipeline.dry_run`/`Pipeline.run` sections. https://distilabel.argilla.io/latest/sections/how_to_guides/basic/pipeline/. Fetched 2026-08-11.

[17] distilabel FAQ page (cache-on-`Pipeline.run`-failure guarantee, `.cache/distilabel` location, Free Serverless Endpoint overload workaround with `input_batch_size=1`). https://distilabel.argilla.io/latest/sections/getting_started/faq/. Fetched 2026-08-11.

[18] distilabel Distiset API reference (`Distiset.log_filename_path` property description). https://distilabel.argilla.io/latest/api/distiset/. Fetched 2026-08-11.

[19] distilabel pipeline-execution how-to guide, "Stopping the pipeline" and "Cache" sections (Ctrl+C behavior, `use_cache` argument). https://distilabel.argilla.io/latest/sections/how_to_guides/basic/pipeline/. Fetched 2026-08-11.

[20] distilabel Distiset API reference (`Distiset` class docstring, `save_to_disk`, `load_from_disk`, `push_to_hub` signatures and behavior including `pipeline.yaml`/`pipeline.log` upload). https://distilabel.argilla.io/latest/api/distiset/. Fetched 2026-08-11.

[21] Live HTTP status check of the versioned docs URL forms, run 2026-08-11: `https://distilabel.argilla.io/1.5.3/` and `https://distilabel.argilla.io/latest/sections/how_to_guides/basic/pipeline/` both returned 200.

[22] distilabel `mkdocs.yml` navigation tree at commit 313fac85b1a2472dd88db1a31c2b754599f46476, giving the page-slug patterns for how-to guides and API reference pages. https://raw.githubusercontent.com/argilla-io/distilabel/313fac85b1a2472dd88db1a31c2b754599f46476/mkdocs.yml. Fetched 2026-08-11.

[23] distilabel GitHub repository (the item's home page). https://github.com/argilla-io/distilabel. Fetched 2026-08-11.
