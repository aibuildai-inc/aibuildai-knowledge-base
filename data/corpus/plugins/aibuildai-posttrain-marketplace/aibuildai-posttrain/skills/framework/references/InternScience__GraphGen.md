# GraphGen

A knowledge-graph-guided synthetic data generator that produces SFT/pretrain/RLVR training data from raw text; it does not train models itself.

**GraphGen** is "a framework for synthetic data generation guided by knowledge graphs" [1]. It builds a fine-grained knowledge graph from source text, scores an LLM's knowledge gaps with an expected-calibration-error-style metric to prioritize long-tail knowledge, and generates question-answer pairs through multi-hop neighborhood sampling and style-controlled generation [1]. It was originally published as `open-sciencelab/GraphGen` and now lives at `InternScience/GraphGen` (the shortlist row records the former name) [2]. At the current git source (the screening commit), its API is a YAML-declared node pipeline - read, chunk, build_kg, quiz, judge, partition, generate - executed as a Ray dataset job through `python -m graphgen.run --config_file <yaml>` [1][3]. This is NOT the same pipeline `pip install graphg` delivers: the published package resolves to an older, pre-Ray release whose console script is named `graphgen` and runs a different, flat-config `GraphGen` class instead (see Install, below) [4][5]. The repository lives at https://github.com/InternScience/GraphGen [2].

**When to pick it**: pick GraphGen when you need to turn raw text (files, PDFs, search engines, or bio/chem databases such as UniProt, NCBI, RNAcentral) into knowledge-graph-derived QA data for SFT, pretraining rephrase, or RLVR, using its knowledge-gap-targeting and multi-hop sampling as the differentiator over generic prompted data generation [1]. It is a data-generation front end, not a trainer: the README's own next step after generation is to fine-tune with LLaMA-Factory or xtuner [1], so weigh trl, verl, LLaMA-Factory, or xtuner (cross-reference; not covered here) for the actual training loop. The README's own effectiveness tables are the deciding numbers behind that pitch: with Qwen2.5-7B-Instruct as baseline, a model whose SFT data was over 50% GraphGen-generated scores 65.9 vs. 51.5 on SeedBench, 20.6 vs. 16.7 on AIME24, and 22.7 vs. 7.2 on AIME25, though it trails the baseline on CMMLU (73.6 vs. 75.8); a Qwen2.5-7B base model RL-tuned directly on GraphGen-synthesized verifiable data (no prior SFT) scores 66.8 vs. 51.5 on SeedBench, 87.1 vs. 80.7 on MedQA, and 55.3 vs. 49.6 on BBH [1].

**Methods it ships**: none in the trainer sense - GraphGen implements no PPO/GRPO/DPO/SFT training loop. What it ships under `graphgen/models/evaluator/qa/` are four QA-pair quality scorers used to filter or rank generated data: `reward_evaluator.py`, which scores a question-answer pair by feeding it to a pretrained HuggingFace sequence-classification reward model (default `OpenAssistant/reward-model-deberta-v3-large-v2`) and returning its `reward_score` logit [6]; `length_evaluator.py`, `mtld_evaluator.py`, and `uni_evaluator.py` sit alongside it in the same directory [7]. This card reads `reward_evaluator.py` in full; the other three evaluators were only confirmed to exist by directory listing, not read. The "REWARD" method tag on this row names a data-quality filter, not a reward-model-training implementation.

**Scale it handles**: each pipeline node's YAML block sets `execution_params.replicas` and `batch_size`, and the whole pipeline runs as a Ray dataset job (`ray.data.from_items` in `graphgen/run.py`) [3][8]. The README's own changelog says the pipeline was "[r]efactored ... using ray to improve the efficiency of distributed execution and resource management" [1], which documents the mechanism but publishes no throughput or multi-node benchmark, and no cluster-launch (Ray head/worker, SLURM) instructions were found on the pages read for this card. GPU use enters only through the pluggable LLM backend: local HuggingFace/vLLM/SGLang inference backends take `SYNTHESIZER_NUM_GPUS` / `SYNTHESIZER_TP_SIZE` (and the `TRAINEE_*` equivalents) for tensor-parallel local model serving [9].

**Install**: `pip install graphg` installs whatever version PyPI's own JSON API names as `info.version` for the package, which is `20250416`, uploaded 2025-04-22 - not the more recently tagged `0.1.0.post20250930` [4]. This is a PEP 440 ordering trap: the bare numeric string `20250416` sorts as a higher version than `0.1.0.post20250930` under PEP 440, so pip's resolver prefers it even though GitHub's own release list shows `0.1.0.post20250930` published on 2025-09-30, over five months after the `20250416` release's 2025-04-22 upload date [4][10]. The published wheel confirms the actually-installed shape at this version: its `entry_points.txt` declares the console script as `graphgen = graphgen.generate:main` (not `graphg`) [5], and its `METADATA` declares no `Requires-Python` and no `License` classifier, matching `info.requires_python: null` and `info.license: null` on PyPI [5][4]. `Requires-Dist` at this version is tqdm, openai, python-dotenv, numpy, networkx, graspologic, tiktoken, pyecharts, wikipedia, tenacity, nltk, jieba, plotly, pandas, `gradio>=5.25.0`, `gradio-i18n==0.3.0`, kaleido, pyyaml, langcodes - no ray, kuzu, rocksdict, torch, or transformers [5]. The corresponding git tag is `20250422` (commit `e34135d7a1f64c07360652c9625f26a4bf8540cd`) [11]; at that commit `setup.py`'s classifiers claim Python 3.8-3.12 [12], and the package runs the older flat-config `GraphGen` class (`.insert()`/`.quiz()`/`.judge()`/`.traverse()`) via `graphgen/generate.py`, reading a single YAML `--config_file` and env-var-selected `SYNTHESIZER_MODEL`/`TRAINEE_MODEL` OpenAI-style clients - it has no `engine.py`, no `run.py`, and no kuzu/rocksdb backend [13][14]. The node-graph YAML pipeline, Ray execution, and kuzu/rocksdb backends shown in the current README's "Run from Source" quickstart exist only in the git checkout at the screening commit (`d9b8bedb5152499718e816d803668d40676f9781`) and have never been cut into a PyPI release [15][16][8]. License is Apache-2.0 per the repository's own license field [2], though the PyPI wheel itself declares no license classifier, as noted above. No CUDA or hardware minimum is stated in any file read for this card.

**Maintained by**: originally the open-sciencelab organization, now InternScience [2]; backed by an arXiv paper (2505.20416) and, per the README's changelog, a follow-up paper built on GraphGen, "Knowledge-to-Verification: Exploring RLVR for LLMs in Knowledge-Intensive Domains," accepted to ACL 2026 Main Conference as of the 2026-04-13 entry [1]. The commit read for this card (`d9b8bedb5...9781`, pushed 2026-05-19) is authored by Nanqing Dong (SII) [17]; GitHub lists two tagged releases, the latest `v0.1.0.post20250930` on 2025-09-30 [10].

## Quick start

The README's "Run from PyPI" path prints the command as `graphg --output_dir cache` [1], but the wheel this install line actually downloads names its console script `graphgen`, not `graphg` (see Install, above) - the corrected form is:

```bash
uv pip install graphg
SYNTHESIZER_MODEL=your_synthesizer_model_name \
SYNTHESIZER_BASE_URL=your_base_url_for_synthesizer_model \
SYNTHESIZER_API_KEY=your_api_key_for_synthesizer_model \
TRAINEE_MODEL=your_trainee_model_name \
TRAINEE_BASE_URL=your_base_url_for_trainee_model \
TRAINEE_API_KEY=your_api_key_for_trainee_model \
graphgen --output_dir cache
```

From the README's "Run from Source" path, which is the only path that exercises the Ray/node-graph pipeline described above [1]:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --depth=1 https://github.com/InternScience/GraphGen
cd GraphGen
uv venv --python 3.10
uv pip install -r requirements.txt
cp .env.example .env   # set SYNTHESIZER_*/TRAINEE_* backend and model
bash examples/generate/generate_aggregated_qa/generate_aggregated.sh
```

That script runs `python3 -m graphgen.run --config_file examples/generate/generate_aggregated_qa/aggregated_config.yaml` [18]. Generated data appears at `ls cache/output` [1].

## Start it

- One process is the only launch form documented: `python -m graphgen.run --config_file <yaml>` for the current git source's Ray node-graph pipeline, or the `graphgen` console script (installed by `pip install graphg`) for the older flat-config pipeline that PyPI actually ships [1][3][5]. No accelerate/torchrun/deepspeed launcher is involved in either path - both are single-process jobs (a Ray dataset job in the source pipeline, a plain Python loop in the PyPI one), not a model-parallel trainer.
- The Config surface is the pipeline YAML: a `global_params` block (`working_dir`, `graph_backend: kuzu|networkx`, `kv_backend: rocksdb|json_kv`) and a `nodes` list, where each node names an `op_name`, its `dependencies`, an `execution_params` block (`replicas`, `batch_size`) that sets its Ray parallelism, and op-specific `params` - e.g. the `generate` node's `method` (atomic/aggregated/multi_hop/cot/vqa/...) and `data_format` (Alpaca/Sharegpt/ChatML) [19]. There is no RL/SFT-style effective-batch arithmetic here; `batch_size` is a Ray `map_batch` chunk size per node, not a training batch.
- Generation-layout choice: which model does synthesis vs. which model is being profiled for knowledge gaps is set independently through `SYNTHESIZER_BACKEND`/`SYNTHESIZER_MODEL` and `TRAINEE_BACKEND`/`TRAINEE_MODEL` env vars, with `*_NUM_GPUS`/`*_TP_SIZE` for the local vLLM/SGLang backends [9]. The file's own top-of-section comment names eight backends - `http_api`, `openai_api`, `ollama_api`, `ollama`, `huggingface`, `tgi`, `sglang`, `tensorrt` - but the example blocks that follow it in the same file demonstrate a different set of six - `openai_api`/`http_api` sharing one block, plus `azure_openai_api`, `ollama_api`, `huggingface`, `sglang`, `vllm` - so the comment and the worked examples disagree on the exact backend list, and only the six with worked blocks are shown configured end to end [9]. One documented incompatibility: "TRAINEE with ollama_api backend is not supported yet as ollama_api does not support logprobs" [9].
- Out-of-memory first aid: not published on the pages read for this card (README, `.env.example`, `run.py`, the example config) beyond the local-backend GPU-count/tensor-parallel knobs above; no generation-pipeline-specific OOM guidance was found.

## Watch it

This section is mechanics only; GraphGen has no training-signal semantics to interpret, since it runs no training loop.

- Logging is a plain file logger, not a metrics tracker: `graphgen/run.py` calls `set_logger(log_path, name="GraphGen", if_stream=True)` with `log_path = <working_dir>/logs/Driver.log`, so every run streams to stdout and to that file by default [8]. No wandb, TensorBoard, trackio, or other experiment-tracker integration was found in the README, `requirements.txt`, or `run.py` at the screening commit - state plainly: none found, searched README.md, requirements.txt, and graphgen/run.py.
- Sample-level output is the product, not a side channel: the `generate` node's `save_output: true` writes the generated QA pairs in the configured `data_format` under `<working_dir>/output/<unix-timestamp>/`, a directory `run.py` creates from `int(time.time())` at start and also writes a copy of the resolved `config.yaml` into [19][8].
- Evaluation during generation is the `quiz`/`judge`/`partition` nodes: `quiz` samples QA probes on the trainee model, `judge` scores them, and `partition` groups knowledge-graph units using `method: ece` - "a custom partition method based on comprehension loss" per the example config's comment - prioritizing units the trainee model gets wrong [19][1]. Post-hoc filtering is separate: the `reward`/`length`/`mtld`/`uni` evaluators under `graphgen/models/evaluator/qa/` score already-generated pairs [7].
- No stopping rule applies: GraphGen's pipeline is finite, each node runs once over its input dataset and the process exits (`run.py` calls `engine.execute(...)` once, then returns) [8] - there is no iterative training loop, so no early-stopping threshold to publish.

## Save it

- There is no checkpoint directory, no optimizer/RNG state, and no adapter-vs-full-model distinction to report, because GraphGen trains nothing. What lands on disk is the generated dataset plus its provenance: `<working_dir>/output/<unique_id>/` holds the QA pairs in the run's `data_format`, and `run.py` writes the resolved pipeline config to `<output_dir>/config.yaml` alongside it via a dedicated `save_config()` call [8][19].
- There is no resume-from-checkpoint call; re-running the same config re-executes the pipeline (a new `<unique_id>` output directory each time, per `int(time.time())`) [8].
- Loader handoff: the README states the intended next step in one sentence - "you can use LLaMA-Factory and xtuner to finetune your LLMs" on the generated data [1]. ChatML/Alpaca/Sharegpt are all formats those trainers and trl-family SFT trainers consume, but this card does not verify field-level schema compatibility with any specific loader - check the target trainer's own dataset-format docs before use.

## Find it in the docs

- Docs are hosted on GitBook at `https://chenzihong.gitbook.io/graphgen-cookbook/`, which renders client-side; fetching the HTML directly returns the page shell, not the article text, so treat the slugs below as a link map rather than summarized content [20]. Slugs found in that page's navigation: `graphgen-cookbook` (index), `graphgen-cookbook/quick_start`, `graphgen-cookbook/data-preparation`, `graphgen-cookbook/parameters`, `graphgen-cookbook/best-practice`, `graphgen-cookbook/progress`, `graphgen-cookbook/appendix` [20].
- The README separately links a DeepWiki-generated architecture writeup ("System Architecture") and a maintainer-authored "best practice" thread on GitHub issue #17 [1].
- Runnable references: one launcher pair per output format under `examples/generate/generate_<format>_qa/` - a `.sh` script plus a matching `_config.yaml` - covering `cot`, `atomic`, `aggregated`, `multi-hop`, `vqa`, `multi_choice`, `multi_answer`, `fill_in_blank`, and `true_false` [1]. Input-format examples sit under `examples/input_examples/` [19].
- Community layer: the README's own acknowledgements are prior-art/infra credits (LightRAG, ROGRAG, DB-GPT, SiliconFlow), not a curated tutorial list [1]; no separate community-tutorials page was found. No official MCP endpoint for these docs was found.
- Honest boundary: GraphGen ships no trainer, so it cannot run SFT, DPO, GRPO, or any other post-training method on its own output - pair it with a trainer library for that step [1]. The PyPI-vs-source split above (Install field) is the trap most likely to bite a first-time installer: `pip install graphg` silently gives the pre-Ray, pre-kuzu architecture, not the pipeline the current README quickstarts against [15][16].

## Sources

Ecosystem tools named only in passing (LLaMA-Factory, xtuner, Ray, kuzu, rocksdb, vLLM, SGLang, LightRAG, ROGRAG, DB-GPT) are reached through the README's own links [1] and are not separately enumerated. Unless a release tag/commit is named, all GitHub file reads are pinned to the screening commit `d9b8bedb5152499718e816d803668d40676f9781`; the Install field's claims about the `pip install graphg` code path are pinned separately to PyPI release `20250416` / git tag `20250422` (commit `e34135d7a1f64c07360652c9625f26a4bf8540cd`), which is what PyPI's dependency resolver actually serves - this is ahead of neither commit, it is a wholly separate, older release from the screening commit. All fetches were made 2026-08-12.

[1] GraphGen README (What is GraphGen, Latest Updates, Support List, Quick Start, System Architecture, Effectiveness of GraphGen, Acknowledgements sections). https://github.com/InternScience/GraphGen/blob/main/README.md, fetched from `raw.githubusercontent.com/InternScience/GraphGen/main/README.md`. Fetched 2026-08-12.

[2] InternScience/GraphGen repository metadata (license, former org name, current URL). https://github.com/InternScience/GraphGen via `api.github.com/repos/InternScience/GraphGen`. Fetched 2026-08-12.

[3] GraphGen `run.py` argument parser and console-script entry point at the screening commit. https://github.com/InternScience/GraphGen/blob/d9b8bedb5152499718e816d803668d40676f9781/graphgen/run.py. Fetched 2026-08-12.

[4] `graphg` package JSON API on PyPI (`info.version`, `info.requires_python`, `info.license`, and per-file `upload_time` for every release: `20250416` uploaded 2025-04-22T09:01:01, `0.1.0.post20250930` uploaded 2025-09-30T08:55:53). https://pypi.org/project/graphg/ via `pypi.org/pypi/graphg/json`. Fetched 2026-08-12.

[5] `graphg-20250416-py3-none-any.whl`, the actual wheel PyPI serves for `pip install graphg`, downloaded from `files.pythonhosted.org` and inspected directly (`*.dist-info/entry_points.txt` gives the console-script name; `*.dist-info/METADATA` gives `Requires-Dist`, absence of a `Requires-Python` field, and absence of a `License` classifier). https://files.pythonhosted.org/packages/3c/86/2e6207be3334d2ef068205fb69ee97faf70e1a33b65c26e17b052a3ef729/graphg-20250416-py3-none-any.whl. Fetched 2026-08-12.

[6] `graphgen/models/evaluator/qa/reward_evaluator.py` at the screening commit (full file read). https://github.com/InternScience/GraphGen/blob/d9b8bedb5152499718e816d803668d40676f9781/graphgen/models/evaluator/qa/reward_evaluator.py. Fetched 2026-08-12.

[7] `graphgen/models/evaluator/qa/` directory listing at the screening commit (file names only, not read). https://github.com/InternScience/GraphGen/tree/d9b8bedb5152499718e816d803668d40676f9781/graphgen/models/evaluator/qa. Fetched 2026-08-12.

[8] `graphgen/run.py` full body (logger setup, output-directory construction, `save_config`, single `engine.execute` call) at the screening commit. Same URL as [3]. Fetched 2026-08-12.

[9] GraphGen `.env.example` at the screening commit (backend selection, GPU/tensor-parallel knobs, ollama trainee limitation). https://github.com/InternScience/GraphGen/blob/d9b8bedb5152499718e816d803668d40676f9781/.env.example. Fetched 2026-08-12.

[10] GitHub Releases list for InternScience/GraphGen (two releases, latest `v0.1.0.post20250930` published 2025-09-30). https://github.com/InternScience/GraphGen/releases via `api.github.com/repos/InternScience/GraphGen/releases`. Fetched 2026-08-12.

[11] GitHub tags for InternScience/GraphGen, resolving `20250422` to commit `e34135d7a1f64c07360652c9625f26a4bf8540cd` and `v0.1.0.post20250930` to commit `fd6fa6b4c2e13de9f3bdad246eadb134b34ab085`. https://github.com/InternScience/GraphGen/tags via `api.github.com/repos/InternScience/GraphGen/tags`. Fetched 2026-08-12.

[12] `setup.py` at git tag `20250422` (commit `e34135d7a1f64c07360652c9625f26a4bf8540cd`): console-script entry point `graphgen=graphgen.generate:main`, Python 3.8-3.12 classifiers. https://github.com/InternScience/GraphGen/blob/20250422/setup.py. Fetched 2026-08-12.

[13] `graphgen/` package directory listing at git tag `20250422` (commit `e34135d7a1f64c07360652c9625f26a4bf8540cd`): `generate.py`/`graphgen.py`/`judge.py`/`evaluate.py`/`version.py`, no `engine.py` or `run.py`. https://github.com/InternScience/GraphGen/tree/20250422/graphgen. Fetched 2026-08-12.

[14] `requirements.txt` and `graphgen/generate.py` at git tag `20250422` (commit `e34135d7a1f64c07360652c9625f26a4bf8540cd`): dependency list has no kuzu/rocksdict/ray/torch/transformers entry; `generate.py` runs the flat-config `GraphGen.insert()/.quiz()/.judge()/.traverse()` chain via a single `--config_file` YAML and `SYNTHESIZER_*`/`TRAINEE_*` env-var OpenAI-style clients. https://github.com/InternScience/GraphGen/blob/20250422/requirements.txt and https://github.com/InternScience/GraphGen/blob/20250422/graphgen/generate.py. Fetched 2026-08-12.

[15] `graphgen/` package directory listing at the `v0.1.0.post20250930` release commit (`generate.py`/`graphgen.py`/`evaluate.py` entrypoints, bundled `configs/` dir - no `engine.py` or `run.py`). https://github.com/InternScience/GraphGen/tree/fd6fa6b4c2e13de9f3bdad246eadb134b34ab085/graphgen. Fetched 2026-08-12.

[16] `setup.py` at the `v0.1.0.post20250930` release tag (console-script entry point `graphg=graphgen.generate:main`, Python 3.10-3.12 classifiers), byte-identical at the screening commit. https://github.com/InternScience/GraphGen/blob/v0.1.0.post20250930/setup.py and https://github.com/InternScience/GraphGen/blob/d9b8bedb5152499718e816d803668d40676f9781/setup.py. Fetched 2026-08-12.

[17] Commit metadata for `d9b8bedb5152499718e816d803668d40676f9781` (author, date, message). https://github.com/InternScience/GraphGen/commit/d9b8bedb5152499718e816d803668d40676f9781 via `api.github.com/repos/InternScience/GraphGen/commits/d9b8bedb5152499718e816d803668d40676f9781`. Fetched 2026-08-12.

[18] `examples/generate/generate_aggregated_qa/generate_aggregated.sh` at the screening commit. https://github.com/InternScience/GraphGen/blob/d9b8bedb5152499718e816d803668d40676f9781/examples/generate/generate_aggregated_qa/generate_aggregated.sh. Fetched 2026-08-12.

[19] `examples/generate/generate_aggregated_qa/aggregated_config.yaml` at the screening commit (node-graph pipeline schema, `execution_params`, `ece` partition method comment, `generate` node's `method`/`data_format` params). https://github.com/InternScience/GraphGen/blob/d9b8bedb5152499718e816d803668d40676f9781/examples/generate/generate_aggregated_qa/aggregated_config.yaml. Fetched 2026-08-12.

[20] GraphGen Cookbook docs site navigation (page slugs only; body content is client-side rendered and not captured by this fetch). https://chenzihong.gitbook.io/graphgen-cookbook/. Fetched 2026-08-12.
