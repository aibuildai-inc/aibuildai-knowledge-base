# LazyLLM

A low-code Python framework for building multi-agent LLM applications that also wraps several third-party fine-tuning backends (LLaMA-Factory, Alpaca-LoRA, Collie, FlagEmbedding, and a GRPO/DAPO-capable EasyR1/verl fork) behind one chained `TrainableModule` API - post-training is a feature of an app-building tool, not the tool's primary purpose.

LazyLLM's own package description calls it "A Low-code Development Tool For Building Multi-agent LLMs Applications." [1]. It is built and maintained by LazyAGI [2][3], with `pyproject.toml`-listed authors on a `sensetime.com` domain [4]. Its core object is `TrainableModule`, which chains `.trainset()`, `.finetune_method()`, `.mode('finetune')` and `.deploy_method()` calls and executes them with a single `.update()` call, or `.start()` for inference-only deployment [5]. It lives at https://github.com/LazyAGI/LazyLLM [3].

**When to pick it**: you are already building a multi-agent or RAG application in LazyLLM and want to fine-tune the model that sits inside it without leaving the framework - `TrainableModule` chains fine-tune, deploy, and inference in one call and you never write a training script directly [5]. It is not a dedicated RL/post-training library like trl or verl: only SFT (via LLaMA-Factory) is documented and tutorial-supported [5][6]; GRPO/DAPO support exists in the code (an `EasyR1Finetune` class wrapping a verl fork) but is absent from every docs page checked, including the page that lists supported fine-tuning methods, so picking LazyLLM specifically for RL post-training means working from source, not docs (cross-reference: trl or verl directly for a documented RL path; not covered here).

**Methods it ships**: `lazyllm.finetune.*` exposes five functor classes confirmed in `lazyllm/components/finetune/__init__.py` at the screening commit [7]: `AlpacaloraFinetune` (wraps the tloen/alpaca-lora method, LoRA SFT), `CollieFinetune` (wraps the Collie framework), `LlamafactoryFinetune` (wraps LLaMA-Factory, default LoRA SFT), `FlagembeddingFinetune` (embedding-model fine-tuning), and `EasyR1Finetune`. Only `lazyllm.finetune.auto` (AutoFinetune), `lazyllm.finetune.llamafactory`, `lazyllm.finetune.collie`, and `lazyllm.finetune.flagembedding` appear in the docs' own "Supported Fine-tuning Methods" list on the LocalModel page - `AlpacaloraFinetune` is exported by the package but is absent from that list - and Collie is marked there as deprecated: "Collie has stopped iteration and will be removed in future versions" [6]. `EasyR1Finetune` runs `python -m verl.trainer.main` against a config whose default `algorithm.adv_estimator` is `grpo`, i.e. GRPO is the default algorithm; DAPO is reachable through a shipped `dapo.py` reward-function script plus `online_filtering`/`filter_low`/`filter_high` config knobs on that same GRPO-based run, not a separate trainer or `adv_estimator` value [8][9]. `EasyR1Finetune` is not mentioned on the docs' Best Practice, Tutorial, or API Reference/components pages checked for this card, and the extras description that names LazyLLM's "extra training tools" for the `full` install lists only AlpacaloraFinetune and CollieFinetune, omitting it as well [10][11] - the taxonomy of what's documented moves, so recheck the live `API Reference/components/` page [11] before relying on this list.

**Scale it handles**: single GPU is the default (`launchers.remote(ngpus=1, sync=True)` inside both `EasyR1Finetune` and `LlamafactoryFinetune`) [8][12]. Multi-GPU and multi-node are reached by passing a different launcher into `finetune_method`'s config dict, e.g. `launchers.sco(ngpus=8)` in the docs' own worked fine-tuning example [5]; the launcher classes are `EmptyLauncher` (local), `RemoteLauncher`, `SlurmLauncher` (srun, with `partition`, `nnode`, `nproc`, `ngpus` arguments - genuine multi-node), `ScoLauncher` (targets SenseCore, one of the platforms the README lists alongside Slurm and bare metal - the README does not state who operates SenseCore), and `K8sLauncher`, all documented on the live API Reference/launcher page [13][14][15]. This is documented mechanism with no published multi-node fine-tuning benchmark in any source checked for this card.

**Install**: `pip3 install lazyllm` for the minimal package, then `lazyllm install full` (a package-provided CLI command, not a pip extra) to pull every optional dependency [15]. PyPI's own release metadata for the current release, v1.2.2 (uploaded 2026-08-02), states `requires_python: "<3.13,>=3.10"` [16]; the `pyproject.toml` at the v1.2.2 tag itself instead declares `requires-python = ">=3.10,<3.14"` [4] - the two disagree by one minor version at the upper bound, and PyPI's is what governs `pip install` version resolution. Licence is Apache-2.0 per the GitHub repo's licence field [2]; PyPI's own licence metadata field is empty [16]. The bare-install core dependency list in `[project] dependencies` at v1.2.2 is minimal (fastapi>=0.111.0, pydantic>=2.11.7,<3.0.0, uvicorn>=0.23.2, and similar floors, 16 packages total, no ML framework) [4]; training-relevant hard pins live in the optional `[tool.poetry.dependencies]` table at the same tag: `transformers==4.57.1`, `peft==0.17.1`, `accelerate==1.6.0`, `vllm==0.10.1`, `lazyllm-llamafactory==0.9.4.dev2` (a LazyAGI-maintained fork of LLaMA-Factory, pulled in by the `full`, `finetune-all`, and `llama-factory` extras, but NOT by `standard`), and `lazyllm-verl==0.3.2.dev2` (a fork of verl, pulled in only by the `standard` or `full` extras, not by `finetune-all` or `llama-factory`) [4]. `torch` itself is an unpinned floor (`>=2.1.2`) [4]. `EasyR1Finetune.cmd()` requires the `trl` package via a runtime `thirdparty.check_packages(['verl', 'trl'])` call, but `trl` does not appear as a pinned dependency anywhere in `pyproject.toml`'s optional-dependency table - it is an undocumented extra install the user must add themselves [8]. Neither the installation-relevant files read for this card nor the README state a CUDA or GPU hardware minimum [4][15].

**Maintained by**: LazyAGI [2][3]; 3,858 GitHub stars per the shortlist row's screening snapshot (not a ranking signal); the repository shows active releases, with v1.2.2 published 2026-08-02 and v1.2.1 (the screening commit `28b6df1376f40f81974facdb9f206aad09c00fbc`) published 2026-07-31, one release behind [17]. Claims about `easyr1.py`, `dapo.py`, `easy_r1/config.yaml`, `llamafactory.py`, `sft.yaml`, and `base.py` below were read at the v1.2.1 tag/commit `28b6df1376f40f81974facdb9f206aad09c00fbc`, one release behind the v1.2.2 release the Install section describes [7][8][9][12][18][19].

## Quick start

The docs' own worked fine-tuning-to-deployment example, from the Fine-Tuning Tutorial page (Tutorial/9) [5]:

```python
import lazyllm
from lazyllm import finetune, deploy, launchers

model = lazyllm.TrainableModule(model_path) \
    .mode('finetune') \
    .trainset(train_data_path) \
    .finetune_method((finetune.llamafactory, {
        'learning_rate': 1e-4,
        'cutoff_len': 5120,
        'max_samples': 20000,
        'val_size': 0.01,
        'per_device_train_batch_size': 2,
        'num_train_epochs': 2.0,
        'launcher': launchers.sco(ngpus=8)
    })) \
    .prompt(dict(system='You are a helpful assistant.', drop_builtin_system=True)) \
    .deploy_method(deploy.Vllm)
model.evalset(eval_data)
model.update()
```

`.update()` runs fine-tuning, then deployment, then automatically infers over the evaluation set [5]. The inference-only form from the LocalModel best-practice page is `model = lazyllm.TrainableModule(model_path); model.start(); model("hello")` [6]. A CLI entry point exists for pre-built apps only (`lazyllm run chatbot --model=<name>`, `lazyllm run rag --documents=<path> --model=<name>`), not for launching a fine-tuning job [15].

**The deciding number**: in that same tutorial, fine-tuning InternLM2-Chat-7B via `finetune.llamafactory` (LoRA SFT) on the CMRC2018 Chinese reading-comprehension dataset (10,142 training questions) raised the exact-match rate on a 1,002-question held-out test set from 2.10% (21/1002) before fine-tuning to 39.72% (398/1002) after - a 37.62-point gain that also beat the much larger DeepSeek-V3 model's zero-shot 5.29% (53/1002) on the same test set; the same run raised "original text inclusion" (fraction of output words found in the source passage) from 5.19% to 94.91% [5].

## Start it

- One process, one GPU is the default: omit `launcher` in `finetune_method`'s config dict and both `EasyR1Finetune` and `LlamafactoryFinetune` fall back to `launchers.remote(ngpus=1, sync=True)` [8][12].
- More GPUs or nodes: pass a different launcher instance as the `launcher` key inside the same config dict, e.g. `launchers.sco(ngpus=8)` (the docs' own example [5]; `ScoLauncher` targets the SenseCore platform per the README [15]), or `launchers.slurm(partition=..., nnode=N, nproc=..., ngpus=...)` for srun-based multi-node, both documented on the live API Reference/launcher page alongside `EmptyLauncher`, `RemoteLauncher`, and `K8sLauncher` [13][14].
- Effective batch size and gradient accumulation are LLaMA-Factory/verl config keys passed straight through the `finetune_method` config dict (e.g. `per_device_train_batch_size`, shown as `2` in the tutorial example) [5]; LazyLLM does not compute or rename this arithmetic itself.
- For SFT via `finetune.llamafactory`, the shipped default config is `lazyllm/components/finetune/llama_factory/sft.yaml`: `finetuning_type: lora`, `lora_rank: 8`, `learning_rate: 1.0e-04`, `num_train_epochs: 3.0`, and `fp16: true` with `bf16: false` - a silent fp16-not-bf16 default that assumes fp16-friendly hardware unless the caller overrides it [18].
- For GRPO/DAPO via `finetune.easy_r1` (undocumented, see Methods it ships), the shipped default `easy_r1/config.yaml` sets `algorithm.adv_estimator: grpo`, `trainer.total_epochs: 15`, and points at a placeholder cluster model path and the `hiyouga/math12k` train/test split by default - both must be overridden for a real run [9].
- Out-of-memory first aid is not published by LazyLLM itself for either path in the sources checked for this card; the SFT config's own knobs to try are `per_device_train_batch_size` (down) and `lora_rank` (down), both plain LLaMA-Factory fields passed through unmodified [18].

## Watch it

This section is the mechanics only; what a logged value means for GRPO or DAPO training health is on those methods' own cards, not here.

- **SFT (`finetune.llamafactory`)**: the shipped default config sets `report_to: tensorboard` and `logging_steps: 10` [18] - TensorBoard is the default sink, not "nowhere," but only when the shipped default config is used unmodified; a caller who overrides `report_to` gets whatever LLaMA-Factory's own `report_to` field accepts. The exact per-step metric names are LLaMA-Factory's, not enumerated on any LazyLLM docs page checked for this card.
- **GRPO/DAPO (`finetune.easy_r1`)**: the shipped default config sets `trainer.logger: ["console"]` [9] - console-only by default, no tracker configured; metric names are verl's, not enumerated on any LazyLLM docs page checked for this card, since this path is undocumented (see Methods it ships).
- Stopping-rule search, run 2026-08-11 over the pages that would carry one: the LocalModel best-practice page [6], the Fine-Tuning Tutorial [5], the API Reference/components page [11], and the API Reference/configs page [20] (LazyLLM's runtime-configuration/environment-variable reference, which does document a `timeout` field, but scoped only to tracing-backend requests, not to training). None of the four publishes an RL or SFT stopping-rule, threshold, or resource-limit warning; "none found" is the result of this specific search, not a claim that no such field exists anywhere in the library.
- Sample-level generation logging and evaluation-during-training are handled at the `TrainableModule` level, not the trainer level: `.evalset(eval_data)` plus `.update()` runs inference over the evaluation set once training and deployment finish, and the tutorial's own comprehensive evaluation script computes exact-match, cosine-similarity (via a second `TrainableModule` loading `bge-large-zh-v1.5`), and text-inclusion scores over that evalset, saved to a JSON file (`eval/infer_true_cp.json`) [5]. This is a one-shot post-training evaluation, not periodic in-loop evaluation with a configurable cadence field documented anywhere checked for this card.

## Save it

- `LazyLLMFinetuneBase.__call__` - the shared base every finetune functor implements - returns `self.merge_path` if a merge path was set, otherwise `self.target_path`; this is the value the rest of the `TrainableModule` chain (deployment, inference) consumes as "the trained model" [19].
- `LlamafactoryFinetune.cmd()` automatically runs `llamafactory-cli export` after a successful LoRA training run, merging the LoRA adapter into a full model at a separate `merge_path`, and for full (non-LoRA) fine-tuning instead rsyncs the trained files directly - so the object handed to deployment is a full merged model either way, unlike bare trl/peft workflows where LoRA merging is a separate manual step [12].
- Checkpoint retention and save cadence for the SFT path are plain LLaMA-Factory config fields passed through the same `sft.yaml`/config dict: `save_steps: 500`, `save_total_limit: null` (unlimited retention by default), `save_only_model: false` (optimizer/scheduler state kept, so resuming training is possible) in the shipped default [18].
- For GRPO/DAPO, the shipped default `easy_r1/config.yaml` sets `trainer.save_freq: 5` and `trainer.save_limit: 3` (keeps at most 3 checkpoints) [9]; LazyLLM's own code does not document a resume call for this path in the sources checked for this card - resuming is whatever `python -m verl.trainer.main` itself supports, invoked by `EasyR1Finetune.cmd()` [8].
- Loader handoff: whether an external evaluator can load the saved directory directly follows from the merge behavior above - the LlamafactoryFinetune path yields a full merged-model directory, not a bare LoRA adapter directory, so a standard `from_pretrained()`-style load is expected to work without pairing it with a base model; this card did not fetch a shared loading-contract reference to confirm the loader side independently.

## Find it in the docs

The docs are the live source; this section teaches the lookup, it does not mirror the content.

- Address pattern: `https://docs.lazyllm.ai/en/<version>/<Section>/<page>/`, where `<version>` is `stable` or `latest` (both fetched and confirmed working for this card; no version-pinned numeric tag form like `/en/v1.2.2/` was tested) [6][11]. `<Section>` uses title case with spaces that must be percent-encoded as `%20` in a raw URL (e.g. `Best%20Practice`, `API%20Reference`) [6][13].
- Question-to-page map: how to chain fine-tune/deploy/inference on one model -> `Best%20Practice/LocalModel/` [6]; the full worked fine-tuning example with evaluation -> `Tutorial/9/` (titled "Domain-Specific Fine-Tuning") [5]; class-level docs for any finetune functor or launcher -> `API%20Reference/components/` and `API%20Reference/launcher/` respectively [11][13].
- The docs site's `Tutorial/` index page lists the fine-tuning tutorial directly, as "Chapter 9: Fine-Tuning in Practice — Help Large Models and Embedding Models Better Understand Your Domain", alongside its mostly RAG/agent-focused chapters; the LocalModel best-practice page also links to it in-page [5][6][21].
- Runnable references beyond the docs: the Fine-Tuning Tutorial page links out to GitHub-hosted code for its data-preparation and evaluation scripts (labelled "Code GitHub link" inline) [5]; its worked example trains on the CMRC2018 dataset, which the tutorial describes as consisting of nearly 15,000 real-world questions annotated by human experts on Wikipedia paragraphs, used to build Chinese reading-comprehension information-extraction capability [5].
- No dedicated community-tutorials page (curated third-party posts) was found on the docs site among the pages checked for this card (Home, Best Practice, API Reference, Tutorial sections; a RoadMap section link is visible in the site nav but its page was not fetched for this card); the closest thing is the numbered in-house Tutorial series itself (Tutorial/1 through at least Tutorial/9, per the homepage nav) [22].
- No official MCP endpoint for querying LazyLLM's own docs was found in the pages checked for this card; `mcp` appears only as a floor-pinned runtime dependency (`mcp>=1.7.0`) for LazyLLM's own agent features, not as a docs-query tool [4].
- Honest boundary: RL/GRPO/DAPO fine-tuning (`EasyR1Finetune`) is implemented and exported in code but undocumented everywhere checked (see Methods it ships) - a reader who wants that path is working from source, not docs, and should expect the docs' own "Supported Fine-tuning Methods" and extras-description text to undersell what the package can do [6][10][11]. No maintainer reply in a closed GitHub issue was found describing a specific trap for either fine-tuning path; none is claimed here.

## Sources

All pages are `main`-branch docs or live pages read on 2026-08-11 unless a commit or tag is named. Method names (SFT, GRPO, DAPO) are deliberately cited to nothing here; their defining papers live on the methodology cards, not this one.

[1] LazyLLM on PyPI (package summary/description). https://pypi.org/project/lazyllm/. Fetched 2026-08-11.

[2] LazyAGI/LazyLLM GitHub repository API metadata (licence, description, stars, archived/fork status, timestamps). https://api.github.com/repos/LazyAGI/LazyLLM. Fetched 2026-08-11.

[3] LazyAGI/LazyLLM GitHub repository. https://github.com/LazyAGI/LazyLLM. Fetched 2026-08-11.

[4] `pyproject.toml` at the v1.2.2 tag (authors, requires-python, `[project] dependencies`, `[tool.poetry.dependencies]` version pins, `[tool.poetry.extras]` groups, `[tool.lazyllm.extras_descriptions]`). https://raw.githubusercontent.com/LazyAGI/LazyLLM/v1.2.2/pyproject.toml. Fetched 2026-08-11.

[5] LazyLLM docs, Tutorial/9 "Domain-Specific Fine-Tuning" (the fine-tuning-to-deployment quick start, launcher example, and before/after CMRC2018 evaluation numbers). https://docs.lazyllm.ai/en/stable/Tutorial/9/. Fetched 2026-08-11.

[6] LazyLLM docs, Best Practice/LocalModel (TrainableModule usage, the docs' own "Supported Fine-tuning Methods" list, Collie deprecation notice). https://docs.lazyllm.ai/en/stable/Best%20Practice/LocalModel/. Fetched 2026-08-11.

[7] `lazyllm/components/finetune/__init__.py` at commit `28b6df1376f40f81974facdb9f206aad09c00fbc` (exported finetune functor classes). https://raw.githubusercontent.com/LazyAGI/LazyLLM/28b6df1376f40f81974facdb9f206aad09c00fbc/lazyllm/components/finetune/__init__.py. Fetched 2026-08-11.

[8] `lazyllm/components/finetune/easyr1.py` at commit `28b6df1376f40f81974facdb9f206aad09c00fbc` (`EasyR1Finetune` implementation: package requirements, default launcher, `verl.trainer.main` command construction). https://raw.githubusercontent.com/LazyAGI/LazyLLM/28b6df1376f40f81974facdb9f206aad09c00fbc/lazyllm/components/finetune/easyr1.py. Fetched 2026-08-11.

[9] `lazyllm/components/finetune/easy_r1/config.yaml` and `easy_r1/reward_function/dapo.py` at commit `28b6df1376f40f81974facdb9f206aad09c00fbc` (default `adv_estimator: grpo`, DAPO filter knobs, `trainer.logger`, `save_freq`, `save_limit`, `total_epochs`). https://raw.githubusercontent.com/LazyAGI/LazyLLM/28b6df1376f40f81974facdb9f206aad09c00fbc/lazyllm/components/finetune/easy_r1/config.yaml and .../easy_r1/reward_function/dapo.py. Fetched 2026-08-11.

[10] `[tool.lazyllm.extras_descriptions]` table in [4] (the `full` extra's description naming "extra training tools").

[11] LazyLLM docs, API Reference/components (documented Finetune classes: AlpacaloraFinetune, CollieFinetune, LlamafactoryFinetune, FlagembeddingFinetune, AutoFinetune, DummyFinetune, LazyLLMFinetuneBase; no EasyR1Finetune entry, in both the `stable` and `latest` builds checked). https://docs.lazyllm.ai/en/stable/API%20Reference/components/ and https://docs.lazyllm.ai/en/latest/API%20Reference/components/. Fetched 2026-08-11.

[12] `lazyllm/components/finetune/llamafactory.py` at commit `28b6df1376f40f81974facdb9f206aad09c00fbc` (default launcher, `cmd()` method including the post-training `llamafactory-cli export` merge step). https://raw.githubusercontent.com/LazyAGI/LazyLLM/28b6df1376f40f81974facdb9f206aad09c00fbc/lazyllm/components/finetune/llamafactory.py. Fetched 2026-08-11.

[13] LazyLLM docs, API Reference/launcher (`LazyLLMLaunchersBase`, `EmptyLauncher`, `RemoteLauncher`, `SlurmLauncher`, `ScoLauncher`, `K8sLauncher`, all confirmed present in the rendered page). https://docs.lazyllm.ai/en/stable/API%20Reference/launcher/. Fetched 2026-08-11.

[14] `lazyllm/launcher/slurm.py` at commit `28b6df1376f40f81974facdb9f206aad09c00fbc` (`SlurmLauncher.__init__` signature: `partition`, `nnode`, `nproc`, `ngpus`, `timeout`). https://raw.githubusercontent.com/LazyAGI/LazyLLM/28b6df1376f40f81974facdb9f206aad09c00fbc/lazyllm/launcher/slurm.py. Fetched 2026-08-11.

[15] LazyLLM README at the v1.2.2 tag (Installation section: `pip3 install lazyllm`, `lazyllm install full`, source install, and the `lazyllm run chatbot`/`lazyllm run rag` CLI forms). https://raw.githubusercontent.com/LazyAGI/LazyLLM/v1.2.2/README.md. Fetched 2026-08-11.

[16] PyPI JSON API for `lazyllm` (release version, `requires_python`, licence metadata field, upload date). https://pypi.org/pypi/lazyllm/json. Fetched 2026-08-11.

[17] GitHub Releases API for LazyAGI/LazyLLM (v1.2.2 and v1.2.1 publish dates) and Tags API (tag-to-commit mapping confirming `v1.2.1` = `28b6df1376f40f81974facdb9f206aad09c00fbc`). https://api.github.com/repos/LazyAGI/LazyLLM/releases and https://api.github.com/repos/LazyAGI/LazyLLM/tags. Fetched 2026-08-11.

[18] `lazyllm/components/finetune/llama_factory/sft.yaml` at commit `28b6df1376f40f81974facdb9f206aad09c00fbc` (default SFT config: `finetuning_type`, `lora_rank`, `learning_rate`, `num_train_epochs`, `fp16`/`bf16`, `report_to`, `logging_steps`, `save_steps`, `save_total_limit`, `save_only_model`). https://raw.githubusercontent.com/LazyAGI/LazyLLM/28b6df1376f40f81974facdb9f206aad09c00fbc/lazyllm/components/finetune/llama_factory/sft.yaml. Fetched 2026-08-11.

[19] `lazyllm/components/finetune/base.py` at commit `28b6df1376f40f81974facdb9f206aad09c00fbc` (`LazyLLMFinetuneBase.__call__` merge_path/target_path contract). https://raw.githubusercontent.com/LazyAGI/LazyLLM/28b6df1376f40f81974facdb9f206aad09c00fbc/lazyllm/components/finetune/base.py. Fetched 2026-08-11.

[20] LazyLLM docs, API Reference/configs page (runtime configuration/environment-variable reference, including the `timeout` field for tracing-consume-backend requests). https://docs.lazyllm.ai/en/stable/API%20Reference/configs/. Fetched 2026-08-11.

[21] LazyLLM docs, Tutorial/ index page (in-body chapter list including the Chapter 9 fine-tuning link). https://docs.lazyllm.ai/en/stable/Tutorial/. Fetched 2026-08-11.

[22] LazyLLM docs homepage nav (Tutorial/1 through Tutorial/19 links enumerated in the site navigation). https://docs.lazyllm.ai/en/stable/. Fetched 2026-08-11.
