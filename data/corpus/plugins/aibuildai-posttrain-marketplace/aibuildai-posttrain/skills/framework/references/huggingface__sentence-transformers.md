# sentence-transformers

Hugging Face's library for embedding, reranker, and sparse-encoder models: one `Trainer` per model family, built on the same `transformers.Trainer` base as trl.

**sentence-transformers** describes itself as "an easy method to compute embeddings for accessing, using, and training state-of-the-art embedding and reranker models," covering three model families through one Hub-native API: `SentenceTransformer` (dense embeddings / bi-encoders), `CrossEncoder` (rerankers, a.k.a. cross-encoders), and `SparseEncoder` (sparse embeddings) [1]. It is built and maintained by Hugging Face, with Nils Reimers and Tom Aarsen as the listed authors and Tom Aarsen the current maintainer [2]. Each family exposes a `Trainer` class (`SentenceTransformerTrainer`, `CrossEncoderTrainer`, `SparseEncoderTrainer`) that all subclass a shared `BaseTrainer(Trainer, ABC)`, itself a `transformers.Trainer` subclass, and take a model, a Hub dataset, and a loss object into `.train()` [3]. It lives at https://github.com/huggingface/sentence-transformers [4].

**When to pick it**: fine-tuning or training embedding, reranker, or sparse-retrieval models specifically, as opposed to trl's generative-LM post-training methods (SFT, DPO, GRPO, ...); both frameworks subclass `transformers.Trainer` [3]. Pick something else if you need decoder-only RLHF-style methods; this library's method menu is embedding/ranking-specific (below).

**Methods it ships**: not RL algorithms but loss families per model type, one `Trainer` and `TrainingArguments` class per family, all inheriting from shared `Base*` classes [3]. Sentence-Transformer (dense) losses include `MultipleNegativesRankingLoss`, `CoSENTLoss` (the `SentenceTransformerTrainer`'s fallback when no loss is passed, per its `get_default_loss()` [23]), `ContrastiveLoss`/`OnlineContrastiveLoss`, `TripletLoss`, `MatryoshkaLoss`/`Matryoshka2dLoss`/`AdaptiveLayerLoss`, `GISTEmbedLoss`, and cached (gradient-cache) variants `CachedMultipleNegativesRankingLoss`, `CachedGISTEmbedLoss`, `MegaBatchMarginLoss` [6]. Cross-Encoder losses mirror a reranker-appropriate subset (`BinaryCrossEntropyLoss`, `CrossEntropyLoss`, `LambdaLoss`, `RankNetLoss`, its own `CachedMultipleNegativesRankingLoss`) and Sparse-Encoder losses add `SpladeLoss`/`CachedSpladeLoss`, `FlopsLoss`, `CSRLoss` alongside sparse counterparts of the dense losses [6]. The cached losses implement GradCache, described in the docs as solving the problem that in-batch-negative contrastive losses "cannot work either" with naive batch-scaling because the datapoints in a batch are non-independent; they embed in mini-batches (`mini_batch_size`, default 32, or the newer `mini_batch_num_tokens`) and replay cached gradients to hold memory constant at large batch sizes [7]. This taxonomy is the live package-reference page, not a fixed list - recheck it at [6].

**Scale it handles**: single GPU is the base case; multi-GPU is Data Parallel (DP) or Distributed Data Parallel (DDP) via `torchrun`/`accelerate launch`, with a published throughput comparison; FSDP (Fully Sharded Data Parallel) is explicitly flagged as "not fully supported" - the docs list concrete limitations (no evaluator support during FSDP training, a special save path, and required `fsdp`/`fsdp_config` launch arguments) [8]. No multi-node launch form or benchmark is published on this page - only the single-node DP/DDP/FSDP comparison below.

**Install**: `pip install -U sentence-transformers`; version 5.7.0 released 2026-08-06, resolved at the `v5.7.0` tag to commit `b2a9529cf6312d2b2a8ffa2b64d82fabc1571bd8` [9][10]. Python >=3.10; Apache-2.0 [10][2]. Core pins at that tag/release: `transformers>=4.41.0,<6.0.0`, `torch>=1.11.0` (unpinned upper bound), `huggingface-hub>=0.23.0`, `tokenizers>=0.19`; the `[train]` extra adds `datasets>=2.0.0` and `accelerate>=0.20.3`; other extras (`image`, `audio`, `video`, `onnx`, `onnx-gpu`, `openvino`, `dev`) are named in `pyproject.toml` at that tag [10]. No CUDA or hardware minimum is stated anywhere in the installation docs beyond "you must install PyTorch with CUDA support" and a link to PyTorch's own install page [11]. The shortlist screening commit, `5de24293729a8905c2b0baf7bda76a130b221e31` (pushed 2026-07-31), predates the v5.7.0 release commit above by six days and sits between the v5.6.1 (2026-07-23) and v5.7.0 (2026-08-06) tags [12][13] - the reverse of the usual case: any card claim sourced from that commit's code - [3] (`base/trainer.py`, cited in the opening paragraph, When to pick it, and Methods it ships) and [23] (`sentence_transformer/trainer.py`, cited in Methods it ships), plus the trainer and model-card files cited in Watch it/Save it below - is therefore for a pre-5.7.0 development snapshot, not for what `pip install sentence-transformers` delivers today, and 5.7.0 rewrote the cached-loss internals cited above [7][14].

**Maintained by**: Hugging Face, with Tom Aarsen as the current listed maintainer [2]; the v5.7.0 release (2026-08-06) describes itself as "a correctness and performance-focused release" that rebuilt every gradient-cached loss on one shared engine after finding several bugs that "silently corrupted gradients" - for example, Cross-Encoder `CachedMultipleNegativesRankingLoss` on GPU used different dropout masks in its forward and backward passes, silently biasing gradients for any reranker trained with dropout on CUDA or MPS [14].

## Quick start

Inference, from the README (a complete program once a model name is filled in) [1]:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
similarities = model.similarity(embeddings, embeddings)
```

Training, the full end-to-end script from the Sentence Transformer training-overview page [5]:

```python
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator

model = SentenceTransformer(
    "microsoft/mpnet-base",
    model_card_data=SentenceTransformerModelCardData(
        language="en", license="apache-2.0", model_name="MPNet base trained on AllNLI triplets",
    ),
)
dataset = load_dataset("sentence-transformers/all-nli", "triplet")
train_dataset = dataset["train"].select(range(100_000))
eval_dataset = dataset["dev"]
loss = MultipleNegativesRankingLoss(model)
args = SentenceTransformerTrainingArguments(
    output_dir="models/mpnet-base-all-nli-triplet",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=0.1,
    fp16=True,
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet",
)
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"], positives=eval_dataset["positive"], negatives=eval_dataset["negative"],
    name="all-nli-dev",
)
trainer = SentenceTransformerTrainer(
    model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
    loss=loss, evaluator=dev_evaluator,
)
trainer.train()
model.save_pretrained("models/mpnet-base-all-nli-triplet/final")
model.push_to_hub("mpnet-base-all-nli-triplet")
```

`CrossEncoder`/`SparseEncoder` follow the same shape with their own `Trainer`/`TrainingArguments`/loss classes [6].

## Start it

- One GPU is the plain script above.
- Multiple GPUs on one node: DP or DDP, both launched with the same script via `torchrun` or `accelerate launch`; the docs note the launch entry point must sit behind `if __name__ == "__main__":` for these to work, and that the evaluator only runs on a single device even under multi-GPU training [8]. A published single-node comparison (samples/sec, on the same hardware) shows DDP as the strongest option: No Parallelism 2724, DP 3675 (1.349x), DDP 6980 (2.562x) [8].
- FSDP is available but explicitly "not fully supported": it measured 5782 samples/sec (2.122x) in the same comparison - faster than no parallelism but slower than DDP - and carries three concrete limitations the docs spell out: the `Evaluator` does not work during FSDP training, checkpoint saving needs a different procedure, and launching requires passing `fsdp` and `fsdp_config` arguments [8]. A closed GitHub issue (#3023, opened 2024) shows two distinct FSDP problems in one thread: a contributor (association CONTRIBUTOR) first hit `RuntimeError: 'weight' must be 2-D` while training with FSDP, which they traced to a model-wrapping condition in the trainer's `compute_loss` that skipped a required model-in-loss update; after working around that condition, the same contributor separately reported that "the sample code still doesn't work because of the evaluator," and the maintainer (Tom Aarsen, repo member) confirmed "the evaluator is too naive for FSDP," linking the same docs section - the RuntimeError and the evaluator limitation are separate failures, not one [15].
- The effective batch is per-device batch x device count x gradient accumulation steps, the same arithmetic as any `transformers.Trainer`-based library; the quickstart script above sets `per_device_train_batch_size=16` with no accumulation.
- `SentenceTransformerTrainingArguments` extends `transformers.TrainingArguments`/a shared `BaseTrainingArguments` with library-specific fields: `prompts` (per-column prompt injection), `batch_sampler` (default `BatchSamplers.BATCH_SAMPLER`; the quickstart overrides it to `NO_DUPLICATES` because `MultipleNegativesRankingLoss` needs no duplicate samples in a batch), `multi_dataset_batch_sampler` (default `MultiDatasetBatchSamplers.PROPORTIONAL`, for training on several datasets/losses at once), `router_mapping`, and `learning_rate_mapping` [16]. It does not change the `fp16`/`bf16` defaults from base `transformers.TrainingArguments` (both default `False`), so the quickstart's `fp16=True` is an explicit per-run choice, not a library default [16].
- Out-of-memory first aid is the cached-loss family itself, not a training-argument flag: swap a large-batch contrastive loss (e.g. `MultipleNegativesRankingLoss`) for its `Cached*` counterpart (`CachedMultipleNegativesRankingLoss`, `CachedGISTEmbedLoss`, `CachedSpladeLoss` for sparse models), which trains at the same effective batch size in mini-batches of `mini_batch_size` (default 32) or, as of the v5.7.0 rework, a token-based `mini_batch_num_tokens`, replaying cached gradients so memory stays roughly constant [7][14].

## Watch it

Mechanics only - what a metric means for training health is out of scope here.

- **Enable it**: no logging destination is on by default. The docs list three optional `transformers.TrainerCallback` integrations that activate automatically when the package is installed: `WandbCallback` (installing `wandb`), `TensorBoardCallback` (installing `tensorboard`), and `CodeCarbonCallback` (installing `codecarbon`, which also feeds carbon-emission numbers into the auto-generated model card) [17][11]. The install page's own warning: "Don't forget to add the module names to `report_to` in the Training Arguments when training, or they will not be used" [11].
- **Metric names are evaluator-defined, not trainer-fixed**: during training the `Trainer.evaluate()` machinery prefixes whatever dict an `Evaluator` callable returns with `eval_` (collapsing a per-dataset `eval_<name>_` prefix down to plain `eval_` when only one eval dataset is active), so the concrete keys come from which `Evaluator` you pass [18]. Example from the `EmbeddingSimilarityEvaluator` docs: constructing it with `name="sts_dev"` gives `evaluator.primary_metric == "sts_dev_pearson_cosine"`, which becomes the logged key `eval_sts_dev_pearson_cosine` under the trainer's prefixing [19]. `BinaryClassificationEvaluator` similarly names its primary metric after its own `name` argument (docs example: `"quora_duplicates_dev_cosine_ap"`) [19].
- **A logging mechanic independent of any tracker**: a `BaseModelCardCallback(TrainerCallback)`, attached automatically to every trainer, records a training-loss (and, when evaluating, validation-loss) row per step into the model's `model_card_data.training_logs` on `on_log`/`on_evaluate`, regardless of whether `report_to` is set - so the generated model card carries a loss table even for a run with no external tracker configured [20]. It also auto-tags the card `generated_from_trainer`, records the loss class(es) in use, and pulls a few widget examples from the eval or train dataset [20].
- **Evaluate during training**: pass `evaluator=` to the `Trainer` (as in the quickstart) and set the standard `transformers` fields `eval_strategy`/`eval_steps` on the `TrainingArguments`; the evaluator itself is a separate callable (`TripletEvaluator`, `EmbeddingSimilarityEvaluator`, `InformationRetrievalEvaluator`, `NanoBEIREvaluator`, `BinaryClassificationEvaluator`, `RerankingEvaluator`, and sparse/cross-encoder counterparts) run at that cadence, not a metric computed from the loss [19].
- **Stopping**: no RL-style stopping rule applies here since these are supervised contrastive/classification losses, not RL objectives; the docs read for this card (installation, training-overview, distributed-training, and the loss/evaluator package-reference pages [1][5][8][11][17][19]) publish no early-stopping threshold specific to this library. `transformers.EarlyStoppingCallback` (patience/threshold on a tracked metric) is available generically through the shared `Trainer` base but is not sentence-transformers-specific and not documented on any of the pages above.

## Save it

- `trainer.train()` writes numbered checkpoints under the `TrainingArguments.output_dir` on the same `save_strategy`/`save_steps`/`save_total_limit` fields as base `transformers.TrainingArguments` [16]; internally `_save()` calls `self.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)` and also writes the processing class and the `TrainingArguments` object into that directory [3].
- Final save/reload: `model.save_pretrained("path")` at the end of the quickstart script, and `model.push_to_hub("repo_name")` to upload it [5]; a saved directory reloads with the same `SentenceTransformer(...)`/`CrossEncoder(...)`/`SparseEncoder(...)` constructor used to load any pretrained model.
- Resume: this card found no sentence-transformers-specific resume argument beyond the base `transformers.Trainer.train(resume_from_checkpoint=...)` surface, which `BaseTrainer` inherits; `_load_from_checkpoint()` reconstructs the model from the checkpoint path and copies its `state_dict` into the live model [3].
- The trainer's checkpoint push path (`_push_from_checkpoint`) explicitly skips re-uploading optimizer/scheduler/RNG-state files and standard weight/config files already covered by the base `transformers` checkpoint logic, layering sentence-transformers-specific files (`modules.json`, `README.md`, per-module subfolders) on top [3]; PEFT adapter files (`adapter_config.json`/weights) are excluded from that push list the same way when `peft` is installed, but this card found no sentence-transformers-specific text describing adapter-only saves the way trl documents them - treat adapter saving here as inherited PEFT/`transformers` behavior, not a library-specific contract.
- Loader handoff: this card did not verify whether the saved directory format is accepted by any specific external evaluator; check that evaluator's own loading contract separately.

## Find it in the docs

- Address pattern: `https://sbert.net/docs/<page>.html`, no version-tag prefix - checked 2026-08-10: `sbert.net/docs/installation.html` returns 200, while `sbert.net/v5.7.0/docs/...`, `sbert.net/en/stable/docs/...`, and `sbert.net/en/latest/docs/...` all return 404, so the docs are unversioned/latest-only, built with Sphinx and Read the Docs branding [11][21]. There is no way to pin the rendered docs to the release you installed; only the source `.rst`/code at a git tag can be pinned (as done above for the install pins).
- Page-slug recipes: `docs/installation.html`; `docs/quickstart.html` (per-family anchors `#sentence-transformer`, `#cross-encoder`, `#sparse-encoder`); `docs/sentence_transformer/training_overview.html` (and the `cross_encoder`/`sparse_encoder` siblings) for the full Trainer walkthrough; `docs/sentence_transformer/training/distributed.html` for DP/DDP/FSDP; `docs/package_reference/sentence_transformer/losses.html`, `.../evaluation.html`, `.../training_args.html` for the class-level API reference (and their `cross_encoder`/`sparse_encoder` counterparts) [1][5][8][6][16][19].
- Question-to-slug map: "which loss for my data shape" and multi-dataset training -> the relevant `training_overview.html` page's Loss Function / Multi-Dataset Training sections [5]; "how do I scale this" -> `training/distributed.html` [8]; "what does the trainer log" -> the loss/evaluator package-reference pages, since there is no single dedicated metrics page - metric names are defined by whichever `Evaluator` class you pick [19].
- Runnable references beyond the docs: the `examples/` tree in the GitHub repo, split into `sentence_transformer/`, `cross_encoder/`, and `sparse_encoder/` subdirectories at the v5.7.0 tag [22]; the quickstart's own datasets (`sentence-transformers/all-nli`) and models (`sentence-transformers/all-MiniLM-L6-v2`, `microsoft/mpnet-base`) are ready-made smoke tests [1][5].
- Known boundary, stated where it bites: FSDP training is explicitly unsupported for evaluation (see Start it) [8][15], and the v5.7.0 release notes carry a forward-looking deprecation - "loading models whose modules import classes from outside `sentence_transformers` will require `trust_remote_code=True` from v6.0" [14] - relevant to anyone loading custom-module community checkpoints.
- Community layer and MCP endpoint: this card did not fetch a curated community-tutorials page or confirm an official MCP endpoint for these docs; unlike the trl card, no such page was located during this session's research, so none is cited here.

## Sources

All pages fetched 2026-08-10 unless a git tag/commit is named. Method names for losses are cited to the package-reference docs, not to their original papers, since methodology-level citations belong on separate method cards; the two loss papers named inside a direct quote ([7]) are cited only because the quote names them.

[1] sentence-transformers README (GitHub, master branch). https://raw.githubusercontent.com/huggingface/sentence-transformers/master/README.md. Fetched 2026-08-10.

[2] sentence-transformers on PyPI (JSON API), version 5.7.0. https://pypi.org/pypi/sentence-transformers/json. Fetched 2026-08-10.

[3] `sentence_transformers/base/trainer.py` at commit 5de24293729a8905c2b0baf7bda76a130b221e31 (the shortlist screening commit, 2026-07-31). https://raw.githubusercontent.com/huggingface/sentence-transformers/5de24293729a8905c2b0baf7bda76a130b221e31/sentence_transformers/base/trainer.py. Fetched 2026-08-10.

[4] sentence-transformers GitHub repository (API metadata). https://api.github.com/repos/huggingface/sentence-transformers. Fetched 2026-08-10.

[5] Sentence Transformer training-overview page (Trainer end-to-end quickstart script). https://sbert.net/docs/sentence_transformer/training_overview.html. Fetched 2026-08-10.

[6] sentence-transformers package-reference losses page (loss taxonomy across all three model families). https://sbert.net/docs/package_reference/sentence_transformer/losses.html. Fetched 2026-08-10.

[7] `CachedMultipleNegativesRankingLoss` docstring, on the same losses page as [6] (GradCache mechanism, `mini_batch_size`/`mini_batch_num_tokens` parameters). https://sbert.net/docs/package_reference/sentence_transformer/losses.html. Fetched 2026-08-10.

[8] Sentence Transformer distributed-training page (DP/DDP/FSDP launch forms, throughput comparison table, FSDP limitations). https://sbert.net/docs/sentence_transformer/training/distributed.html. Fetched 2026-08-10.

[9] sentence-transformers GitHub releases API (latest release, v5.7.0). https://api.github.com/repos/huggingface/sentence-transformers/releases/latest. Fetched 2026-08-10.

[10] `pyproject.toml` at the v5.7.0 tag (dependency floors, extras, Python floor, license). https://raw.githubusercontent.com/huggingface/sentence-transformers/v5.7.0/pyproject.toml. Fetched 2026-08-10.

[11] Installation page (extras, tracker install guidance and the `report_to` warning, CUDA statement). https://sbert.net/docs/installation.html. Fetched 2026-08-10.

[12] GitHub API commit lookup for the shortlist commit 5de24293729a8905c2b0baf7bda76a130b221e31 (date 2026-07-31T08:19:35Z). https://api.github.com/repos/huggingface/sentence-transformers/commits/5de24293729a8905c2b0baf7bda76a130b221e31. Fetched 2026-08-10.

[13] GitHub API tags/releases list (v5.6.1, v5.6.0, v5.5.1, v5.5.0, v5.7.0 with dates and commit SHAs). https://api.github.com/repos/huggingface/sentence-transformers/releases and https://api.github.com/repos/huggingface/sentence-transformers/tags. Fetched 2026-08-10.

[14] v5.7.0 release notes body (GradCache overhaul, the dropout-mask gradient bug, the breaking-change and deprecation notices). https://api.github.com/repos/huggingface/sentence-transformers/releases/latest. Fetched 2026-08-10.

[15] GitHub issue #3023, "FSDP Training with Sentence Transformer" (maintainer Tom Aarsen, association MEMBER, replies dated 2024-10-28 through 2025-03-28 in the issue thread). https://github.com/huggingface/sentence-transformers/issues/3023 and its comments API https://api.github.com/repos/huggingface/sentence-transformers/issues/3023/comments. Fetched 2026-08-10.

[16] `SentenceTransformerTrainingArguments` package-reference page (full field list and defaults, including `batch_sampler`, `multi_dataset_batch_sampler`, `fp16`/`bf16` defaults). https://sbert.net/docs/package_reference/sentence_transformer/training_args.html. Fetched 2026-08-10.

[17] Training-overview page, Callbacks section (WandbCallback/TensorBoardCallback/CodeCarbonCallback integration). https://sbert.net/docs/sentence_transformer/training_overview.html. Fetched 2026-08-10.

[18] `sentence_transformers/base/trainer.py` at the shortlist commit, `evaluation_loop()` (the `eval_`/`eval_<name>_` metric-key prefixing logic). https://raw.githubusercontent.com/huggingface/sentence-transformers/5de24293729a8905c2b0baf7bda76a130b221e31/sentence_transformers/base/trainer.py. Fetched 2026-08-10.

[19] Sentence Transformer package-reference evaluation page (`EmbeddingSimilarityEvaluator`, `BinaryClassificationEvaluator`, `TripletEvaluator`, and the rest, with `primary_metric` naming examples). https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html. Fetched 2026-08-10.

[20] `sentence_transformers/base/model_card.py` at the shortlist commit, `BaseModelCardCallback` (`on_init_end`/`on_log`/`on_evaluate`, the always-on `training_logs` mechanic). https://raw.githubusercontent.com/huggingface/sentence-transformers/5de24293729a8905c2b0baf7bda76a130b221e31/sentence_transformers/base/model_card.py. Fetched 2026-08-10.

[21] Direct fetch of candidate versioned-docs URL forms (`/v5.7.0/docs/...`, `/en/stable/docs/...`, `/en/latest/docs/...`) alongside the working unversioned form, to confirm the address pattern. Checked against https://sbert.net/docs/installation.html. Fetched 2026-08-10.

[22] GitHub API contents listing of the `examples/` directory at the v5.7.0 tag. https://api.github.com/repos/huggingface/sentence-transformers/contents/examples?ref=v5.7.0. Fetched 2026-08-10.

[23] `sentence_transformers/sentence_transformer/trainer.py` at the shortlist commit (`SentenceTransformerTrainer.get_default_loss()`, which instantiates `CoSENTLoss` as the fallback when no loss is passed). https://raw.githubusercontent.com/huggingface/sentence-transformers/5de24293729a8905c2b0baf7bda76a130b221e31/sentence_transformers/sentence_transformer/trainer.py. Fetched 2026-08-10.
