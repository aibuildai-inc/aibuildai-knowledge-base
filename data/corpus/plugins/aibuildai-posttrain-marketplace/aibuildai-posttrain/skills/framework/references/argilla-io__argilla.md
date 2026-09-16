# argilla

A human-in-the-loop data-annotation platform for building and curating the datasets post-training methods consume - it ships no trainer of its own.

**Argilla** describes itself as "a collaboration tool for AI engineers and domain experts who need to build high-quality datasets for their projects" [1]. It is built by the `argilla-io` GitHub organization [2] and ships as a client SDK (`pip install argilla`) that talks to a separately deployed Argilla Server (Docker or a Hugging Face Space) over a Python API: define a `Settings` object (fields and questions), create a `Dataset`, then `dataset.records.log(...)` to push records in for human or AI-assisted labeling in the web UI [1]. It lives at https://github.com/argilla-io/argilla [2].

**When to pick it**: pick Argilla when the bottleneck is collecting or curating human (or AI) feedback into a labeled dataset - preference pairs, SFT demonstrations, classification labels - before any post-training run starts; it is not an alternative to trl, verl, or similar trainer libraries, because the current 2.x package contains no trainer class and no training loop of any kind, confirmed by listing `argilla/src/argilla` at commit `5338519`, which holds API/client/dataset/records/workspace modules and no trainer module [3]. The shortlist evidence that tagged this repo with "DPO" is a mock of trl's `DPOTrainer.push_to_hub` inside a test fixture (`mocked_trainer_push_to_huggingface`) in `argilla-v1/tests/integration/conftest.py`, for a legacy `ArgillaTrainer` wrapper that shipped only in the 1.x SDK [4]. That 1.x line's own README, read at the same commit, states "We have stopped development for the 1.x SDK version, while still committing to bug fixes," and directs readers to the 2.x README for the current SDK [5] - so the DPO-integration code this row was surfaced for is in a version-frozen, bug-fix-only line, not the library a reader would install today.

**Methods it ships**: none in the current (2.x) package - no method taxonomy, no method-specific config classes, no training entry point exist to list here [3]. The now-frozen 1.x `ArgillaTrainer` accepted a `framework="trl"` argument and, per its own integration test suite, drove SFT, reward-modeling (RM), PPO, and DPO training tasks (`SFTReturnTypes`, `RMReturnTypes`, `PPOReturnTypes`, `DPOReturnTypes`) by calling `dataset.prepare_for_training(framework="trl", task=...)` and then `ArgillaTrainer(...).train(...)`, with `trl`, `transformers`, and `peft` imported directly in that test module [6]; the fixture in [4] mocks the `push_to_hub` methods this wrapper calls on trl's `SFTTrainer`, `RewardTrainer`, and `DPOTrainer` afterward. This wrapper is not present in the current 2.x package [3][4][6].

**Scale it handles**: not applicable - there is no training loop to scale. The deployable unit is Argilla Server (Docker, or the one-click Hugging Face Spaces template linked from the README), which scales as an annotation web service, not as a distributed training job [1].

**Install**: `pip install argilla`; PyPI's release metadata for the current version, 2.8.0, records the wheel as uploaded 2025-03-10T09:42:27Z, requiring Python `>=3.9`, licensed Apache-2.0 [7]. (GitHub's release page separately timestamps the same tag's publication as 2025-03-11T08:58:41Z [8] - a day later, because it marks when the GitHub Release object was published rather than when the package was uploaded to PyPI.) This package is the client SDK only; using it requires separately deploying Argilla Server, for which the README points to a free Hugging Face Spaces template rather than a pinned server image. The v2.8.0 tag resolves to commit `78cb518`, and `argilla/pyproject.toml` at that commit pins `pydantic>=2.6.0, <3.0.0` (an upper-bounded pin a reader may already hold at a different version elsewhere in their stack), plus `httpx>=0.26.0`, `huggingface_hub>=0.22.0`, `datasets>=2.0.0`, and `pillow>=9.5.0`; no CUDA or GPU minimum is stated anywhere in that file, consistent with Argilla being a CPU-side annotation client rather than a training package [9].

**Maintained by**: the `argilla-io` GitHub organization [2]. The README at the pinned commit `5338519` opens with a maintainer notice: "The original authors have moved on to exciting new projects! The codebase is mature and stable, having served users reliably for years. While we won't be adding new features going forward, we're committed to solve bug fixes and publish patches as needed," and it invites new maintainers to open an issue [1]. Consistent with that notice, the GitHub API reports the repository's most recent push as 2026-08-03 [2] - recent activity, but per the maintainers' own statement it should be read as bug-fix maintenance rather than active feature development. The README's contributor-support link routes to a Hugging Face staff Calendly ("david-berenstein-huggingface"), consistent with Argilla's ties to the Hugging Face ecosystem [1].

## Quick start

The library has no training quick start; its quickstart is dataset creation and annotation, quoted verbatim from the README at commit `5338519` [1]:

```python
import argilla as rg

client = rg.Argilla(api_url="https://[your-owner-name]-[your_space_name].hf.space", api_key="owner.apikey")
```

```python
settings = rg.Settings(
    guidelines="Classify the reviews as positive or negative.",
    fields=[
        rg.TextField(
            name="review",
            title="Text from the review",
            use_markdown=False,
        ),
    ],
    questions=[
        rg.LabelQuestion(
            name="my_label",
            title="In which category does this article fit?",
            labels=["positive", "negative"],
        )
    ],
)
dataset = rg.Dataset(
    name=f"my_first_dataset",
    settings=settings,
    client=client,
)
dataset.create()
```

```python
from datasets import load_dataset

data = load_dataset("imdb", split="train[:100]").to_list()
dataset.records.log(records=data, mapping={"text": "review"})
```

No CLI training form exists; there is nothing to `.train()`.

## Start it

Not applicable. There is no launcher, no GPU count to choose, no batch-size arithmetic, and no OOM guidance to give, because the current package does not run a training process - it runs a client against Argilla Server for dataset authoring and annotation [1][3].

## Watch it

Not applicable in the training sense - there is no loss, reward, or KL to log. What Argilla exposes at runtime is annotation-workflow state (records logged, responses submitted, dataset progress) inside its own web UI, not a metrics stream a post-training operator would watch; this card's fetched sources ([1], [3], [7]) do not cover the UI's monitoring surface in the depth this section requires, and reading it further is out of scope for a card about picking a post-training framework.

## Save it

Not applicable to model checkpoints - Argilla persists annotated datasets and their configuration on the Argilla Server, and `dataset.records.log(...)` is how records are written into it [1]; it never writes a training checkpoint, optimizer state, or adapter, so there is no save/resume/loader contract to state here.

## Find it in the docs

- The README at the pinned commit `5338519` links its documentation as `https://argilla-io.github.io/argilla/latest/`; fetched live on 2026-08-10 that URL returns an HTTP 301 redirect to `https://docs.argilla.io/latest/`, which itself returns HTTP 200 - so the working current-docs address today is `https://docs.argilla.io/latest/`, reached via the README's own link [1]. An older address, `https://docs.v2.argilla.io/latest/`, is what the unpinned `main`-branch copy of the README links as of this fetch, but it returns HTTP 404 live and should not be used [10].
- The now-frozen 1.x docs live at a separate `https://docs.v1.argilla.io/` domain; the 1.x README explicitly separates itself from the current SDK this way [5].
- The `argilla/src/argilla` module tree at commit `5338519` - `_api`, `client.py`, `datasets`, `records`, `settings`, `users`, `webhooks`, `workspaces` - is the accurate map of what the current SDK does, read directly from the repository rather than a doc page [3].
- Runnable references beyond the docs: the repo's own `examples/` directory (not opened for this card - name only) [2].
- Honest boundary, stated where a reader would hit it: a reader who wants a DPO, SFT, PPO, or other post-training trainer inside Argilla will not find one in the current release - that capability existed only as a thin wrapper over `trl`/`transformers`/`peft` in the bug-fix-only 1.x line, and even there it deferred training to those libraries rather than implementing it, so it never carried method-level configuration, logging semantics, or checkpoint handling of its own [3][4][5][6].

## Sources

[1] argilla README, read at the pinned commit `5338519accb13ae422f8bf9c0642651c249c49af` (product description, maintainer notice, quickstart code, docs link, contributor Calendly link). https://github.com/argilla-io/argilla/blob/5338519accb13ae422f8bf9c0642651c249c49af/README.md. Fetched 2026-08-10.

[2] argilla GitHub repository (owning organization, repo home, most recent push timestamp). https://github.com/argilla-io/argilla. Fetched 2026-08-10 via the GitHub API (`api.github.com/repos/argilla-io/argilla`).

[3] Directory listing of `argilla/src/argilla`, read at the pinned commit `5338519accb13ae422f8bf9c0642651c249c49af`, showing no trainer module. https://github.com/argilla-io/argilla/tree/5338519accb13ae422f8bf9c0642651c249c49af/argilla/src/argilla. Fetched 2026-08-10 via the GitHub API contents endpoint.

[4] `argilla-v1/tests/integration/conftest.py`, read at the pinned commit `5338519accb13ae422f8bf9c0642651c249c49af`: the `mocked_trainer_push_to_huggingface` fixture mocking `trl.trainer.dpo_trainer.DPOTrainer.push_to_hub` and related SFT/reward/PEFT/setfit push-to-hub calls - the source of this row's "DPO" evidence. https://github.com/argilla-io/argilla/blob/5338519accb13ae422f8bf9c0642651c249c49af/argilla-v1/tests/integration/conftest.py. Fetched 2026-08-10.

[5] `argilla-v1/README.md`, read at the pinned commit `5338519accb13ae422f8bf9c0642651c249c49af`, stating the 1.x SDK is bug-fix-only and pointing readers to the current 2.x README. https://github.com/argilla-io/argilla/blob/5338519accb13ae422f8bf9c0642651c249c49af/argilla-v1/README.md. Fetched 2026-08-10.

[6] `argilla-v1/tests/integration/client/feedback/training/test_trl.py`, read at the pinned commit `5338519accb13ae422f8bf9c0642651c249c49af`: imports `trl`, `transformers`, and `peft` directly and drives `ArgillaTrainer` through SFT, RM, PPO, and DPO tasks via `dataset.prepare_for_training(framework="trl", task=...)`. https://github.com/argilla-io/argilla/blob/5338519accb13ae422f8bf9c0642651c249c49af/argilla-v1/tests/integration/client/feedback/training/test_trl.py. Fetched 2026-08-10.

[7] argilla on PyPI (version 2.8.0, wheel upload timestamp, Python floor, licence). https://pypi.org/project/argilla/. Fetched 2026-08-10 via `pypi.org/pypi/argilla/json`.

[8] argilla v2.8.0 GitHub Release (release-publication timestamp, distinct from the PyPI upload timestamp in [7]). https://github.com/argilla-io/argilla/releases/tag/v2.8.0. Fetched 2026-08-10 via the GitHub API releases endpoint.

[9] `argilla/pyproject.toml`, read at commit `78cb5183f72e59857cf4da3015c42a91c3b46205` (the commit the `v2.8.0` tag resolves to): dependency floors and the `pydantic` upper-bound pin. https://github.com/argilla-io/argilla/blob/78cb5183f72e59857cf4da3015c42a91c3b46205/argilla/pyproject.toml. Fetched 2026-08-10.

[10] argilla README, `main` branch (unpinned, mutable; distinct from the pinned commit `5338519` used for [1]), showing the `docs.v2.argilla.io` documentation link, which 404s live. https://raw.githubusercontent.com/argilla-io/argilla/main/README.md. Fetched 2026-08-10; the `docs.v2.argilla.io/latest/` link's HTTP 404 was independently confirmed live on 2026-08-10.
