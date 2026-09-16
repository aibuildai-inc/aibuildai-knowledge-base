# stellalisy/MediQ_AskDocs_preference

A gated, MIT-licensed collection of synthetic preference pairs of clinical follow-up questions, built over real r/AskDocs posts, shipped as four filter/attribute variants each with a train and a validation JSONL file.

**stellalisy/MediQ_AskDocs_preference** is the preference-data half of the MediQ-AskDocs release accompanying "ALFA: Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning" [1]; its sibling `stellalisy/MediQ_AskDocs` carries the SFT half, as the card for this repository states directly - "This dataset is the preference data subset of MediQ_AskDocs, for the SFT subset, see MediQ_AskDocs" [2]. The paper builds the pairs by taking real follow-up questions from r/AskDocs threads and prompting an LLM to write "enhanced" and "corrupted" counterfactual rewrites of each question along one attribute at a time (e.g. clarity, medical accuracy), then keeping the resulting better-vs-worse pairs as DPO/PPO-style training data [1]. **The repository is gated with automatic approval, and every content endpoint we probed - the raw README, the Croissant metadata, the datasets-server info/size/rows APIs, and a HEAD request on a data file - returned an authentication error rather than data, so no split sizes, columns, or rows could be confirmed from outside an authenticated session [3].**

**Use it for**: preference-pair training (DPO/PPO-style, chosen-vs-rejected question pairs) only - the dataset's own card points SFT users at the separate `MediQ_AskDocs` repository instead [2]. No source we could reach states the column schema, so whether the served format is implicit-prompt (paired full text) or explicit-prompt (separate prompt/chosen/rejected fields) is not stated; check the preference-format section of the method card that matches whichever trainer you use once you have gated access.

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`) [3]. The one catch: the repository is gated (`"gated":"auto"`) - Hugging Face auto-approves access requests, but every file, viewer, and API request we sent without a granted, authenticated token was rejected with "Access to dataset ... is restricted. You must have access to it and be authenticated to access it" [3].

**Shape**: not confirmed from outside authentication. The repository tree lists eight JSONL files under four directories - `filtered/`, `unfiltered/`, `coarse_filtered/`, `coarse_unfiltered/` - each with a `train.jsonl` and a `validation.jsonl`, and no config/split metadata is exposed to an unauthenticated request [3].

**Hold out**: not confirmed which rows, if any, overlap the paper's held-out sets, because we could not read the served rows to check join keys. The paper does describe disjoint source splits upstream: it samples 4,463 training, 433 validation, and 620 test source questions from r/AskDocs "ensuring no post overlap" before generating pairs [1] - but the shipped repository has no `test.jsonl`, so it is not stated whether or how the paper's held-out test questions are kept out of these `train`/`validation` files.

**Origin**: built by Shuyue Stella Li and co-authors; underlying questions are real Reddit posts, and the preference labels come from an automated pipeline - a Llama-3.1-405B-Instruct-FP8 model generates the enhanced/corrupted rewrites and a GPT-4o judge verifies and filters them, not human raters [1]. Hub API at the check date: `downloads` 236, `downloadsAllTime` 1,467, `likes` 2 [3].

**Trained-on-by**: none found - this is the origin paper's own released data, and we found no independent adoption in the sources we fetched [1][2].

**Introduced by**: [1] (Li et al.).

## Shape

No source we could reach without authentication states row counts, columns, or byte sizes for this repository: the datasets-server `/info` and `/size` endpoints both return "The dataset does not exist, or is not accessible without authentication (private or gated)" [4], and the live dataset page's cached sidebar reports a single aggregate figure of 192,897 rows across the whole repository with no per-file or per-split breakdown [3]. The repository tree itself lists these eight files, with no sizes given in the unauthenticated tree listing [3]:

| directory | files |
| --- | --- |
| `filtered/` | `train.jsonl`, `validation.jsonl` |
| `unfiltered/` | `train.jsonl`, `validation.jsonl` |
| `coarse_filtered/` | `train.jsonl`, `validation.jsonl` |
| `coarse_unfiltered/` | `train.jsonl`, `validation.jsonl` |

The paper's own implementation appendix gives the closest thing to a shape table, reporting per-attribute train/dev pair counts after its quality filter: Accuracy 11,994/1,155, Answerable 11,933/1,154, Avoiding DDX Bias 12,548/1,223, Clarity 10,250/986, Focus 10,660/1,095, Relevance 11,756/1,142, summing to an "All" fine-grained total of 69,141 train / 6,755 dev pairs, plus a separate single-attribute "Coarse" condition of 12,939 train / 1,246 dev pairs [1]. These are the paper's post-filter (i.e. `filtered`/`coarse_filtered`) counts; the paper does not give a matching pre-filter table, so the `unfiltered`/`coarse_unfiltered` row counts are not stated. No source states sequence-length or token statistics for any of the eight files.

## Quality

- `filtered` and `coarse_filtered` differ from `unfiltered`/`coarse_unfiltered` by an LLM-judge quality filter: the paper reports it "filtered out 13.9% of generated pairs where LLM-judge ratings misalign with intended perturbation directions" [1]. Table 5 of the paper gives the deciding comparison of that filter's effect on downstream models - training on the filtered data (`Alfa`) versus the unfiltered data (`Alfa-Unfiltered`) moved diagnostic accuracy (MediQ-AD) from 85.43 to 87.75 at 3B scale and from 86.09 to 88.08 at 8B, with question-quality win-rate moving from 64.19 to 64.97 (3B) and 63.31 to 65.13 (8B) [1].
- `coarse_filtered`/`coarse_unfiltered` collapse the six theory-grounded attributes (accuracy, answerability, avoiding diagnosis bias, clarity, focus, relevance) into one undifferentiated "better vs. worse" label, instead of a separate pair per attribute; the paper's Table 2 shows fine-grained attribute training reaching a higher MediQ-AD diagnostic accuracy than the coarse condition (87.75 vs. 83.77 at 3B, 88.08 vs. 85.76 at 8B) while the two conditions post similar or identical win-rates on LLM-judge and expert evaluation [1].
- The paper's own quality note on the filter step: "clarity has the lowest data quality before filtering," with only 85.8% of its enhanced-vs-corrupted comparisons surviving the LLM-judge check, versus 96.0-99.8% for most other attributes and directions [1].
- The paper's ethics statement flags that the underlying r/AskDocs data may not reflect diverse patient populations or questioning strategies, and that its human expert annotators (used for the separate evaluation task, not for labeling these pairs) are all U.S.-based, English-speaking clinicians, limiting generalizability [1].
- No source states a measured duplicate rate or contamination rate for this dataset; the paper does state that its upstream source-question sampling for train/validation/test "ensur[ed] no post overlap" [1], but does not extend that guarantee to the shipped preference-pair files.

## Load it

We could not fetch a pinned, reproducible row count because the datasets-server API rejects unauthenticated requests for this repository [4]; the Hub API confirms the commit this card's other numbers were read at:

```python
import datasets

REV = "7df550ec82b3ffc63b3897747a437dea3596a679"  # main at the check date
train = datasets.load_dataset(
    "stellalisy/MediQ_AskDocs_preference",
    data_files="filtered/train.jsonl",
    revision=REV,
    split="train",
)
```

**Trap**: the repository is gated with automatic approval - `load_dataset` will fail with a 401/403 until you have visited the dataset page while logged in, accepted the access request, and passed a Hugging Face token with `use_auth_token=True` (or `HF_TOKEN` in the environment); every unauthenticated request we sent, including a plain `HEAD` on `filtered/train.jsonl`, returned `x-error-code: GatedRepo` [3]. There is no default config: you must pick one of the eight `data_files` paths above, since a bare `load_dataset("stellalisy/MediQ_AskDocs_preference")` has no declared default split structure in the unauthenticated metadata we could read [3][4].

## Neighbors

- `stellalisy/MediQ_AskDocs` - the SFT counterpart from the same paper and repository family, ungated, with `original/{train,validation,test}.jsonl` and `synthetic/{train,validation}.jsonl` files; its own card points back at this repository for preference data [5]. Use this one for supervised fine-tuning and the preference repository for DPO/PPO; do not mix the two into one preference-training run, since the SFT repository's rows are not pair-labeled.
- No other re-release, cleaned, or binarized fork of this preference dataset was found in the sources we fetched.

## A row

No row could be fetched. Every content-serving endpoint we tried for this repository - `raw/main/README.md`, `raw/main/readme.md`, the Croissant metadata endpoint, the `datasets-server` `/first-rows` endpoint, and a direct `resolve/main/filtered/train.jsonl` request - returned an authentication error instead of data, each carrying the same message: "Access to dataset stellalisy/MediQ_AskDocs_preference is restricted. You must have access to it and be authenticated to access it. Please log in." [3][4]. The dataset's `gated: "auto"` status means a logged-in user who accepts the access request should be able to read rows, but that access is not available to an unauthenticated fetch, so no schema or example row is reported here.

## Where it came from

The underlying questions come from r/AskDocs, a public health-advice subreddit where expert users can display a moderator-verified credential flair; the authors parsed 2013-2021 threads where a community member's first follow-up comment contained a question, then used GPT-4o (`gpt-4o-2024-08-06`) to decompose each conversation into atomic questions, conclusions, and presence of positive feedback, yielding 17,425 threads, 13,496 unique posts, and 24,263 questions [1]. From this pool the authors sampled a seed set of 4,463 training, 433 validation, and 620 test questions, balanced across eight proxy quality groups (expert-author status, conversation outcome, positive feedback) with no post overlap across the three splits [1]. For each seed question, `meta-llama/Llama-3.1-405B-Instruct-FP8` (temperature 1.0, run on 8 A100 GPUs via vLLM) generated one "enhanced" and one "corrupted" rewrite along a single targeted attribute, producing three candidate preference pairs per attribute per question (enhanced-vs-original, enhanced-vs-corrupted, original-vs-corrupted); a GPT-4o judge then verified that each pair's direction matched the intended perturbation, discarding pairs where it disagreed, at an overall discard rate of 13.9% [1]. The `filtered`/`unfiltered` split of the repository names whether that judge-based discard step was applied, and `coarse_filtered`/`coarse_unfiltered` names whether the pairs used the paper's six decomposed attributes or a single undifferentiated "better vs. worse" attribute [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and claim above. Hub repositories are mutable and this one is also gated, so the pinned commit in Load it is what our metadata-only numbers (licence, tags, file tree, downloads) were read at - it does not and cannot pin row counts, since no row-serving endpoint was reachable without authentication.

[1] Li, Mun, Brahman, Ilgen, Tsvetkov, and Sap, "ALFA: Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning", 2025. https://arxiv.org/abs/2502.14860 - dataset curation (Appendix B), counterfactual perturbation and filtering pipeline (§4.2, Appendix C.1), Tables 2, 5, and 7, and the ethics statement. Current title read from the live abs page. Read via the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2502.14860. Fetched 2026-08-11.

[2] Hugging Face Hub API record for stellalisy/MediQ_AskDocs_preference, `description` field (a truncated preview of the repository's README, since the raw README itself returned an access error). https://huggingface.co/api/datasets/stellalisy/MediQ_AskDocs_preference?full=true Fetched 2026-08-11.

[3] Hugging Face Hub page and API for stellalisy/MediQ_AskDocs_preference: the API record (licence, gate status, tags, sha, downloads, likes, siblings tree, last-modified date), the rendered dataset page's embedded JSON (cached `numRows: 192897` sidebar figure, `downloadsAllTime`), and direct probes that each returned an authentication error - `raw/main/README.md`, `raw/main/readme.md`, the `/viewer/default/train` page, the Croissant metadata endpoint, and a `HEAD` request on `resolve/main/filtered/train.jsonl` (which returned HTTP 401 with `x-error-code: GatedRepo`). https://huggingface.co/api/datasets/stellalisy/MediQ_AskDocs_preference?full=true and https://huggingface.co/datasets/stellalisy/MediQ_AskDocs_preference Fetched 2026-08-11.

[4] datasets-server info and size endpoints, both returning the same authentication error for this dataset. https://datasets-server.huggingface.co/info?dataset=stellalisy%2FMediQ_AskDocs_preference and https://datasets-server.huggingface.co/size?dataset=stellalisy%2FMediQ_AskDocs_preference and https://datasets-server.huggingface.co/first-rows?dataset=stellalisy%2FMediQ_AskDocs_preference&config=default&split=train Fetched 2026-08-11.

[5] stellalisy/MediQ_AskDocs dataset card (README) and Hub API record - the ungated SFT sibling repository, its file tree, and its own pointer back to this preference repository. https://huggingface.co/datasets/stellalisy/MediQ_AskDocs/raw/main/README.md and https://huggingface.co/api/datasets/stellalisy/MediQ_AskDocs?full=true Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as preference-pair training data, gated behind an auto-approved access request: the pairs are attribute-specific preference judgments over rewrites of real r/AskDocs follow-up questions, produced by an LLM-generation-plus-LLM-judge pipeline rather than human raters, matching the screening row's note [6]. No hold-out risk is confirmed either way, since no served row could be read to check for overlap with the paper's own held-out question sets.

### The screening row

The row's own note [6]: "Preference pairs over doctor answers to patient questions taken from the r/AskDocs MediQ study, shipped as filtered, unfiltered, coarse_filtered and coarse_unfiltered directories each holding a train and a validation jsonl; access is auto-granted on accepting the gate, which is why the viewer serves nothing." The row carries no flag.

[6] The corpus screening row for `stellalisy/MediQ_AskDocs_preference`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.
