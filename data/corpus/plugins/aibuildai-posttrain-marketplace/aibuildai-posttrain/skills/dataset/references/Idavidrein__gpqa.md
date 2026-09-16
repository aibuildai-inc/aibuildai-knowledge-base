# Idavidrein/gpqa

1,252 rows across four CSV configs of PhD-written, multiple-choice science questions - a gated benchmark, not a training corpus.

**Idavidrein/gpqa** is the Hub release of GPQA, introduced in "GPQA: A Graduate-Level Google-Proof Q&A Benchmark" [1]: 448 (later extended to 546) multiple-choice questions in biology, physics and chemistry, written by annotators who have or are pursuing a PhD in the question's subdomain, then checked by a second domain expert and by non-expert validators with unrestricted web access [1]. **This is an evaluation benchmark, not training data: the paper embeds a canary string in the release specifically so training corpora can filter it out, and the Hub repository gates the files behind an agreement not to reveal examples online, "to reduce the risk of leakage into foundation model training corpora"** [1][2]. It lives at https://huggingface.co/datasets/Idavidrein/gpqa .

**Use it for**: held-out scoring of scientific reasoning and knowledge, not for any training split; the closest fit in this corpus is a multiple-choice-QA eval harness, not the SFT or preference method cards. The served columns are flat CSV rows (question, four answer choices, explanation, plus writer/validator metadata), not a chat or preference format - a harness must build its own prompt template from `Question`, `Correct Answer` and the three `Incorrect Answer N` columns [3].

**Licence**: CC BY 4.0 in the repository's `cardData` [2]. The catch is separate from the licence: access to every file is gated behind a click-through agreement not to post examples in plain text or images online [2][1], and the README additionally mentions "MIT" in its body text per the corpus screening record [4] - the README itself sits behind that same gate, so this card cannot confirm what that second mention covers.

**Shape**: 1 repository, 4 configs (`gpqa_extended`, `gpqa_main`, `gpqa_diamond`, `gpqa_experts`), each a single `train` split, 1,252 rows total, CSV format [2][4].

**Hold out**: everything. GPQA is built to stay out of training data entirely - the paper's canary string exists so that corpora can be scrubbed of it before pretraining or fine-tuning [1], and no split here is a training split. Screen any training pool against all four configs before a scored run.

**Origin**: built and released by the paper's authors (David Rein et al.); questions are written by human domain-PhD annotators, with answers and explanations also human-written [1]. Hub API as of this check: 107,275 downloads, 499 likes, 2,009,285 all-time downloads [2].

**Trained-on-by**: not applicable in the training sense - GPQA is designed as a held-out eval, and no source here states a model trained on it. As an evaluation benchmark it is widely reported: Meta's Llama 3 herd of models paper reports 0-shot chain-of-thought GPQA scores for the Llama 3 8B/70B/405B models against several other model families, citing this paper [5].

**Introduced by**: [1] (Rein et al.), no separate blog post fetched for this card.

## Shape

Configs, splits and row counts, from the corpus screening record for this dataset (the Hub datasets-server `/info` and `/size` endpoints are blocked by the gate without authentication, so this card relies on the record's own fetch of them) [4]:

| config | split | rows |
| --- | --- | --- |
| `gpqa_extended` | `train` | 546 |
| `gpqa_main` | `train` | 448 |
| `gpqa_diamond` | `train` | 198 |
| `gpqa_experts` | `train` | 60 |
| total | | 1,252 |

These counts match the paper: it reports collecting 564 questions, removing 18 to an unreleased held-out set, leaving the 546-question GPQA Extended; GPQA (main) is the 448-question subset where at least 1/2 experts agree and at most 2/3 non-experts answer correctly; GPQA Diamond is the 198-question subset where both experts agree and at most 1/3 non-experts answer correctly [1]. The paper does not name or describe a fourth, 60-row "experts" config, so what distinguishes `gpqa_experts` is not stated by the paper.

The repository's file list is four CSVs (`gpqa_diamond.csv`, `gpqa_experts.csv`, `gpqa_extended.csv`, `gpqa_main.csv`) plus `eval.yaml`, `README.md`, `license.txt`, and `.gitattributes` [2]. The screening record's flattened column list (89 columns) mixes clearly per-question fields (`Question`, `Correct Answer`, `Incorrect Answer 1-3`, `Explanation`, `Subdomain`, the `Pre-Revision` and `Extra Revised` variants) with fields that look like per-person records (`Name`, `Domain`, `Description of Expertise`, `Qualifications`, `Num Correct Expert Validations`, `Expert Accuracy on Questions Written`) [4]. Since the gate blocks a direct per-config fetch, this card cannot confirm which of those 89 columns actually populate `gpqa_experts` versus the three question configs.

The paper states question length for the 546-question extended set: a median of 561 characters including answer choices, or 146 tokens by OpenAI's tiktoken byte-pair tokenizer [1]. No source states token or length statistics for the main, diamond or experts subsets specifically.

The Hub API's `usedStorage` for the whole repository is 7,085,662 bytes [2]; this is git-repository storage (including any history overhead), not the same quantity as a datasets-server content-byte count, and this card has no independently fetched datasets-server figure to compare it against.

## Quality

- Every question passes through a fixed four-stage pipeline: a domain-PhD writes it; a first expert in the same domain answers it and gives feedback; the writer may revise based on that feedback; a second expert answers the (possibly revised) version and a post-hoc agreement judgment is recorded, then three non-expert validators with web access attempt it [1].
- Measured accuracy on the 546-question extended set: first expert validator 66.5% ± 4.0%, second expert validator (post-revision) 64.8% ± 4.0% (95% CI); non-expert validators reach 34.1%; random guessing on four choices is 25% [1].
- The paper's own objectivity estimate, after manually re-checking the 191 cases where the second expert disagreed with the question writer, is that about 74% of GPQA Extended questions have uncontroversibly correct answers [1].
- Sufficient-expertise self-reports from the second expert validator are 90.7% on the extended set, 93.5% on main, 97.0% on diamond [1].
- The paper's strongest baseline at release, few-shot chain-of-thought GPT-4, reached 38.7-39.7% accuracy across the three subsets - close to non-expert human accuracy and far below expert accuracy, which the paper offers as evidence the benchmark is "Google-proof" [1].
- No source fetched for this card states a measured contamination or duplicate rate for this Hub release; the canary string [1] is a leakage-prevention mechanism, not a measured contamination figure.

## Load it

The dataset is gated: `load_dataset` requires a Hugging Face account that has accepted the repository's access agreement and an authentication token, neither of which this card's fetch environment had - no README, row, or size figure that requires authentication could be independently verified for this card, and those gaps are marked "not stated" or attributed to the screening record above rather than invented [2].

```python
import datasets

REV = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"  # main at the check date
ds = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main", revision=REV, split="train", token=True)
```

**Trap**: the config argument is mandatory - `gpqa_extended`, `gpqa_main`, `gpqa_diamond` or `gpqa_experts` - and `gpqa_main` and `gpqa_diamond` are filtered subsets of `gpqa_extended`, so loading more than one of them into the same pool duplicates rows [1][4]. A `token=True` (or a pre-cached `huggingface-cli login`) is required or the load fails with an authentication error, as this card's own unauthenticated fetches did [2].

## Neighbors

- `gpqa_diamond` is described by the paper as "our highest quality subset", and is the config most commonly used for headline benchmark numbers because it filters hardest for both expert agreement and non-expert difficulty [1].
- No other Hugging Face Hub repository was fetched for this card, so no sibling or successor release outside this one repository is confirmed; none found.

## A row

The repository is gated and no authentication token was available to this card's fetch environment, so no live row could be pulled from any of the four configs; the datasets-server `/first-rows` endpoint returned an authentication error for each config tried (`gpqa_diamond`, `gpqa_main`, `gpqa_extended`, `gpqa_experts`) [6]. The screening record's column list [4] indicates the question-bearing configs (`gpqa_extended`, `gpqa_main`, `gpqa_diamond`) carry columns including `Question`, `Correct Answer`, `Incorrect Answer 1`-`3`, `Explanation`, `Subdomain`, `Record ID` and `High-level domain`, but no card content states literal cell values because none were read.

## Where it came from

Built and released by David Rein and co-authors [1]. The authors hired 61 contractors through Upwork, each holding or pursuing a PhD in their field, to write and validate questions in a subdomain of biology, physics or chemistry chosen for producing difficult, objective questions [1]. Every question passes the four-stage pipeline described in Quality above (write, first-expert-validate, revise, second-expert-validate, then three non-expert validations), and the extended/main/diamond split is a post-hoc filter on the resulting expert- and non-expert-accuracy statistics, not a separate collection round [1]. The Hub repository additionally ships `eval.yaml`, a `license.txt`, and a canary string embedded in the data to support automated exclusion from training corpora [1][2].

## Sources

Fetched on 2026-08-11; that date covers every number, quote and API response above. Hugging Face Hub repositories are mutable (a repo can be force-pushed or re-gated), which is why Load it pins the revision recorded by the corpus screening record. This dataset is gated: its README, `eval.yaml`, `license.txt`, and every datasets-server endpoint (`/info`, `/size`, `/first-rows`) returned an authentication error to this card's unauthenticated fetches; those gaps are stated explicitly rather than filled in.

[1] Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark", 2023. https://arxiv.org/abs/2311.12022 - abstract and full text (via the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2311.12022): question counts, split definitions, accuracy figures, canary string, collection pipeline. Fetched 2026-08-11.

[2] Hugging Face Hub API record for Idavidrein/gpqa. https://huggingface.co/api/datasets/Idavidrein/gpqa?full=true - `sha`, `cardData` (license, gated prompt, configs), `siblings` (file list), `downloads`, `likes`, `createdAt`, `usedStorage`; all-time downloads read via the same endpoint's `expand[]=downloadsAllTime` variant. This endpoint does not require authentication. Fetched 2026-08-11.

[3] Inferred column names from the corpus screening record's `columns` field [4]; no separate schema documentation was fetched because the README is gated.

[4] The corpus screening row for `Idavidrein/gpqa`, supplied with this card's request - its declared `rows_served`, `splits`, `columns`, `bytes`, `licence_card_field`, `licence_body_mentions` and `config_names` fields, used because the equivalent live datasets-server endpoints are blocked by the Hub gate without authentication. Checked 2026-08-11.

[5] Meta AI, "The Llama 3 Herd of Models", 2024. https://arxiv.org/abs/2407.21783 - Table of reasoning benchmark results reporting 0-shot CoT GPQA scores for Llama 3 models, citing this dataset's paper. Read via the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2407.21783. Fetched 2026-08-11.

[6] datasets-server first-rows endpoint, tried for all four configs. https://datasets-server.huggingface.co/first-rows?dataset=Idavidrein%2Fgpqa&config=gpqa_diamond&split=train (and `gpqa_main`, `gpqa_extended`, `gpqa_experts`) - each returned `{"error":"The dataset does not exist, or is not accessible without authentication (private or gated)..."}`. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable only as a held-out evaluation set, never as training data: the paper's own canary string and stated leakage concerns, and the Hub gate's access agreement not to post examples online, both establish this as the card's central restriction (quoted in the opening paragraph) [1][2]. The screening row's note independently confirms the same reading.

### The screening row

The row's own note [4]: "GPQA graduate-level science questions written by domain PhDs; gated on the Hub and a canonical eval set." The row carries no flag.
