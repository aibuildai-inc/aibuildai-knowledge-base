# meta-math/MetaMathQA

395,000 grade-school and competition math question/solution pairs in one `train` split, built by prompting GPT-3.5-Turbo to rewrite GSM8K and MATH training questions four ways and to write a chain-of-thought solution for each rewrite.

**meta-math/MetaMathQA** is the training data behind MetaMath, introduced in "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models" [1]. The paper bootstraps new questions from the training splits of GSM8K and MATH by four augmentation methods - Answer Augmentation (resampling reasoning paths for the original question), Rephrasing (an LLM-rewritten restatement of the question), and two backward-reasoning methods, Self-Verification and FOBAR, which mask a number in the question and ask the model to recover it from the given answer [1]. GPT-3.5-Turbo generates both the rephrased/backward questions and every reasoning-path solution, at temperature 0.7 [1]. The resulting query/response pairs feed supervised fine-tuning for chain-of-thought math reasoning. **Every source question is drawn only from the GSM8K and MATH training splits: the dataset card states in bold that none of the augmented data comes from either benchmark's test split, so evaluating a model trained on this data against GSM8K or MATH test accuracy is not contaminated by construction** [2]. It lives at https://huggingface.co/datasets/meta-math/MetaMathQA .

**Use it for**: reasoning-trace SFT - each row is a `query`/`response` chat pair with a full chain-of-thought solution, not a preference pair; maps to the SFT method card's single-turn instruction format. No decontamination step is needed against GSM8K/MATH test sets given the train-only sourcing above [2].

**Licence**: MIT (`cardData.license` is `"mit"`, tag `license:mit`), ungated (`"gated": false, "private": false`) [3].

**Shape**: 395,000 rows in one config (`default`), one split `train`, four string columns (`type`, `query`, `original_question`, `response`) [4][5].

**Hold out**: nothing. Every row is built from a GSM8K or MATH training-split question, and the dataset card states in bold that no row is derived from either benchmark's test split [2]; the paper's own accuracy claims are measured against those external test sets (GSM8K 1,319 rows, MATH 5,000 rows), which are not part of this release [1].

**Origin**: built by the MetaMath authors from GSM8K/MATH training questions rewritten and solved by GPT-3.5-Turbo, with a supervised consistency check on the rephrased questions [1]. Hub API at the check date: `downloads` 62,578, `downloadsAllTime` 522,597, `likes` 466 [3][6].

**Trained-on-by**: the paper's own MetaMath-7B/13B/70B and MetaMath-Mistral-7B models [1][2]; the meta-math/MetaMath-7B-V1.0 model card declares `datasets: meta-math/MetaMathQA` in its metadata [7]. The dataset card additionally names, without a metric or citation of its own, OpenChat-3.5, CausalLM-14B, a Zephyr-7B-alpha variant, and Ziya2-13B-Base as models the authors say were trained with MetaMathQA [2], and describes Arithmo-Mistral-7B as combining MetaMathQA with the MathInstruct dataset [2].

**Introduced by**: [1] (Yu et al.).

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 395,000 |

One config, `default`, four columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `type` | string |
| `query` | string |
| `original_question` | string |
| `response` | string |

`type` takes 8 values, matching the paper's Table 1 ("Number of samples in the proposed MetaMathQA") breakdown by source benchmark and augmentation method exactly (datasets-server `/statistics`, `column_name="type"`) [8][1]:

| type | rows |
| --- | --- |
| `GSM_AnsAug` | 80,000 |
| `GSM_Rephrased` | 80,000 |
| `GSM_SV` | 40,000 |
| `GSM_FOBAR` | 40,000 |
| `MATH_AnsAug` | 75,000 |
| `MATH_Rephrased` | 50,000 |
| `MATH_SV` | 15,000 |
| `MATH_FOBAR` | 15,000 |

Sizes (datasets-server `/size`) [4]: 395,626,321 bytes of original JSON download, 187,622,686 bytes as Parquet, 369,474,919 bytes decoded in memory. No source states token counts, but datasets-server `/statistics` gives per-column string-length statistics (characters, not tokens): `query` ranges 0-4,409 characters, mean 213.5, median 198; `original_question` ranges 16-4,309, mean 196.7, median 181; `response` ranges 15-5,367, mean 498.2, median 426 [8]. The `/size`, `/info`, `/statistics`, and `/first-rows` calls behind this card's numbers take no revision parameter, so they read whatever datasets-server currently serves for `main`; they are reported here as live reads, not as covered by the `Load it` revision pin, though the Hub API's `sha` for `main` at the check date is that same pinned commit [3][4][5][8][13].

## Quality

- Every solution (`response`) is a GPT-3.5-Turbo generation filtered to have the correct final answer against the source question's ground-truth answer, not a human-written or human-checked solution [1].
- For the Rephrasing method, the paper reports that Complexity-based CoT prompting with GPT-3.5-Turbo answers the rephrased questions at 76.30% accuracy, which it calls comparable to answering the original questions, as its evidence that the rephrased questions preserve the original's meaning [1].
- Table 3 of the paper ("Effect of different question augmentation with LLaMA-2-7B finetuned on GSM8K or MATH") is the deciding ablation: plain SFT on the GSM8K training set alone (no augmentation) scores 41.6% on GSM8K, versus 64.4% when fine-tuned on the full GSM8K-sourced augmentation (AnsAug + Rephrasing + SV + FOBAR); on MATH, plain SFT on the MATH training set alone scores 4.7%, versus 17.7% with the full MATH-sourced augmentation [1]. The paper's headline result, MetaMath-7B trained on the combined 395K-row set, reaches 66.5% on GSM8K and 19.8% on MATH [1].
- No source states a duplicate-row rate or an annotator/model agreement rate for the release as a whole; none is invented here.

## Load it

```python
import datasets

REV = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"  # main at the check date
train = datasets.load_dataset("meta-math/MetaMathQA", revision=REV, split="train")  # 395,000 rows
```

**Trap**: the repository ships a single `MetaMathQA-395K.json` file behind the `default` config and one `train` split - there is no held-out split to accidentally merge in, but `original_question` is present on every row and is identical to `query` for the `AnsAug` rows (unmodified) while differing from `query` on the `Rephrased`/`SV`/`FOBAR` rows (the bootstrapped question); do not assume `query` always equals the GSM8K/MATH source wording [4][5].

## Neighbors

meta-math also hosts smaller and derived releases; row counts and columns below were read live at the check date [9].

- `meta-math/MetaMathQA-40K` - 40,000 rows, columns `query`/`response`/`type` (no `original_question`); five sampled rows carry `type` values (`GSM_SV`, `GSM_FOBAR`, `GSM_AnsAug`, `MATH_Rephrased`, `MATH_AnsAug`) drawn from the same 8-value label set as this release, and its query/response text has the same GSM8K/MATH chain-of-thought shape, but no fetched source confirms row-for-row that its 40,000 rows are a subset of this release's 395,000, and its own card states nothing about how it was sampled [9][10].
- `meta-math/MetaMathQA_GSM8K_zh` - 231,685 rows, adding `query_zh`/`response_zh` machine-translated-to-Chinese columns alongside the original `query`/`response`; its card states only that the question-answer pairs are translated from MetaMathQA by GPT-3.5-Turbo with few-shot prompting, without naming which rows or types were selected [9][11].
- `meta-math/GSM8K_zh` - a Chinese translation of GSM8K itself (not of MetaMathQA), 7,473 training and 1,319 test rows per its card; a source dataset relative, not a derivative, of this release [9][11].
- `meta-math/GSM8K_Backward` - 1,270 rows in a `test` split only, columns `question`/`answer`/`original_question`/`original_answer`; this is an evaluation set for the backward-reasoning ability the paper's SV/FOBAR augmentations target, not additional training data, and should not be mixed into training [9][12].

This corpus prefers the full `meta-math/MetaMathQA` release over `MetaMathQA-40K` for training when the larger set is affordable, since the 40K set carries strictly fewer columns (no `original_question` traceback) for the same row shape [9][10].

## A row

The repository serves one config and one schema, so one row covers it. From `config="default"`, `split="train"`, a `GSM_AnsAug`-type row (datasets-server `/first-rows`) [13]:

```json
{
  "type": "GSM_AnsAug",
  "query": "The town of Belize has 400 homes. One fourth of the town's homes are white. One fifth of the non-white homes have a fireplace. How many of the non-white homes do not have a fireplace?",
  "original_question": "The town of Belize has 400 homes. One fourth of the town's homes are white. One fifth of the non-white homes have a fireplace. How many of the non-white homes do not have a fireplace?",
  "response": "One fourth of the town's homes are white, so there are 400/4 = 100 white homes.\nThe remaining non-white homes are 400 - 100 = 300 homes.\nOne fifth of the non-white homes have a fireplace, so there are 300/5 = 60 non-white homes with a fireplace.\nTherefore, the number of non-white homes without a fireplace is 300 - 60 = 240.\n#### 240\nThe answer is: 240"
}
```

For an `AnsAug` row `query` equals `original_question` verbatim, since Answer Augmentation resamples only the reasoning path, not the question [1]. On a `Rephrased`, `SV`, or `FOBAR` row `query` instead holds the LLM-bootstrapped question while `original_question` holds the GSM8K/MATH source wording, e.g. the first served row (`config="default"`, `split="train"`, `row_idx=0`, `type="MATH_AnsAug"`) [13] pairs `query` "Gracie and Joe are choosing numbers on the complex plane..." with an identical `original_question`, while row 4 (`type="GSM_FOBAR"`) [13] pairs a backward-reasoning `query` that masks a number as `x` against an `original_question` phrased as the forward question.

## Where it came from

Built by the MetaMath authors from the training splits of GSM8K and MATH: for each source question, GPT-3.5-Turbo either resamples additional chain-of-thought reasoning paths with temperature sampling (Answer Augmentation), rewrites the question from a different perspective (Rephrasing), or masks a number in the question and asks the model to recover it from the stated answer, in either a Self-Verification declarative-statement form or a FOBAR form (Backward Reasoning) [1]. Rephrased questions pass a supervised consistency check against the original meta-question before being kept [1]. GPT-3.5-Turbo also generates every solution, filtered to keep only reasoning paths whose final answer matches the ground truth [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. The datasets-server endpoints ([4], [5], [8], [13]) take no revision parameter and were read against whatever `main` currently serves; at the check date the Hub API's `sha` for `main` is the same commit this card pins, aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18 [3], but a future force-push to `main` would move those live endpoints without moving the pin.

[1] Yu et al., "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models", 2023. https://arxiv.org/abs/2309.12284 - the origin paper; current title read from the live abs page (v4). Body text (augmentation methods, GPT-3.5-Turbo generation, Table 1, Table 2, rephrasing-consistency accuracy, headline results) read from the arXiv-hosted HTML of v4, https://arxiv.org/html/2309.12284v4. Fetched 2026-08-11.

[2] meta-math/MetaMathQA dataset card (README). https://huggingface.co/datasets/meta-math/MetaMathQA/raw/main/README.md - no-test-data note, named downstream models, Arithmo-Mistral-7B combination note. Fetched 2026-08-11.

[3] Hugging Face Hub API record for meta-math/MetaMathQA. https://huggingface.co/api/datasets/meta-math/MetaMathQA?full=true - licence, gate, `sha`, `downloads`, `likes`, siblings. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=meta-math%2FMetaMathQA Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=meta-math%2FMetaMathQA Fetched 2026-08-11.

[6] Hugging Face Hub API record for meta-math/MetaMathQA with `expand[]=downloadsAllTime`. https://huggingface.co/api/datasets/meta-math/MetaMathQA?expand[]=downloadsAllTime Fetched 2026-08-11.

[7] meta-math/MetaMath-7B-V1.0 model card (README). https://huggingface.co/meta-math/MetaMath-7B-V1.0/raw/main/README.md - `datasets: meta-math/MetaMathQA` metadata field. Fetched 2026-08-11.

[8] datasets-server statistics endpoint. https://datasets-server.huggingface.co/statistics?dataset=meta-math%2FMetaMathQA&config=default&split=train - `type` column frequencies. Fetched 2026-08-11.

[9] Hugging Face Hub API listing for author meta-math, plus per-dataset datasets-server `/size` calls for each neighbor named above: `meta-math/MetaMathQA-40K`, `meta-math/MetaMathQA_GSM8K_zh`, `meta-math/GSM8K_zh`, `meta-math/GSM8K_Backward`. https://huggingface.co/api/datasets?author=meta-math and https://datasets-server.huggingface.co/size?dataset=<id> - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[10] meta-math/MetaMathQA-40K dataset card, datasets-server `/info`, and datasets-server `/first-rows` (5 rows read at `config="default"`, `split="train"`, offset 0). https://huggingface.co/datasets/meta-math/MetaMathQA-40K/raw/main/README.md , https://datasets-server.huggingface.co/info?dataset=meta-math%2FMetaMathQA-40K , https://datasets-server.huggingface.co/first-rows?dataset=meta-math%2FMetaMathQA-40K&config=default&split=train - columns, card gives no sampling description, sampled rows' `type` values and text shape. Fetched 2026-08-11.

[11] meta-math/MetaMathQA_GSM8K_zh dataset card (full README) and datasets-server `/info`; meta-math/GSM8K_zh Hub API description (translation method, row counts) via https://huggingface.co/api/datasets?author=meta-math. https://huggingface.co/datasets/meta-math/MetaMathQA_GSM8K_zh/raw/main/README.md , https://datasets-server.huggingface.co/info?dataset=meta-math%2FMetaMathQA_GSM8K_zh Fetched 2026-08-11.

[12] meta-math/GSM8K_Backward datasets-server `/info`. https://datasets-server.huggingface.co/info?dataset=meta-math%2FGSM8K_Backward - `test`-only split, column names. Fetched 2026-08-11.

[13] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=meta-math%2FMetaMathQA&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data for math chain-of-thought fine-tuning, with nothing to hold out: the dataset is built only from GSM8K/MATH training questions, and the dataset card's bold no-test-data statement [2] is corroborated by the paper's own methodology, which augments only the training splits [1] - the same two facts the screening row's note names.

### The screening row

The row's own note [screening row for `meta-math/MetaMathQA`, corpus check 2026-08-11]: "GSM8K+MATH TRAIN problems rewritten (rephrase, self-verify, FOBAR) with GPT-3.5-written solutions; card asserts no test data and probes agree." The row carries no flag.
