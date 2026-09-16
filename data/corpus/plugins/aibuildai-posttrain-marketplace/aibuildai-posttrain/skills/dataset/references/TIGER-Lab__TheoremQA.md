# TIGER-Lab/TheoremQA

800 STEM theorem-driven question-answer pairs, human-expert authored, one `test` split, a minority of rows carrying a diagram image alongside the text question.

**TIGER-Lab/TheoremQA** is a question-answering benchmark introduced in "TheoremQA: A Theorem-driven Question Answering dataset" [1], curated to test whether large language models can apply named theorems - the abstract lists examples like Taylor's theorem, Lagrange's theorem, Huffman coding, and quantum and elasticity theorems - to solve challenging science problems in Math, Physics, EE&CS, and Finance [1]. The dataset card describes the 800 QA pairs as collected by human experts with very high quality, covering 350+ theorems, and pitched as a benchmark to test the limit of large language models in applying theorems to solve challenging university-level questions [2]. **This is a published evaluation benchmark with only a `test` split: it is eval-only, and any model evaluated on it must be checked for prior exposure to these exact questions before the score is reported.** It lives at https://huggingface.co/datasets/TIGER-Lab/TheoremQA .

**Use it for**: held-out evaluation of theorem-application and quantitative-reasoning ability, not training - the repository ships only a `test` split and the paper's own use of it is to score 16 LLMs and code models under Chain-of-Thoughts and Program-of-Thoughts prompting [1]. Rows are single-turn question/answer pairs (`Question`, `Answer`, `Answer_type`, optional `Picture`), not a chat-format or preference-pair dataset, so this does not map to an SFT or DPO method card as training data; it maps to an eval harness that parses `Answer_type` to grade `Answer`.

**Licence**: MIT (`cardData.license` is `"mit"`, repo tag `license:mit`, and the raw README front matter also states `license: mit`) [2][3]. No further catch stated - the eval-only restriction above comes from the dataset having no `train` split, not from the licence.

**Shape**: 800 rows, one config (`default`), one split (`test`), four columns (`Question`, `Answer`, `Answer_type`, `Picture`) [4][5].

**Hold out**: the entire dataset is a held-out evaluation set by design - there is no `train` split to draw from, and the row's own screening note calls it "a published benchmark" [6]. Anyone training a model should decontaminate against these 800 questions rather than holding out a slice of them.

**Origin**: built by TIGER-Lab (Wenhu Chen et al.); questions and answers are human-expert authored, not model-generated [1][2]. Hub API at the check date: `downloads` 3,427, `likes` 20 [3].

**Trained-on-by**: not stated - no source fetched for this card names a specific model or training recipe that trained on TheoremQA; as a published eval benchmark, the concern with adoption evidence would run the other way (evaluation coverage and contamination risk), and no source states a documented contamination check either. None found.

**Introduced by**: [1] (Chen, Yin, Ku, Lu, Wan, Ma, Xu, Wang, Xia et al., "TheoremQA: A Theorem-driven Question Answering dataset").

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `test` | 800 |

One config, `default`, four columns (datasets-server `/info`, matching the Hub `cardData.dataset_info`) [5][2]:

| column | dtype |
| --- | --- |
| `Question` | string |
| `Answer` | string |
| `Answer_type` | string |
| `Picture` | Image |

Sizes (datasets-server `/size`) [4]: 4,949,475 bytes as Parquet/original download, 4,750,688 bytes decoded in memory. No source states sequence-length or token statistics for this dataset.

## Quality

- The dataset card describes the QA pairs as collected by human experts with very high quality [2]; no further annotation-process detail (number of annotators, agreement rate) is stated in any source fetched for this card.
- No source states a measured contamination or duplicate rate for this release.
- `Answer_type` is a categorical grading key. Of 200 of the 800 rows read (offsets 0-100 and 700-800 of `test`), the observed values were: `float` 90, `integer` 52, `bool` 32, `list of integer` 18, `option` 5, `list of float` 3 - six distinct types across the two sampled ranges; the full 800-row distribution is not confirmed beyond this sample.
- Of the same 200-row sample, 17 rows (8.5%) carried a non-null `Picture` image alongside the text question; the remaining rows had `Picture` set to null. This share is a sample statistic from the two ranges read, not a confirmed full-split count.
- The origin paper's deciding number for the benchmark's difficulty: GPT-4 reached 51% accuracy with Program-of-Thoughts prompting, while "all the existing open-sourced models are below 15%, barely surpassing the random-guess baseline" [1].

## Load it

Only one split exists, and it is the evaluation set itself. Pin the revision this card's numbers were read at (the shortlist's own commit, which matches the Hub API `sha` at the check date; the repo was last modified 2024-05-15) [3]:

```python
import datasets

REV = "a340b1782960a712843aae3ed25f1e013cc008a5"  # main at the check date
test = datasets.load_dataset("TIGER-Lab/TheoremQA", revision=REV, split="test")  # 800 rows - eval only
```

**Trap**: there is no `train` split to load - `load_dataset("TIGER-Lab/TheoremQA")` with no split argument still only yields the single `test` split under the key `"test"` [2]; there is nothing to accidentally train on here, but a pipeline that assumes every dataset has a `train` split will fail silently or error on this one.

## Neighbors

- `kranthigv/TheoremQA_standardized` and `HydraLM/TheoremQA_standardized` - the former serves 1,600 rows in a single `train` split with four columns per datasets-server, twice this release's 800, and its README could not be fetched (`Entry not found`), so what the reformatting or the row doubling represents is not stated by any source fetched for this card [7].
- `mm-eval/TheoremQA` - a 53-row `test`-split repackaging with six columns (`id`, `media`, `messages`, `answer`, `question_type`, `answer_type`), evidently a multimodal-only subset (its own `dataset_info` names a `media` list-of-image field) rather than the full 800-row set [7][8].
- `ChuGyouk/TheoremQA-Ko`, `dongboklee/TheoremQA-*` (per-model output logs), and `jinulee-v/bright-theoremqa_theorems` also surfaced in a Hub search for "TheoremQA" but were not fetched for this card; their content relative to this release is not stated here.
- This original TIGER-Lab release is the one the origin paper's own benchmark numbers were computed on [1], and is preferred whenever the full 800-question set with images is needed.

## A row

The repository serves one config and one split, so one row covers the text-only shape; a second row shows the image-carrying shape, both from `config="default"`, `split="test"` [9][10].

Text-only row, `row_idx=0` [9]:

```json
{
  "Question": "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?",
  "Answer": "11760",
  "Answer_type": "integer",
  "Picture": null
}
```

Image-carrying row, `row_idx=2` [9]:

```json
{
  "Question": "Consider the following graph, with links costs listed, and assume we are using shortest-path (or lowest-cost) routing, and that routing has equilibrated to a constant set of routing tables. The routin...",
  "Answer": "5",
  "Answer_type": "integer",
  "Picture": {"src": "https://datasets-server.huggingface.co/assets/TIGER-Lab/TheoremQA/.../Picture/image.png", "height": 550, "width": 644}
}
```

Both rows share the same four-column schema; the `Picture` field is null for questions that need no figure and an image object (with a served URL, height, width) for questions that do.

## Where it came from

Built by TIGER-Lab (Wenhu Chen and co-authors). The paper describes the dataset as "curated by domain experts", covering 800 questions across 350 theorems spanning Math, Physics, EE&CS, and Finance [1]; the Hub card likewise describes the QA pairs as human-expert annotated [2]. The origin paper additionally provides a code pipeline to prompt LLMs and grade their outputs against WolframAlpha, and the Hub card links the same GitHub repository (`wenhuchen/TheoremQA`) for that evaluation code [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Chen, Yin, Ku, Lu, Wan, Ma, Xu, Wang, Xia et al., "TheoremQA: A Theorem-driven Question Answering dataset", 2023. https://arxiv.org/abs/2305.12524 - the origin paper; abstract giving theorem examples, domain coverage, and the GPT-4 51% / open-source <15% accuracy comparison; current title read from the live abs page. Fetched 2026-08-11.

[2] TIGER-Lab/TheoremQA dataset card (README), including its YAML front matter. https://huggingface.co/datasets/TIGER-Lab/TheoremQA/raw/main/README.md - "annotated by human experts", 350+ theorems, load instructions, GitHub code link, `license: mit`, `dataset_info` schema. Fetched 2026-08-11.

[3] Hugging Face Hub API record for TIGER-Lab/TheoremQA. https://huggingface.co/api/datasets/TIGER-Lab/TheoremQA?full=true - `cardData`, `sha`, `downloads`, `likes`, `gated`, `private`, last-modified date. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=TIGER-Lab%2FTheoremQA Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=TIGER-Lab%2FTheoremQA Fetched 2026-08-11.

[6] The corpus screening row for `TIGER-Lab/TheoremQA`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[7] Hugging Face Hub dataset search for "TheoremQA" and datasets-server `/size` for each neighbor listed (`kranthigv/TheoremQA_standardized`, `mm-eval/TheoremQA`). https://huggingface.co/api/datasets?search=TheoremQA and https://datasets-server.huggingface.co/size?dataset=<id> - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[8] mm-eval/TheoremQA dataset card (README), YAML front matter only. https://huggingface.co/datasets/mm-eval/TheoremQA/raw/main/README.md - `dataset_info` schema showing `media`, `messages`, `answer`, `question_type`, `answer_type` columns and 53-row `test` split. Fetched 2026-08-11.

[9] datasets-server rows endpoint, offset 0. https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FTheoremQA&config=default&split=test&offset=0&length=100 Fetched 2026-08-11.

[10] datasets-server rows endpoint, offset 700. https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FTheoremQA&config=default&split=test&offset=700&length=100 - together with [9], the 200-row sample used for the `Answer_type` and `Picture` statistics above. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as an evaluation-only benchmark: the dataset provides only a `test` split of expert-authored theorem questions, matching the row's own characterization, and the card above establishes no training-shape use for it - it should feed an eval harness, and any model scored on it should be checked for prior exposure to these 800 questions.

### The screening row

The row's own note [6]: "800 STEM theorem questions annotated by human experts; only a `test` split, and it is a published benchmark." The row carries no flag.
