# jondurbin/airoboros-2.2

44,838 single-turn and multi-turn instruction/response rows, machine-generated mostly by GPT-4, spanning dozens of task categories (orca-style QA, coding, writing, roleplay, trivia, multiple choice, and more) - one release in Jon Durbin's ongoing "airoboros" series.

**jondurbin/airoboros-2.2** is built by Jon Durbin using his `airoboros` pipeline, a heavily modified, seed-free reimplementation of the Self-Instruct approach [1] that queries GPT-4 (and other APIs) to generate synthetic instructions and responses [2]. The dataset card describes it as "mostly a continuation" of the prior `jondurbin/airoboros-2.1` release, with fixes to contamination filtering, a new "awareness" instructor, a text-editing instructor, regenerated writing samples, and 500 summarization examples folded in from `mattpscott/airoboros-summarization` [2]. There is no dedicated paper for this dataset; it is documented only by its Hub card [2] and the `airoboros` GitHub repository [1]. It lives at https://huggingface.co/datasets/jondurbin/airoboros-2.2 .

**The card's own contamination section says the pipeline only checked embedding similarity against TruthfulQA (removing anything with a similarity score below 0.15) and did not check the hundreds of thousands of other instructions against other benchmarks, because "the instructions aren't typically verbatim with the benchmark questions" and a full check was judged too computationally expensive [2]. Treat any benchmark overlap outside TruthfulQA as unverified; decontaminate before a scored eval run.**

**Use it for**: SFT chat-style instruction tuning - the SFT method card. Each row already carries `instruction`/`response`/`system` fields with an explicit `category` label and a `skip_prompt_formatting` flag, so it maps directly onto a single-turn or multi-turn SFT record without a preference-pair conversion step.

**Licence**: `license: other` in the card's front matter, no separate licence text or file in the repository [3]. The one catch: the card's own usage note says most of the data was generated via GPT-4 API calls, which "has a restriction in the ToS about 'competing' models," and tells users seeking commercial use to "seek legal advice" [2].

**Shape**: 44,838 rows, one config (`default`), one split (`train`), five columns (`instruction`, `response`, `category`, `skip_prompt_formatting`, `system`) [4][5].

**Hold out**: no split or benchmark-set flag ships with the repository - the entire 44,838 rows sit in `train` [5]. The unresolved risk is the one named above: the card admits it did not check most benchmarks for overlap, so a reader running a scored eval should decontaminate against that eval's own benchmark before training, rather than trusting an all-clear.

**Origin**: built and released by Jon Durbin; responses are model-generated, mostly by GPT-4 per the card's own usage note [2]. Hub API at the check date: `downloads` 153, `likes` 24 [3].

**Trained-on-by**: Jon Durbin's own `jondurbin/airoboros-l2-70b-2.2` model lists `jondurbin/airoboros-2.2` in its `datasets:` front matter and describes itself as trained on a "clean" (non-uncensored) version of this dataset [6]. No other adopting model or recipe was found in this check.

**Introduced by**: no paper - the dataset card [2]; the generation pipeline itself is documented in the `airoboros` GitHub repository, which frames itself as a modified take on the Self-Instruct paper [1][7].

## Shape

Rows and split, from the datasets-server size endpoint [4]:

| split | rows |
| --- | --- |
| `train` | 44,838 |

One config, `default`, five columns, from the datasets-server info endpoint [5]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `response` | string |
| `category` | string |
| `skip_prompt_formatting` | bool |
| `system` | string |

Sizes (datasets-server `/size`) [4]: 87,194,316 bytes of original JSON download, 48,822,041 bytes as Parquet, 82,161,302 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The card states its only checked contamination signal: instructions were embedded with `thenlper/gte-small`, indexed with faiss, and anything scoring a similarity below 0.15 against TruthfulQA was removed; the card gives no dataset name for what "0.15 similarity" was measured against beyond TruthfulQA [2].
- The same section states the pipeline has "~1279 multiple choice questions, all randomly GPT generated," which the author judges to carry "probably little-to-no overlap" with other benchmarks, but this is stated as the author's own expectation, not a measured rate [2].
- The card documents one known prior defect: an earlier `airoboros-2.1`-trained model "accidentally included some of the benchmark data," inflating its TruthfulQA leaderboard score, which the author says triggered the 0.15-similarity filtering described above for this release [2].
- Of the first 90 served rows, 23 are labelled `orca`, 11 `general`, 10 `coding`, 9 `writing`, and the remainder spread across trivia, contextual QA, roleplay, jokes, and other categories; the served response itself is marked `truncated: true`, meaning the datasets-server endpoint stopped short of returning a full 100-row preview, though no individual row in this 90-row sample carries a truncated field [8].

## Load it

Pin the revision this card's numbers were read at (the shortlist row's `commit`, which matches the Hub API's current `sha` for `main`) [3]:

```python
import datasets

REV = "c6a0c2a5e58f2914bdb74dd5dee36f76df64b7f0"  # main at the check date
train = datasets.load_dataset("jondurbin/airoboros-2.2", revision=REV, split="train")  # 44,838 rows
```

**Trap**: the repository ships a single `instructions.jsonl` tree file with no held-out split, so `load_dataset` returns everything as `train`; there is no built-in mechanism to exclude the unverified-overlap rows described above - filtering, if wanted, has to be done downstream by the caller.

## Neighbors

Jon Durbin's `airoboros` series has many prior and later releases under the same author; the dataset card itself names `jondurbin/airoboros-2.1` as this release's direct predecessor [2]. Row counts below were read live at the check date [4][9]:

- `jondurbin/airoboros-2.1` (36,306 rows) - the direct predecessor this release continues; its own README carries no body text beyond an `apache-2.0` license front-matter block, so its content is described only through this card's own "continuation of" statement [2][9].
- `jondurbin/airoboros-2.2.1` (42,731 rows) - the immediate successor; its own card states it is "a slight update to 2.2," regenerating writing responses that had been produced with `gpt-4-0613` (found to be shorter and lower-scoring on readability metrics) using `gpt-4-0314` instead [9][10]. Prefer 2.2.1 over 2.2 when only the latest fixed writing responses are wanted; prefer 2.2 for reproducing results tied to this exact release.
- `jondurbin/airoboros-3.0` (45,902 rows) - a later release that switches the on-disk format to ShareGPT-style multi-turn conversations and adds a `mathjson` category not present in this release's five-column schema, so its rows do not share a collator with this dataset [9][11].
- `jondurbin/airoboros-3.1` (59,277 rows) and `jondurbin/airoboros-3.2` (58,709 rows) - later releases in the same series [9].
- `mattpscott/airoboros-summarization` (3,949 rows) - the separate, BSD-3-Clause-licensed summarization dataset that this release's card says it folded 500 examples from [2][9][12].

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=1` (datasets-server `/first-rows`) [8]:

```json
{
  "instruction": "In what classic board game would you find properties named Boardwalk and Park Place?",
  "response": "Monopoly",
  "category": "trivia",
  "skip_prompt_formatting": false,
  "system": "You are a world class trivia AI - provide accurate, succinct responses."
}
```

Row shapes vary by category: `row_idx=0` (category `rp`, not shown here) carries a multi-paragraph character-card `instruction` and a long-form `response`, while short-answer categories like `trivia` and `general` (shown above) carry brief instructions and one- or two-word responses; both share the same five columns [8].

## Where it came from

Built and released by Jon Durbin using his `airoboros` pipeline, a self-hosted, seed-free reimplementation of the Self-Instruct approach that queries chat/completion APIs (mostly GPT-4) to generate synthetic instructions, with an in-memory vector index used for diversity/similarity filtering instead of ROUGE scoring [1][2]. The card describes this specific release as a continuation of `jondurbin/airoboros-2.1`, adding a new "awareness" instructor that conditions responses on system-prompt context (time, location, senses), a text-editing instructor built via a reverse-prompt mechanism (corrupting well-written text, then training the model to correct it), regenerated "Once upon a time..." writing samples, rebuilt roleplay/GTKM conversation data with USER/ASSISTANT prefixes stripped, ASCII normalization of UTF-8 punctuation, and 500 folded-in summarization examples credited to `mattpscott/airoboros-summarization` [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] `jondurbin/airoboros` GitHub repository README. https://github.com/jondurbin/airoboros - describes the pipeline as a modified take on the Self-Instruct paper (https://arxiv.org/abs/2212.10560), with no human-generated seeds, using /v1/completions or /v1/chat/completions APIs. Fetched 2026-08-12.

[2] jondurbin/airoboros-2.2 dataset card (README). https://huggingface.co/datasets/jondurbin/airoboros-2.2/raw/main/README.md - overview, contamination section, awareness/editor/writing/roleplay/UTF-8/summarization notes, usage/licence note. Fetched 2026-08-12.

[3] Hugging Face Hub API record for jondurbin/airoboros-2.2. https://huggingface.co/api/datasets/jondurbin/airoboros-2.2?full=true - licence field, `sha`, `downloads`, `likes`, siblings, last-modified date. Fetched 2026-08-12.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=jondurbin%2Fairoboros-2.2 Fetched 2026-08-12.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=jondurbin%2Fairoboros-2.2 Fetched 2026-08-12.

[6] jondurbin/airoboros-l2-70b-2.2 model card (README) and Hub API record. https://huggingface.co/jondurbin/airoboros-l2-70b-2.2/raw/main/README.md and https://huggingface.co/api/models/jondurbin/airoboros-l2-70b-2.2 - `datasets: [jondurbin/airoboros-2.2]` in front matter; card text describing training on a "clean" version of the airoboros-2.2 dataset. Fetched 2026-08-12.

[7] Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions", 2022. https://arxiv.org/abs/2212.10560 - the paper the `airoboros` pipeline describes itself as a modified take on; linked from the GitHub README [1]. Not independently fetched beyond its abs-page title as referenced in [1].

[8] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=jondurbin%2Fairoboros-2.2&config=default&split=train Fetched 2026-08-12.

[9] datasets-server size endpoint, one call per neighbor: `jondurbin/airoboros-2.1`, `jondurbin/airoboros-2.2.1`, `jondurbin/airoboros-3.0`, `jondurbin/airoboros-3.1`, `jondurbin/airoboros-3.2`, `mattpscott/airoboros-summarization`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-12.

[10] jondurbin/airoboros-2.2.1 dataset card (README). https://huggingface.co/datasets/jondurbin/airoboros-2.2.1/raw/main/README.md - states it is "a slight update to 2.2," describes regenerated writing responses. Fetched 2026-08-12.

[11] jondurbin/airoboros-3.0 dataset card (README). https://huggingface.co/datasets/jondurbin/airoboros-3.0/raw/main/README.md - states the format changed to ShareGPT and adds a `mathjson` category. Fetched 2026-08-12.

[12] mattpscott/airoboros-summarization dataset card (README). https://huggingface.co/datasets/mattpscott/airoboros-summarization/raw/main/README.md - states its BSD-3-Clause license and description as an adaptation of the Booksum dataset for airoboros. Fetched 2026-08-12.

[13] The corpus screening row for jondurbin/airoboros-2.2, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable for SFT, with an unresolved contamination caveat: the card's own contamination section only verifies TruthfulQA overlap via embedding similarity and explicitly leaves other benchmarks unchecked, so any scored-eval use needs its own decontamination pass rather than relying on this release's filtering. This rests on the card's own contamination-section text, quoted and paraphrased above [2], and matches the screening row's flag [13].

### The screening row

The row's own note [13]: "airoboros GPT-4-generated instruction set; contains deliberately toxic content." Its flag: "unverified: - Generator known (GPT-4 airoboros self-instruct) but the card admits the 2.1 pipeline shipped benchmark data, filters only TruthfulQA and leaves other-benchmark overlap unchecked and typically non-verbatim, which verbatim probes cannot clear."
