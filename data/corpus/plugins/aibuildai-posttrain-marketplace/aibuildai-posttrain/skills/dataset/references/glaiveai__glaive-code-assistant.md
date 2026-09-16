# glaiveai/glaive-code-assistant

136,109 single-turn code question/answer pairs, generated entirely by Glaive's synthetic data platform, apache-2.0 licensed.

**glaiveai/glaive-code-assistant** was built and released by Glaive AI; there is no origin paper, only the dataset's own card [1]. The README describes it as roughly 140,000 code problems and solutions produced by Glaive's synthetic data generation platform, with questions phrased to resemble the way real users ask code-related questions, and reports that about 60% of the samples are Python [1]. Each row is a `question`/`answer` string pair, so it serves single-turn code-assistant SFT rather than multi-turn dialogue or preference data. It lives at https://huggingface.co/datasets/glaiveai/glaive-code-assistant .

**Use it for**: single-turn SFT on code-assistant instruction following - the SFT method card - by mapping each row's `question` to a user turn and `answer` to the assistant turn; the dataset carries no chat template of its own (`chat_dialect` is unset in the shipped file) [2], so a chat template must be applied before training. No usage-shape restriction is stated by the source.

**Licence**: apache-2.0, ungated. The repository ships a full `LICENSE.md` containing the Apache License 2.0 text [3], and the Hub API's `cardData.license` reads `"apache-2.0"` [4]; no other licence catch is stated.

**Shape**: one config, one split - `train`, 136,109 rows - two string columns, `question` and `answer` [5][6].

**Hold out**: nothing. No source - the dataset card, the Hub API, or the corpus screening note - states or implies that any rows overlap a held-out evaluation set [1][4][7].

**Origin**: built by Glaive AI; both the questions and answers are synthetic generations from Glaive's own data-generation platform, and the README does not name the generating model [1]. Hub API at the check date: `downloads` 2,785, `downloadsAllTime` 15,514, `likes` 104 [4].

**Trained-on-by**: Glaive's own `glaive-coder-7b`, a CodeLlama-7b fine-tune, lists this dataset as its training data and reports 63.1% pass@1 on HumanEval and 45.2% pass@1 on MBPP [8]. `openchat/openchat_3.5` lists it among the datasets in its training mix [9]. `Weyaxi/Einstein-v4-7B` also lists it in its `datasets:` metadata [10].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and columns (datasets-server `/size` and `/info`) [5][6]:

| split | rows | columns |
| --- | --- | --- |
| `train` | 136,109 | `question` (string), `answer` (string) |

Sizes (datasets-server `/size`) [5]: 218,840,602 bytes of original JSON download, 103,952,789 bytes as Parquet, 207,368,464 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The README states the data is synthetically generated end-to-end by Glaive's own data-generation platform - both questions and answers - and does not name the generating model [1].
- The README puts the language mix at roughly 60% Python; no per-language breakdown or measured figure beyond that estimate is given [1].
- No source states a measured contamination rate, duplicate rate, or a quality-complaint log for this dataset; none is invented here.
- No annotation or human-review process is described; the README frames the data as machine-generated and invites the reader to report problems via Discord rather than describing an internal QA pass [1].

## Load it

Single split, single config; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date, matching the last-modified timestamp 2023-09-27) [4]:

```python
import datasets

REV = "398ce053096475671e8d3872b8e5b28b59f8fcec"  # main at the check date
train = datasets.load_dataset("glaiveai/glaive-code-assistant", revision=REV, split="train")  # 136,109 rows
```

**Trap**: the repository ships one raw JSON file (`c9bc9129-eba0-4b10-8292-4ae70fc7fa0d.json`, 218,840,602 bytes) rather than a `data/` directory of pre-partitioned Parquet shards [11]; `load_dataset` reads it through the generic `json` builder and materializes the whole 136,109-row `train` split in one piece - there is no smaller split or config to sample from first.

## Neighbors

Glaive released two direct successors under the same organization, both with the identical `question`/`answer` string schema confirmed live at the check date [12][13][14]:

- `glaiveai/glaive-code-assistant-v2` - ~215k rows (215,166 served); its README states it was built on top of this v1 release [12][14].
- `glaiveai/glaive-code-assistant-v3` - ~950k rows (950,384 served); its README states it was built on top of `v2` and that it already folds in both v1 and v2 [13][14].

Because v3's own card says it already contains this release's rows, mixing v1 into a v3-based training run duplicates data; pick one version, and prefer v3 for the largest single pull or this v1 release when reproducing the smaller `glaive-coder-7b` training set it was built for [8][13].

## A row

One shape is served (`config="default"`, `split="train"`). From `row_idx=0` (datasets-server `/first-rows`) [15]:

```json
{
  "question": "How can I output bold text in Bash? I have a Bash script that prints some text to the screen using the `echo \"Some Text\"` command. Is there a way I can format the text to make it bold?",
  "answer": "Yes, you can format the output text in Bash to make it bold. Bash allows you to use special escape sequences for text decoration. To make some text bold in bash, you would use the escape sequence `\\033[1m`, and to reset the formatting, you would use `\\033[0m`. \n\nHere's how you can update your `echo` statement to print bold text:\n\n```bash\necho -e \"\\033[1mSome Text\\033[0m\"\n```\n\nIn this code:\n\n- The `-e` option of `echo` allows the interpretation of backslash escapes.\n- The `\\033[1m` sequence sets the text to be bold.\n- The `Some Text` part is the actual text that will be printed in bold.\n- The `\\033[0m` sequence resets the text formatting to the default, so anything printed afterwards will be in the default format.\n\nRemember that these escape sequences may not work in all terminals and circumstances, but they should work in most common situations."
}
```

## Where it came from

Built and released by Glaive AI. The README states the questions and answers were both generated by Glaive's own synthetic data-generation platform, worded to resemble how real users phrase code-related questions, with no human-authored or human-annotated source pool, and no named generating model [1]. The same organization later built `glaive-code-assistant-v2` on top of this release and `glaive-code-assistant-v3` on top of v2 [12][13].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] glaiveai/glaive-code-assistant dataset card (README). https://huggingface.co/datasets/glaiveai/glaive-code-assistant/raw/main/README.md - description, ~140k/~60% python claims, no-paper status. Fetched 2026-08-11.

[2] The shortlist row supplied with this card's request, `chat_dialect: "none"` field. Checked 2026-08-11.

[3] LICENSE.md in the repository. https://huggingface.co/datasets/glaiveai/glaive-code-assistant/raw/main/LICENSE.md - full Apache License 2.0 text. Fetched 2026-08-11.

[4] Hugging Face Hub API record for glaiveai/glaive-code-assistant. https://huggingface.co/api/datasets/glaiveai/glaive-code-assistant?full=true - `cardData.license`, `sha`, `downloads`, `likes`, `lastModified`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=glaiveai%2Fglaive-code-assistant Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=glaiveai%2Fglaive-code-assistant Fetched 2026-08-11.

[7] The corpus screening row for `glaiveai/glaive-code-assistant`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[8] glaiveai/glaive-coder-7b model card (README). https://huggingface.co/glaiveai/glaive-coder-7b/raw/main/README.md - training-data attribution, HumanEval/MBPP pass@1 figures. Fetched 2026-08-11.

[9] openchat/openchat_3.5 model card (README). https://huggingface.co/openchat/openchat_3.5/raw/main/README.md - `datasets:` metadata and prose listing this dataset in its training mix. Fetched 2026-08-11.

[10] Weyaxi/Einstein-v4-7B model card (README). https://huggingface.co/Weyaxi/Einstein-v4-7B/raw/main/README.md - `datasets:` metadata listing this dataset. Fetched 2026-08-11.

[11] Repository tree listing. https://huggingface.co/api/datasets/glaiveai/glaive-code-assistant/tree/main - single-file JSON layout, file size. Fetched 2026-08-11.

[12] glaiveai/glaive-code-assistant-v2 dataset card (README). https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v2/raw/main/README.md - states it was built on top of this release, ~215k rows claim. Fetched 2026-08-11.

[13] glaiveai/glaive-code-assistant-v3 dataset card (README). https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3/raw/main/README.md - states it was built on top of v2 and already folds in v1 and v2, ~1M rows claim. Fetched 2026-08-11.

[14] datasets-server size and info endpoints for the two neighbors. https://datasets-server.huggingface.co/size?dataset=glaiveai%2Fglaive-code-assistant-v2 , https://datasets-server.huggingface.co/info?dataset=glaiveai%2Fglaive-code-assistant-v2 , https://datasets-server.huggingface.co/size?dataset=glaiveai%2Fglaive-code-assistant-v3 , https://datasets-server.huggingface.co/info?dataset=glaiveai%2Fglaive-code-assistant-v3 - row counts 215,166 and 950,384, and matching `question`/`answer` string schema. Fetched 2026-08-11.

[15] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=glaiveai%2Fglaive-code-assistant&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as single-turn code-assistant SFT data: apply a chat template to the `question`/`answer` pairs and train on the full 136,109-row `train` split, with nothing to hold out. This rests on the dataset card's own description of the data as synthetic QA pairs for code-assistant training [1], the confirmed two-column schema [6], and the screening row's note, which states the same size, origin, and language-mix facts [7].

### The screening row

The row's own note [7]: "~140k code questions and answers produced by Glaive's synthetic data platform (model not named); ~60% Python." The row carries no flag.
