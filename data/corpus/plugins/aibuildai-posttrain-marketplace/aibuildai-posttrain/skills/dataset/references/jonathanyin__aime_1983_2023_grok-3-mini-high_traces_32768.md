# jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768

866 AIME competition problems (years 1983-2023), each paired with one Grok-3-mini-high reasoning trace capped at a 32,768-token budget and already wrapped in a chat template.

**jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768** is a Hugging Face dataset published by the user `jonathanyin`. Its 12-column schema (`ID`, `Year`, `Problem Number`, `Question`, `Answer`, `Part`, `Reasoning`, `Solution Attempt`, `Answer Attempt`, `Correct`, `templated_response`, `Token Count`) extends the same author's 919-problem `AIME_1983_2023` problem set, which carries only `ID`/`Year`/`Problem Number`/`Question`/`Answer`/`Part` [1][2]. The dataset card carries no prose - only a YAML `dataset_info`/`configs` block - so there is no paper describing how the traces were produced; the dataset card itself is the only description available [1]. The task shape is reasoning-trace SFT: a math competition prompt paired with a chain-of-thought trace and a final boxed answer. **This is the same AIME 1983-2023 problem archive the corpus also uses for other Grok-trace training rows, and the corpus's own screening flags AIME competition items generically as sitting on the surface of the `aime2025` evaluation benchmark; every one of the 866 `Year` values read here falls in 1983-2023 with none in 2025, but decontaminate against whatever AIME-based eval split is used downstream before a scored run** [3]. It lives at https://huggingface.co/datasets/jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768 .

**Use it for**: reasoning-trace SFT on math word problems - short prompt, long chain-of-thought, boxed final answer - restricted in practice to already-correct traces, since every one of the 866 served rows has `Correct` set to `True` [4]. The `templated_response` column is a fully pre-rendered chat-template string (Qwen `<|im_start|>`/`<|im_end|>` markers), so it can be used as-is as the SFT target text without re-templating; see the SFT method card. Decontaminate against AIME-based eval sets first, per the restriction above [3].

**Licence**: not stated. `cardData` carries no `license` field and the card body mentions none; the repository is public and ungated [1].

**Shape**: 866 rows, one split (`train`), one config (`default`), 12 columns [5][6].

**Hold out**: no row in this release falls outside 1983-2023 - every one of the 866 `Year` values read is in that range, none is 2025 - so no row-level overlap with an "AIME 2025" eval set is measurable from this release alone. The corpus's rule-risk flag nonetheless treats AIME competition items generically as the `aime2025` evaluation surface; decontaminate against whichever AIME eval split is used downstream before scoring on it [3].

**Origin**: built by Hub user `jonathanyin`; the reasoning traces are attributed to Grok-3-mini-high by the repository's own name, and no source states whether the `Correct` label was machine-graded or human-checked. Hub API at the check date: `downloads` 143, `downloadsAllTime` 1,447, `likes` 0 [1][7].

**Trained-on-by**: none found. No source located during this check names a model or training recipe that used this dataset.

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and splits (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 866 |

One config, `default`, 12 columns (datasets-server `/info` and `cardData.dataset_info`) [1][6]:

| column | dtype |
| --- | --- |
| `ID` | string |
| `Year` | int64 |
| `Problem Number` | int64 |
| `Question` | string |
| `Answer` | string |
| `Part` | string |
| `Reasoning` | string |
| `Solution Attempt` | string |
| `Answer Attempt` | string |
| `Correct` | bool |
| `templated_response` | string |
| `Token Count` | int64 |

Sizes (datasets-server `/size`) [5]: 23,288,245 bytes of Parquet download, 50,458,628 bytes decoded in memory. No source states a fixed sequence-length limit, but reading all 866 `Token Count` values gives a minimum of 1,379, a maximum of 32,752, and a mean of about 12,246 - consistent with the "32768" figure in the repository name acting as an upper token budget that no row exceeds [8].

## Quality

- Reading all 866 served rows (nine `rows` calls covering offsets 0 through 865) shows every one has `Correct` set to `True`; the release appears pre-filtered to traces where the model's extracted `Answer Attempt` matched `Answer` [8].
- The same full read shows `Part` is `"I"` for 326 rows, `"II"` for 327 rows, and null for 213 rows - the null rows are the pre-2000-era AIME years that had a single exam rather than separate `I`/`II` sittings [8].
- `Token Count` never exceeds 32,752, one below the 32,768 figure the name implies is the cap [8].
- The `templated_response` field embeds a Qwen system message ("You are Qwen, created by Alibaba Cloud...") even though the reasoning content is attributed to Grok-3-mini-high by the repository name; no source explains this, so the chat template is presumably applied for downstream SFT on a Qwen-family base model rather than reflecting who generated the reasoning [9].
- The assistant turn in `templated_response` opens with two consecutive template markers, `<|im_start|>assistant\n<|im_start|>think\n`, and the reasoning, solution and final `\boxed{...}` answer all sit inside that single turn with no closing think tag before `<|im_end|>` - a nonstandard structure a reader building a collator needs to reproduce exactly rather than assume a standard `<think>...</think>` split [9].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2025-04-23) [1]:

```python
import datasets

REV = "320311cce0fa82666c8d17d60d8c4d57979c0214"  # main at the check date
train = datasets.load_dataset(
    "jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768",
    revision=REV,
    split="train",
)  # 866 rows
```

**Trap**: there is only one split (`train`) and one config (`default`), so no split/config argument is needed to get the full 866 rows - but every row already has `Correct == True`, so filtering on that column changes nothing; a reader expecting a mix of correct and incorrect traces to filter will find none to remove [8].

## Neighbors

Same author, same underlying AIME 1983-2023 archive, different trace budgets or chat templates; row counts read live at the check date [10]. Prefer this 32,768-token release when the training run's context window can absorb traces up to that length and a larger, less-truncated pool of correct traces is wanted.

- `jonathanyin/AIME_1983_2023` - the 919-problem source set with no traces (`ID`/`Year`/`Problem Number`/`Question`/`Answer`/`Part` only); this release covers 866 of those 919 problems [1][2].
- `jonathanyin/aime_1983_2023_grok-3-mini-high_traces` - the same trace collection with no stated token cap, 865 rows; a spot read of its first 100 and last 65 rows found `Correct` also `True` in all 165 checked [11].
- `jonathanyin/aime_1983_2023_grok-3-mini-high_traces_16384` - the same collection capped at a smaller token budget, only 704 rows: fewer problems are solved correctly within the tighter budget [10].
- `jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768_r1_formatted` - the same 866 problem/trace pairs re-templated: its first row drops the Qwen system message and instead wraps the reasoning in a `<think>` block after a "Please reason step by step, and put your final answer within \boxed{}." user instruction, an R1-style template rather than this release's Qwen-style one [12].
- `jonathanyin/aime_1983_2023_grok-3-mini-high_traces_16392_r1_formatted` - the R1-style template at the smaller (~16k) token budget, 702 rows [10].
- `jonathanyin/aime_1983_2023_grok-3-mini-high_traces_r1_formatted` - the R1-style template with no stated token cap, 866 rows and byte sizes identical to the 32768_r1_formatted variant, per `cardData` in both cards' README [10].

## A row

One config, one split, so one row covers the served shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/rows`, offset 0) [8], with `Reasoning` and `templated_response` truncated:

```json
{
  "ID": "1983-1",
  "Year": 1983,
  "Problem Number": 1,
  "Question": "Let $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_xw=24$ , $\\log_y w = 40$ and $\\log_{xyz}w=12$ . Find $\\log_zw$ .",
  "Answer": "60",
  "Part": null,
  "Reasoning": "The problem states that x, y, and z all exceed 1, and w is a positive number. Given are log base x of w equals 24, log base y of w equals 40, and log base xyz of w equals 12. I need to find log base z of w. [...]",
  "Solution Attempt": "[...omitted for length...]",
  "Answer Attempt": "60",
  "Correct": true,
  "templated_response": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nLet $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_xw=24$ , $\\log_y w = 40$ and $\\log_{xyz}w=12$ . Find $\\log_zw$ .\n\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<|im_start|>think\nThe problem states that x, y, and z all exceed 1 [...] \\boxed{60}<|im_end|>\n",
  "Token Count": 4466
}
```

## Where it came from

Built by Hub user `jonathanyin` from that author's own 919-problem `AIME_1983_2023` set [2]. The dataset card gives no methodology prose - no builder statement, no generation script, no grading description - only the `dataset_info` YAML that records the schema and split sizes [1]. The `Reasoning`, `Solution Attempt`, `Answer Attempt`, `Correct` and `Token Count` columns, and the model name in the repository id, indicate the traces come from Grok-3-mini-high, but no source states the collection or grading procedure explicitly.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. That pin covers only the `load_dataset` call: the datasets-server endpoints used throughout this card ([4], [5], [6], [8], [9], and the neighbor calls in [10][11][12]) take no revision parameter - passing one is silently ignored and the endpoint still returns HTTP 200 with the same figures - so every row count, column list, `Correct`/`Part`/`Token Count` statistic, and sampled row in this card, not only the Neighbors figures, is a live read against `main` at the check date and is not covered by the `sha` pin in Load it.

[1] Hugging Face Hub API record for jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768. https://huggingface.co/api/datasets/jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768?full=true - `sha`, `cardData.dataset_info`, tags, gated/private status, `lastModified`; `downloadsAllTime` via the same endpoint's `expand[]=downloadsAllTime` variant. The dataset card (README) itself: https://huggingface.co/datasets/jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768/raw/main/README.md - YAML-only, no prose. Fetched 2026-08-11.

[2] Hugging Face Hub API record for jonathanyin/AIME_1983_2023. https://huggingface.co/api/datasets/jonathanyin/AIME_1983_2023?full=true - schema and split size of the source problem set. Fetched 2026-08-11.

[3] The corpus screening row for `jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768`, supplied with this card's request - its `note` and `flag`, read back in the appendix. Checked 2026-08-11.

[4] datasets-server rows endpoint, full split read (see [8]) - basis for the `Correct == True` claim used in Use it for.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=jonathanyin%2Faime_1983_2023_grok-3-mini-high_traces_32768 Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=jonathanyin%2Faime_1983_2023_grok-3-mini-high_traces_32768 Fetched 2026-08-11.

[7] Hugging Face Hub API record with `expand[]=downloadsAllTime`. https://huggingface.co/api/datasets/jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768?expand[]=downloadsAllTime Fetched 2026-08-11.

[8] datasets-server rows endpoint, `config=default`, `split=train`, read in nine calls at offsets 0, 100, 200, 300, 400, 500, 600, 700, 800 (length 100 each) to cover all 866 rows. https://datasets-server.huggingface.co/rows?dataset=jonathanyin%2Faime_1983_2023_grok-3-mini-high_traces_32768&config=default&split=train&offset=<n>&length=100 - basis for the full-split `Correct`, `Part`, and `Token Count` statistics, and for the sampled row. Fetched 2026-08-11.

[9] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=jonathanyin%2Faime_1983_2023_grok-3-mini-high_traces_32768&config=default&split=train - used to inspect the `templated_response` chat-template structure. Fetched 2026-08-11.

[10] datasets-server size endpoint, one call per neighbor, for the row counts of `jonathanyin/aime_1983_2023_grok-3-mini-high_traces_16384`, `jonathanyin/aime_1983_2023_grok-3-mini-high_traces_16392_r1_formatted`, and `jonathanyin/aime_1983_2023_grok-3-mini-high_traces_r1_formatted`; and each neighbor's dataset card (README) for its `cardData.dataset_info` byte sizes. https://datasets-server.huggingface.co/size?dataset=<id> and https://huggingface.co/datasets/<id>/raw/main/README.md - these endpoints take no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[11] datasets-server rows endpoint for jonathanyin/aime_1983_2023_grok-3-mini-high_traces, offsets 0 and 800 (length 100 each, 165 rows total read out of 865). https://datasets-server.huggingface.co/rows?dataset=jonathanyin%2Faime_1983_2023_grok-3-mini-high_traces&config=default&split=train&offset=<n>&length=100 Fetched 2026-08-11.

[12] datasets-server first-rows endpoint for jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768_r1_formatted. https://datasets-server.huggingface.co/first-rows?dataset=jonathanyin%2Faime_1983_2023_grok-3-mini-high_traces_32768_r1_formatted&config=default&split=train - used to compare the `templated_response` structure against this release's. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data, with the decontamination caveat above resting on facts already established in this card: every one of the 866 served rows falls in 1983-2023 with `Correct` true, and the row's own flag treats AIME items generically as overlapping an `aime2025` evaluation surface even though no row here carries a 2025 year [3][8].

### The screening row

The row's own note [3]: "Same problems with Grok-3-mini-high traces at 32,768 tokens." Its flag [3]: "rule-risk: aime2025 - Same AIME archive as train rows with Grok traces; AIME competition items are the aime2025 surface"
