# jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768

716 correct DeepSeek-R1-Distill-Qwen-1.5B reasoning traces on AIME problems from 1983 to 2023, one row per problem the model solved, chat-templated and ready for SFT.

**jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768**, at https://huggingface.co/datasets/jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768 [2], pairs each problem from the builder's own `jonathanyin/AIME_1983_2023` pool [1] with a DeepSeek-R1-Distill-Qwen-1.5B generation sampled at a 32,768-token budget, keeping only the 716 of 919 problems (77.9%) the model answered correctly [2][3]. The repository's README carries no descriptive text at all, only the machine-written `dataset_info`/`configs` YAML block that Hugging Face Hub renders as the card [2]; there is no paper behind this release - "no paper - the dataset card" is the accurate citation [2]. **The problems are drawn from the same 1983-2023 AIME archive that backs the builder's sibling trace datasets used elsewhere for training, and the corpus screening flags a rule-risk that AIME items were not removed for potential evaluation overlap [4]; this archive itself only spans 1983-2023 and does not contain AIME 2024 or 2025 problems, confirmed by reading every row's `Year` field [3].**

**Use it for**: reasoning-trace SFT, chat-templated already (the `templated_response` column is a ready-to-tokenize string); the SFT method card is the relevant mapping. Do not treat this as a benchmark-eval file - it is filtered to correct rollouts and is explicitly a training row (screening `use` is `train`) [4].

**Licence**: not stated - `cardData` carries no `license` key and the repo's tags include no `license:` tag [2]. Ungated, public (`"gated": false`, `"private": false`) [2].

**Shape**: 716 rows, one config (`default`), one split (`train`), 12 columns [2][5][6].

**Hold out**: no exact row-level overlap with a named evaluation split is confirmed here - the archive covers AIME 1983-2023 only, not AIME 2024 or 2025, verified by reading all 716 rows' `Year` values (range 1983-2023) [3]. The corpus screening flag nonetheless marks a rule-risk that this same 1983-2023 archive is reused across the builder's other training rows without any decontamination step against evaluation sets; see the appendix screening row for the flag's own words [4].

**Origin**: built by Hugging Face user `jonathanyin`; the `Reasoning`, `Solution Attempt`, and `Answer Attempt` columns are generations from DeepSeek-R1-Distill-Qwen-1.5B, filtered to rows where `Correct` is true [2][3]. Hub API at the check date: `downloads` 177, `downloadsAllTime` 854, `likes` 0 [2][7].

**Trained-on-by**: none found - no source names a model or training recipe that consumed this specific repository.

**Introduced by**: no paper - the dataset card [2].

## Shape

Rows and split (datasets-server `/size`) [5]:

| split | rows |
| --- | --- |
| `train` | 716 |

One config, `default`, with 12 columns (datasets-server `/info`, matching `cardData.dataset_info`) [6][2]:

| column | dtype |
| --- | --- |
| `ID` | string |
| `Year` | int64 |
| `Problem Number` | int64 |
| `Question` | string |
| `Answer` | int64 |
| `Part` | string |
| `Reasoning` | string |
| `Solution Attempt` | string |
| `Answer Attempt` | string |
| `Correct` | bool |
| `templated_response` | string |
| `Token Count` | int64 |

`Part` is null for years without an AIME I/II split (1983-1999 in the rows read) and holds `"I"` or `"II"` for the two-sitting years; reading all 716 rows confirms `Part` only ever takes the values null, `"I"`, or `"II"` [3]. Sizes (datasets-server `/size`) [5]: 12,302,910 bytes of Parquet download, 30,307,252 bytes decoded in memory. The `Token Count` column ranges 1,361-31,048 across all 716 rows, mean 8,175, read directly from the served data; no source states which field it counts or which tokenizer produced it [3].

## Quality

- Every one of the 716 served rows has `Correct` equal to true - confirmed by reading all 716 rows, not a sample - because the repository keeps only the rollouts the 1.5B model answered correctly out of the 919-problem pool in `jonathanyin/AIME_1983_2023` [1][3].
- That 716/919 = 77.9% solve rate is the lowest among the builder's five same-archive, same-32,768-token-budget trace releases read at the check date: DeepSeek-R1 (full) kept 877/919 (95.4%), QwQ-32B kept 889/919 (96.7%), DeepSeek-R1-Distill-Qwen-7B kept 830/919 (90.3%), and DeepSeek-R1-Distill-Qwen-14B kept 740/919 (80.5%) [8]. This is the number that decides how strong a teacher the 1.5B model is for this pool: weakest of the five, and only 2.6 points above the 14B distillation despite being an order of magnitude smaller.
- No source states an annotator-agreement figure, a duplicate-row rate, or a held-out-eval contamination rate for this specific repository; none is invented here. The rule-risk flag in the appendix is the only stated contamination-adjacent signal [4].

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main`, last modified 2025-04-25) [2]:

```python
import datasets

REV = "929c6ac8d9d606b376ffdacbbaeb78afbb80cb3a"  # main at the check date
train = datasets.load_dataset(
    "jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768",
    revision=REV,
    split="train",
)  # 716 rows, all Correct == True
```

**Trap**: there is only one split (`train`) and it is pre-filtered to correct rollouts - there is no held-out validation or test split shipped in this repository, so any eval split must be carved out manually before training [2][5].

## Neighbors

All from the same Hub author, `jonathanyin`, and all four 32,768-budget trace siblings share this dataset's 12 column names - but not its dtypes. Fetching `/info` for all four shows every one of them disagrees with this dataset on `Answer` and/or `Answer Attempt`: this dataset has `Answer: int64`, `Answer Attempt: string`; the DeepSeek-R1 sibling has them swapped (`Answer: string`, `Answer Attempt: int64`); the QwQ-32B sibling has `Answer: string`, `Answer Attempt: float64`; the 7B sibling has both as `string` (matching this dataset only on `Answer Attempt`); the 14B sibling has both as `int64` (matching this dataset only on `Answer`) [9]. No sibling matches this dataset on both columns, so a collator built for this dataset's `Answer`/`Answer Attempt` fields is not a safe drop-in on any of the four.

- `jonathanyin/AIME_1983_2023` - the underlying 919-problem pool with no traces (`ID`, `Year`, `Problem Number`, `Question`, `Answer`, `Part` only); every trace dataset in this family, including this one, is a filtered subset of it [1].
- `jonathanyin/aime_1983_2023_deepseek-r1_traces_32768` - same pool, same 32,768-token budget, DeepSeek-R1 (full model, not distilled) traces, 877 rows kept [8][9].
- `jonathanyin/aime_1983_2023_qwq-32b_traces_32768` - same pool and budget, QwQ-32B traces, 889 rows kept, the strongest teacher of the family read at the check date [8][9].
- `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-7b_traces_32768` - same pool and budget, the 7B distillation, 830 rows kept [8][9].
- `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768` - same pool and budget, the 14B distillation, 740 rows kept [8][9].
- Matching datasets at a 16,384-token budget instead of 32,768 exist for several of the same teachers (e.g. `jonathanyin/aime_1983_2023_deepseek-r1_traces_16384`, `jonathanyin/aime_1983_2023_qwq-32b_traces_16384`) - a shorter generation budget, not fetched in detail here [10].
- This corpus prefers the strongest available teacher for SFT traces; among the family, that is `jonathanyin/aime_1983_2023_qwq-32b_traces_32768` or `jonathanyin/aime_1983_2023_deepseek-r1_traces_32768` on solve-rate grounds, not this 1.5B release, unless the training goal specifically calls for a weak-teacher or self-distillation trace [8].

## A row

One config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/rows`) [3], with long fields truncated:

```json
{
  "ID": "1983-1",
  "Year": 1983,
  "Problem Number": 1,
  "Question": "Let $x$ , $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\\log_xw=24$ , $\\log_y w = 40$ and $\\log_{xyz}w=12$ . Find $\\log_zw$ .",
  "Answer": 60,
  "Part": null,
  "Reasoning": "Okay, so I have this problem where I need to find log base z of w, given some logarithmic equations. Let me try to understand what the problem is saying and how to approach it. [...] Thus, we have: [...] Therefore, log_z w = 60.",
  "Solution Attempt": "Given the problem, we need to find \\(\\log_z w\\) given the following logarithmic equations: [...] Thus, \\(z = w^{1/60}\\). Therefore, \\(\\log_z w = 60\\).",
  "Answer Attempt": "60",
  "Correct": true,
  "templated_response": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nLet $x$ , $y$ and $z$ all exceed $1$ [...] Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<|im_start|>think\nOkay, so I have this problem [...] **Final Answer**\n\\boxed{60}\n<|im_start|>answer\nGiven the problem, we need to find [...] \\boxed{60}\n<|im_end|>",
  "Token Count": 3282
}
```

`templated_response` concatenates a Qwen-style system/user turn with an assistant turn split into two custom role tags, `<|im_start|>think` holding the `Reasoning` text and `<|im_start|>answer` holding the `Solution Attempt` text, closed with a single `<|im_end|>`; there is no `</think>` closing tag anywhere in the field, confirmed by searching the full string [3].

## Where it came from

Built by Hub user `jonathanyin` from their own `jonathanyin/AIME_1983_2023` problem pool, 919 AIME problems spanning 1983-2023 with no traces [1]. For each problem, DeepSeek-R1-Distill-Qwen-1.5B was sampled for a reasoning trace under a 32,768-token generation budget - the count is named in the repository id and matches the `Token Count` column's observed range (1,361-31,048 across all 716 rows) [2][3]; no source states the exact sampling or decoding settings (temperature, number of samples per problem) beyond this. The resulting `Reasoning`, `Solution Attempt`, and `Answer Attempt` were graded against the pool's `Answer` field to produce the boolean `Correct` column, and only the 716 rows with `Correct == true` are served in this repository [2][3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] `jonathanyin/AIME_1983_2023` dataset card (YAML `dataset_info`) and datasets-server `/size`. https://huggingface.co/datasets/jonathanyin/AIME_1983_2023/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=jonathanyin%2FAIME_1983_2023 - 919-row, 6-column base pool with no trace columns. Fetched 2026-08-11.

[2] Hugging Face Hub API record and raw README for `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768`. https://huggingface.co/api/datasets/jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768?full=true and https://huggingface.co/datasets/jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768/raw/main/README.md - `sha`, `gated`, `private`, `downloads`, `likes`, `lastModified`, `cardData.dataset_info`/`configs`, tags. Fetched 2026-08-11.

[3] datasets-server `/rows` endpoint, read across offsets 0, 100, 200, 300, 400, 500, 600, 700 (length 100 each) to cover all 716 rows of `split=train`, `config=default`. https://datasets-server.huggingface.co/rows?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768&config=default&split=train - used for the `Correct`, `Year`, `Part`, and `Token Count` counts above and for the sampled row. This endpoint takes no revision parameter, so these counts are live as of the fetch, not pinned to the `929c6ac...` revision used in Load it. Fetched 2026-08-11.

[4] The corpus screening row for `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768 - takes no revision parameter; the 716-row, 12,302,910-byte figures are live as of the fetch, not pinned. Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768 - takes no revision parameter; the column/dtype table is live as of the fetch, not pinned, though it matches the pinned `cardData.dataset_info` in source [2]. Fetched 2026-08-11.

[7] Hugging Face Hub API `downloadsAllTime` expansion. https://huggingface.co/api/datasets/jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768?expand%5B%5D=downloadsAllTime Fetched 2026-08-11.

[8] datasets-server size endpoints, one call per sibling, for the row counts compared above: `jonathanyin/aime_1983_2023_deepseek-r1_traces_32768` (877), `jonathanyin/aime_1983_2023_qwq-32b_traces_32768` (889), `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-7b_traces_32768` (830), `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768` (740). https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[9] Hugging Face Hub API list of datasets by author `jonathanyin`, and datasets-server `/info` for all four 32,768-budget siblings, compared column-by-column against this dataset's own `/info` (source [6]) for the dtype table above. https://huggingface.co/api/datasets?author=jonathanyin&limit=100 , https://datasets-server.huggingface.co/info?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1_traces_32768 , https://datasets-server.huggingface.co/info?dataset=jonathanyin%2Faime_1983_2023_qwq-32b_traces_32768 , https://datasets-server.huggingface.co/info?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-7b_traces_32768 , https://datasets-server.huggingface.co/info?dataset=jonathanyin%2Faime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768 - none take a revision parameter, so the dtypes compared are live as of the fetch. Fetched 2026-08-11.

[10] The same Hub API author listing as source [9], read for the names of the 16,384-token-budget sibling repositories mentioned in Neighbors; their contents were not fetched. https://huggingface.co/api/datasets?author=jonathanyin&limit=100 Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as training data, with a decontamination caveat: the dataset is a clean, pre-filtered, chat-templated set of correct reasoning traces on a fixed, dated (1983-2023) problem archive [2][3], but the corpus screening flags a rule-risk that the same archive backs other training rows in this sourcing project without having AIME items removed for potential evaluation overlap [4]. The archive's own year range (1983-2023, confirmed by reading all 716 rows) does not include AIME 2024 or 2025 [3], so the specific overlap the flag names is not reproduced by this card's own reading of the data; the flag is reported here as the screening row states it, not resolved.

### The screening row

The row's own note [4]: "Same problems with DeepSeek-R1-Distill-Qwen-1.5B traces; the weakest teacher here, barely above the model being trained." Its flag [4]: "rule-risk: aime2025 - Same AIME archive as train rows with distilled-1.5B traces; AIME items not removed."
