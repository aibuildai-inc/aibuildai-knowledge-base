# HuggingFaceH4/Bespoke-Stratos-17k

16,710 reasoning-trace SFT rows - math, code, and science/puzzle questions each paired with a DeepSeek-R1 chain-of-thought and final answer - a TRL-formatted mirror of Bespoke Labs' original release.

**HuggingFaceH4/Bespoke-Stratos-17k** ( https://huggingface.co/datasets/HuggingFaceH4/Bespoke-Stratos-17k ) is Hugging Face's re-serialization of `bespokelabs/Bespoke-Stratos-17k`; its card states only that it is "a TRL-compatible version" of that source and points readers to the source dataset for details [1]. Bespoke Labs built the source dataset by replicating and improving the Berkeley Sky-T1 data pipeline, using SFT distillation traces sampled from DeepSeek-R1, and used it to fine-tune Qwen2.5-32B-Instruct and Qwen2.5-7B-Instruct into the Bespoke-Stratos-32B/7B reasoning models [2]. The source card's own reference runs show this DeepSeek-R1 distillation recipe beating its QwQ-distilled Sky-T1-32B predecessor and o1-preview on the benchmarks it reports: Bespoke-Stratos-32B scores AIME2024 63.3 / MATH500 93.0 / GPQA-Diamond 58.1, against Sky-T1-32B's 43.3 / 82.4 / 56.8 and o1-preview's 40.0 / 81.4 / 75.2; at 7B scale, Bespoke-Stratos-7B scores AIME2024 20.0 against un-distilled Qwen2.5-7B-Instruct's 10.0 [2]. Each row is a single-turn question with a long `<|begin_of_thought|>...<|begin_of_thought|>`-style DeepSeek-R1 reasoning trace and boxed answer, so the task shape is reasoning-trace SFT, not preference data. **Its math portion is drawn from the "AIME, MATH, and Olympiads subsets of the NuminaMATH dataset" by the source card's own description [2][3], but NuminaMath's own source breakdown has no distinct "aime" category - only a combined `amc_aime` bucket of 4,072 rows [4] - and a live sample of this dataset's math rows turned up both open-answer items and AMC-style multiple-choice items (`\textbf{(A)}...\textbf{(E)}`) [5]. Neither this card, the Bespoke Labs card, nor the Sky-T1 card states any decontamination or removal of AIME/AMC benchmark problems before release [1][2][3]; treat any AIME- or AMC-benchmarked evaluation of a model trained on this data as unverified for contamination until you check it yourself.**

**Use it for**: reasoning-trace SFT - long chain-of-thought distillation from a reasoning teacher model, matching the SFT method card. The `messages` column (`role`/`content`, no system turn) is TRL's conversational chat format and is what an SFT trainer such as TRL's `SFTTrainer` consumes directly [1][6]; the `conversations` column (`from`/`value`) is the ShareGPT-style equivalent for non-TRL frameworks. The `system` column holds a separate long system-prompt string used by neither `messages` nor `conversations`, and must be prepended by the caller if wanted.

**Licence**: not stated on this repository - `cardData` carries no `license` key and the card body states none [1]. The upstream `bespokelabs/Bespoke-Stratos-17k` is licensed `apache-2.0` [2], but no source states that licence transfers to this re-serialization; treat the licence as unconfirmed for this specific repo.

**Shape**: 16,710 rows in one config (`default`), split `train` 16,610 / `test` 100, three columns (`system`, `conversations`, `messages`) [7][8].

**Hold out**: `test` (100 rows) from any training run against `train` (16,610 rows) [7]. Beyond that split, no source states which specific rows overlap a named benchmark, so no row-level contamination list exists to hold out - the bolded restriction above is a usage-shape caution, not a resolved row count.

**Origin**: builder Hugging Face (`HuggingFaceH4`), reformatting Bespoke Labs' dataset; the reasoning traces are DeepSeek-R1 generations with rejection-sampling-based correctness filtering by Bespoke Labs [2]. Hub API at the check date: `downloads` 2,973, `downloadsAllTime` 29,072, `likes` 19 [9][10].

**Trained-on-by**: at the check date the Hub API lists 115 models tagged `dataset:HuggingFaceH4/Bespoke-Stratos-17k` [11], including Hugging Face's own `loubnabnl/Llama-8B-Bespoke-H4` family, whose model tags name `library:trl` and `sft` [11]. The upstream `bespokelabs/Bespoke-Stratos-17k` (not this mirror) trained Bespoke-Stratos-32B and Bespoke-Stratos-7B, per its own card [2].

**Introduced by**: no paper - the Bespoke Labs dataset card [2], plus a Bespoke Labs blog post ("Bespoke-Stratos: The unreasonable effectiveness of reasoning distillation") cited in that card's own BibTeX entry [2].

## Shape

Rows served and splits (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 16,610 |
| `test` | 100 |
| total | 16,710 |

One config, `default`, with three columns (datasets-server `/info`) [8]:

| column | dtype |
| --- | --- |
| `system` | string |
| `conversations` | list<struct<from: string, value: string>> |
| `messages` | list<struct<role: string, content: string>> |

Sizes (datasets-server `/size`) [7]: 253,238,353 bytes of Parquet download, 588,450,890 bytes decoded in memory. The card's `dataset_info` block states the same split counts and a slightly different in-memory byte figure (599,464,968 total, computed as non-integer sub-byte estimates per split) [1]. No source states token or sequence-length statistics for this dataset.

## Quality

- The reasoning traces are DeepSeek-R1 outputs filtered by rejection sampling: the source card states this "involves filtering out reasoning traces with incorrect solutions," verified with GPT-4o-mini for math instead of Sky-T1's original parsing logic, which it says raised the retained-correct-solution rate "from 25% to 73%" [2].
- The source card states the Sky-T1 pipeline was ported into Bespoke Labs' Curator tool and the whole DeepSeek-R1 generation run took about 1.5 hours at a cost of $800 [2].
- No source states a measured duplicate rate or an explicit contamination check against any benchmark for this dataset or its upstream; the AIME/AMC composition risk in the opening paragraph is the only source-stated compositional detail bearing on contamination, not a stated measurement of it.
- A live sample of this mirror's `train` split (30 rows read across offsets 0 and 200) shows a mix of multiple-choice items carrying `\textbf{(A)}`-style answer choices, open-answer math items, and Python coding-function prompts beginning "Generate an executable Python function..." [5], consistent with the source card's stated mix of math, code (APPS/TACO), and science/puzzle (STILL-2) content [2][3].

## Load it

Pin the revision this card's numbers were read at (the shortlist's recorded commit) [12]:

```python
import datasets

REV = "384140b6bba79a3f50697536c3b4192b86ddce1d"
train = datasets.load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", revision=REV, split="train")  # 16,610 rows
test = datasets.load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", revision=REV, split="test")    # 100 rows - hold out
```

**Trap**: the `system` column is not folded into either `conversations` or `messages` - both start directly with the user turn [5]. A trainer that reads `messages` as the full prompt without also reading `system` silently drops the long system instruction the source data was generated under, which begins "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process..." [5].

## Neighbors

- `bespokelabs/Bespoke-Stratos-17k` - the source this mirror reformats, same 17k-row pool but a single `train` split (no `test`) and only the `system`/`conversations` columns, no `messages` [2][13]. This mirror exists to add the TRL `messages` format on top of it; prefer the source repo if the `messages` column is not needed, since it carries the `apache-2.0` licence tag this mirror lacks [2].
- `bespokelabs/Bespoke-Stratos-35k` - a larger successor pool from the same builder, 35,028 rows in a single `train` split, `system`/`conversations` columns only [14].
- `NovaSky-AI/Sky-T1_data_17k` - the original Berkeley Sky-T1 data this pipeline replicates, a 17k-row JSON file (not a Parquet dataset with row/split metadata on the Hub) generated with QwQ instead of DeepSeek-R1 [3].
- Dozens of further community re-formats and reductions of the same pool exist on the Hub (e.g. `natolambert/bespoke_stratos_17k_converted`, `Seungyoun/Bespoke-Stratos-17k-revised-format`); none is a Bespoke Labs or Hugging Face release and none is evaluated here [15].

## A row

One config and one shared row shape across `train` and `test`; both splits were sampled and matched this schema [5][16]. From `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [5], with the reasoning trace truncated:

```json
{
  "system": "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. [...]",
  "conversations": [
    {"from": "user", "value": "Return your final response within \\boxed{}. Ms. Blackwell gives an exam to two classes. The mean of the scores of the students in the morning class is $84$, and the afternoon class's mean score is $70$. The ratio of the number of students in the morning class to the number of students in the afternoon class is $\\frac{3}{4}$. What is the mean of the scores of all the students?\n$\\textbf{(A)} ~74 \\qquad\\textbf{(B)} ~75 \\qquad\\textbf{(C)} ~76 \\qquad\\textbf{(D)} ~77 \\qquad\\textbf{(E)} ~78$"},
    {"from": "assistant", "value": "<|begin_of_thought|>\n\nOkay, let's see. I need to find the mean score of all the students combined from both the morning and afternoon classes. Hmm, the problem gives me the means of each class and the ratio of the number of students. [...]"}
  ],
  "messages": [
    {"role": "user", "content": "Return your final response within \\boxed{}. Ms. Blackwell gives an exam to two classes. [...] $\\textbf{(A)} ~74 \\qquad\\textbf{(B)} ~75 \\qquad\\textbf{(C)} ~76 \\qquad\\textbf{(D)} ~77 \\qquad\\textbf{(E)} ~78$"},
    {"role": "assistant", "content": "<|begin_of_thought|>\n\nOkay, let's see. I need to find the mean score of all the students combined from both the morning and afternoon classes. [...]"}
  ]
}
```

## Where it came from

Hugging Face's `HuggingFaceH4` org built this repository as a reformatting of `bespokelabs/Bespoke-Stratos-17k`, adding the TRL-style `messages` column while keeping the source's `system` and `conversations` columns; its own card states nothing further about the reformatting process [1]. Bespoke Labs built the upstream pool: crowdworker-free, model-only generation, sampling reasoning traces from DeepSeek-R1 over a Sky-T1-style question pool of "5k coding data from APPs and TACO, and 10k math data from AIME, MATH, and Olympiads subsets of the NuminaMATH dataset, and 1k science and puzzle data from STILL-2," noting "the exact problems included may differ due to the rejection sampling process" versus Sky-T1's own set [2]. The Sky-T1 pipeline being replicated is described in NovaSky-AI's own card in the same terms - "5k coding data from APPs and TACO, and 10k math data from AIME, MATH, and Olympiads subsets of the NuminaMATH dataset," plus "1k science and puzzle data from STILL-2" [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] HuggingFaceH4/Bespoke-Stratos-17k dataset card (README) and Hub API record. https://huggingface.co/datasets/HuggingFaceH4/Bespoke-Stratos-17k/raw/main/README.md and https://huggingface.co/api/datasets/HuggingFaceH4/Bespoke-Stratos-17k?full=true - card text, `dataset_info`, licence absence, split/column declarations. Fetched 2026-08-11.

[2] bespokelabs/Bespoke-Stratos-17k dataset card (README). https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k/raw/main/README.md - pipeline description, source-model composition, rejection-sampling filtering, models trained, licence, BibTeX citation of the Bespoke Labs blog post. Fetched 2026-08-11.

[3] NovaSky-AI/Sky-T1_data_17k dataset card (README). https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k/raw/main/README.md - the original Sky-T1 pipeline's stated data composition (AIME/MATH/Olympiads subsets of NuminaMATH, APPS/TACO, STILL-2). Fetched 2026-08-11.

[4] AI-MO/NuminaMath-CoT dataset card (README), source breakdown table. https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/raw/main/README.md - lists `amc_aime` (4,072), `math` (7,478), `olympiads` (150,581) among NuminaMath's named source categories; no separate `aime` category exists. Fetched 2026-08-11.

[5] datasets-server first-rows endpoint, `train` and `test` splits. https://datasets-server.huggingface.co/first-rows?dataset=HuggingFaceH4%2FBespoke-Stratos-17k&config=default&split=train and &split=test - row schema, system-prompt text, and the sampled MCQ/open/code row mix at offset 0 (10 rows per split). Fetched 2026-08-11.

[6] TRL dataset formats documentation. https://raw.githubusercontent.com/huggingface/trl/main/docs/source/dataset_formats.md - conversational format definition (`role`/`content` `messages` lists) consumed directly by TRL trainers. A `main`-branch file, unpinned and mutable. Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=HuggingFaceH4%2FBespoke-Stratos-17k Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=HuggingFaceH4%2FBespoke-Stratos-17k Fetched 2026-08-11.

[9] Hugging Face Hub API record for HuggingFaceH4/Bespoke-Stratos-17k. https://huggingface.co/api/datasets/HuggingFaceH4/Bespoke-Stratos-17k?full=true - `downloads`, `likes`, `sha`, `lastModified`. Fetched 2026-08-11.

[10] Hugging Face Hub API record, `downloadsAllTime` expansion. https://huggingface.co/api/datasets/HuggingFaceH4/Bespoke-Stratos-17k?expand[]=downloadsAllTime Fetched 2026-08-11.

[11] Hugging Face Hub API model-search endpoint, filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:HuggingFaceH4/Bespoke-Stratos-17k - 115 tagged models, including the `loubnabnl/Llama-8B-Bespoke-H4` family with `library:trl` and `sft` tags. Fetched 2026-08-11.

[12] The corpus shortlist row for `HuggingFaceH4/Bespoke-Stratos-17k`, supplied with this card's request - `commit` field, read back for the pinned revision in Load it. Checked 2026-08-11.

[13] bespokelabs/Bespoke-Stratos-17k Hub API record and datasets-server info/size endpoints. https://huggingface.co/api/datasets/bespokelabs/Bespoke-Stratos-17k?full=true, https://datasets-server.huggingface.co/info?dataset=bespokelabs%2FBespoke-Stratos-17k, https://datasets-server.huggingface.co/size?dataset=bespokelabs%2FBespoke-Stratos-17k - single `train` split, 16,710 rows, two columns (`system`, `conversations`), `apache-2.0` licence tag. Fetched 2026-08-11.

[14] bespokelabs/Bespoke-Stratos-35k dataset card and datasets-server size endpoint. https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-35k/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=bespokelabs%2FBespoke-Stratos-35k - 35,028-row single `train` split, `system`/`conversations` columns. Fetched 2026-08-11.

[15] Hugging Face Hub API dataset-search endpoint. https://huggingface.co/api/datasets?search=Bespoke-Stratos - enumerates community derivatives of the Bespoke-Stratos pool. Fetched 2026-08-11.

[16] datasets-server first-rows endpoint, `test` split, row 0. https://datasets-server.huggingface.co/first-rows?dataset=HuggingFaceH4%2FBespoke-Stratos-17k&config=default&split=test - confirms the same three-column schema as `train`. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data, with an unresolved contamination caution rather than a resolved one: the screening row's flag is a `rule-risk` on AIME2025-style evaluation, and this card's own reading of the sources confirms the underlying fact the flag points at - the math pool is described as drawn from NuminaMath's AIME/MATH/Olympiads subsets, but NuminaMath's own source breakdown has no distinct AIME category (only a mixed `amc_aime` bucket), a live sample of rows shows AMC-style multiple-choice items, and no source in the chain states a decontamination or removal step [2][3][4][5]. This is not resolved into a specific held-out row count because no source names which rows overlap which benchmark; a user targeting AIME- or AMC-adjacent evaluation should treat that as open until checked directly.

### The screening row

The row's own note [12]: "reasoning traces distilled from DeepSeek-R1 (Bespoke Labs replication of Sky-T1); TRL-formatted mirror." Its flag, class `rule-risk` (`aime2025`): "Card calls it a Sky-T1 pipeline replication; rows are AMC-style competition problems from the NuminaMath pool, no removal claim."
