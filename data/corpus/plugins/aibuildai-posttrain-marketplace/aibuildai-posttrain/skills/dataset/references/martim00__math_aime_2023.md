# martim00/math_aime_2023

975 instruction/output rows of AIME competition math problems paired with worked solutions, still in raw MediaWiki markup, split 682 train / 293 test.

**martim00/math_aime_2023** pairs an `instruction` column holding an AIME-style problem statement with an `output` column holding a step-by-step solution, both wrapped in an Alpaca-style `### Instruction:`/`### Response:`/`### Solution:` template [1]. The dataset card itself is only YAML front matter with no prose - no builder statement, no description, no license field [1]. Despite the repo name, a sample of 375 of the 975 rows (182 of 682 train rows at offsets 0-99 and 600-681; 193 of 293 test rows at offsets 0-99 and 200-292) shows problems and solutions referencing AIME years 2005, 2007, 2021 and 2022 as well as 2023, and one solution embeds a file reference to `[[File:2022 AIME II 3.png|300px|right]]` [2] - this is not an AIME-2023-only set. **Decontaminate before using this as reasoning-trace SFT data: the corpus screening record flags this as a rule-risk because it packages raw AIME problems and solutions as training rows, an AIME-2023 problem was confirmed present, and although probes for AIME-2024 and AIME-2025 items found no matches, matching items are not removed from the set on principle [3].** It lives at https://huggingface.co/datasets/martim00/math_aime_2023 .

**Use it for**: reasoning-trace SFT (instruction-in, worked-solution-out) - maps to the SFT method card's instruction/response format, after decontaminating against AIME evaluation benchmarks (e.g. `math-ai/aime25`, the 30-problem AIME 2025 benchmark) as noted above [3][4]. Not preference data - there is no chosen/rejected pair, only one `output` per `instruction` [1].

**Licence**: not stated. `cardData` carries no `license` key and the README body has no license mention or badge [1]; the repo is public and ungated [5].

**Shape**: 975 rows, one config (`default`), two string columns (`instruction`, `output`), split `train` 682 / `test` 293 [1][6].

**Hold out**: the `test` split (293 rows) for basic train/test separation, and additionally decontaminate `train` (682 rows) against external AIME evaluation sets before using it for a scored run - the rule-risk flag above is what makes this the right call, since the screening probe's zero hits for AIME-2024/2025 items do not mean the set is clear, only that a probe on this particular data did not find a match [3].

**Origin**: uploaded by Hub user `martim00` on 2024-04-05; no source states who wrote the solutions, but their prose style and embedded `[[wiki links]]`, `<math>` tags, `<asy>` Asymptote code and `[[File:...]]` image references match AoPS Wiki's AIME problem/solution page markup, the same markup convention a neighboring dataset explicitly names as its source [2][7]. Hub API at the check date: 2,729 downloads, 6,832 all-time downloads, 0 likes [5].

**Trained-on-by**: none found - no adoption evidence surfaced in any source fetched for this card.

**Introduced by**: no paper - the dataset card [1], which is YAML metadata only and states no origin.

## Shape

Rows and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 682 |
| `test` | 293 |
| total | 975 |

One config, `default`, two columns (datasets-server `/info`) [8]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `output` | string |

Sizes (datasets-server `/size`) [6]: 1,042,346 bytes of original/parquet download, 2,205,691 bytes decoded in memory. No source states sequence-length or token statistics for this dataset; none is invented here.

## Quality

- The card carries no prose - no annotation process, no measured contamination rate, no duplicate-rate figure [1].
- Of the 375 sampled rows (182 train at offsets 0-99 and 600-681; 193 test at offsets 0-99 and 200-292), every `instruction` begins with the literal string `### Instruction:` and every `output` begins with `### Solution:`, so the template is consistent across the sample [2].
- Across that same 375-row sample, no `instruction` string in the sampled train rows matched any `instruction` string in the sampled test rows [2] - offsets 100-599 of `train` and 100-199 of `test` were not read, so this does not cover the full split.
- The rule-risk flag on this dataset is a corpus-screening judgment, not a measured contamination rate: it is that raw AIME problems are training rows at all, regardless of which exact years matched a probe [3].

## Load it

```python
import datasets

REV = "9ba3750276136bcb3c4ef51afe6d011a9060d926"  # main at the check date
train = datasets.load_dataset("martim00/math_aime_2023", revision=REV, split="train")  # 682 rows
test = datasets.load_dataset("martim00/math_aime_2023", revision=REV, split="test")    # 293 rows - hold out
```

**Trap**: both `instruction` and `output` still carry the raw MediaWiki/Alpaca template (`### Instruction:` ... `### Response:` inside `instruction`; `### Solution:` at the head of `output`) and unrendered `<math>`, `[[wiki link]]`, `<asy>` and `[[File:...]]` markup [2] - a consumer expecting clean LaTeX or plain text must strip or render this markup itself; no source states that any cleaning step has been applied.

## Neighbors

- `di-zhang-fdu/AIME_1983_2024` - 933 rows spanning AIME 1983-2023 plus part of 2024, in `problem`/`answer`-style columns rather than instruction/output prose; its own README states it is sourced from the AoPS wiki AIME problem/solution pages and says explicitly "Disclaimer: This is a Benchmark dataset! Do not using in training!" [7]. Prefer this dataset over that one only where instruction-tuned prose solutions (not just final answers) are needed for SFT; `di-zhang-fdu/AIME_1983_2024`'s own card rules it out for training.
- `math-ai/aime25` - the AIME 2025 evaluation benchmark itself, 30 `problem`/`answer`/`id` rows under an Apache-2.0 license [4]. This is the kind of external eval set the decontamination restriction above is protecting against, not a training-data alternative.
- `AI-MO/aimo-validation-aime` - surfaced in a Hub search for "aime" and referenced by `di-zhang-fdu/AIME_1983_2024`'s README as covering AIME 2024 part 1 [7]; not otherwise fetched for this card, so its shape is not described here.

## A row

One config, one schema, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/rows`) [2], with the long `output` field truncated:

```json
{
  "instruction": "### Instruction:\nThe increasing [[geometric sequence]] <math>x_{0},x_{1},x_{2},\\ldots</math> consists entirely of [[integer|integral]] powers of <math>3.</math> Given that\n\n<math>\\sum_{n=0}^{7}\\log_{3}(x_{n}) = 308</math> and <math>56 \\leq \\log_{3}\\left ( \\sum_{n=0}^{7}x_{n}\\right ) \\leq 57,</math>\n\nfind <math>\\log_{3}(x_{14}).</math>\n\n### Response:\n",
  "output": "### Solution:\nSuppose that <math>x_0 = a</math>, and that the common [[ratio]] between the terms is <math>r</math>. \n\n\nThe first conditions tells us that <math>\\log_3 a + \\log_3 ar + \\ldots + \\log_3 ar^7 = 308</math>. Using the rules of [[logarithm]]s, we can simplify that to <math>\\log_3 a^8r^{1 + 2 + \\ldots + 7} = 308</math>. Thus, <math>a^8r^{28} = 3^{308}</math>. [...]"
}
```

## Where it came from

The dataset card names no builder, no upstream pool, no collection method and no generating model - the only content in the README is a `dataset_info` YAML block describing the schema and split sizes [1]. What can be read directly from the served rows: every `instruction`/`output` pair is formatted as an Alpaca-style instruction/response template wrapping a math competition problem and its worked solution, and the wiki markup inside that template ([[double-bracket links]], `<math>` LaTeX tags, `<asy>` Asymptote figure code, `[[File:...]]` image embeds) matches the formatting convention of AoPS Wiki's AIME problem/solution pages, the same convention a neighboring dataset cites by name as its source [2][7]. No source states that this particular repository was built by scraping that site.

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] martim00/math_aime_2023 dataset repository, https://huggingface.co/datasets/martim00/math_aime_2023 , with its card (README) at https://huggingface.co/datasets/martim00/math_aime_2023/raw/main/README.md and Hub API record at https://huggingface.co/api/datasets/martim00/math_aime_2023?full=true - schema, split sizes, absence of license and prose. Fetched 2026-08-11.

[2] datasets-server rows endpoint, martim00/math_aime_2023, config `default`, split `train` (offsets 0-99, 600-681) and split `test` (offsets 0-99, 200-292). https://datasets-server.huggingface.co/rows?dataset=martim00%2Fmath_aime_2023&config=default&split=train&offset=0&length=100 (and the corresponding train offset=600, test offset=0, test offset=200 calls) - row content, template consistency, year references, cross-split instruction comparison. Fetched 2026-08-11.

[3] The corpus screening row for `martim00/math_aime_2023`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[4] math-ai/aime25 dataset card and datasets-server info/size. https://huggingface.co/datasets/math-ai/aime25/raw/main/README.md , https://datasets-server.huggingface.co/info?dataset=math-ai%2Faime25 , https://datasets-server.huggingface.co/size?dataset=math-ai%2Faime25 - AIME 2025 benchmark identity, 30 rows, Apache-2.0 license. Fetched 2026-08-11.

[5] Hugging Face Hub API record for martim00/math_aime_2023. https://huggingface.co/api/datasets/martim00/math_aime_2023?full=true and https://huggingface.co/api/datasets/martim00/math_aime_2023?expand[]=downloadsAllTime - gate status, private status, `sha`, `downloads`, `downloadsAllTime`, `likes`, last-modified date. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=martim00%2Fmath_aime_2023 Fetched 2026-08-11.

[7] di-zhang-fdu/AIME_1983_2024 dataset card and datasets-server info/size. https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/raw/main/README.md , https://datasets-server.huggingface.co/info?dataset=di-zhang-fdu%2FAIME_1983_2024 , https://datasets-server.huggingface.co/size?dataset=di-zhang-fdu%2FAIME_1983_2024 - AoPS wiki source statement, "do not use in training" disclaimer, 933 rows, columns. Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=martim00%2Fmath_aime_2023 Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for reasoning-trace SFT only after decontamination against external AIME benchmarks. This rests on facts established above: the dataset packages raw AIME problems and worked solutions as training rows, spans multiple AIME years despite its 2023-only name, and the screening record's own probe confirmed an AIME-2023 problem present while finding no AIME-2024/2025 matches - without those items being removed from the set [1][2][3].

### The screening row

The row's own note [3]: "AoPS-wiki AIME problems with wiki solutions in raw wiki markup (682 train plus a test split); an AIME-2023 problem present, AIME-2024/2025 probes found nothing." Its flag [3]: "rule-risk: aime2025 - AoPS AIME problems plus wiki solutions as train rows; AIME-2025 probe matched 0, but AIME items are not removed."
