# garage-bAInd/Open-Platypus

24,926 instruction/output rows in Alpaca format, blended from eleven STEM- and logic-focused sources, deduplicated against benchmark test sets and used to train the Platypus2 model family.

**garage-bAInd/Open-Platypus** is a curated blend built by the garage-bAInd team and introduced in "Platypus: Quick, Cheap, and Powerful Refinement of LLMs" [1], which describes it as focused on improving LLM logical reasoning and states it was used to train the Platypus2 models [2]. The rows are collected from eleven named sources - PRM800K, MATH, ScienceQA, SciBench, ReClor, TheoremQA, `nuprl/leetcode-solutions-python-testgen-gpt4`, `jondurbin/airoboros-gpt4-1.4.1`, `TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k`, ARB, and `timdettmers/openassistant-guanaco` - filtered by keyword search for STEM/logic content and then deduplicated with Sentence Transformer embeddings at an 80%-similarity threshold [2]. The paper frames this as instruction-tuning data in the Alpaca format (instruction, input, output), with input left empty for most sources and populated only for the small multiple-choice subsets [1]. It lives at https://huggingface.co/datasets/garage-bAInd/Open-Platypus . **Three restrictions apply at use time. First, licensing is mixed and not resolved to one SPDX id: the card's own per-source table marks ScienceQA "Creative Commons Attribution-NonCommercial-ShareAlike 4.0" and ReClor "Non-commercial" [2], so the blended dataset as a whole cannot be treated as permissively licensed. Second, 564 of the rows are labeled `data_source: theoremqa`, and the only public TheoremQA release on the Hub ships a single 800-row `test` split with no train split [3] - so training on this subset risks training on (a large fraction of) the TheoremQA benchmark's own held-out test questions, independent of the dataset's own near-duplicate filtering. Third, 1,317 of the rows are labeled `data_source: scienceqa`; the public `derek-thomas/ScienceQA` release does carry an independent train split, unlike TheoremQA, but it also carries a 4,241-row `test` split [4], and no source read for this card states which ScienceQA split(s) these 1,317 rows were drawn from - so overlap with the ScienceQA test set cannot be ruled out from what was checked here.**

**Use it for**: SFT on instruction/input/output triples for STEM reasoning and logic tasks - the SFT method card, not preference or reward-model training (there is no chosen/rejected pairing here) [2]. The rows already sit in a single-turn instruction format usable directly by an Alpaca-style SFT collator; no `chat_template` or dialect is declared on the repository. **Do not train on this dataset if you plan to evaluate on the TheoremQA benchmark, and treat any ScienceQA-benchmarked evaluation as at risk until the source split of the 1,317 `scienceqa` rows is confirmed**; treat the ScienceQA- and ReClor-licensed portions as non-commercial-restricted for any downstream release.

**Licence**: no repository-level SPDX id - the Hub API record's `cardData` carries no `license` key [5]. The dataset card instead gives a per-source table: PRM800K, MATH, SciBench MIT; TheoremQA MIT; `tigerbot-kaggle-leetcodesolutions-en-2k` and `openassistant-guanaco` Apache-2.0; `leetcode-solutions-python-testgen-gpt4` "None listed"; `airoboros-gpt4-1.4.1` "other"; ARB "CC BY 4.0"; ScienceQA CC BY-NC-SA 4.0; ReClor "Non-commercial" [2]. The one catch: the CC BY-NC-SA and non-commercial entries mean the blend is not cleanly redistributable or usable commercially as a whole.

**Shape**: one config (`default`), one split (`train`, 24,926 rows), four string columns: `instruction`, `input`, `output`, `data_source` [6][7].

**Hold out**: nothing shipped as a separate split - this repository has only `train`. If evaluating on the TheoremQA benchmark, exclude the 564 rows where `data_source == "theoremqa"` (offsets 14944-15507 in the served `train` split, read via the datasets-server `/rows` endpoint) [8]. If evaluating on ScienceQA, treat the 1,317 rows where `data_source == "scienceqa"` (offsets 13011-14327, same endpoint) [8] as at risk too: `derek-thomas/ScienceQA` publishes a 4,241-row `test` split alongside its train split [4], and this card's own reading does not establish which split these 1,317 rows came from. The screening flag names both risks together [9]; this line is where the card settles them.

**Origin**: built by garage-bAInd from eleven upstream sources, with outputs drawn verbatim from those sources rather than freshly generated for this repository [2]. Hub API at the check date: 13,621 downloads (window), 303,743 downloads all-time, 422 likes [5].

**Trained-on-by**: the garage-bAInd Platypus2 family itself - `Platypus2-13B`, `Platypus2-70B`, `Platypus2-70B-instruct` - per the dataset card's own statement that Open-Platypus "was used to train the Platypus2 models" [2]. The paper describes the `Camel-Platypus2` and `Stable-Platypus2` variants as produced by merging the Platypus LoRA adapter into other base models, e.g. "the Camel-Platypus2-70B model, when merged with the Platypus adapter" [1]. Independently of garage-bAInd, `Open-Orca/OpenOrca-Platypus2-13B` names `garage-bAInd/Open-Platypus` directly as one of its training datasets in its own model-card metadata, and states it merges `garage-bAInd/Platypus2-13B` with `Open-Orca/OpenOrcaxOpenChat-Preview2-13B` [10].

**Introduced by**: [1] (Lee, Hunter, and Ruiz).

## Shape

Rows and splits (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 24,926 |

One config, `default`, four columns (datasets-server `/info`) [7]:

| column | dtype |
| --- | --- |
| `input` | string |
| `output` | string |
| `instruction` | string |
| `data_source` | string |

`data_source` is a per-row provenance tag, not a config or split; reading the served rows at their source boundaries via the datasets-server `/rows` endpoint gives an exact count per upstream source that the card's own text does not state [8]:

| `data_source` value | rows | offsets | matches README source |
| --- | --- | --- | --- |
| `MATH/PRM-800K` | 12,298 | 0-12297 | PRM800K + MATH (merged label) |
| `ARB` | 713 | 12298-13010 | ARB |
| `scienceqa` | 1,317 | 13011-14327 | ScienceQA |
| `scibench` | 616 | 14328-14943 | SciBench |
| `theoremqa` | 564 | 14944-15507 | TheoremQA |
| `reclor` | 4,530 | 15508-20037 | ReClor |
| `tigerbot-kaggle` | 386 | 20038-20423 | `TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k` |
| `leetcode_ne` | 1,100 | 20424-21523 | `nuprl/leetcode-solutions-python-testgen-gpt4` |
| `airoboros` | 2,605 | 21524-24128 | `jondurbin/airoboros-gpt4-1.4.1` |
| `guanaco` | 797 | 24129-24925 | `timdettmers/openassistant-guanaco` |

These counts were obtained by binary-searching the source boundaries in the served `train` split (the rows are stored contiguously by `data_source`), and they sum to exactly 24,926 [8]. Byte sizes (datasets-server `/size`) [6]: 15,565,850 bytes of original Parquet download, 26,178,830 bytes decoded in memory. No source states token- or sequence-length statistics for this repository.

The size, info, and rows counts above come from datasets-server endpoints that take no revision parameter and always reflect the current default-branch state; they are not covered by the commit pin used in Load it below, so a re-fetch at a later date could show different numbers even though the pinned file load stays reproducible.

## Quality

- The card states approximately 200 training questions were removed for appearing in Hugging Face benchmark test sets, and points to the paper for the method [2].
- The paper's own Table 1 gives per-source exact-duplicate ("leaked question") counts against benchmark test sets for the sources actually included in Open-Platypus: PRM800K 77, MATH 77, ScienceQA 0, SciBench 0, ReClor 0, TheoremQA 0, leetcode-solutions 0, airoboros-gpt4-1.4.1 13, tigerbot-kaggle 0, ARB 0, openassistant-guanaco 13 (sum 180) [1]. The paper defines "duplicate" contamination as the only category counted here; two further categories it calls "gray-area" and "similar but different" were also removed from training but are not tallied in this table [1].
- Despite the paper's own check reporting 0 exact-duplicate leaks for TheoremQA, that check compares training rows against a fixed list of benchmark test sets and does not change the structural fact established above: TheoremQA's only public release is an 800-row `test` split with no separate train pool [3], so the 564 `theoremqa`-sourced training rows here are drawn from that same 800-question pool by construction, not from an independent training split.
- ScienceQA differs structurally: `derek-thomas/ScienceQA` does carry an independent train split (12,726 rows) alongside its 4,241-row `test` split [4], so its 1,317 rows in Open-Platypus are not necessarily drawn from the test pool the way TheoremQA's are. But the paper's own 0-leaked-question count for ScienceQA only rules out near-duplicate text matches [1], and no source read for this card states which ScienceQA split(s) fed these 1,317 rows, so overlap with the 4,241-row test set cannot be ruled out from what was checked here.
- Of a sample of roughly 360 served rows read across all ten `data_source` boundaries (13 batches of 20 rows plus the first 100 rows, all via the datasets-server `/rows` and `/first-rows` endpoints), `input` was empty in all but a small minority: non-empty for `reclor` rows sampled at offsets 17000-17002 (each populated with "Choose A, B, C or D as your solution.") and for one `scienceqa` row read at offset 13013; `ARB` rows sampled at offsets 13000-13010 all had empty `input`, differing from the paper's general description that ARB uses the same multiple-choice input format as ReClor [1][8].
- No source states a duplicate-rate, contamination-rate, or annotator-agreement figure beyond the counts above.

## Load it

```python
import datasets

REV = "37141edbdb7826378cce118c46a109b813e1f038"  # main at the check date
train = datasets.load_dataset("garage-bAInd/Open-Platypus", revision=REV, split="train")  # 24,926 rows
```

**Trap**: there is only one split (`train`) and no shipped eval/test split - there is nothing to hold out structurally except the `theoremqa`- and `scienceqa`-sourced rows named above, which a reader must filter out manually, e.g. `train.filter(lambda r: r["data_source"] not in ("theoremqa", "scienceqa"))`, before any TheoremQA- or ScienceQA-benchmarked evaluation. The four-column Alpaca schema is uniform across all ten `data_source` values, but `input` is populated for only a minority of rows (see Quality); a collator that assumes `input` is always present or always empty will silently mishandle the other case.

## Neighbors

One reformatted sibling was found, from a different builder: `chargoddard/Open-Platypus-Chat` converts this dataset into sharegpt-style chat rows, and its own card states it is Open-Platypus converted to sharegpt format with a handful of potential refusals removed [11]. It carries 24,887 rows [11], 39 fewer than this release's 24,926 [6], consistent with a small number of rows being removed rather than a resample. Its schema is a single `conversations` column holding a list of `{from, value}` turns, not the four-column instruction/input/output/data_source shape here [11]; row 0 of both repositories is the identical spinner-probability MATH problem, reshaped from instruction/output fields into a two-turn `conversations` list [7][11]. Prefer this original garage-bAInd release for instruction/output-shaped SFT collators; reach for `chargoddard/Open-Platypus-Chat` only when a `conversations`-shaped chat collator is required directly.

## A row

One config, one split, one schema - a row from `config="default"`, `split="train"`, `row_idx=0`, `data_source="MATH/PRM-800K"` (datasets-server `/first-rows`) [12]:

```json
{
  "input": "",
  "instruction": "A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.",
  "output": "To find the probability of the spinner landing on $C$, I need to subtract the probabilities of the spinner landing on $A$ and $B$ from $1$, since the sum of the probabilities of all possible outcomes is $1$. I can write this as an equation: $P(C) = 1 - P(A) - P(B)$. [...] I get: $P(C) = \\frac{1}{4}$. ",
  "data_source": "MATH/PRM-800K"
}
```

A row where `input` is populated, from `row_idx=17000`, `data_source="reclor"` [8]:

```json
{
  "input": "Choose A, B, C or D as your solution.",
  "instruction": "Area resident: Childhood lead poisoning has declined steadily since the 1970s, when leaded gasoline was phased out and lead paint was banned. But recent statistics indicate that 25 percent of this ar[...]",
  "output": "C",
  "data_source": "reclor"
}
```

## Where it came from

Built by garage-bAInd from eleven upstream sources - PRM800K, MATH, ScienceQA, SciBench, ReClor, TheoremQA, `nuprl/leetcode-solutions-python-testgen-gpt4`, `jondurbin/airoboros-gpt4-1.4.1`, `TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k`, ARB, and `timdettmers/openassistant-guanaco` - selected by keyword search for STEM and logic content, then deduplicated by Sentence Transformer cosine similarity at an 80% threshold, keeping the more verbose answer when a near-duplicate pair was found [2][1]. The paper additionally ran a contamination check against Hugging Face benchmark test sets, categorizing near-matches as duplicate, gray-area, or similar-but-different, and removing all three categories from the training set [1]. Outputs are the pre-existing answers/solutions from each upstream source (e.g., PRM800K step-by-step solutions, ARB expert-written solutions), not freshly generated for this repository [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision; the Shape/Quality section notes that the datasets-server endpoints used elsewhere on this card take no revision parameter and are not covered by that pin.

[1] Lee, Hunter, and Ruiz, "Platypus: Quick, Cheap, and Powerful Refinement of LLMs", 2023. https://arxiv.org/abs/2308.07317 - the origin paper; contamination-check methodology, Table 1 per-source leaked-question counts, Alpaca-format description, ARB/ReClor input-formatting description, Camel-Platypus2/Stable-Platypus2 merge description. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2308.07317). Current title read from the live abs page. Fetched 2026-08-11.

[2] garage-bAInd/Open-Platypus dataset card (README). https://huggingface.co/datasets/garage-bAInd/Open-Platypus/raw/main/README.md - source list, per-source license table, contamination-check statement, training-use statement. Fetched 2026-08-11.

[3] TIGER-Lab/TheoremQA dataset card and datasets-server size endpoint. https://huggingface.co/datasets/TIGER-Lab/TheoremQA/raw/main/README.md and https://datasets-server.huggingface.co/size?dataset=TIGER-Lab%2FTheoremQA - single 800-row `test` split, no train split. Fetched 2026-08-11.

[4] derek-thomas/ScienceQA datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=derek-thomas%2FScienceQA - train (12,726), validation (4,241), test (4,241) split sizes. Fetched 2026-08-11.

[5] Hugging Face Hub API record for garage-bAInd/Open-Platypus. https://huggingface.co/api/datasets/garage-bAInd/Open-Platypus?full=true - `cardData` (no `license` key), `sha`, `downloads`, `likes`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=garage-bAInd%2FOpen-Platypus Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=garage-bAInd%2FOpen-Platypus Fetched 2026-08-11.

[8] datasets-server rows endpoint, sampled at offsets 0, 1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000, 23000, and 24800 to locate `data_source` boundaries and check `input` population. https://datasets-server.huggingface.co/rows?dataset=garage-bAInd%2FOpen-Platypus&config=default&split=train&offset=<n>&length=20 Fetched 2026-08-11.

[9] The corpus screening row for `garage-bAInd/Open-Platypus`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[10] Open-Orca/OpenOrca-Platypus2-13B model card. https://huggingface.co/Open-Orca/OpenOrca-Platypus2-13B/raw/main/README.md - `datasets: - garage-bAInd/Open-Platypus` in its metadata, and its description as a merge of `garage-bAInd/Platypus2-13B` and `Open-Orca/OpenOrcaxOpenChat-Preview2-13B`. Fetched 2026-08-11.

[11] chargoddard/Open-Platypus-Chat Hub API record and datasets-server size/info/first-rows endpoints. https://huggingface.co/api/datasets/chargoddard/Open-Platypus-Chat?full=true , https://datasets-server.huggingface.co/size?dataset=chargoddard%2FOpen-Platypus-Chat , https://datasets-server.huggingface.co/info?dataset=chargoddard%2FOpen-Platypus-Chat , https://datasets-server.huggingface.co/first-rows?dataset=chargoddard%2FOpen-Platypus-Chat&config=default&split=train - description, row count, `conversations` schema, row 0 content. Fetched 2026-08-11.

[12] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=garage-bAInd%2FOpen-Platypus&config=default&split=train Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable for SFT, with the TheoremQA subset excluded before any TheoremQA-benchmarked evaluation, and the ScienceQA subset treated as at-risk until its source split is confirmed. This rests on facts already established above: the dataset card documents no chosen/rejected pairing, so it is SFT-shaped, not preference data [2]; and the screening row's flag identifies the same TheoremQA and ScienceQA overlap risk this card resolves in the Hold out line and the opening paragraph [9].

### The screening row

The row's own note [9]: "logic/STEM blend of PRM800K, MATH, ScienceQA, SciBench, ReClor, TheoremQA, GPT-4 leetcode solutions and airoboros." Its flag [9]: "contamination risk, not a duplicate — it embeds TheoremQA (which appears in this same range as a test-only benchmark) and ScienceQA; training on it contaminates both." (flag class: contamination)
