# agentlans/train-of-thought

Alpaca-format instruction-tuning data with think-on/think-off reasoning traces, 969,020 rows in its default config, readapted by the same author from agentlans/think-more's aggregation of nine chain-of-thought datasets.

**agentlans/train-of-thought** is built by Hugging Face user agentlans by reformatting **agentlans/think-more** - agentlans's own compilation of DeepSeek-R1 and OpenAI o1 reasoning traces from nine upstream datasets - into Alpaca-style `instruction`/`input`/`output` triples for instruction-tuned chain-of-thought training; there is no paper behind either release, only the two dataset cards [1][2]. Each of the 969,020 original think-more rows was randomly assigned to "thinking off" (`output` holds only the final answer) or "thinking on" (`output` wraps a `<think>...</think>` trace followed by a `<response>...</response>` answer) [1]. **Because 226,052 of these 969,020 rows (23.3%) come from open-r1/OpenR1-Math-220k and nvidia/OpenMathReasoning - two of think-more's nine source datasets, per its own source table - and the corpus screening flags this pairing as a rule-risk for AIME 2025 overlap, decontaminate against AIME 2025 (and other math-competition eval sets) before scoring a run on this data [2][3].**

It lives at https://huggingface.co/datasets/agentlans/train-of-thought .

**Use it for**: reasoning-trace SFT - Alpaca-style `instruction`+`input` as the prompt and `output` as the target, with the boolean `thinking` column already baking the `<think>...</think><response>...</response>` wrapping into `output` when true; maps to the SFT method card. Decontaminate against AIME 2025 (and any other math-competition eval you score against) before a scored run, per the restriction above.

**Licence**: CC BY 4.0 (`cardData.license` is `cc-by-4.0` on this repo, matching the `license:cc-by-4.0` tag), ungated (`"gated": false`, `"private": false`) [4]. The one catch: the card's own licence section only says to "refer to the original agentlans/think-more...dataset licence for usage terms" because "this adaptation inherits the same licence" [1]; think-more's card in turn states only "Creative Commons Attribution 4.0" as a one-line grant over its combined output, with no source-by-source licence audit of the nine upstream reasoning datasets it draws from [2]. No second licence is stated anywhere in the body - the only "MIT"-like substring in the README is inside the phrase "limit points" in the example JSON, not a licence clause [1].

**Shape**: 4 configs, 1 split each - `train` 969,020 rows (default), `100K` 100,000, `10K` 10,000, `1K` 1,000; the three smaller configs are literal prefixes of `train`, not additional rows [5][6].

**Hold out**: before a scored run, hold out or decontaminate against AIME 2025 (and other math-competition benchmarks) any row whose `source` column reads `open-r1/OpenR1-Math-220k` or `nvidia/OpenMathReasoning` - 226,052 of 969,020 default-config rows (23.3%), computed from think-more's own source-table counts of 64,293 and 161,759 [2]. The corpus screening's own note names this pairing as the AIME 2025 risk [3].

**Origin**: built by Hugging Face user agentlans, who also built the upstream think-more compilation; think-more's own summary names DeepSeek R1 and OpenAI o1 as the generating models for its traces [2], and one of its nine upstream sources, nvidia/OpenMathReasoning, separately names QwQ-32B as a second generation model for its own solutions [7]. The Alpaca-format conversion with the random think-on/off split is agentlans's own [1]. Hub API at the check date: `downloads` 199, `downloadsAllTime` 1,744, `likes` 5 [4].

**Trained-on-by**: None found - neither dataset card names a model or training recipe that used this data, and among agentlans's 276 other Hub dataset repositories, none whose name matches "think", "thought", "reason", or "cot" references it as a training source [1][2][8].

**Introduced by**: no paper - the dataset card [1]; the row composition is documented only in think-more's dataset card [2].

## Shape

Configs and splits, from the datasets-server size endpoint [6]:

| config | split | rows |
| --- | --- | --- |
| train | train | 969,020 |
| 100K | train | 100,000 |
| 10K | train | 10,000 |
| 1K | train | 1,000 |
| total served | | 1,080,020 |

Five columns, identical across every config, from the info endpoint [5]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |
| `thinking` | bool |
| `source` | string |

Byte sizes from the size endpoint [6]: 1,995,796,864 bytes of original downloaded files across the whole dataset, 4,306,302,378 bytes as Parquet, 9,393,224,026 bytes decoded in memory (of which `train` alone accounts for 8,431,825,793). No source states sequence-length or token statistics for any config. Think-more's card states only that its own full decompressed file is 15.1 GB [2], a file-size figure, not a token count.

## Quality

- Think-more's stated cleaning steps, which flow into this release since it is a direct reformat: for prompts with multiple traces, only the shortest was kept; rows with incorrect or annotated-wrong reasoning were removed; entries were deduplicated and rows with missing answers or thoughts were dropped [2].
- Think-more's stated limitations, likewise inherited: coverage is "mostly limited to math and science problems"; content is mostly English "with some Chinese entries"; because only correct and complete traces were kept, the mix is biased "toward easier problems with successful solutions"; and the data carries "the biases of both DeepSeek and ChatGPT" [2].
- The train-of-thought README's own example row shows `"thinking": "on"` as a string, but the column actually served is a boolean: the sampled rows below return `true`/`false`, matching the boolean dtype reported by the info endpoint - the README's example does not match the served dtype [1][5][9].
- Of the first 30 rows read at offset 0 of the `train` config, 13 have `thinking=true` and 17 have `thinking=false`, and the `source` values cover 7 of think-more's nine upstream sets (GeneralReasoning/GeneralThought-430K, open-thoughts/OpenThoughts-114k, nvidia/OpenMathReasoning, open-r1/OpenR1-Math-220k, zwhe99/DeepMath-103K, open-r1/Mixture-of-Thoughts, O1-OPEN/OpenO1-SFT), with FreedomIntelligence/medical-o1-reasoning-SFT and nvidia/OpenCodeReasoning absent from this sample; 4 of the 30 (13%) read `nvidia/OpenMathReasoning` and 3 of the 30 (10%) read `open-r1/OpenR1-Math-220k`, together 7/30 (23%) - close to the 23.3% share computed from the full source-table counts. This is a 30-row sample at offset 0 of one config only, not a claim about the full 969,020-row split [9].

## Load it

```python
import datasets

REV = "3a5ea3a5c42a70eb3d9d271427fe504c5c06d8cc"  # main at the check date
train = datasets.load_dataset("agentlans/train-of-thought", split="train", revision=REV)               # default config, 969,020 rows
sample_100k = datasets.load_dataset("agentlans/train-of-thought", "100K", split="train", revision=REV)  # first 100,000 rows of `train`
```

**Trap**: `1K`, `10K`, and `100K` are not extra data - each is a literal prefix of `train`, confirmed by reading row 0 and row 5 of `train` and of `1K` and finding them identical [9][10]. Loading `train` alongside any of the smaller configs in the same training run duplicates rows.

## Neighbors

- **agentlans/think-more** - the direct, pre-Alpaca upstream: the same 969,020 rows and the same CC BY 4.0 licence declaration, but columns `question`/`answer`/`thought`/`source` instead of `instruction`/`input`/`output`/`thinking`, and no think-on/off variant. Prefer train-of-thought for an Alpaca-style SFT triple with a think switch; prefer think-more for the raw question/answer/thought triple [2].
- No other reformatting of think-more, and no other train-of-thought-style release, appears among agentlans's 276 other Hub dataset repositories: of the 11 whose name matches "think", "thought", "reason", or "cot", none of their READMEs mentions "think-more" or "train-of-thought" [8].

## A row

All four configs share one schema, so one row covers every served shape. From `config="train"`, `split="train"`, `row_idx=1` (datasets-server first-rows) [9], with the long `output` field truncated:

```json
{
  "instruction": "Use a chain-of-thought approach within <think>...</think> tags and summarize the result in <response>...</response>.",
  "input": "What happens to the photon after it transfers energy to an electron in the photoelectric effect?A: gets reflected\nB: changes frequency\nC: transforms into an electron\nD: disappears",
  "output": "<think>Okay, let's try to figure out this physics problem. Hmm, the question is about what happens to a photon after it transfers energy to an electron in the photoelectric effect. [...] </think><response>[...] - **D (Disappears)**: Correct. The photon is entirely absorbed, transferring all its energy to the electron, and thus no longer exists.\n\n\\boxed{D}</response>",
  "thinking": true,
  "source": "open-r1/Mixture-of-Thoughts"
}
```

## Where it came from

agentlans first built think-more by aggregating and cleaning nine reasoning-trace datasets: FreedomIntelligence/medical-o1-reasoning-SFT (12,971 rows), GeneralReasoning/GeneralThought-430K (336,262), O1-OPEN/OpenO1-SFT (61,709), nvidia/OpenCodeReasoning (11,732), nvidia/OpenMathReasoning (161,759), open-r1/OpenR1-Math-220k (64,293), open-thoughts/OpenThoughts-114k (113,943), zwhe99/DeepMath-103K (99,197), and open-r1/Mixture-of-Thoughts (107,154), for a combined 969,020 rows [2]. nvidia/OpenMathReasoning's problems are sourced from AoPS competition-math forums, and its card reports a results table naming AIME24 and AIME25 as evaluation benchmarks for models trained on it [7]. open-r1/OpenR1-Math-220k's problems come from NuminaMath 1.5, with reasoning traces generated by DeepSeek R1 [11]. agentlans then readapted all 969,020 think-more rows into this release's Alpaca-style triples, randomly assigning each to a "thinking on" or "thinking off" variant [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was fetched on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] agentlans/train-of-thought dataset card (README). https://huggingface.co/datasets/agentlans/train-of-thought/raw/main/README.md - structure, thinking-flag description, licence pointer, example row, load instructions. Fetched 2026-08-12.

[2] agentlans/think-more dataset card (README). https://huggingface.co/datasets/agentlans/think-more/raw/main/README.md - source table, processing steps, limitations, licence line, decompressed file size. Fetched 2026-08-12.

[3] The corpus screening row for `agentlans/train-of-thought`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-12.

[4] Hugging Face Hub API record for agentlans/train-of-thought. https://huggingface.co/api/datasets/agentlans/train-of-thought?full=true - `sha`, licence tag, gate/private status, `downloads`, `likes`, `lastModified`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=agentlans%2Ftrain-of-thought Fetched 2026-08-12.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=agentlans%2Ftrain-of-thought Fetched 2026-08-12.

[7] nvidia/OpenMathReasoning dataset card (README). https://huggingface.co/datasets/nvidia/OpenMathReasoning/raw/main/README.md - AoPS-forum provenance, QwQ-32B as a generation model, AIME24/AIME25 evaluation table. Fetched 2026-08-12.

[8] Hugging Face Hub API listing of agentlans's dataset repositories. https://huggingface.co/api/datasets?author=agentlans&limit=1000 - returns all 276 of agentlans's datasets; used to keyword-filter for "think"/"thought"/"reason"/"cot" in the name (11 matches besides think-more and train-of-thought) and, after fetching each match's README, to check for other reformattings of think-more or other releases that trained on this data. Fetched 2026-08-12.

[9] datasets-server first-rows endpoint, `config=train`. https://datasets-server.huggingface.co/first-rows?dataset=agentlans%2Ftrain-of-thought&config=train&split=train Fetched 2026-08-12.

[10] datasets-server first-rows endpoint, `config=1K`. https://datasets-server.huggingface.co/first-rows?dataset=agentlans%2Ftrain-of-thought&config=1K&split=train Fetched 2026-08-12.

[11] open-r1/OpenR1-Math-220k dataset card (README). https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/raw/main/README.md - NuminaMath 1.5 provenance, DeepSeek R1 generation. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data, with a stated decontamination step: because 226,052 of the 969,020 default-config rows trace to open-r1/OpenR1-Math-220k and nvidia/OpenMathReasoning, and nvidia/OpenMathReasoning's own card ties itself to AIME24/AIME25 evaluation, rows sourced from those two sets should be checked against AIME 2025 (and other math-competition eval sets) before a scored run, as established above [2][3][7].

### The screening row

The row's own note [3]: "Alpaca-format CoT with think-on/think-off variants, readapted from agentlans/think-more, which aggregates DeepSeek-R1 and o1-style traces (OpenO1-SFT, OpenMathReasoning, OpenR1-Math)." Its flag [3]: "rule-risk: aime2025 - Readapted from agentlans/think-more, whose source table lists open-r1/OpenR1-Math-220k (64,293) and nvidia/OpenMathReasoning (161,759 AoPS)."
