# TheTokenFactory/sec-contracts-financial-extraction-instructions

7,683 instruction examples that extract exact terms - effective dates, parties, dollar amounts, debt covenants, executive pay - from real SEC contract exhibits and proxy statements into JSON, served three times over in three chat formats. Silver labels written by a 2B model. The closest public match to Harvey LAB's "extract key terms" task families.

**TheTokenFactory/sec-contracts-financial-extraction-instructions** holds "7,683 instruction-tuning examples for training LLMs to extract structured financial data from SEC filings" across 368 S&P 500 companies: 1,028 Exhibit 10 material contracts and 150 DEF 14A proxy statements [1]. Each example gives a system prompt with a JSON schema, a chunk of filing text, and the extracted JSON [1][2]. It lives at https://huggingface.co/datasets/TheTokenFactory/sec-contracts-financial-extraction-instructions .

**The three configs (`sharegpt`, `alpaca`, `openai`) are the same examples in three formats: load one. The labels were produced by Gemma 4 2B, not by people; the card calls them "silver-standard labels" [1].**

**Use it for**: SFT on exact-term extraction from real agreements - the skill behind LAB criteria such as "states the annual contract value as approximately $6.2 million". Use the `train` split for positive examples; the `corrective` split adds negatives that teach an empty answer when the text holds no value.

**Licence**: CC BY 4.0 in the card metadata and body [1][3]. The source filings are public SEC records.

**Shape**: 23,049 rows served = 7,683 examples x 3 formats [4]. Per format: `train` 3,430, `corrective` 4,253 [1][4].

**Hold out**: nothing of its own; no evaluation split. Last changed 2026-04-10, before LAB went public [3]. Measured against LAB anyway: one LAB document reaches exactly 50% coverage, a Sarbanes-Oxley certification whose boilerplate also appears in real filings; no LAB rubric or instruction is contained [5].

**Origin**: filing text is human-drafted; the JSON targets are model-written: "Extraction model: Gemma 4 2B (Q4_K_M quantized) at temperature 0.1", filtered through "14+ quality gates" [1]. Hub API at the check date: `downloads` 3,183, `downloadsAllTime` 4,432, `likes` 1 [3].

**Trained-on-by**: three quantised Gemma 4 E2B fine-tunes from the same author, `TheTokenFactory/gemma-4-E2B-sec-extraction-GGUF` (107 downloads) and its `-v2` (92) and `-v3` (87) [6].

**Introduced by**: [1] (dataset card; no paper).

## Shape

| config | split | rows [4] |
| --- | --- | --- |
| `sharegpt` (default) | `train` / `corrective` | 3,430 / 4,253 |
| `alpaca` | `train` / `corrective` | 3,430 / 4,253 |
| `openai` | `train` / `corrective` | 3,430 / 4,253 |

Columns of `sharegpt`: `conversations` (list of `from`/`value`), `metadata` (source file, task type, company, ticker, confidence and correction flags) [2].

Task types, measured on the `sharegpt` files [7]:

| task type | `train` | `corrective` |
| --- | ---: | ---: |
| `financial_extraction` | 1,434 | 1,600 |
| `metadata_extraction` | 1,028 | 1,027 |
| `compensation_extraction` | 293 | - |
| `covenant_extraction` | 264 | 433 |
| `governance_extraction` | 261 | 160 |
| `exec_metadata_extraction` | 150 | - |
| table and preamble extractions | - | 1,033 |

## Quality

- The card's corrective breakdown does not match the file. The card lists 1,968 positive-corrected, 95 corrective and 2,190 negative examples [1]; counted in `sharegpt_corrective.jsonl`: 3,347 `positive_corrected`, 240 `corrective`, 666 `negative` [7].
- Duplication, measured [7]: 13 of 3,430 `train` conversations (0.38%) and 15 of 4,253 `corrective` (0.35%) are exact repeats. 30.67% of `train` rows share their input text with another row: the same contract chunk is posed as several tasks, so split by `source_file`, not by row.
- MinHash (5-gram shingles, Jaccard at least 0.8) flags 16.47% of `train` conversations as near-duplicates [7]; each task type repeats one long system prompt, which inflates this figure for short inputs.
- Temporal scope is a six-month filing window and the universe is S&P 500 only [1].

## Load it

```python
import datasets

REV = "c90ac749f9056901c8c3ed22ae1ff7c894c9b696"  # main at the check date
train = datasets.load_dataset("TheTokenFactory/sec-contracts-financial-extraction-instructions", "sharegpt", revision=REV, split="train")        # 3,430
corrective = datasets.load_dataset("TheTokenFactory/sec-contracts-financial-extraction-instructions", "sharegpt", revision=REV, split="corrective")  # 4,253
```

**Trap**: loading all three configs, or the dataset without a config name in a tool that concatenates configs, triples every example. Pick one format.

## Neighbors

- `chenghao/sec-material-contracts` - 1.1 million Exhibit 10 contracts; the unlabelled pool these chunks come from.
- `theatticusproject/cuad-qa` - expert-labelled clause extraction from EDGAR contracts; gold labels where this set has silver.
- `TheTokenFactory/sec-contracts-corrective-extraction` - a companion set named in the author's model tags [6]; not screened here.

## A row

From `config="sharegpt"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`), contract text shortened [2]:

```json
{
  "conversations": [
    {"from": "system", "value": "You are a strict legal extraction AI. Read this contract preamble. Identify the primary effective date and the exact legal names of the two primary contracting parties. Output strictly as JSON. ..."},
    {"from": "human", "value": "EX-10.1 3 tm2529220d2_ex10-1.htm EXHIBIT 10.1 ... Project Comet US$3,050,000,000 Bridge Facility Commitment Letter ..."},
    {"from": "gpt", "value": "{\"effective_date\": \"2025-10-27\", \"primary_party_1\": \"Skyworks Solutions, Inc.\", \"primary_party_2\": \"Goldman Sachs Bank USA\"}"}
  ],
  "metadata": {"source_file": "0001_tm2529220d2_ex10-1.htm", "task_type": "metadata_extraction", "company": "SKYWORKS SOLUTIONS, INC.", "ticker": "SWKS", "pipeline": "exhibit10", "model_version": "gemma-4-2b-base"}
}
```

## Where it came from

Built by TheTokenFactory with a six-stage pipeline: harvest from EDGAR, chop into targeted sections, extract with Gemma 4 2B, validate through quality gates, normalise company names by CIK, and join inputs to validated outputs [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-24; that date covers every number, quote, and row above unless a line says it was measured. Hub repositories are mutable, which is why Load it pins the revision.

[1] Dataset card (README). https://huggingface.co/datasets/TheTokenFactory/sec-contracts-financial-extraction-instructions/raw/c90ac749f9056901c8c3ed22ae1ff7c894c9b696/README.md. Fetched 2026-09-24.

[2] datasets-server info and first-rows endpoints. https://datasets-server.huggingface.co/first-rows?dataset=TheTokenFactory%2Fsec-contracts-financial-extraction-instructions&config=sharegpt&split=train. Fetched 2026-09-24.

[3] Hugging Face Hub API record. https://huggingface.co/api/datasets/TheTokenFactory/sec-contracts-financial-extraction-instructions?full=true - licence field, `sha`, `lastModified` 2026-04-10; `downloads`, `downloadsAllTime`, `likes` through the `expand[]` variant. Fetched 2026-09-24.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=TheTokenFactory%2Fsec-contracts-financial-extraction-instructions - 23,049 rows, `partial` false. Fetched 2026-09-24.

[5] This skill's own measurement, `references/contamination.md`, section "Harvey LAB". Run 2026-09-24.

[6] Hub model search by dataset tag. https://huggingface.co/api/models?filter=dataset:TheTokenFactory/sec-contracts-financial-extraction-instructions&sort=downloads. Fetched 2026-09-24.

[7] This skill's own count of the `sharegpt` files at the pinned revision: task and example types, exact, cleaned and MinHash duplication. Run 2026-09-24.

## Appendix: screening record

### Screening verdict

Passed as SFT data for LAB's extraction tasks, with silver labels. Stage 0: readable, not gated, `partial` false. Stage 1: chat SFT with a JSON target. Stage 2: licence field and body agree. Stage 3: 0.38% exact duplicates; inputs repeat across task types. Stage 4: no LAB rubric or instruction contained. Stage 5 (overlap with carded datasets) was not run. The corrective-split counts on the card do not match the file.

### The screening row

The row's own note: "SEC contract and proxy extraction SFT, 7,683 examples x 3 formats; silver labels from Gemma 4 2B; card's corrective counts wrong; added for the LAB target on 2026-09-24." The row carries the flag `model-written-labels`.
