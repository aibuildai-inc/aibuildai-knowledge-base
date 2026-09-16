# argilla/distilabel-intel-orca-dpo-pairs

12,859 DPO preference pairs, a re-labelled and re-scored version of Intel/orca_dpo_pairs, built with Argilla's distilabel pipeline and a GPT-4-based judge.

**argilla/distilabel-intel-orca-dpo-pairs** re-labels Intel/orca_dpo_pairs [2], which itself samples about 12k question/response pairs from Open-Orca/OpenOrca [3] - a FLAN-derived instruction corpus of roughly 1M GPT-4 and 3.2M GPT-3.5 completions built to align with the distribution described in "Orca: Progressive Learning from Complex Explanation Traces of GPT-4" [4]. Rather than assume the original dataset's GPT-4 response always beats the GPT-3.5 response, Argilla re-scored both candidates with a JudgeLM-style prompt run on `gpt-4-1106-preview`, keeping the higher-scored response as `chosen`; this flips the label on roughly 2,000 of the 12,859 pairs relative to the original, and about 4,000 pairs are marked as a rated tie [1]. The dataset serves DPO/preference-pair training and adds a `chosen_score` and an `in_gsm8k_train` decontamination flag for filtering [1]. It lives at https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs .

**Use it for**: preference-pair training - DPO or another method that consumes a chosen/rejected pair - the same shape as the SFT/DPO method card's preference-pair format, already in `chosen`/`rejected` columns [1]. The card's own recommended filter (`status != "tie"`, `chosen_score >= 8`, `not in_gsm8k_train`) drops the set to 5,922 rows [1]; on the linked `distilabeled-OpenHermes-2.5-Mistral-7B` fine-tune, that filtered subset scored 54.04 average on the AGIEval/GPT4All/TruthfulQA/Bigbench suite versus 53.51 for `mlabonne/NeuralHermes-2.5-Mistral-7B`, trained with the same DPO recipe on the unfiltered original Intel/orca_dpo_pairs data [8].

**Licence**: Apache-2.0 (`license: apache-2.0` in the card's front matter and in the Hub API's `cardData`), ungated [1][5]. The one catch: the upstream Intel/orca_dpo_pairs card is also Apache-2.0 [2], but Open-Orca/OpenOrca two levels upstream is licensed MIT [3] - this card states no additional licence term beyond its own Apache-2.0 grant.

**Shape**: 12,859 rows, one config (`default`), one split (`train`) [6][7].

**Hold out**: the card reports checking the 12,859 rows for overlap with GSM8K's train split (its own TF-IDF decontamination pass), flagging 79 rows `in_gsm8k_train=True`, and states it found no matches against GSM8K's test split [1]; no other holdout is stated. Filter or hold out the 79 `in_gsm8k_train=True` rows before any GSM8K-adjacent evaluation.

**Origin**: built by Argilla from Intel/orca_dpo_pairs, using `gpt-4-1106-preview` as the re-scoring judge [1]; the underlying chosen/rejected responses are themselves GPT-4 and GPT-3.5 completions from OpenOrca [3]. Hub API as of the check date: 10,907 downloads, 139,419 all-time downloads, 182 likes [5].

**Trained-on-by**: Argilla's own `distilabeled-OpenHermes-2.5-Mistral-7B`, a DPO fine-tune of OpenHermes-2.5-Mistral-7B released alongside this dataset and tagged `dataset:argilla/distilabel-intel-orca-dpo-pairs` [8]. A live Hub API query for models tagged with this dataset returns at least 100 matches, including `argilla/distilabeled-Marcoro14-7B-slerp`, `decruz07/kellemar-DPO-7B-v1.01`, and several community GGUF/GPTQ/MLX re-quantizations of the Hermes DPO model [9].

**Introduced by**: no paper - the dataset card [1].

## Shape

Rows and split, from the datasets-server `/size` endpoint [6]:

| split | rows |
| --- | --- |
| `train` | 12,859 |

One config, `default`, with 16 columns, from the datasets-server `/info` endpoint [7]:

| column | dtype |
| --- | --- |
| `system` | string |
| `input` | string |
| `chosen` | string |
| `rejected` | string |
| `generations` | list\<string\> |
| `order` | list\<string\> |
| `labelling_model` | string |
| `labelling_prompt` | list\<struct\<content: string, role: string\>\> |
| `raw_labelling_response` | string |
| `rating` | list\<float64\> |
| `rationale` | string |
| `status` | string |
| `original_chosen` | string |
| `original_rejected` | string |
| `chosen_score` | float64 |
| `in_gsm8k_train` | bool |

No source states sequence-length or token statistics for this dataset. Sizes from `/size` [6]: 79,210,071 bytes of Parquet download, 160,754,769 bytes decoded in memory - about 4.15x Intel/orca_dpo_pairs's own 19,093,564-byte Parquet download and about 4.63x its 34,684,414-byte in-memory size for the same 12,859 rows [10], consistent with this release adding the judge's rating, rationale, and raw response columns on top of the original four columns.

## Quality

- The re-scoring judge is a single GPT-4 checkpoint (`gpt-4-1106-preview`) run once per pair with a JudgeLM-style prompt that rates both shuffled candidates 1-10 and gives a natural-language rationale; the card reports the resulting label distribution as roughly 7,000 `unchanged`, about 4,000 `tie`, and about 2,000 `swapped` (rejected preferred over the original chosen) [1].
- The card states its GSM8K decontamination used TF-IDF cosine similarity between each row's `input` and GSM8K's train (and train-socratic) questions, with a similarity threshold of 0.8 chosen because it reports that lower thresholds introduced false positives on manual inspection [1]. It reports finding no matches against GSM8K's test split, though the reproduction code shown only checks against the train and train-socratic splits [1].
- The `in_gsm8k_train` column reproduces the card's stated result exactly: of the 12,859 rows, 79 are flagged `True` and 12,780 `False` [1], matching the `bool` column served in the dataset [7].
- No source states an annotator-agreement figure, a duplicate-row rate, or a measured error rate for the judge's ratings; none is invented here.

## Load it

Single split, no filtering needed to load; pin the revision this card's numbers were read at [5]:

```python
from datasets import load_dataset

REV = "0b10ec0df32c919f95126b203c8f5962b6875896"  # main at the check date
dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", revision=REV, split="train")  # 12,859 rows
```

**Trap**: the card's own recommended filter - `status != "tie"`, `chosen_score >= 8`, `not in_gsm8k_train` - is not applied by default; loading without it keeps all 12,859 rows, including the ~4,000 rated ties and the 79 rows flagged `in_gsm8k_train=True` [1]:

```python
dataset = dataset.filter(
    lambda r:
        r["status"] != "tie" and
        r["chosen_score"] >= 8 and
        not r["in_gsm8k_train"]
)  # card reports this yields 5,922 rows
```

## Neighbors

- `Intel/orca_dpo_pairs` - the unmodified upstream this dataset re-labels: same 12,859 rows, same `system`/`input`/`chosen`/`rejected` columns, but every `chosen` is assumed to be the GPT-4 response with no judge score, rating, or GSM8K flag [1][2][10]. This corpus prefers the distilabel release for its added `chosen_score`, `status`, and `in_gsm8k_train` filtering columns.
- `Open-Orca/OpenOrca` - the much larger upstream pool (roughly 1M GPT-4 and 3.2M GPT-3.5 completions) that Intel/orca_dpo_pairs was sampled from; not a preference-pair dataset itself [3].
- No sibling "binarized" or "cleaned" re-release of this specific distilabel dataset was found on the Hub at the check date.

## A row

One config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [11], with the long `labelling_prompt` and `raw_labelling_response`/`rationale` fields truncated:

```json
{
  "system": "",
  "input": "You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplet [...]",
  "chosen": "[\n  [\"AFC Ajax (amateurs)\", \"has ground\", \"Sportpark De Toekomst\"],\n  [\"Ajax Youth Academy\", \"plays at\", \"Sportpark De Toekomst\"]\n]",
  "rejected": " Sure, I'd be happy to help! Here are the RDF triplets for the input sentence:\n\n[AFC Ajax (amateurs), hasGround, Sportpark De Toekomst]\n[Ajax Youth Academy, playsAt, Sportpark De Toekomst] [...]",
  "generations": ["[\n  [\"AFC Ajax (amateurs)\", \"has ground\", \"Sportpark De Toekomst\"], [...]", " Sure, I'd be happy to help! [...]"],
  "order": ["chosen", "rejected"],
  "labelling_model": "gpt-4-1106-preview",
  "labelling_prompt": [
    {"content": "You are a helpful and precise assistant for checking the quality of the answer.", "role": "system"},
    {"content": "[Question]\nYou will be given a definition of a task first [...]", "role": "user"}
  ],
  "raw_labelling_response": "9 9\n\nBoth Assistant 1 and Assistant 2 provided correct RDF triplets for the given sentence. [...]",
  "rating": [9.0, 9.0],
  "rationale": "\nBoth Assistant 1 and Assistant 2 provided correct RDF triplets for the given sentence. [...]",
  "status": "tie",
  "original_chosen": "[\n  [\"AFC Ajax (amateurs)\", \"has ground\", \"Sportpark De Toekomst\"],\n  [\"Ajax Youth Academy\", \"plays at\", \"Sportpark De Toekomst\"]\n]",
  "original_rejected": " Sure, I'd be happy to help! Here are the RDF triplets for the input sentence: [...]",
  "chosen_score": 9.0,
  "in_gsm8k_train": false
}
```

This first row has a `rating` of `[9.0, 9.0]`, so `status` is `tie` and `chosen`/`original_chosen` match - the judge scored both candidates equally rather than swapping the label.

## Where it came from

Built by Argilla using the distilabel pipeline library [1]. The pipeline loads Intel/orca_dpo_pairs, shuffles each pair's `chosen`/`rejected` response into a `generations` list and records the original `order` to avoid positional bias, then labels both candidates with a JudgeLM-style prompt run on `gpt-4-1106-preview`, storing the raw response, per-candidate `rating`, and a natural-language `rationale` [1]. A post-processing step derives `status` (`unchanged`, `tie`, or `swapped`) from whether the judge's higher-rated candidate matches the original `chosen`, sets `chosen_score` to the higher rating, and - when the judge disagreed with the original label - swaps `chosen`/`rejected` while preserving the pre-swap values in `original_chosen`/`original_rejected` [1]. A separate decontamination pass computes TF-IDF cosine similarity between each row's `input` question and GSM8K's train and train-socratic questions, flagging matches at or above a 0.8 threshold in the `in_gsm8k_train` column [1]. Intel/orca_dpo_pairs, the dataset this release re-labels, states it contains about 12k examples sampled from Open-Orca/OpenOrca [2], an instruction dataset of GPT-4 and GPT-3.5 completions built to align with the distribution in the Orca paper [3][4].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] argilla/distilabel-intel-orca-dpo-pairs dataset card (README). https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs/raw/main/README.md - description, licence, recommended filter, GSM8K decontamination method and result, reproduction code, label-distribution chart. Fetched 2026-08-11.

[2] Intel/orca_dpo_pairs dataset card (README). https://huggingface.co/datasets/Intel/orca_dpo_pairs/raw/main/README.md - upstream dataset description, licence, source pool. Fetched 2026-08-11.

[3] Open-Orca/OpenOrca dataset card (README). https://huggingface.co/datasets/Open-Orca/OpenOrca/raw/main/README.md - GPT-4/GPT-3.5 completion counts, licence, alignment with the Orca paper. Fetched 2026-08-11.

[4] Mukherjee et al., "Orca: Progressive Learning from Complex Explanation Traces of GPT-4", 2023. https://arxiv.org/abs/2306.02707 - the methodology OpenOrca targets; current title read from the live abs page. Fetched 2026-08-11.

[5] Hugging Face Hub API record for argilla/distilabel-intel-orca-dpo-pairs. https://huggingface.co/api/datasets/argilla/distilabel-intel-orca-dpo-pairs?full=true and the `expand[]=downloadsAllTime` variant - licence, gate, `sha`, `downloads`, `downloadsAllTime`, `likes`. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=argilla%2Fdistilabel-intel-orca-dpo-pairs Fetched 2026-08-11.

[7] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=argilla%2Fdistilabel-intel-orca-dpo-pairs Fetched 2026-08-11.

[8] argilla/distilabeled-OpenHermes-2.5-Mistral-7B model card (README) and Hub API record. https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B/raw/main/README.md and https://huggingface.co/api/models/argilla/distilabeled-OpenHermes-2.5-Mistral-7B - confirms the DPO fine-tune trained on this dataset, and its README's benchmark table compares it (trained on the filtered 5,922-row subset) against `mlabonne/NeuralHermes-2.5-Mistral-7B` (unfiltered original data, same recipe). Fetched 2026-08-11.

[9] Hugging Face Hub API, models filtered by this dataset. https://huggingface.co/api/models?filter=dataset:argilla/distilabel-intel-orca-dpo-pairs&limit=100 - this endpoint takes no revision parameter, so the count is live, not pinned. Fetched 2026-08-11.

[10] datasets-server size endpoint for Intel/orca_dpo_pairs. https://datasets-server.huggingface.co/size?dataset=Intel%2Forca_dpo_pairs - byte-size comparison. Fetched 2026-08-11.

[11] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=argilla%2Fdistilabel-intel-orca-dpo-pairs&config=default&split=train Fetched 2026-08-11.

[12] The corpus screening row for `argilla/distilabel-intel-orca-dpo-pairs`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as DPO preference-pair data, filtered by the card's own recommended thresholds before training. This rests on facts already established above: the dataset's own card frames it as a preference-tuning replacement for Intel/orca_dpo_pairs with a stated filter that improves downstream performance [1], and the screening row's note describes the same re-labelling and GSM8K-flagging story [12].

### The screening row

The row's own note [12]: "12,859 DPO pairs re-labelled from Intel/orca_dpo_pairs with distilabel: the GPT-4 and GPT-3.5 answers were re-scored by an AI judge instead of assuming GPT-4 always wins, and a column marks rows whose question is in the GSM8K train split, which the card says found no test examples." The row carries no flag.
