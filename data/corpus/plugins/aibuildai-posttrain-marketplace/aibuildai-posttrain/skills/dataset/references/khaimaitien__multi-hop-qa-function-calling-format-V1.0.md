# khaimaitien/multi-hop-qa-function-calling-format-V1.0

28,733 multi-hop and single-hop question-answering dialogues, rewritten as OpenAI-style function-calling message lists with `user`, `assistant` and `function` roles, split into train (25,547) and validation (3,186).

**khaimaitien/multi-hop-qa-function-calling-format-V1.0** is a reformatting, by the same author (Khai Mai), of the paragraph/answer dataset `khaimaitien/qa-expert-multi-hop-qa-V1.0` into a list of chat messages: each `user` turn holds the question, each `assistant` turn either answers in `content` or issues a `function_call` (`name: "retrieve"`, `arguments: {"query": ...}`), and each `function` turn returns the retrieved paragraph [1]. The upstream dataset backs the `qa-expert-7B-V1.0` model, which decomposes a multi-hop question into single questions, calls a retrieval function for each, then summarizes the answers [2]. There is no separate origin paper; the upstream card cites only a GitHub-hosted write-up of the same project [3]. **A meaningful share of rows carry a `tag` containing `"musique"` or `"un_musique"`, and the upstream generation write-up states that `"musique"`-tagged rows are built from the public MuSiQue multi-hop QA benchmark [3]: decontaminate against any MuSiQue-based evaluation before a scored run.** It lives at https://huggingface.co/datasets/khaimaitien/multi-hop-qa-function-calling-format-V1.0 .

**Use it for**: SFT on function-calling / tool-use dialogue - training a model to alternate between free-text `assistant` turns and `function_call` turns across a fixed `user` → `assistant` (call) → `function` (result) → `assistant` (answer) pattern, repeated once per sub-question [1]. The `messages` column is already the OpenAI function-calling message-list shape, so it maps directly to the SFT chat method card without an `extract_prompt`-style conversion step. Restriction: hold out `validation`, and decontaminate the MuSiQue-derived rows before scoring on MuSiQue-based benchmarks (see Quality for the measured share).

**Licence**: not stated - neither this repository's card nor the upstream `qa-expert-multi-hop-qa-V1.0` card carries a `license` field or tag [1][4][5]; the repo is ungated (`"gated": false`) [6].

**Shape**: 28,733 rows, one config (`default`), splits `train` 25,547 / `validation` 3,186, four columns (`tag`, `meta_info`, `multihop`, `messages`) [7][8].

**Hold out**: `validation` (3,186 rows). Beyond the split, rows whose `tag` contains `"musique"` or `"un_musique"` are built from the MuSiQue benchmark per the upstream generation write-up [3]; Quality below gives the measured share in the rows actually read.

**Origin**: converted by Khai Mai (`khaimaitien`) from their own `qa-expert-multi-hop-qa-V1.0` [1][4]; the upstream card states generation was mostly done with `gpt-3.5-turbo-instruct` [4], though the served rows' own `meta_info.llm` field disagrees in the sample read - see Quality. Hub API at the check date: `downloads` 141, `downloadsAllTime` 1,763, `likes` 9 [6].

**Trained-on-by**: `khaimaitien/qa-expert-7B-V1.0`, a fine-tune of `mistralai/Mistral-7B-v0.1`, states it was trained on `khaimaitien/qa-expert-multi-hop-qa-V1.0` [2] - the upstream, pre-conversion dataset this repository reformats, not this function-calling repository itself. No source confirms a model trained on this reformatted repository specifically.

**Introduced by**: no paper - the dataset card [1], reformatting the upstream dataset card [4], whose own citation block points only to a GitHub project write-up, not a paper [3].

## Shape

Rows served and splits (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 25,547 |
| `validation` | 3,186 |
| total | 28,733 |

One config, `default`, four columns (datasets-server `/info`) [8]:

| column | dtype |
| --- | --- |
| `tag` | string |
| `meta_info` | struct (generation metadata - category, entities, attributes, `llm`, `negatives`, `selected`, `slot_dic`, `src`; fields are populated or null depending on which generation pathway produced the row) |
| `multihop` | bool |
| `messages` | list of struct (`content`: string or null, `function_call`: struct(`arguments`, `name`) or null, `role`: string) |

Sizes (datasets-server `/size`) [7]: 118,272,384 bytes of original JSON download, 34,311,596 bytes as Parquet (30,863,262 for `train` and 3,448,334 for `validation`), 85,259,992 bytes decoded in memory. No source states sequence-length or token statistics for this release; none is invented here.

## Quality

- The upstream generation write-up says the dataset was created to fill a gap in public multi-hop QA training data, and describes two generation paths: synthetic entity/attribute-comparison questions generated with `gpt-3.5-turbo-instruct`, and questions adapted from the MuSiQue dataset, identified by a `tag` field containing `"musique"` [3]. Of the first 55 `train` rows (offset 0) returned by the datasets-server `first-rows` endpoint, 21 carry a `tag` containing `"musique"`; of the first 59 `validation` rows (offset 0, the full sample the endpoint returned for that split), 27 do [9]. This card reports only those two samples and does not extrapolate to the full 28,733 rows.
- In the same 114 sampled rows, the `meta_info.llm` field reads `"wizard_lm"` for 84 rows and `"gpt-3.5-turbo-instruct"` for 30 [9] - the opposite of what the upstream card says, that generation was done mostly with `gpt-3.5-turbo-instruct` [4], and `"wizard_lm"` is not named anywhere in the upstream generation write-up's description of the pipeline [3]. This card states the discrepancy as read and does not resolve it.
- The upstream write-up describes a deliberate "negative paragraph" generation step, applied to a subset of the entity- and attribute-comparison rows, that replaces a retrieved paragraph with one that does not answer the sub-question, to make the model learn to say a question is not answerable from the given context [3]. In the sample above, `meta_info.selected.answerable` (populated only on rows from this pathway) reads `false` for 14 rows and `true` for 21, out of the 35 sampled rows where `meta_info.selected` is non-null; the field is null on the other 79 sampled rows [9].
- No source states a measured contamination rate against downstream evaluation benchmarks, a duplicate-row rate, or an annotator-agreement figure for this dataset.

## Load it

Train on `train`, hold out `validation`, and pin the revision this card's numbers were read at:

```python
import datasets

REV = "0ac8a58c37f06c68e7ff4cebd2404f67e5ef5d91"
train = datasets.load_dataset("khaimaitien/multi-hop-qa-function-calling-format-V1.0", revision=REV, split="train")            # 25,547 rows
validation = datasets.load_dataset("khaimaitien/multi-hop-qa-function-calling-format-V1.0", revision=REV, split="validation")  # 3,186 rows - hold out
```

**Trap**: within each `messages` entry, `content` and `function_call` are mutually exclusive - an `assistant` turn that issues a retrieval call has `content: null` and a populated `function_call`, while an `assistant` turn that answers has `content` set and `function_call: null` [1][9]. A collator that assumes every message has a non-null string `content` will drop or crash on the function-call turns, which are exactly the turns this dataset exists to teach.

## Neighbors

- `khaimaitien/qa-expert-multi-hop-qa-V1.0` - the upstream, pre-conversion release this repository reformats: the same 25,547/3,186 split [10], but each row is a flat JSON of `question`, `sub_questions` (each with `question`, `paragraph`, `long_answer`), `final_answer`, `answer` and `meta_info`, not a chat message list [4]. This corpus's function-calling reformatting is the one to use for chat/tool-use SFT; use the upstream directly only if a non-chat, paragraph-QA shape is wanted instead.
- A Hub search for other repositories built from this project (`qa-expert`, `multi-hop-qa-function-calling`) by any author returned no further re-releases, forks, or successors as of the check date [11].

## A row

One config and one message schema serve both splits; message-list length varies with `multihop` and the number of sub-questions rather than with a different shape. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `first-rows`) [9], with the two retrieved-paragraph fields truncated:

```json
{
  "tag": "long_entities-train.json",
  "meta_info": {
    "category": "Drug in Over-the-counter medications",
    "comparison_attribute": "dosage",
    "entity_1": "Tylenol",
    "entity_2": "Advil",
    "list_of_attributes": "active ingredient, purpose, dosage, side effects",
    "llm": "gpt-3.5-turbo-instruct",
    "negatives": null,
    "selected": null,
    "slot_dic": {
      "category": "Drug in Over-the-counter medications",
      "popularity_1": "1",
      "popularity_2": "1",
      "question_type": "yes/no question"
    },
    "src": "gen_qa"
  },
  "multihop": true,
  "messages": [
    {"role": "user", "content": "Is the dosage for Tylenol higher than the dosage for Advil?", "function_call": null},
    {"role": "assistant", "content": null, "function_call": {"name": "retrieve", "arguments": "{\"query\": \"What is the dosage for Tylenol?\"}"}},
    {"role": "function", "content": "Tylenol, also known by its generic name acetaminophen... [...] The recommended dosage for adults is 325-650 mg every 4-6 hours, with a maximum of 4,000 mg in a 24-hour period. [...]", "function_call": null},
    {"role": "assistant", "content": "The recommended dosage for adults is 325650 mg every 46 hours... [...]", "function_call": {"name": "retrieve", "arguments": "{\"query\": \"What is the dosage for Advil?\"}"}},
    {"role": "function", "content": "Advil, also known as ibuprofen... [...] The recommended dosage for adults is 200-400 milligrams every four to six hours, as needed. [...]", "function_call": null},
    {"role": "assistant", "content": "The recommended dosage for adults is 200400 milligrams every four to six hours... [...] Summary: [...] Answer: Yes, the dosage for Tylenol is higher than the dosage for Advil. [...]", "function_call": null}
  ]
}
```

## Where it came from

Built by Khai Mai (`khaimaitien`) by converting `khaimaitien/qa-expert-multi-hop-qa-V1.0` into function-calling message lists [1]. That upstream dataset's rows come from two pathways, per its GitHub-hosted generation write-up [3]: (1) synthetic multi-hop questions comparing two entities or two attributes, generated with `gpt-3.5-turbo-instruct` through a multi-step prompt chain (random category, entity generation, attribute selection, question generation, paragraph generation, answer generation), with a subset further augmented with "negative paragraphs" that do not contain the answer; and (2) questions adapted from the MuSiQue multi-hop QA dataset, filtering out malformed questions and generating complete natural-language answers for MuSiQue's span answers. Single-question (non-multi-hop) rows reuse the sub-questions from the multi-hop generation pathway [3].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] khaimaitien/multi-hop-qa-function-calling-format-V1.0 dataset card (README). https://huggingface.co/datasets/khaimaitien/multi-hop-qa-function-calling-format-V1.0/raw/main/README.md - message-role/format description, conversion source. Fetched 2026-08-12.

[2] khaimaitien/qa-expert-7B-V1.0 model card (README). https://huggingface.co/khaimaitien/qa-expert-7B-V1.0/raw/main/README.md - states the model is a Mistral-7B-v0.1 fine-tune trained on khaimaitien/qa-expert-multi-hop-qa-V1.0. Fetched 2026-08-12.

[3] Generation write-up for the upstream dataset, on the linked project GitHub repository. https://raw.githubusercontent.com/khaimt/qa_expert/main/gen_data/README.md - generation pipeline, negative-paragraph step, MuSiQue adaptation and the `"musique"` tag convention; linked from the upstream dataset card as "how we created this dataset". Fetched 2026-08-12.

[4] khaimaitien/qa-expert-multi-hop-qa-V1.0 dataset card (README). https://huggingface.co/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0/raw/main/README.md - row counts, "mostly generated using... gpt-3.5-turbo-instruct" claim, citation block. Fetched 2026-08-12.

[5] Hugging Face Hub API record for khaimaitien/qa-expert-multi-hop-qa-V1.0. https://huggingface.co/api/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0?full=true - `cardData` has no `license` key. Fetched 2026-08-12.

[6] Hugging Face Hub API record for khaimaitien/multi-hop-qa-function-calling-format-V1.0. https://huggingface.co/api/datasets/khaimaitien/multi-hop-qa-function-calling-format-V1.0?full=true - `sha`, `gated`, `downloads`, `likes`, `lastModified`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-12.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=khaimaitien%2Fmulti-hop-qa-function-calling-format-V1.0 Fetched 2026-08-12.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=khaimaitien%2Fmulti-hop-qa-function-calling-format-V1.0 Fetched 2026-08-12.

[9] datasets-server first-rows endpoint, `config=default`, read for both splits. https://datasets-server.huggingface.co/first-rows?dataset=khaimaitien%2Fmulti-hop-qa-function-calling-format-V1.0&config=default&split=train (55 rows returned) and the same URL with `split=validation` (59 rows returned). Fetched 2026-08-12.

[10] datasets-server size endpoint for the upstream dataset. https://datasets-server.huggingface.co/size?dataset=khaimaitien%2Fqa-expert-multi-hop-qa-V1.0 - confirms the exact 25,547/3,186 split match with this repository. Fetched 2026-08-12.

[11] Hugging Face Hub dataset search API. https://huggingface.co/api/datasets?search=qa-expert and https://huggingface.co/api/datasets?search=multi-hop-qa-function-calling - live search, unpinned. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as SFT chat/function-calling training data, with a decontamination step before any MuSiQue-based scored run. Two facts decide it: the `messages` column already carries the OpenAI function-calling shape the SFT method card expects [1], and a substantial share of rows (21/55 and 27/59 in the samples read) are built from the MuSiQue benchmark per the upstream generation write-up [3], which is the risk the screening row's note also names.

### The screening row

The row's own note [screening record for `khaimaitien/multi-hop-qa-function-calling-format-V1.0`, checked 2026-08-12]: "multi-hop QA rewritten as OpenAI function-call messages; the upstream card says the data was mostly generated with gpt-3.5-turbo-instruct." The row carries no flag.
