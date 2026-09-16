# axiong/pmc_llama_instructions

513,999 single-turn medical instruction/input/output rows, assembled from six benchmark and knowledge-graph sources, released as part of the instruction-tuning data behind PMC-LLaMA-13B.

**axiong/pmc_llama_instructions** was released by the PMC-LLaMA authors alongside "PMC-LLaMA: Towards Building Open-source Language Models for Medicine" [1], the paper that fine-tunes a LLaMA-13B medical model on a larger instruction corpus the paper calls MedC-I [1]. The dataset card states this repository holds six of that corpus's seven source components - MedQA, MedMCQA, PubMedQA, LiveQA, MedicationQA and UMLS - and explicitly excludes the seventh, ChatDoctor, telling users to merge it in separately for the complete mix [2]. Each row is a single instruction/input/output triple, shaped for supervised instruction fine-tuning rather than multi-turn dialogue. It lives at https://huggingface.co/datasets/axiong/pmc_llama_instructions . **The MedQA, MedMCQA and PubMedQA components are drawn from those benchmarks' official training partitions, not their test partitions: the row-level `source` field literally names one of them `medqa_train`, and the origin paper states instruction tuning "start[ed] with the training sets" of USMLE (MedQA), PubMedQA and MedMCQA, holding their official test splits out for evaluation instead [1]. A scored run against any of those three benchmarks' own test sets should still explicitly hold those test sets out, since this dataset only documents having excluded them at the paper's build time, not at a downstream user's evaluation time.**

**Use it for**: single-turn SFT on instruction/input/output triples covering medical multiple-choice reasoning, PubMed abstract QA, consumer health QA, and UMLS knowledge-graph QA/relation prompts - maps directly to the SFT method card by concatenating `instruction`+`input` as the prompt and `output` as the target. Not preference data (no chosen/rejected pair) and not multi-turn dialogue.

**Licence**: `license:openrail` tag (cardData `license: openrail`), ungated, public [4]. The catch: the repository ships no `LICENSE` file naming which specific OpenRAIL variant applies, and the `openrail` tag covers this repackaging only - the underlying MedQA/MedMCQA/PubMedQA/UMLS source datasets carry their own separate original licences that this tag does not state or supersede.

**Shape**: 513,999 rows, one config (`default`), one split (`train`), 5 columns [5][6].

**Hold out**: nothing within this repository itself - it is a single, undivided `train` split with no reserved test rows. Before any scored run against MedQA/USMLE, MedMCQA, or PubMedQA, hold out those benchmarks' own official test sets, because this dataset is built from their training partitions and a downstream evaluation harness could still reintroduce overlap if it does not do so independently [1][2].

**Origin**: released by axiong (the PMC-LLaMA team); content is human-authored - exam questions (MedQA/USMLE, MedMCQA), PubMed research-question abstracts (PubMedQA), consumer health questions (LiveQA, MedicationQA), and UMLS ontology entries - restructured into instruction/input/output triples with templated instruction wording [1][2]. Hub API at the check date: 157 downloads, 33 likes, last modified 2023-11-23 [4].

**Trained-on-by**: PMC-LLaMA-13B, per this repository's own card, which states the repository "provides part of the dataset used for PMC-LLaMA-13B's instruction tuning" [2] - PMC-LLaMA-13B's own model card does not itself list this dataset repository by name, so this line rests on the dataset card's statement, not on a Hub auto-link [8]. Separately, the Hub's own "Models trained or fine-tuned on" list for this dataset shows six linked models, none of them PMC-LLaMA-13B [10]: two independent base finetunes - Henrychur/MMed-Llama-3-8B-EnIns, whose model card lists `axiong/pmc_llama_instructions` in its training `datasets:` and says the model was trained only on it for English instruction fine-tuning [9], and Technoculture/MT7Bi-sft, whose model card lists it alongside three other training datasets [12] - plus four GGUF quantization repositories of those two base models.

**Introduced by**: [1] (Wu et al., "PMC-LLaMA: Towards Building Open-source Language Models for Medicine").

## Shape

Rows, config and columns from the datasets-server `/size` and `/info` endpoints [5][6]:

| split | rows |
| --- | --- |
| `train` | 513,999 |

| column | dtype |
| --- | --- |
| `output` | string |
| `source` | string |
| `instruction` | string |
| `input` | string |
| `sample_id` | int64 |

Sizes [5]: 699,885,733 bytes of original JSON download, 309,278,832 bytes as Parquet, 675,228,674 bytes decoded in memory. No source states sequence-length or token statistics for this specific 513,999-row release. The origin paper states its full instruction corpus, MedC-I - which additionally includes the ChatDoctor component this repository excludes - totals 202M tokens [1]; that figure is not this repository's own token count.

Reading the `source` column at sampled offsets across the full row range (0, 5,000, 9,000-31,000, 50,000-200,000, 250,000-410,000, 414,000-415,200, 450,000, 500,000, 513,900) turned up exactly seven distinct values and the approximate order in which they run through the file [7]:

| approximate row range (sampled) | `source` value | matches |
| --- | --- | --- |
| 0 - ~20,000-30,000 | `medqa_train` | MedQA/USMLE, explicitly the train partition |
| ~30,000 - ~200,000-250,000 | `medmcqa` | MedMCQA |
| ~250,000 - ~410,000-414,000 | `pubmedqa.ori_pqaa` | PubMedQA's PQA-A artificial subset |
| ~414,000 - ~414,600 | `liveqa` | LiveQA |
| ~414,600 - ~415,000 | `medicationqa` | MedicationQA |
| ~415,000 - ~450,000-500,000 | `umls` | UMLS entity descriptions |
| ~500,000 - 513,999 | `umls_relation` | UMLS entity relations |

These are the boundaries observed in the offsets actually sampled, not an exhaustive scan; exact transition rows were not located.

## Quality

- Every sampled row's `source` value names its origin component, and one of them, `medqa_train`, says outright that it is the MedQA/USMLE training partition, not a held-out split [7].
- The origin paper corroborates this for all three multiple-choice/abstract-QA components: instruction tuning "start[ed] with the training sets of the open-source medical multi-choice question-answering datasets, such as USMLE..., PubMedQA... and MedMCQA," and separately reports MedMCQA's official split as 182,822 train questions and 4,183 test questions, and PubMedQA's PQA-A (211.3k artificially generated pairs) as the paper's own train set, versus PQA-L (1k manually labeled pairs) as the paper's own test set [1]. This repository's `medmcqa` row count (observed in the range above) sits far closer to 182,822 than to 4,183, and every sampled PubMedQA row carries the `pubmedqa.ori_pqaa` tag, never a PQA-L tag, consistent with a train-only PQA-A source [1][7].
- The `sample_id` column is populated (0, 1, 2, ...) only in the `medqa_train` rows read; every `medmcqa`, `pubmedqa.ori_pqaa`, and `umls_relation` row read carried `sample_id: null` [7].
- MedQA questions repeat with multiple paraphrased instruction wordings: the first three served rows all carry `sample_id: 0` and the same question, each with a differently worded `instruction` field [7]. This is consistent with the origin paper's description of using GPT-4 to generate several "synonymous sentences" per instruction template to diversify phrasing, though the paper describes that step for its ChatDoctor/MedAlpaca conversational data specifically, not by name for the QA components [1].
- Sampled `medqa_train` and `medmcqa` rows carry two distinct output styles: a short plain answer (e.g. "###Answer: OPTION D IS CORRECT.") and, in other rows for the same or different questions, a longer "###Rationale: ..." explanation walking through the answer choices - both styles appear repeatedly within the offsets read, so this repository's own rows do carry the ChatGPT-generated causal-rationale text the origin paper describes as "Medical Rationale QA" [1][7]. Sampled `pubmedqa.ori_pqaa` rows instead carry the source abstract's own long-answer text as `output`, not a generated rationale [7].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this specific 513,999-row release.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-11-23) [4]:

```python
import datasets

REV = "5142689a3888786db7d376e60c16c5c1e7201191"  # main at the check date
train = datasets.load_dataset("axiong/pmc_llama_instructions", revision=REV, split="train")  # 513,999 rows
```

**Trap**: the repository serves a single file, `release.json`, as one undivided `train` split - there is no train/test split to select. The dataset card itself says this repository "provides *part of* the dataset used for PMC-LLaMA-13B's instruction tuning" and excludes ChatDoctor's roughly 100K rows, telling users to merge that component in separately for the full mix [2]; loading this repository alone does not reproduce the paper's full MedC-I corpus [1].

## Neighbors

- `HoangHa/pmc_llama_instructions` - a Parquet-native re-upload of the same content: identical row count (513,999), identical five columns, and row 0 fetched from its served endpoint matches this repository's row 0 field-for-field [11]. Use it only if a Parquet-native repo is preferred over this repository's single JSON file; it carries no additional documentation of its own.
- `axiong/pmc_oa` is a separate dataset from the same author (PubMed Central image-caption pairs for continued pretraining), not a re-release of this instruction data and does not share its columns; it is not a neighbor by shape.
- No other repackaging of this specific instruction mix, and no rationale-augmented or ChatDoctor-merged re-release, was found.

## A row

The repository serves one config and one schema, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7]:

```json
{
  "output": "###Answer: OPTION D IS CORRECT.",
  "source": "medqa_train",
  "instruction": "You're a doctor, kindly address the medical queries according to the patient's account.\nAnswer with the best option directly.",
  "input": "###Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?\n###Options:\nA. Ampicillin\nB. Ceftriaxone\nC. Doxycycline\nD. Nitrofurantoin\n",
  "sample_id": 0
}
```

## Where it came from

Built and released by the PMC-LLaMA authors (axiong on the Hub) as part of the data behind "PMC-LLaMA: Towards Building Open-source Language Models for Medicine" [1]. The paper describes each component's collection: the multiple-choice QA components (USMLE/MedQA, MedMCQA) and PubMedQA come from those benchmarks' own published training partitions, restructured into instruction/input/output triples with GPT-4-diversified instruction phrasing for the paper's broader conversational data [1]; the UMLS component is built by querying the UMLS medical knowledge graph to construct entity-description and entity-relationship QA pairs, aligning the model with clinicians' terminology [1]. The dataset card names LiveQA and MedicationQA only by link, to `truehealth/liveqa` and `truehealth/medicationqa` on the Hub, without further collection detail in this card [2]. The card also names a seventh component, ChatDoctor (roughly 100K rows), that the paper's broader corpus uses but this specific repository does not include [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Wu et al., "PMC-LLaMA: Towards Building Open-source Language Models for Medicine", 2023. https://arxiv.org/abs/2304.14454 - the origin paper; current title read from the live abs page, full text read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2304.14454). Fetched 2026-08-11.

[2] axiong/pmc_llama_instructions dataset card (README). https://huggingface.co/datasets/axiong/pmc_llama_instructions/raw/main/README.md - component list, ChatDoctor exclusion, licence and task tags. Fetched 2026-08-11.

[3] The corpus screening row for `axiong/pmc_llama_instructions`, supplied with this card's request - its `note` and `flag`, read back in the row's own words in the appendix. Checked 2026-08-11.

[4] Hugging Face Hub API record for axiong/pmc_llama_instructions. https://huggingface.co/api/datasets/axiong/pmc_llama_instructions?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date. Fetched 2026-08-11.

[5] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=axiong%2Fpmc_llama_instructions Fetched 2026-08-11.

[6] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=axiong%2Fpmc_llama_instructions Fetched 2026-08-11.

[7] datasets-server rows and first-rows endpoints, sampled at offsets 0, 5,000, 9,000-31,000, 50,000-200,000, 250,000-410,000, 414,000-415,200, 450,000, 500,000 and 513,900 (100 or fewer rows per call). https://datasets-server.huggingface.co/rows?dataset=axiong%2Fpmc_llama_instructions&config=default&split=train&offset=<n>&length=<n> and https://datasets-server.huggingface.co/first-rows?dataset=axiong%2Fpmc_llama_instructions&config=default&split=train Fetched 2026-08-11.

[8] axiong/PMC_LLaMA_13B model card (README) and Hub API record. https://huggingface.co/axiong/PMC_LLaMA_13B/raw/main/README.md and https://huggingface.co/api/models/axiong/PMC_LLaMA_13B - the model card does not list this dataset repository by name in its `datasets:` metadata. Fetched 2026-08-11.

[9] Henrychur/MMed-Llama-3-8B-EnIns model card (README). https://huggingface.co/Henrychur/MMed-Llama-3-8B-EnIns/raw/main/README.md - lists `axiong/pmc_llama_instructions` under `datasets:` and states the model was trained only on it for English instruction tuning. Fetched 2026-08-11.

[10] axiong/pmc_llama_instructions dataset page, "Models trained or fine-tuned on" section. https://huggingface.co/datasets/axiong/pmc_llama_instructions - live HTML rendering of the Hub-linked models: Henrychur/MMed-Llama-3-8B-EnIns, mradermacher/MT7Bi-sft-i1-GGUF, Technoculture/MT7Bi-sft, mradermacher/MT7Bi-sft-GGUF, tensorblock/MMed-Llama-3-8B-EnIns-GGUF, itlwas/MMed-Llama-3-8B-EnIns-Q4_K_M-GGUF. Fetched 2026-08-11.

[11] HoangHa/pmc_llama_instructions dataset: Hub API, datasets-server size endpoint, and a sampled row. https://huggingface.co/api/datasets/HoangHa/pmc_llama_instructions?full=true , https://datasets-server.huggingface.co/size?dataset=HoangHa%2Fpmc_llama_instructions , https://datasets-server.huggingface.co/rows?dataset=HoangHa%2Fpmc_llama_instructions&config=default&split=train&offset=0&length=1 - row-for-row match against this repository's row 0. These endpoints take no revision parameter, so this comparison is live, not pinned. Fetched 2026-08-11.

[12] Technoculture/MT7Bi-sft Hub API record. https://huggingface.co/api/models/Technoculture/MT7Bi-sft - `cardData.datasets` lists `axiong/pmc_llama_instructions` alongside three other training datasets. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as single-turn SFT instruction data, with the contamination question the screening flagged now resolved by source evidence: the `source` field and the origin paper both show the MedQA, MedMCQA and PubMedQA components are drawn from those benchmarks' training partitions, not their test partitions [1][7]. A downstream scored run against those three benchmarks should still explicitly hold their own official test sets out, since this dataset documents only what the paper's build excluded, not what any given evaluation harness will exclude on its own.

### The screening row

The row's own note [3]: "Medical instruction tuning assembled from MedQA, MedMCQA, PubMedQA, LiveQA, MedicationQA, UMLS (and ChatDoctor, not included); exam questions and abstracts written by people." Its flag [3]: "contamination risk: built on the MedQA / MedMCQA / PubMedQA benchmark families; row counts match their train splits but no card states the split."
