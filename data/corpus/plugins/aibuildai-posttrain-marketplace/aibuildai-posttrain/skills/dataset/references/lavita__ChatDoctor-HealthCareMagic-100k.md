# lavita/ChatDoctor-HealthCareMagic-100k

112,165 single-turn patient-question / doctor-answer pairs in Alpaca-style instruction/input/output form, a Parquet mirror of the HealthCareMagic-100k set behind the ChatDoctor project.

**lavita/ChatDoctor-HealthCareMagic-100k** packages the HealthCareMagic-100k corpus behind "ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge" [1]. The project's GitHub repository describes how that corpus was built: about 100,000 real doctor-patient conversations were scraped from the online consultation site HealthCareMagic.com, filtered both manually and automatically, stripped of patient and doctor identity information, and passed through language tools to correct grammar, before the result was named HealthCareMagic-100k [2]. Every row carries the same fixed instruction string ("If you are a doctor, please answer the medical questions based on the patient's description.") with the patient's message in `input` and the doctor's reply in `output` [3], so the dataset serves single-turn instruction-tuning (SFT), not multi-turn dialogue or preference training. **No usage-shape restriction is stated for this dataset repository itself** - see Licence below for the separate, model-level restriction the source project states. It lives at https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k .

**Use it for**: single-turn instruction SFT - fine-tuning a base model to answer a patient's free-text question in one turn. Maps to the SFT method card by treating `instruction` + `input` as the prompt and `output` as the target completion (a fixed Alpaca-style triple, not a chat-template dialogue) [3]. No preference pairs, no reasoning traces, no reward-model fields.

**Licence**: not stated for this Hub repository - the card's YAML front matter carries no `license` key and the repo carries no `license:` tag [4]. The one catch: the source GitHub project's README says the ChatDoctor model itself is restricted to academic research only, non-commercial, and not licensed for healthcare use, inheriting LLaMA's non-commercial terms [5] - that statement is about the fine-tuned model, and no source states it also binds this raw HealthCareMagic-100k text.

**Shape**: one config (`default`), one split, `train` = 112,165 rows, three string columns (`instruction`, `input`, `output`) [4][6].

**Hold out**: nothing found in this repository or its sources - it ships one split with no eval rows. The source project's own held-out evaluation set is a separate 10k-conversation collection from a different site, iCliniq, published as the sibling repo `lavita/ChatDoctor-iCliniq` (7,321 rows served) [2][7]; it is not part of this release and carries a different schema (see Neighbors).

**Origin**: mirrored to the Hub by user/org `lavita`; the underlying conversations are real human patient and doctor messages from HealthCareMagic.com, not model-generated [2]. Hub API at the check date: `downloads` 8,848, `downloadsAllTime` 60,795, `likes` 114 [4][8].

**Trained-on-by**: the source project's own ChatDoctor model (LLaMA fine-tuned on this data) [1]. Beyond that, the Hub's models-by-dataset filter for this exact repo returns 50+ community fine-tunes as of the check date (e.g. `kingabzpro/Gemma-2-9b-it-chat-doctor`, `vikash06/doctorLLM`, several `mradermacher` GGUF re-quantizations) [9]; none of these is a widely cited foundation-model release.

**Introduced by**: [1] (Li et al., "ChatDoctor").

## Shape

Rows and splits, live at the check date (datasets-server `/size`) [6]:

| split | rows |
| --- | --- |
| `train` | 112,165 |

One config, `default`, three columns, all string (datasets-server `/info`) [10]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |

Byte sizes: the repo's pinned `cardData.dataset_info` states `download_size` 70,518,148 and decoded `dataset_size` 126,454,896 for 112,165 examples [4]; the live datasets-server `/size` endpoint agrees on the 70,518,148-byte Parquet download but reports a different in-memory decoded size, 123,706,442 bytes [6] - the two sources compute "decoded size" differently and do not match to the byte, though both agree on row count and download size. No source states token or sequence-length statistics for this dataset.

## Quality

- The project's GitHub README states the conversations were filtered both manually and automatically, had doctor and patient identity information removed, and were passed through language tools to correct grammatical errors before release as HealthCareMagic-100k [2]. No source gives a measured filtering or error rate.
- Of the 100 rows read at offset 0 and a further 100 rows read at offset 112,065 (the end of `train`), `instruction` held the single fixed string quoted above in all 200 rows, and no row in either sample had an empty `input` or `output` [3][11]. This does not establish the whole 112,165-row split is free of empty fields; it only covers the 200 rows actually read.
- No source states an annotator-agreement figure (there is no separate annotation step beyond the automatic/manual filtering above) or a duplicate-row rate.
- The Hub card body itself is a stub ("More Information needed") and states no quality information [4].

## Load it

```python
import datasets

REV = "505443eac4e99ccedeffbb6f640061223d1d4bb3"  # main at the check date
train = datasets.load_dataset(
    "lavita/ChatDoctor-HealthCareMagic-100k", revision=REV, split="train"
)  # 112,165 rows
```

**Trap**: there is only one split (`train`) and one config (`default`); passing a `data_dir` or expecting a held-out split will fail, since none is declared on the card or served by the API [4][6]. If an eval split is needed, it must come from elsewhere (e.g. the sibling iCliniq repo, a different schema - see Neighbors), not from this repository.

## Neighbors

Row counts below were read live at the check date, unpinned [7][12].

- `lavita/ChatDoctor-iCliniq` - the same builder's mirror of the source project's separate iCliniq evaluation collection: 7,321 rows, four columns (`input`, `answer_icliniq`, `answer_chatgpt`, `answer_chatdoctor`), used by the project to score model answers against a real doctor's answer and against ChatGPT's, not for training [2][7]. Different schema; do not merge into `train` above.
- `wangrongsheng/HealthCareMagic-100k-en` - same row count (112,165) and the same three columns (`instruction`, `input`, `output`) as this repository, but carries no dataset card describing what, if anything, changed [12][13]; treat as an unverified re-upload rather than a confirmed duplicate.
- `RafaelMPereira/HealthCareMagic-100k-Chat-Format-en` - same row count (112,165) reshaped into a single `text` column under an Apache-2.0 license tag the source repository itself does not carry [12][14]; useful only if a pre-templated chat string is wanted instead of the three-field form.
- Prefer the present repository (`lavita/ChatDoctor-HealthCareMagic-100k`) for the three-column instruction/input/output shape: its `cardData` figures are pinned and directly reproducible via the revision recorded above, unlike the uncarded `wangrongsheng` mirror.

## A row

One served shape (`config="default"`, `split="train"`), row 0 (datasets-server `/first-rows`) [3]:

```json
{
  "instruction": "If you are a doctor, please answer the medical questions based on the patient's description.",
  "input": "I woke up this morning feeling the whole room is spinning when i was sitting down. I went to the bathroom walking unsteadily, as i tried to focus i feel nauseous. I try to vomit but it wont come out. [...]",
  "output": "Hi, Thank you for posting your query. The most likely cause for your symptoms is benign paroxysmal positional vertigo (BPPV), a type of peripheral vertigo. In this condition, the most common symptom is dizziness or giddiness, which is made worse with movements. Accompanying nausea and vomiting are [...]"
}
```

## Where it came from

The ChatDoctor project's authors scraped roughly 100,000 real patient-doctor conversations from the online consultation platform HealthCareMagic.com, filtered them both manually and automatically, removed patient and doctor identity information, and corrected grammar with language tools, naming the cleaned result HealthCareMagic-100k [2]. They used this set (together with Stanford Alpaca data for general conversational ability) to fine-tune LLaMA into the ChatDoctor model, and reserved a separate, smaller collection scraped from iCliniq.com as held-out evaluation data [2]. This Hub repository, maintained by `lavita`, is a Parquet re-packaging of that HealthCareMagic-100k set into the `instruction`/`input`/`output` schema; its own card is a content-free stub ("More Information needed") that states nothing about what, if anything, was changed in the re-packaging [4].

## Sources

Checked 2026-08-11; Hub repositories are mutable, which is why Load it pins the revision. The live datasets-server endpoints (`/size`, `/info`, `/first-rows`, `/rows`) and the Hub models-by-dataset filter take no revision parameter and are reported as of the check date only.

[1] Li et al., "ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge". https://arxiv.org/abs/2303.14070 - origin paper; current title and abstract (100,000 patient-doctor dialogues, cleaned and anonymized) read from the live abs page. Only the abstract was fetched, not the full text. Fetched 2026-08-11.

[2] Kent0n-Li/ChatDoctor GitHub repository README, "Patient-physician Conversation Dataset" section. https://raw.githubusercontent.com/Kent0n-Li/ChatDoctor/main/README.md - describes the HealthCareMagic scrape, manual/automatic filtering, de-identification, grammar correction, and the separate iCliniq-10k evaluation collection. Fetched 2026-08-11.

[3] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=lavita%2FChatDoctor-HealthCareMagic-100k&config=default&split=train - row 0 and the fixed `instruction` string across the first 100 rows. Fetched 2026-08-11.

[4] Hugging Face Hub API record for lavita/ChatDoctor-HealthCareMagic-100k. https://huggingface.co/api/datasets/lavita/ChatDoctor-HealthCareMagic-100k?full=true - `cardData.dataset_info`, tags, `sha`, `downloads`, `likes`, `createdAt`/`lastModified`, siblings; also the raw README (https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k/raw/main/README.md), a stub with no license field and no body beyond "More Information needed". Fetched 2026-08-11.

[5] Kent0n-Li/ChatDoctor GitHub repository README, "Limitations" section. https://raw.githubusercontent.com/Kent0n-Li/ChatDoctor/main/README.md - states the ChatDoctor model is for academic research only, non-commercial, inheriting LLaMA's license. Fetched 2026-08-11.

[6] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=lavita%2FChatDoctor-HealthCareMagic-100k Fetched 2026-08-11.

[7] datasets-server size endpoint for the sibling repo. https://datasets-server.huggingface.co/size?dataset=lavita%2FChatDoctor-iCliniq - and its raw README, https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq/raw/main/README.md, for its four-column schema. Fetched 2026-08-11.

[8] Hugging Face Hub API record, `downloadsAllTime` expansion. https://huggingface.co/api/datasets/lavita/ChatDoctor-HealthCareMagic-100k?expand[]=downloadsAllTime Fetched 2026-08-11.

[9] Hugging Face Hub API, models filtered by this dataset. https://huggingface.co/api/models?filter=dataset:lavita/ChatDoctor-HealthCareMagic-100k&limit=50 - returned 50 models at the check date (limit reached); none independently confirmed as a widely cited foundation-model release beyond this dataset's own listing. Fetched 2026-08-11.

[10] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=lavita%2FChatDoctor-HealthCareMagic-100k Fetched 2026-08-11.

[11] datasets-server rows endpoint, offset near the end of `train`. https://datasets-server.huggingface.co/rows?dataset=lavita%2FChatDoctor-HealthCareMagic-100k&config=default&split=train&offset=112065&length=100 Fetched 2026-08-11.

[12] datasets-server size endpoint, one call per neighbor: `wangrongsheng/HealthCareMagic-100k-en`, `RafaelMPereira/HealthCareMagic-100k-Chat-Format-en`. https://datasets-server.huggingface.co/size?dataset=<id> - live, unpinned. Fetched 2026-08-11.

[13] datasets-server info endpoint for `wangrongsheng/HealthCareMagic-100k-en`. https://datasets-server.huggingface.co/info?dataset=wangrongsheng%2FHealthCareMagic-100k-en - confirms the same three-column schema; its own README returns "Entry not found". Fetched 2026-08-11.

[14] `RafaelMPereira/HealthCareMagic-100k-Chat-Format-en` raw README and datasets-server info. https://huggingface.co/datasets/RafaelMPereira/HealthCareMagic-100k-Chat-Format-en/raw/main/README.md (license: apache-2.0) and https://datasets-server.huggingface.co/info?dataset=RafaelMPereira%2FHealthCareMagic-100k-Chat-Format-en (single `text` column). Fetched 2026-08-11.

[15] The corpus screening row for `lavita/ChatDoctor-HealthCareMagic-100k`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as single-turn instruction-SFT data: one `train` split, no held-out rows to withhold from within this repository, no restriction stated on the Hub repository itself. The screening row's own note describes the data this way, and the dataset's fixed-instruction, question/answer schema confirmed above matches it [15].

### The screening row

The row's own note [15]: "112k real patient-doctor consultations from HealthCareMagic in instruction form." The row carries no flag.
