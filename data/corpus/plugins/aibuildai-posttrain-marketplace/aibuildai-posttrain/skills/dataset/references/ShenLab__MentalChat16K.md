# ShenLab/MentalChat16K

16,084 English instruction/input/output rows of mental-health counseling dialogue, mixing GPT-3.5-generated synthetic sessions with paraphrased real clinical-interview transcripts, served as a single unsplit `train` split.

**ShenLab/MentalChat16K** was built by a University of Pennsylvania team and introduced as "MentalChat16K: A Benchmark Dataset for Conversational Mental Health Assistance" [1]. It combines two sources into one `instruction`/`input`/`output` schema: a synthetic counselor-client dataset generated with OpenAI GPT-3.5 Turbo via a customized Airoboros self-generation framework, and a dataset of anonymized, human-transcribed interview transcripts between behavioral-health coaches and caregivers of hospice/palliative-care patients, paraphrased into single-turn QA pairs by a local Mistral-7B-Instruct-v0.2 model [2][1]. **The paper's own limitations section says combining the synthetic and interview halves during fine-tuning "did not consistently improve model performance" and in some cases hurt it, and it recommends handling the two halves separately [1] - a distinction this Hub repository does not preserve, since it serves both sources pre-merged into one `train` split with no source column (see Shape).** The paper uses this data to fine-tune LLMs and evaluates the fine-tuned models against a separate, hand-curated 200-question benchmark drawn from Reddit and Mental Health Forum posts, which is not part of this repository [1]. It lives at https://huggingface.co/datasets/ShenLab/MentalChat16K .

**Use it for**: SFT-style instruction fine-tuning (`instruction` + `input` -> `output`), the shape the paper itself used with QLoRA on 7B models [1]. Because the two source halves have different quality/behavior profiles and the paper says mixing them can degrade performance, split by source before training rather than treating all 16,084 rows as one homogeneous set (see Shape for how to recover the split). Maps to the SFT method card.

**Licence**: MIT (`cardData.license` is `"mit"`, repo tag `license:mit`), ungated (`"gated": false`, `"private": false`) [3]. No further usage restriction is stated in the licence itself; the mixing caution above comes from the paper, not the licence.

**Shape**: 16,084 rows, one config (`default`), one split (`train`); three string columns (`instruction`, `input`, `output`) [4][5].

**Hold out**: nothing declared by the card or paper as an evaluation split inside this repository - the paper's own 200-question evaluation set is a separately curated, external resource, not rows drawn from this dataset [1]. No source states or implies that the training rows here overlap that external benchmark.

**Origin**: built by ShenLab (University of Pennsylvania); outputs are OpenAI GPT-3.5 Turbo generations for the synthetic half and Mistral-7B-Instruct-v0.2 paraphrases of human-transcribed interviews for the interview half [1][2]. Hub API at the check date: `downloads` 1,726, `downloadsAllTime` 31,642, `likes` 62 [3].

**Trained-on-by**: the origin paper's own experiments, which used QLoRA to fine-tune seven base models on this data (individually on each half and on the combination): LLaMA-2-7B, Mistral-7B-v0.1, Mixtral-8x7B-v0.1, Mistral-7B-Instruct-v0.2, Mixtral-8x7B-Instruct-v0.1, Vicuna-7B-v1.5, and Zephyr-7B-Alpha [1]. No other adoption by named external models or recipes was found.

**Introduced by**: [1] (Xu et al.).

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 16,084 |

One config, `default`, three columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `instruction` | string |
| `input` | string |
| `output` | string |

The repository ships two source CSV files that this single config concatenates: `Interview_Data_6K.csv` and `Synthetic_Data_10K.csv` [6]. Neither the served rows nor `/info`'s `download_checksums` order name which rows came from which file, but the underlying `csv` loader concatenates files in alphabetical filename order, and the served row content confirms it: row 0's `input` field is verbatim the interview file's first data row, and row 6,310's `input` field is verbatim the synthetic file's first data row [7]. Downloading and parsing both CSVs directly (accounting for embedded newlines in fields) gives 6,310 data rows in `Interview_Data_6K.csv` and 9,774 in `Synthetic_Data_10K.csv`, summing to exactly the 16,084 rows served [7]. So rows 0-6,309 are the interview half and rows 6,310-16,083 are the synthetic half; the join key is row-index range, since no explicit source column exists.

These counts differ from the paper's own prose, which states 9,775 synthetic QA pairs and 6,338 interview QA pairs (also the numbers in this repository's README and in the shortlist row's note) [1][2]. The paper's own Table 1 additionally prints those same two counts (9,775 and 6,338) attached to the opposite dataset labels from its prose - the "Interview Data" table row shows 9,775 rows and 378 sessions, while its own prose says the interview data has 378 transcripts yielding 6,338 pairs (378 times 16.8 average QA-per-session, the table's own figure, is approximately 6,350, not 9,775); the "Synthetic Data" row shows 6,338 rows and 33 topics, while the prose and the README both attribute the 33-topic synthetic conversations to the roughly 9,775-pair count [1][2]. Table 1 also gives average word counts per row, correctly paired with the dataset identity by the sessions/topics cross-check above: interview rows average 69.94 input words and 235.85 output words; synthetic rows average 111.24 input words and 363.94 output words [1]. No source states token counts.

## Quality

- The interview half's outputs are Mistral-7B-Instruct-v0.2 paraphrases/summaries of one page of transcript into a single QA turn, not the coach's or caregiver's verbatim words; the paper filtered out any resulting QA pair with fewer than 40 words in the question or answer [1].
- The synthetic half's outputs are direct GPT-3.5 Turbo generations, not paraphrases of any recorded conversation [2].
- The paper's own ablation (its Table 3, per the limitations section) found that fine-tuning on the combined synthetic+interview data "did not consistently improve model performance" over fine-tuning on either half alone, and "in some cases, it led to performance degradation" [1]. It attributes this to the interview data's narrower demographic (specific caregivers/patients) versus the synthetic data's broader topic spread, and recommends handling the two datasets separately [1].
- Under its GPT-4 Turbo judge, the paper reports that for the "Active Listening" metric the synthetic-only fine-tune beat the base, interview-fine-tuned, and combination-fine-tuned variants for all seven base models (7 of 7), and that across its other six metrics the synthetic-only fine-tune's win count was 6 of 7 for one metric and 7 of 7 for the remaining five, i.e. the synthetic-only fine-tune dominated under this judge [1]. Under its separate Gemini Pro judge, the paper reports the opposite tilt: Gemini "seem[s] to place more value on the depth and realism provided by interview data", especially on the Safety & Trustworthiness and Boundaries & Ethical metrics [1]. Its separate human-evaluator panel gives no matching per-metric win counts, only a qualitative statement that evaluators "often preferred" the synthetic-fine-tuned model but that interview- and combination-fine-tuned models "also performed well" in several cases [1].
- The paper states that the anonymization process "may inadvertently strip conversations of contextual nuances essential for effective mental health support", and separately that the paraphrasing process "may introduce minor deviations or potential hallucinations" [1].
- No source states a measured duplicate-row rate or annotator-agreement figure for either half.

## Load it

Pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2025-07-14):

```python
import datasets

REV = "5f60cd380cfc58f0f12d44892bed41ee3670a70a"  # main at the check date
ds = datasets.load_dataset("ShenLab/MentalChat16K", revision=REV, split="train")  # 16,084 rows, both halves merged

interview_half = ds.select(range(0, 6310))     # Interview_Data_6K.csv, 6,310 rows
synthetic_half = ds.select(range(6310, 16084))  # Synthetic_Data_10K.csv, 9,774 rows
```

**Trap**: `load_dataset` with no arguments beyond the split merges both source CSVs into one `train` split with no column marking which file a row came from; the paper's own limitations section says mixing them can hurt fine-tuning performance, so a reader who wants to reproduce the paper's per-source or combined runs must recover the split by row-index range as above, not by any served field [1][6][7].

## Neighbors

Several forks of this repository exist on the Hub, all uploaded by third parties, not ShenLab; row counts read live at the check date [8]:

- `TVRRaviteja/MentalChat16K-split` - the same schema (`instruction`/`input`/`output`) pre-divided into `train` (12,867 rows) and `test` (3,217 rows) splits, still totaling 16,084 rows; a reader who needs a ready-made train/test split of this exact corpus without deriving one from the source-file boundary can use this instead [8][9].
- `Devique/mentalchat16k_sharegpt` - the same content reformatted into a single ShareGPT-style `conversation` column, 24,104 rows in one `train` split - the row count does not match this release's 16,084, so it is a reformatted and apparently re-chunked derivative, not a 1:1 remap [8][10].
- A long tail of other unofficial re-uploads under names like `Cikycn/MentalChat16K`, `Jpnm89/MentalChat16K`, `ProfPsyko/MentalChat16K`, etc. were found in a Hub search for "mentalchat" but were not individually verified; every one of them, including these three, shows 0 likes (versus this repository's 62) and at most 29 downloads (versus this repository's 1,726), and none is the builder's own release [8].
- No cleaned or binarized re-release from ShenLab itself was found; this repository is ShenLab's only dataset on the Hub [8].

## A row

The repository serves one config and one schema for both source halves, so one row from each half is shown. From `config="default"`, `split="train"`, `row_idx=0` (interview half) and `row_idx=6310` (synthetic half), via the datasets-server `/rows` endpoint [7], with long fields truncated:

```json
{
  "row_idx": 0,
  "instruction": "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. \nThe assistant gives helpful, comprehensive, and appropriate answers to the user's questions. ",
  "input": "I've been struggling with my mental health for a while now, and I can't seem to find a way to cope with it. I've tried visualization, positive thinking, and even medication, but nothing seems to work. [...]",
  "output": "I understand that you've been dealing with a sense of confusion and chaos in your thoughts and emotions for some time now. [...] This process is called Cognitive Behavioral Therapy (CBT) [...]"
}
```

```json
{
  "row_idx": 6310,
  "instruction": "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. \nThe assistant gives helpful, comprehensive, and appropriate answers to the user's questions. ",
  "input": "I think I might be developing a substance abuse problem. Lately, I've been relying on alcohol as a way to cope with my emotions and escape from reality. [...]",
  "output": "I'm really glad that you reached out and shared what you've been going through. [...] Firstly, it's great that you're recognizing the impact this habit is having on various aspects of your life. [...]"
}
```

Both rows share the identical fixed `instruction` string; only `input` and `output` vary. The two halves are not distinguishable by schema or by any field value pattern found in these two sampled rows - only by the row-index boundary established in Shape.

## Where it came from

Built by ShenLab at the University of Pennsylvania. The interview half comes from 378 transcripts of an ongoing clinical trial, audio-recorded behavioral-intervention sessions between behavioral-health coaches and caregivers of individuals in palliative or hospice care, transcribed by human experts; participants signed informed consent, sessions were recorded on team-provided devices and stored on secure institutional cloud servers, and identifying information was stripped and stored separately from the study data under unique identifiers [1]. Each transcript page was fed to a local Mistral-7B-Instruct-v0.2 model with a summarization prompt to produce one QA turn, and pairs with fewer than 40 words in the question or answer were filtered out [1]. The synthetic half was generated with OpenAI GPT-3.5 Turbo using a customized adaptation of the Airoboros self-generation framework, with topic proportions specified in the generation prompt across 33 mental health topics [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Xu, Wei, Hou, Orzechowski, Yang, Jin, Paulbeck, Wagenaar, Demiris, Shen, "MentalChat16K: A Benchmark Dataset for Conversational Mental Health Assistance", 2025. https://arxiv.org/abs/2503.13509 - the origin paper; current title read from the live abs page; body text and Table 1 read from the arXiv HTML rendering (v2). Fetched 2026-08-12.

[2] ShenLab/MentalChat16K dataset card (README). https://huggingface.co/datasets/ShenLab/MentalChat16K/raw/main/README.md - dataset description, generation method, topic count. Fetched 2026-08-12.

[3] Hugging Face Hub API record for ShenLab/MentalChat16K. https://huggingface.co/api/datasets/ShenLab/MentalChat16K?full=true and the `expand[]=downloadsAllTime` variant - licence, gate, `sha`, `downloads`, `downloadsAllTime`, `likes`, last-modified date. Fetched 2026-08-12.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=ShenLab%2FMentalChat16K Fetched 2026-08-12.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=ShenLab%2FMentalChat16K Fetched 2026-08-12.

[6] Hub repository file tree for ShenLab/MentalChat16K. https://huggingface.co/api/datasets/ShenLab/MentalChat16K/tree/main - lists `Interview_Data_6K.csv` and `Synthetic_Data_10K.csv` as the two data files. Fetched 2026-08-12.

[7] The dataset's own served rows and source CSV files, read directly: datasets-server `/rows` endpoint at offsets 0, 6,305-6,314, 9,770-9,784, and 16,070-16,083 (https://datasets-server.huggingface.co/rows?dataset=ShenLab%2FMentalChat16K&config=default&split=train), cross-checked against the two CSV files downloaded whole from https://huggingface.co/datasets/ShenLab/MentalChat16K/resolve/main/Interview_Data_6K.csv and .../Synthetic_Data_10K.csv and parsed with Python's `csv` module (6,310 and 9,774 data rows respectively). Fetched 2026-08-12.

[8] Hub dataset search and datasets-server size endpoint, for every neighbor named above: search at https://huggingface.co/api/datasets?search=mentalchat and https://huggingface.co/api/datasets?author=ShenLab, sizes at https://datasets-server.huggingface.co/size?dataset=<id>. These endpoints take no revision parameter, so the counts are live, not pinned. Fetched 2026-08-12.

[9] Hub API record for TVRRaviteja/MentalChat16K-split. https://huggingface.co/api/datasets/TVRRaviteja/MentalChat16K-split?full=true - `dataset_info` split sizes confirming the train/test division and matching schema. Fetched 2026-08-12.

[10] datasets-server info endpoint for Devique/mentalchat16k_sharegpt. https://datasets-server.huggingface.co/info?dataset=Devique%2Fmentalchat16k_sharegpt - confirms the single `conversation` column name and row count. Fetched 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as SFT instruction data, with the paper's own caution attached: fine-tune on the merged rows as a starting point is possible, but the paper's ablation found mixing the two source halves inconsistent (sometimes worse than fine-tuning on either half alone), so a reader aiming to reproduce or improve on the paper's results should split by the row-index boundary in Shape and treat the halves separately or compare against single-source fine-tunes. No held-out evaluation split exists inside this repository; the paper's own evaluation used an external 200-question set not distributed here.

### The screening row

The row's own note [as supplied with this card's request]: "9,775 counseling chats generated by OpenAI GPT-3.5 Turbo plus 6,338 QA pairs from human-transcribed clinical interviews." The row carries no flag.
