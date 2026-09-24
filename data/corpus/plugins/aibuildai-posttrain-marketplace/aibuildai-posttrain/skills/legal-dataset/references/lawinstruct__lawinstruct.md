# lawinstruct/lawinstruct

A multilingual legal instruction-tuning collection - 55 source datasets in 24 languages rendered as instruction, prompt and answer, 10.6 GB compressed - whose training files carry every LegalBench citizenship question and nearly every MAUD item.

**lawinstruct/lawinstruct** is "a diverse multilingual dataset for legal instruction tuning" [1], introduced by Niklaus et al. in the paper now titled "LawInstruct: A Resource for Studying Language Model Adaptation to the Legal Domain" [2]. The data sits in 142 `.jsonl.xz` files named `<dataset>-<subset>-train-<shard>.jsonl.xz` [3], covering 55 source datasets from Brazilian bar exams and CAIL to LexGLUE, MAUD, ContractNLI and Swiss judgment prediction [1]. Its card's citation block still uses the paper's earlier title, "FLawN-T5: An Empirical Examination of Effective Instruction-Tuning Data Mixtures for Legal Reasoning" [1]. It lives at https://huggingface.co/datasets/lawinstruct/lawinstruct .

**LawInstruct is built from the training splits of legal benchmarks, and several of those benchmarks feed LegalBench. Measured: its International Citizenship Law Questions files hold all 9,306 LegalBench `international_citizenship_questions` test items, and its MAUD files hold 4,530 of 4,598 LegalBench `maud_*` items.**

**Use it for**: multilingual legal instruction tuning, file by file. Choose sources by task and language, and drop every file whose source is a benchmark you will report - see `references/contamination.md` for which files overlap LegalBench.

**Licence**: MIT in the card metadata [4], but the card's Licensing Information reads "[More Information Needed]" [1], and the collection repackages sources under their own licences - ContractNLI, for instance, is CC BY-NC-SA 4.0 in its Hub release [5]. The one catch: the effective licence of each file is its source's.

**Shape**: 142 data files, 10.6 GB compressed, one `train` split [3][1]. The viewer cannot run the loading script: "To be able to use lawinstruct/lawinstruct, you need to install the following dependency: absl" [6].

**Hold out**: nothing is held out inside the repository; every file is a `train` file [1]. Measured against LegalBench test [7]: the two International Citizenship files hold 6,457 and 2,849 items (all 9,306); each MAUD file holds 4,530 of the 4,598 MAUD items (the `answer` file 4,085); the ContractNLI file 303 of 1,927 ContractNLI items; the LexGLUE `unfair_tos` file 190 of 3,614 `unfair_tos` items; the PrivacyQA file 403, mostly OPP-115 (272) and `privacy_policy_*` (124) items.

**Origin**: mixed: human-written legal NLP datasets, with instructions written by the authors and machine-translated into 24 languages - measured, the 14,010 ContractNLI rows carry instructions in 23 different languages, and all 185,200 PrivacyQA rows carry Bulgarian instructions over English passages [8]. Hub API at the check date: `downloads` 618, `downloadsAllTime` 13,147, `likes` 32 [4].

**Trained-on-by**: the paper's FLawN-T5 models are trained on it [2]; no model on the Hub declares it through its dataset tag [9].

**Introduced by**: [2] (Niklaus et al.).

## Shape

The viewer serves no rows [6]. Row counts of the files downloaded for this card [8]:

| file | rows |
| --- | --- |
| `PrivacyQA-privacy_qa-train-0.jsonl.xz` | 185,200 |
| `LexGLUE-case_hold-train-0.jsonl.xz` | 45,000 |
| `MAUD-category-train-0.jsonl.xz` (same for `question`, `text_type`) | 25,827 |
| `ContractNLI-contract_nli-train-0.jsonl.xz` | 14,010 |
| `MAUD-answer-train-0.jsonl.xz` | 10,751 |
| `InternationalCitizenshipLawQuestions-..._mode_acq-train-0.jsonl.xz` | 6,460 |
| `LexGLUE-unfair_tos-train-0.jsonl.xz` | 5,532 |
| `InternationalCitizenshipLawQuestions-..._mode_loss-train-0.jsonl.xz` | 2,850 |

The largest files are Brazilian case-law tasks (`BrCAD5-brcad5_topic-train-0.jsonl.xz` 18.6 GiB uncompressed) [1]. Each file's rows carry `dataset_name`, `subset_name`, `source`, `instruction_language`, `prompt_language`, `answer_language`, `jurisdiction`, `task_type`, `downloaded_timestamp`, `instruction`, `prompt` and `answer` [8].

## Quality

- The card's field list documents a single `text` field "consisting of the prompt and the answer" [1], but the files have separate `instruction`, `prompt` and `answer` fields and no `text` field [8]. Code written from the card reads empty strings.
- Instructions are machine-translated regardless of the passage language: the first LexGLUE `unfair_tos` row pairs a Latvian instruction with an English terms-of-service sentence and an English answer [8].
- The same MAUD rows appear four times, once per MAUD subset file (`answer`, `category`, `question`, `text_type`), each asking a different question about the same passage [8].

## Load it

Read the files you want directly - there is no parquet copy, and the script needs `absl`:

```python
from huggingface_hub import hf_hub_download
import json, lzma

REV = "578e100783654ba791e55e5ebaaac343c66f408d"  # main at the check date
f = hf_hub_download("lawinstruct/lawinstruct", "data/ContractNLI-contract_nli-train-0.jsonl.xz",
                    repo_type="dataset", revision=REV)
rows = [json.loads(l) for l in lzma.open(f, "rt", encoding="utf-8")]   # 14,010 rows
sft = [{"prompt": r["instruction"] + "\n\n" + r["prompt"], "response": r["answer"]} for r in rows]
```

**Trap**: the card documents a `text` column that does not exist [1][8]; the rows have `instruction`, `prompt` and `answer`. The first containment run for this card read `text` and measured zero overlap with every benchmark - a false clean bill that only the positive control caught (`references/contamination.md`).

## Neighbors

- `joelniklaus/Multi_Legal_Pile` - the same authors' pretraining corpus.
- `nguha/legalbench` - the benchmark several LawInstruct sources overlap.

## A row

The viewer serves no rows. Line 1 of `data/LexGLUE-unfair_tos-train-0.jsonl.xz`, downloaded [8], instruction truncated:

```json
{
  "dataset_name": "LexGLUE",
  "subset_name": "unfair_tos",
  "source": "https://huggingface.co/datasets/lex_glue",
  "instruction_language": "lv",
  "prompt_language": "en",
  "answer_language": "en",
  "jurisdiction": "UNKNOWN",
  "task_type": "TEXT_CLASSIFICATION",
  "downloaded_timestamp": "08-31-2023",
  "instruction": "Šajā uzdevumā jūs pārbaudīsiet teikumu no tiešsaistes platformas Pakalpojumu sniegšanas noteikumu (ToS) dokumenta un noteiksit negodīgu līguma noteikumu veidus [...]",
  "prompt": "Passage: notice to california subscribers : you may cancel your subscription , without penalty or obligation , at any time prior to midnight of the third business day following the date you subscribed . \n",
  "answer": "Label(s): n/a"
}
```

## Where it came from

Built by Joel Niklaus, Lucia Zheng, Arya D. McCarthy and co-authors at Stanford and elsewhere; the build code is at https://github.com/JoelNiklaus/LawInstruct [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] lawinstruct/lawinstruct dataset card (README). https://huggingface.co/datasets/lawinstruct/lawinstruct/raw/main/README.md. Fetched 2026-09-23.

[2] Niklaus et al., "LawInstruct: A Resource for Studying Language Model Adaptation to the Legal Domain", arXiv:2404.02127, 2024. https://arxiv.org/abs/2404.02127 - current title read from the live abs page. Fetched 2026-09-23.

[3] Repository file tree for lawinstruct/lawinstruct. https://huggingface.co/api/datasets/lawinstruct/lawinstruct/tree/main?recursive=true. Fetched 2026-09-23.

[4] Hugging Face Hub API record for lawinstruct/lawinstruct. https://huggingface.co/api/datasets/lawinstruct/lawinstruct?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[5] Hugging Face Hub API record for kiddothe2b/contract-nli. https://huggingface.co/api/datasets/kiddothe2b/contract-nli?full=true - `license: cc-by-nc-sa-4.0`. Fetched 2026-09-23.

[6] datasets-server size, info and splits endpoints for lawinstruct/lawinstruct. https://datasets-server.huggingface.co/size?dataset=lawinstruct%2Flawinstruct - each answered with the error quoted in the card instead of a size. Fetched 2026-09-23.

[7] This skill's own measurement, `references/contamination.md`, row "LawInstruct" - n-gram containment with its three controls; method and script in that file - LegalBench test vs each downloaded LawInstruct file; word 8-grams. Run 2026-09-23.

[8] lawinstruct/lawinstruct data files downloaded from https://huggingface.co/datasets/lawinstruct/lawinstruct/resolve/main/data/<file> and read line by line - row counts, field names, `instruction_language` values, first rows. Fetched 2026-09-23.

[9] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:lawinstruct/lawinstruct&sort=downloads - live list, unpinned. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable for multilingual legal instruction tuning, file by file, after dropping the files that overlap the benchmarks you will report; LegalBench citizenship and MAUD scores are meaningless after training on the whole collection. Read the files directly - the card's field list is wrong.

### The screening row

The row's own note: "multilingual legal instruction collection; built from benchmark train splits; card documents a nonexistent `text` field." The row carries the flag `contains-benchmark-items`.
