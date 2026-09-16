# starmpcc/Asclepius-Synthetic-Clinical-Notes

158,114 clinical-note/question/answer instruction triples, spanning eight clinical NLP task types, where both the discharge-summary note and the instruction-answer pair are entirely GPT-3.5-generated text.

**starmpcc/Asclepius-Synthetic-Clinical-Notes** is the official training data for Asclepius, introduced in "Publicly Shareable Clinical Large Language Model Built on Synthetic Clinical Notes" [1]: the authors took real case-report summaries from the PMC-Patients corpus [2][7] and used GPT-3.5-turbo to rewrite each one into a synthetic discharge-summary-style clinical note, then used GPT-3.5-turbo again, in a separate seeded-bootstrap step, to generate an instruction (`question`) and an `answer` for each note across eight task types - Named Entity Recognition, Relation Extraction, Temporal Information Extraction, Coreference Resolution, Question Answering, Abbreviation Expansion, Summarization, and Paraphrasing [1]. The result is a single-turn, note-grounded instruction-tuning corpus for clinical NLP. It lives at https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes . **The data carries a CC-BY-NC-SA 4.0 licence (non-commercial, share-alike) [2][3], and both the note and the question/answer are model-generated: nothing in this file traces back to a real patient record - the source case reports are themselves public literature summaries, not raw charts [1][7].**

**Use it for**: instruction/SFT training on clinical-note-grounded tasks (map `note` + `question` to a prompt and `answer` to the completion) - the SFT method card's format - restricted to non-commercial use by the licence, and understood as synthetic-on-synthetic text rather than a proxy for real clinical documentation quality [1][2][3].

**Licence**: CC-BY-NC-SA 4.0 (`license: cc-by-nc-sa-4.0` in the repo's card metadata), ungated (`"gated": false`, `"private": false`) [3]. The catch: share-alike requires derivatives built on it to carry the same licence, on top of the non-commercial restriction - no source states a separate commercial-use exception [2][3].

**Shape**: 158,114 rows, one config (`default`), one split (`train`), five columns [4][5].

**Hold out**: nothing - the repository serves only a `train` split, and no fetched source states that any row overlaps a named evaluation benchmark; the paper's own held-out test sets for Asclepius are built from separate real-note corpora (MIMIC-III, MIMIC-IV, i2b2, CASI, DiSCQ), not from this file [1].

**Origin**: built by the Asclepius team (Kweon et al.) [1]; both the `note` field and the `question`/`answer` fields are GPT-3.5-turbo (version 0314) generations, with physicians involved only in seed-example and prompt design, not per-row authorship [1][2]. Hub API at the check date: `downloads` 646, `downloadsAllTime` 18,236, `likes` 116 [3].

**Trained-on-by**: the paper's own Asclepius-7B and Asclepius-13B checkpoints, stated to be "trained on synthetic notes" [1]; two of the Hub-hosted checkpoints, Asclepius-7B and Asclepius-Llama3-8B, declare this dataset id directly in their own Hub card metadata (`datasets: - starmpcc/Asclepius-Synthetic-Clinical-Notes`) [8][9]. The dataset's own README additionally lists Asclepius-Llama2-7B/13B and Asclepius-Mistral-7B-v0.3 as further checkpoints from the same project, without independently confirming each one's training-data tag here [2].

**Introduced by**: [1] (Kweon et al.).

## Shape

Rows and splits (datasets-server `/size`) [4]:

| split | rows |
| --- | --- |
| `train` | 158,114 |

One config, `default`, with five columns (datasets-server `/info`) [5]:

| column | dtype |
| --- | --- |
| `patient_id` | int64 |
| `note` | string |
| `question` | string |
| `answer` | string |
| `task` | string |

The README's Data Fields section documents the note column under the name `patient` ("Case report text") [2], but the column actually served, per both `/info` and the sampled rows, is named `note` [5][6] - a reader following the README's field name alone would not find it.

Byte sizes (datasets-server `/size`) [4]: 401,793,563 bytes of original CSV download, 198,605,402 bytes as Parquet, 403,104,396 bytes decoded in memory. No source states sequence-length or token statistics for this release.

Of the first 72 rows read at offset 0 (datasets-server `/first-rows`) [6], all eight task types named in the README and paper appear (Paraphrasing 9, Coreference Resolution 7, Summarization 12, Relation Extraction 16, Temporal Information Extraction 7, Abbreviation Expansion 7, Named Entity Recognition 6, Question Answering 8), and `patient_id` values in that slice are all distinct, running from 0 to 76 with some values absent from the run - consistent with roughly one instruction-answer pair generated per synthetic note rather than all eight task types per note, but this is read only from those 72 rows and not verified across the full 158,114.

## Quality

- The paper reports a perplexity check on note fidelity: under a LLaMA model fine-tuned on 57k real MIMIC-III discharge summaries, real hospital notes scored between 2.186 (in-domain MIMIC-III) and 5.178 (out-of-domain i2b2) perplexity; the source PMC-Patients case reports scored 71.719 before transformation, and the resulting synthetic notes scored 4.816 - inside the real-note range - after transformation [1].
- Instruction-answer pairs were generated by a two-step, seeded-bootstrap process (five hand-crafted, professional-verified seed examples per task, then model-generated instructions, then model-generated answers fed the note back in) rather than free generation, which the paper states produced more detailed instructions and answers than generating both together [1].
- No source states a measured error rate, duplicate rate, or per-row human-review coverage for the 158,114 pairs; physicians are stated to have been involved in seed-example and prompt design, not in reviewing individual generated rows [1].

## Load it

Single config, single split; pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2024-06-14) [3]:

```python
import datasets

REV = "52af6897eb1bf275a8937ae54bf42da381fe9421"  # main at the check date
ds = datasets.load_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes", revision=REV, split="train")  # 158,114 rows
```

**Trap**: the repository ships a single `synthetic.csv` loaded through the `csv` builder [5]; there is no held-out split to pass, so any train/test split for downstream evaluation has to be carved out by the caller, and the README's documented field name `patient` will not match a real column - use `note` [2][5].

## Neighbors

A Hub API listing of datasets by the `starmpcc` author returns only this one dataset repository [3][10] - there is no Hub-hosted sibling or binarized variant. The README names one non-Hub sibling: "Asclepius-R", built the same way but from real MIMIC-III discharge summaries instead of PMC-Patients case reports, with the instruction-answer pairs and the models trained on them made available on PhysioNet [2]. This card did not fetch the PhysioNet page - an attempted fetch failed with an authentication error - so no claim is made here about its access terms beyond what the README states, which is only that it is "now available on Physionet" [2]. Choose Asclepius-R only if real (not synthetic) discharge summaries are needed and separate verification of PhysioNet access terms is done first.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [6]:

```json
{
  "patient_id": 0,
  "note": "Discharge Summary:\n\nPatient: 60-year-old male with moderate ARDS from COVID-19\n\nHospital Course:\n\nThe patient was admitted to the hospital with symptoms of fever, dry cough, and dyspnea. During physical therapy on the acute ward, the patient experienced coughing attacks that induced oxygen desaturation and dyspnea with any change of position or deep breathing. [...] Exercise progression was low initially but increased daily until hospital discharge to a rehabilitation clinic on day 10.\n\nClinical Outcome:\n\nThe patient was discharged on day 10 to a rehabilitation clinic making satisfactory progress with all symptoms resolved.\n\nFollow-up:\n\nThe patient will receive follow-up care at the rehabilitation clinic, with regular monitoring of progress and further rehabilitation exercises until full recovery. Any new symptoms or concerns should be reported to the clinic immediately.\n\nOverall Impression:\n\nThe patient responded well to treatment, and with appropriate medical intervention, was able to overcome the difficulties faced during hospitalization for ARDS from COVID-19. [...]",
  "question": "Can you provide a simplified paraphrase of the sentence, 'To avoid rapid deterioration and respiratory failure, a step-by-step approach was used for position changes' in the patient's discharge summary?",
  "answer": "The healthcare team used a gradual approach to changing the patient's position to avoid worsening of the respiratory status and prevent respiratory failure.",
  "task": "Paraphrasing"
}
```

## Where it came from

Built and released by the Asclepius project team (Kweon, Kim, et al.) [1]. The pipeline runs in two GPT-3.5-turbo stages over case reports drawn from the PMC-Patients dataset of PubMed Central case-report summaries [2][7]: first, each case report is rewritten into a synthetic discharge-summary-style clinical note, validated by the perplexity comparison above; second, for each of the eight defined task types, five professional-verified seed examples per task seed a bootstrapped generation of an instruction, and the note plus generated instruction are then fed back to the model to produce the answer, yielding 158,114 note-instruction-answer triples in total [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision `52af6897eb1bf275a8937ae54bf42da381fe9421` - but that pin covers only the repository file (`synthetic.csv`) fetched through `load_dataset`. The datasets-server endpoints used for Shape, Quality, and A row [4][5][6] take no revision parameter and were read live at the check date; they are reported as check-date snapshots, not as covered by the revision pin.

[1] Kweon et al., "Publicly Shareable Clinical Large Language Model Built on Synthetic Clinical Notes", 2023. https://arxiv.org/abs/2309.00237 - the origin paper; generation pipeline, perplexity validation, task-type list, 158,114-pair count, downstream model claims. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2309.00237). Fetched 2026-08-11.

[2] starmpcc/Asclepius-Synthetic-Clinical-Notes dataset card (README). https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes/raw/main/README.md - dataset summary, documented field names, licence statement, related-model list, Asclepius-R/PhysioNet mention. Fetched 2026-08-11.

[3] Hugging Face Hub API record for starmpcc/Asclepius-Synthetic-Clinical-Notes. https://huggingface.co/api/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes?full=true - licence field, gate/private status, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=starmpcc%2FAsclepius-Synthetic-Clinical-Notes Fetched 2026-08-11.

[5] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=starmpcc%2FAsclepius-Synthetic-Clinical-Notes Fetched 2026-08-11.

[6] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=starmpcc%2FAsclepius-Synthetic-Clinical-Notes&config=default&split=train Fetched 2026-08-11.

[7] zhengyun21/PMC-Patients dataset card (README). https://huggingface.co/datasets/zhengyun21/PMC-Patients/raw/main/README.md - upstream case-report source, its own field names and ReCDS-benchmark split. Fetched 2026-08-11.

[8] starmpcc/Asclepius-7B model card (README). https://huggingface.co/starmpcc/Asclepius-7B/raw/main/README.md - `datasets:` metadata field naming this dataset id. Fetched 2026-08-11.

[9] starmpcc/Asclepius-Llama3-8B model card (README). https://huggingface.co/starmpcc/Asclepius-Llama3-8B/raw/main/README.md - `datasets:` metadata field naming this dataset id. Fetched 2026-08-11.

[10] Hugging Face Hub API dataset listing filtered by author. https://huggingface.co/api/datasets?author=starmpcc&limit=100 - confirms no other Hub dataset repository from the same author. Fetched 2026-08-11.

[11] The corpus screening row for `starmpcc/Asclepius-Synthetic-Clinical-Notes`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT/instruction data, subject to its non-commercial licence, with the doubly-synthetic nature of the text (both the note and the question/answer are GPT-3.5 generations, per the opening paragraph) [1] weighed against the paper's own perplexity evidence that the synthetic notes fall inside the real-hospital-note perplexity range [1]. The screening row's own note names this as the central risk to weigh.

### The screening row

The row's own note [11]: "GPT-3.5 synthesized the discharge summaries AND the instruction/answer pairs; fully synthetic clinical text, the riskiest kind of medical synthetic data because nothing here traces to a real patient record." The row carries no flag.
