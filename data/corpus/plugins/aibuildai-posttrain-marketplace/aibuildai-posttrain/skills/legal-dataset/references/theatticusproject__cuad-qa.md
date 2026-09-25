# theatticusproject/cuad-qa

CUAD as extractive QA: 510 EDGAR commercial contracts, 41 clause-type questions each, answers as spans - 22,450 train and 4,182 test questions - whose training contracts hold most of the text of LegalBench's CUAD tasks.

**theatticusproject/cuad-qa** is the question-answering form of CUAD v1, "a corpus of more than 13,000 labels in 510 commercial legal contracts that have been manually labeled to identify 41 categories of important clauses that lawyers look for when reviewing contracts in connection with corporate transactions" [1], introduced by Hendrycks et al. [2]. Each question asks for the span of one clause type in one contract, SQuAD-style [1]. It lives at https://huggingface.co/datasets/theatticusproject/cuad-qa .

**LegalBench's 38 `cuad_*` tasks are clauses from these same contracts. Most LegalBench CUAD and `contract_qa` test items come from the 408 training contracts. Training on `train` and reporting LegalBench CUAD tasks measures memory, not skill.**

**Use it for**: SFT for contract clause extraction - given a contract and a clause type, return the span or "none". The expert annotations make it the strongest contract-review training data here.

**Licence**: CC BY 4.0: "free to the public for commercial and non-commercial use" [1][3]. The card adds that "The creators make no representations or warranties regarding the license status of the underlying contracts, which are publicly available and downloadable from EDGAR" [1]. The one catch: the contracts themselves carry no licence grant.

**Shape**: `train` 22,450 / `test` 4,182 questions, per the card [1]; 408 and 102 contracts respectively, counted in the source archive [4]. The viewer does not run the loading script [5]. Five fields: `id`, `title`, `context`, `question`, `answers` [1].

**Hold out**: `test` (4,182 questions, 102 contracts); no test contract title appears in train [4]. LegalBench's `cuad_*` and `contract_qa` tasks are built from these contracts [6]. Hold out LegalBench `cuad_*` entirely if you train on either split.

**Origin**: contracts filed on SEC EDGAR, "manually labeled" by The Atticus Project [1][2]. Hub API at the check date: `downloads` 2,088, `downloadsAllTime` 61,756, `likes` 68 [3].

**Trained-on-by**: the Hub's dataset tag lists many contract models, led by `Ihteshamstar/qwen3-4b-cuad-extractor` (402 downloads) and `CSroseX/Legal_llama3.1_part5` (118) [7].

**Introduced by**: [2] (Hendrycks et al.).

## Shape

The viewer answers "The dataset viewer doesn't support this dataset because it runs arbitrary Python code" [5]. Counts from the card and from the archive the script downloads [1][4]:

| split | questions | contracts | archive file |
| --- | --- | --- | --- |
| `train` | 22,450 | 408 | `train_separate_questions.json` |
| `test` | 4,182 | 102 | `test.json` |
| all (`CUADv1.json`) | 20,910 | 510 | not loaded by the script |

The loading script downloads `https://github.com/TheAtticusProject/cuad/raw/main/data.zip` at load time [8], so the dataset's content lives on GitHub, not in this repository, and the revision pin covers only the script. `train_separate_questions.json` has more questions (22,450 for 408 contracts) than the combined file has for all 510 (20,910), because it separates multi-part questions [4].

## Quality

- Of the 20,910 questions in `CUADv1.json`, 6,702 have at least one answer span; the rest are "clause absent" examples [4].
- No source states a measured duplicate rate.

## Load it

The script fetches from GitHub; pin this repository and, for reproducibility, keep a copy of `data.zip`:

```python
import datasets

REV = "6d9a118fb629814c4384a1e63cfc84f5c01894b3"  # main at the check date
train = datasets.load_dataset("theatticusproject/cuad-qa", revision=REV, split="train", trust_remote_code=True)  # 22,450
test = datasets.load_dataset("theatticusproject/cuad-qa", revision=REV, split="test", trust_remote_code=True)    # 4,182 - hold out
```

**Trap**: about two thirds of questions have no answer [4]; an SFT template that drops empty-answer rows trains a model that always finds a clause. Keep them, with an explicit "no such clause" target.

## Neighbors

- `theatticusproject/cuad` - the raw CUAD v1 release (contract PDFs and text, the master JSON and CSV); see its card.
- `chenghao/cuad_qa` - a re-upload found by the Hub search [9]; not screened here.
- `nguha/legalbench` - its `cuad_*` tasks are clauses from these contracts.

## A row

The viewer serves no rows [5]. The card's documented field layout [1]:

```json
{
  "id": "<contract>__<Clause Type>",
  "title": "<contract file name>",
  "context": "<full contract text>",
  "question": "Highlight the parts (if any) of this contract related to \"<Clause Type>\" that should be reviewed by a lawyer. Details: <definition>",
  "answers": {"text": ["<span>"], "answer_start": [<offset>]}
}
```

This is a schematic of the declared fields, not a fetched row.

## Where it came from

Created by The Atticus Project with Dan Hendrycks, Collin Burns, Anya Chen and Spencer Ball; code and data at https://github.com/TheAtticusProject/cuad [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] theatticusproject/cuad-qa dataset card (README). https://huggingface.co/datasets/theatticusproject/cuad-qa/raw/main/README.md. Fetched 2026-09-23.

[2] Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review", arXiv:2103.06268, 2021. https://arxiv.org/abs/2103.06268 - current title read from the live abs page. Fetched 2026-09-23.

[3] Hugging Face Hub API record for theatticusproject/cuad-qa. https://huggingface.co/api/datasets/theatticusproject/cuad-qa?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] The CUAD archive the loading script downloads, https://github.com/TheAtticusProject/cuad/raw/main/data.zip - files `CUADv1.json`, `train_separate_questions.json`, `test.json`; contracts, questions and answered questions counted by this skill. Fetched 2026-09-23.

[5] datasets-server size, info and splits endpoints for theatticusproject/cuad-qa. https://datasets-server.huggingface.co/size?dataset=theatticusproject%2Fcuad-qa - each answered with the error quoted in the card instead of a size. Fetched 2026-09-23.

[6] This skill's own check by n-gram containment against the benchmark's test items (word 8-grams, or character 15-grams for Chinese), with self, words-sorted and positive controls, at the pinned revision. Run 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:theatticusproject/cuad-qa&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] theatticusproject/cuad-qa repository file `cuad-qa.py`. https://huggingface.co/datasets/theatticusproject/cuad-qa/raw/main/cuad-qa.py. Fetched 2026-09-23.

[9] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=cuad&sort=downloads. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable as the best contract-review SFT data here: train on `train`, hold out `test`. Its training contracts contain most of LegalBench's CUAD test items, so drop LegalBench `cuad_*` from any report after training on it.

### The screening row

The row's own note: "CUAD extractive QA; contracts overlap LegalBench cuad_* tasks." The row carries the flag `overlaps-LegalBench`.
