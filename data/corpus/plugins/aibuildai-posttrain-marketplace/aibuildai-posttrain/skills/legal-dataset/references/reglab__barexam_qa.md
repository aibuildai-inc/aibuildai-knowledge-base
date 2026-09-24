# reglab/barexam_qa

Bar Exam QA: 1,195 historical Multistate Bar Exam questions with four choices, the answer, and a gold supporting passage annotated by a law student, plus a ~900K-passage case-law retrieval pool.

**reglab/barexam_qa** is "a dataset for legal information retrieval and question answering" from Stanford RegLab [1]. It "contains the public MBE subset of questions (historical exams)", released with gold passages from primary and secondary sources "annotated by a law student, simulating a legal research workflow" [1]. The authors also evaluate on a private BarBri test set that is not released [1]. It lives at https://huggingface.co/datasets/reglab/barexam_qa .

**Use it for**: evaluating or training legal multiple-choice reasoning and retrieval-augmented answering on U.S. bar-exam law. With only 954 training questions, it suits RAG training or a small SFT mix, not a whole stage.

**Licence**: CC BY-SA 4.0 in the card metadata [2]. The loading script's `_LICENSE` string is empty [3]. The one catch: historical MBE questions are published by the NCBE, and the card does not address their terms.

**Shape**: `qa`: `train` 954 / `validation` 124 / `test` 117 (1,195 in all); `passages`: a pool of about 900K passages, `train`/`validation`/`test` files of 447,902,075 / 54,947,161 / 55,782,786 bytes [4][5]. The viewer does not run the script [6].

**Hold out**: `qa` `test` (117 questions). Measured: no `test` question text appears in `train` [4].

**Origin**: questions from historical Multistate Bar Exams (every `idx` is prefixed `mbe`; row 0's source is `MBE-1972-78-part2`); gold passages chosen by a law student from Caselaw Access Project opinions and secondary sources [4][1]. Hub API at the check date: `downloads` 652, `downloadsAllTime` 8,954, `likes` 5 [2].

**Trained-on-by**: no model on the Hub declares this dataset through its dataset tag [7].

**Introduced by**: the RegLab legal RAG benchmarks paper linked from the card, https://reglab.github.io/legal-rag-benchmarks/ [1].

## Shape

The viewer serves no rows [6]. `qa` rows counted from `data/qa/*.csv` [4]:

| split | rows |
| --- | --- |
| `train` | 954 |
| `validation` | 124 |
| `test` | 117 |
| `data/qa/qa.csv` (all) | 1,195 |

`qa` columns: `idx`, `dataset`, `example_id`, `prompt_id`, `source`, `subject`, `question_number`, `prompt`, `question`, `choice_a`-`choice_d`, `answer`, `gold_passage`, `gold_idx` [4].

## Quality

- Some questions are old: row 0 of `test` comes from the 1972-78 MBE [4], and law has changed since. Scores mix current and historical doctrine.
- `subject` is empty on the sampled `test` row [4].
- The card's field list spells the passage column `gold_passge` [1]; the CSV header is `gold_passage` [4].

## Load it

The script reads CSVs from the repository; pin it:

```python
import datasets

REV = "b7c245a7e4739f6c785faa2489e283c8daa449d1"  # main at the check date
qa_train = datasets.load_dataset("reglab/barexam_qa", "qa", revision=REV, split="train", trust_remote_code=True)  # 954
qa_test = datasets.load_dataset("reglab/barexam_qa", "qa", revision=REV, split="test", trust_remote_code=True)    # 117 - hold out
```

**Trap**: `prompt` holds shared context and `question` the actual question; several questions share one `prompt` (`prompt_id`) [1][4]. Build the input from both, or questions lose their fact pattern.

## Neighbors

- `reglab/housing_qa` - the same group's state housing-law benchmark.
- `isaacus/mteb-barexam-qa` - a retrieval-benchmark repackaging found by the Hub search [8]; not screened here.

## A row

The viewer serves no rows. The first data row of `data/qa/test.csv`, downloaded [4], truncated:

```json
{
  "idx": "mbe_46",
  "dataset": "mbe",
  "source": "MBE-1972-78-part2",
  "subject": "",
  "question_number": "47",
  "prompt": "An ordinance of City makes it unlawful to park a motor vehicle on a City street within ten feet ofa fire hydrant. At 1:55 p.m. Parker, realizing he mu [...]",
  "question": "If Ned asserts a claim against Parker, the most likely result is that Ned will",
  "choice_a": "recover because Parker's action was negligence per se",
  "choice_b": "recover because Parker's action was a continuing wrong which contributed to Ned's injuries",
  "choice_c": "not recover because a reasonably Prudent person could not foresee injury to Ned as a result of Parker's action",
  "choice_d": "not recover because a violation of a city ordinance does not give rise to a civil cause of action",
  "answer": "C",
  "gold_idx": "mbe_2606"
}
```

## Where it came from

Released by Stanford RegLab with gold passages annotated by a law student [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] reglab/barexam_qa dataset card (README). https://huggingface.co/datasets/reglab/barexam_qa/raw/main/README.md. Fetched 2026-09-23.

[2] Hugging Face Hub API record for reglab/barexam_qa. https://huggingface.co/api/datasets/reglab/barexam_qa?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[3] reglab/barexam_qa repository file `barexam_qa.py`. https://huggingface.co/datasets/reglab/barexam_qa/raw/main/barexam_qa.py. Fetched 2026-09-23.

[4] reglab/barexam_qa files `data/qa/train.csv`, `validation.csv`, `test.csv`, `qa.csv`, downloaded - rows counted, headers read, `test` questions matched against `train`. Fetched 2026-09-23.

[5] Repository file tree for reglab/barexam_qa. https://huggingface.co/api/datasets/reglab/barexam_qa/tree/main?recursive=true. Fetched 2026-09-23.

[6] datasets-server size, info and splits endpoints for reglab/barexam_qa. https://datasets-server.huggingface.co/size?dataset=reglab%2Fbarexam_qa - each answered with the error quoted in the card instead of a size. Fetched 2026-09-23.

[7] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:reglab/barexam_qa&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[8] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=barexam&sort=downloads. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

Usable for bar-exam QA and legal RAG: train on `qa` `train`, hold out `test`. Small, partly historical, and with NCBE question terms unaddressed.

### The screening row

The row's own note: "public MBE questions + gold passages; 1,195 questions." The row carries no flag.
