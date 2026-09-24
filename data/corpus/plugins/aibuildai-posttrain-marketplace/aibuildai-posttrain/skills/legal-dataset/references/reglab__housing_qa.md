# reglab/housing_qa

HousingQA: 6,853 yes/no questions about 2021 U.S. state housing and eviction law, each with the state statutes that answer it, plus a larger 9,297-question set without statute labels and a statute corpus for RAG.

**reglab/housing_qa** is "a benchmark evaluating LLMs' abilities to answer questions about housing law in different states" from Stanford RegLab [1]. It supports three questions: whether a model knows state housing law from its weights, whether it can answer given the relevant statute excerpt, and whether a retrieval system can find the statute first [1]. Questions come from the Legal Services Corporation Eviction Laws Database and statutes were scraped from Justia; "all questions and statutes are only accurate as of 2021" [1]. It lives at https://huggingface.co/datasets/reglab/housing_qa .

**This is an evaluation set. And 64.8% of the 6,853 `questions` answers are "No", so a model that always answers "No" scores 64.8%.**

**Use it for**: evaluating jurisdiction-specific legal knowledge and statute-grounded answering. The `questions` config has statute annotations and supports RAG evaluation; `questions_aux` is larger but "does not contain statute annotations" [1].

**Licence**: CC BY-SA 4.0 in the card metadata [2]. The one catch: statutes were scraped from Justia [1], whose terms the card does not discuss.

**Shape**: `questions` 6,853; `questions_aux` 9,297; `statutes` a corpus in a 731,952,596-byte zip (3.36 GB unzipped) [3][4]. The viewer does not run the script [5].

**Hold out**: all of it. `questions_aux` contains all of `questions` [1].

**Origin**: questions and answers from the Legal Services Corporation Eviction Laws Database; statutes from Justia [1]. Hub API at the check date: `downloads` 282, `downloadsAllTime` 6,427, `likes` 3 [2].

**Trained-on-by**: an evaluation set; no model declares it [6].

**Introduced by**: the RegLab legal RAG benchmarks paper, https://reglab.github.io/legal-rag-benchmarks/ [1].

## Shape

The viewer serves no rows [5]. Counted from the downloaded zips [3]:

| config | split | rows | answer "No" / "Yes" | states |
| --- | --- | --- | --- | --- |
| `questions` | `test` | 6,853 | 4,439 / 2,414 | 48 |
| `questions_aux` | `test` | 9,297 | 6,086 / 3,211 | 55 |
| `statutes` | `corpus` | - | - | - |

Each `questions` row has `idx`, `state`, `question`, `answer`, `question_group` and a `statutes` list of `{statute_idx, citation, excerpt}` [3].

## Quality

- The same question text is asked for every state (`question_group` links them) [3]; per-state accuracy is more informative than the pooled number.
- Law accurate "as of 2021" [1]: a model that knows 2025 law can be marked wrong.

## Load it

Evaluate on `test`, pinned:

```python
import datasets

REV = "761550cc974fa1d9141ffd39014db89efa2a7230"  # main at the check date
q = datasets.load_dataset("reglab/housing_qa", "questions", revision=REV, split="test", trust_remote_code=True)          # 6,853
statutes = datasets.load_dataset("reglab/housing_qa", "statutes", revision=REV, split="corpus", trust_remote_code=True)  # large
```

**Trap**: the `statutes` config's split is named `corpus`, and the question configs' split is `test` - there is no `train` [7].

## Neighbors

- `reglab/barexam_qa` - the same group's bar-exam QA and retrieval set.

## A row

The viewer serves no rows. The first entry of `questions.json` from `data/questions.json.zip`, downloaded [3], truncated:

```json
{
  "idx": 0,
  "state": "Alabama",
  "question": "Is there a state/territory law regulating residential evictions?",
  "answer": "Yes",
  "question_group": 69,
  "statutes": [{"statute_idx": 431263, "citation": "ALA. CODE § 35-9A-141(11)", "excerpt": "(11) “premises” means a dwelling unit and the structure of which it is a part [...]"}]
}
```

## Where it came from

Built by Stanford RegLab from the Legal Services Corporation Eviction Laws Database and Justia's statute pages [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] reglab/housing_qa dataset card (README). https://huggingface.co/datasets/reglab/housing_qa/raw/main/README.md. Fetched 2026-09-23.

[2] Hugging Face Hub API record for reglab/housing_qa. https://huggingface.co/api/datasets/reglab/housing_qa?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[3] reglab/housing_qa files `data/questions.json.zip` and `data/questions_aux.json.zip`, downloaded - rows, answer balance and states counted. Fetched 2026-09-23.

[4] Repository file tree for reglab/housing_qa. https://huggingface.co/api/datasets/reglab/housing_qa/tree/main?recursive=true. Fetched 2026-09-23.

[5] datasets-server size, info and splits endpoints for reglab/housing_qa. https://datasets-server.huggingface.co/size?dataset=reglab%2Fhousing_qa - each answered with the error quoted in the card instead of a size. Fetched 2026-09-23.

[6] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:reglab/housing_qa&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[7] reglab/housing_qa repository file `housing_qa.py`. https://huggingface.co/datasets/reglab/housing_qa/raw/main/housing_qa.py. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

An evaluation set: report per state and against the 64.8% always-"No" baseline. Law is as of 2021.

### The screening row

The row's own note: "HousingQA, 2021 state housing law; imbalanced yes/no." The row carries no flag.
