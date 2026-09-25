# ricdomolm/caselawqa-8k

CaselawQA: 10,000 legal classification questions over full U.S. court opinions, half from the Supreme Court Database and half from the Songer Court of Appeals Database, sampled from the Lawma test splits - an evaluation set.

**ricdomolm/caselawqa-8k** is CaselawQA, "a benchmark comprising legal classification tasks, drawing from the Supreme Court and Songer Court of Appeals legal databases" [1]. "The majority of its 10,000 questions are multiple-choice, with 5,000 sourced from each database," and "The questions are randomly selected from the test sets of the Lawma tasks" [1]. The `8k` in the name matches the per-tokenizer length variants of the Lawma instructions; the card itself does not explain it. It lives at https://huggingface.co/datasets/ricdomolm/caselawqa-8k .

**This is an evaluation set: hold out every split. Never train on it, and never train on `ricdomolm/lawma-tasks` `test` splits if you will report CaselawQA.**

**Use it for**: evaluating long-context legal classification. The `1k` config (1,000 `test` questions) is a small slice for quick runs; `sc` and `songer` give 5,000 each [2].

**Licence**: none on the card - no `license` field [3]. Ungated [3]. The one catch: no licence is stated; the source `ricdomolm/lawma-tasks` is MIT.

**Shape**: 22,000 rows over three configs, each with `val` and `test`: `1k` 1,000 / 1,000; `sc` 5,000 / 5,000; `songer` 5,000 / 5,000 [2]. Six columns (`opinion`, `instruction`, `question`, `choices`, `answer`, `task`) [4].

**Hold out**: the whole dataset - it is a benchmark. Its source, the Lawma `test` splits, must also be kept out of training.

**Origin**: the Lawma tasks' expert-coded labels over U.S. opinions [1]. Hub API at the check date: `downloads` 59, `downloadsAllTime` 1,867, `likes` 3 [3].

**Trained-on-by**: an evaluation set; no model declares it through its dataset tag [5].

**Introduced by**: [6] (Dominguez-Olmedo et al.); the card points to https://github.com/socialfoundations/lawma [1].

## Shape

| config | split | rows |
| --- | --- | --- |
| `1k` | `val` | 1,000 |
| `1k` | `test` | 1,000 |
| `sc` | `val` | 5,000 |
| `sc` | `test` | 5,000 |
| `songer` | `val` | 5,000 |
| `songer` | `test` | 5,000 |
| all 3 configs | `val` 11,000 / `test` 11,000 | 22,000 |

All three configs share one schema; `1k` shown [4]:

| column | dtype |
| --- | --- |
| `opinion` | string |
| `instruction` | string |
| `question` | string |
| `choices` | list<string> |
| `answer` | list<int64> |
| `task` | string |

## Quality

- No source states whether the `val` splits are drawn from Lawma `val` or `test`; the card says only that questions come from the test sets [1].
- No source states whether `1k` is a subset of `sc` and `songer`.

## Load it

Evaluate on `test`; use `val` only to choose prompts. Pin the revision:

```python
import datasets

REV = "2cf6113b00220ef83343de87e46479a4c2214522"  # main at the check date
quick = datasets.load_dataset("ricdomolm/caselawqa-8k", "1k", revision=REV, split="test")   # 1,000 rows
sc = datasets.load_dataset("ricdomolm/caselawqa-8k", "sc", revision=REV, split="test")      # 5,000 rows
```

**Trap**: the split is named `val`, not `validation` [2]; `split="validation"` fails. And, as in the parent, `answer` is a list of choice indices, not a letter [4].

## Neighbors

- `ricdomolm/lawma-tasks` - the source; its `test` splits contain these questions.
- `ricdomolm/caselawqa-subtasks-8k` and `ricdomolm/caselawqa_leaderboard_results` - a per-subtask variant and published scores, found by the Hub search [7]; not screened here.

## A row

From `config="1k"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`) [8], with long fields truncated:

```json
{
  "opinion": "WEST POINT WHOLESALE GROCERY CO. v. CITY OF OPELIKA, ALABAMA.\nNo. 478.\nArgued April 24, 1957.\nDecided June 17, 1957.\nM. R. Schlesinger argued the cause for appellant. With him on the brief were N. D. Denson and Tom B. Slade.\nR. E. L. Cope argued the  [...]",
  "instruction": "What follows is an opinion from the Supreme Court of the United States. Your task is to determine whether the decision of the court whose decision the Supreme Court reviewed was itself liberal or conservative. In the context of issues pertaining to c [...]",
  "question": "What is the ideological direction of the decision reviewed by the Supreme Court?",
  "choices": [
    "Conservative",
    "Liberal",
    "Unspeciﬁable"
  ],
  "answer": [
    1
  ],
  "task": "sc_lcdispositiondirection"
}
```

## Where it came from

Sampled by the Lawma authors from the test splits of the Lawma tasks, 5,000 questions per database [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-23; that date covers every number, quote, and corpus row above unless a line says it was measured. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] ricdomolm/caselawqa-8k dataset card (README). https://huggingface.co/datasets/ricdomolm/caselawqa-8k/raw/main/README.md. Fetched 2026-09-23.

[2] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=ricdomolm%2Fcaselawqa-8k - takes no revision parameter; a live figure. Fetched 2026-09-23.

[3] Hugging Face Hub API record for ricdomolm/caselawqa-8k. https://huggingface.co/api/datasets/ricdomolm/caselawqa-8k?full=true - licence field, gate, `sha`, `downloads`, `likes`, tags; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-09-23.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=ricdomolm%2Fcaselawqa-8k - column schema; live, no revision parameter. Fetched 2026-09-23.

[5] Hugging Face Hub model search filtered by dataset tag. https://huggingface.co/api/models?filter=dataset:ricdomolm/caselawqa-8k&sort=downloads - live list, unpinned. Fetched 2026-09-23.

[6] Dominguez-Olmedo et al., "Lawma: The Power of Specialization for Legal Annotation", arXiv:2407.16615, 2024. https://arxiv.org/abs/2407.16615 - current title read from the live abs page. Fetched 2026-09-23.

[7] Hugging Face Hub dataset search. https://huggingface.co/api/datasets?search=caselawqa&sort=downloads. Fetched 2026-09-23.

[8] datasets-server first-rows endpoint, one call per config and split. https://datasets-server.huggingface.co/first-rows?dataset=ricdomolm%2Fcaselawqa-8k&config=<config>&split=<split> - up to 100 rows read at offset 0. Fetched 2026-09-23.

## Appendix: screening record

### Screening verdict

An evaluation set: hold out all of it. Its source is the `test` split of `ricdomolm/lawma-tasks`, which must be excluded from any training mix that will report CaselawQA.

### The screening row

The row's own note: "CaselawQA benchmark, drawn from Lawma test splits." The row carries no flag.
