# crosbylegal/RedlineBench

RedlineBench: 140 agentic contract-negotiation tasks - an agent marks up a SaaS or services agreement as a real `.docx` with tracked changes and comments over four alternating turns - graded by an LLM judge panel against attorney-written weighted rubrics. An evaluation set for agentic contract markup.

**crosbylegal/RedlineBench** "measures contract negotiation as a sequence of judgment calls rather than a collection of isolated clause edits", asking agents to produce "a real Word `.docx` with native tracked changes and threaded margin comments, graded against attorney-authored rubrics by an LLM judge panel" [1]. It has 140 Harbor tasks over 3 negotiation scenarios and 4 turns; tasks in one input group share the same input and differ only in which attorney's rubric grades them [1]. It lives at https://huggingface.co/datasets/crosbylegal/RedlineBench , with code at https://github.com/crosbylegal/redline-bench .

**This is an evaluation set: hold out all 140 tasks. The rubrics and the golden attorney redlines sit in each task's `tests/` folder beside the inputs; never train on that folder.**

**Use it for**: evaluating an agent's contract redlining: producing a marked-up `.docx` with tracked changes and comments, over several negotiation turns, judged against attorney rubrics.

**Licence**: CC BY 4.0 for the data, MIT for the code ("© 2026 Crosby Legal") [1][2].

**Shape**: 140 rows in one config (`tasks`), split `test` [3]; each row indexes one task bundle. The repository holds 4,237 files: 348 `.docx`, 156 `.pdf` playbooks, 140 `rubrics.json`, and per-task Docker environments [4].

**Hold out**: all of it. The rubrics and golden redlines sit under each task's `tests/` folder in the same repository.

**Origin**: synthetic scenarios with fictional parties (AgentCo, LargeCo, GiantCo); rubrics written "in real time" by senior technology-transactions attorneys as they negotiated each turn; 138 of 140 tasks carry a golden attorney redline [1]. Hub API at the check date: `downloads` 1,525, `downloadsAllTime` 8,509, `likes` 16 [2].

**Trained-on-by**: an evaluation set; the Hub's model search by dataset tag returns no model [5].

**Introduced by**: [1] (dataset card) and the published report at https://intelligence.crosby.ai/benchmark .

## Shape

| config | split | rows |
| --- | --- | --- |
| `tasks` | `test` | 140 [3] |

Viewer columns [1][3]: `task_id`, `scenario_id`, `scenario_label`, `turn`, `represented_party`, `counterparty`, `rubric_variant`, `instruction_preview` (first 500 characters of `instruction.md`), `rubric_count`, `rubric_category_counts`, `rubric_criteria_preview` (first five criteria), `contract_path`, `rubrics_path`, `attorney_redline_doc_path`, `source_task_path`.

Rubric criteria by dimension [1]: commercial context 33.4%, legal correctness 25.7%, negotiation quality 17.0%, deal-closing orientation 13.7%, counterparty-acceptance prediction 10.2%. Weights run from -10 to 10, and a task's reward is a weighted pass rate, not an all-pass score [1].

## Quality

- The viewer rows are an index, not the task: the instruction, contract, playbooks and rubrics are files under `tasks/<task_id>/` [1].
- One contract is negotiated from three sides across four turns, so the 140 tasks are far fewer independent documents than the row count suggests [1].
- Two turn-4 tasks have no golden redline by design: the right move is to accept and close [1].

## Load it

```python
from huggingface_hub import snapshot_download

REV = "eee1b6790982ed1279e86bec7616b662a61993e6"  # main at the check date
path = snapshot_download("crosbylegal/RedlineBench", repo_type="dataset", revision=REV)
# run with the authors' driver: https://github.com/crosbylegal/redline-bench
```

**Trap**: loading the `tasks` config gives 140 index rows and truncated previews, not the tasks. Use the task bundles, and keep each bundle's `tests/` folder (rubrics, judge, golden redline) away from the agent and out of any training mix.

## Neighbors

- `harveyai/harvey-labs` - Harvey LAB; its markup task families test the same deliverable across many practice areas.
- `theatticusproject/cuad-qa` - clause identification in commercial contracts, the reading half of redlining.

## A row

From `config="tasks"`, `split="test"`, `row_idx=0` (datasets-server `/first-rows`), previews shortened [6]:

```json
{
  "task_id": "redline-s1-t1-g01a",
  "scenario_label": "Vendor-led SaaS MSA",
  "turn": 1,
  "represented_party": "AgentCo",
  "counterparty": "LargeCo",
  "rubric_count": 15,
  "rubric_criteria_preview": [
    "Rejects in Section 1.3 the inclusion of PCI-DDS Standards in the definition of Applicable law",
    "Replaces in Section 1.4 unilateral Confidential Information protection for LargeCo systems with mutual Confidential Information protection."
  ],
  "contract_path": "tasks/redline-s1-t1-g01a/environment/app/contract.docx",
  "rubrics_path": "tasks/redline-s1-t1-g01a/tests/rubrics.json"
}
```

## Where it came from

Crosby Legal with micro1 ("Crosby-micro1 RedlineBench"); registered as a Hugging Face benchmark with one leaderboard, `redline_overall` [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-24; that date covers every number, quote, and row above unless a line says it was measured. Hub repositories are mutable, which is why Load it pins the revision.

[1] crosbylegal/RedlineBench dataset card (README). https://huggingface.co/datasets/crosbylegal/RedlineBench/raw/eee1b6790982ed1279e86bec7616b662a61993e6/README.md. Fetched 2026-09-24.

[2] Hugging Face Hub API record. https://huggingface.co/api/datasets/crosbylegal/RedlineBench?full=true - licence field, `sha`, `lastModified` 2026-06-18; `downloads`, `downloadsAllTime`, `likes` through the `expand[]` variant. Fetched 2026-09-24.

[3] datasets-server size and info endpoints. https://datasets-server.huggingface.co/size?dataset=crosbylegal%2FRedlineBench - 140 rows, `partial` false. Fetched 2026-09-24.

[4] Repository tree. https://huggingface.co/api/datasets/crosbylegal/RedlineBench/tree/eee1b6790982ed1279e86bec7616b662a61993e6?recursive=true - 4,237 files. Fetched 2026-09-24.

[5] Hub model search by dataset tag. https://huggingface.co/api/models?filter=dataset:crosbylegal/RedlineBench&sort=downloads - empty. Fetched 2026-09-24.

[6] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=crosbylegal%2FRedlineBench&config=tasks&split=test. Fetched 2026-09-24.

## Appendix: screening record

### Screening verdict

An evaluation set: hold out every task. Stages 0-2 answered from the Hub records: readable, not gated, `partial` false, licence field and body agree. Stage 1 confirms an evaluation set: `eval.yaml` at the repository root, rubrics under `tests/`.

### The screening row

The row's own note: "RedlineBench, attorney-rubric contract redlining benchmark; 140 tasks; evaluation only." The row carries no flag.
