# harveyai/harvey-labs

Harvey LAB, the Legal Agent Benchmark: 2,010 agentic legal tasks on GitHub, each a folder of synthetic matter documents, an instruction, and a pass/fail rubric that two LLM judges grade all-or-nothing. An evaluation set with no training split, and an example of the agentic legal work - reading a matter file and producing work product - that most other legal benchmarks do not test.

**harveyai/harvey-labs** is "an open-source benchmark for evaluating agents on real legal work" [1], announced by Harvey on 2026-05-06 [2]. Each task gives an agent a workspace with a read-only `documents/` folder and asks for deliverables, usually a `.docx` memo, markup or term sheet, written with six tools (`bash`, `read`, `write`, `edit`, `glob`, `grep`) and a `finish` signal [3]. Every task's rubric is inline in its `task.json` as equally weighted pass/fail criteria. The judges are `claude-sonnet-4-6` and `gpt-5.5` by default, and a task scores 1.0 only if every criterion passes [4]. It lives at https://github.com/harveyai/harvey-labs .

**This is an evaluation set with no split: nothing in it is training data unless you carve a split yourself and report only the held-out part.**

**Use it for**: evaluating an agent on long-horizon legal work product: reading a matter file, extracting exact facts, spotting issues, drafting and marking up documents. Its rubrics can also serve as an RL reward on a split you carve, at the cost of the held-out claim for that split.

**Licence**: MIT (`LICENSE`, "Copyright (c) 2026 Harvey AI") [5]. Documents are synthetic: CONTRIBUTING requires "synthetic people, companies, law firms, funds, products, addresses, and matter facts" [6].

**Shape**: at commit `1dd8140`, 2,010 `task.json` files in 27 top-level areas, 114,437 rubric criteria (median 54 per task), and 48,687 source documents with extractable text (measured [7]). The README badge says 1,671 tasks and the evaluation guide 1,660 tasks with about 101,000 criteria [1][4]; they may count scenarios and `firm-knowledge` differently.

**Hold out**: every task. It was published on 2026-05-06, so a training set whose files were all committed before that date cannot contain it; anything newer that mentions Harvey or LAB, including agent trajectories, needs screening before training.

**Origin**: synthetic documents "generated in large batches, under the guidance and review of human lawyers" [8]; tasks started "with real client matters handled by practicing lawyers in each practice area, then decomposed into discrete associate-level assignments" [2]. Version tags `v1.0` and `v1.1.0` [9].

**Trained-on-by**: an evaluation set.

**Introduced by**: [2] (Harvey, announcement post); cited as "Harvey LAB: The Legal Agent Benchmark", v1.0 [1].

## Shape

Tasks by area at commit `1dd8140` (measured [7]):

| area | tasks | area | tasks |
| --- | --- | --- | --- |
| `contracts` | 498 | `healthcare-life-sciences` | 43 |
| `firm-knowledge` | 250 | `emerging-companies-venture-capital` | 43 |
| `corporate-ma` | 161 | `international-trade-sanctions` | 41 |
| `intellectual-property` | 147 | `employment-labor` | 39 |
| `corporate-governance` | 97 | `banking-finance` | 37 |
| `trusts-estates-private-client` | 77 | `arbitration-international-dispute-resolution` | 37 |
| `funds-asset-management` | 66 | `bankruptcy-restructuring` | 36 |
| `litigation-dispute-resolution` | 52 | `capital-markets` | 35 |
| `real-estate` | 44 | `tax` | 34 |
| `data-privacy-cybersecurity` | 44 | `antitrust-competition` | 33 |
| `environmental-esg` | 44 | 6 more areas | 152 |

Work types (`work_type` field): `analyze` 488, `draft` 444, `review` 306, `research` 24; the 498 `contracts` tasks and 250 `firm-knowledge` tasks carry none [7]. Deliverables: 1,560 `.docx`, 261 `.md`, 45 `.xlsx`. Source files: 33,954 `.docx`, 10,575 `.xlsx`, 5,169 `.eml`, 1,091 `.pptx`, 889 `.txt`; median 8 per task outside `firm-knowledge` [7]. Jurisdiction is mostly U.S.: Delaware appears in 29% of task instructions and rubrics, Texas 13%, the SEC or the Securities and Exchange Acts 10%, the EU or GDPR 9%, English law 2% (measured by keyword [7]).

`firm-knowledge` is different in kind: its 250 tasks share one document store of 9,288 files (`tasks/firm-knowledge/dms/`) and each has a single criterion answered in `response.md` [7].

## Quality

- Scenario variants (`scenario-01`, `scenario-02`, ...) of one task are separate matters with their own documents; `firm-knowledge` tasks share one document store, so a split inside it shares documents.
- Rubric criteria test exact facts as well as judgement ("states the TerraNode annual contract value as approximately $6.2 million" [4]), and all-pass grading means one missed fact fails the task.
- The system prompt forbids the agent to read `task.json`, because the rubric sits beside the documents [10]. An RL environment built on LAB must hide it too.
- Two paid judge models grade every run by default [4]; a scored run costs judge calls on top of the agent's.

## Load it

```bash
git clone https://github.com/harveyai/harvey-labs.git
cd harvey-labs && git checkout 1dd81403b2fbb60596f7aea3fcecafad7bf73143   # main at the check date
```

```python
import json, glob
tasks = {p[len("tasks/"):-len("/task.json")]: json.load(open(p)) for p in glob.glob("tasks/**/task.json", recursive=True)}
```

**Trap**: each task's rubric sits in `task.json` next to its `documents/` folder. An agent or RL environment that can read the task directory can read the answers; the harness forbids it in the system prompt [10], and any environment built on LAB must hide the file.

## Neighbors

- `crosbylegal/RedlineBench` - multi-turn contract redlining benchmark; a separate evaluation set.
- `nguha/legalbench` - short-answer legal reasoning; a different skill from LAB's agentic work product.

## A row

`tasks/real-estate/extract-psa-key-terms/scenario-01/task.json`, first two of its 75 criteria [7]:

```json
{
  "title": "Extract Key Terms from Commercial Property Purchase and Sale Agreement",
  "work_type": "analyze",
  "instructions": "Extract key terms from the attached PSA and flag issues using the supporting documents; produce a detailed term sheet organized by topic with section references.\n\nOutput: `psa-term-sheet.docx`",
  "criteria": [
    {"id": "C-001", "title": "Output file is named psa-term-sheet.docx", "deliverables": ["psa-term-sheet.docx"],
     "match_criteria": "PASS if the agent produces a file named 'psa-term-sheet.docx' (or a substantially similar name like 'psa-term-sheet' in docx format). FAIL if no such file is produced."},
    {"id": "C-002", "title": "Term sheet identifies Buyer as Calverley Capital Partners LLC", "deliverables": ["psa-term-sheet.docx"],
     "match_criteria": "PASS if the term sheet identifies the Buyer as Calverley Capital Partners LLC. FAIL if the Buyer entity name is missing or incorrect."}
  ],
  "deliverables": {"psa-term-sheet.docx": "psa-term-sheet.docx"}
}
```

## Where it came from

Built by Harvey AI; tasks and harness in one repository, with a score-impact `CHANGELOG.md` recording changes to harness, grading, dataset and adapters [1][11].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-09-24; that date covers every number, quote and row above unless a line says it was measured. The repository is mutable, which is why Load it pins the commit.

[1] harveyai/harvey-labs README. https://github.com/harveyai/harvey-labs/blob/1dd81403b2fbb60596f7aea3fcecafad7bf73143/README.md. Fetched 2026-09-24.

[2] Harvey, "Introducing Harvey's Legal Agent Benchmark", 2026-05-06. https://www.harvey.ai/blog/introducing-harveys-legal-agent-benchmark. Fetched 2026-09-24.

[3] docs/architecture.md, sections Harness, Agent Loop and Tools. https://github.com/harveyai/harvey-labs/blob/1dd81403b2fbb60596f7aea3fcecafad7bf73143/docs/architecture.md. Fetched 2026-09-24.

[4] docs/eval-strategies.md. https://github.com/harveyai/harvey-labs/blob/1dd81403b2fbb60596f7aea3fcecafad7bf73143/docs/eval-strategies.md - all-pass scoring, default judge pair, the $6.2 million example criterion, the 1,660-task total. Fetched 2026-09-24.

[5] LICENSE. https://github.com/harveyai/harvey-labs/blob/1dd81403b2fbb60596f7aea3fcecafad7bf73143/LICENSE. Fetched 2026-09-24.

[6] CONTRIBUTING.md. https://github.com/harveyai/harvey-labs/blob/1dd81403b2fbb60596f7aea3fcecafad7bf73143/CONTRIBUTING.md. Fetched 2026-09-24.

[7] This skill's own count on a clone at commit `1dd8140`: tasks, criteria, documents, work types, file types and jurisdiction keywords. Run 2026-09-24.

[8] docs/tutorial.md. https://github.com/harveyai/harvey-labs/blob/1dd81403b2fbb60596f7aea3fcecafad7bf73143/docs/tutorial.md. Fetched 2026-09-24.

[9] Repository tags. https://github.com/harveyai/harvey-labs/tags - `v1.0` at commit `1da47501`, `v1.1.0` at `1dd81403`. Fetched 2026-09-24.

[10] lab_core/harness/system_prompt.md. https://github.com/harveyai/harvey-labs/blob/1dd81403b2fbb60596f7aea3fcecafad7bf73143/lab_core/harness/system_prompt.md. Fetched 2026-09-24.

[11] CHANGELOG.md. https://github.com/harveyai/harvey-labs/blob/1dd81403b2fbb60596f7aea3fcecafad7bf73143/CHANGELOG.md. Fetched 2026-09-24.

## Appendix: screening record

### Screening verdict

An evaluation set: hold out every task you report on. It has no official split; if you carve one for training or RL, split by task family, keep `firm-knowledge` whole on one side, and say so in the report.

### The screening row

The row's own note: "Harvey LAB, the Legal Agent Benchmark on GitHub; agentic legal work product; no split." The row carries no flag.
