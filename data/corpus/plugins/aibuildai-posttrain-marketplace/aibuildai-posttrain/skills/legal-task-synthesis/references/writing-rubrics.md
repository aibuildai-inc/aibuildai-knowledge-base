# Writing the rubric for a synthesized legal task

Read when the task's output is work product and the grade will be a list of criteria. `legal-evaluation` covers how published benchmarks score rubrics and how far LLM judges agree with lawyers (`rubric-scoring.md`, `llm-judges.md`). This card covers writing the rubric for a task you built yourself.

For a synthesized task the rubric is also the training signal. A vague criterion becomes a noisy reward, and a criterion that the instruction gives away becomes a free one.

## What the expert-written sets require of a criterion

PRBench gave its authors five requirements [1]:

- **Constructive**: "Each criterion should be correct, precise, and free of internal errors or misconceptions."
- **Mutually exclusive and collectively exhaustive**: no criterion repeated, and together "thorough enough to cover all aspects of a perfect response."
- **Atomic**: "Each rubric criterion evaluates exactly one distinct aspect and should contain no bundling of multiple criteria into a single criterion."
- **Objective**: "Criteria should be binary (true or false)".
- **Self-contained**: "All info needed to score a response must be included in the criterion."

Weights ran "between -10 and 10, excluding 0", and "a criterion with a negative score describes undesired properties" [1]. APEX-Agents uses the same form: "self-contained, short, descriptive statements, which can be graded as true or false" [2]. Harvey LAB breaks a deliverable into "atomic, binary pass/fail criteria: facts, conclusions, citations, severity ratings, recommendations, deadlines, dollar amounts, and formatting choices" [3].

## How they checked the rubric

- **A second expert.** In PRBench "A second domain expert then reviews, edits, and provides feedback", and independent validation reached "93.9% agreement between experts on the validity of rubrics" [1].
- **Grade the gold answer.** APEX-Agents experts "manually graded gold outputs against the rubric when creating them", so that prompt and rubric agree [2].
- **Have someone else do the task.** For 20% of APEX-Agents tasks, experts who had not seen them did them from scratch; in 10% of those "we identified a minor problem with the prompt, rubric, or metadata that needed to be fixed" [2]. One task in ten had a defect its authors missed.
- **Many reviews.** Each GDPval task "received an average of five human reviews (with a minimum of three reviews)" [4].

## Rubrics written by a model

Rubrics as Rewards has a model write a rubric of 7 to 20 self-contained items from the question and a reference answer, labelled Essential, Important, Optional or Pitfall. Its ablation is the finding to remember: "Synthetic rubrics generated without access to reference answers perform notably worse" [5]. A model asked to write a rubric from the instruction alone writes what a good answer usually contains, not what this matter requires.

So for a synthesized legal task, give the rubric writer something that fixes the answer: the facts ledger and planted issues of a document set (`document-sets.md`), the record's own answer (`labels-from-the-record.md`), or a reference answer a lawyer has checked.

## Criteria for legal work

- **Facts**: one criterion per exact fact the work product must state, with the value in the criterion ("states that the cap is 12 months of fees"). Take them from the ledger.
- **Issues**: one per planted or known issue, phrased as identification plus consequence.
- **Authority**: the provision or case that must be cited, where one must be. Add a negative criterion for citing authority that does not exist or is from the wrong jurisdiction; an LLM judge will not check existence on its own (`legal-evaluation`: `llm-judges.md`), so pair the rubric with a citation check (`checking-synthesized-tasks.md`).
- **Errors that matter in practice**, as negative criteria: a missed deadline, advice the facts do not support, a statement contradicted by a document in the set.
- **Form**, only where the work requires it: the requested file type, tracked changes where a markup was asked for.

Keep the instruction and the rubric apart. If the instruction lists the issues to address, the criteria for spotting them measure nothing.

## Three tests before a rubric is used

1. **The gold answer passes every positive criterion.** If it does not, the rubric or the answer is wrong.
2. **A blank or boilerplate answer passes none.** A criterion that a generic memo satisfies carries no signal.
3. **A deliberately wrong answer fails the criteria it should**, and trips the negative ones. Write one wrong answer per task for this; it also becomes a hard negative for judge calibration.

## What to do

1. Write or generate the gold answer first, from the ledger or the record, then the rubric from the gold answer and the ledger.
2. Keep criteria binary and self-contained, with the expected value inside the criterion.
3. Vary the number of criteria with the task. Published sets run from a mean of 4 to dozens per task (`task-anatomy.md`).
4. Run the three tests, with the judge you will train against.
5. Choose aggregation for the reward from `legal-evaluation`: `rubric-scoring.md`. All-pass is a reporting metric; the criterion fraction is the usual reward.

## Sources

Every source was read on 2026-10-03.

[1] "PRBench: Large-Scale Expert Rubrics for Evaluating High-Stakes Professional Reasoning". https://arxiv.org/html/2511.11562

[2] "APEX-Agents". https://arxiv.org/html/2601.14242

[3] Harvey, "Open-Sourcing Harvey's Long Horizon Legal Agent Benchmark". https://www.harvey.ai/blog/introducing-harveys-legal-agent-benchmark

[4] "GDPval". https://arxiv.org/html/2510.04374

[5] "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains". https://arxiv.org/html/2507.17746
