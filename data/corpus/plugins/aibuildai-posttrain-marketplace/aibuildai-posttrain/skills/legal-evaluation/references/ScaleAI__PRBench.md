# PRBench

PRBench: 1,100 professional-reasoning tasks in legal and finance, written by 182 qualified professionals (JDs, CFAs, or six or more years of experience), with 19,356 expert-written rubric criteria in the paper [1]. Its prompts cover 114 countries and dependencies and 47 U.S. jurisdictions across both domains, which makes it the broadest rubric-graded legal advice set with a published judge agreement.

**Grades**: rubric, binary per criterion, weighted.

**Score**: per task, the sum of the weights of met criteria divided by the sum of positive weights; weights run from −10 to +10; the overall score is floored at 0 [1].

**Judge**: `o4-mini`. On 101 tasks its Cohen's κ against experts was 0.605, against 0.589 between experts [1].

**Access and licence**: open, CC BY 4.0, HF `ScaleAI/PRBench`: splits `legal` 500 and `legal_hard` 250 (finance 600 and 300). The card counts 18,692 criteria where the paper says 19,356 [2].

**Use it for**: rubric-graded legal advice across jurisdictions; a judge-validated reference point for building a legal rubric reward.

**Trap**: jurisdiction varies task to task. A model trained on one jurisdiction's law will fail criteria that assume another; filter by jurisdiction before comparing.

## How it is graded

Each criterion is judged met or not met. Negative-weight criteria describe errors: a met negative criterion subtracts. Top models scored 0.37 on the hard legal subset in the paper [1].

## Sources

Every source was read on 2026-09-29.

[1] "PRBench: Large-Scale Expert Rubrics for Evaluating High-Stakes Professional Reasoning". https://arxiv.org/abs/2511.11562 ; https://arxiv.org/html/2511.11562

[2] Hugging Face Hub API record and card. https://huggingface.co/api/datasets/ScaleAI/PRBench ; https://huggingface.co/datasets/ScaleAI/PRBench
