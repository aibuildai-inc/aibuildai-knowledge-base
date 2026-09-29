# Vals Legal AI Report

The Vals Legal AI Report (VLAIR): industry studies that compare legal AI products with a lawyer baseline on the same tasks [1][2]. Two rounds with two different graders, which makes it a useful contrast between LLM-judged and lawyer-graded evaluation.

**Grades**: round one: per-element pass or fail against a reference response; legal research round: lawyer-graded rubric.

**Score**:
- Round one (updated 2025-02-27): seven tasks (data extraction, document Q&A, summarisation, redlining, transcript analysis, chronology, EDGAR research), 500+ samples; accuracy per task [1].
- Legal research round (updated 2025-10-14): 200 questions; accuracy 50% (0-3), authoritativeness 40% (0-3, whether valid sources are cited), appropriateness 10% (0-2); at least two graders per answer [2].

**Judge**: round one, an LLM judge given the reference and grading guidance, with human review of failures [1]; research round, lawyers and law librarians, "not using any automated systems" [2].

**Access and licence**: reports only; the tasks are not public.

**Use it for**: the design of a lawyer baseline (same instructions, same documents, time recorded) and a research rubric that weights authority at 40%.

**Trap**: the two rounds use different graders and cannot be compared; a claim that a system beats lawyers depends on the time and effort the lawyers were given.

## Sources

Every source was read on 2026-09-29.

[1] Vals AI, Legal AI Report, first round. https://www.vals.ai/industry-reports/vlair-2-27-25

[2] Vals AI, Legal AI Report, legal research. https://www.vals.ai/industry-reports/vlair-10-14-25
