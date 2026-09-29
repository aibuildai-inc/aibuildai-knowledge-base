# BigLaw Bench

BigLaw Bench: Harvey's benchmark of transactional and litigation tasks drawn from large-firm practice, graded by attorney-written rubrics that give points for requirements and take them away for hallucination and extraneous material [1][2]. Only samples are public.

**Grades**: rubric with positive and negative points; a separate source score.

**Score**: answer score = (positive points − negative points) / total positive points, read as "what % of a lawyer-quality work product" the model produced. Source score = share of correct statements backed by an accurate source [1].

**Judge**: not stated in the launch post [1].

**Access and licence**: samples only: six core tasks with rubrics, SPA deal-point rows and retrieval queries; "for access to the full dataset ... contact Harvey". No licence stated [2].

**Use it for**: a model of how a firm writes legal rubrics; the sample rubrics show the penalty style (for example "-1 point for every hallucination", "-0.5 point for every statement of accurate but extraneous or misconstrued information") [2].

**Trap**: open-ended penalties ("every hallucination") require the grader to enumerate errors, so longer answers carry more exposure to deductions, and scores depend on how thoroughly the grader counts.

## Related Harvey evaluations

- BigLaw Bench: Arena, pairwise preference by lawyers aggregated to Elo; see `building-a-legal-eval.md` [3].
- JudgmentBench was built from 30 BigLaw Bench-style tasks and found rubric scores recovered quality orderings poorly; see `judgmentbench__JudgmentBench.md`.
- Harvey LAB, the agentic benchmark, is carded in `legal-dataset` and graded as in `rubric-scoring.md`.

## Sources

Every source was read on 2026-09-29.

[1] Harvey, "Introducing BigLaw Bench" (2024-08-29). https://www.harvey.ai/blog/introducing-biglaw-bench

[2] BigLaw Bench repository at commit `138fd481b459a00bbd98eeb710f69ada1052bd47`: README, `core-samples.csv`. https://github.com/harveyai/biglaw-bench

[3] Harvey, "Introducing BigLaw Bench: Arena" (2025-11-07). https://www.harvey.ai/blog/introducing-biglaw-bench-arena
