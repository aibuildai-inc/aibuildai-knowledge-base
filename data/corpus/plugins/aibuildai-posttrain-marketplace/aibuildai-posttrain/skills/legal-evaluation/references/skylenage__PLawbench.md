# PLawBench

PLawBench: 850 questions across 13 Chinese legal-practice scenarios (case analysis, consultation, document drafting) with about 12,500 rubric items written by legal practitioners and doctoral candidates [1]. Rubric-graded practice work in a civil-law system.

**Grades**: rubric with penalties.

**Score**: scoring rate = Σ points earned / Σ maximum points. Penalties range from −1 to −6 for misleading guidance and similar faults, and harmful or illegal advice forfeits all points [1].

**Judge**: Gemini-3.0-Pro-Preview, chosen for the highest agreement with experts; Pearson 0.809 on analysis tasks [1].

**Access and licence**: partial release: 280 of the 850 questions (250 case analysis, 18 consultation, 12 drafting). No licence stated [2].

**Use it for**: rubric-graded Chinese practice tasks, including drafting; an example of a forfeit rule for harmful advice.

**Trap**: no licence and a partial release: the public 280 are not the paper's 850, so reproduce on the released subset and say so.

## Sources

Every source was read on 2026-09-29.

[1] "PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice", including Table 4. https://arxiv.org/abs/2601.16669 ; https://arxiv.org/html/2601.16669

[2] PLawBench repository at commit `f6ee07b36c1f4c0bfc2af8ca27183d3dd3175c56`: README. https://github.com/skylenage/PLawbench
