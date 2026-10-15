# LexRubric

LexRubric: 649 Chinese legal items (473 consultation questions, 176 judicial-exam questions) with 12,337 atomic scoring criteria in six dimensions [1]. A rubric benchmark whose open judge model agrees closely with experts.

**Grades**: rubric, binary per criterion, weighted −10 to +10.

**Score**: Σ weight × 1{criterion met} / Σ max(weight, 0), as a percentage [1].

**Judge**: Qwen3.6-27B at temperature 0.0. Against experts: Kendall τ-b 0.800, Spearman 0.886, pairwise accuracy 90% [1].

**Access and licence**: open. HF `chenyifan0929/LexRubric` states CC BY-NC 4.0; the GitHub repository `foggpoy/LexRubric` states MIT. Use the data under the non-commercial terms [2][3].

**Use it for**: a rubric reward with an open-weights judge, which a training loop can run locally at scale.

**Trap**: the data licence is non-commercial, while the code is MIT. A commercial training run cannot use the data even though the repository says MIT.

## Sources

Every source was read on 2026-09-29.

[1] "LexRubric: A Rubric-Guided Diagnostic Benchmark for Open-Ended Legal Tasks". https://arxiv.org/html/2606.09389

[2] Hugging Face Hub API record. https://huggingface.co/api/datasets/chenyifan0929/LexRubric

[3] LexRubric repository at commit `9141ee4b3715834ac7e1507cbfc225cf858f7328`: LICENSE, README. https://github.com/foggpoy/LexRubric
