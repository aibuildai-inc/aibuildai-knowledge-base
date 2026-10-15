# Magis-Bench

Magis-Bench: 74 questions from eight Brazilian magistrate entrance exams, 2023-2025: 58 discursive questions and 16 sentence drafts (8 civil, 8 criminal), graded against the exam boards' official rubrics [1][2].

**Grades**: rubric (official exam rubrics), 0-10.

**Score**: each judge scores each answer 0-10 against the official rubric [1].

**Judge**: the paper uses four frontier models (GPT-5.1, Gemini-2.5-Pro, Gemini-3-Pro-Preview, Claude-4.5-Opus), with inter-judge Kendall's W 0.984 and no validation against human graders, which the authors note [1]. The repository at the pinned commit defaults to a single judge, `anthropic/claude-opus-4-6`, and keeps the four-judge setup at the tag `paper-icail2026` [2].

**Access and licence**: open, Apache-2.0; data and model answers are in the repository [2].

**Use it for**: judicial drafting and exam writing in Portuguese under Brazilian law, graded by the rubrics real examiners use.

**Trap**: judges agreeing with each other is not judges agreeing with examiners, and the repository's default judge differs from the paper's panel. Say which setup produced a number, and treat scores as a ranking among models, not as a pass mark.

## Sources

Every source was read on 2026-09-29.

[1] Magis-Bench paper. https://arxiv.org/html/2605.08437v1

[2] Magis-Bench repository at commit `b93d0e82795a37ff063bcf7cf89b91dcedebe921`: README, LICENSE. https://github.com/maritaca-ai/magis-bench
