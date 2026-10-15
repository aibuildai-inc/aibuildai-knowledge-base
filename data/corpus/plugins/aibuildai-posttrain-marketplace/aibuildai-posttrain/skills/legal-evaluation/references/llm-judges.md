# LLM judges on legal text: how far to trust them, and how to use one as a reward

Read in step 1 when a legal score comes from an LLM judge, and again before using a judge as a training reward.

Most long-form legal benchmarks now grade with an LLM. The published agreement between those judges and lawyers ranges from close to expert-expert agreement down to barely above chance, and depends mostly on how the grading question is framed. A judge also has a disposition, and a policy trained against it learns that disposition.

## Measured agreement with legal experts

| Study | What was judged | Judge | Agreement with experts |
|---|---|---|---|
| PRBench [1] | legal and finance advice, binary rubric criteria, 101 tasks | `o4-mini` | Cohen's κ 0.605 against experts; expert against expert 0.589 |
| LexRubric [2] | Chinese consultation and exam answers, weighted rubric | Qwen3.6-27B | Kendall τ-b 0.800, Spearman 0.886 |
| PLawBench [3] | Chinese practice tasks, rubric | Gemini-3.0-Pro-Preview | Pearson 0.809 on analysis tasks |
| CaseGen [4] | Chinese judgment sections, 1-10 against a reference | `gpt-4o-2024-11-20` | Spearman 0.750 |
| LEXam [5] | Swiss exam answers, 0-1 against a reference | minimum of GPT-4o, Qwen3-32B, DeepSeek-V3 | passed an alternative-annotator test against each of three legal experts on 50 samples; the experts agreed with each other at Pearson 0.70 |
| LeMAJ [6] | contract Q&A split into "legal data points" | Claude 3.5 Sonnet | Pearson 0.26-0.70 by dataset and dimension |
| JudgmentBench [7] | 30 legal tasks, rubric scores versus pairwise preference, practising attorneys | GPT-5.4, GPT-5.4-mini | quality orderings implied by the autograder agreed with the attorneys' at Spearman 0.200-0.242 under rubrics and 0.375-0.611 under pairwise comparison |
| Polish appeal-chamber exam [8] | written judgments | GPT-4o | examiners gave 37, 30 and 8 out of 100; the LLM grader gave 88, 90 and 85 |

Three patterns hold across the table:

- **Atomic, binary criteria written by lawyers agree best** (PRBench, LexRubric). Holistic scores against a reference agree moderately (CaseGen). Holistic scores with no reference or rubric can be far too lenient (the Polish exam) [8].
- **Judges miss invented law.** The Polish exam's grader rated the use of legal provisions highly and overlooked the errors the examiners marked [8]. Judges read fluency; they do not look up authority. Add a citation check (`citations-and-grounding.md`).
- **Rubric scores can be less faithful than preferences.** JudgmentBench found both attorneys and LLM graders recovered quality better by comparing two answers than by filling a rubric, and pairwise took less than half the annotation time [7]. When building an evaluation (not reproducing one), consider pairwise judgment for the headline and rubrics for diagnosis.

## A judge has a disposition, and it can be optimised against

On LEXam, Qwen3-32B grades more leniently and DeepSeek-V3 more strictly; optimising prompts against one judge's feedback raised that judge's score by 2.8-6.7 points and transferred unevenly to the other, while prompts optimised against the judge ensemble gained under one point [9]. Several judges combined are harder to game than one. Training against a judge is the same optimisation at larger scale.

## Rubric rewards: what the general papers do

No peer-reviewed legal rubric-reward RL paper was found; the legal RL papers found reward verifiable labels (for example LegalΔ on CAIL2018 and JEC-QA [10]). The general-domain rubric-reward papers agree on a common recipe:

- **Criteria are binary; the reward is a normalised weighted sum.** Rubrics as Rewards weights criteria as essential 1.0, important 0.7, optional 0.3 and pitfall 0.9, with a `gpt-4o-mini` judge, and reports that asking the judge for one holistic score with the rubric in view ("implicit") did best on HealthBench [11]. HealthBench divides met points by positive points and clips the mean score to [0, 1] [12]; PRBench and LexRubric also normalise by positive points.
- **Rubric generation matters.** Decomposing and decorrelating criteria improved judge accuracy on JudgeBench from 55.6 to 73.3 [13].
- **Guard the reward.** Zero reward on a judge parse failure and a frozen judge at low temperature [14]; randomising answer order to cut position sensitivity [15].

## Rules for a legal run

1. Before training against a judge, grade 30-50 outputs with a lawyer or law-trained reviewer and measure agreement, stratified by task type. A judge that agrees on consultation answers may not agree on drafting.
2. Choose the reward judge by measuring it against a stronger reference. Harvey picked GPT-5 Mini as the grader for training a legal agent after it stayed above 97% alignment with a frontier-model consensus [16]. Where the budget allows, also select and report checkpoints with a judge or panel other than the reward judge, so the policy's exploitation of one judge's disposition shows up.
3. Pin the judge model, prompt, temperature and renderer. A judge change is a benchmark change.
4. Read a sample of passing outputs, not only failing ones: leniency shows up as confident, fluent answers that pass while citing nothing real.

## Sources

Every source was read on 2026-09-29.

[1] "PRBench: Large-Scale Expert Rubrics for Evaluating High-Stakes Professional Reasoning". https://arxiv.org/abs/2511.11562

[2] "LexRubric: A Rubric-Guided Diagnostic Benchmark for Open-Ended Legal Tasks". https://arxiv.org/abs/2606.09389

[3] "PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice", Table 4. https://arxiv.org/abs/2601.16669

[4] CaseGen paper. https://arxiv.org/abs/2502.17943

[5] Fan et al., "LEXam: Benchmarking Legal Reasoning on 340 Law Exams". https://arxiv.org/abs/2505.12864

[6] LeMAJ (Legal LLM-as-a-Judge), NLLP 2025. https://aclanthology.org/2025.nllp-1.23.pdf

[7] Yang et al., "JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment". https://arxiv.org/abs/2605.25240

[8] Study of LLMs on the Polish National Appeal Chamber qualification exam. https://arxiv.org/abs/2511.04205

[9] "Exploiting LLM-as-a-Judge Disposition on Free Text Legal QA via Prompt Optimization". https://arxiv.org/abs/2604.20726

[10] "LegalΔ: Enhancing Legal Reasoning in LLMs via Reinforcement Learning with Chain-of-Thought Guided Information Gain". https://arxiv.org/abs/2508.12281

[11] "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains". https://arxiv.org/abs/2507.17746

[12] HealthBench grader, `healthbench_eval.py`, at commit `652c89d0ca9df547706735883097e9537d40dc47`; paper https://arxiv.org/abs/2505.08775 . https://github.com/openai/simple-evals

[13] "Rethinking Rubric Generation for Improving LLM Judge and Reward Modeling for Open-ended Tasks". https://arxiv.org/abs/2602.05125

[14] "Rubric-Grounded Reinforcement Learning: Structured Judge Rewards for Generalizable Reasoning in Language Models". https://arxiv.org/abs/2605.08061

[15] "Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training". https://arxiv.org/abs/2602.01511

[16] Harvey, "Training a Legal Agent With Applied Compute" (2026-06-22). https://www.harvey.ai/blog/training-a-legal-agent-with-applied-compute
