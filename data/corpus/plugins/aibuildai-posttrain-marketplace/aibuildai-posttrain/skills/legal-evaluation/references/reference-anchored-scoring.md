# Reference-anchored scoring: open answers compared with a model answer

Read in step 1 when the legal target is an open answer (an exam answer, a judgment section, a summary, a research answer) and the grader holds a reference answer to compare against.

There are three ways to do the comparison: lexical overlap (ROUGE, BLEU, METEOR), embedding similarity (BERTScore, BARTScore), or an LLM judge that reads the reference. In law, the first two track expert judgement poorly, and the third inherits the reference's framing.

## How each one compares

| Benchmark | What is generated | Comparison | Read in |
|---|---|---|---|
| LEXam open questions (2,841, Swiss law exams, English and German) | an exam answer | LLM judge sees the question, the answer and a professor-written reference; returns `[[0.0]]`-`[[1.0]]` in steps of 0.1. The paper uses the minimum of three judges (GPT-4o, Qwen3-32B, DeepSeek-V3); the public scripts call one judge per run | `customized_judge_async.py`; paper [1][2] |
| CaseGen, Chinese civil cases (500 cases, 4 stages) | defence, trial facts, reasoning, judgment | `gpt-4o-2024-11-20` scores 1-10 per dimension against a reference that the prompts place at 8 (10 for the judgment stage); BLEU, ROUGE-L and BERTScore also reported | `eval/llm_eval.py`; paper [3] |
| KoBLEX, Korean statute questions (226, multi-hop) | an answer grounded in provisions | token F1, plus a GPT-4o judge on 1-10 against the reference provisions and answer | paper [4] |
| LexEval, Chinese (23 tasks, 14,150 questions) | generation tasks among multiple choice | ROUGE-L, BERTScore, BARTScore; no judge | paper, README [5] |
| CLERC, U.S. federal case law | the analysis paragraph of an opinion | ROUGE and BARTScore, plus citation recall, precision and false-positive rate | paper [6] |
| LegalAgentBench, Chinese (300 tasks, 37 tools) | a final answer after tool use | fraction of gold key strings found verbatim in the answer; a process rate over intermediate keys in the trajectory; no judge | `src/evaluation/eval.py` [7] |
| LawBench generation tasks (1-1, 2-7, 3-2, 3-8) | legal text | ROUGE-L on segmented Chinese; the README itself says ROUGE-L is not a good metric for long output | [8] |

## What the evidence says about the comparison

- **Lexical and embedding metrics disagree with lawyers.** On Supreme Court syllabi, CaseSumm found the model that won most automatic metrics also hallucinated precedent and facts, and law-student raters preferred another model; ROUGE correlated negatively with their specificity ratings [9]. On Indian judgments, ROUGE-L correlated 0.13 and BERTScore 0.07 with expert overall scores [10]. CLERC's highest-ROUGE model hallucinated the most citations [6].
- **A judge with a reference does better, with a cost.** CaseGen reports Spearman 0.750 between its judge and human ratings, above ROUGE-L and BERTScore [3]; LEXam's judge ensemble passed an alternative-annotator test against each of three legal experts on 50 samples, whose agreement with one another was Pearson 0.70 [1]. Both judges score closeness to the reference, so a correct answer argued another way loses points, and LEXam's prompt tells the judge to penalise content that is not in the reference unless it is certainly correct [2].
- **Keyword keys reward copying.** LegalAgentBench counts exact substrings, so pasting retrieved text raises the score and a correct paraphrase lowers it [7].

## What to do in training

- When the grader is lexical, train on reference-style text for the number, and never treat that number as legal quality; add a judge or lawyer check before choosing a checkpoint.
- When the grader is a judge with a reference, answer the question the reference answers, in its order and at its level of detail. Extra material earns nothing under LEXam, and costs points unless the judge is certain it is correct and relevant.
- Pair any overlap metric with a citation check (`citations-and-grounding.md`): fluent text with invented authority scores well on overlap.

## Sources

Every source was read on 2026-09-29.

[1] Fan et al., "LEXam: Benchmarking Legal Reasoning on 340 Law Exams". https://arxiv.org/abs/2505.12864

[2] LEXam repository at commit `64f2e548a4a84b6154be3498dd5fed8966990bca`, `customized_judge_async.py` and README. https://github.com/LEXam-Benchmark/LEXam

[3] CaseGen paper and repository (`eval/llm_eval.py`, `eval/template/`). https://arxiv.org/abs/2502.17943 ; https://github.com/CSHaitao/CaseGen

[4] KoBLEX paper. https://arxiv.org/abs/2509.01324

[5] LexEval paper and README. https://arxiv.org/abs/2409.20288 ; https://github.com/CSHaitao/LexEval

[6] CLERC paper. https://arxiv.org/abs/2406.17186

[7] LegalAgentBench repository at commit `ec14d8ab1fc439bfd99cbefba08d8e27dafca534`, `src/evaluation/eval.py`; paper https://arxiv.org/abs/2412.17259 . https://github.com/CSHaitao/LegalAgentBench

[8] LawBench repository at commit `e30981bb3ff54c41571f222e0b23e92d27375388`, README. https://github.com/open-compass/LawBench

[9] CaseSumm paper. https://arxiv.org/abs/2501.00097

[10] Shukla et al., "Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation". https://arxiv.org/abs/2210.07544
