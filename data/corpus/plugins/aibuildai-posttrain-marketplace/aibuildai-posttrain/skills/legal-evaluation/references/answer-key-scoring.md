# Answer-key scoring: labels, letters and short answers

Read in step 1 when the legal target has one correct answer per item: a label, a multiple-choice letter, a yes/no, a number.

Most older legal benchmarks are graded this way: LegalBench, LexGLUE, CaseHOLD, the multiple-choice half of LawBench and LEXam, CaselawQA, Bar Exam QA, HousingQA, LEXTREME and COLIEE's entailment task. The model's reasoning earns nothing. What earns the point is a final string that the extraction rule finds and the metric counts as right, so the extraction rule is the first thing to read.

## How each one extracts and counts

| Benchmark | Extraction from a generation | Metric | Read in |
|---|---|---|---|
| LegalBench, most of 162 tasks | none: the whole generation is normalised (punctuation and outer whitespace stripped, lowercased) and must equal the label | balanced accuracy (`sklearn.balanced_accuracy_score`) | `evaluation.py`, `evaluate_exact_match_balanced_accuracy` [1] |
| LegalBench `sara_numeric` | first run of digits after `,` and `.` are removed | correct within 10% of the gold number | `evaluation.py` [1] |
| LegalBench `successor_liability`, `ssla_*` | substring match of each gold item | micro-F1 | `evaluation.py` [1] |
| LegalBench `citation_prediction_open` | gold case name is a substring of the output; reporter and court are not checked | accuracy | `evaluation.py` [1] |
| LegalBench `rule_qa` and the rule-application tasks | none | graded by hand; `evaluate` raises for `rule_qa` | `evaluation.py`, paper [1][2] |
| LexGLUE, seven tasks | none: built for encoder classifiers; a generative model needs your own label parser | micro-F1 and macro-F1; the repository aggregates across tasks with arithmetic, harmonic and geometric means | `experiments/*.py`, `statistics/` [3] |
| CaseHOLD | argmax over five holdings | the original paper reports F1; LexGLUE reports micro- and macro-F1, where micro-F1 equals accuracy for one label out of five | [3][4] |
| LawBench multiple choice (1-2, 2-8, 3-6 with letters; 2-2, 2-4 with category names as options) | counts which options appear anywhere in the output; correct only if the gold option is the only one present; none present is an abstention | accuracy, plus abstention rate reported separately | `evaluation/utils/function_utils.py`, `multi_choice_judge` [5] |
| LEXam multiple choice | last `###X###` in the output | accuracy with bootstrap variance | `evaluation.py`, `extract_letter` [6] |
| CaselawQA (Lawma) | log-probability of each label, or a generated "The final answer is ..." | example-weighted accuracy; each task's majority class is capped at 50%, so a majority-class guess averages about 40% | paper, README [7] |
| Bar Exam QA, HousingQA | answer-letter likelihood or open generation, per model | accuracy; retrieval scored separately with Recall@k and MRR@10 | paper [8] |
| LEXTREME | classification heads | macro-F1, then harmonic means over languages, configs and datasets | paper [9] |

## What it rewards, and the traps

- **The bare label, not the argument.** LegalBench compares the entire normalised output with the label, so "Yes, because ..." is a wrong answer, not a right one with extra words [1]. LawBench marks a correct letter wrong when the reasoning mentions any other option letter, and an English "Answer: B" contains the letter A [5]. Train the exact final form the extractor reads, and score with and without chain of thought before deciding reasoning helps.
- **Class balance is handled differently in each.** Balanced accuracy (LegalBench), a majority cap (CaselawQA), permuted options (LEXam), macro-F1 (LexGLUE, LEXTREME). A number from one is not comparable with another, and a model that predicts the majority class can look strong under plain accuracy.
- **Number parsers are brittle.** `sara_numeric` removes the decimal point, so 1,234.56 is read as 123456, and it takes the first number, so reasoning text before the answer breaks it [1]. LawBench's damages task counts an answer correct if the gold amount is among any numbers in the output, which a list of numbers games [5].
- **Versions move.** LegalBench rebuilt its OPP-115 tasks and re-extracted MAUD passages on 2026-03-29 and fixed `unfair_tos` labels in 2024 [1]; LEXam corrected its multiple-choice statements in 2025-11 [6]. Pin the commit.
- **The training source is often the benchmark's source.** `legal-dataset` holds the hold-out table for these families; check it before training.

## Sources

Every source was read on 2026-09-29.

[1] LegalBench repository at commit `b46bf4ffae90524b2b72aaa30e7745fe9db64481`: `evaluation.py`, `CHANGELOG.md`. https://github.com/HazyResearch/legalbench

[2] Guha et al., "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models". https://arxiv.org/abs/2308.11462

[3] LexGLUE repository at commit `419a49d0fe82ecb69eb8e9343a9eb3382f1040fd` and dataset card. https://github.com/coastalcph/lex-glue ; https://huggingface.co/datasets/coastalcph/lex_glue

[4] Zheng et al., "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset of 53,000+ Legal Holdings". https://arxiv.org/abs/2104.08671

[5] LawBench repository at commit `e30981bb3ff54c41571f222e0b23e92d27375388`: `evaluation/utils/function_utils.py`, `evaluation/evaluation_functions/`. https://github.com/open-compass/LawBench

[6] LEXam repository at commit `64f2e548a4a84b6154be3498dd5fed8966990bca`: `evaluation.py`, README. https://github.com/LEXam-Benchmark/LEXam

[7] Dominguez-Olmedo et al., "Lawma: The Power of Specialization for Legal Annotation". https://arxiv.org/abs/2407.16615 ; https://github.com/socialfoundations/lawma

[8] Stanford RegLab paper introducing Bar Exam QA and Housing Statute QA. https://arxiv.org/abs/2505.03970

[9] Niklaus et al., "LEXTREME: A Multi-Lingual and Multi-Task Benchmark for the Legal Domain". https://arxiv.org/abs/2301.13126
