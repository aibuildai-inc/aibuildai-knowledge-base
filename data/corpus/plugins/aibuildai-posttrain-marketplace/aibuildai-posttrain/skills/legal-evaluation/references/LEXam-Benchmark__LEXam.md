# LEXam

LEXam: 7,537 questions from 340 Swiss law-school exams in 116 courses, in English and German. 4,696 multiple-choice questions scored by answer key, and 2,841 open questions scored by an LLM judge against a professor-written reference answer [1][2]. The one widely used legal benchmark that grades both short answers and long reasoning.

**Grades**: answer key (multiple choice) and reference-anchored judge (open questions).

**Score**: multiple choice: accuracy on the last `###X###` in the output. Open: the judge returns 0.0-1.0 in steps of 0.1, reported ×100 [3].

**Judge**: the paper grades with the minimum of GPT-4o, Qwen3-32B and DeepSeek-V3. It validated the ensemble against three legal experts (Swiss law graduates pursuing or holding a doctorate) on 50 samples with an alternative-annotator test, which the judge passed against each expert; the experts agreed with each other at Pearson 0.70. The public scripts call one judge per run (`--llm gpt-4o`) [1][3].

**Access and licence**: open. Data CC BY 4.0, code Apache-2.0. HF `LEXam-Benchmark/LEXam`: open questions `test` 2,541 and `dev` 300; multiple choice with 4, 8, 16 and 32 options: 1,655, 1,463, 1,028 and 550 [2][3].

**Use it for**: exam-style legal reasoning in a civil-law system, with a hard multiple-choice ladder (up to 32 options) that punishes guessing.

**Trap**: the judge prompt tells the judge to penalise content not in the reference unless it is certainly correct and relevant, and to require precise statute citations (Abs., Ziff., lit.) [3]. Extra material the judge cannot confirm costs points; answer the reference's sub-points at its level of detail.

## How it is graded

- Multiple choice: `evaluation.py::extract_letter` takes the last `###([A-Z])###`; no match is wrong. Accuracy with a 1,000-sample bootstrap variance [3].
- Open questions: the judge sees the question, the answer and the reference. If the reference lists sub-points, credit is proportional to sub-points covered. The score is the first `[[x.x]]` in the judge's output; no match scores 0 [3].
- Scores before the 2025-11 correction of multiple-choice statements are not comparable with later ones [3].

## Where it sits

Swiss federal and cantonal law; a correct answer under another jurisdiction's law is wrong here. The open-question judge is one of the few in legal evaluation with published agreement numbers; see `llm-judges.md`.

## Sources

Every source was read on 2026-09-29.

[1] Fan et al., "LEXam: Benchmarking Legal Reasoning on 340 Law Exams". https://arxiv.org/abs/2505.12864

[2] Hugging Face Hub API record. https://huggingface.co/api/datasets/LEXam-Benchmark/LEXam

[3] LEXam repository at commit `64f2e548a4a84b6154be3498dd5fed8966990bca`: README, `evaluation.py`, `customized_judge_async.py`, `LICENSE_DATA`. https://github.com/LEXam-Benchmark/LEXam
