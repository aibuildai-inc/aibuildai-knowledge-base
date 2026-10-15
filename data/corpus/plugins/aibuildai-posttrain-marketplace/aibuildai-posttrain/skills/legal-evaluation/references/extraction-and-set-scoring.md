# Extraction and set scoring: clauses, spans, articles and passages

Read in step 1 when the legal target is to find things in documents: clauses in a contract, deal points in a merger agreement, the statute articles or charges a judgment rests on, the passages that answer a question.

These scorers compare a set the model returns with a gold set. The matching rule (exact, word overlap, character overlap, substring) and the counting rule (what an extra or an empty answer costs) decide what training should aim at.

## How each one matches and counts

| Benchmark | Matching rule | Metric | Read in |
|---|---|---|---|
| CUAD clause extraction (41 clause types) | Jaccard over word sets after removing `. , ; :`, lowercasing and turning `/` into a space; a match at overlap ≥ 0.5. For the "Parties" question, a gold string contained in the prediction also matches | TP, FP and FN counted over confidence thresholds from 0.99 down to 0; AUPR, and precision at 80% and 90% recall | `evaluate.py` [1] |
| MAUD deal points (92 questions) | multiple choice, not spans | one-versus-rest AUPR per answer class on the minority class, averaged per question, then across questions | `src/maud/pr_curves.py` [2] |
| LawBench set tasks (2-3, 2-9, 3-1, 3-3), article prediction | set F1 over extracted items; articles normalised from Chinese numerals with `cn2an` | set F1 averaged per item | `evaluation/evaluation_functions/` [3] |
| LawBench sentence prediction (3-4, 3-5) | first months figure, else years × 12 | normalised log distance: `(log 216 − mean |log(gt+1) − log(pred+1)|) / log 216`; a missing prediction takes the full penalty | `ljp_imprison.py` [3] |
| JuDGE, Chinese criminal judgments | charges and penal-code articles pulled from the generated judgment by regular expressions | precision, recall and F1 for each; prison term and fine scored `1 − |gold − generated| / max`, zero on a sign or death-penalty mismatch | `evaluation/calc.py` [4] |
| LegalBench-RAG, retrieval over contracts and policies | character overlap between retrieved and gold spans | precision = overlap / retrieved characters, recall = overlap / gold characters | `run_benchmark.py` [5] |

## What it rewards, and the traps

- **Every extra span costs precision, and an answer where none exists is always a false positive.** In CUAD, any prediction on a question with no gold answer counts as a false positive, while an empty prediction is ignored [1]. About 10% of contract text carries a label [6], so a model that learns to say nothing when nothing applies gains most of its precision there. Keep negative examples in training.
- **Word-overlap matching tolerates boundaries, not paraphrase.** A span that shares half its words with the gold clause matches; a correct paraphrase does not. Train on verbatim spans when the grader is CUAD-like.
- **Regular expressions define correctness.** JuDGE and LawBench find charges, articles and terms with patterns written for court style [3][4]. Output in a different format scores zero even when the law is right, so match the court's phrasing.
- **Chunk size moves precision and recall in opposite directions.** Under character-level scoring, returning whole documents gives high recall and low precision [5]. Report both at a fixed k.
- **Sources overlap with training data.** CUAD, MAUD and ContractNLI feed LegalBench and LegalBench-RAG; see the hold-out table in `legal-dataset`.

## Sources

Every source was read on 2026-09-29.

[1] CUAD repository at commit `67faa0e6023b04fcaae6cc09497ab00e5d63a2a2`, `evaluate.py`. https://github.com/TheAtticusProject/cuad

[2] MAUD repository at commit `4640316078dcd370debb877f350c39a28b181ffe`, `src/maud/pr_curves.py`; paper https://arxiv.org/abs/2301.00876 . https://github.com/TheAtticusProject/maud

[3] LawBench repository at commit `e30981bb3ff54c41571f222e0b23e92d27375388`, `evaluation/evaluation_functions/`. https://github.com/open-compass/LawBench

[4] JuDGE repository, `evaluation/calc.py`; paper https://arxiv.org/abs/2503.14258 . https://github.com/oneal2000/JuDGE

[5] LegalBench-RAG repository at commit `431bc8f2488a81569ab7259fa633dcc50ab77f9a`, `run_benchmark.py`; paper https://arxiv.org/abs/2408.10343 . https://github.com/zeroentropy-ai/legalbenchrag

[6] Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review". https://arxiv.org/abs/2103.06268
