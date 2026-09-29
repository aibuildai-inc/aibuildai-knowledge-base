# Legal evaluations by grading family

Every legal evaluation this skill describes, grouped by how it is graded. **Card** says where the details are: a file in this skill's `references/`, or a card in the `legal-dataset` skill, which holds the evaluation sets that are also Hugging Face datasets (with load lines, rows and hold-out notes). Checked 2026-09-29.

Families: **key** = answer key; **set** = extraction and set scoring; **ref** = reference-anchored (lexical metric or judge with a reference); **rubric** = criterion-level rubric; **cite** = citation and grounding; **pair** = pairwise or lawyer grading.

## Short answers and classification

| Evaluation | Jurisdiction, language | Family | Metric | Judge | Card |
|---|---|---|---|---|---|
| LegalBench, 162 tasks | U.S., English | key | balanced accuracy on exact match; F1 or tolerance for a few tasks; `rule_qa` by hand | none | `legal-dataset`: `nguha__legalbench.md`; `answer-key-scoring.md` |
| LexGLUE, 7 tasks | ECHR, U.S., EU | key | micro- and macro-F1 | none | `legal-dataset`: `coastalcph__lex_glue.md` |
| CaseHOLD | U.S. | key | F1 (paper); micro- and macro-F1 (LexGLUE) | none | `legal-dataset`: `casehold__casehold.md` |
| CaselawQA | U.S. | key | example-weighted accuracy, majority class capped at 50% | none | `legal-dataset`: `ricdomolm__caselawqa-8k.md` |
| Bar Exam QA, HousingQA | U.S. | key, retrieval | accuracy; Recall@k, MRR@10 | none | `legal-dataset`: `reglab__barexam_qa.md`, `reglab__housing_qa.md` |
| AdaptLLM law tasks | U.S., EU | key | per task | none | `legal-dataset`: `AdaptLLM__law-tasks.md` |
| AGIEval JEC-QA-KD | China | key | accuracy | none | `legal-dataset`: `hails__agieval-jec-qa-kd.md` |
| LawBench, 20 tasks | China | key, set, ref | accuracy, set F1, ROUGE-L, log distance; abstention rate | none | `legal-dataset`: `doolayer__LawBench.md`; `answer-key-scoring.md` |
| LexEval, 23 tasks | China | key, ref | accuracy or F1; ROUGE-L, BERTScore, BARTScore | none | `CSHaitao__LexEval.md` |
| LEXam, 7,537 questions | Switzerland, English and German | key, ref | accuracy; judge 0-1 against a reference | GPT-4o, Qwen3-32B, DeepSeek-V3 (minimum, paper) | `LEXam-Benchmark__LEXam.md` |
| LEXTREME | 24+ languages | key | macro-F1, harmonic means | none | `answer-key-scoring.md` |

## Extraction and retrieval

| Evaluation | Jurisdiction, language | Family | Metric | Judge | Card |
|---|---|---|---|---|---|
| CUAD | U.S. contracts | set | AUPR, precision at 80% and 90% recall, word Jaccard ≥ 0.5 | none | `legal-dataset`: `theatticusproject__cuad-qa.md`; `extraction-and-set-scoring.md` |
| MAUD | U.S. merger agreements | set | minority-class AUPR per answer class | none | `legal-dataset`: `theatticusproject__maud.md`; `extraction-and-set-scoring.md` |
| JuDGE | China, criminal | set, ref | charge and article F1, term and fine distance, METEOR, BERTScore | none | `oneal2000__JuDGE.md` |
| LegalBench-RAG | U.S. contracts, policies | set | character-level precision and recall | none | `zeroentropy-ai__legalbenchrag.md` |
| Legal RAG Bench | Australia (Victoria), criminal | ref, retrieval | answers and relevant passages for 100 questions; see its card | per card | `legal-dataset`: `isaacus__legal-rag-bench.md` |

## Open answers against a reference

| Evaluation | Jurisdiction, language | Family | Metric | Judge | Card |
|---|---|---|---|---|---|
| CaseGen | China, civil | ref | 1-10 per dimension against the real text; BLEU, ROUGE-L, BERTScore | `gpt-4o-2024-11-20` | `CSHaitao__CaseGen.md` |
| KoBLEX | Korea, statutes | ref | token F1; 1-10 judge against reference provisions | GPT-4o | `daehuikim__KoBLEX.md` |
| CLERC | U.S. case law | ref, cite | ROUGE, BARTScore; citation recall, precision, false positives | none | `legal-dataset`: `jhu-clsp__CLERC.md`; `citations-and-grounding.md` |
| Legal summarisation (BillSum, EUR-Lex-Sum, Multi-LexSum) | U.S., EU | ref | ROUGE | none | `legal-dataset`: `lighteval__legal_summarization.md` |
| LegalAgentBench | China | ref (keyword keys) | fraction of gold keys in the answer and trajectory | none | `CSHaitao__LegalAgentBench.md` |

## Work product graded by rubric

| Evaluation | Jurisdiction, language | Family | Task score | Judge | Card |
|---|---|---|---|---|---|
| Harvey LAB | mostly U.S., English, agentic | rubric | all-pass, averaged over two judges | `claude-sonnet-4-6`, `gpt-5.5` | `legal-dataset`: `harveyai__harvey-labs.md`; `rubric-scoring.md` |
| RedlineBench | U.S. commercial contracts, agentic | rubric | weighted with penalties, clamped to [0, 1] | three-judge majority | `legal-dataset`: `crosbylegal__RedlineBench.md`; `rubric-scoring.md` |
| BigLaw Bench | U.S., large-firm tasks | rubric, cite | (positive − negative) / positive; source score | not stated | `harveyai__biglaw-bench.md` |
| PRBench, legal half | many countries and U.S. jurisdictions | rubric | weighted, normalised by positive weights, floored at 0 | `o4-mini` | `ScaleAI__PRBench.md` |
| PLawBench | China | rubric | earned / maximum, with penalties | Gemini-3.0-Pro-Preview | `skylenage__PLawbench.md` |
| LexRubric | China | rubric | weighted, normalised by positive weights | Qwen3.6-27B | `chenyifan0929__LexRubric.md` |
| Magis-Bench | Brazil, Portuguese | rubric | 0-10 on official exam rubrics | four frontier judges (paper); one default judge (repository) | `maritaca-ai__magis-bench.md` |
| Vals Legal AI Report | U.S. | rubric, pair | per-element pass rate; lawyer-graded research rubric | LLM, then lawyers | `vals-ai__legal-ai-report.md` |

## Citations, grounding and grading methods

| Evaluation | Jurisdiction, language | Family | Metric | Judge | Card |
|---|---|---|---|---|---|
| Legal RAG hallucination study (RegLab) | U.S. | cite | correct, incorrect or refusal × grounded, ungrounded or misgrounded | lawyers | `legal-dataset`: `reglab__legal_rag_hallucinations.md`; `citations-and-grounding.md` |
| Large Legal Fictions (RegLab) | U.S. federal | cite | hallucination rate against case metadata | none (reference-based) | `citations-and-grounding.md` |
| LegalCiteBench | U.S. | cite | citation overlap; misleading answer rate | none | `legalcitebench__LegalCiteBench.md` |
| JudgmentBench | U.S., large-firm tasks | pair | recovery of known quality orderings, rubric versus pairwise | attorneys, LLM graders | `judgmentbench__JudgmentBench.md` |

## Method cards

| File | Read it when |
|---|---|
| `answer-key-scoring.md` | the answer is a label, letter or number |
| `extraction-and-set-scoring.md` | the answer is a set of spans, clauses, articles or passages |
| `reference-anchored-scoring.md` | an open answer is compared with a model answer |
| `rubric-scoring.md` | work product is graded criterion by criterion |
| `citations-and-grounding.md` | the answer cites authority |
| `llm-judges.md` | any score comes from an LLM judge, or a judge will be the reward |
| `building-a-legal-eval.md` | no public evaluation fits, or checkpoints must be compared |
