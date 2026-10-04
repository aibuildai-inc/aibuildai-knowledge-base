# Model-written legal tasks: what the published pipelines do, and what they filter

Read when the record does not hold the answer and a model has to write the question, the answer, or both. It lists the generation recipes that have been published with results, and the controls that separated the ones that helped from the ones that hurt.

A model can turn any legal text into question-answer pairs at almost no cost. The cost appears later: answers that the text does not support, questions that only restate a heading, thousands of near-identical rows. The pipelines below differ mainly in how they prevent that.

## Published recipes, legal

| Pipeline | Source text | Generation | Filter | Reported result |
|---|---|---|---|---|
| Open Australian Legal QA | 2,124 documents sampled from the Open Australian Legal Corpus | one chunk of up to 384 tokens per document; `gpt-4` at temperature 0 writes a decontextualised question and an answer "extracted from the snippet" that names the source | none stated beyond the prompt | 2,124 pairs, each keeping its source chunk (`legal-dataset` card) |
| LegalKit | 53,000 articles of French codes | three LLaMA-3-70B instances each write a query for the article; a fourth picks the best | selection by the fourth model | one query per article, with validity dates (`legal-dataset` card) |
| German statute QA, difficulty-graded [1] | provisions of the BGB, StGB, SGB and Basic Law, each stored with law and section identifiers | GPT-4 writes four levels: clause comprehension, a client-style paraphrase without statute numbers, a short scenario applying the provision, and a scenario spanning several provisions | a reviewer prompt that sees only the statute text and checks answerability, support and redundancy; it can "only discard candidates, not generate new question types" | 31,777 candidates, 23,905 kept; Gemma 3 12B rose from 36.8% to 76.4% on the authors' BGB questions |
| German statute QA, single prompt [1] | the same provisions | one prompt, passage to QA "in one step" | none | 15,734 pairs; mixed results, and LLaMA 3.1 fell from 43.0 to 35.4 on the open-ended, LLM-judged LegalMC4 questions |
| LawGPT knowledge-guided generation [2] | not stated in the abstract | knowledge-guided generation "to enhance diversity", then a refinement step that corrects errors | a verification step that filters low-quality data | 50,000 examples; only the abstract was read |
| MASER [3] | interactive legal scenarios | a multi-agent simulation generates the interaction data | a supervisory mechanism inside the simulation | only the abstract was read |
| SEC contract extraction | 1,028 Exhibit 10 contracts and 150 proxy statements | a 2B model fills a JSON schema from each chunk | none stated; the card calls the labels "silver-standard" | 3,430 positive and 4,253 "corrective" examples that teach an empty answer when the text holds no value (`legal-dataset` card) |

LawInstruct is a different operation: it rewrites 58 existing annotated datasets, from 17 jurisdictions and 24 languages, into 12 million instruction examples [4]. Nothing new is generated, so it inherits every benchmark those datasets feed (`legal-dataset`: `lawinstruct__lawinstruct.md`).

## Published recipes, general-domain, that take raw documents

| Method | What it does | Reported |
|---|---|---|
| Bonito [5] | a model fine-tuned on 1.65 million examples to convert unannotated text into task-specific training data, conditioned on a task type | 22.1 F1 points over the zero-shot baseline, where plain next-word training on the same text lost 0.8 |
| Source2Synth [6] | generation "grounded in real-world data sources", followed by a curation step that discards low-quality items | 22.57% on HotpotQA and 25.51% on WikiSQL over fine-tuned baselines |
| AgentInstruct [7] | three flows: transform the raw document, generate seed instructions from it, then refine them for difficulty; "uses raw documents for seeding" | about 22 million generated pairs; a Mistral-7B fine-tune improved 40% on AGIEval |
| Synthetic continued pretraining (EntiGraph) [8] | extract the salient entities of a small corpus and generate varied text about the relations between them, then continue pretraining on it | proposed because learning a fact from a few documents is data-inefficient; the abstract reports that models learn small domain corpora more effectively this way |

## What separates the pipelines that helped

- **The answer is tied to supplied text.** Every legal recipe above generates from a specific passage and keeps it on the row. A question written from the model's memory of the law has no check.
- **A reviewer that sees only the source.** In the German study the filter read the statute and the candidate, nothing else, and could only delete. It removed about a quarter of candidates, mostly at the easier levels; all 441 multi-provision items survived [1].
- **Graded difficulty.** The same study's single-prompt data lowered one model's score while the graded data improved every model on every test the authors report [1]. One prompt yields one kind of question.
- **Negatives.** The SEC extraction set has more "corrective" rows than positive ones. Without no-answer items a model learns that every question has an answer in the text.
- **Split by source passage.** The German study assigns "all QA pairs generated from a given section" to one split [1].

## What goes wrong

- **Volume without content.** `stindardlogic/legal-reasoning-dpo-100k` is 16 distinct rows repeated 6,250 times each, and `Alignment-Lab-AI/Lawyer-Instruct` is flagged off-topic (`legal-dataset`). Count distinct rows before counting rows.
- **Nobody checked the law.** None of the legal pipelines above reports review of the generated answers by lawyers. A filter that checks support in the passage does not check that the passage was the right law for the question.
- **The generator's terms travel with the data.** Answers written by `gpt-4` or LLaMA-3 carry those models' output terms on top of the source's licence (`legal-dataset` cards for the two datasets).
- **The generator's style becomes the target.** Training on one model's answers teaches its phrasing. `training` card `34-distillation.md` covers this as fixed-teacher distillation.

## What to do

1. Generate from a stored passage, at several difficulty levels, and keep the passage on the row.
2. Run a discard-only reviewer that sees only the passage and the candidate. Report candidates in and items kept.
3. Add no-answer and false-premise items.
4. Check every case citation in a generated answer (`checking-synthesized-tasks.md`).
5. Have a law-trained reader review a sample before training, and say in the dataset's documentation that the rest is unreviewed.
6. Record the generator model, prompt version and date on every row.

## Sources

Every source was read on 2026-10-03.

[1] Bashir et al., "Domain-Adaptation through Synthetic Data: Fine-Tuning Large Language Models for German Law". https://arxiv.org/html/2601.14160

[2] "LawGPT: Knowledge-Guided Data Generation and Its Application to Legal LLM" (abstract). https://arxiv.org/abs/2502.06572

[3] "Multi-Agent Simulator Drives Language Models for Legal Intensive Interaction" (abstract). https://arxiv.org/abs/2502.06882

[4] "LawInstruct" (abstract). https://arxiv.org/abs/2404.02127

[5] Nayak, Nan, Trost and Bach, "Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation". https://arxiv.org/abs/2402.18334

[6] "Source2Synth" (abstract). https://arxiv.org/abs/2409.08239

[7] Mitra et al., "AgentInstruct: Toward Generative Teaching with Agentic Flows". https://arxiv.org/html/2407.03502

[8] Yang, Band, Li, Candès and Hashimoto, "Synthetic continued pretraining". https://arxiv.org/abs/2409.07431
