# [3rd] Synthetic Data to Teach OA Fundamentals

Competition: deep-past-initiative-machine-translation
Rank: #3
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/3rd-synthetic-data-to-teach-oa-fundamentals

Thank you Kaggle for hosting this interesting competition. Special shout-out to @deeppast for active engagement throughout. It was great to see performance improving right up to the final hour! At a broader level, we think some of the techniques here would generalize to teaching an LLM a new skill (e.g. a new programming language) or encoding new knowledge (e.g. post cutoff recent events).

# TL;DR

We ensembled a `ByT5-Large` and a `ByT5-XL` model, using a fine-tuned `Qwen3-8B` pairwise Reward Model to select the better translation per sentence. Each model was trained in two stages: 
- Continued seq2seq pre-training (CPT)
- Fine-tuning (FT)

CPT stage teaches Old Assyrian (OA) language fundamentals (Vocab, Grammar, Morphology etc.) using synthetic data, while having a light exposure to the actual translation task. FT stage was aimed at teaching the translation style of OA scholars using only high-quality scholar-translated OA (transliteration, translation) pairs.



## Agentic Tooling

We used Claude Code extensively throughout this competition — for data extraction, quality auditing, training pipeline development, and deep research. A separate post on key learnings from using agentic tools (Claude Code skills, MCP integrations, parallel subagents) will follow.

# Data Sources

This competition was heavily data-driven.  We spent the majority of our time preparing high-quality training data: extracting parallel pairs from publication PDFs, OCR texts, and generating synthetic drills to teach OA fundamentals.

## Synthetic Data
We used `Claude 4.5 Sonnet` to generate diverse synthetic tasks to teach OA fundamentals to ByT5 models.

### Vocab Drills

We extracted ~1k OA lemmas (headword) from the CAD and eSAD dictionaries: [Old Assyrian Grammars and other resources](https://www.kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources). Each lemma had one or more attested senses (glosses) with example (transliteration, translation) pairs. For each (lemma, gloss) pair (~3k), we asked an LLM to generate a list of 32 synthetic sentence pairs, resulting in ~90k examples.

For polysemous lemmas, we sampled and concatenated up to 4 examples from different senses of the same lemma into a single training pair. This forces the model to learn sense disambiguation from context. This gave us ~60k additional training pairs by repurposing the earlier generated examples.

### Grammar Drills

We extracted ~500 grammar rules from the [Kouwenberg 2017 GOA book](https://www.kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources?select=Kouwenberg_2017_GOA_HdO118.pdf). Rules had the following structure:

```
  {
    "title": "G-stem as passive of factitive D-stem (anti-factitive)",
    "rule": "In certain intransitive verbs with a frequently used factitive D-stem...",
    "transformation": "For seeds containing D-stem forms of abātum, kabātum, lamādum, nasāḫum, aqāpum, or nasārum: change D → G for passive, or G → D to add an agent. D-stem prt pattern: ú-PaRRiS (e.g., ú-bu-ú, ú-ka-bi-it, ú-na-si-iḫ). G-stem stative: PaRiS (e.g., a-bi, ka-bi-it, na-si-iḫ).",
    "examples": [
      {
        "transliteration": "ki-ma ú-bu-ú",
        "translation": "As he cleared (the debt).",
        "grammar": "D prt 3ms of abātum"
      },
      {
        "transliteration": "ḫu-bu-la-tum ib-bi-a",
        "translation": "The debts have been cleared.",
        "grammar": "G stv 3fp of abātum (anti-factitive)"
      },
...
    ]
  },
```
We then use LLM to apply the rules on seed sentences (sampled from the competition training tablets) to transform them by making minimal changes e.g. swap tense, person, mood, or clause structure. We generated ~50k examples following this approach.

### Slot-Fill Templates 

Extracted 600+ templates with typed slots (PN, GN, amount, commodity, deadline, month, occupation etc). Example:
```
{
    "transliteration": "{AMT_TL} KÙ.BABBAR ṣa-ru-pá-am i-ṣé-er {PN1_TL} {PN2_TL} i-šu",
    "translation": "{PN1_TR} owes {AMT_TR} of refined silver to {PN2_TR}."
}
```
and filled them programmatically from curated candidates for the typed slots (e.g. names sampled from host provided [onomasticon.csv](https://www.kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources?select=onomasticon.csv)). We created ~150K pairs this way.

### PN vs Word Contrastive

Created a small seed dataset with around 60 transliteration forms which are ambiguous i.e. they can be a personal name or a common word depending on the context. This seed dataset included few shot pairs from real tablets showing the same form used as a NAME in one sentence and as a WORD in another. Then we used LLM to generate more such examples for the same forms.

### Other Synthetic data
- Weight Conversion Drills
- Synthetic Tablets: coherent multi-sentence tablets generated via augmenting real tables e.g via clause extension, name substitution, interpolation across real tablets.

## Real Tablets

### Competition Data
The competition's `train.csv` (1,561 tablets) and `Sentences_Oare` CSV were combined and deduplicated by `oare_id` to produce ~2,700 tablet-level `(transliteration, translation)` pairs. These are full-tablet translations, so we used Gemini 3 Pro with CoT to split each tablet into sentence-level pairs, aligning OA segments to their corresponding English translation fragments. A LLM judge then scored each pair on translation quality (high/medium/low) evaluating faithfulness to the OA. This produced ~17k sentence pairs with 85% high, 7% medium, 8% low quality distribution.

### PDF Extraction Pipeline
We independently extracted parallel pairs from the [Kültepe Tablets PDFs](https://www.kaggle.com/datasets/deeppast/old-assyrian-kltepe-tablets-in-pdf) using the multimodal capabilities of `Gemini-3.0-Pro`. PDF pages were rendered at 576 DPI, critical for correctly recognizing Akkadian diacritics (`š, ṣ, ṭ, ḫ`) and subscript digits (`₄, ₆`) and sent as images in 10-page overlapping chunks. A [demo notebook](https://www.kaggle.com/code/darraghdog/k-ltepe-tablets-with-gemini-flash) shows the full pipeline. The prompt went through 24+ iterations. Key design choices:
  - Visual layout: Identifies transliteration blocks by visual structure (margin line counts, Vs./Rs. markers, indentation) rather than text parsing.
  - Marker alignment: Uses superscript line ranges in the translation (e.g., <sup>1-2)</sup>, <sup>3-6)</sup>) as anchors for semantic alignment.
  - Trust the scholar: Not to alter the published translation, entities (names, commodities). This prevents hallucinating names not in the cuneiform.
  - Complete extraction: Explicit "include ALL lines" instruction. Without it, the gemini silently skipped damaged tablets, losing ~30% of pairs.
  - In-line translation: For German/Turkish sources, a domain vocabulary supplement enables English translation in the same API call.

We extracted parallel (transliteration, translation) pairs from the PDFs shared in [Old Assyrian Kültepe Tablets in PDF](https://www.kaggle.com/datasets/deeppast/old-assyrian-kltepe-tablets-in-pdf). 3.3k tablets producing around 36k sentence pairs.

### OCR Texts
Publications not available as PDFs were extracted from the OCR data in the competition's [publications.csv](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/data?select=publications.csv). It provided ~20k sentence pairs with scholar translation.

### Synthetic Translations
The competition's `published_texts.csv` contains ~8k tablets with transliterations. We used Gemini to generate translations and split them into sentence-level pairs. The prompt inlined the host's dataset Instructions covering name normalization, Sumerogram conventions, gap marker parallelism (`<gap>` in transliteration → `<gap>` in translation at the same semantic positions), standard letter/debt/witness formulas, and unit handling. 

We provided 80 dynamically sampled expert few-shot examples to anchor the translation style. In a second round, we replaced random sampling with embedding-based retrieval (`NVIDIA NV-Embed`) to find the 8 most semantically similar expert pairs per sentence. This improved self-reported high-quality rate from 61% to 82%. Produced ~35k sentence pairs across after deduplication against the `Competition Data`.

### Additional Sources
We used OA publication PDFs from Hecker's HPM catalog (9.9k transliteration-only segments), Michel legal case publications (HAL/CNRS open access), Dergipark Turkish academic journals (ArAn, Belleten, DTCF — Turkey's national open-access platform), and assorted open-access dissertations. These PDFs were extracted using the same Gemini multimodal recipe as the Kültepe tablets. All external sources are either open-access by institutional mandate (Dergipark/TÜBİTAK, HAL/CNRS), competition-provided, or public domain (CAD/Oriental Institute).

### Pseudo Labelling
We used our fine-tuned ByT5 models to generate translations for unlabelled transliterations in the `published_texts.csv` and `publications.csv`.

# Datamix
For **CPT**, we used all of the synthetic drill examples. In addition, we used the examples from the competition dataset, PDF extraction pipeline, extracted pairs from OCR texts and synthetic translations. We upsampled high quality subsets (e.g. competition Data,  PDF Extraction data) by 2x.

For **FT**, we used 
- High-quality scholar created translations: competition Data, PDF Extraction Pipeline, pairs from the OCR Texts (`publications.csv`)
- Pseudo-labeled translations on Unlabeled Transliterations
- A small amount of carry over synthetic drills from CPT (5k examples)

# Normalization
Both transliteration and translation required deterministic normalization to match the test format. The normalization details can be found in our [inference notebook](https://www.kaggle.com/code/conjuring92/dpc-a08-inference-rm?scriptVersionId=305681862).

# Ablations - ByT5 Large

| Config | What's Added | Public | Private | 
|--------|-------------|--------|---------|
| Baseline | No CPT, direct train on only scholar data | 39.9 | 40.3 |
| CPT only | CPT checkpoint, no FT stage| 39.6 | 40.4 |
| B0 | CPT -> FT on scholar data | 40.6 | 41.0 |
| B1 | B0 + carry over synthetic drills (5k sentences) + upsampling competition tablets | 40.8 | 41.4 |
| B2 | B1 + pseudo-labeled data (27.7K sentences) | 41.2 | 41.4 |
| B2_ns | B2 + name swap augmentation | 41.3 | 41.2 |
| M2 | B2_ns + More PL data + more pretrain drill carry | 41.0 | 41.3 |

# Training
## Stage 1: Continued Pre-training (CPT)
- Effective batch size 128, 3 epochs on the CPT datamix (18K gradient descent steps)
- Long warmup (3.6k steps) followed by constant LR: this gives many viable checkpoints to choose from for Stage 2 FT
- Gradient clipping (max norm 0.3)
- First ~1k steps of training (when ByT5 encoder got exposed to OA translations) were aggressive with very high gradient norms. Long warmup + gradient clipping helped with numerical stability.
- AWP (Adversarial Weight Perturbation) to mitigate overfitting risk
- Checkpoint selection: We decided to use the checkpoint after 14k steps for stage 2 FT as per LLM qualitative analysis, which found:
    - **Steps 0 to 6K (rapid learning)**: The model learns to decode common Sumerograms, master formulaic translations, and render high frequency PN/GNs correctly. Almost everything gets better.
    - **Steps 6K–14K (refinement, diminishing returns):** Specialized commercial and legal vocabulary solidifies. Rarer Sumerograms and kinship terms gradually converge toward correct readings. Additional steps improved fewer sentences with rare names start regressing toward frequent neighbors.
    - **Steps 14K–18K (regression onset):** Nearly as many translations degrade as improve. Name frequency bias intensifies, vocabulary that was correct for 14k steps suddenly regresses and hallucinated names appear.

## Stage 2: Fine-tuning (FT)
Starting from the CPT checkpoint at step 14K, we fine-tuned on the high-quality FT datamix described above. For this stage, we used
- Cosine Decay LR schedule with short warmup
- Name swap augmentation
- AWP
- EMA
- Label Smoothing

# Inference
We used beam search with 8 beams. Our inference notebook is available [here](https://www.kaggle.com/code/conjuring92/dpc-a08-inference-rm?scriptVersionId=305681862).
