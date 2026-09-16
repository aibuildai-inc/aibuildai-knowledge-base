# 10th Place Solution — Seq2Seq + CPT + Pseudo-Labeling

Competition: deep-past-initiative-machine-translation
Rank: #10
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/10th-place-solution-seq2seq-cpt-pseudo-label

# Overview
Our final solution is an ensemble of three sequence-to-sequence models — ByT5-Large, ByT5-XL, and MADLAD-400-3B-MT — all initialised from continual pretraining (CPT) checkpoints and further improved with pseudo-labeled data on `published_texts.csv`.

The overall pipeline consists of three stages:
1. Continual pretraining (CPT) on domain-specific Akkadian text
2. Supervised fine-tuning (SFT) on sentence-level translation data
3. Pseudo-labeling and retraining using additional unlabeled corpus data

The final submission ensembles predictions from all three retrained models + MBR decoding.

---

# External Data
To supplement the official training data, we incorporated two external resources from Kaggle:
- Old Assyrian grammar references and linguistic materials
- Scanned PDFs of Kültepe tablets, primary archaeological sources for Old Assyrian cuneiform

Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/668485

---

# Data Preprocessing
Scanned tablet PDFs were processed with **GLM-OCR (0.9B)**, with each page handled independently to extract Akkadian/English text pairs.

## Post-OCR Processing
We then used the Gemini, GPT, and Claude APIs for:
- Sentence alignment
- OCR noise removal
- Text normalisation

The same alignment and cleaning pipeline was applied to the `published_texts` corpus before using it for pseudo-labeling.

To ensure a reliable evaluation setup, we reserved one held-out fold exclusively for validation and excluded it from all training stages.

---

# Model Candidates
We experimented with several strong sequence-to-sequence backbones:
- ByT5 variants
- mT5 variants
- MADLAD-400-3B-MT

ByT5 was particularly well-suited to this task: its byte-level tokenisation handles rare scripts and low-frequency transliteration patterns more robustly than standard subword tokenisation. MADLAD-400-3B-MT was included to add diversity to the final ensemble.

---

# Stage 1 — Continual Pretraining (CPT)
Each model was first trained for 3 epochs of continual pretraining on document-level Akkadian text, combining:
- Official training data
- Cleaned external corpus data

This stage adapts the pretrained multilingual models to the Akkadian domain before introducing translation supervision.

---

# Stage 2 — Supervised Fine-tuning (SFT)
Starting from CPT checkpoints, each model was fine-tuned at the sentence level on parallel translation data, again combining:
- Official training data
- Cleaned external data

The CPT → SFT pipeline consistently outperformed direct fine-tuning, likely because CPT first helps the model internalise the structure and distribution of the target language.

---

# Stage 3 — Pseudo-Labeling on Published Texts
We selected the best single model based on cross-validation and leaderboard performance, and used it to generate pseudo-labels for the `published_texts` corpus at the sentence level.

Three final models were retrained on the combined dataset — official training data, cleaned external Assyrian data, and pseudo-labeled published texts:

| Model | Public | Private |
|---|---|---|
| ByT5-Large | 37.4 | 39.3 |
| ByT5-XL | 37.6 | 39.2 |
| MADLAD-400-3B-MT | 37.3 | 38.9 |
| **Ensemble** | **38.5** | **39.9** |
