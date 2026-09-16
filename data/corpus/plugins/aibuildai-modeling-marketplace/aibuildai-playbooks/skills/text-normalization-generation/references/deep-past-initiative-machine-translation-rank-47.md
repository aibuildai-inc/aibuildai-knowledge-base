# Multi-phase curriculum training

Competition: deep-past-initiative-machine-translation
Rank: #47
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/multi-phase-curriculum-training

**Team members:** @arnavmishra02
 
Huge thanks to @takamichitoda for the professional base code that served as the foundation for this solution, and to @phucthaiv02 for the PDF-extracted Akkadian translation dataset 🙏
 
**Model used:** `google/byt5-base`
 
Why ByT5? Akkadian transliterations are full of diacritics (š, ṣ, ṭ, ḫ), subscript numbers, and Unicode fractions (⅓, ⅔). Subword tokenizers choke on these. ByT5 operates at the byte level — every character is a token — so it handles all of this natively without vocabulary issues.
 
## Datasets
 
We used **7 data sources** of increasing quality, trained in curriculum order:
 
| Phase | Source | What it is |
|---|---|---|
| 0 | OA Lexicon + eBL Dictionary + Onomasticon | Word-level pairs (form → English definition, name spellings → canonical names) |
| 0.5 | Gemini back-translations | Synthetic sentence pairs from back-translating English into Akkadian |
| 0.9 | HuggingFace `phucthaiv02/akkadian-translation` | Translation pairs extracted from academic PDFs |
| 1 | Gemini pseudo-labels (OARE corpus) | LLM-generated translations of the full OARE transliteration corpus |
| 1.5 | CDLI 10K + Gemini pseudo-labels | 10K CDLI texts translated by Gemini |
| 1.9 | Published OARE translations | Translations from scholarly publications |
| 2 | Competition `train.csv` | Human gold-standard annotations |
 
All phases were augmented with **Joint Dropout (JD)** — perturbed copies of training pairs where tokens are randomly dropped from both source and target. This is huge for cuneiform because tablets are almost always damaged/fragmentary, so the model needs to handle missing text gracefully. Adding JD consistently boosted the LB by **~0.2 per dataset** — small individually, but it compounds across all 7 phases.
 
Anti-leak filtering was applied rigorously: JD pairs matching validation set IDs or translations were removed before training in every phase.
 
## Preprocessing
 
Mostly based on the host's discussion post, plus:
 
- All break/gap markers (large break, N broken lines, `…`, `[x]`, lone `x`) → single `<gap>` token; consecutive gaps collapsed
- Determinatives preserved: `(d)` → `{d}`, `(ki)` → `{ki}`
- `Ḫ`/`ḫ` → `H`/`h`, subscript digits → regular digits
- Decimal fractions → Unicode fractions (`0.3333` → `⅓`)
- `KÙ.B.` → `KÙ.BABBAR` (silver abbreviation fix)
- Target side: `-gold` → `pašallum gold`, `-tax` → `šadduātum tax`, `-textiles` → `kutānum textiles`
- Shekel fraction arithmetic: `1/12 (shekel)` → `15 grains`, `5/12 shekel` → `⅓ shekel 15 grains`, etc.
- Roman numeral months → Arabic: `month XII` → `month 12`
- `PN` → `<gap>`, removed linguistic annotations (`fem.`, `sing.`, `pl.`, `(?)`)
 
A simple sentence aligner splits multi-sentence documents into 1:1 line pairs when source lines and target sentences match in count.
 
## Training
 
**7-phase curriculum** — each phase loads the previous checkpoint and continues training. The idea: start noisy and broad, finish clean and precise.
 
| Phase | Epochs | LR | Key detail |
|---|---|---|---|
| 0 (Dictionary) | 10 | 2e-4 | Teaches word-level mappings |
| 0.5 (Back-trans) | 10 | 1e-4 | Synthetic fluency |
| 0.9 (PDF) | 5 | 1e-4 | Academic style exposure |
| 1 (Gemini) | 5 | 1e-4 | Broad OARE coverage |
| 1.5 (CDLI) | 5 | 1e-4 | Broader cuneiform genres |
| 1.9 (Published) | 5 | 1e-4 | Scholarly quality |
| 2 (Gold) | 9 | 1e-5 | Early stopping (patience=3), best checkpoint by eval_loss |
 
All phases: batch size 16, gradient accumulation 2 (effective 32), label smoothing 0.1, weight decay 0.01, cosine scheduler.
 
Phase 2 uses a **10× lower LR** (1e-5) to prevent catastrophic forgetting of all the pretraining knowledge. Early stopping + `load_best_model_at_end` ensures we pick the best checkpoint.
 
Prefix for all inputs: `"translate Akkadian to English: "`
 
## Validation
 
Train/val split: 90/10 random split (seed=42) on each phase's data independently. Phase 2 evaluates with `eval_loss` for checkpoint selection, plus BLEU, chrF++, and geometric mean `√(BLEU × chrF++)` for reporting.
 
## What Worked
 
- **ByT5 byte-level approach** — no tokenizer headaches with Akkadian special characters
- **Curriculum learning** — noisy-to-clean progression was key; without the pretraining phases, gold-only training underperforms significantly
- **Joint Dropout augmentation** — essential for robustness to fragmentary tablets
- **Domain-specific preprocessing** — fraction normalization, gap standardization, and determinative handling made a real difference
- **Low LR for final fine-tuning** — 1e-5 with early stopping was the sweet spot for not destroying pretrained knowledge
 
## Failed / Not Tried
 
- ByT5-Large or bigger (not investigated in depth)
- Ensemble decoding across phase checkpoints
- Retrieval-augmented generation with the dictionary at inference time
- Using a separate LLM for post-processing / deduplication
