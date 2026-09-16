# [21th] ByT5-Base + Gemini Augmentation, No Ensemble

Competition: deep-past-initiative-machine-translation
Rank: #21
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/21th-byt5-base-gemini-augmentation-no-ensembl

# ByT5-Base + Gemini Augmentation, No Ensemble

**Final: Private 38.9 / Public 38.3 (Version 137)**

Solo entry, single GPU on AutoDL, ByT5-base throughout.

---

## 1. Overview

```
  ┌──────────────────┐     ┌───────────────────┐     ┌────────────────┐
  │  train.csv       │     │  published_texts   │     │  Gemini API    │
  │  (1500 labeled)  │     │  (unlabeled)       │     │  augmentation  │
  └────────┬─────────┘     └────────┬──────────┘     └───────┬────────┘
           │                        │                         │
           ▼                        ▼                         ▼
  ┌──────────────────┐     ┌───────────────────┐     ┌────────────────┐
  │  DeepSeek split  │     │  Gemini pseudo-    │     │  Synthetic     │
  │  + clean/align   │     │  label translation │     │  ~7800 samples │
  └────────┬─────────┘     └────────┬──────────┘     └───────┬────────┘
           │                        │                         │
           └────────────────┬───────┘─────────────────────────┘
                            ▼
                   ┌─────────────────────┐
                   │  Combined ~30k rows  │
                   │  weighted sampling   │
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │  ByT5-base training  │
                   │  5 epochs, single GPU│
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Epoch 3-5 avg      │
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Inference          │
                   │  beam=4, post-proc  │
                   └─────────────────────┘
```

## 2. Data Engineering

### 2.1 Sentence Splitting

Many samples in train.csv are paragraph-level (multiple sentences concatenated), while the test set is evaluated at sentence level. Used **DeepSeek** to split paragraphs into sentences, then cleaned out samples with abnormal IO length ratios and empty fields. This produced ~**6750 rows** of gold data and gave ~+2 points improvement.

### 2.2 Pseudo-labeling Published Texts

Used **Gemini-3 Pro** to split and translate published_texts.csv at sentence level. Iterated over multiple rounds, expanding to ~40k rows of silver data. This was the single biggest improvement at ~**+4 points**.

### 2.3 Gemini Synthetic Augmentation

For each paragraph in the training set, asked Gemini to generate a semantically related but different paragraph (e.g. reply letter, counter-claim, follow-up transaction) as a transliteration-translation pair. Produced ~**7800** augmented samples, contributing ~+0.5 points.

### 2.4 Preprocessing

- **Subscript character removal:** Akkadian transliterations contain Unicode subscript characters (e.g. ₂, ₃). Based on information from Gemini, these subscripts only mark different cuneiform sign variants in the original writing and do not affect pronunciation — and Akkadian word meaning depends solely on pronunciation. Removing them improved training results. Not sure how other participants handled this.
- Other preprocessing (gap normalization, parenthetical removal, number/fraction normalization, etc.) followed ideas shared in public discussions.

## 3. Training

Model: **google/byt5-base** (~580M params). Byte-level tokenization, naturally suited for the special characters and diacritics in Akkadian transliteration.

- **Optimizer:** AdamW, weight_decay=0.01
- **LR:** 3e-4, Warmup (6%) + Cosine Decay
- **Precision:** BF16
- **Max Length:** 512
- **Epochs:** 5
- **Weighted training:** gold data weighted higher than silver data in loss computation
- **Checkpoint averaging:** averaged parameters from epochs 3-5. Observed ~+0.1 LB improvement in a couple of A/B tests. Not sure if optimal, but makes sense for reducing variance in epoch selection.

## 4. Inference & Post-processing

- Beam search, num_beams=4
- Gap compression: trained with `~` as placeholder, restored to `<gap>` at inference
- Post-processing: `<big gap>` → `<gap>`, remove parenthetical content, number normalization

## 5. What Worked

| Method | Improvement | Notes |
|--------|-------------|-------|
| Subscript removal | +0.7 | Subscripts only mark cuneiform sign variants, not pronunciation |
| Sentence splitting | +2.0 | Paragraph → sentence level, aligned with test set granularity |
| Published_texts pseudo-labels (Gemini) | +4.0 | ~40k silver rows, single biggest gain |
| Gold data cleaning + val set recovery | +0.5 | IO ratio filtering, folding val set into training |
| Gemini synthetic augmentation + weighted training | +0.5 | ~7800 augmented samples, gold/silver weighting |
| Checkpoint averaging | +0.1 | Epoch 3-5 parameter averaging, reduces variance |

Overall trajectory: baseline 31.1 → splitting 33.8 → pseudo-labels 37.5 → refinement 38.3 → **final 38.9 Private** (V137)

## 6. What Didn't Work

- **1024 max_length training:** Test set is sentence-level; training on long paragraphs actually hurt (31.1 vs 33.8).
- **Context concatenation at inference (sep training):** Concatenating adjacent sentences at inference; single mode consistently outperformed.
- **DPO:** Base model too weak, DPO training didn't work at all.
- **Gold fine-tuning after silver pre-training:** Fine-tuning a silver-trained model with gold data actually decreased scores.
- **Bidirectional training (akk↔eng):** No improvement for a single model.
- **MBR decoding:** Minimal gain (0.0-0.3) for a single model, not worth doubling inference time.
- **MADLAD-400-3B:** Tried a different backbone, only reached 33.1 after 5 epochs — insufficient training.
- **ByT5-small:** Ceiling around ~36.5, clearly worse than base.
- **Name swap augmentation:** No LB improvement.
- **Disabling early stopping:** LB dropped from 37.5 to 36.6.

---

**Final:** Private **38.9** / Public 38.3 (Version 137) | ~140 submissions, 50+ experiments, 4 weeks, solo
