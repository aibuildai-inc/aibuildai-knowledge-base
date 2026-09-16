# [32nd] ByT5-base Fine-tuning | No LLMs, No Synthetic Data, No Ensembles

Competition: deep-past-initiative-machine-translation
Rank: #32
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/32nd-byt5-base-fine-tuning-no-llms-no-synthet

# [32nd] ByT5-base Fine-tuning | No LLMs, No Synthetic Data, No Ensembles

**Rank:** 32 / 2674 teams | **Public:** 37.6 | **Private:** 38.1

---

## Overview

No Gemini, no synthetic data, no large-scale PDF extraction. Our strategy was simpler: start from a strong base model, train carefully, iterate continuously, and use the leaderboard as the real validation signal.

This writeup is for participants who want to understand what is achievable with only public data and a disciplined iterative process.

---

## Two-Stage Training

**Stage 1: Akkademia Pre-training**

ByT5-base was first fine-tuned on ~69k sentence pairs from the Akkademia corpus. Akkademia is not Old Assyrian — the relationship is similar to Portuguese vs Spanish. The goal was not to learn the competition domain directly, but to give the model familiarity with Akkadian grammar and morphology before seeing competition data. Without this stage, the model treats Old Assyrian as completely foreign.

**Stage 2: Iterative Fine-tuning**

Due to Kaggle runtime limits (~12h/session), we adopted a progressive fine-tuning loop in cycles of 3 epochs:

1. Train 3 epochs from best checkpoint
2. Save → create Kaggle dataset
3. Run inference → submit
4. Analyze score
5. Continue or adjust

We repeated this loop over nearly 3 months, reaching **21 total epochs**. Each iteration taught us which normalization rules helped, which hurt, and what decoding parameters worked best. The leaderboard became our continuous evaluation framework.

---

## Data

Public data only:

| Source | Rows |
|---|---|
| `train.csv` | 1,561 documents |
| `published_texts.csv` | ~199 recovered transliterations |
| `Sentences_Oare_FirstWord_LinNum.csv` | 841 gold sentence pairs |
| `akkademia_train_CLEAN.csv` | ~69k pre-training pairs |

---
After splitting long documents: **~2,800 sentence-level training samples**

The `Sentences_Oare_FirstWord_LinNum.csv` file was a critical late discovery ; it contains gold sentence boundaries using `first_word_spelling` as anchor, which closely matches how the hidden test set was constructed.

---

## Key Findings from Leaderboard Submissions

- `early_stopping=True` causes repetition artifacts → hurts score
- `length_penalty=0.91` consistently outperforms lower values
- Complex postprocessing provides no benefit; the model already internalized normalizations during training
- Splitting documents to sentence level improves alignment with test distribution

---

## Normalization

Applied to training data following host recommendations:

- `ḫ/Ḫ → h/H` (confirmed absent from test set)
- Decimals → unicode fractions (`0.3333 → ⅓`)
- `<gap><gap>` / `<big_gap>` → single `<gap>`
- Subscripts → integers (`₄ → 4`)
- Removed `fem.`, `pl.`, `(?)` annotations
- `PN → <gap>`
- Roman months → integers

---

## Model & Inference

**Model:** [akk-v24-ep21 on Kaggle](https://www.kaggle.com/datasets/assiaazzouz/akk-v24-ep21)

- Base: `google/byt5-base`
- Total epochs: 21
- `num_beams=6`, `length_penalty=0.91`, `early_stopping=False`
- No postprocessing

### Inference : Less is More

Our final inference is intentionally minimal:

```python
out = model.generate(
    **enc,
    max_new_tokens=300,
    num_beams=6,
    length_penalty=0.91,
    # NO early_stopping
)
predictions = tokenizer.batch_decode(out, skip_special_tokens=True)
# NO postprocessing applied
```

**No preprocessing of input. No postprocessing of output.**

This was a late but important discovery. We spent months building a careful postprocessing pipeline; normalizing fractions, removing annotations, fixing gaps. Every rule was based on host-confirmed guidelines.

Yet when we tested simple inference against our full pipeline, simple inference scored the same or higher.

**Why?** The model was trained on already-cleaned data. It learned to output correct fractions, correct quotes, correct gaps directly. Our postprocessing was correcting things that were already correct; and occasionally breaking things that were fine.

The lesson: **preprocess training data carefully, postprocess model output as little as possible.** Interestingly, the 1st place team said exactly the same thing in their writeup.


---

## What Limited Us

- No ByT5-Large/XL (top teams used XL on 8×H20)
- No PDF extraction pipeline (top teams built 30k+ pairs with Gemini)
- No pseudo-labeling or synthetic data
- Kaggle runtime forced 3-epoch increments

---

## Final Thoughts

The 1st place writeup says it best: **"Data quality dictates everything."** Top teams had 30k+ clean sentence pairs. We had ~2,800. Despite this gap, the results show that careful iteration and data alignment can compensate for limited resources.

**37.6 public → 38.1 private. Public data only. ByT5-base.**

---
