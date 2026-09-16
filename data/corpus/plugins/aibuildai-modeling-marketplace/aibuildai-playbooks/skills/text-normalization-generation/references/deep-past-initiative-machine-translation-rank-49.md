# 2-Stage CV, CPT & MBR Decoding

Competition: deep-past-initiative-machine-translation
Rank: #49
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/2-stage-cv-cpt-and-mbr-decoding

# Akkadian Translation: 2-Stage CV, CPT & MBR Decoding

First of all, congratulations to the winners and many thanks to the organizers for hosting this competition. My strategy was designed around building a stable pipeline that leverages continual pre-training and advanced decoding techniques.

Here is a high-level overview of my approach.

---

## 1. Foundation: Continual Pre-Training (CPT)
Instead of starting fine-tuning directly from `google/byt5-base`, I established a baseline foundation through **Continual Pre-Training (CPT)**. 
I took the vanilla `byt5-base` model and subjected it to an unsupervised/self-supervised pre-training phase over a corpus of **domain-specific Akkadian text**. 

**Why Continual Pre-Training?**
Ancient languages exhibit massive domain shifts compared to modern languages (which standard T5 models are trained on). By allowing the model to freely learn the morphological structures, cuneiform transliteration patterns, and phonotactics of Akkadian *before* introducing the translation alignment objective, the network builds a structural understanding of the source language. This CPT phase transformed a generalized multilingual model into an Akkadian-aware backbone, providing a stable prior for downstream fine-tuning.

## 2. Data Strategy & Normalization
Machine translation for ancient languages is often noisy. I built preprocessors and postprocessors optimized with regex and vectorized operations:
* **Gaps & Breaks:** Unified all gap representations (`< gap >`, `big gap`, `broken lines`, `x x`) into a canonical `<gap>` token.
* **Fractions:** Mapped numeric decimals accurately to exact Unicode fractions (e.g., `0.8333 $\to$ 5/6`).
* **Diacritics & Determinatives:** Handled uppercase/lowercase transliteration conventions, converting numeric notations (like $a_2, a_3$) to properly accented characters (á, à) to harmonize the input space.
* **Leakage Prevention:** When mixing **Cleaned external data**, overlap checks were implemented so validation targets were unseen during training.

## 3. Training Strategy: Supervised Fine-Tuning (SFT) via 2-Stage 3-Fold CV
Starting from the CPT-enhanced Base Model, I utilized a **Group K-Fold (k=3)** grouped by unique identifiers. Inside each fold, the Supervised Fine-Tuning (SFT) process follows a 2-stage approach:

* **Stage 1 (Mixed SFT):** The model is trained on a combination of the **Official training data** and **Cleaned external data** for **10 epochs**. This expands the vocabulary and syntax understanding.
* **Stage 2 (Refinement SFT):** A secondary fine-tuning pass **only** on the **clean Official training data** for **5 epochs**, with the learning rate scaled down by `0.1x`. This helps the model adapt precisely to the target distribution of the competition. 

Early stopping and grouped evaluation were used to monitor overfitting.

## 4. The Final Soup (Averaging the Folds)
At the end of the cross-validation phase, three checkpoints emerge. 

Standard prediction ensembling—where multiple models generate translations simultaneously and vote—is memory-intensive and often unfeasible under strict inference time limits. To solve this, I applied **Weight Averaging (Model Souping)** across the 3 fold models. 

**Why a Final Soup?**
By mathematically averaging the weights of the 3 fold checkpoints (giving 1/3 weight to each) into a single unified model, the architecture achieves some of the variance-reduction benefits of an ensemble, but at the computational inference cost of a single model. It essentially smooths the loss landscape without any latency penalty during the decoding phase.

## 5. Inference Optimization & MBR Decoding
The inference engine was optimized to maintain translation quality under compute limits:
* **Hardware Accel:** BF16 precision + optimized transformer backends.
* **Bucket Batching:** Sorted samples by length to minimize padding overhead. Adaptive beam sizes (short texts got fewer beams) kept generation fast.
* **Minimum Bayes Risk (MBR) Decoding:**
  * Generated a pool of up to 10 candidates using Beam Search (beams=12).
  * Ranked candidates using a modified **CHRF++** algorithm.
  * Added penalties for **Gap-Mismatches** (comparing `<gap>` counts between the Akkadian source and English candidate).
  * Factored in a self-confidence bonus based on original beam rank.

It was a great learning experience blending modern NLP techniques with Assyriology. Thanks for reading!
