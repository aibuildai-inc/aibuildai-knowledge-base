# 5th Place Solution

Competition: cafa-6-protein-function-prediction
Rank: #5
Source: https://www.kaggle.com/c/cafa-6-protein-function-prediction/writeups/5th-place-solution

First of all, I would like to thank the host team and the Kaggle team for organizing and running this competition. As one of the kaggle users, I am truly delighted that CAFA — an important benchmark study in the field of bioinformatics — once again chose Kaggle as its hosting platform, following CAFA5 in 2023. CAFA5 was also a memorable competition for me, as it was the first time I won a gold medal. So I am deeply honored to have earned a gold medal once again, three years on.<br>
I would also like to thank the participants who shared insightful discussions and notebooks across both CAFA5 and CAFA6 — this solution builds on a number of ideas generously shared by the community.

---

## 1. Overview
My solution is an ensemble whose backbone is a weighted average of 6 prediction sources — three supervised models that share most of their features, plus three kNN label-transfer models on distinct modalities (sequence, 3D-structure, PPI). At the final merge stage, a shared set of post-processing steps is applied: GO hierarchy propagation, filtering to scorable GO terms, and taxon constraints.

An overview of the full pipeline is shown in the figure below.


- MCM-NN is a custom neural network built for this competition; detailed in §5.2
- Throughout this writeup, "supervised models" refers to the three models that learn parameters from the training labels (MCM-NN, logistic regression, PyBoost), as distinguished from the "kNN label-transfer" methods, which learn no parameters and instead transfer labels from retrieved neighbours.

The final submission is a weighted average of the following 6 sources:

| Source | Category | weight | Aspects | Notes |
| --- | --- | :---: | :---: | --- |
| Set 1 (PLM + taxon-Phylum + NEA) | Supervised | 0.20 | BP / MF / CC | use base feature set |
| Set 2 (PLM + **taxon-Order** + NEA) | Supervised | 0.16 | BP / MF / CC | more precise taxonomy information |
| Set 3 (PLM + taxon-Phylum + **PubMed-emb (MedCPT)** + NEA) | Supervised | 0.16 | BP / MF / CC | use PubMed literature information |
| Foldseek kNN | Label transfer | 0.16 | BP / MF / CC | 3D structure based |
| DIAMOND kNN | Label transfer | 0.16 | BP / MF / CC | Sequence similarity based |
| PPI kNN | Label transfer | 0.16 | **BP only** | PPI confidence based (STRING database) |

*\* The merge is a NaN-aware weighted average: for any [protein, GO] pair where a given source has no prediction, that source's weight is dropped and the remaining weights are renormalized. Thus for MF / CC — where PPI is absent — the average implicitly renormalizes over the remaining 5 sources.*

---

## 2. General (Common settings)

### 2.1 Per-aspect training and prediction

The three ontologies (BP / MF / CC) are trained and predicted independently and concatenated only at the very end. Each of the three supervised models (NN / logreg / pyboost) is fit per aspect, and the kNN models are also processed aspect-wise so that only the training labels for the target aspect are consulted.

### 2.2 Label (GO term) selection for supervised training

The following filters are applied in order to select the GO terms used as training labels.

- **1. Frequency-based filter**: training labels are restricted to GO terms that occur at least 41 times in the training data.
- **2. IA-based filter (only for logreg and pyboost)**: For logreg and pyboost training, GO terms with `IA (Information Accretion) = 0` are dropped. For MCM-NN, `IA = 0` terms are instead kept in the label set, which empirically gave a better LB score. The intuition is that learning labels not directly scored by the CAFA metric acts as an auxiliary task — it helps the shared backbone learn richer GO-hierarchy representations. The per-label loss weight in MCM-NN is set to `1 + IA`.

### 2.3 Common post-processing

Regardless of whether a component is supervised or kNN-based, all of them flow into the final merge through the following common post-processing pipeline.

1. **GO hierarchy propagation**: propagate child-GO scores up to parents via *max*, enforcing hierarchical consistency.
2. **Scorable-GO filter**: keep only GO terms that are scored.
3. **Drop known labels**: any [protein, GO] pair already present in `train_terms.tsv` is removed.
4. **Taxon constraint**: GO terms that cannot exist in a given taxonomic group (forbidden GO) are dropped.
5. **Top-N per (protein, aspect)**: keep only the top N entries by descending score per aspect.
6. **Score ≥ 0.01 filter**: anything below this threshold is excluded from the submission file.

---

## 3. Cross-validation setup

I implemented a multilabel / group-aware splitter (`MultilabelStratifiedGroupKFold`, n_splits = 5) tailored to the two requirements below.

- **Group integrity**: identical sequences must not be split between train and val. Even when the UniProt accessions differ, *any pair of proteins with identical sequences is treated as the same group*.
- **Multilabel stratification**: the positive rate of every label should be balanced across the 5 folds, and even rare classes must have positive examples in every fold.

#### Algorithm (summary)

1. **Pre-aggregation**: proteins are collapsed into groups (one group per unique sequence), and for each group the number of proteins carrying each label is counted, yielding a `(num_groups × num_labels)` positive-count matrix.
2. **Step 1 — rare-class-first assignment**: labels are processed in ascending order of group support (rarest first). For each label, the unassigned groups holding it are dealt out evenly across folds, so that even rare classes are guaranteed positives in every fold.
3. **Step 2 — greedy assignment of remaining groups**: every remaining group is added to the fold with the lowest cost, `cost(fold) = (current sample count) + 0.1 × (deviation of the fold's label distribution from the global average)`. The first term enforces size balance, the second enforces stratification; the coefficient 0.1 keeps size balance dominant.

Clustering by sequence similarity (e.g. CD-HIT, MMseqs2) would have grouped near-duplicate sequences as well, which would likely have made the CV more robust. This solution did not go that far.

---

## 4. Speeding up local scoring

Given the timeline of CAFA, putting too much trust in the public LB is risky, and iteration should also take local cross-validation score into consideration. What enabled fast iteration here was [@antoninadolgorukova](https://www.kaggle.com/antoninadolgorukova)'s shared [idea for speeding up cafaeval](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/discussion/664359).

I simply had Claude Code read her post and quickly created a fork of the original CAFA-evaluator-PK (https://github.com/claradepaolis/CAFA-evaluator-PK), and even such a casual, unoptimized version resulted in about a 4× speedup and greatly helped shorten my iteration loop. With a more thorough optimization, she reports an astonishing speedup of well over 10×.

---

## 5. Supervised models

All three "Sets" share the same internal structure — an ensemble of three models: MCM-NN (x0.4) + logreg (x0.3) + pyboost (x0.3). The only thing that differs between Sets is the input features.

| Set   | Input features                                             | Notes              |
| ----- | ---------------------------------------------------------- | ------------------ |
| Set 1 | PLM + taxon (Phylum level) + NEA                           | base features      |
| Set 2 | PLM + **taxon (Order level)** + NEA                        | finer taxon level  |
| Set 3 | PLM + taxon (Phylum level) + **PubMed-emb (MedCPT)** + NEA | add literature feature |

---

### 5.1 Feature engineering

All supervised models draw from the same pool of four feature families. The first three are model-agnostic (the same vector is fed to every model); the NEA tensor is preprocessed in two flavours, one for the NN and one for logreg / pyboost.

#### 5.1.1 PLM (Protein Language Model) embeddings

**ESM-2 (esm2_t36_3B_UR50D, 2560 dim)** + **ProtT5-XL (1024 dim)**, both mean-pooled and concatenated, for a total of 3584 dim.

#### 5.1.2 Taxon hierarchy one-hot
Unlike CAFA5, using the raw taxon ID (the leaf level) in CAFA6 is far too sparse (some taxa appear only rarely in train, others only in test). Therefore, the taxon tree was retrieved using the [approach](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/discussion/613750) shared by [@mtinti](https://www.kaggle.com/mtinti) and a higher level was used instead (`Domain → Phylum → Class → Order → ... → Species`). Climbing to a higher level reduces label sparsity, but the coarser grouping carries less information as a feature — a trade-off. Set 1 and 3 use a coarser level; Set 2 uses a finer one. Categories with too few examples were bucketed into `[UNK]`.

#### 5.1.3 PubMed literature embeddings (Set 3 only)
Keyed by UniProt accession, all related PubMed articles for each protein were fetched, and their title + abstract were embedded using MedCPT (https://huggingface.co/ncbi/MedCPT-Article-Encoder), a biomedical text encoder available on Hugging Face. The literature embedding is added only to Set 3, as an additional information source distinct from the protein sequence itself.

#### 5.1.4 NEA (Non-Experimental Annotation)
NEA features are derived from the GOA `(protein, GO, evidence_code)` triples that carry non-experimental [evidence codes](https://geneontology.org/docs/guide-go-evidence-codes/) (IBA, IEA, ISO, ISS, ND, ...). After GO hierarchy propagation, the top 1,000 most frequent non-experimental GO terms are turned into features.

NEA is shared across all Sets and all three models, but the preprocessing differs by model:

| Model | NEA preprocessing |
| --- | --- |
| MCM-NN | Kept as a structured tensor that preserves the evidence-code axis, fed into a dedicated NonExperimental Module (see §5.2). |
| logreg / pyboost | Collapsed (mean over the evidence-code axis) to a flat vector, then `hstack`ed with the other features. |

Logistic Regression and PyBoost cannot consume a structured tensor directly, so the evidence axis is averaged away. This loses "which evidence code the information came from," but retains the per-GO assignment probability. The NN, in contrast, exploits the evidence dimension through a dedicated sub-module.

---

### 5.2 MCM-NN

#### Overview

This neural network draws heavily on the [CAFA5 3rd-place solution](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/writeups/tito-3rd-place-solution-for-the-cafa-5-protein-fun) shared by [@its7171](https://www.kaggle.com/its7171) and the [SPROF-GO paper](https://academic.oup.com/bib/article/24/3/bbad117/7085635?login=false). The architecture is a two-branch design: a Main MLP fed with the flat feature vector, and a NonExperimental Module fed with the NEA (Non-Experimental Annotation) features. Each branch emits logits independently; the two logit vectors are summed element-wise, passed through a sigmoid, and then refined by the Max Constraint Module (MCM).



#### NonExperimental Module

This module follows the design principle from the Non-Experimental Annotation module introduced in [CAFA5 3rd-place solution](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/writeups/tito-3rd-place-solution-for-the-cafa-5-protein-fun) by [@its7171](https://www.kaggle.com/its7171), whose key idea is to exploit non-experimental GO annotations — which are excluded from the ground-truth labels — as additional input features, keeping them separated by evidence code. I reimplemented this principle to fit the input shape used here, but the internal design is otherwise specific to my own implementation and omitted from this writeup.

#### Max Constraint Module (MCM) and MCLoss

MCM and MCLoss are taken from the [SPROF-GO paper](https://academic.oup.com/bib/article/24/3/bbad117/7085635?login=false). The core idea of MCM is to embed in the forward pass the hierarchical-consistency constraint that "**the output score of a parent GO term must be at least the max over the scores of all its descendants**." Concretely, the GO hierarchy is precomputed into a descendant matrix `M ∈ {0,1}^(K×K)` (`M[i,j] = 1` ⇔ term *i* is an ancestor of *j*, including itself), and for each term *i* the output is the max of the descendant scores selected by `M[i, :]`.

SPROF-GO further uses an **asymmetric design in which the set of descendants used for the max differs between inference and training**:

- **At inference**: the max is taken over *all descendants*, without consulting the label, yielding a hierarchy-consistent distribution.
- **At training**: if the parent is positive (`y = 1`), the max is taken only over *positive descendants*; if negative (`y = 0`), over all descendants. This prevents the shortcut where a positive parent satisfies its loss just because some *negative* child happens to receive a high score.

The loss is BCE evaluated on the post-MCM scores (= **MCLoss**), which folds hierarchical consistency into the training objective and removes the need for max-propagation as a separate post-processing step. For the full formulation, see the [SPROF-GO paper](https://academic.oup.com/bib/article/24/3/bbad117/7085635?login=false) and [GitHub repo](https://github.com/biomed-AI/SPROF-GO).

#### Training settings (only batch_size differs per aspect)

| Settings | Value |
|---|---|
| optimizer | AdamW (weight_decay = 0.02) |
| LR scheduler | CosineAnnealingLR, 1e-3 → 1e-6 over the full run (no restarts) |
| n_epochs | 200 (upper bound; EarlyStopping on `val_loss_ema`, patience = 8) |
| Loss | MCLoss (BCE-based) + label smoothing (1e-3, training only) |
| loss weight per label | 1 + IA |
| Mixup | p = 0.8, alpha = 0.2 |
| EMA decay | 0.99 |
| batch size | 8 for BP, 256 for CC and MF |

The batch_size for BP had to be set extremely small. The VRAM bottleneck is the MCM forward pass, which produces a `[B, K, K]` intermediate tensor (K = output dimension = number of GO terms used for training). Memory therefore scales as K².<br>
After the label-selection procedure in §2.2, the output dimension per aspect is BP = 4,747, CC = 685, MF = 859. BP is 5–7× larger than CC/MF, which corresponds to roughly 25–49× more memory in the `[B, K, K]` tensor. The batch_size ratio (256 → 8 = 1/32) was chosen to offset this K² scaling.<br>
Although the textbook practice when changing batch_size by such a factor is to also retune lr, training remained stable with a unified lr = 1e-3 across all aspects. Per-aspect lr tuning would likely have improved results further, but was skipped due to time constraints. The training probably got away with the unified (and likely suboptimal) learning rate because (a) AdamW's adaptive per-parameter learning rate is much less sensitive to batch size than SGD-style optimisers, and (b) EMA, Mixup, and label smoothing together smooth the loss landscape.

---

### 5.3 Logistic Regression

- A much simpler algorithm provides patterns that the NN does not capture — an option that many top CAFA5 solutions also used.
- **RAPIDS cuML** is used rather than scikit-learn so training runs fast on GPU.
- The input is a single flat vector: PLM + taxon + (PubMed for Set 3) + NEA (flat-vector flavour), concatenated with `hstack`.

### 5.4 PyBoost

- PyBoost can train multilabel targets in a single model with strong GPU efficiency, making it a natural fit for this scale (~100k+ proteins × thousands of labels).
- I reuse the hyperparameters from the [CAFA5 2nd-place solution](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/writeups/u900-private-2nd-public-5th-solution-py-boost-and-) verbatim and did not optimize further.
- The input features are identical to Logistic Regression.
- Reference for PyBoost:
    - [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a36c3dbe676fa8445715a31a90c66ab3-Paper-Conference.pdf)
    - [GitHub repository](https://github.com/sb-ai-lab/Py-Boost)

---

## 6. kNN label-transfer models

These are **training-free, retrieval-based methods**: no model parameters are learnt. For each test protein, nearby train proteins under some similarity metric are retrieved, and their labels are aggregated via a **similarity-weighted average** to predict the test protein's labels.

The three sources differ only in the nearest-neighbour search algorithm. **DIAMOND-kNN and PPI-kNN share a common scoring formula** (shown below); **Foldseek-kNN** uses a related scoring scheme provided by the [ProFun](https://github.com/SamusRam/ProFun) library shared by [@samusram](https://www.kaggle.com/samusram) in CAFA5, but the spirit (similarity-weighted neighbour-label aggregation) is the same.

### Shared scoring formula (used by DIAMOND-kNN and PPI-kNN)

```
weight       = similarity metric of the neighbour (DIAMOND: bitscore; PPI: combined_score / 1000)
score(term)  = Σ_neighbour (weight × IC(term)) / Σ_neighbour weight
IC(term)     = -log2(p(term)) / max_IC   (from training-set occurrence probability, normalised by the max)
```

In words: take a **similarity-weighted average over the top-k neighbours** and multiply by the GO term's **information content (IC)**. Only labels that actually appear in the retrieved neighbours can ever be predicted, so the output is naturally sparse.

### 6.1 Foldseek-kNN

- Foldseek — essentially BLAST for 3D structures — was run against an AlphaFold2 structure database, downloaded in advance for all train + test proteins.
- The implementation uses the **`FoldseekMatching` model from the [ProFun](https://github.com/SamusRam/ProFun) library** (the kNN aggregation itself is not reimplemented from scratch).
- The exact scoring formula is delegated to ProFun; it is similarity-weighted in spirit and very close to the shared formula above.

### 6.2 DIAMOND-kNN

- DIAMOND is a fast BLAST replacement that outputs sequence-similarity bitscores.
- Workflow: build a DIAMOND DB from the train sequences → query with test sequences → score the hit labels with the **shared formula above**, using **bitscore as the weight**.

### 6.3 PPI-kNN (BP aspect only)

- Data source: the protein–protein interaction network from [STRING database](https://string-db.org/). Only edges with `combined_score ≥ 400` are kept.
- The weight is normalised as `weight = combined_score / 1000` and plugged into the **shared formula above**.
- Althogh BP terms correlate strongly with the functions of neighbouring proteins, little gain was observed on MF and CC in my experiments. So PPI-kNN is used only for BP.

---

## 7. Final ensemble merge

Once every component has produced its submission file, they are merged as follows.

1. Merge the 6 sources by a **NaN-aware weighted average** (weights renormalised over the sources that actually have a prediction). Weights: Set1 = 0.20, others = 0.16 each.
2. GO hierarchy propagation (child → parent, max).
3. Scorable-GO filter / known-label removal / taxon constraints.
4. Truncate to top 200 per (protein, aspect) and apply score ≥ 0.01.

Each of Sets 1–3 is itself an internal 3-model ensemble (`NN × 0.4 + logreg × 0.3 + pyboost × 0.3`), so the overall structure is effectively a **multi-stage ensemble**.
