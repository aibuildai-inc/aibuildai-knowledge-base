# 16th Place Solution -- Trust your cv

Competition: playground-series-s6e2
Rank: #16
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/16th-place-solution-trust-your-cv

## 1. Introduction  
Happy to be writing this writeup for 16th place solution. My final submission scored **0.95534 on the private LB**.  

The key to my solution was **not a single dominant model**, but a **diverse collection of models** combined with a **custom rank‑based weight optimiser** that also learned a non‑linear power transform.  

---

## 2. Overall Strategy  
My pipeline consisted of several stages:

1. **Generate multiple feature representations** from the original data.  
2. **Train a wide variety of models** (tree‑based, neural, tabular transformers, AutoGluon) on each representation.  
3. **Collect ~200 OOF predictions** using a consistent 5‑fold stratified split.  
4. **Select a diverse, high‑quality subset** based on AUC and Spearman correlation.  
5. **Optimise ensemble weights** with a custom hill‑climbing algorithm that also learns a power transform.  
6. **Apply the learned weights and power to the test set** for final submission.  

---

## 3. Feature Engineering – Creating Multiple Views  
Different images of data with different feature subsets used to generate diversity. The notable engineered features included:

Target Encoding: Applied within each fold to avoid leakage, generating statistics such as mean, standard deviation, and count for each categorical value. This helped capture the relationship between categories and the target in a smooth, regularised way.

Categorical Conversion: All features were converted to categorical dtype and passed natively to models that support categorical inputs. At the same time, the original numerical versions of these features were retained, giving models the flexibility to choose the most informative representation.

---

## 4. Cross‑Validation Strategy & Leakage Prevention  
-  **5‑fold StratifiedKFold** (shuffle=True, random_state=42)**.
- **Stratification** was based on the target feature **Heart Disease**.  

--- 

## 5. Model Zoo – The 200+ OOF Predictions  
For every feature representation, the same split indices were used across all models to keep OOF predictions aligned.

| Model Family | Specific Models / Variants 
|--------------|----------------------------
| **XGBoost** | `gbtree`, `dart`, `gblinear` (each tuned via Optuna with 40‑50 trials) 
| **LightGBM** | `gbdt`, `goss`, `dart` (separate Optuna runs) 
| **CatBoost** | Default and tuned (used `Pool` for efficiency) 
| **PyTorchTabular** | `RealMLP`, `DANet`, `AutoInt`, `NODE`, `GANDALF`, `TabNet` 
| **PyTabKit** | `RealMLP_TD`, `TabM_D`, `ResNet_RTDL_D`, `FFT_Transformer` 
| **AutoGluon** | BestQuality and Extreme presets (each produced several internal models) 

---

## 6. Selecting a Diverse Elite Subset  
With over 200 OOF predictions, a simple average degraded performance. I built a custom **diversity‑aware selector**:

1. **Anchor Model**: Started with the best CV model (`catboost`, AUC 0.955705). 
2. **Quality Filter**: Retained only models with AUC ≥ 0.954 (158 models qualified).
3. **Diversity Selection**: Greedily added models with lowest average Spearman correlation to the current set.
4. **Size Cap**: Selected the top 100 most diverse models.

The final elite set had an average pairwise correlation of ~0.995 – still high, but this greedy approach extracted maximum possible diversity given the data.

---

## 7. The Final Optimiser – Power‑Rank Hill Climbing  
Instead of a simple weighted average, I used a **custom hill‑climbing algorithm** that also learns a **power transform** on the ranks of predictions.

### 7.1 Why Ranking + Power?
- **ROC AUC only cares about ranking**, so converting predictions to ranks (uniform [0,1]) removes scale differences between models.  
- A **power transform** (rank raised to a power p) applies a non‑linear warp:  
  - p > 1 emphasises high ranks,  
  - p < 1 spreads out low ranks.  
- Optimising both weights and a single global power p adds an extra degree of freedom that can better match the true relationship.

### 7.2 Algorithm Outline
The algorithm performs a batch‑based random search with annealing:
- Start with equal weights and p = 1.0.
- At each step, generate a batch of candidate weight vectors by adding Gaussian noise to the current best weights, then renormalising.
- Also generate small random perturbations to p.
- Evaluate all candidates using the AUC on the OOF set.
- Update the best weights and p if improvement is found.
- Reduce step size over time (annealing) and stop early after a patience period.

### 7.3 Results
The optimiser ran  achieved an **OOF AUC of 0.955785** – the best among all my attempts.  
The optimal power was **p = 0.8**, compressing the rank distribution.

---

## 8. Final Test Predictions  
The learned weights and optimal power were applied to the test set: test predictions are first ranked, then raised to the learned power, and finally combined via weighted sum.  

The final CSV scored **0.95534 on the private LB**, securing 16th place.

---

## 9. What Did Not Work  
- **Pseudo‑labelling** (both soft and hard) – provided no measurable gain. Interestingly, when using logistic regression as the meta‑model with hill climbing results as pseudo‑labels, one of my submissions achieved a better private score despite having a slightly lower CV – that could've secured me an even better rank.

- **Public LB chasing** – I deliberately avoided optimising for public leaderboard scores and placed my trust entirely in cross‑validation.

---

## 11. Acknowledgements  .  
-Thanks to @cdeotte, @tilii, @optimistix, @mahoganybuttstrings , @masayakawamata for their discussions, code and their previous writeups. Learned from the very best.
