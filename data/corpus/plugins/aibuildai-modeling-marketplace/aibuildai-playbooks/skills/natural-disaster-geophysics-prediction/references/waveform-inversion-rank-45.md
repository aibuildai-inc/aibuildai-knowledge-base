# 45th Place Solution

Competition: waveform-inversion
Rank: #45
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587553

## Introduction
First of all, I would like to express my deep gratitude to the organizers Yale, UNC-CH, Kaggle. This competition was interesting and educational, and I'm glad I participated.
I would also like to deeply thank @brendanartley for providing the public code. My solution is developed based on Bartley's public code.

---
## 1.Overview
- Trained several models using ConvNeXt and CAFormer models with different model-size configurations
- Best single model was ConvNeXt-Base (CV=23.87, LB=25.7)
- Final submission was a weighted median ensemble of 9 submissions achieving Public MAE = 23.2, Private MAE = 23.1
- Due to the large dataset size, cross-validation was challenging, so I adopted an extended hold-out approach, which maintained high correlation with Public LB

## 2. Dataset
- **OpenFWI (all data)** only  
- No resizing, no augmentation, no synthetic generation  

## 3. Validation Strategy
- **Expanded hold-out** split: **Train ≈ 88 % / Val ≈ 12 %** from the public code. 

## 4. Single-Model Results  

**Table&nbsp;1 – Single-model performance (lower is better)**  

| Backbone | Epochs | CV MAE | Public MAE | Notes |
|----------|-------:|-------:|-----------:|-------|
| ConvNeXt-Base | 200 | **23.87** | **25.7** | Best single model |
| CAFormer-m36  | 180 | 24.08 | 26.0 | StarReLU |
| CAFormer-b36  | 180 | 24.96 | 29.5 | StarReLU |
| ConvNeXt-Base | 150 | 25.58 | 26.7 | Checkpoint fine-tuned twice |
| ConvNeXt-Large | 150 | 26.08 | 27.8 | — |
| Flash InternImage-b | 50 | 34.60 | 37.6 | FP32, ≈ 210 min / epoch |
| ConvNeXt-Base (full data) | 218 | — | 27.7 | Trained on 100 % data |
| ConvNeXt-Base (full data) | 243 | — | 26.7 | Trained on 100 % data |
| ConvNeXt-Base (full data) | 258 | — | 26.6 | Trained on 100 % data |

Hyper-parameters tuned: `weight_decay`, `dropout_rate`, `decoder_channels`, `clip_grad_norm`  
LR restarts: resumed from the previous terminal LR (`1 e-5`) with `CosineAnnealingLR(T_max = 50)`, which proved more stable than warm restarts.

## 5. Ensembling & Post-processing  

**Table&nbsp;2 – Ensemble performance (Public MAE)**  

| Method | Submissions | Public MAE | Δ vs. Best Single |
|--------|-----------:|----:|------------------:|
| Simple median | 9 | 23.4 | −2.3 |
| **Weighted median (final)** | **9** | **23.2** | **−2.5** |

- Median-based methods clearly outperform means for MAE.  
- Weighted median improved MAE by **0.2** over the simple median.  
- Type-wise weighted means (after predicting sample type with 99.5 % accuracy) did **not** beat median.  
- Post-processing is limited to **`clip(1500, 4500)`** (additional –0.002 CV MAE).

## 6. Speed-up Techniques
| Category | Key settings |
|----------|--------------|
| **Compiler / Optimizer** | `torch.compile(mode="max-autotune")`, **Fused AdamW** |
| **Mixed Precision** | AMP (**bfloat16**) for all ConvNeXt / CAFormer runs |
| **DataLoader** | `num_workers` tuned, `prefetch_factor` tuned, `pin_memory=True`, `persistent_workers=True` |
| **Hardware** | Experiments run on **RTX 3090** & **A100**<br>Flash InternImage remained FP32 due to DCNv4 → **≈ 210 min / epoch (A100)** |

“num_workers was experimentally tuned—the value is a sweet spot: too many or too few threads slowed training, so we benchmark-benchmarked different settings and used the fastest one for all runs.”


## 7. What Didn’t Work

| What failed | Likely cause / lesson |
|-------------|-----------------------|
| Type-wise weights, heavy post-processing | Median already robust; extra weighting brought no gain |
| Flash InternImage-b | AMP unstable; FP32 made training prohibitively slow, limiting epochs |
| EVA family | Mishandled window / resize parameters – never converged |
| Full-data training (100 %) | Slightly worse Public MAE vs. 88 % split, much longer runtime |

## 8. Results & Lessons Learned
- **Final scores:** Public MAE 23.2 / Private MAE 23.1 (weighted median)  
- **Ensembling gain** saturated at ≈ 2.5 MAE → future effort should target single-model quality.  
- LR scheduling and restart strategy had greater impact than backbone size beyond *ConvNeXt-Base*.  
- Further gains likely lie in synthetic data and higher-resolution training, as hinted by top teams.

---

## References
- Bartley Baseline Notebook  
  <https://www.kaggle.com/code/brendanartley/convnext-full-resolution-baseline>  
  <https://www.kaggle.com/code/brendanartley/caformer-full-resolution-improved>
- Training / Speed-up Discussion  
  <https://www.kaggle.com/competitions/waveform-inversion/discussion/583896>

---
