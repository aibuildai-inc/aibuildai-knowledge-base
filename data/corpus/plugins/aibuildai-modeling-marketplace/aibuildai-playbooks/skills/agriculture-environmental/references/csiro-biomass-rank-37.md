# Top 10 Public | Top 38 Private solution

Competition: csiro-biomass
Rank: #37
Source: https://www.kaggle.com/c/csiro-biomass/writeups/top-10-public-top-38-private-solution

# CSIRO - Image2Biomass: State-of-the-Art Biomass Quantification Framework

## Abstract
This document details our winning methodology for the CSIRO - Image2Biomass competition. Our solution addresses the complex regression task of quantifying pasture biomass from high-variance imagery. By leveraging cutting-edge Foundation Models (DinoV3, DinoV2) and implementing a robust ensemble strategy, we overcame significant challenges related to severe illumination shifts, occlusion, and extreme data scarcity. Our final architecture integrates Multi-Scale Patch Learning, Multiple Instance Learning (MIL), and advanced feature aggregation techniques to achieve superior generalization.

## Problem Formulation & Challenges
The core challenge involved regressing biomass values ($kg/ha$) from unstructured visual data under unconstrained environmental conditions.
*   **High-Variance Illumination:** Drastic dynamic range differences complicated the semantic segmentation of senescent (dry) vs. photosynthetic (green) vegetation.
*   **Small-Shot Regime:** The limited dataset size posed a high risk of overfitting, rendering traditional fully-supervised training of large CNNs or ViTs ineffective.

## Methodology: Ensemble of Foundation Models
Our strategy shifted from training scratch architectures to leveraging the semantic richness of Self-Supervised Vision Transformers. We constructed a heterogeneous ensemble of 5 distinct regression pipelines, stabilizing the aleatoric uncertainty inherent in the data.

### Architecture I: Coarse-Grained Spatial Aggregation (2x2 Grid)
*   **Backbone:** **Dino V3 ViT-7B** (State-of-the-Art Foundation Model).
*   **Feature Extraction:** Utilized a 4-patch spatial decomposition (2x2 Grid) to capture local-global context.
*   **Regression Head:** A lightweight Multi-Layer Perceptron (MLP) trained on frozen embeddings.
*   **Loss Function:** **SmoothL1 Loss** was selected over MSE to reduce sensitivity to label outliers and improve convergence stability.
*   **Robustness:** Implemented Seed Averaging (Stochastic Weight Averaging proxy) to minimize variance.
*   **Data Cleaning:** Manually curated the dataset to exclude samples with ground-truth label noise (2 instances removed).

### Architecture II: Fine-Grained Spatial Aggregation (3x3 Grid)
*   **differentiation:** Increased spatial resolution of feature extraction.
*   **Preprocessing:** Image decomposition into a 9-patch grid (3x3), allowing the model to focus on finer textural details of the pasture structure.
*   **Backbone & Head:** Identical MLP projection on Dino V3 ViT-7B embeddings.

### Architecture III: Deep Multiple Instance Learning (MIL)
*   **Architecture:** **MilTransformer**. Treated the biomass estimation as a bag-of-features problem.
*   **Backbone:** **Dino V2 ViT-G14 (Register-Equipped)**. The register tokens enhance the model's ability to discard background noise.
*   **Optimization:** Trained via standard MSE Loss.
*   **Augmentation Strategy:** Introduced Vertical Flips to enforce rotation invariance without distorting the biomass distribution prior.

### Architecture IV: Dense Strided Feature Extraction
*   **differentiation:** High-density feature sampling.
*   **Mechanism:** Employed a sliding window approach with a dense stride of 112px over Dino V3 ViT-7B embeddings. This ensures maximum coverage and overlap of critical visual features.

### Architecture V: Large-Scale Receptive Field Analysis
*   **differentiation:** Expanded context window.
*   **Mechanism:** Increased patch size to 336px to capture macro-level biomass patterns and texture density, complementing the fine-grained models.

## Engineering: Scaled Inference Pipeline
To satisfy computational constraints while maximizing throughput, we architected a distributed inference pipeline, parallelizing the massive feature extraction workload across dual GPU nodes.

## Ablation Studies & Research Limitations
We conducted extensive experiments to validate our hypothesis space. Due to the low-data regime, classical deep learning techniques proved detrimental:
*   **Negative Results:** Standard CNN fine-tuning, aggressive TTA (Test Time Augmentation), and end-to-end Transformer training led to rapid overfitting.
*   **Synthetic Domain Adaptation:** We generated synthetic samples using **Nano Banana** generative models to augment the training distribution. While this degraded Public Leaderboard performance (likely due to distribution shift), post-hoc analysis revealed it yielded the **SOTA score on the Private Leaderboard**, suggesting improved generalization capability that was penalized by the public test split distribution.

## Results Summary
Our ensemble demonstrates a significant performance uplift over individual weak learners.

| Model Architecture | Public Score (R2) |
| :--- | :--- |
| I. DinoV3 (2x2 Patching) | 0.74 |
| II. DinoV3 (3x3 Patching) | 0.74 |
| III. MIL Transformer (DinoV2) | 0.74 |
| IV. DinoV3 (Dense Stride) | 0.76 |
| V. DinoV3 (Large Patch) | 0.75 |
| **Final Ensemble (Weighted Blend)** | **0.779** |
