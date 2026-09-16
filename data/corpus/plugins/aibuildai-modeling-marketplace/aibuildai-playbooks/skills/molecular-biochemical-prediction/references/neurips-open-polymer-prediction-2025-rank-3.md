# 3rd Place Solution

Competition: neurips-open-polymer-prediction-2025
Rank: #3
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/3rd-place-solution

This is my first year on Kaggle and also my first Kaggle competition. Encountering repeated errors with the test set in my very first Featured Competition was very unfriendly to a newcomer like me. This current version of the solution was written following the requirements from the email.
Initially, I tried to use Uni-Mol for predictions and wasted a lot of time before realizing I couldn't use it well, and my local computing power was completely insufficient, so I had no choice but to give up. However, working with Uni-Mol made me realize the importance of data preprocessing, which made me much more proficient when I later switched to using GNNs.

#code,PPT,data here:
https://github.com/fresnellll/kaggle-NeurIPS-polymer-prediction-solution

# Overview
My winning method is a fusion scheme that combines a Graph Attention Network (GATv2Conv) with selected Morgan Fingerprint features. I believe the main factors for my success are:
1,GATv2's attention mechanism can already capture complex, high-level geometric and chemical environment information. The simple, bit-based Morgan Fingerprints provide complementary, low-level substructure information without creating conflicting signals.
2,Recognizing that the data might have biases, I used the validation set as a baseline to calibrate the data and used 5-fold cross-validation to mitigate the risks posed by potential data anomalies.

Core GNN Backbone: Each GATv2Conv layer utilizes 8 Attention Heads, allowing the model to focus on different neighborhood information patterns simultaneously.The GNN's hidden dimension is 384. With 8 heads, each head processes a 48-dimensional feature space (384 / 8 = 48).
Residual Connections are implemented via simple element-wise addition, enabling stable training of the 6-layer deep network.
Feature Fusion Architecture:The fusion of global and local features is a critical step. We use simple but effective concatenation. 

| Graph Embedding (384 dims) | Top 50 FP (50 dims) |
Independent MLP Prediction Heads:
Following the fusion layer, each of the 5 target properties has its own dedicated prediction head.
Each head is a 2-layer MLP with a ReLU activation and Dropout for regularization:
Linear(in=434, out=384)
ReLU()
Dropout(p=0.2)
Linear(in=384, out=1)
[model overview]

**Training Pipeline **
Robust Ensembling with 5-Fold Cross-Validation:
The final submission is a simple average ensemble of the predictions from the 5 models. Each model is given an equal weight of 20%.
Training Hyperparameters & Optimization:
Optimizer: We use AdamW, which often provides better regularization and performance on deep learning models compared to the standard Adam.
Learning Rate: A constant learning rate of 1e-4 is used throughout the training.
Early Stopping: Training is halted if the validation wMAE score does not improve for 40 consecutive epochs (patience=40). The model from the best epoch is saved.
Batch Size: We use a batch size of 64 during training.
Custom Loss Function for Direct Metric Optimization:
We implemented a custom Weighted MAE (wMAE) Loss Function in PyTorch.
This is critical because it ensures the model's training objective is perfectly aligned with the official competition metric, leading to more direct and efficient optimization.
Post-Hoc Calibration Details:
The Linear Calibrator is a simple LinearRegression() model from scikit-learn. For each target, it learns a linear transformation (y = a*x + b) to map the GNN's raw predictions to the true values, effectively correcting simple biases.

#Feature Selection/engineering
**Fundamental Transformation: SMILES to Graph Representation**
We convert 1D SMILES strings into rich graph structures where atoms are nodes and bonds are edges.
Node features include atomic number, degree, hybridization, etc. (7 features total).
This allows our GNN model to learn directly from the polymer’s 2D topology, capturing complex structural information.

**Key Engineering Step: Chemical Data Augmentation**
We implemented a function to programmatically extend monomer repeat units (*A* -> *A-A-A*).
This creates a larger and more realistic training set, forcing the model to learn features from longer, more representative polymer chains, which significantly boosts generalization.
[Before augmentation]
[After aufmentation]

**Winning Strategy: Hybrid & Task-Specific Feature Selection**
We combine two feature types: learned global embeddings from the GNN and curated local features from Morgan Fingerprints.
For each of the 5 targets, we independently select the Top 50 most predictive Morgan Fingerprints using F-regression, creating a tailored feature set for each prediction task.

**e.g.**
We used F-regression to score all 1024 Morgan Fingerprint bits against the Tg target, identifying the most predictive chemical substructures.
The plot visualizes the Top 20 bits with the highest scores. These bits, such as fp_587 and fp_80, represent the most influential local chemical features for Tg prediction.
This data-driven selection process allowed us to create a compact and powerful 50-feature set for the Tg prediction head, instead of using the full noisy 1024-bit vector.
[top 20 FP for Tg]
Relationship to Target: Impact of the Top Feature (fp_587) on Tg
To visualize the impact of our most important feature, we compared the Tg distributions for two groups of polymers: those that possess the fp_587 fingerprint bit and those that do not.
The violin plot clearly shows a significant positive correlation: polymers with the fp_587 substructure have a markedly higher median Tg (approx. 280°C vs 150°C).
This strong visual evidence validates our feature selection method and confirms that our model is learning meaningful chemical relationships.
[relationship]

#Training Methods
**Custom Weighted MAE Loss with AdamW Optimizer.**
We implemented a custom loss function in PyTorch that perfectly mirrors the official wMAE competition metric. This ensures the model directly optimizes for the final score.
AdamW optimizer was used with a constant learning rate of 1e-4.
Robust 5-Fold Cross-Validation Ensemble.
The final solution is a simple average ensemble of 5 models trained on different 80/20 splits of the data.
Batch size of 64 was used during training.
**Two-Stage Training with Early Stopping & Refitting.**
Stage 1: Find Best Epoch. We train on 80% of the data and use the remaining 20% as a validation set for Early Stopping (patience=40), preventing overfitting.
Stage 2: Refit on Full Data. A new model is then re-trained from scratch on 100% of the fold's data for exactly the best number of epochs found in Stage 1. This maximizes data utilization for each final model.
**Post-Hoc Linear Calibration.**
After training, a simple LinearRegression model is fitted for each target on the validation set's predictions. This effectively corrects any systematic bias from the GNN model, providing a final small but crucial boost in performance.

#Important&interesting Findings
**Finding 1: Polymer Chain Extension is Superior to Isomer Generation**
We experimented with two distinct data augmentation strategies.
Strategy A (Chain Extension): Simulating polymerization by extending monomer repeat units (*A* -> *A-A-A*). This was highly effective.
Strategy B (Isomer Generation): Creating multiple different but chemically valid SMILES strings (isomers) for the same monomer. This provided no performance gain.
[different Augmentation]

**Finding 2:The "Sweet Spot" of Batch Size - Bigger is Not Always Better**
Despite having enough GPU memory for larger batch sizes (128, 256), we found that a smaller batch size of 64 consistently yielded the best results.
Larger batch sizes led to a degradation in the final PB score.

**Finding 3: Surprising Synergy Between GNN Architectures and Feature Types**
We tested different combinations of GNN backbones and auxiliary features. The results revealed a strong and unexpected interaction effect.
GATv2 + Morgan Fingerprints was the clear winner.
Interestingly, RDKit Descriptors hurt the performance of GATv2 a lot, but helped the performance of NNConv a little.
[different models&features]
Hypothesis:
GATv2's attention mechanism already captures complex, high-level geometric and chemical environments. The simple, bit-based Morgan Fingerprints provide complementary, low-level substructure information without creating conflicting signals.
NNConv, being a simpler message-passing network, may benefit more from the pre-computed, human-engineered RDKit Descriptors, which provide it with high-level chemical concepts it struggles to learn on its own.

**Finding 4: Systematic Bias Correction via Post-Hoc Calibration**
Analysis of our Out-of-Fold (OOF) predictions across all 5 folds revealed a consistent, systematic bias: the model's raw predictions (blue line) consistently overestimate low Tg values and underestimate high Tg values.
This distribution shift is a common challenge, and correcting for it was key to maximizing our score.

We found this bias can be effectively corrected with a simple LinearRegression model fitted on each fold's validation predictions.
As shown in the table, this simple post-hoc step provides a significant 5.47% improvement to the final wMAE score. It's a small change that makes a crucial impact.Our PB Score from 0.080 to 0.078.We must thanks to https://www.kaggle.com/hengck23 in https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/589360

#Simple Model
To understand the contribution of each component, we evaluated several simplified versions of our final model.
A single model (1-fold) without ensembling already achieves a competitive score.
[Simple Model]
**Post-Competition Finding: The Untapped Potential of RemoveHs**
After the competition ended, we tested a simple change in the data preprocessing step: removing hydrogen atoms (RemoveHs in RDKit) before graph conversion.
Surprisingly, a single model trained with RemoveHs achieved a Public LB score of ~0.083, which is even better than our single model with hydrogens.
Conclusion: This suggests that if we had applied the RemoveHs preprocessing to our full 5-fold ensemble pipeline, our final score could have been even higher. The model performs better when focusing on the heavy-atom backbone of the polymer.

Thank you
hongyu Guo
