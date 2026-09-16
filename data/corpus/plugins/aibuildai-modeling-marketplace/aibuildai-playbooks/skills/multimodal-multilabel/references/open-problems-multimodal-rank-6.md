# [6th private - 3rd public] Summary of our solution

Competition: open-problems-multimodal
Rank: #6
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/366417

First of all, thanks to the Kaggle team and the host for providing this cutting-edge technology dataset and hosting this great challenge. I learned a lot from the competition. Many thanks to my teammates @mathormad @nguyenvlm for hard-working days. This is a summary of what we did to get 3rd place in LB and 6th place in private

# Local CV (for both cite and multiome task)
We use **stratified-5-Fold** (stratify by day/donor/celltype) CV strategy. We do not use GroupKFold to avoid overfitting in public, and the private dataset day is also quite "far" from the training dataset, so it's a bit risky to use a CV split by day (we've tried it, but both CV and LB drop). In our opinion, **this is the key to our stable placement in both public and private leaderboards. Our best submission in CV are also the best on Public and Private leaderboard.**


# Feature engineering (cite)
- Feature selection: remove all 0 cols group by metadata (day/donor/cell_type/train/test) --> remove ~4,500 features
- Dimension reduction: we use different methods to increase the diversity for the final ensemble: 240 n_components SVD/quantiledPCA and denoise Autoencoder (256 latent dims). For SVD, increase n_iter params slightly improve CV by 2e-4 (but run longer) (ours is 50 iter, the default of sklearn library is only 5). The denoise autoencoder helps us improve CV by 1e-3 (comparing to SVD/PCA)
- Feature importance
    - Use name matching from this discussion https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349242
    - Search using xgb feature importance: for cite, we fit multiple xgb for each target (full 22k feature) then choose the top 5  important features of each target. In total, we get around 500 important features for our model.
- TargetEncoder for xgb: we apply target encode the cell_type for each target (each cell_type will be represented by a 140-dims vector)


# Feature engineering (multiome)
- Feature selection: remove chY features, which is not correlated to our target. We also remove all 0 cols group by metadata (day/donor/cell_type/train/test) --> remove ~ 500 features
- Dimension reduction: we use 256 n_components SVD fitted with 200iter

# Training process
- Use custom loss (weighted correlation and MSE loss)
- Use c-mixup to increase the diversity for tabnet
- SWA when training MLP, Denoise Autocoder, Tabnet and 1D-CNN
- Adam optimizer with high learning rate (1e-2)
- (Multiome) XGB is trained with the PCA of target as label

# External data
For cite task, we **apply the whole training process above for the [raw count dataset](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/359355)**, then ensemble with the original one. This significantly boosts the performance by 1e-3
For multiome task, we do not use the raw count dataset because it's lacking some rows, we cannot match the OOF between the original and raw count so we do not have CV score to validate. I think it should work but we do not have enough time to rematch the OOF

# Stacking feature & Pseudo-labeling
We concatenate the prediction of 1 model with current features as input to another model (e.g. use output of xgb as input of MLP, and vice versa), which improves the performance of a single model around 5e-4. 
We also use pseudo labeling (not much improvement, about 2-3e-4)

# Post process
- (Cite) Apply standard scaler (axis 1) for the output of each fold before taking the average
- (Multiome) Some of the target is all 0, so we remove them from the training process and replace 0 later. This helps us increase our CV by 1e-3

# Ensemble
- (Cite) Our final submission is a blending of the following models:
  - MLP
  - XGB
  - Tabnet
  - 1D-CNN
Each model is trained on the original and raw count dataset -> 8 models in total
- (Multiome) Blending of:
    - MLP
    - XGB
    - Tabnet


# Tryhard
We worked 6 hours/day, from the very beginning of the competition to the last hour. Especially for this competition when we have 2 tasks with 2 datasets, a lot of work and experiments to do. The most important thing I learned during this competition is that the more time you spend, the higher place you will be.

# What does not work
- Encoder for multiome task
- Use important features in multiome task
- Some bio techniques such as Ivis (dimension reduction), Magic (denoising)
- Use raw label for training
- ...
