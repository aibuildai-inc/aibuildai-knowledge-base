# 12th Place Solution: E2E NN

Competition: ariel-data-challenge-2025
Rank: #12
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/12th-place-solution-e2e-nn

**Preface**
Thanks to the host and Kaggle team for organizing this competition, I learned a lot from this first contact with astronomy and I hope to participate again.
My approach is quite simple and just a neural network end to end. I ended up ensembling 3 for my final submission, but a single model can reach 0.552.

**Calibration**
I adapted this amazing [notebook](https://www.kaggle.com/code/ilu000/ariel25-quick-data-prep-improved) by Pascal with minor changes, keeping the 15 binning.

**Preprocessing**
Signals were clipped to remove outliers then group norm (over lambda) was applied.
Orbit parameters in star_info were not used.

**Transits**
During inference the model internally outputs transit points t1 to t4, and they are trained in a semi-supervised manner with labels derived from the gradients of the savgol smoothed signal and filtered for validity.

**Architecture**
The model is a bert style transformer encoder and takes preprocessed signals as input and outputs mu and sigma (logvar).
The intermediary transit points are fed into the final output head.

**Training**
Since the model outputs both mu and sigma, I trained directly on the competition metric with gaussian NLL, and the (intermediary) transit head was trained with smooth L1 and an unsupervised symmetry loss (also smooth l1).
For LR, cosine decay scheduling was used in addition to very high weight decay and a little dropout.

**Augmentation**
Flip and aperture augmentation were applied to the signals during training as well as adding noise to the transit points.

**Inference**
Flip TTA boosted LB by quite a bit and ensembling was done with inverse variance weighting.

**Caveats**
Training variance was high (without deterministic algorithms), the same seed can converge at quite different places, likely because of the sensitivity of GNLL. This made it quite difficult to tune hyperparameters reliably with my limited compute.
Extra signals didn't have a noticeable effect during training nor inference.

**Todo**
Utilize orbit parameters and disentangle gain drift.
