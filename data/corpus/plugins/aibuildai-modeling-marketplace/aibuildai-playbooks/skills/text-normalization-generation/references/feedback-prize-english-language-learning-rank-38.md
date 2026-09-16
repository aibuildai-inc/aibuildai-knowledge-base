# 38th place solution

Competition: feedback-prize-english-language-learning
Rank: #38
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/370459

Thanks to kaggle team, host for hosting this competition and to teammates. Thanks also to the Kaggler's for sharing their helpful codes and discussions. This was my second time to participate in NLP competition after the previous fp2 competition, and I learned a lot.

## Overview

My final submission is ensemble of 15 models optimized for weights with optuna. Each models are trained with various hyperparameter such as seed, num_fold, max_length, and so on.



## What Worked

* SmoothL1Loss was the best in CV.
* Using optuna to determine ensemble weights for each target.
* Use various custom head.
  * mean pooling, lstm, concat last 4 layer
* Adversarial Weight Perturbation
* Use various model. (not only the hugging face model, but also lightgbm)
  * Single model is not good, but ensemble has improved score a little.
  * refs (lightgbm model): <https://www.kaggle.com/code/dlaststark/fpe-no-fancy-stuff>

## What Didn’t Work

* Different loss functions
* mixout
* Use text features as meta feature when I stacking models.

These methods did not improve cv scores, but few were effective in ensembles.

## Important Citations

* <https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train>
@yasufuminakama. His notebook has taught me a lot.
* <https://www.kaggle.com/code/electro/deberta-layerwiselr-lastlayerreinit-tensorflow>
* <https://www.kaggle.com/code/kojimar/fb3-single-pytorch-model-train/notebook>

## Thanks and Acknowledgements

Finally, I'd like to thank Kaggle, host for hosting such an iteresting competition and all who participated discussions.
Thank you and Congratulations to my team mates - @mitsuruueki
