# 13th place solution

Competition: nbme-score-clinical-patient-notes
Rank: #13
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323074

First, thanks to kaggle team and organizers for hosting this competition!
I learned a lot from this competition, especially for pseudo labeling because this is my first time to use pseudo labeling.

# Overview of our solution
Our models are basically same as others. Our final submissions consist of the ensemble of deberta-based QA-style models trained with pseudo labeled instances.
The network structure is bert (last hidden state) -> dropout -> fc (output_dim=1) -> BCE loss.
We simply took ensemble over the output probabilities with the uniform weight.

# Pseudo labeling
What we tried for PL at the beginning is selecting the trustful pseudo instances by setting threshold over the predicted probabilities to reduce the noise.
Unfortunately, we found that setting any threshold got worse LB score than without threshold (precisely, without-threshold is effectively same as threshold=50%).
We also tried the confidence-based selection method (https://openreview.net/forum?id=-ODN6SbiUU ), but it also didn't outperform the one with no-threshold (I was personally sad for this result).
My thought is, we need difficult instances to improve the model's decision ability but selection methods remove such instances, so there is trade-off. As many prior works have succeeded by selecting the reliable instances, there should be some factors that leads the selection not to work well for this dataset, but I'm not sure which is it because of lacking in my knowledge (maybe if the performance of teacher model is enough good, the selection could be relatively unnecessary?)

Although the threshold-based selection doesn't work well, it's also true that there are noises on pseudo instances.
Instead of setting the threshold, we tried building pseudo instances for each CV split and taking intersection of them (i.e, retain the instance if all CV models output same predictions for it). This reduces roughly 1/3 of pseudo instances and boosted LB +0.001~0.002.
The drawback is, this leaks the information of OOF more so our CV scores go to moon🚀 (0.91~0.92).
We also tried building pseudo instances from the ensemble output over CV and it gives similar CV/LB scores to the one from the intersection. I feel the intersection-version is little further from direct leakage of OOF than the ensemble-version, so I decided to use the intersection-version.

# Training settings
We used Radam for the optimizer. We kept the learning rate fixed during the training and didn't use any learning scheduler such as cosine annealing.
We clipped the gradient by norm with the maximum norm of 1.0.
We initialized the weight of the output fc layer by transformers.PreTrainedModel._init_weights method, and this really helped to stabilize the training.
No MLM training.

# Tuning
When I used deberta-v3-large, I tried tuning all hyper-parameters but most parameters affected the CV score only a little. The exception is the learning rate, so for other deberta-variants, I used same values as deberta-v3-large for hyper-parameters other than the learning rate and only tuned the learning rate for each model.

# Ensemble
- v3-large + v3-large (other CV seed) + v1-xlarge + v1-large
LB 0.893, Private 0.892
- v3-large + v3-large (other CV seed) + v3-large (one more other CV seed) + v1-xlarge + v2-xlarge
LB 0.893, Private 0.892

Single (w/ PL):
| Model | LB | Private |
| --- | --- | --- |
| v3-large | 0.889 | 0.889 |
| v3-large (other CV seed) | 0.890 | 0.891 |
| v3-large (one more other CV seed) | 0.888 | 0.889 |
| v1-xlarge | 0.888 | 0.891 |
| v1-large | 0.886 | 0.888 |
| v2-xlarge | 0.888 | 0.889 |

CV n_splits=4 for all.


Again, thank you all for this great competition!
