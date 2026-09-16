# 23rd Place Solution: Supervised Contrastive Learning Meets Domain Generalization (with TF code)

Competition: rfcx-species-audio-detection
Rank: #23
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220972

### Introduction

Thanks Kaggle for this exciting competition and our team ( @dathudeptrai @mcggood @akensert @theoviel @ratthachat ) congratulate all winners -- we have learned a lot from this competition and all winners’ solutions !!

From the winner solutions, it turns out there are mainly 4 ways to break 0.95x

1. Masked loss
2. Post-processing
3. Pseudo-labelling
4. Extra labeling


We actually tried the first three but unfortunately could not really make them work effectively.
Here, as an alternative solution, we would love to share our own solution which is able to reach 0.943 Private. 

### Training pipeline (improve from 0.80x baseline to 0.92x)

#### Baseline

Our baseline models score slightly above 0.8. We adopt an audio tagging approach using a densenet121 backbone and the BCE loss. 

Our training pipeline includes the following tricks :
 
- Class-balanced sampling, using 8 classes per batch. Batch sizes were usually 64, and 32 for bigger models.
- Cyclical learning rate with min_lr is 0.0001 and max_lr is 0.001, step_size is 3 epochs. We train models for 100 epochs with early stopping. 
- LookAhead with Adam optimizer (sync_period is 10 and slow_step_size is 0.5)

#### Pretraining with Supervised contrastive Learning (SCL) [0.81x -> 0.85x]

Because of the small amount of data, models overfit quickly. To solve this problem,two options were using external data and cleverly pretraining our models. Unlike a lot of competitors, we focused on self-pretraining techniques : @dathudeptrai tried auto-encoders, GANs, SimCLR, Cola, and [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) which ultimately was the only thing to work. 


#### Non-overlap time Cutmix [0.85x -> 0.88x]



Our sampling strategy consists of randomly selecting a crop containing the label. Most of the time, crops are bigger than labels which introduces false positives. One idea to make full use of our windows was to adapt cutmix to concatenate samples such that labels are entirely kept (when possible). 

#### Domain Generalization with MixStyle [0.88x -> 0.89x]

Domain shift always exists in deep learning, in both practice and kaggle challenges, especially for small data. Therefore, domain generalization techniques should help with robustness. We applied a simple yet effective technique called [Mixstyle](https://openreview.net/pdf?id=6xHJ37MVxxp).


#### Multi Scale inference (MSI) [0.89x -> 0.91x]

Duration of species’ call varies quite a lot. For example, for class 3 it is around 0.7 seconds while for class 23 is around 8 seconds. To use this prior information, we use multiple window sizes (instead of using a single one). For each class, we choose the one that yields the best CV. In case we have multiple window sizes reaching the maximum, we take the largest window. Although our CV setup which consists of crops centered around the labels did not correlate really well with LB, the 2% CV improvement reflected on LB quite well.


#### Positive learning and Negative learning  [0.91x -> 0.92x]

We used the following assumption to improve the training of our models : 
For a given recording, if a species has a FP and no TP, then it is not in the call. Our BCE was then updated to make sure the model predicts 0 for such species. 

### Ensembling

Our best single model densenet121 scores around 0.92 public and 0.93 private. Averaging some models with different backbones, we were able to reach 0.937. We tried many different ensembling, scale fixing and post-processing ideas, and were able to improve our score a bit, but unfortunately we could not find the real magic.

In the end, we empirically analyzed the most uncertain class predictions from our best models, and averaged predictions with other (weaker) models. We relied on diversity to make our submission more robust. Our final ensemble scored public 0.942 and private 0.943.

##### Thanks for reading ! 

[TensorFlow Code Here](https://github.com/dathudeptrai/rfcx-kaggle)
