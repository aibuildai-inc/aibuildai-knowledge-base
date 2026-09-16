# 7th place solution

Competition: stanford-ribonanza-rna-folding
Rank: #7
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460190

## Summary
- Transformer based solution
- Masked Conv1D instead of MLP
- bpp injection into attention matrix, bpp-based token mixing, matrix mixing, and dual stream setup (attention boosting) for incorporating bpp

## Introduction
Our team would like to thank the organizers and Kaggle for making this competition possible. Also, I want to express my gratitude to my outstanding teammates @drhabib and @martynoveduard for their incredible contribution toward our final result.

## Details
### Data
**BPP**: We generated additional bpp using vienna_2 (we also used SS output), contrafold_2, rnaformerv1, and rnafm. However, we saw nearly negligible improvement in comparison to using only bpp provided by organizers.
[**EX data**](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/454397): We tried to do fine-tuning on this data since it is only the source that provides GT for sequence ends. We got 5-10 bps CV improvement from this procedure (Iafoss/DrHB) and got the best visualization for long-range dependencies. However, at the sequence ends the predictions tend to get values close to zero for unknown reasons. At private LB EX data did not give any boost.
**CV split**. slime used a random 4-fold split. DrHB and Iafoss used a split based on sequence similarity to avoid any possible leaks and also we excluded any train data overlapping with the test test from the train. The size of the val set is ~20k samples that passed SN criterion.

### Model
At the beginning of the competition we quickly realized that bpp is very helpful for model performance. Initially, we tried to use it as an auxiliary output computed as the mean attention matrix, but convergence was quite slow (the typical run was 200-250 epochs). Then we build setups that also take bpp as an input. It drastically accelerated convergence, so that it took only several tens of epochs, while giving similar results.
We considered several setups:
(1) **bpp injection** (Iafoss): add bpp (in the logit form) directly to the attention matrix (in case of multiple bpp, each bpp is added to the particular head). We gradually attenuate the injection towards zero in the last layers to give the mode the freedom to develop interactions missed in bpp. The critical part of the model was replacing MLP with a masked 1d x5 convolution that performs the mixing of neighboring tokens. The transformer has 384 width, a head size of 64, 24 depth, a droppath of 0.3, dropout of 0.1. We use rotary enc to define the relative position of tokens. Single bpp and 6 bpp setups were considered. The best single model got 0.14375 private and 0.14013 public LB and 0.14292 and 0.13973 with using pseudo labels.
(2) **bpp mixing** (DrHB): instead of injection of bpp into the attention matrix, we tried to add matrix multiplication modules performing mixing tokens based on bpp. The total structure of the model: 3 x [bpp eterna -> conv ->transformer -> bpp rest mean ->conv ->transformer ->ss- as adj -> graph transformer ].
(3) **matrix mixing** (slime): trying to follow the previous competition solutions we added a learnable attention bias produced by a convolutional stream applied to bpp and ALIBI-like positional encoding. We used 2 bpp sources. The matrix mixer is represented by several conv layers with SE modules. The transformer consists of 12 matrix mixing + 12 regular transformer blocks with a width of 384. The best single model is 0.14509 private and 0.14066 public LB. This model used a different training pipeline and cannot be directly compared to others.
(4) **dual stream** (Iafoss). The drawback of matrix mixing is that the attention bias is updated independently from the transformer based on the input bpp. How about attention boosting? We perform a simple projection of the attention state followed by scaled tanh nonlinearity to limit accumulated values and stabilize training. This value is added to the Attention matrix and then the result is fed to the next transformer layer. This modification has improved the model performance giving 0.14296 at private and 0.13697 public LB as a single model. Unfortunately, we discovered this setup only a few days before the end of the competition, and didn't have a chance to run multiple trainings and PL setup, which we expect to give a further ~10 bps improvement of LB.

The models are schematically depicted in the image below:


### Training
**(Iafoss/DrHB)** The loss is weighted based on the error, while we do not downselect data based on SN criterion: `w = 1/sqrt(1/6 + err.clip(100))`. Use AdamW, cosine annealing with warmup, lr=5e-4, wd=0.05, bs=16 (small bs was working better for the reason we could not identify). We used flip augmentation for both train and TTA, but the key thing here is using bpp computed for the correct order of nucleotides. It quite improves CV giving 10-15 bps boost. We also used bin based auxiliary loss which gave a slight improvement.

We split the training portion of the data into 4 folds, train 4 models (40-48 epochs), fine-tune on external data + train data (5-6 epochs with x10 EX data oversampling), fine-tune on noise-free samples for 12 epochs. Then we corrected the provided train data weighting GT and PL based on the inverse error (assuming 0.15 error for PL). The provided train data has a large level of noise slowing down the convergence. Test data and the sequence ends were labeled solely based on PL. Then we train a PL model on the whole train (excluding val samples) + test data, fine-tune on EX data, and fine-tune on noise-free samples. This procedure improved CV by 10 bpp, and the weighted average of all models generated in the procedure gets further improvement.

**(slime)** *Pre-training*: First we performed MLM pre-training on a whole dataset for 5 epochs (40% of input tokens are replaced with mask tokens). We used cosine decay to zero with one epoch warmup and AdamW optimizer with base_lr= 5e-4 and wd=0.05. The model was trained to predict missing nucleotides with cross-entropy loss

*Fine-tuning*: During the fine-tuning stage we initialized our model with weights from MLM-pretraining, it gave a noticeable improvement to the final result [10 bps CV]. We fine-tuned our models for 50 epochs with batch_size=16 on samples where either SN(DMS) == 1 or SN(2A3) == 1, we masked the loss depending on SN of the given test [10 bps boost compared to filtering train dataset with SN(DMS) == 1 and SN(2a3) == 1). In addition, we weighted the samples based on their reactivity error provided by the ground-truth data as `loss *= torch.log(1.1 + snr) / 2`. Similar to pertaining, we used AdamW optimizer [lr=5e-4, wd=0.05] and cosine scheduler with lr decay to zero. Since in matrix mixing model masking was not considered in convolutions, this model was trained with length-matching batch sampling (samples of exactly the same length).

### Best singe model end ensemble
Best single model (dual stream model): 0.14292 at private and 0.13711 public LB, which can take top 10 itself. This result would be further improved by 10 bps to ~0.1419 if if we had time to run our full PL pipeline.
Our final submission is a combination of ~20 models that got 0.14189 at private and 0.13604 at public LB.

### Things didn't work
- EX data was helpful at CV and public LB (5-10 bps boost), but not helpful at private LB
- Additional bpps gave only a negligible improvement in comparison to the use of single bpp provided by organizers
- MLM on 30M external RNA sequence dataset
- EMA, AWP, Floyd-warshall distance matrices 
- 2D Ushape models
